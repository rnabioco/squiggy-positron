"""Search functionality for Squiggy viewer"""

import asyncio

from PySide6.QtWidgets import QMessageBox

from .utils import (
    get_bam_references,
    get_reads_in_region,
    get_reference_sequence_for_read,
    parse_region,
    reverse_complement,
)


class SearchManager:
    """Manages search operations for reads"""

    def __init__(self, parent_window):
        """
        Initialize SearchManager

        Args:
            parent_window: The parent QMainWindow for displaying dialogs
        """
        self.parent = parent_window

    def filter_by_read_id(self, read_tree, search_text):
        """
        Filter the read tree based on read ID search input

        Args:
            read_tree: ReadTreeWidget to filter
            search_text: Text to search for in read IDs
        """
        read_tree.filter_by_read_id(search_text)

    async def filter_by_region(self, bam_file, read_tree, region_str):
        """
        Filter reads based on genomic region query (requires BAM file)

        Args:
            bam_file: Path to BAM file
            read_tree: ReadTreeWidget to filter
            region_str: Region string (e.g., "chr1:1000-2000")

        Returns:
            tuple: (success: bool, visible_count: int, message: str, alignment_info: dict)
        """
        if not region_str:
            # Clear filter - show all reads
            read_tree.show_all_reads()
            return True, 0, "Ready", {}

        # Check if BAM file is loaded
        if not bam_file:
            QMessageBox.warning(
                self.parent,
                "BAM File Required",
                "Reference region search requires a BAM file.\n\n"
                "Please load a BAM file first.",
            )
            return False, 0, "BAM file required", {}

        # Parse region
        chromosome, start, end = parse_region(region_str)
        if chromosome is None:
            QMessageBox.warning(
                self.parent,
                "Invalid Region",
                f"Could not parse region: {region_str}\n\n"
                "Expected format: chr1, chr1:1000, or chr1:1000-2000",
            )
            return False, 0, "Invalid region format", {}

        try:
            # Run query in background thread
            reads_in_region = await asyncio.to_thread(
                get_reads_in_region, bam_file, chromosome, start, end
            )

            # Filter tree to show only reads in region
            visible_count = read_tree.filter_by_region(reads_in_region)

            # Build region description
            region_desc = f"{chromosome}"
            if start is not None and end is not None:
                region_desc += f":{start}-{end}"
            elif start is not None:
                region_desc += f":{start}"

            message = f"Found {visible_count} reads in region {region_desc}"
            return True, visible_count, message, reads_in_region

        except ValueError as e:
            QMessageBox.critical(self.parent, "Query Failed", str(e))
            return False, 0, "Query failed", {}
        except Exception as e:
            QMessageBox.critical(
                self.parent, "Error", f"Unexpected error querying BAM file:\n{str(e)}"
            )
            return False, 0, "Error", {}

    async def browse_references(self, bam_file):
        """
        Get list of references from BAM file for browsing

        Args:
            bam_file: Path to BAM file

        Returns:
            tuple: (success: bool, references: list or None)
        """
        if not bam_file:
            QMessageBox.warning(
                self.parent,
                "No BAM File",
                "Please load a BAM file first to view available references.",
            )
            return False, None

        try:
            # Get references in background thread
            references = await asyncio.to_thread(get_bam_references, bam_file)
            return True, references

        except Exception as e:
            QMessageBox.critical(
                self.parent,
                "Error",
                f"Failed to load BAM references:\n{str(e)}",
            )
            return False, None

    async def search_sequence(
        self, bam_file, read_id, query_seq, include_revcomp=True
    ):
        """
        Search for a DNA sequence in the reference

        Args:
            bam_file: Path to BAM file
            read_id: Read ID to search in
            query_seq: Query sequence (should be uppercase)
            include_revcomp: Whether to include reverse complement search

        Returns:
            tuple: (success: bool, matches: list, message: str)
        """
        if not query_seq:
            return True, [], "Ready"

        # Check if BAM file is loaded
        if not bam_file:
            QMessageBox.warning(
                self.parent,
                "BAM File Required",
                "Sequence search requires a BAM file with reference alignment.\n\n"
                "Please load a BAM file first.",
            )
            return False, [], "BAM file required"

        # Validate sequence (should be DNA: A, C, G, T, N)
        valid_bases = set("ACGTN")
        if not all(base in valid_bases for base in query_seq):
            QMessageBox.warning(
                self.parent,
                "Invalid Sequence",
                f"Invalid DNA sequence: {query_seq}\n\n"
                "Only A, C, G, T, N characters are allowed.",
            )
            return False, [], "Invalid sequence"

        try:
            # Search for sequence in background thread
            matches = await asyncio.to_thread(
                self._search_sequence_in_reference,
                bam_file,
                read_id,
                query_seq,
                include_revcomp,
            )

            if not matches:
                message = (
                    f"No matches found for '{query_seq}'"
                    + (" (or reverse complement)" if include_revcomp else "")
                )
                return True, [], message
            else:
                message = f"Found {len(matches)} match(es) for '{query_seq}'"
                return True, matches, message

        except Exception as e:
            QMessageBox.critical(
                self.parent, "Search Failed", f"Failed to search sequence:\n{str(e)}"
            )
            return False, [], "Search failed"

    def _search_sequence_in_reference(
        self, bam_file, read_id, query_seq, include_revcomp=True
    ):
        """
        Search for sequence in the reference (blocking function)

        Args:
            bam_file: Path to BAM file
            read_id: Read ID to search in
            query_seq: Query sequence (uppercase)
            include_revcomp: Whether to search reverse complement

        Returns:
            list: List of match dictionaries with keys:
                  strand, ref_start, ref_end, base_start, base_end, sequence
        """
        # Get reference sequence for this read
        ref_seq, ref_start, aligned_read = get_reference_sequence_for_read(
            bam_file, read_id
        )

        if not ref_seq:
            raise ValueError(f"Could not extract reference sequence for read {read_id}")

        matches = []

        # Search forward strand
        start_pos = 0
        while True:
            pos = ref_seq.find(query_seq, start_pos)
            if pos == -1:
                break

            # Convert reference position to base position in alignment
            ref_pos = ref_start + pos
            ref_end = ref_pos + len(query_seq)

            # Map to base position (0-indexed in the aligned read)
            base_start = pos
            base_end = pos + len(query_seq)

            matches.append(
                {
                    "strand": "Forward",
                    "ref_start": ref_pos,
                    "ref_end": ref_end,
                    "base_start": base_start,
                    "base_end": base_end,
                    "sequence": query_seq,
                }
            )

            start_pos = pos + 1

        # Search reverse complement if requested
        if include_revcomp:
            revcomp_seq = reverse_complement(query_seq)
            if revcomp_seq != query_seq:  # Only search if different
                start_pos = 0
                while True:
                    pos = ref_seq.find(revcomp_seq, start_pos)
                    if pos == -1:
                        break

                    ref_pos = ref_start + pos
                    ref_end = ref_pos + len(revcomp_seq)
                    base_start = pos
                    base_end = pos + len(revcomp_seq)

                    matches.append(
                        {
                            "strand": "Reverse",
                            "ref_start": ref_pos,
                            "ref_end": ref_end,
                            "base_start": base_start,
                            "base_end": base_end,
                            "sequence": revcomp_seq,
                        }
                    )

                    start_pos = pos + 1

        return matches
