#!/usr/bin/env python3
"""
Filter reads from BAM file and extract matching reads from POD5 file(s).

Usage:
    python filter_reads.py <bam_file> <pod5_path> [options]

Examples:
    # Single POD5 file
    python filter_reads.py alignments.bam reads.pod5 --chromosome chrIII --output filtered_reads.pod5

    # Directory of POD5 files with BAM output
    python filter_reads.py alignments.bam pod5_dir/ --chromosome chrIII --output filtered_reads.pod5 --output-bam filtered_reads.bam
"""

import argparse
import pysam
import pod5
from pathlib import Path
from typing import List, Dict, Any, Union


def filter_reads_from_bam(
    bam_path: str,
    chromosome: str = "chrIII",
    min_mapq: int = 30,
    target_length: int = 1000,
    length_tolerance: int = 100,
    region_size: int = 10000,
    max_reads: int = 100,
) -> List[Dict[str, Any]]:
    """
    Filter reads from BAM file based on quality and genomic criteria.

    Args:
        bam_path: Path to indexed BAM file
        chromosome: Target chromosome (default: chrIII)
        min_mapq: Minimum mapping quality (default: 30)
        target_length: Target read length in bp (default: 1000)
        length_tolerance: Allowed deviation from target length (default: 100)
        region_size: Size of genomic window in bp (default: 10000)
        max_reads: Maximum number of reads to return (default: 100)

    Returns:
        List of dictionaries containing read information
    """
    min_length = target_length - length_tolerance
    max_length = target_length + length_tolerance

    reads = []
    positions = []

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        # First pass: collect candidate reads
        for read in bam.fetch(chromosome):
            # Skip unmapped, secondary, and supplementary alignments
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue

            # Filter by mapping quality
            if read.mapping_quality < min_mapq:
                continue

            # Filter by read length
            read_length = read.query_length
            if read_length < min_length or read_length > max_length:
                continue

            # Store read info
            positions.append(read.reference_start)
            reads.append(
                {
                    "read_id": read.query_name,
                    "position": read.reference_start,
                    "length": read_length,
                    "mapq": read.mapping_quality,
                    "strand": "-" if read.is_reverse else "+",
                }
            )

    if not reads:
        print(f"No reads found matching criteria on {chromosome}")
        return []

    # Find a 10kb window with maximum read density
    positions.sort()
    best_window_start = positions[0]
    max_count = 0

    for start_pos in positions:
        end_pos = start_pos + region_size
        count = sum(1 for pos in positions if start_pos <= pos < end_pos)

        if count > max_count:
            max_count = count
            best_window_start = start_pos

    # Filter reads to those within the best window
    best_window_end = best_window_start + region_size
    filtered_reads = [
        r
        for r in reads
        if best_window_start <= r["position"] < best_window_end
    ]

    # Limit to max_reads
    filtered_reads = filtered_reads[:max_reads]

    return filtered_reads


def extract_reads_from_pod5(
    pod5_path: Union[str, Path],
    read_ids: List[str],
    output_path: str,
) -> tuple[int, List[str]]:
    """
    Extract specific reads from POD5 file(s) and write to new POD5 file.

    Args:
        pod5_path: Path to input POD5 file or directory containing POD5 files
        read_ids: List of read IDs to extract
        output_path: Path to output POD5 file

    Returns:
        Tuple of (number of reads extracted, list of successfully extracted read IDs)
    """
    extracted_count = 0
    pod5_path = Path(pod5_path)

    # Collect all POD5 files to search
    if pod5_path.is_file():
        pod5_files = [pod5_path]
    elif pod5_path.is_dir():
        # Find all .pod5 files in the directory
        pod5_files = sorted(pod5_path.glob("*.pod5"))
        if not pod5_files:
            print(f"Warning: No POD5 files found in directory: {pod5_path}")
            return 0
        print(f"Found {len(pod5_files)} POD5 files in directory")
    else:
        raise FileNotFoundError(f"POD5 path does not exist: {pod5_path}")

    # Track which reads we still need to find
    remaining_read_ids = set(read_ids)
    skipped_end_reason = {}  # Track reads skipped due to end_reason

    with pod5.Writer(output_path) as writer:
        # Search through all POD5 files
        for pod5_file in pod5_files:
            if not remaining_read_ids:
                break  # All reads found

            try:
                with pod5.Reader(pod5_file) as reader:
                    # Use selection parameter to efficiently get only the requested reads
                    for read_record in reader.reads(selection=list(remaining_read_ids), missing_ok=True):
                        read_id_str = str(read_record.read_id)

                        # Check end_reason - only include signal_positive reads
                        if read_record.end_reason.name != "signal_positive":
                            skipped_end_reason[read_id_str] = read_record.end_reason.name
                            remaining_read_ids.remove(read_id_str)
                            continue

                        # Convert ReadRecord to Read object before adding to writer
                        writer.add_read(read_record.to_read())
                        # Convert UUID to string for comparison with BAM read IDs
                        remaining_read_ids.remove(read_id_str)
                        extracted_count += 1

                # Progress update for directory mode
                if len(pod5_files) > 1:
                    print(f"  Processed {pod5_file.name}: {extracted_count}/{len(read_ids)} reads found")

            except Exception as e:
                # Skip problematic files and continue
                if len(pod5_files) > 1:
                    print(f"  Warning: Skipped {pod5_file.name}: {type(e).__name__}: {str(e)[:100]}")
                else:
                    # If it's a single file, re-raise the error
                    raise

    if skipped_end_reason:
        print(f"Warning: {len(skipped_end_reason)} reads skipped due to end_reason filter (not signal_positive):")
        for read_id, end_reason in skipped_end_reason.items():
            print(f"  {read_id}: {end_reason}")

    if remaining_read_ids:
        print(f"Warning: {len(remaining_read_ids)} reads not found in POD5 file(s)")

    # Calculate which reads were successfully extracted (found and passed end_reason filter)
    extracted_read_ids = [read_id for read_id in read_ids if read_id not in remaining_read_ids and read_id not in skipped_end_reason]

    return extracted_count, extracted_read_ids


def extract_reads_from_bam(
    bam_path: str,
    read_ids: List[str],
    output_path: str,
) -> int:
    """
    Extract specific reads from BAM file and write to new BAM file.

    Args:
        bam_path: Path to input BAM file
        read_ids: List of read IDs to extract
        output_path: Path to output BAM file

    Returns:
        Number of reads successfully extracted
    """
    extracted_count = 0
    read_id_set = set(read_ids)

    with pysam.AlignmentFile(bam_path, "rb") as bam_in:
        # Create output BAM with same header
        with pysam.AlignmentFile(output_path, "wb", header=bam_in.header) as bam_out:
            # Iterate through all reads in the BAM
            for read in bam_in.fetch(until_eof=True):
                if read.query_name in read_id_set:
                    bam_out.write(read)
                    extracted_count += 1

    return extracted_count


def main():
    parser = argparse.ArgumentParser(
        description="Filter reads from BAM and extract from POD5 file(s)"
    )
    parser.add_argument("bam_file", help="Path to indexed BAM file")
    parser.add_argument("pod5_path", help="Path to input POD5 file or directory containing POD5 files")
    parser.add_argument(
        "--chromosome",
        default="chrIII",
        help="Target chromosome (default: chrIII)",
    )
    parser.add_argument(
        "--min-mapq",
        type=int,
        default=30,
        help="Minimum mapping quality (default: 30)",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=1000,
        help="Target read length in bp (default: 1000)",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=100,
        help="Length tolerance in bp (default: 100)",
    )
    parser.add_argument(
        "--region-size",
        type=int,
        default=10000,
        help="Genomic window size in bp (default: 10000)",
    )
    parser.add_argument(
        "--max-reads",
        type=int,
        default=100,
        help="Maximum reads to extract (default: 100)",
    )
    parser.add_argument(
        "--output",
        default="filtered_reads.pod5",
        help="Output POD5 file (default: filtered_reads.pod5)",
    )
    parser.add_argument(
        "--output-bam",
        help="Output BAM file (default: None - no BAM output)",
    )
    parser.add_argument(
        "--read-ids-file",
        help="Optional: Also write read IDs to text file",
    )

    args = parser.parse_args()

    print(f"Filtering reads from: {args.bam_file}")
    print(f"Chromosome: {args.chromosome}")
    print(f"Min MAPQ: {args.min_mapq}")
    print(
        f"Length range: {args.length - args.tolerance}-{args.length + args.tolerance} bp"
    )
    print(f"Region size: {args.region_size} bp")
    print()

    # Step 1: Filter reads from BAM
    filtered_reads = filter_reads_from_bam(
        bam_path=args.bam_file,
        chromosome=args.chromosome,
        min_mapq=args.min_mapq,
        target_length=args.length,
        length_tolerance=args.tolerance,
        region_size=args.region_size,
        max_reads=args.max_reads,
    )

    if not filtered_reads:
        return

    # Print summary
    print(f"Found {len(filtered_reads)} reads in optimal {args.region_size}bp window")
    window_start = min(r["position"] for r in filtered_reads)
    window_end = max(r["position"] for r in filtered_reads)
    print(
        f"Window: {args.chromosome}:{window_start:,}-{window_end + args.region_size:,}"
    )
    print()

    # Print read details
    print("Read ID\tPosition\tLength\tMAPQ\tStrand")
    for read in filtered_reads:
        print(
            f"{read['read_id']}\t{read['position']}\t{read['length']}\t{read['mapq']}\t{read['strand']}"
        )
    print()

    # Step 2: Extract reads from POD5
    print(f"Extracting reads from: {args.pod5_path}")
    read_ids = [r["read_id"] for r in filtered_reads]

    extracted_count, extracted_read_ids = extract_reads_from_pod5(
        pod5_path=args.pod5_path,
        read_ids=read_ids,
        output_path=args.output,
    )

    print(f"Extracted {extracted_count}/{len(read_ids)} reads to: {args.output}")

    # Step 3: Extract reads from BAM if requested
    if args.output_bam:
        print(f"\nExtracting {len(extracted_read_ids)} reads from BAM: {args.bam_file}")
        bam_extracted_count = extract_reads_from_bam(
            bam_path=args.bam_file,
            read_ids=extracted_read_ids,
            output_path=args.output_bam,
        )
        print(f"Extracted {bam_extracted_count}/{len(extracted_read_ids)} reads to: {args.output_bam}")

        # Index the output BAM file
        print(f"Indexing {args.output_bam}...")
        pysam.index(args.output_bam)
        print(f"Created index: {args.output_bam}.bai")

    # Write read IDs to file if requested
    if args.read_ids_file:
        with open(args.read_ids_file, "w") as f:
            for read_id in extracted_read_ids:
                f.write(f"{read_id}\n")
        print(f"Read IDs written to: {args.read_ids_file}")


if __name__ == "__main__":
    main()
