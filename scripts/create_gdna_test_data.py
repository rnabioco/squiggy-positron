#!/usr/bin/env python3
"""
Script to extract a curated subset of reads from genomic DNA POD5/BAM files for testing.

This creates a smaller test dataset with long reads from a high-coverage region,
useful for testing squiggy with genomic nanopore data.

Usage:
    # Auto-detect high-coverage region:
    python create_gdna_test_data.py \
        --input-pod5-dir /path/to/pod5_pass/ \
        --input-bam /path/to/alignments.bam \
        --output-dir squiggy/data/

    # Specify a region:
    python create_gdna_test_data.py \
        --input-pod5-dir /path/to/pod5_pass/ \
        --input-bam /path/to/alignments.bam \
        --region "chr1:1000000-2000000" \
        --output-dir squiggy/data/
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import pod5
import pysam


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract subset of genomic DNA POD5/BAM data for testing"
    )
    parser.add_argument(
        "-i",
        "--input-pod5-dir",
        required=True,
        type=Path,
        help="Directory containing POD5 files (handles unmerged data)",
    )
    parser.add_argument(
        "-b",
        "--input-bam",
        required=True,
        type=Path,
        help="Input BAM file with alignments (must be indexed)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Genomic region to extract (e.g., 'chr1:1000000-2000000'). "
        "If not specified, auto-detects a high-coverage region.",
    )
    parser.add_argument(
        "--num-reads",
        type=int,
        default=10,
        help="Target number of reads to extract (default: 10)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory for test files (default: current directory)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="gdna",
        help="Prefix for output files (default: gdna)",
    )
    parser.add_argument(
        "--min-mapq",
        type=int,
        default=20,
        help="Minimum mapping quality (default: 20)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=1000,
        help="Minimum read length in bp (default: 1000)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=100000,
        help="Window size for coverage scanning in bp (default: 100000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    return parser.parse_args()


def parse_region(region_str):
    """
    Parse a region string like 'chr1:1000000-2000000'.

    Returns:
        Tuple of (chrom, start, end) or None if invalid
    """
    try:
        chrom, coords = region_str.split(":")
        start, end = coords.split("-")
        return (chrom, int(start), int(end))
    except (ValueError, AttributeError):
        return None


def find_high_coverage_region(bam_path, min_mapq, min_length, window_size):
    """
    Scan BAM to find a region with good coverage of long reads.

    Args:
        bam_path: Path to indexed BAM file
        min_mapq: Minimum mapping quality
        min_length: Minimum read length
        window_size: Size of windows to scan

    Returns:
        Tuple of (chrom, start, end) for best region
    """
    print(f"Scanning BAM for high-coverage regions (window size: {window_size:,} bp)...")

    # Track coverage per window
    window_counts = defaultdict(lambda: defaultdict(int))

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        # Get reference lengths
        ref_lengths = dict(zip(bam.references, bam.lengths))

        for read in bam.fetch():
            if read.is_unmapped:
                continue
            if read.mapping_quality < min_mapq:
                continue
            if read.query_length < min_length:
                continue

            # Determine which window this read falls into
            chrom = read.reference_name
            window_idx = read.reference_start // window_size
            window_counts[chrom][window_idx] += 1

    # Find the window with most long reads
    best_chrom = None
    best_window = None
    best_count = 0

    for chrom, windows in window_counts.items():
        for window_idx, count in windows.items():
            if count > best_count:
                best_count = count
                best_chrom = chrom
                best_window = window_idx

    if best_chrom is None:
        return None

    # Convert window index back to coordinates
    start = best_window * window_size
    end = min(start + window_size, ref_lengths.get(best_chrom, start + window_size))

    print(f"  Best region: {best_chrom}:{start:,}-{end:,} ({best_count} long reads)")

    return (best_chrom, start, end)


def collect_candidate_reads(bam_path, region, min_mapq, min_length):
    """
    Collect candidate reads from specified region.

    Args:
        bam_path: Path to BAM file
        region: Tuple of (chrom, start, end)
        min_mapq: Minimum mapping quality
        min_length: Minimum read length

    Returns:
        List of tuples: (read_name, read_length, start_pos, end_pos)
    """
    chrom, start, end = region
    candidates = []

    print(f"Collecting candidate reads from {chrom}:{start:,}-{end:,}")

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for read in bam.fetch(chrom, start, end):
            if read.is_unmapped:
                continue
            if read.mapping_quality < min_mapq:
                continue
            if read.query_length < min_length:
                continue

            candidates.append(
                (
                    read.query_name,
                    read.query_length,
                    read.reference_start,
                    read.reference_end,
                )
            )

    print(f"  Found {len(candidates)} candidate reads")

    # Sort by length (longest first) for reporting
    candidates.sort(key=lambda x: x[1], reverse=True)

    if candidates:
        lengths = [c[1] for c in candidates]
        print(f"  Length range: {min(lengths):,} - {max(lengths):,} bp")
        print(f"  Median length: {sorted(lengths)[len(lengths)//2]:,} bp")

    return candidates


def select_reads(candidates, num_reads, seed):
    """
    Select reads from candidates, prioritizing longer reads.

    Args:
        candidates: List of (read_name, length, start, end) tuples
        num_reads: Target number of reads
        seed: Random seed

    Returns:
        Set of selected read names
    """
    random.seed(seed)

    if len(candidates) <= num_reads:
        selected = {c[0] for c in candidates}
        print(f"\nSelected all {len(selected)} reads (fewer than target)")
        return selected

    # Take a mix: half from longest, half random
    # This ensures we get some impressive long reads while also having variety
    n_longest = num_reads // 2
    n_random = num_reads - n_longest

    # Candidates already sorted by length (longest first)
    longest = [c[0] for c in candidates[:n_longest]]

    # Random sample from the rest
    remaining = [c[0] for c in candidates[n_longest:]]
    random_picks = random.sample(remaining, min(n_random, len(remaining)))

    selected = set(longest + random_picks)

    print(f"\nSelected {len(selected)} reads:")
    print(f"  {n_longest} longest reads")
    print(f"  {len(random_picks)} randomly sampled")

    # Report lengths of selected reads
    selected_info = [c for c in candidates if c[0] in selected]
    lengths = [c[1] for c in selected_info]
    print(f"  Selected length range: {min(lengths):,} - {max(lengths):,} bp")

    return selected


def find_pod5_files(pod5_dir):
    """
    Find all POD5 files in directory.

    Args:
        pod5_dir: Directory to search

    Returns:
        List of Path objects for POD5 files
    """
    pod5_files = list(pod5_dir.glob("*.pod5"))
    print(f"Found {len(pod5_files)} POD5 files in {pod5_dir}")
    return pod5_files


def extract_pod5_subset(pod5_dir, output_path, selected_reads, target_count=None):
    """
    Extract subset of reads from POD5 files in directory.

    Iterates through POD5 files individually to handle corrupted files gracefully.

    Args:
        pod5_dir: Directory containing POD5 files
        output_path: Output POD5 file
        selected_reads: Set of read IDs to extract (candidates)
        target_count: Stop after extracting this many reads (default: all)

    Returns:
        Set of read IDs that were actually extracted
    """
    print(f"\nExtracting POD5 subset to: {output_path}")

    pod5_files = find_pod5_files(pod5_dir)

    if not pod5_files:
        print("  ERROR: No POD5 files found!")
        return set()

    if target_count is None:
        target_count = len(selected_reads)

    print(
        f"  Searching for up to {target_count} reads "
        f"(from {len(selected_reads)} candidates) across {len(pod5_files)} files..."
    )

    # Convert to set for O(1) lookup
    candidate_reads = set(selected_reads)
    extracted_reads = set()
    corrupted_files = []

    with pod5.Writer(str(output_path)) as writer:
        for pod5_file in pod5_files:
            if len(extracted_reads) >= target_count:
                break  # Reached target

            try:
                with pod5.Reader(str(pod5_file)) as reader:
                    for read in reader.reads(selection=candidate_reads, missing_ok=True):
                        read_id = str(read.read_id)
                        if read_id in candidate_reads and read_id not in extracted_reads:
                            writer.add_read(read.to_read())
                            extracted_reads.add(read_id)
                            candidate_reads.discard(read_id)
                            print(
                                f"    Extracted read {read_id[:8]}... "
                                f"({len(extracted_reads)}/{target_count})"
                            )
                            if len(extracted_reads) >= target_count:
                                break
            except RuntimeError as e:
                if "Invalid signature" in str(e) or "IOError" in str(e):
                    corrupted_files.append(pod5_file.name)
                else:
                    raise

    if corrupted_files:
        print(f"  WARNING: Skipped {len(corrupted_files)} corrupted POD5 files")

    print(f"  Completed: Extracted {len(extracted_reads)} reads")

    if len(extracted_reads) < target_count:
        missing_count = target_count - len(extracted_reads)
        print(f"  WARNING: Only found {len(extracted_reads)} of {target_count} target reads")

    return extracted_reads


def extract_bam_subset(input_path, output_path, selected_reads, region):
    """
    Extract subset of BAM file containing only selected reads.

    Args:
        input_path: Input BAM file
        output_path: Output BAM file
        selected_reads: Set of read names to extract
        region: Tuple of (chrom, start, end) for the region
    """
    print(f"\nExtracting BAM subset to: {output_path}")

    chrom, start, end = region
    reads_written = 0

    with pysam.AlignmentFile(str(input_path), "rb") as bam_in:
        # Create output BAM with same header
        with pysam.AlignmentFile(str(output_path), "wb", header=bam_in.header) as bam_out:
            for read in bam_in.fetch(chrom, start, end):
                if read.query_name in selected_reads:
                    bam_out.write(read)
                    reads_written += 1

    print(f"  Extracted {reads_written} alignments")

    # Index the output BAM
    print("  Indexing BAM file...")
    pysam.index(str(output_path))
    print(f"  Created index: {output_path}.bai")

    return reads_written


def generate_statistics(bam_path, pod5_path, region):
    """
    Generate statistics about the extracted dataset.

    Args:
        bam_path: Path to output BAM file
        pod5_path: Path to output POD5 file
        region: Tuple of (chrom, start, end)

    Returns:
        Dict with statistics
    """
    chrom, reg_start, reg_end = region

    stats = {
        "region": f"{chrom}:{reg_start:,}-{reg_end:,}",
        "total_reads": 0,
        "lengths": [],
        "positions": [],
    }

    # Analyze BAM
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for read in bam.fetch():
            stats["total_reads"] += 1
            stats["lengths"].append(read.query_length)
            if read.is_mapped:
                stats["positions"].append((read.reference_start, read.reference_end))

    if stats["lengths"]:
        stats["min_length"] = min(stats["lengths"])
        stats["max_length"] = max(stats["lengths"])
        stats["median_length"] = sorted(stats["lengths"])[len(stats["lengths"]) // 2]
        stats["total_bases"] = sum(stats["lengths"])

    if stats["positions"]:
        all_starts = [p[0] for p in stats["positions"]]
        all_ends = [p[1] for p in stats["positions"]]
        stats["coverage_start"] = min(all_starts)
        stats["coverage_end"] = max(all_ends)
        stats["coverage_span"] = stats["coverage_end"] - stats["coverage_start"]

    # Get file sizes
    stats["pod5_size_mb"] = pod5_path.stat().st_size / (1024 * 1024)
    stats["bam_size_kb"] = bam_path.stat().st_size / 1024

    return stats


def print_summary(stats, output_pod5, output_bam):
    """Print a summary of the extracted data."""
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print(f"\nOutput files:")
    print(f"  POD5: {output_pod5} ({stats['pod5_size_mb']:.2f} MB)")
    print(f"  BAM:  {output_bam} ({stats['bam_size_kb']:.1f} KB)")

    print(f"\nDataset summary:")
    print(f"  Region:        {stats['region']}")
    print(f"  Total reads:   {stats['total_reads']}")
    print(f"  Total bases:   {stats.get('total_bases', 0):,} bp")

    if stats["lengths"]:
        print(f"\nRead lengths:")
        print(f"  Min:    {stats['min_length']:,} bp")
        print(f"  Median: {stats['median_length']:,} bp")
        print(f"  Max:    {stats['max_length']:,} bp")

    if stats.get("coverage_span"):
        print(f"\nCoverage:")
        print(f"  Span: {stats['coverage_start']:,} - {stats['coverage_end']:,}")
        print(f"  Width: {stats['coverage_span']:,} bp")


def main():
    """Main execution function."""
    args = parse_args()

    # Validate inputs
    if not args.input_pod5_dir.exists():
        print(f"Error: Input POD5 directory not found: {args.input_pod5_dir}")
        sys.exit(1)

    if not args.input_pod5_dir.is_dir():
        print(f"Error: --input-pod5-dir must be a directory: {args.input_pod5_dir}")
        sys.exit(1)

    if not args.input_bam.exists():
        print(f"Error: Input BAM file not found: {args.input_bam}")
        sys.exit(1)

    # Check BAM index
    bai_path = Path(str(args.input_bam) + ".bai")
    csi_path = Path(str(args.input_bam) + ".csi")
    if not bai_path.exists() and not csi_path.exists():
        print(f"Error: BAM index not found. Please run: samtools index {args.input_bam}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    output_pod5 = args.output_dir / f"{args.output_prefix}_reads.pod5"
    output_bam = args.output_dir / f"{args.output_prefix}_mappings.bam"

    print("=" * 70)
    print("Genomic DNA Test Data Extraction")
    print("=" * 70)
    print(f"Input POD5 dir:   {args.input_pod5_dir}")
    print(f"Input BAM:        {args.input_bam}")
    print(f"Target reads:     {args.num_reads}")
    print(f"Min MAPQ:         {args.min_mapq}")
    print(f"Min length:       {args.min_length:,} bp")
    print(f"Random seed:      {args.seed}")
    print(f"Output POD5:      {output_pod5}")
    print(f"Output BAM:       {output_bam}")
    print("=" * 70)

    # Step 1: Determine region
    if args.region:
        region = parse_region(args.region)
        if region is None:
            print(f"Error: Invalid region format: {args.region}")
            print("  Expected format: chr1:1000000-2000000")
            sys.exit(1)
        print(f"\nUsing specified region: {args.region}")
    else:
        print("\nAuto-detecting high-coverage region...")
        region = find_high_coverage_region(
            args.input_bam, args.min_mapq, args.min_length, args.window_size
        )
        if region is None:
            print("Error: Could not find any region with qualifying reads")
            sys.exit(1)

    # Step 2: Collect candidate reads from region
    candidates = collect_candidate_reads(
        args.input_bam, region, args.min_mapq, args.min_length
    )

    if not candidates:
        print("\nError: No candidate reads found in region")
        sys.exit(1)

    # Step 3: Select 3x target reads (to account for reads missing from POD5)
    oversample_factor = 3
    candidate_count = min(len(candidates), args.num_reads * oversample_factor)
    selected_reads = select_reads(candidates, candidate_count, args.seed)

    # Step 4: Extract POD5 subset first (stop when we hit target)
    extracted_reads = extract_pod5_subset(
        args.input_pod5_dir, output_pod5, selected_reads, target_count=args.num_reads
    )

    if not extracted_reads:
        print("\nError: No reads were extracted from POD5 files")
        sys.exit(1)

    # Step 5: Extract BAM subset (only for reads we actually got from POD5)
    extract_bam_subset(args.input_bam, output_bam, extracted_reads, region)

    # Step 6: Generate and print statistics
    stats = generate_statistics(output_bam, output_pod5, region)
    print_summary(stats, output_pod5, output_bam)

    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    main()
