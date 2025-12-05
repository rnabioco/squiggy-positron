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

import numpy as np
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
        "-d",
        "--min-depth",
        type=int,
        default=10,
        help="Minimum per-base coverage depth required (default: 10)",
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


def find_high_coverage_position(bam_path, min_mapq, min_length, window_size, min_depth=10):
    """
    Scan BAM to find a position with high per-base coverage.

    Uses count_coverage() to find actual per-base depth, then identifies
    positions where coverage meets the minimum depth requirement.

    Args:
        bam_path: Path to indexed BAM file
        min_mapq: Minimum mapping quality
        min_length: Minimum read length
        window_size: Size of windows to scan for candidates
        min_depth: Minimum per-base coverage depth required

    Returns:
        Tuple of (chrom, position, depth) for best position, or None
    """
    print(f"Scanning BAM for positions with >= {min_depth}x per-base coverage...")

    # First, find candidate windows with enough reads to potentially have high coverage
    window_counts = defaultdict(lambda: defaultdict(int))

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        ref_lengths = dict(zip(bam.references, bam.lengths, strict=False))

        for read in bam.fetch():
            if read.is_unmapped:
                continue
            if read.mapping_quality < min_mapq:
                continue
            if read.query_length < min_length:
                continue

            chrom = read.reference_name
            # Count reads in overlapping windows (not just start position)
            start_window = read.reference_start // window_size
            end_window = read.reference_end // window_size
            for w in range(start_window, end_window + 1):
                window_counts[chrom][w] += 1

    # Sort windows by read count (most reads first)
    candidate_windows = []
    for chrom, windows in window_counts.items():
        for window_idx, count in windows.items():
            if count >= min_depth:  # Must have at least min_depth reads to have that coverage
                candidate_windows.append((chrom, window_idx, count, ref_lengths.get(chrom, 0)))

    candidate_windows.sort(key=lambda x: x[2], reverse=True)

    if not candidate_windows:
        print(f"  No windows found with >= {min_depth} overlapping reads")
        return None

    print(f"  Found {len(candidate_windows)} candidate windows, checking per-base coverage...")

    # Check actual per-base coverage in top candidate windows
    best_result = None

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for chrom, window_idx, _read_count, ref_len in candidate_windows[:20]:  # Check top 20
            start = window_idx * window_size
            end = min(start + window_size, ref_len)

            # Get per-base coverage (returns tuple of 4 arrays: A, C, G, T)
            try:
                coverage = bam.count_coverage(
                    chrom, start, end, quality_threshold=min_mapq, read_callback="nofilter"
                )
            except ValueError:
                continue

            # Sum all bases to get total depth per position
            total_depth = np.array(coverage[0]) + np.array(coverage[1]) + \
                         np.array(coverage[2]) + np.array(coverage[3])

            # Find positions meeting minimum depth
            high_cov_positions = np.where(total_depth >= min_depth)[0]

            if len(high_cov_positions) > 0:
                # Find the position with maximum coverage
                max_idx = np.argmax(total_depth)
                max_depth = total_depth[max_idx]
                max_pos = start + max_idx

                if best_result is None or max_depth > best_result[2]:
                    best_result = (chrom, max_pos, int(max_depth))
                    print(f"    {chrom}:{max_pos:,} has {max_depth}x coverage")

                    # If we found a great position, stop searching
                    if max_depth >= min_depth * 2:
                        break

    if best_result:
        print(f"  Best position: {best_result[0]}:{best_result[1]:,} ({best_result[2]}x coverage)")

    return best_result


def find_high_coverage_region(bam_path, min_mapq, min_length, window_size, min_depth=10):
    """
    Find a region centered on a high-coverage position.

    Args:
        bam_path: Path to indexed BAM file
        min_mapq: Minimum mapping quality
        min_length: Minimum read length
        window_size: Size of region to return around high-coverage position
        min_depth: Minimum per-base coverage depth required

    Returns:
        Tuple of (chrom, start, end) for region, or None
    """
    result = find_high_coverage_position(bam_path, min_mapq, min_length, window_size, min_depth)

    if result is None:
        return None

    chrom, position, depth = result

    # Create a region centered on the high-coverage position
    half_window = window_size // 2
    start = max(0, position - half_window)
    end = position + half_window

    return (chrom, start, end)


def collect_candidate_reads(bam_path, region, min_mapq, min_length, target_position=None):
    """
    Collect candidate reads from specified region.

    If target_position is provided, only collects reads that overlap that position.
    This ensures all selected reads share a common overlapping point.

    Args:
        bam_path: Path to BAM file
        region: Tuple of (chrom, start, end)
        min_mapq: Minimum mapping quality
        min_length: Minimum read length
        target_position: Optional position that all reads must overlap

    Returns:
        List of tuples: (read_name, read_length, start_pos, end_pos)
    """
    chrom, start, end = region
    candidates = []

    if target_position is not None:
        print(f"Collecting reads overlapping position {chrom}:{target_position:,}")
    else:
        print(f"Collecting candidate reads from {chrom}:{start:,}-{end:,}")

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for read in bam.fetch(chrom, start, end):
            if read.is_unmapped:
                continue
            if read.mapping_quality < min_mapq:
                continue
            if read.query_length < min_length:
                continue

            # If target position specified, only include reads that overlap it
            if target_position is not None:
                if not (read.reference_start <= target_position < read.reference_end):
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


def generate_statistics(bam_path, pod5_path, region, target_position=None):
    """
    Generate statistics about the extracted dataset.

    Args:
        bam_path: Path to output BAM file
        pod5_path: Path to output POD5 file
        region: Tuple of (chrom, start, end)
        target_position: Optional target position that all reads overlap

    Returns:
        Dict with statistics
    """
    chrom, reg_start, reg_end = region

    stats = {
        "region": f"{chrom}:{reg_start:,}-{reg_end:,}",
        "target_position": target_position,
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

        # Calculate actual coverage at target position
        if target_position is not None:
            try:
                coverage = bam.count_coverage(
                    chrom, target_position, target_position + 1
                )
                depth = sum(c[0] for c in coverage)
                stats["target_depth"] = depth
            except (ValueError, IndexError):
                stats["target_depth"] = stats["total_reads"]

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

    print("\nOutput files:")
    print(f"  POD5: {output_pod5} ({stats['pod5_size_mb']:.2f} MB)")
    print(f"  BAM:  {output_bam} ({stats['bam_size_kb']:.1f} KB)")

    print("\nDataset summary:")
    print(f"  Region:        {stats['region']}")
    if stats.get("target_position") is not None:
        print(f"  Target pos:    {stats['target_position']:,}")
        if stats.get("target_depth") is not None:
            print(f"  Depth @ pos:   {stats['target_depth']}x")
    print(f"  Total reads:   {stats['total_reads']}")
    print(f"  Total bases:   {stats.get('total_bases', 0):,} bp")

    if stats["lengths"]:
        print("\nRead lengths:")
        print(f"  Min:    {stats['min_length']:,} bp")
        print(f"  Median: {stats['median_length']:,} bp")
        print(f"  Max:    {stats['max_length']:,} bp")

    if stats.get("coverage_span"):
        print("\nCoverage:")
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
    print(f"Min depth:        {args.min_depth}x per-base coverage")
    print(f"Random seed:      {args.seed}")
    print(f"Output POD5:      {output_pod5}")
    print(f"Output BAM:       {output_bam}")
    print("=" * 70)

    # Step 1: Determine region and target position
    target_position = None
    if args.region:
        region = parse_region(args.region)
        if region is None:
            print(f"Error: Invalid region format: {args.region}")
            print("  Expected format: chr1:1000000-2000000")
            sys.exit(1)
        print(f"\nUsing specified region: {args.region}")
    else:
        print("\nAuto-detecting high-coverage position...")
        position_result = find_high_coverage_position(
            args.input_bam, args.min_mapq, args.min_length, args.window_size, args.min_depth
        )
        if position_result is None:
            print(f"Error: Could not find any position with >= {args.min_depth}x coverage")
            sys.exit(1)

        chrom, target_position, depth = position_result

        # Create region centered on the high-coverage position
        half_window = args.window_size // 2
        start = max(0, target_position - half_window)
        end = target_position + half_window
        region = (chrom, start, end)

    # Step 2: Collect candidate reads that overlap the target position
    candidates = collect_candidate_reads(
        args.input_bam, region, args.min_mapq, args.min_length, target_position
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
    stats = generate_statistics(output_bam, output_pod5, region, target_position)
    print_summary(stats, output_pod5, output_bam)

    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    main()
