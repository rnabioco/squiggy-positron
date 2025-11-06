#!/usr/bin/env python3
"""
Script to extract a curated subset of reads from POD5/BAM files for testing.

This creates a smaller test dataset with reads mapping to specific tRNA references,
useful for testing the multi-read aggregate plotting feature.

Usage:
    python create_test_data.py \
        --input-pod5 /path/to/full_data.pod5 \
        --input-bam /path/to/full_mappings.bam \
        --references "tRNA-Phe-GAA,tRNA-Tyr-GUA" \
        --reads-per-ref 60 \
        --output-dir tests/data/
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
        description="Extract subset of POD5/BAM data for testing"
    )
    parser.add_argument(
        "--input-pod5",
        required=True,
        type=Path,
        help="Input POD5 file with all reads",
    )
    parser.add_argument(
        "--input-bam",
        required=True,
        type=Path,
        help="Input BAM file with alignments",
    )
    parser.add_argument(
        "--references",
        required=True,
        type=str,
        help="Comma-separated list of reference names to extract (e.g., 'tRNA-Phe-GAA,tRNA-Tyr-GUA')",
    )
    parser.add_argument(
        "--reads-per-ref",
        type=int,
        default=60,
        help="Target number of reads per reference (default: 60)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/data"),
        help="Output directory for test files (default: tests/data/)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="trna",
        help="Prefix for output files (default: trna)",
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
        default=50,
        help="Minimum alignment length (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    return parser.parse_args()


def collect_candidate_reads(bam_path, target_refs, min_mapq, min_length):
    """
    Collect candidate reads mapping to target references.

    Args:
        bam_path: Path to BAM file
        target_refs: Set of reference names to extract
        min_mapq: Minimum mapping quality
        min_length: Minimum alignment length

    Returns:
        Dict mapping reference names to lists of read names
    """
    candidates = defaultdict(list)

    print(f"Scanning BAM file: {bam_path}")
    print(f"Target references: {', '.join(sorted(target_refs))}")

    with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            # Skip if not mapped
            if not read.is_mapped:
                continue

            # Skip if not mapping to target reference
            if read.reference_name not in target_refs:
                continue

            # Check quality filters
            if read.mapping_quality < min_mapq:
                continue

            aln_length = read.reference_end - read.reference_start
            if aln_length < min_length:
                continue

            # Check for move table
            if not read.has_tag("mv"):
                print(
                    f"Warning: Read {read.query_name} has no move table, skipping"
                )
                continue

            candidates[read.reference_name].append(read.query_name)

    # Print statistics
    print("\nFound candidate reads:")
    for ref in sorted(candidates.keys()):
        print(f"  {ref}: {len(candidates[ref])} reads")

    return candidates


def select_reads(candidates, reads_per_ref, seed):
    """
    Randomly sample reads from candidates to get target number per reference.

    Args:
        candidates: Dict mapping reference names to lists of read names
        reads_per_ref: Target number of reads per reference
        seed: Random seed

    Returns:
        Set of selected read names
    """
    random.seed(seed)
    selected = set()

    print(f"\nSelecting {reads_per_ref} reads per reference...")

    for ref, read_names in candidates.items():
        if len(read_names) <= reads_per_ref:
            # Take all reads if we don't have enough
            selected.update(read_names)
            print(
                f"  {ref}: Selected all {len(read_names)} reads (fewer than target)"
            )
        else:
            # Random sample
            sampled = random.sample(read_names, reads_per_ref)
            selected.update(sampled)
            print(f"  {ref}: Randomly sampled {len(sampled)} reads")

    print(f"\nTotal selected reads: {len(selected)}")
    return selected


def extract_pod5_subset(input_path, output_path, selected_reads):
    """
    Extract subset of POD5 file containing only selected reads.

    Args:
        input_path: Input POD5 file
        output_path: Output POD5 file
        selected_reads: Set of read IDs to extract
    """
    print(f"\nExtracting POD5 subset to: {output_path}")

    reads_found = 0
    reads_written = 0

    with pod5.Reader(str(input_path)) as reader:
        with pod5.Writer(str(output_path)) as writer:
            for read in reader.reads():
                reads_found += 1

                # Convert UUID to string for comparison
                read_id = str(read.read_id)

                if read_id in selected_reads:
                    writer.add_read(read.to_read())
                    reads_written += 1

                # Progress indicator
                if reads_found % 1000 == 0:
                    print(f"  Processed {reads_found} reads, extracted {reads_written}")

    print(f"  Completed: Extracted {reads_written} reads from {reads_found} total")


def extract_bam_subset(input_path, output_path, selected_reads):
    """
    Extract subset of BAM file containing only selected reads.

    Args:
        input_path: Input BAM file
        output_path: Output BAM file
        selected_reads: Set of read names to extract
    """
    print(f"\nExtracting BAM subset to: {output_path}")

    reads_written = 0

    with pysam.AlignmentFile(str(input_path), "rb", check_sq=False) as bam_in:
        # Create output BAM with same header
        with pysam.AlignmentFile(
            str(output_path), "wb", header=bam_in.header
        ) as bam_out:
            for read in bam_in.fetch(until_eof=True):
                if read.query_name in selected_reads:
                    bam_out.write(read)
                    reads_written += 1

    print(f"  Extracted {reads_written} alignments")

    # Index the output BAM
    print("  Indexing BAM file...")
    pysam.index(str(output_path))
    print(f"  Created index: {output_path}.bai")


def generate_statistics(bam_path, pod5_path, output_refs):
    """
    Generate statistics about the extracted dataset.

    Args:
        bam_path: Path to output BAM file
        pod5_path: Path to output POD5 file
        output_refs: Set of reference names

    Returns:
        Dict with statistics
    """
    stats = {
        "total_reads": 0,
        "refs": defaultdict(lambda: {"count": 0, "total_length": 0, "positions": []}),
    }

    # Analyze BAM
    with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            stats["total_reads"] += 1

            if read.is_mapped and read.reference_name in output_refs:
                ref_stats = stats["refs"][read.reference_name]
                ref_stats["count"] += 1
                ref_stats["total_length"] += read.reference_end - read.reference_start
                ref_stats["positions"].append(
                    (read.reference_start, read.reference_end)
                )

    # Calculate coverage ranges
    for _ref, ref_stats in stats["refs"].items():
        if ref_stats["positions"]:
            all_starts = [p[0] for p in ref_stats["positions"]]
            all_ends = [p[1] for p in ref_stats["positions"]]
            ref_stats["min_pos"] = min(all_starts)
            ref_stats["max_pos"] = max(all_ends)
            ref_stats["span"] = ref_stats["max_pos"] - ref_stats["min_pos"]
            ref_stats["avg_length"] = ref_stats["total_length"] // ref_stats["count"]

    # Get POD5 file size
    stats["pod5_size_mb"] = pod5_path.stat().st_size / (1024 * 1024)
    stats["bam_size_kb"] = bam_path.stat().st_size / 1024

    return stats


def update_readme(output_dir, prefix, stats, target_refs):
    """
    Update README.md with information about the new test files.

    Args:
        output_dir: Output directory
        prefix: File prefix
        stats: Statistics dict
        target_refs: List of target reference names
    """
    readme_path = output_dir / "README.md"

    # Generate documentation section
    doc_section = f"""
### {prefix.upper()} Files (Multi-read testing)

- **{prefix}_reads.pod5** - POD5 file with reads mapping to multiple tRNA genes
  - Used by: Multi-read aggregate plotting tests
  - Size: ~{stats['pod5_size_mb']:.1f}MB
  - Total reads: {stats['total_reads']}
  - Contents: Nanopore signal data for testing aggregate signal analysis

- **{prefix}_mappings.bam** - BAM file with alignments to tRNA references
  - Used by: Multi-read aggregate plotting tests
  - Size: ~{stats['bam_size_kb']:.0f}KB
  - References: {', '.join(sorted(target_refs))}
  - Read distribution:
"""

    # Add per-reference statistics
    for ref in sorted(stats["refs"].keys()):
        ref_stats = stats["refs"][ref]
        doc_section += f"    - {ref}: {ref_stats['count']} reads, "
        doc_section += f"coverage {ref_stats['min_pos']:,}-{ref_stats['max_pos']:,} "
        doc_section += f"({ref_stats['span']:,} bp span)\n"

    doc_section += "  - Note: All reads have move table (mv tag) for event-aligned plotting\n"

    print("\n" + "=" * 70)
    print("README.md section to add:")
    print("=" * 70)
    print(doc_section)
    print("=" * 70)
    print(
        f"\nPlease manually add the above section to {readme_path} in the appropriate location."
    )


def main():
    """Main execution function."""
    args = parse_args()

    # Validate inputs
    if not args.input_pod5.exists():
        print(f"Error: Input POD5 file not found: {args.input_pod5}")
        sys.exit(1)

    if not args.input_bam.exists():
        print(f"Error: Input BAM file not found: {args.input_bam}")
        sys.exit(1)

    # Parse reference names
    target_refs = {ref.strip() for ref in args.references.split(",")}

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    output_pod5 = args.output_dir / f"{args.output_prefix}_reads.pod5"
    output_bam = args.output_dir / f"{args.output_prefix}_mappings.bam"

    print("=" * 70)
    print("Test Data Extraction Configuration")
    print("=" * 70)
    print(f"Input POD5:       {args.input_pod5}")
    print(f"Input BAM:        {args.input_bam}")
    print(f"Target refs:      {', '.join(sorted(target_refs))}")
    print(f"Reads per ref:    {args.reads_per_ref}")
    print(f"Min MAPQ:         {args.min_mapq}")
    print(f"Min length:       {args.min_length}")
    print(f"Random seed:      {args.seed}")
    print(f"Output POD5:      {output_pod5}")
    print(f"Output BAM:       {output_bam}")
    print("=" * 70)

    # Step 1: Collect candidate reads
    candidates = collect_candidate_reads(
        args.input_bam, target_refs, args.min_mapq, args.min_length
    )

    if not candidates:
        print("\nError: No candidate reads found matching criteria")
        sys.exit(1)

    # Step 2: Select reads
    selected_reads = select_reads(candidates, args.reads_per_ref, args.seed)

    # Step 3: Extract POD5 subset
    extract_pod5_subset(args.input_pod5, output_pod5, selected_reads)

    # Step 4: Extract BAM subset
    extract_bam_subset(args.input_bam, output_bam, selected_reads)

    # Step 5: Generate statistics
    print("\n" + "=" * 70)
    print("Generating statistics...")
    print("=" * 70)
    stats = generate_statistics(output_bam, output_pod5, target_refs)

    print("\nFinal statistics:")
    print(f"  Total reads:    {stats['total_reads']}")
    print(f"  POD5 size:      {stats['pod5_size_mb']:.1f} MB")
    print(f"  BAM size:       {stats['bam_size_kb']:.0f} KB")
    print("\n  Per-reference breakdown:")
    for ref in sorted(stats["refs"].keys()):
        ref_stats = stats["refs"][ref]
        print(f"    {ref}:")
        print(f"      Reads:          {ref_stats['count']}")
        print(f"      Avg length:     {ref_stats['avg_length']} bp")
        print(
            f"      Coverage range: {ref_stats['min_pos']:,} - {ref_stats['max_pos']:,}"
        )
        print(f"      Span:           {ref_stats['span']:,} bp")

    # Step 6: Generate README documentation
    update_readme(args.output_dir, args.output_prefix, stats, target_refs)

    print("\n" + "=" * 70)
    print("COMPLETE: Test data extraction successful!")
    print("=" * 70)


if __name__ == "__main__":
    main()
