#!/usr/bin/env python3
"""
Filter reads from BAM file and extract matching reads from POD5 file.

Usage:
    python filter_reads.py <bam_file> <pod5_file> [options]

Example:
    python filter_reads.py alignments.bam reads.pod5 --chromosome chrIII --output filtered_reads.pod5
"""

import argparse
import pysam
import pod5
from typing import List, Dict, Any


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
    pod5_path: str,
    read_ids: List[str],
    output_path: str,
) -> int:
    """
    Extract specific reads from POD5 file and write to new POD5 file.

    Args:
        pod5_path: Path to input POD5 file
        read_ids: List of read IDs to extract
        output_path: Path to output POD5 file

    Returns:
        Number of reads successfully extracted
    """
    extracted_count = 0

    with pod5.Reader(pod5_path) as reader:
        with pod5.Writer(output_path) as writer:
            # Use selection parameter to efficiently get only the requested reads
            for read in reader.reads(selection=read_ids, missing_ok=True):
                writer.add_read(read)
                extracted_count += 1

    return extracted_count


def main():
    parser = argparse.ArgumentParser(
        description="Filter reads from BAM and extract from POD5"
    )
    parser.add_argument("bam_file", help="Path to indexed BAM file")
    parser.add_argument("pod5_file", help="Path to input POD5 file")
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
    print(f"Extracting reads from: {args.pod5_file}")
    read_ids = [r["read_id"] for r in filtered_reads]

    extracted_count = extract_reads_from_pod5(
        pod5_path=args.pod5_file,
        read_ids=read_ids,
        output_path=args.output,
    )

    print(f"Extracted {extracted_count}/{len(read_ids)} reads to: {args.output}")

    # Write read IDs to file if requested
    if args.read_ids_file:
        with open(args.read_ids_file, "w") as f:
            for read_id in read_ids:
                f.write(f"{read_id}\n")
        print(f"Read IDs written to: {args.read_ids_file}")


if __name__ == "__main__":
    main()
