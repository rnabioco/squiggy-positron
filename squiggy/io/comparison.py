"""Comparison functions for multi-sample workflows"""

from .kernel import squiggy_kernel


def get_common_reads(sample_names: list[str]) -> set[str]:
    """
    Get reads that are present in all specified samples

    Finds the intersection of read IDs across multiple samples.

    Args:
        sample_names: List of sample names to compare

    Returns:
        Set of read IDs present in all samples

    Raises:
        ValueError: If any sample name not found

    Examples:
        >>> from squiggy import get_common_reads
        >>> common = get_common_reads(['model_v4.2', 'model_v5.0'])
        >>> print(f"Common reads: {len(common)}")
    """
    if not sample_names:
        return set()

    # Get first sample
    first_sample = squiggy_kernel.get_sample(sample_names[0])
    if first_sample is None:
        raise ValueError(f"Sample '{sample_names[0]}' not found")

    # Start with reads from first sample
    common = set(first_sample._read_ids)

    # Intersect with remaining samples
    for name in sample_names[1:]:
        sample = squiggy_kernel.get_sample(name)
        if sample is None:
            raise ValueError(f"Sample '{name}' not found")
        common &= set(sample._read_ids)

    return common


def get_unique_reads(
    sample_name: str, exclude_samples: list[str] | None = None
) -> set[str]:
    """
    Get reads unique to a sample (not in other samples)

    Finds reads that are only in the specified sample.

    Args:
        sample_name: Sample to find unique reads for
        exclude_samples: Samples to exclude from (default: all other samples)

    Returns:
        Set of read IDs unique to the sample

    Raises:
        ValueError: If sample not found

    Examples:
        >>> from squiggy import get_unique_reads
        >>> unique_a = get_unique_reads('model_v4.2')
        >>> unique_b = get_unique_reads('model_v5.0')
    """
    sample = squiggy_kernel.get_sample(sample_name)
    if sample is None:
        raise ValueError(f"Sample '{sample_name}' not found")

    sample_reads = set(sample._read_ids)

    # Determine which samples to exclude
    if exclude_samples is None:
        # Exclude all other samples
        exclude_samples = [
            name for name in squiggy_kernel.list_samples() if name != sample_name
        ]

    # Remove reads that appear in any excluded sample
    for exclude_name in exclude_samples:
        exclude_sample = squiggy_kernel.get_sample(exclude_name)
        if exclude_sample is None:
            raise ValueError(f"Sample '{exclude_name}' not found")
        sample_reads -= set(exclude_sample._read_ids)

    return sample_reads


def compare_samples(sample_names: list[str]) -> dict:
    """
    Compare multiple samples and return analysis

    Generates a comprehensive comparison of samples including read overlap,
    reference validation, and model provenance information.

    Args:
        sample_names: List of sample names to compare

    Returns:
        Dict with comparison results:
            - samples: List of sample names
            - read_overlap: Read ID overlap analysis
            - reference_validation: Reference compatibility (if BAM files loaded)
            - sample_info: Basic info about each sample

    Examples:
        >>> from squiggy import compare_samples
        >>> result = compare_samples(['model_v4.2', 'model_v5.0'])
        >>> print(result['read_overlap'])
    """
    from ..utils import compare_read_sets, validate_sq_headers

    # Validate samples exist
    for name in sample_names:
        if squiggy_kernel.get_sample(name) is None:
            raise ValueError(f"Sample '{name}' not found")

    result: dict = {
        "samples": sample_names,
        "sample_info": {},
        "read_overlap": {},
    }

    # Add basic info about each sample
    for name in sample_names:
        sample = squiggy_kernel.get_sample(name)
        result["sample_info"][name] = {
            "num_reads": len(sample._read_ids),
            "pod5_path": sample._pod5_path,
            "bam_path": sample._bam_path,
        }

    # Compare read sets for all pairs
    if len(sample_names) >= 2:
        for i, name_a in enumerate(sample_names):
            for name_b in sample_names[i + 1 :]:
                sample_a = squiggy_kernel.get_sample(name_a)
                sample_b = squiggy_kernel.get_sample(name_b)
                pair_key = f"{name_a}_vs_{name_b}"
                result["read_overlap"][pair_key] = compare_read_sets(
                    sample_a._read_ids, sample_b._read_ids
                )

    # Validate references if BAM files are loaded
    if len(sample_names) >= 2:
        bam_pairs = []
        for i, name_a in enumerate(sample_names):
            for name_b in sample_names[i + 1 :]:
                sample_a = squiggy_kernel.get_sample(name_a)
                sample_b = squiggy_kernel.get_sample(name_b)
                if sample_a._bam_path and sample_b._bam_path:
                    bam_pairs.append(
                        (name_a, name_b, sample_a._bam_path, sample_b._bam_path)
                    )

        if bam_pairs:
            result["reference_validation"] = {}
            for name_a, name_b, bam_a, bam_b in bam_pairs:
                pair_key = f"{name_a}_vs_{name_b}"
                try:
                    validation = validate_sq_headers(bam_a, bam_b)
                    result["reference_validation"][pair_key] = validation
                except Exception as e:
                    result["reference_validation"][pair_key] = {"error": str(e)}

    return result
