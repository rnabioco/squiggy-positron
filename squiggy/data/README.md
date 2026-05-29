# Squiggy Demo Data

This directory contains demo data files bundled with the squiggy Python package.

## Biological Context

E. coli tRNA data comparing wildtype (WT) vs TruB mutant at 15 minutes (control condition). TruB catalyzes pseudouridine (Ψ) at position 55 in the T-loop of tRNAs. The TruB mutant lacks this modification, providing a clear biological comparison for demonstrating modification visualization.

**Source**: 2026 phage infection experiment, charged-only reads from `wt-15-ctl-01` and `tb-15-ctl-01`.

## Files

### WT Sample (wildtype, has Ψ55)
- **ecoli_trna_wt_reads.pod5** (1.6 MB) - 180 reads (60 per reference)
- **ecoli_trna_wt_mappings.bam** (113 KB) - Alignments with modifications
- **ecoli_trna_wt_mappings.bam.bai** - BAM index

### TruB Sample (TruB mutant, lacks Ψ55)
- **ecoli_trna_tb_reads.pod5** (1.5 MB) - 180 reads (60 per reference)
- **ecoli_trna_tb_mappings.bam** (98 KB) - Alignments with modifications
- **ecoli_trna_tb_mappings.bam.bai** - BAM index

### Shared Reference
- **ecoli_trna.fa** - Reference FASTA (3 tRNA sequences with adapters)
- **ecoli_trna.fa.fai** - FASTA index

## References Included

| Reference | Length | Description |
|-----------|--------|-------------|
| host-tRNA-Arg-ACG-1-1 | 141 bp | Arginine tRNA |
| host-tRNA-Gly-GCC-1-1 | 140 bp | Glycine tRNA |
| host-tRNA-Ala-TGC-1-1 | 140 bp | Alanine tRNA |

## Read Selection Criteria

All reads were selected to meet these quality requirements:
- **MAPQ >= 20** - High mapping quality
- **mv tag** - Move table present (signal-to-base alignment)
- **Both 5' and 3' PT tags** - Full adapter coverage (5p_adapter + 3p_adapter_charged)

## BAM Tags

All reads include:
- `mv` - Move table (signal-to-base alignment)
- `ts` - Trim start offset
- `MM`/`ML` - Base modification calls (8 types: A+17596, A+69426, A+a, C+19228, C+m, G+19229, T+17802, T+19227)
- `PT` - Adapter boundary positions (both 5' and 3' for every read)

## Usage

### From Positron Extension

The Squiggy extension provides a **"Load Demo Session"** button that loads both WT and TruB samples for side-by-side comparison.

### From Python

```python
import squiggy

data_dir = squiggy.get_test_data_path()

# Load WT sample
squiggy.load_pod5(f'{data_dir}/ecoli_trna_wt_reads.pod5')
squiggy.load_bam(f'{data_dir}/ecoli_trna_wt_mappings.bam')
```

## Size

Total: ~3.4 MB (2 POD5 + 2 BAM + FASTA files)
