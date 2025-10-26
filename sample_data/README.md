# Sample Data for Squiggy

This directory contains sample POD5 files for users to practice with Squiggy.

## Files

### sample.pod5
A small POD5 file containing representative Oxford Nanopore sequencing reads.

- **Size**: ~500KB - 1MB
- **Reads**: 5-10 reads
- **Purpose**: Quick demonstration of Squiggy's visualization capabilities

## Usage

### Option 1: Download via CLI (Recommended)

```bash
# Download sample data to current directory
squiggy-download-sample

# Or specify a custom location
squiggy-download-sample --output ~/Downloads/sample.pod5
```

### Option 2: Manual Download

1. Download `sample.pod5` from the GitHub repository
2. Open it in Squiggy:
   ```bash
   squiggy
   ```
   Then use File → Open to select the downloaded file.

### Option 3: Direct Open (if installed)

```bash
squiggy --file sample_data/sample.pod5
```

## Creating Your Own Sample Data

To create a minimal POD5 file from your own data:

```python
import pod5

# Extract first 10 reads from a larger file
with pod5.Reader("your_large_file.pod5") as reader:
    with pod5.Writer("sample.pod5") as writer:
        for i, read in enumerate(reader.reads()):
            if i >= 10:
                break
            writer.add_read(read)
```

## Hosting Sample Data

For distribution, sample data can be hosted:
- In this repository (via Git LFS)
- On GitHub Releases
- On a public file server (S3, Google Drive, etc.)

The download URL is configured in `squiggy/main.py` in the `SAMPLE_DATA_URL` constant.
