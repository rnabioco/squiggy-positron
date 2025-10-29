#!/bin/bash
# Generate screenshots for Squiggy documentation
# Uses the CLI headless export mode to create PNG images

set -e  # Exit on error

# Paths
POD5="tests/data/yeast_trna_reads.pod5"
BAM="tests/data/yeast_trna_mappings.bam"
OUTPUT_DIR="docs/images"

# Sample read IDs from yeast tRNA data
READ1="36d401b5-ba9c-4862-b003-7cd309ce5281"
READ2="407164ac-c89a-4b42-a99e-a808577f3756"
READ3="42361bc3-ef8b-497f-8e02-6e2976699816"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo "Generating documentation screenshots..."
echo "Using sample data: $POD5"
echo "Output directory: $OUTPUT_DIR"
echo ""

# 1. Single read mode (default view with median normalization)
echo "1/6 Generating single read mode screenshot..."
squiggy -p "$POD5" \
  --read-id "$READ1" \
  --mode single \
  --normalization median \
  --export "$OUTPUT_DIR/single_read.png" \
  --export-width 1400 \
  --export-height 900

# 2. Event-aligned mode with base annotations
echo "2/6 Generating event-aligned mode screenshot..."
squiggy -p "$POD5" -b "$BAM" \
  --read-id "$READ1" \
  --mode eventalign \
  --normalization median \
  --show-bases \
  --export "$OUTPUT_DIR/eventalign_mode.png" \
  --export-width 1400 \
  --export-height 900

# 3. Event-aligned mode with dwell time scaling
echo "3/6 Generating event-aligned with dwell time screenshot..."
squiggy -p "$POD5" -b "$BAM" \
  --read-id "$READ1" \
  --mode eventalign \
  --normalization median \
  --show-bases \
  --dwell-time \
  --export "$OUTPUT_DIR/eventalign_dwell.png" \
  --export-width 1400 \
  --export-height 900

# 4. Overlay mode with multiple reads (z-score normalization)
echo "4/6 Generating overlay mode screenshot..."
squiggy -p "$POD5" \
  --reads "$READ1" "$READ2" "$READ3" \
  --mode overlay \
  --normalization znorm \
  --export "$OUTPUT_DIR/overlay_mode.png" \
  --export-width 1400 \
  --export-height 900

# 5. Stacked mode with multiple reads
echo "5/6 Generating stacked mode screenshot..."
squiggy -p "$POD5" \
  --reads "$READ1" "$READ2" "$READ3" \
  --mode stacked \
  --normalization median \
  --export "$OUTPUT_DIR/stacked_mode.png" \
  --export-width 1400 \
  --export-height 900

# 6. Single read with signal points shown
echo "6/6 Generating signal points screenshot..."
squiggy -p "$POD5" \
  --read-id "$READ1" \
  --mode single \
  --normalization median \
  --show-points \
  --downsample 5 \
  --export "$OUTPUT_DIR/signal_points.png" \
  --export-width 1400 \
  --export-height 900

echo ""
echo "✓ All screenshots generated successfully!"
echo "Output location: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/*.png
