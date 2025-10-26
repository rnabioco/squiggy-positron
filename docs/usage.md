# Usage

## Opening POD5 Files

1. Launch Squiggy
2. Click the **"Select POD5 File"** button
3. Navigate to and select your POD5 file
4. The application will load all available reads

## Viewing Reads

Once a POD5 file is loaded:

- **Read List**: All read IDs from the file appear in the left panel
- **Search**: Use the search box to filter reads by ID
- **Select**: Click any read ID to visualize its signal data

## Understanding Squiggle Plots

The squiggle plot shows:

- **X-axis**: Time (seconds) - calculated from sample rate
- **Y-axis**: Signal (pA) - raw current measurements from the nanopore
- **Blue line**: The time-series signal trace

Each peak and valley represents ionic current changes as DNA/RNA molecules pass through the nanopore.

## Tips

- **Large files**: POD5 files can contain thousands of reads. Use the search box to quickly find specific reads
- **Plot quality**: Plots are generated at high resolution suitable for screenshots and presentations
- **Performance**: The first read visualization may take a moment as libraries initialize; subsequent plots render quickly

## Keyboard Shortcuts

- **Ctrl/Cmd + O**: Open file dialog
- **Ctrl/Cmd + Q**: Quit application (standard Qt shortcuts)

## File Format Support

Squiggy supports POD5 files, which are the standard format for Oxford Nanopore raw signal data:

- POD5 is an HDF5-based format with VBZ compression
- Files typically have `.pod5` extension
- Contains raw signal data plus metadata (sample rate, read IDs, etc.)

## Troubleshooting

**Application won't start**
- On macOS: Check System Preferences → Security to allow the app
- On Linux: Ensure the executable has run permissions (`chmod +x Squiggy`)

**Can't open POD5 file**
- Verify the file is a valid POD5 format
- Check that the file isn't corrupted or incomplete

**Plots not displaying**
- Ensure the read has valid signal data
- Try selecting a different read to verify functionality
