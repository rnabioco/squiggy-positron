# Squiggy 0.1.0 (2025-10-28)

Initial release of Squiggy - a desktop GUI application for visualizing Oxford Nanopore sequencing data from POD5 files with optional base annotations from BAM files.

## Features
- Interactive squiggle plot visualization with four plot modes: SINGLE, OVERLAY, STACKED, and EVENTALIGN
- Support for POD5 file format with VBZ compression
- Base annotation overlay from BAM files with event-aligned data
- Multiple signal normalization methods: Z-score, median-centered, and median absolute deviation (MAD)
- Export plots to HTML, PNG, and SVG formats with zoom-level preservation
- Three search modes: Read ID filtering, reference region queries, and sequence motif search
- Dark mode UI support with qt-material theme
- Sample data bundled with application for quick testing
- Cross-platform support: Windows, macOS, and Linux standalone executables

## Improvements
- Migrated from plotnine to Bokeh for better interactive plotting performance
- Async I/O operations with qasync for non-blocking UI
- Colorblind-friendly Okabe-Ito palette for base coloring
- Signal downsampling using LTTB algorithm for large datasets
- Collapsible UI sections for better space management
- Reference browser dialog for exploring BAM reference sequences

## Internal
- Automated GitHub Actions workflows for multi-platform builds
- Comprehensive test suite with pytest
- PyInstaller configuration for standalone executable packaging
- MkDocs documentation website with Material theme
- `/release` and `/worktree` slash commands for development workflow
