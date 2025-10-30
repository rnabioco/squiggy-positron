# Task: Add Signal/Sequence Space Toggle for OVERLAY Mode

## Description
Add UI controls to allow users to switch between signal space (sample indices) and sequence space (reference positions) when viewing multiple reads in OVERLAY mode.

## Objective
Enable OVERLAY mode to plot reads aligned to reference coordinates (like AGGREGATE mode) when a BAM file is loaded, while maintaining backward compatibility with signal-space plotting.

## Context (from Issue #6)
Currently OVERLAY mode plots in signal space (X = sample index), while AGGREGATE mode plots in sequence space (X = reference position). Users need the ability to overlay multiple reads in sequence space to compare signal patterns at specific genomic positions.

## Files to Modify
- `src/squiggy/constants.py` - Add CoordinateSpace enum
- `src/squiggy/ui_components.py` - Add radio buttons to AdvancedOptionsPanel
- `src/squiggy/viewer.py` - State management and signal connections
- `src/squiggy/plotting/overlay.py` - Implement sequence-space plotting logic
- `src/squiggy/plotting/base.py` - Extract/share alignment utilities if needed
- `tests/test_plotting.py` - Add tests for sequence-space OVERLAY

## Success Criteria
- [x] CoordinateSpace enum added (SIGNAL, SEQUENCE)
- [x] UI toggle added (radio buttons: "Signal Space" / "Sequence Space")
- [x] Toggle enabled only when BAM file is loaded (in OVERLAY mode)
- [x] OVERLAY mode can plot in signal space (default, current behavior)
- [x] OVERLAY mode can plot in sequence space (aligned to reference)
- [x] Reads without BAM alignment are excluded gracefully (no warning, just skipped)
- [x] All existing tests still pass (377 tests passing)
- [x] New tests added for sequence-space plotting (6 new tests)

## Implementation Notes
- Use move table processing similar to AGGREGATE mode for alignment
- Default to signal space for backward compatibility
- Show warning in status bar for excluded reads (no BAM data)
- X-axis label should update based on coordinate space selection

## Technical Details
- Move table format: `[stride, move_0, move_1, ...]`
- Stride values: 5 for DNA models, 10-12 for RNA
- Reference position increments when move=1
- Signal index increments by stride for each move

## Notes
- Branch: add-coordinate-space-toggle
- Base: main
- Created: 2025-10-29
- Worktree: ../squiggy-add-coordinate-space-toggle
- Related Issue: #6 (comment about plottability in sequence and signal space)
