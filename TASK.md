# Task: Refactor plotter.py into Package Structure

## Description
Refactor the large `plotter.py` file (1,654 lines) into a well-organized package structure with separate modules for each plot mode. This will improve maintainability and reduce complexity.

## Objective
Split `src/squiggy/plotter.py` into a new `src/squiggy/plotting/` package with the following structure:

```
src/squiggy/plotting/
├── __init__.py           # SquigglePlotter class facade (maintains backward compatibility)
├── base.py               # Shared utilities, figure creation, data processing (~400 lines)
├── single.py             # SINGLE mode plotting (~250 lines)
├── overlay.py            # OVERLAY mode plotting (~60 lines)
├── stacked.py            # STACKED mode plotting (~70 lines)
├── eventalign.py         # EVENTALIGN mode plotting (~280 lines)
└── aggregate.py          # AGGREGATE mode plotting (~300 lines)
```

## Module Responsibilities

**base.py** - Core utilities and figure creation:
- Signal processing: `normalize_signal()`, `process_signal()`
- Figure creation: `create_figure()`, `format_plot_title()`, `format_html_title()`
- Renderers: `add_signal_renderers()`, `add_hover_tool()`, `configure_legend()`
- Data sources: `create_signal_data_source()`
- Theme utilities: `get_base_colors()`, `get_signal_line_color()`
- Constants: `MULTI_READ_COLORS`

**single.py** - Single read visualization:
- `plot_single_read()` - Main entry point
- `add_base_annotations_single_read()` - Single read annotations
- `calculate_base_regions_time_mode()` - Time-based base regions
- `add_dwell_time_patches()`, `add_base_type_patches()`
- `add_base_labels_time_mode()`, `add_simple_labels()`

**overlay.py** - Overlay multiple reads:
- `plot_overlay()` - Overlay mode implementation

**stacked.py** - Stack multiple reads:
- `plot_stacked()` - Stacked mode implementation

**eventalign.py** - Event-aligned visualization:
- `plot_eventalign()` - Main entry point
- `add_base_annotations_eventalign()` - Event-aligned annotations
- `plot_eventalign_signals()` - Signal plotting for event-aligned mode
- `calculate_base_regions_position_mode()` - Position-based base regions
- `add_base_labels_position_mode()` - Position-based labels

**aggregate.py** - Aggregate multi-read view:
- `plot_aggregate()` - Three-track aggregate visualization

**__init__.py** - Backward-compatible facade:
- `SquigglePlotter` class with static methods that delegate to module functions
- Maintains exact same API for existing code
- Re-exports commonly used functions

## Files to Modify
- Create: `src/squiggy/plotting/__init__.py`
- Create: `src/squiggy/plotting/base.py`
- Create: `src/squiggy/plotting/single.py`
- Create: `src/squiggy/plotting/overlay.py`
- Create: `src/squiggy/plotting/stacked.py`
- Create: `src/squiggy/plotting/eventalign.py`
- Create: `src/squiggy/plotting/aggregate.py`
- Update: `src/squiggy/viewer.py` (import change)
- Update: `src/squiggy/cli.py` (import change)
- Update: `tests/test_plotting.py` (no change needed - imports work via __init__.py)
- Update: `tests/test_aggregate_plotting.py` (no change needed)
- Update: `tests/test_signal_accuracy.py` (no change needed)
- Update: `CLAUDE.md` (documentation)
- Delete: `src/squiggy/plotter.py`

## Success Criteria
- [x] Create new plotting package directory
- [ ] Implement base.py with shared utilities
- [ ] Implement single.py for SINGLE plot mode
- [ ] Implement overlay.py for OVERLAY plot mode
- [ ] Implement stacked.py for STACKED plot mode
- [ ] Implement eventalign.py for EVENTALIGN plot mode
- [ ] Implement aggregate.py for AGGREGATE plot mode
- [ ] Implement __init__.py with SquigglePlotter class facade
- [ ] Update imports in viewer.py and cli.py
- [ ] All existing tests pass without modification
- [ ] No functionality changes (pure refactoring)
- [ ] Update CLAUDE.md documentation
- [ ] Delete old plotter.py file

## Backward Compatibility

All existing code continues to work without changes:
```python
from squiggy.plotter import SquigglePlotter  # Still works!
SquigglePlotter.plot_single_read(...)        # Still works!
```

The `SquigglePlotter` class in `__init__.py` will delegate to the module functions while maintaining the exact same API surface.

## Notes
- Branch: refactor-plotter-package
- Base: main
- Created: 2025-10-29
- Worktree: /Users/jayhesselberth/devel/rnabioco/squiggy-refactor-plotter-package
- Original file size: 1,654 lines
- Target: ~7 focused modules (~200-400 lines each)
- Strategy: Convert SquigglePlotter static methods to module-level functions
- Maintain 100% backward compatibility through facade pattern
