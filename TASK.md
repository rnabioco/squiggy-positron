# Refactor Plotting Architecture with Strategy Pattern

**Branch**: `refactor-plotting-architecture`
**Base**: `main`
**Created**: 2025-11-02
**Timeline**: 5-6 weeks
**Related Issue**: #61 (Multi-File A/B Comparison)

---

## Objective

Refactor squiggy's plotting module from a monolithic 2,000+ line class into a modular, extensible architecture using the Strategy Pattern. This enables easy addition of new plot types (especially the A/B comparison plot from #61) while eliminating code duplication and improving testability.

---

## Problem Statement

### Current Architecture Issues

1. **Monolithic `SquigglePlotter` class** (plotter.py:66-2033)
   - All 5 plot types in one 2,000-line class
   - Violates Single Responsibility Principle
   - Difficult to add new plot types without breaking existing ones

2. **Code Duplication in Base Annotation Logic**
   - Time-based calculation (lines 351-472)
   - Position-based calculation (lines 475-542)
   - Nearly identical logic but split into separate functions

3. **Mode-Specific Helper Functions**
   - Functions appear "generic" but are tightly coupled to specific plot types
   - Cannot reuse base annotation logic across plot types
   - Hidden dependencies make testing difficult

4. **Excessive `global` Keyword Usage**
   - `global _squiggy_session` declared even when just mutating attributes
   - Legacy globals (`_current_pod5_reader`, etc.) maintained for backward compatibility
   - Code smell that can be eliminated

5. **Limited Modification Support**
   - Modification tracks only work with SINGLE mode (lines 984-1026)
   - Cannot add modifications to EVENTALIGN or AGGREGATE modes
   - Logic hardcoded into single plot type

6. **Scattered Theme Logic**
   - Theme colors applied ad-hoc throughout plotter.py
   - 45 lines of theme application in `_create_figure()` alone
   - No centralized theme management

---

## Solution Overview

### Strategy Pattern Architecture

```
PlotFactory
    ↓ creates
PlotStrategy (ABC)
    ↓ implements
┌─────────────────────────────────────────────┐
│  SingleReadPlotStrategy                     │
│  EventAlignPlotStrategy                     │
│  OverlayPlotStrategy                        │
│  StackedPlotStrategy                        │
│  AggregatePlotStrategy                      │
│  ComparisonPlotStrategy (NEW for #61)       │
└─────────────────────────────────────────────┘
    ↓ uses
┌─────────────────────────────────────────────┐
│  Shared Components:                         │
│  - ThemeManager                             │
│  - BaseAnnotationRenderer                   │
│  - ModificationTrackBuilder                 │
└─────────────────────────────────────────────┘
```

### Key Benefits

✅ **Easy extensibility** - Add new plot type = create new strategy class
✅ **Eliminate duplication** - Shared components used across all plot types
✅ **Better testability** - Each component tested independently
✅ **Cleaner state management** - Remove `global` keyword entirely
✅ **Reusable modifications** - Mod tracks work with all plot types
✅ **Separation of concerns** - Each strategy has single responsibility
✅ **Enables #61** - Comparison plot fits naturally into architecture

---

## Global State Management Refactoring

### Current Pattern (io.py)

```python
# Module-level state
_squiggy_session = SquiggySession()  # NEW from PR #59
_current_pod5_reader = None          # LEGACY
_current_read_ids = []               # LEGACY
_current_bam_path = None             # LEGACY

def load_pod5(file_path: str):
    global _squiggy_session, _current_pod5_reader, _current_read_ids  # ❌
    # ... mutate session attributes
```

### Refactored Pattern

```python
# Module-level state (single source of truth)
_squiggy_session = SquiggySession()

def load_pod5(file_path: str) -> SquiggySession:
    """Load POD5 file into kernel session"""
    # No global keyword needed! Just mutate the object
    _squiggy_session.close_pod5()
    _squiggy_session.reader = reader
    _squiggy_session.pod5_path = abs_path
    _squiggy_session.read_ids = read_ids
    return _squiggy_session  # Return for user visibility
```

### Why This Works

- **Module-level state is necessary** for kernel/REPL usage
- **`global` keyword only needed when REASSIGNING** the variable itself
- **Mutating object attributes doesn't require `global`**
- **Legacy globals removed** (breaking change, but acceptable per user)
- **`_squiggy_session` becomes sole state container**

---

## Implementation Plan

### Phase 1: Foundation & Cleanup (Week 1, Days 1-2)

#### 1.1 Clean Up Global State Pattern

**Files**: `squiggy/io.py`, `squiggy/__init__.py`

**Tasks**:
- [ ] Remove all `global` keyword declarations (lines 123, 303, 357, 428, 454)
- [ ] Remove legacy global variables (`_current_pod5_reader`, `_current_read_ids`, `_current_bam_path`)
- [ ] Update `load_pod5()` to return `_squiggy_session`
- [ ] Update `load_bam()` to return `_squiggy_session`
- [ ] Update `__init__.py` plot functions to access `_squiggy_session` directly (no imports)
- [ ] Update docstrings to reflect new pattern

**Testing**:
- [ ] All existing tests pass
- [ ] Session visible in Variables pane
- [ ] Kernel state persists across function calls

**Example**:
```python
# io.py - Before
def load_pod5(file_path: str) -> tuple[pod5.Reader, list[str]]:
    global _current_pod5_reader, _squiggy_session
    # ...
    return reader, read_ids

# io.py - After
def load_pod5(file_path: str) -> SquiggySession:
    # No global keyword!
    _squiggy_session.close_pod5()
    # ... setup reader
    _squiggy_session.reader = reader
    _squiggy_session.pod5_path = abs_path
    _squiggy_session.read_ids = read_ids
    return _squiggy_session
```

#### 1.2 Create Base Strategy Interface

**New file**: `squiggy/plot_strategies/base.py`

**Tasks**:
- [ ] Create `PlotStrategy` abstract base class
- [ ] Define `create_plot(data, options)` abstract method
- [ ] Define `validate_data(data)` abstract method
- [ ] Add comprehensive docstrings
- [ ] Create unit tests for interface contract

**Code**:
```python
from abc import ABC, abstractmethod
from bokeh.models import Figure
from ..constants import Theme

class PlotStrategy(ABC):
    """Base class for all plot type implementations"""

    def __init__(self, theme: Theme):
        self.theme = theme

    @abstractmethod
    def create_plot(self, data: dict, options: dict) -> tuple[str, Figure]:
        """
        Generate Bokeh plot HTML and figure

        Args:
            data: Plot data (signal, annotations, modifications, etc.)
            options: Plot-specific options (normalization, downsample, etc.)

        Returns:
            Tuple of (html_string, bokeh_figure)
        """
        pass

    @abstractmethod
    def validate_data(self, data: dict) -> None:
        """
        Validate that required data is present for this plot type

        Raises:
            ValueError: If required data is missing
        """
        pass
```

---

### Phase 2: Extract Reusable Components (Week 1, Days 3-5)

#### 2.1 Theme Manager

**New file**: `squiggy/theme_manager.py`

**Tasks**:
- [ ] Create `ThemeManager` class
- [ ] Extract theme logic from `plotter.py:265-321`
- [ ] Add `apply_to_figure()` method
- [ ] Add color getter methods
- [ ] Write unit tests for theme application
- [ ] Verify visual output unchanged

**Code**:
```python
from bokeh.models import Figure
from .constants import Theme, LIGHT_THEME, DARK_THEME, BASE_COLORS, BASE_COLORS_DARK

class ThemeManager:
    """Centralized theme management for all plot types"""

    def __init__(self, theme: Theme):
        self.theme = theme
        self.colors = DARK_THEME if theme == Theme.DARK else LIGHT_THEME
        self.base_colors = BASE_COLORS_DARK if theme == Theme.DARK else BASE_COLORS

    def apply_to_figure(self, figure: Figure) -> None:
        """Apply theme colors to Bokeh figure"""
        figure.background_fill_color = self.colors["background"]
        figure.border_fill_color = self.colors["background"]
        figure.outline_line_color = self.colors["grid"]
        # ... (extract from current _create_figure())

    def get_signal_color(self) -> str:
        return self.colors["signal_line"]

    def create_figure(self, width: int, height: int, **kwargs) -> Figure:
        """Create themed Bokeh figure"""
        fig = figure(width=width, height=height, **kwargs)
        self.apply_to_figure(fig)
        return fig
```

**Refactor locations**:
- `plotter.py:265-273` - Color getters
- `plotter.py:276-321` - `_create_figure()` theme logic

#### 2.2 Base Annotation Renderer

**New file**: `squiggy/annotation_renderer.py`

**Tasks**:
- [ ] Create `AnnotationMode` enum (TIME_BASED, POSITION_BASED)
- [ ] Create `BaseRegionData` dataclass (unified return type)
- [ ] Create `BaseAnnotationRenderer` class
- [ ] Extract and unify time-based logic (`plotter.py:351-472`)
- [ ] Extract and unify position-based logic (`plotter.py:475-542`)
- [ ] Implement `calculate_regions()` method (dispatches to mode)
- [ ] Implement `render_patches()` method
- [ ] Implement `render_labels()` method
- [ ] Write comprehensive unit tests (both modes)
- [ ] Test with sample data from `tests/data/`

**Code**:
```python
from enum import Enum
from dataclasses import dataclass
from bokeh.models import Figure, LinearColorMapper
from .constants import Theme, BASE_COLORS, BASE_COLORS_DARK

class AnnotationMode(Enum):
    TIME_BASED = "time"
    POSITION_BASED = "position"

@dataclass
class BaseRegionData:
    """Unified return type for base annotation calculations"""
    regions: dict[str, list]  # Base type -> list of regions
    labels: dict[str, list]   # Base type -> label data
    dwell_times: list[float] | None = None
    color_mapper: LinearColorMapper | None = None

class BaseAnnotationRenderer:
    """Handles all base annotation rendering across plot types"""

    def __init__(self, theme: Theme):
        self.theme = theme
        self.base_colors = BASE_COLORS_DARK if theme == Theme.DARK else BASE_COLORS

    def calculate_regions(
        self,
        mode: AnnotationMode,
        sequence: str,
        seq_to_sig_map: list[int],
        signal_range: tuple[float, float],
        sample_rate: int,
        show_dwell_time: bool = False,
        scale_dwell_time: bool = False,
        **kwargs
    ) -> BaseRegionData:
        """
        Unified region calculation for time or position modes

        Consolidates logic from:
        - _calculate_base_regions_time_mode() (plotter.py:351-472)
        - _calculate_base_regions_position_mode() (plotter.py:475-542)
        """
        if mode == AnnotationMode.TIME_BASED:
            return self._calculate_time_based(
                sequence, seq_to_sig_map, signal_range, sample_rate,
                show_dwell_time, scale_dwell_time
            )
        else:
            return self._calculate_position_based(
                sequence, seq_to_sig_map, signal_range
            )

    def render_patches(self, figure: Figure, regions: BaseRegionData) -> None:
        """Add background base patches to figure"""
        # Extract from _add_base_annotations_single_read() and _add_base_annotations_eventalign()
        pass

    def render_labels(
        self,
        figure: Figure,
        regions: BaseRegionData,
        interval: int = 10,
        mode: AnnotationMode = AnnotationMode.TIME_BASED
    ) -> None:
        """Add base labels to figure"""
        # Extract from _add_base_labels_time_mode() and _add_base_labels_position_mode()
        pass
```

**Refactor locations**:
- `plotter.py:351-472` - Time-based calculation
- `plotter.py:475-542` - Position-based calculation
- `plotter.py:1029-1106` - Single read annotation rendering
- `plotter.py:1539-1600` - EventAlign annotation rendering
- `plotter.py:607-637` - Time-based labels
- `plotter.py:640-777` - Position-based labels

#### 2.3 Modification Track Builder

**New file**: `squiggy/modification_track.py`

**Tasks**:
- [ ] Create `ModificationTrackBuilder` class
- [ ] Extract modification track logic (`plotter.py:1108-1287`)
- [ ] Implement `create_track()` method
- [ ] Implement `create_column_layout()` method for linking axes
- [ ] Write unit tests with sample modification data
- [ ] Verify track appearance unchanged

**Code**:
```python
from bokeh.models import Figure, Column
from bokeh.layouts import column
from .constants import Theme
from .modifications import ModificationAnnotation

class ModificationTrackBuilder:
    """Separate builder for modification tracks (reusable across plot types)"""

    def __init__(self, theme: Theme):
        self.theme = theme

    def create_track(
        self,
        sequence: str,
        seq_to_sig_map: list[int],
        time_ms: np.ndarray,
        modifications: list[ModificationAnnotation],
        min_probability: float = 0.5,
        enabled_types: list[str] | None = None,
        height: int = 150,
        **options
    ) -> Figure | None:
        """
        Create modification track figure

        Extracted from plotter.py:1108-1287
        """
        # ... implementation
        pass

    def create_column_layout(
        self,
        main_figure: Figure,
        mod_track: Figure
    ) -> Column:
        """
        Link axes and create column layout

        Synchronizes x-axis zoom/pan between main plot and mod track
        """
        # Link x-ranges
        mod_track.x_range = main_figure.x_range

        # Create column layout
        return column(main_figure, mod_track)
```

**Refactor location**:
- `plotter.py:1108-1287` - `_create_modification_track()`
- `plotter.py:999-1026` - Column layout creation in `plot_single_read()`

**Benefit**: Enables modification tracks for EVENTALIGN and AGGREGATE modes!

---

### Phase 3: Implement Strategy Pattern (Week 2-3)

#### 3.1 SingleReadPlotStrategy

**New file**: `squiggy/plot_strategies/single_read.py`

**Tasks**:
- [ ] Create `SingleReadPlotStrategy` class extending `PlotStrategy`
- [ ] Extract logic from `plotter.py:808-1027`
- [ ] Use `ThemeManager` for figure creation
- [ ] Use `BaseAnnotationRenderer` for annotations
- [ ] Use `ModificationTrackBuilder` for mod tracks
- [ ] Write characterization tests (compare old vs new HTML output)
- [ ] Write unit tests for strategy

**Code**:
```python
from bokeh.models import Figure
from .base import PlotStrategy
from ..theme_manager import ThemeManager
from ..annotation_renderer import BaseAnnotationRenderer, AnnotationMode
from ..modification_track import ModificationTrackBuilder
from ..constants import Theme, NormalizationMethod

class SingleReadPlotStrategy(PlotStrategy):
    """Strategy for SINGLE plot mode - single read visualization"""

    def __init__(self, theme: Theme):
        super().__init__(theme)
        self.theme_manager = ThemeManager(theme)
        self.annotation_renderer = BaseAnnotationRenderer(theme)
        self.mod_builder = ModificationTrackBuilder(theme)

    def validate_data(self, data: dict) -> None:
        required = ['signal', 'read_id', 'sample_rate']
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing required data: {missing}")

    def create_plot(self, data: dict, options: dict) -> tuple[str, Figure]:
        self.validate_data(data)

        # Extract data
        signal = data['signal']
        read_id = data['read_id']
        sample_rate = data['sample_rate']
        sequence = data.get('sequence')
        seq_to_sig_map = data.get('seq_to_sig_map')
        modifications = data.get('modifications')

        # Extract options
        normalization = options.get('normalization', NormalizationMethod.ZNORM)
        show_dwell_time = options.get('show_dwell_time', False)
        scale_dwell_time = options.get('scale_dwell_time', False)
        # ... other options

        # Create figure using theme manager
        fig = self.theme_manager.create_figure(
            width=1200, height=400,
            title=f"Read: {read_id}",
            # ... toolbar, etc.
        )

        # Calculate x-axis (time or dwell-scaled)
        x_values = self._calculate_x_axis(signal, sample_rate, ...)

        # Add signal line
        fig.line(x_values, signal, color=self.theme_manager.get_signal_color(), ...)

        # Add base annotations if available
        if sequence and seq_to_sig_map:
            regions = self.annotation_renderer.calculate_regions(
                mode=AnnotationMode.TIME_BASED,
                sequence=sequence,
                seq_to_sig_map=seq_to_sig_map,
                signal_range=(signal.min(), signal.max()),
                sample_rate=sample_rate,
                show_dwell_time=show_dwell_time,
                scale_dwell_time=scale_dwell_time
            )
            self.annotation_renderer.render_patches(fig, regions)
            if options.get('show_labels', True):
                self.annotation_renderer.render_labels(fig, regions)

        # Add modification track if present
        if modifications:
            mod_track = self.mod_builder.create_track(
                sequence=sequence,
                seq_to_sig_map=seq_to_sig_map,
                time_ms=x_values,
                modifications=modifications,
                min_probability=options.get('min_mod_probability', 0.5),
                enabled_types=options.get('enabled_mod_types')
            )
            if mod_track:
                layout = self.mod_builder.create_column_layout(fig, mod_track)
                return self._figure_to_html(layout), layout

        return self._figure_to_html(fig), fig
```

#### 3.2 EventAlignPlotStrategy

**New file**: `squiggy/plot_strategies/eventalign.py`

**Tasks**:
- [ ] Create `EventAlignPlotStrategy` class
- [ ] Extract logic from `plotter.py:1462-1720`
- [ ] Reuse `BaseAnnotationRenderer` with `AnnotationMode.POSITION_BASED`
- [ ] Handle multiple reads with shared annotation background
- [ ] Write characterization tests
- [ ] Write unit tests

**Key difference from SingleRead**: Uses position-based x-axis and annotation mode.

#### 3.3 OverlayPlotStrategy & StackedPlotStrategy

**New files**:
- `squiggy/plot_strategies/overlay.py`
- `squiggy/plot_strategies/stacked.py`

**Tasks**:
- [ ] Create `OverlayPlotStrategy` (extract from `plotter.py:1346-1398`)
- [ ] Create `StackedPlotStrategy` (extract from `plotter.py:1400-1460`)
- [ ] Implement color cycling for multiple reads
- [ ] Write tests for both strategies

#### 3.4 AggregatePlotStrategy

**New file**: `squiggy/plot_strategies/aggregate.py`

**Tasks**:
- [ ] Create `AggregatePlotStrategy` class
- [ ] Extract logic from `plotter.py:1723-2032`
- [ ] Create three synchronized tracks (signal, pileup, quality)
- [ ] Use `gridplot` for layout
- [ ] Link x-axes for synchronized zoom/pan
- [ ] Write characterization tests
- [ ] Write unit tests

**Code skeleton**:
```python
from bokeh.layouts import gridplot
from .base import PlotStrategy

class AggregatePlotStrategy(PlotStrategy):
    """Strategy for AGGREGATE plot mode - multi-read aggregate visualization"""

    def create_plot(self, data: dict, options: dict) -> tuple[str, gridplot]:
        # Create 3 tracks
        signal_track = self._create_signal_track(data['aggregate_stats'])
        pileup_track = self._create_pileup_track(data['pileup_stats'])
        quality_track = self._create_quality_track(data['quality_stats'])

        # Link x-axes
        pileup_track.x_range = signal_track.x_range
        quality_track.x_range = signal_track.x_range

        # Create grid layout
        grid = gridplot([[signal_track], [pileup_track], [quality_track]])
        return self._figure_to_html(grid), grid
```

---

### Phase 4: Plot Factory & Public API (Week 3)

#### 4.1 Plot Factory

**New file**: `squiggy/plot_factory.py`

**Tasks**:
- [ ] Create `PlotFactory` class
- [ ] Register all strategy classes in `_strategies` dict
- [ ] Implement `create_plot()` factory method
- [ ] Add error handling for unknown plot modes
- [ ] Write unit tests for factory

**Code**:
```python
from .constants import PlotMode, Theme
from .plot_strategies.base import PlotStrategy
from .plot_strategies.single_read import SingleReadPlotStrategy
from .plot_strategies.eventalign import EventAlignPlotStrategy
from .plot_strategies.overlay import OverlayPlotStrategy
from .plot_strategies.stacked import StackedPlotStrategy
from .plot_strategies.aggregate import AggregatePlotStrategy

class PlotFactory:
    """Factory for creating plot strategies"""

    _strategies = {
        PlotMode.SINGLE: SingleReadPlotStrategy,
        PlotMode.EVENTALIGN: EventAlignPlotStrategy,
        PlotMode.OVERLAY: OverlayPlotStrategy,
        PlotMode.STACKED: StackedPlotStrategy,
        PlotMode.AGGREGATE: AggregatePlotStrategy,
        # PlotMode.COMPARISON: ComparisonPlotStrategy,  # Phase 5
    }

    @classmethod
    def create_plot(cls, mode: PlotMode, theme: Theme) -> PlotStrategy:
        """
        Create plot strategy for given mode

        Args:
            mode: Plot mode enum
            theme: Theme enum

        Returns:
            PlotStrategy instance

        Raises:
            ValueError: If mode is not supported
        """
        strategy_class = cls._strategies.get(mode)
        if not strategy_class:
            raise ValueError(f"Unknown plot mode: {mode}")
        return strategy_class(theme)

    @classmethod
    def register_strategy(cls, mode: PlotMode, strategy_class: type[PlotStrategy]) -> None:
        """Register a new plot strategy (for plugins/extensions)"""
        cls._strategies[mode] = strategy_class
```

#### 4.2 Update Public API Functions

**File**: `squiggy/__init__.py`

**Tasks**:
- [ ] Refactor `plot_read()` to use `PlotFactory`
- [ ] Refactor `plot_reads()` to use `PlotFactory`
- [ ] Refactor `plot_aggregate()` to use `PlotFactory`
- [ ] Remove direct imports of `_current_*` globals
- [ ] Access `_squiggy_session` directly (already in module scope)
- [ ] Update docstrings
- [ ] Verify backward compatibility

**Example refactoring**:
```python
# __init__.py - Before
def plot_read(read_id: str, mode: str = "SINGLE", **options) -> str:
    from .io import _current_bam_path, _current_pod5_reader  # ❌

    if _current_pod5_reader is None:
        raise ValueError("No POD5 file loaded. Call load_pod5() first.")

    # ... extract data manually
    html, figure = SquigglePlotter.plot_single_read(...)
    return html

# __init__.py - After
def plot_read(read_id: str, mode: str = "SINGLE", **options) -> str:
    from .io import _squiggy_session  # ✅ Direct access (already in scope)
    from .plot_factory import PlotFactory

    if _squiggy_session.reader is None:
        raise ValueError("No POD5 file loaded. Call load_pod5() first.")

    # Get read data from session
    read_obj = None
    for read in _squiggy_session.reader.reads():
        if str(read.read_id) == read_id:
            read_obj = read
            break

    if read_obj is None:
        raise ValueError(f"Read not found: {read_id}")

    # Prepare data dict
    data = {
        'signal': read_obj.signal,
        'read_id': read_id,
        'sample_rate': read_obj.run_info.sample_rate,
    }

    # Add sequence/annotations if BAM loaded
    if _squiggy_session.bam_path:
        aligned_read = extract_alignment_from_bam(_squiggy_session.bam_path, read_id)
        if aligned_read:
            data['sequence'] = aligned_read.sequence
            data['seq_to_sig_map'] = [ann.signal_start for ann in aligned_read.bases]
            data['modifications'] = getattr(aligned_read, 'modifications', None)

    # Get strategy from factory
    plot_mode = PlotMode[mode.upper()]
    theme = Theme[options.pop('theme', 'LIGHT').upper()]
    strategy = PlotFactory.create_plot(plot_mode, theme)

    # Generate plot
    html, figure = strategy.create_plot(data, options)
    return html
```

#### 4.3 Deprecate Old Plotter Class

**File**: `squiggy/plotter.py`

**Tasks**:
- [ ] Add deprecation warnings to `SquigglePlotter` class
- [ ] Update docstring to point to new architecture
- [ ] Keep class for backward compatibility (temporary)
- [ ] Plan removal in future version

**Code**:
```python
import warnings

class SquigglePlotter:
    """
    DEPRECATED: This class will be removed in version 0.3.0

    Use the plot_factory and strategy pattern instead:

        from squiggy.plot_factory import PlotFactory
        from squiggy.constants import PlotMode, Theme

        strategy = PlotFactory.create_plot(PlotMode.SINGLE, Theme.LIGHT)
        html, fig = strategy.create_plot(data, options)

    See squiggy/plot_strategies/ for individual strategy implementations.
    """

    def __init__(self):
        warnings.warn(
            "SquigglePlotter is deprecated and will be removed in v0.3.0. "
            "Use PlotFactory and strategy pattern instead.",
            DeprecationWarning,
            stacklevel=2
        )
```

---

### Phase 5: Comparison Implementation (Week 4)

**Goal**: Implement A/B comparison plot from issue #61

#### 5.1 Add COMPARISON Plot Mode

**File**: `squiggy/constants.py`

**Tasks**:
- [ ] Add `COMPARISON = "COMPARISON"` to `PlotMode` enum
- [ ] Document comparison mode in docstring

#### 5.2 Create ComparisonSession Class

**File**: `squiggy/io.py`

**Tasks**:
- [ ] Create `ComparisonSession` class (similar to `SquiggySession`)
- [ ] Add `session_a` and `session_b` attributes (both `SquiggySession` instances)
- [ ] Add model provenance tracking
- [ ] Implement `__repr__()` for Variables pane display
- [ ] Create module-level `_squiggy_comparison` instance

**Code**:
```python
class ComparisonSession:
    """Manages A/B dataset comparison"""

    def __init__(self):
        self.session_a: SquiggySession = SquiggySession()
        self.session_b: SquiggySession = SquiggySession()
        self.model_provenance_a: dict | None = None
        self.model_provenance_b: dict | None = None

    def __repr__(self) -> str:
        """Compact A vs B summary for Variables pane"""
        return f"<ComparisonSession: A: {self.session_a} | B: {self.session_b}>"

    def validate_references(self) -> bool:
        """Check that A and B have matching SQ headers"""
        pass

    def close_all(self):
        """Close both sessions"""
        self.session_a.close_all()
        self.session_b.close_all()

# Global instance
_squiggy_comparison = ComparisonSession()
```

#### 5.3 Add Comparison Load Functions

**File**: `squiggy/io.py`

**Tasks**:
- [ ] Implement `load_dataset_a(pod5_path, bam_path)`
- [ ] Implement `load_dataset_b(pod5_path, bam_path)`
- [ ] Extract model provenance from BAM headers
- [ ] Validate reference compatibility

**Code**:
```python
def load_dataset_a(pod5_path: str, bam_path: str | None = None) -> ComparisonSession:
    """Load dataset A into comparison session"""
    # Load into session_a
    # ... implementation
    return _squiggy_comparison

def load_dataset_b(pod5_path: str, bam_path: str | None = None) -> ComparisonSession:
    """Load dataset B into comparison session"""
    # Load into session_b
    # Validate reference compatibility
    # ... implementation
    return _squiggy_comparison
```

#### 5.4 Create Comparison Utilities

**New file**: `squiggy/comparison.py`

**Tasks**:
- [ ] Implement `calculate_signal_delta(stats_a, stats_b, mode='absolute')`
- [ ] Implement `calculate_pileup_delta(pileup_a, pileup_b)`
- [ ] Implement `calculate_quality_delta(qual_a, qual_b)`
- [ ] Implement `calculate_mod_delta(mod_a, mod_b)` (if applicable)
- [ ] Write unit tests for delta calculations

**Code**:
```python
def calculate_signal_delta(
    stats_a: dict,
    stats_b: dict,
    mode: str = 'absolute'
) -> dict:
    """
    Compute Δ signal track (A - B or (A-B)/B × 100)

    Args:
        stats_a: Aggregate stats for dataset A
        stats_b: Aggregate stats for dataset B
        mode: 'absolute' or 'percentage'

    Returns:
        Delta statistics dict
    """
    if mode == 'absolute':
        delta_mean = stats_a['mean'] - stats_b['mean']
    elif mode == 'percentage':
        delta_mean = (stats_a['mean'] - stats_b['mean']) / stats_b['mean'] * 100
    # ... etc
    return {'delta_mean': delta_mean, ...}
```

#### 5.5 Create ComparisonPlotStrategy

**New file**: `squiggy/plot_strategies/comparison.py`

**Tasks**:
- [ ] Create `ComparisonPlotStrategy` class
- [ ] Implement side-by-side layout (A | B | Δ)
- [ ] Create synchronized tracks for all plot types (signal, pileup, quality, mods)
- [ ] Use Okabe-Ito palette (A=blue, B=orange, Δ=gray)
- [ ] Link x-axes for synchronized zoom/pan
- [ ] Add model mismatch indicator if needed
- [ ] Write tests

**Layout**:
```
┌─────────────┬─────────────┐
│  Dataset A  │  Dataset B  │  ← Signal tracks
├─────────────┼─────────────┤
│  Pileup A   │  Pileup B   │
├─────────────┼─────────────┤
│  Quality A  │  Quality B  │
├─────────────┴─────────────┤
│       Δ Track (A - B)      │  ← Full width
└────────────────────────────┘
```

#### 5.6 Add Public API Function

**File**: `squiggy/__init__.py`

**Tasks**:
- [ ] Implement `plot_comparison(reference_name, **options)`
- [ ] Register `ComparisonPlotStrategy` in `PlotFactory`
- [ ] Update `__all__` exports
- [ ] Write integration tests

**Code**:
```python
def plot_comparison(
    reference_name: str,
    max_reads: int = 100,
    normalization: str = 'ZNORM',
    theme: str = 'LIGHT',
    delta_mode: str = 'absolute',
    tracks: list[str] = ['signal', 'pileup', 'quality']
) -> str:
    """
    Generate A/B comparison plot for a reference sequence

    Args:
        reference_name: Reference name present in both BAM files
        max_reads: Max reads to sample per dataset
        normalization: Normalization method
        theme: Color theme
        delta_mode: 'absolute' or 'percentage' for Δ calculation
        tracks: List of tracks to include

    Returns:
        Bokeh HTML string with comparison plot

    Raises:
        ValueError: If comparison session not loaded
    """
    from .io import _squiggy_comparison
    from .plot_factory import PlotFactory
    from .comparison import calculate_signal_delta, ...

    # Validate both datasets loaded
    if _squiggy_comparison.session_a.reader is None:
        raise ValueError("Dataset A not loaded. Call load_dataset_a() first.")
    if _squiggy_comparison.session_b.reader is None:
        raise ValueError("Dataset B not loaded. Call load_dataset_b() first.")

    # Compute aggregates for both datasets
    # ... (similar to plot_aggregate())

    # Compute deltas
    delta_stats = calculate_signal_delta(stats_a, stats_b, mode=delta_mode)

    # Prepare data
    data = {
        'aggregate_a': stats_a,
        'aggregate_b': stats_b,
        'delta_stats': delta_stats,
        'reference_name': reference_name,
        # ... other tracks
    }

    # Get strategy
    strategy = PlotFactory.create_plot(PlotMode.COMPARISON, Theme[theme.upper()])
    html, figure = strategy.create_plot(data, options)
    return html
```

---

### Phase 6: Extension Integration & Message Passing Validation (Week 5, Days 1-2)

**Critical**: Verify TypeScript ↔ Python communication still works after refactoring

#### 6.1 Fix TypeScript Legacy Global Reference

**File**: `src/commands/file-commands.ts:403`

**Issue**: References `_current_bam_path` which will be removed

**Fix**:
```typescript
// BEFORE (line 403):
await state.positronClient.executeSilent(`
# Clear BAM file state
_current_bam_path = None
`);

// AFTER:
await state.positronClient.executeSilent(`
# Clear BAM file state
from squiggy.io import _squiggy_session
_squiggy_session.close_bam()
`);
```

**Tasks**:
- [ ] Fix `file-commands.ts:403`
- [ ] Search for any other references to legacy globals in TypeScript
- [ ] Update if found

#### 6.2 Extension Integration Testing Checklist

**Test in Positron Extension Development Host**:

##### File Loading Commands
- [ ] `squiggy.openPOD5` - Load POD5 file
  - [ ] Verify session appears in Variables pane
  - [ ] Verify read count correct in File Panel
  - [ ] Check console has no errors

- [ ] `squiggy.openBAM` - Load BAM file
  - [ ] Verify BAM info appears in File Panel
  - [ ] Verify modifications detected (if present)
  - [ ] Verify event alignment status correct

- [ ] `squiggy.closeFiles` - Clear all files
  - [ ] Verify session cleared in Variables pane
  - [ ] Verify File Panel shows "No files loaded"

##### Plot Commands
- [ ] `squiggy.plotRead` - Plot single read (SINGLE mode)
  - [ ] Verify HTML renders in plot panel
  - [ ] Verify base annotations appear (if BAM loaded)
  - [ ] Verify modification track appears (if mods present)
  - [ ] Test with different normalization methods

- [ ] `squiggy.plotRead` - Plot single read (EVENTALIGN mode)
  - [ ] Verify position-based x-axis
  - [ ] Verify base annotations render correctly
  - [ ] Test without BAM (should show error)

- [ ] `squiggy.plotAggregate` - Aggregate plot
  - [ ] Verify three tracks render (signal, pileup, quality)
  - [ ] Verify synchronized zoom/pan
  - [ ] Verify reference base labels on pileup

##### Reads Panel
- [ ] Open Reads View
  - [ ] Verify reads list populated
  - [ ] Verify search functionality works
  - [ ] Verify grouping by reference (if BAM loaded)
  - [ ] Click read → verify plot generated

##### Plot Options Panel
- [ ] Change normalization method
  - [ ] Verify plot updates with new normalization
- [ ] Toggle dwell time coloring
  - [ ] Verify color mapper appears
- [ ] Change theme (LIGHT ↔ DARK)
  - [ ] Verify colors update correctly

##### Modifications Panel
- [ ] View modification types (if present)
- [ ] Filter by modification type
  - [ ] Verify only selected mods shown
- [ ] Adjust probability threshold
  - [ ] Verify filtering works

#### 6.3 Message Passing Validation Tests

**Verify TypeScript → Python → TypeScript round-trip**:

**Test 1: Variable Access**
```typescript
// Test: Can TypeScript read _squiggy_session attributes?
const numReads = await positronClient.getVariable('len(_squiggy_session.read_ids)');
assert(typeof numReads === 'number');

const bamPath = await positronClient.getVariable('_squiggy_session.bam_path');
assert(typeof bamPath === 'string' || bamPath === null);
```

**Test 2: Function Execution**
```typescript
// Test: Can TypeScript call new factory-based API?
await positronClient.executeSilent(`
import squiggy
html = squiggy.plot_read('${readId}', mode='SINGLE', theme='DARK')
`);
const html = await positronClient.getVariable('html');
assert(html.includes('<script'));  // Valid HTML
```

**Test 3: Error Handling**
```typescript
// Test: Does Python raise appropriate errors?
try {
    await positronClient.executeSilent(`
import squiggy
squiggy.plot_read('nonexistent_read')
`);
    fail('Should have raised ValueError');
} catch (error) {
    assert(error.message.includes('Read not found'));
}
```

**Tasks**:
- [ ] Create `tests/integration/test_extension_integration.ts`
- [ ] Implement automated tests for message passing
- [ ] Run tests in CI pipeline

---

### Phase 7: Testing (Week 5, Days 3-4)

#### 7.1 Unit Tests

**New test files to create**:

- [ ] `tests/test_theme_manager.py`
  - Test theme application to figures
  - Test color getters
  - Test LIGHT vs DARK themes

- [ ] `tests/test_annotation_renderer.py`
  - Test time-based region calculation
  - Test position-based region calculation
  - Test patch rendering
  - Test label rendering
  - Test with sample POD5/BAM data

- [ ] `tests/test_modification_track.py`
  - Test track creation
  - Test column layout
  - Test modification filtering
  - Test probability thresholds

- [ ] `tests/test_plot_strategies.py`
  - Test each strategy independently
  - Test data validation
  - Test with various normalization methods
  - Test with/without annotations

- [ ] `tests/test_plot_factory.py`
  - Test factory pattern
  - Test strategy registration
  - Test error handling for unknown modes

- [ ] `tests/test_comparison.py`
  - Test ComparisonSession
  - Test delta calculations (absolute/percentage)
  - Test reference validation
  - Test model provenance parsing

**Tasks**:
- [ ] Create all test files
- [ ] Achieve >90% code coverage for new modules
- [ ] Run pytest with coverage: `pytest tests/ --cov=squiggy --cov-report=html`

#### 7.2 Characterization Tests

**Goal**: Ensure refactored plots look identical to original plots

**Approach**:
1. **Before refactoring**: Capture HTML output for each plot type
2. **After refactoring**: Compare new output with captured baseline
3. **Use**: pytest-snapshot or manual visual diff

**Tasks**:
- [ ] Capture baseline outputs (all plot modes, both themes)
- [ ] Create characterization test suite
- [ ] Run after each strategy implementation
- [ ] Investigate any differences (expected vs actual)

**Test cases**:
- [ ] SINGLE mode - LIGHT theme
- [ ] SINGLE mode - DARK theme
- [ ] SINGLE mode with base annotations
- [ ] SINGLE mode with modification track
- [ ] EVENTALIGN mode
- [ ] OVERLAY mode (multiple reads)
- [ ] STACKED mode
- [ ] AGGREGATE mode (all three tracks)

**Example test**:
```python
def test_single_read_output_unchanged(sample_pod5_file, sample_bam_file):
    """Verify refactored SINGLE mode produces identical output"""
    import squiggy

    # Load files
    squiggy.load_pod5(sample_pod5_file)
    squiggy.load_bam(sample_bam_file)

    # Generate plot with refactored code
    html = squiggy.plot_read('read_001', mode='SINGLE', theme='LIGHT')

    # Compare with baseline (captured before refactoring)
    with open('tests/baselines/single_read_light.html', 'r') as f:
        baseline = f.read()

    # Allow for minor differences (timestamps, etc.)
    assert html_similar(html, baseline, tolerance=0.99)
```

#### 7.3 Integration Tests

**Full workflow tests**: load → plot → export

**Tasks**:
- [ ] Test POD5-only workflow
- [ ] Test POD5 + BAM workflow
- [ ] Test modification filtering workflow
- [ ] Test aggregate workflow
- [ ] Test comparison workflow (new)
- [ ] Test error cases (missing files, invalid read IDs, etc.)

**Example**:
```python
def test_full_workflow_with_bam(sample_data):
    """Test complete workflow: load POD5, load BAM, plot with annotations"""
    import squiggy

    # Load files
    session = squiggy.load_pod5(sample_data['pod5'])
    assert len(session.read_ids) > 0

    session = squiggy.load_bam(sample_data['bam'])
    assert session.bam_path is not None
    assert session.bam_info['has_event_alignment'] is True

    # Plot with event alignment
    html = squiggy.plot_read(
        session.read_ids[0],
        mode='EVENTALIGN',
        normalization='ZNORM'
    )

    # Verify output
    assert '<script' in html  # Valid Bokeh HTML
    assert 'BokehJS' in html

    # Cleanup
    squiggy.close_pod5()
    squiggy.close_bam()
    assert session.reader is None
```

#### 7.4 Performance Tests

**Ensure refactoring doesn't degrade performance**:

**Tasks**:
- [ ] Benchmark plot generation time (before vs after)
- [ ] Test with large datasets (10K+ reads)
- [ ] Profile memory usage
- [ ] Identify any regressions

**Example**:
```python
import time

def test_plot_performance(large_pod5_file):
    """Verify plot generation performance acceptable"""
    import squiggy

    squiggy.load_pod5(large_pod5_file)
    read_id = squiggy.get_read_ids()[0]

    start = time.time()
    html = squiggy.plot_read(read_id, mode='SINGLE')
    elapsed = time.time() - start

    # Should complete in <1 second
    assert elapsed < 1.0, f"Plot took {elapsed:.2f}s (too slow!)"
```

---

### Phase 8: Documentation & Cleanup (Week 5, Day 5)

#### 8.1 Update CLAUDE.md

**File**: `CLAUDE.md`

**Tasks**:
- [ ] Document new architecture (Strategy Pattern)
- [ ] Update module structure section
- [ ] Document PlotFactory usage
- [ ] Update examples to use new pattern
- [ ] Document global state cleanup (no more `global` keyword)
- [ ] Add comparison plot examples

**Example section to add**:
```markdown
### Strategy Pattern Architecture

Squiggy uses the Strategy Pattern for plot generation:

**Creating plots programmatically**:
```python
from squiggy.plot_factory import PlotFactory
from squiggy.constants import PlotMode, Theme

# Get strategy for single read plot
strategy = PlotFactory.create_plot(PlotMode.SINGLE, Theme.LIGHT)

# Prepare data
data = {
    'signal': read_obj.signal,
    'read_id': 'read_001',
    'sample_rate': 4000,
    'sequence': 'ACGTACGT...',
    'seq_to_sig_map': [0, 10, 20, ...],
}

# Generate plot
html, figure = strategy.create_plot(data, options)
```

**Adding new plot types**:
1. Create new strategy class in `squiggy/plot_strategies/`
2. Extend `PlotStrategy` base class
3. Implement `create_plot()` and `validate_data()`
4. Register in `PlotFactory._strategies` dict
5. Add tests

**Reusable components**:
- `ThemeManager` - Centralized theme management
- `BaseAnnotationRenderer` - Base annotation rendering (time/position modes)
- `ModificationTrackBuilder` - Modification track creation
```

#### 8.2 Update Module Docstrings

**Tasks**:
- [ ] Update all new module docstrings (theme_manager.py, annotation_renderer.py, etc.)
- [ ] Add usage examples to docstrings
- [ ] Document parameters and return values
- [ ] Add cross-references between related modules

#### 8.3 Create Migration Guide

**New file**: `docs/MIGRATION.md`

**Content**:
- Breaking changes (legacy globals removed)
- How to update code using old patterns
- New recommended patterns
- Deprecation timeline

**Example**:
```markdown
# Migration Guide: v0.1.x → v0.2.0

## Breaking Changes

### Legacy Global Variables Removed

**Old pattern** (no longer works):
```python
from squiggy.io import _current_pod5_reader, _current_read_ids

print(f"Loaded {len(_current_read_ids)} reads")
```

**New pattern**:
```python
from squiggy.io import _squiggy_session

print(f"Loaded {len(_squiggy_session.read_ids)} reads")
```

### SquigglePlotter Direct Usage Deprecated

**Old pattern** (deprecated):
```python
from squiggy.plotter import SquigglePlotter

html, fig = SquigglePlotter.plot_single_read(...)
```

**New pattern**:
```python
import squiggy

html = squiggy.plot_read(read_id, mode='SINGLE')  # Simpler!
```
```

#### 8.4 Update USER_GUIDE.md

**File**: `docs/USER_GUIDE.md`

**Tasks**:
- [ ] Add "Multi-File Comparison" section
- [ ] Update examples to reflect new patterns
- [ ] Add troubleshooting for common issues

#### 8.5 Cleanup & Polish

**Tasks**:
- [ ] Remove dead code
- [ ] Fix any remaining linting issues: `pixi run lint`
- [ ] Format all code: `pixi run format`
- [ ] Update NEWS.md with changes
- [ ] Update version in `squiggy/__init__.py` and `package.json`

---

## File Structure After Refactoring

```
squiggy/
├── __init__.py                    # Public API (refactored)
├── io.py                          # Session management (no global keyword!)
├── constants.py                   # Add COMPARISON mode
├── theme_manager.py               # NEW - Centralized theme logic
├── annotation_renderer.py         # NEW - Reusable base annotations
├── modification_track.py          # NEW - Modification tracks
├── plot_factory.py                # NEW - Factory pattern
├── comparison.py                  # NEW - Delta calculations for #61
├── plot_strategies/               # NEW DIRECTORY
│   ├── __init__.py                # Export all strategies
│   ├── base.py                    # Abstract base class
│   ├── single_read.py             # SINGLE mode
│   ├── eventalign.py              # EVENTALIGN mode
│   ├── overlay.py                 # OVERLAY mode
│   ├── stacked.py                 # STACKED mode
│   ├── aggregate.py               # AGGREGATE mode
│   └── comparison.py              # COMPARISON mode (NEW for #61)
├── plotter.py                     # DEPRECATED (keep for compatibility)
├── api.py                         # Object-oriented API (unchanged)
├── alignment.py                   # (unchanged)
├── normalization.py               # (unchanged)
├── modifications.py               # (unchanged)
└── utils.py                       # (unchanged)

tests/
├── test_theme_manager.py          # NEW
├── test_annotation_renderer.py    # NEW
├── test_modification_track.py     # NEW
├── test_plot_strategies.py        # NEW
├── test_plot_factory.py           # NEW
├── test_comparison.py             # NEW
├── test_io.py                     # Update for refactored globals
├── test_api.py                    # (mostly unchanged)
├── ... (existing tests)

src/
├── commands/
│   └── file-commands.ts           # FIX: Update _current_bam_path reference
├── backend/
│   └── squiggy-runtime-api.ts     # (no changes needed - already uses _squiggy_session)
└── ... (no other TypeScript changes needed)
```

---

## Success Criteria

### Functional Requirements
- [ ] All existing plot types work identically to before
- [ ] All Positron extension commands work correctly
- [ ] TypeScript ↔ Python message passing works
- [ ] Session state visible in Variables pane
- [ ] No console pollution (silent execution works)

### Code Quality
- [ ] No `global` keyword in codebase
- [ ] Test coverage >90% for new modules
- [ ] All linting checks pass
- [ ] All tests pass (Python + TypeScript)
- [ ] No performance regressions

### Architecture
- [ ] Strategy pattern implemented for all plot types
- [ ] Shared components extracted (ThemeManager, AnnotationRenderer, ModBuilder)
- [ ] PlotFactory provides clean abstraction
- [ ] Easy to add new plot types

### Comparison Feature (#61)
- [ ] ComparisonSession manages A/B state
- [ ] ComparisonPlotStrategy implemented
- [ ] Delta calculations work (absolute/percentage)
- [ ] Public API functions work (`load_dataset_a/b`, `plot_comparison`)

### Documentation
- [ ] CLAUDE.md updated with new architecture
- [ ] Migration guide created
- [ ] USER_GUIDE.md updated
- [ ] All modules have comprehensive docstrings

---

## Risk Assessment & Mitigation

### High Risk Areas

**1. Message Passing Between TypeScript and Python**
- **Risk**: Refactoring could break extension communication
- **Mitigation**:
  - Characterization tests before making changes
  - Incremental testing after each phase
  - Keep public API signatures unchanged
  - Dedicated integration test suite (Phase 6)

**2. Visual Output Changes**
- **Risk**: Plots might look different after refactoring
- **Mitigation**:
  - Capture baseline HTML outputs before refactoring
  - Visual diff tests after each strategy implementation
  - Use identical Bokeh API calls

**3. Performance Degradation**
- **Risk**: Additional abstraction layers slow plot generation
- **Mitigation**:
  - Benchmark before/after
  - Profile hot paths
  - Optimize if needed (strategy pattern shouldn't add overhead)

**4. Breaking User Code**
- **Risk**: Removing legacy globals breaks existing notebooks
- **Mitigation**:
  - User explicitly accepted breaking changes
  - Provide clear migration guide
  - Deprecation warnings before removal

### Low Risk Areas

- Theme manager extraction (isolated change)
- Annotation renderer (consolidates existing logic)
- Modification track builder (currently only used in one place)
- Factory pattern (public API unchanged)

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 1: Foundation & Cleanup | 2 days | Remove `global`, create base interface |
| Phase 2: Extract Components | 3 days | ThemeManager, AnnotationRenderer, ModBuilder |
| Phase 3: Implement Strategies | 2 weeks | All 5 plot type strategies |
| Phase 4: Factory & API | 1 week | PlotFactory, refactored public API |
| Phase 5: Comparison (#61) | 1 week | ComparisonSession, ComparisonPlotStrategy |
| Phase 6: Extension Integration | 2 days | Fix TypeScript, test message passing |
| Phase 7: Testing | 2 days | Unit, characterization, integration tests |
| Phase 8: Documentation | 1 day | Update docs, migration guide |
| **Total** | **5-6 weeks** | Production-ready refactored codebase |

---

## Next Steps

1. **Review this task file** - Ensure plan is clear and complete
2. **Start Phase 1** - Clean up global state pattern (low risk, high value)
3. **Commit incrementally** - Small commits after each sub-task
4. **Test continuously** - Run tests after each change
5. **Update this file** - Check off tasks as completed

---

## Notes & Considerations

### Why Strategy Pattern?

- **Extensibility**: Adding the comparison plot (and future plot types) becomes trivial
- **Maintainability**: Each plot type in separate file with clear responsibility
- **Testability**: Can mock/test each strategy independently
- **Reusability**: Shared components reduce duplication

### Why Remove `global` Keyword?

- **Python best practice**: `global` is considered a code smell
- **Not needed here**: We're mutating object attributes, not reassigning variables
- **Cleaner code**: Makes state management more explicit
- **Still works for kernel**: Module-level state persists in REPL

### Why Keep Module-Level State?

- **Kernel requirement**: Extension needs persistent state across function calls
- **User convenience**: REPL-friendly pattern (`squiggy.load_pod5()` then `squiggy.plot_read()`)
- **Pythonic**: Module-level singletons are acceptable for this use case
- **OO API available**: Users wanting explicit state can use `Pod5File()` / `BamFile()` classes

### Extension Integration Safety

The TypeScript extension already uses `_squiggy_session` extensively (thanks to PR #59), so the refactoring risk is minimal. Only one line needs updating (`file-commands.ts:403`). The strategy pattern is purely Python-internal - TypeScript still calls `squiggy.plot_read()` and gets HTML back.

---

**Branch**: `refactor-plotting-architecture`
**Ready to begin**: ✅
**Estimated completion**: 5-6 weeks from start
