# Migration Guide: Removing Plotnine/Matplotlib Dependencies

This document outlines the steps to fully remove plotnine and matplotlib dependencies after migrating to Bokeh.

## Current Status (Bokeh Branch)

- ✅ Main application (`viewer.py`) fully migrated to Bokeh
- ✅ New `plotter_bokeh.py` module with all plotting functionality
- ❌ Old `plotter.py` still exists (unused by app)
- ❌ Tests still use old plotnine-based plotter
- ❌ Plotnine/matplotlib still in requirements.txt

## Steps to Complete Migration

### 1. Update requirements.txt

Remove these dependencies:

```diff
- plotnine>=0.12.0
```

**Note:** Matplotlib is a plotnine dependency and will be automatically removed when plotnine is removed.

### 2. Delete Old Plotter Module

```bash
rm src/squiggy/plotter.py
```

Or move it to an archive directory if you want to keep it for reference:

```bash
mkdir -p archive
mv src/squiggy/plotter.py archive/
```

### 3. Update Test Files

#### tests/test_plotting.py

Replace all imports:

```diff
- from squiggy.plotter import SquigglePlotter
+ from squiggy.plotter_bokeh import BokehSquigglePlotter
```

Update test methods:

**Old (PNG-based tests):**
```python
def test_plot_to_png_buffer(self, sample_pod5_file):
    plot = SquigglePlotter.create_plot(...)
    buffer = BytesIO()
    plot.save(buffer, format="png", dpi=100)
    assert buffer.tell() > 0
```

**New (HTML-based tests):**
```python
def test_plot_to_html(self, sample_pod5_file):
    html = BokehSquigglePlotter.plot_single_read(...)
    assert len(html) > 0
    assert "<html>" in html
    assert "bokeh" in html.lower()
```

**Methods that need updating:**
- `test_plot_to_png_buffer()` → Test HTML string instead of PNG
- `test_plot_with_subsampled_signal()` → Use `plot_single_read()` instead of `create_plot()`
- `test_signal_dataframe_time_calculation()` → Remove (Bokeh doesn't use DataFrame intermediate)
- `test_multi_read_overlay_plot()` → Test HTML output instead of PNG
- `test_multi_read_stacked_plot()` → Test HTML output instead of PNG
- `test_downsampling_functionality()` → Remove or adapt (Bokeh handles this differently)

#### tests/test_main.py

Similar updates for the `TestSquigglePlotter` class:

```diff
- from squiggy.plotter import SquigglePlotter
+ from squiggy.plotter_bokeh import BokehSquigglePlotter

  class TestSquigglePlotter:
-     """Tests for the SquigglePlotter class."""
+     """Tests for the BokehSquigglePlotter class."""

      def test_create_plot(self, sample_pod5_file):
-         plot = SquigglePlotter.create_plot(...)
+         html = BokehSquigglePlotter.plot_single_read(...)
-         assert plot is not None
+         assert html is not None
+         assert len(html) > 0
```

### 4. Update Normalization Tests

If `tests/test_plotting.py::test_signal_normalization_methods` imports from plotter:

```diff
  def test_signal_normalization_methods(self):
      from squiggy.constants import NormalizationMethod
-     from squiggy.normalization import normalize_signal
+     from squiggy.plotter_bokeh import BokehSquigglePlotter

      signal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

-     znorm = normalize_signal(signal, NormalizationMethod.ZNORM)
+     znorm = BokehSquigglePlotter.normalize_signal(signal, NormalizationMethod.ZNORM)
      # ... rest of test
```

### 5. Run Tests to Verify

```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_plotting.py
pytest tests/test_main.py
```

### 6. Update Documentation

Files that may reference plotnine/matplotlib:

- `README.md` - Update plotting backend description
- `docs/` - Update any documentation mentioning matplotlib
- `CLAUDE.md` - Update architecture section to reference Bokeh instead of plotnine

### 7. Clean Up Any Remaining References

Search for any lingering references:

```bash
# Search entire codebase
grep -r "plotnine\|matplotlib\|SquigglePlotter" src/ tests/ docs/ --exclude-dir=.venv

# Search for old imports
grep -r "from squiggy.plotter import" src/ tests/
```

## Testing Checklist

After completing migration, verify:

- [ ] `pip install -e .` works without plotnine
- [ ] All tests pass (`pytest tests/`)
- [ ] Application launches successfully
- [ ] Can load POD5 files and display plots
- [ ] Interactive features work (zoom, pan, hover)
- [ ] Export to HTML works
- [ ] Multi-read modes work (overlay, stacked)
- [ ] Normalization methods work
- [ ] Base annotations display correctly (if BAM file provided)

## Rollback Plan

If issues arise, you can quickly rollback:

1. Restore `plotter.py` from git: `git checkout src/squiggy/plotter.py`
2. Re-add plotnine to requirements.txt
3. Revert viewer.py changes: `git checkout src/squiggy/viewer.py`

## Benefits of Removing Plotnine/Matplotlib

- **Smaller install size:** ~50-100MB reduction
- **Faster installation:** Fewer dependencies to compile
- **Simpler dependency tree:** Fewer potential conflicts
- **Single plotting backend:** Clearer code architecture
- **Interactive-only:** Forces interactive experience (which is better for signal exploration)

## Notes

- **Export formats:** After migration, plots will only export as interactive HTML files (not PNG/PDF/SVG)
  - Users can still use browser "Print to PDF" or screenshot tools if needed
  - For publication-quality figures, consider using bokeh's export_png (requires additional dependencies)

- **Bokeh export options:** If static image export is needed, you can optionally add:
  ```
  pip install selenium pillow
  ```
  And use `bokeh.io.export_png()` to generate PNGs from bokeh plots

- **Performance:** Bokeh plots may be larger in file size (HTML/JS vs PNG) but provide much better interactivity
