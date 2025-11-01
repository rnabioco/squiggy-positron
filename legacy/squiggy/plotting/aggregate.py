"""Aggregate multi-read visualization with synchronized tracks"""

from bokeh.embed import file_html
from bokeh.layouts import gridplot
from bokeh.models import (
    Band,
    ColorBar,
    ColumnDataSource,
    HoverTool,
    LabelSet,
    LinearColorMapper,
)
from bokeh.palettes import Viridis256
from bokeh.resources import CDN
from bokeh.transform import transform

from ..constants import (
    DARK_THEME,
    LIGHT_THEME,
    MOD_TRACK_HEIGHT,
    MODIFICATION_CODES,
    NormalizationMethod,
    Theme,
)
from .base import (
    add_hover_tool,
    configure_legend,
    create_figure,
    get_base_colors,
)


def plot_aggregate(
    aggregate_stats: dict,
    pileup_stats: dict,
    quality_stats: dict,
    reference_name: str,
    num_reads: int,
    normalization: NormalizationMethod = NormalizationMethod.NONE,
    theme: Theme = Theme.LIGHT,
    modification_pileup_stats: dict | None = None,
) -> tuple[str, object]:
    """Plot aggregate multi-read visualization with synchronized tracks

    Args:
        aggregate_stats: Dict from calculate_aggregate_signal() with keys:
            positions, mean_signal, std_signal, median_signal, coverage
        pileup_stats: Dict from calculate_base_pileup() with keys:
            positions, counts (dict mapping pos to base counts)
        quality_stats: Dict from calculate_quality_by_position() with keys:
            positions, mean_quality, std_quality
        reference_name: Name of reference sequence
        num_reads: Number of reads included in aggregate
        normalization: Normalization method used
        theme: Color theme (LIGHT or DARK)
        modification_pileup_stats: Optional dict from calculate_modification_pileup()
            mapping (ref_pos, mod_type) -> ModPositionStats

    Returns:
        Tuple[str, object]: (HTML string, gridplot object)
    """
    theme_colors = DARK_THEME if theme == Theme.DARK else LIGHT_THEME
    base_colors = get_base_colors(theme)

    # Track 1: Signal aggregate with confidence bands
    p_signal = create_figure(
        title=f"Aggregate Signal - {reference_name} ({num_reads} reads)",
        x_label="Reference Position",
        y_label=f"Signal ({normalization.value})",
        theme=theme,
    )

    positions = aggregate_stats["positions"]
    mean_signal = aggregate_stats["mean_signal"]
    std_signal = aggregate_stats["std_signal"]
    coverage = aggregate_stats["coverage"]

    # Create confidence band (mean ± 1 std dev)
    upper = mean_signal + std_signal
    lower = mean_signal - std_signal

    # Data source for signal
    signal_source = ColumnDataSource(
        data={
            "x": positions,
            "mean": mean_signal,
            "upper": upper,
            "lower": lower,
            "std": std_signal,
            "coverage": coverage,
        }
    )

    # Add confidence band (use theme-appropriate color)
    band_color = (
        "#56B4E9" if theme == Theme.LIGHT else "#0072B2"
    )  # Light blue for light mode, darker blue for dark mode
    band = Band(
        base="x",
        lower="lower",
        upper="upper",
        source=signal_source,
        level="underlay",
        fill_alpha=0.3,
        fill_color=band_color,
        line_width=0,
    )
    p_signal.add_layout(band)

    # Add mean line
    mean_line = p_signal.line(
        "x",
        "mean",
        source=signal_source,
        line_width=2,
        color=theme_colors["signal_line"],
        legend_label="Mean signal",
    )

    # Add hover tool for signal
    add_hover_tool(
        p_signal,
        [mean_line],
        [
            ("Position", "@x"),
            ("Mean", "@mean{0.2f}"),
            ("Std Dev", "@std{0.2f}"),
            ("Coverage", "@coverage"),
        ],
    )

    # Hide legend (title is descriptive enough)
    p_signal.legend.visible = False

    # Track 2: Base pileup (IGV-style stacked bars)
    p_pileup = create_figure(
        title="Base Call Pileup",
        x_label="Reference Position",
        y_label="Base Proportion",
        theme=theme,
    )

    # Calculate proportions for each base at each position
    pileup_positions = pileup_stats["positions"]
    pileup_counts = pileup_stats["counts"]
    reference_bases = pileup_stats.get("reference_bases", {})

    # Prepare data for stacked bars with pre-computed tops and bottoms
    pileup_data = {
        "x": [],
        "A": [],
        "C": [],
        "G": [],
        "T": [],
        "total": [],
        "A_bottom": [],
        "A_top": [],
        "C_bottom": [],
        "C_top": [],
        "G_bottom": [],
        "G_top": [],
        "T_bottom": [],
        "T_top": [],
        "ref_base": [],  # Reference base at each position
    }

    for pos in pileup_positions:
        counts = pileup_counts[pos]
        total = sum(counts.values())
        if total > 0:
            pileup_data["x"].append(pos)
            pileup_data["total"].append(total)

            # Add reference base if available
            ref_base = reference_bases.get(pos, "")
            pileup_data["ref_base"].append(ref_base)

            # Calculate proportions
            proportions = {}
            for base in ["A", "C", "G", "T"]:
                proportions[base] = counts.get(base, 0) / total
                pileup_data[base].append(proportions[base])

            # Calculate cumulative positions for stacking
            cumulative = 0.0
            for base in ["A", "C", "G", "T"]:
                pileup_data[f"{base}_bottom"].append(cumulative)
                cumulative += proportions[base]
                pileup_data[f"{base}_top"].append(cumulative)

    pileup_source = ColumnDataSource(data=pileup_data)

    # Create stacked vbar for each base
    bar_width = 0.8
    renderers = []

    # Stack from bottom: A, then C, then G, then T
    for base in ["A", "C", "G", "T"]:
        if base in base_colors:
            bar = p_pileup.vbar(
                x="x",
                top=f"{base}_top",
                bottom=f"{base}_bottom",
                width=bar_width,
                color=base_colors[base],
                alpha=0.8,
                legend_label=base,
                source=pileup_source,
            )
            renderers.append(bar)

    # Add hover tool for pileup
    hover_pileup = HoverTool(
        renderers=renderers,
        tooltips=[
            ("Position", "@x"),
            ("A", "@A{0.0%}"),
            ("C", "@C{0.0%}"),
            ("G", "@G{0.0%}"),
            ("T", "@T{0.0%}"),
            ("Total", "@total"),
        ],
        mode="vline",
    )
    p_pileup.add_tools(hover_pileup)
    configure_legend(p_pileup)

    # Hide legend (title is descriptive enough)
    p_pileup.legend.visible = False

    # Add reference base labels above bars
    if reference_bases:
        # Extend y-axis range to accommodate labels
        p_pileup.y_range.end = 1.20  # Make room for labels above bars

        # Add separate text labels for each base, grouped by base type for coloring
        for base_type in ["A", "C", "G", "T", "N"]:
            label_data = {
                "x": [],
                "y": [],
                "text": [],
            }

            for i, pos in enumerate(pileup_data["x"]):
                ref_base = pileup_data["ref_base"][i]
                if ref_base == base_type:
                    label_data["x"].append(pos)
                    label_data["y"].append(1.05)  # Slightly above the top of the bar
                    label_data["text"].append(ref_base)

            if label_data["x"]:
                label_source = ColumnDataSource(data=label_data)
                # Use the base color for this base type
                label_color = base_colors.get(base_type, theme_colors["axis_text"])
                labels = LabelSet(
                    x="x",
                    y="y",
                    text="text",
                    source=label_source,
                    text_align="center",
                    text_baseline="bottom",
                    text_color=label_color,
                    text_font_size="11pt",
                    text_font_style="bold",
                )
                p_pileup.add_layout(labels)

    # Track 3: Quality scores
    p_quality = create_figure(
        title="Average Quality Scores",
        x_label="Reference Position",
        y_label="Mean Quality (Phred)",
        theme=theme,
    )

    quality_positions = quality_stats["positions"]
    mean_quality = quality_stats["mean_quality"]
    std_quality = quality_stats["std_quality"]

    # Data source for quality
    quality_source = ColumnDataSource(
        data={
            "x": quality_positions,
            "mean_q": mean_quality,
            "std_q": std_quality,
        }
    )

    # Add quality line (use theme-appropriate color)
    quality_color = (
        "#009E73" if theme == Theme.LIGHT else "#56B4E9"
    )  # Green for light mode, blue for dark mode
    quality_line = p_quality.line(
        "x",
        "mean_q",
        source=quality_source,
        line_width=2,
        color=quality_color,
        legend_label="Mean quality",
    )

    # Add hover tool for quality
    add_hover_tool(
        p_quality,
        [quality_line],
        [
            ("Position", "@x"),
            ("Mean Q", "@mean_q{0.1f}"),
            ("Std Dev", "@std_q{0.1f}"),
        ],
    )

    # Hide legend (title is descriptive enough)
    p_quality.legend.visible = False

    # Track 4: Modification heatmap (optional)
    p_modifications = None
    if modification_pileup_stats:
        p_modifications = _create_modification_heatmap(
            modification_pileup_stats, theme, theme_colors
        )

    # Link x-axes for synchronized zoom/pan
    p_pileup.x_range = p_signal.x_range
    p_quality.x_range = p_signal.x_range
    if p_modifications:
        p_modifications.x_range = p_signal.x_range

    # Set explicit heights and change sizing mode for each track
    # Override the default "stretch_both" from _create_figure
    p_signal.sizing_mode = "stretch_width"
    p_signal.height = 300
    p_pileup.sizing_mode = "stretch_width"
    p_pileup.height = 200
    p_quality.sizing_mode = "stretch_width"
    p_quality.height = 200
    if p_modifications:
        p_modifications.sizing_mode = "stretch_width"
        p_modifications.height = MOD_TRACK_HEIGHT

    # Create grid layout with tracks stacked vertically
    tracks = [[p_signal], [p_pileup], [p_quality]]
    if p_modifications:
        tracks.append([p_modifications])

    grid = gridplot(
        tracks,
        sizing_mode="stretch_width",
        toolbar_location="right",
    )

    # Generate HTML
    html_title = f"Aggregate View - {reference_name} ({num_reads} reads)"
    html = file_html(grid, CDN, title=html_title)

    return html, grid


def _create_modification_heatmap(
    modification_pileup_stats: dict, theme: Theme, theme_colors: dict
):
    """Create a heatmap track for base modifications

    Args:
        modification_pileup_stats: Dict mapping (ref_pos, mod_type) -> ModPositionStats
        theme: Color theme (LIGHT or DARK)
        theme_colors: Theme color dictionary

    Returns:
        Bokeh figure with modification heatmap
    """
    p_mod = create_figure(
        title="Base Modifications (modBAM)",
        x_label="Reference Position",
        y_label="Modification Type",
        theme=theme,
    )

    # Organize data by modification type
    # Group modifications by type
    mod_types = set()
    for (ref_pos, mod_type), stats in modification_pileup_stats.items():
        mod_types.add(mod_type)

    # Sort modification types: strings first, then ChEBI codes
    mod_types_sorted = sorted(mod_types, key=lambda x: (isinstance(x, int), str(x)))

    # Create y-axis mapping (mod type -> y position)
    mod_type_to_y = {mod_type: i for i, mod_type in enumerate(mod_types_sorted)}

    # Prepare data for heatmap
    heatmap_data = {
        "x": [],  # Reference position
        "y": [],  # Y position (categorical)
        "mod_type": [],  # Modification type (for display)
        "mod_name": [],  # Modification name (e.g., "5mC")
        "mean_prob": [],  # Mean probability (for color)
        "coverage": [],  # Number of reads
        "frequency": [],  # Modification frequency (if tau was used)
    }

    for (ref_pos, mod_type), stats in modification_pileup_stats.items():
        heatmap_data["x"].append(ref_pos)
        heatmap_data["y"].append(mod_type_to_y[mod_type])
        heatmap_data["mod_type"].append(str(mod_type))
        mod_name = MODIFICATION_CODES.get(mod_type, str(mod_type))
        heatmap_data["mod_name"].append(mod_name)
        heatmap_data["mean_prob"].append(stats.mean_prob)
        heatmap_data["coverage"].append(stats.coverage)
        # Frequency may be None if no threshold was used
        freq = stats.frequency if stats.frequency is not None else stats.mean_prob
        heatmap_data["frequency"].append(freq)

    heatmap_source = ColumnDataSource(data=heatmap_data)

    # Color mapper for mean probability (0-1 range)
    color_mapper = LinearColorMapper(
        palette=Viridis256, low=0, high=1, nan_color="lightgray"
    )

    # Create rect glyphs for heatmap
    rect_height = 0.8  # Leave small gap between rows
    rect_width = 0.8  # Leave small gap between positions

    rects = p_mod.rect(
        x="x",
        y="y",
        width=rect_width,
        height=rect_height,
        source=heatmap_source,
        fill_color=transform("mean_prob", color_mapper),
        line_color=None,
    )

    # Add hover tool
    hover_mod = HoverTool(
        renderers=[rects],
        tooltips=[
            ("Position", "@x"),
            ("Modification", "@mod_name (@mod_type)"),
            ("Mean Prob", "@mean_prob{0.3f}"),
            ("Coverage", "@coverage"),
            ("Frequency", "@frequency{0.3f}"),
        ],
        mode="mouse",
    )
    p_mod.add_tools(hover_mod)

    # Set y-axis tick labels to modification names
    y_ticks = [
        (i, MODIFICATION_CODES.get(mod_type, str(mod_type)))
        for i, mod_type in enumerate(mod_types_sorted)
    ]
    p_mod.yaxis.ticker = [y for y, _ in y_ticks]
    p_mod.yaxis.major_label_overrides = {y: label for y, label in y_ticks}

    # Add color bar
    color_bar = ColorBar(
        color_mapper=color_mapper,
        width=10,
        location=(0, 0),
        title="Mean Prob",
        title_text_font_size="10pt",
    )
    p_mod.add_layout(color_bar, "right")

    # Set y-axis range to accommodate all modification types
    p_mod.y_range.start = -0.5
    p_mod.y_range.end = len(mod_types_sorted) - 0.5

    return p_mod
