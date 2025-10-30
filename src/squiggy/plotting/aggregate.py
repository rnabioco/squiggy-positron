"""Aggregate multi-read visualization with synchronized tracks"""

from bokeh.embed import file_html
from bokeh.layouts import gridplot
from bokeh.models import Band, ColumnDataSource, HoverTool, LabelSet
from bokeh.resources import CDN

from ..constants import DARK_THEME, LIGHT_THEME, NormalizationMethod, Theme
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
) -> tuple[str, object]:
    """Plot aggregate multi-read visualization with three synchronized tracks

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

    # Link x-axes for synchronized zoom/pan
    p_pileup.x_range = p_signal.x_range
    p_quality.x_range = p_signal.x_range

    # Set explicit heights and change sizing mode for each track
    # Override the default "stretch_both" from _create_figure
    p_signal.sizing_mode = "stretch_width"
    p_signal.height = 300
    p_pileup.sizing_mode = "stretch_width"
    p_pileup.height = 200
    p_quality.sizing_mode = "stretch_width"
    p_quality.height = 200

    # Create grid layout with three tracks stacked vertically
    grid = gridplot(
        [[p_signal], [p_pileup], [p_quality]],
        sizing_mode="stretch_width",
        toolbar_location="right",
    )

    # Generate HTML
    html_title = f"Aggregate View - {reference_name} ({num_reads} reads)"
    html = file_html(grid, CDN, title=html_title)

    return html, grid
