"""
Reference sequence track rendering component

Provides reusable component for rendering reference sequence tracks
across different plot strategies.
"""

from bokeh.models import ColumnDataSource
from bokeh.models.ranges import Range1d

from .theme_manager import ThemeManager


class ReferenceTrackRenderer:
    """
    Renders reference sequence tracks for plots

    Provides consistent reference sequence visualization across different
    plot types. Displays color-coded nucleotides (A/T/G/C) with optional
    mismatch highlighting.

    Examples:
        >>> from squiggy.rendering import ReferenceTrackRenderer, ThemeManager
        >>> from squiggy.constants import Theme
        >>>
        >>> theme_manager = ThemeManager(Theme.LIGHT)
        >>> renderer = ReferenceTrackRenderer(theme_manager)
        >>>
        >>> # Create reference track
        >>> fig = renderer.create_reference_track(
        ...     reference_sequence="ACGTACGT",
        ...     positions=[100, 101, 102, 103, 104, 105, 106, 107],
        ...     x_label="Genomic Position",
        ...     title="Reference Sequence"
        ... )
    """

    def __init__(self, theme_manager: ThemeManager):
        """
        Initialize reference track renderer

        Args:
            theme_manager: ThemeManager instance for consistent styling
        """
        self.theme_manager = theme_manager

    def create_reference_track(
        self,
        reference_sequence: str,
        positions: list | None = None,
        x_label: str = "Position",
        title: str = "Reference Sequence",
        height: int = 80,
        query_sequence: str | None = None,
    ):
        """
        Create a Bokeh figure showing reference sequence

        Args:
            reference_sequence: String of nucleotides (A/T/G/C/N)
            positions: List of x-axis positions for each base (default: 0, 1, 2, ...)
            x_label: Label for x-axis
            title: Plot title
            height: Figure height in pixels
            query_sequence: Optional query sequence for mismatch highlighting

        Returns:
            Bokeh figure with reference sequence track

        Raises:
            ValueError: If reference_sequence is empty or positions length mismatch
        """
        if not reference_sequence:
            raise ValueError("reference_sequence cannot be empty")

        # Default positions if not provided
        if positions is None:
            positions = list(range(len(reference_sequence)))

        if len(positions) != len(reference_sequence):
            raise ValueError(
                f"Position count ({len(positions)}) must match sequence length ({len(reference_sequence)})"
            )

        # Create figure
        fig = self.theme_manager.create_figure(
            title=title,
            x_label=x_label,
            y_label="",  # No y-label for reference track
            height=height,
        )

        # Get base colors from theme
        base_colors = self.theme_manager.get_base_colors()

        # Prepare data for text rendering
        text_data = {
            "x": [],
            "y": [],
            "text": [],
            "color": [],
            "is_mismatch": [],
        }

        y_position = 0.5  # Center vertically

        for i, (pos, ref_base) in enumerate(
            zip(positions, reference_sequence, strict=True)
        ):
            ref_base_upper = ref_base.upper()

            # Determine color
            base_color = base_colors.get(
                ref_base_upper, base_colors.get("N", "#808080")
            )

            # Check for mismatch if query sequence provided
            is_mismatch = False
            if query_sequence and i < len(query_sequence):
                query_base = query_sequence[i].upper()
                is_mismatch = ref_base_upper != query_base and query_base != "N"

            text_data["x"].append(pos)
            text_data["y"].append(y_position)
            text_data["text"].append(ref_base_upper)
            text_data["color"].append(base_color)
            text_data["is_mismatch"].append(is_mismatch)

        # Split into match and mismatch for different styling
        match_data = {
            "x": [],
            "y": [],
            "text": [],
            "color": [],
        }
        mismatch_data = {
            "x": [],
            "y": [],
            "text": [],
            "color": [],
        }

        for i in range(len(text_data["x"])):
            target = mismatch_data if text_data["is_mismatch"][i] else match_data
            target["x"].append(text_data["x"][i])
            target["y"].append(text_data["y"][i])
            target["text"].append(text_data["text"][i])
            target["color"].append(text_data["color"][i])

        # Render matching bases (normal style)
        if match_data["x"]:
            match_source = ColumnDataSource(data=match_data)
            fig.text(
                x="x",
                y="y",
                text="text",
                source=match_source,
                text_color="color",
                text_align="center",
                text_baseline="middle",
                text_font_size="10pt",
            )

        # Render mismatches (bold style with box)
        if mismatch_data["x"]:
            mismatch_source = ColumnDataSource(data=mismatch_data)

            # Add rectangle background for mismatches
            rect_width = (
                (max(positions) - min(positions)) / len(positions)
                if len(positions) > 1
                else 1
            )
            fig.rect(
                x="x",
                y="y",
                width=rect_width * 0.9,
                height=0.8,
                source=mismatch_source,
                fill_color="#ffcccc",  # Light red background
                fill_alpha=0.3,
                line_color="#ff0000",  # Red border
                line_width=1,
            )

            # Add bold text for mismatches
            fig.text(
                x="x",
                y="y",
                text="text",
                source=mismatch_source,
                text_color="color",
                text_align="center",
                text_baseline="middle",
                text_font_size="11pt",
                text_font_style="bold",
            )

        # Configure y-axis
        fig.y_range = Range1d(start=0, end=1)
        fig.yaxis.visible = False  # Hide y-axis for reference track
        fig.ygrid.visible = False

        return fig

    def add_reference_labels_to_plot(
        self,
        fig,
        reference_sequence: str,
        positions: list,
        y_position: float = 1.05,
        query_sequence: str | None = None,
    ):
        """
        Add reference sequence labels to an existing plot

        Useful for adding reference annotation above signal plots.

        Args:
            fig: Existing Bokeh figure to add labels to
            reference_sequence: String of nucleotides
            positions: List of x-axis positions for each base
            y_position: Y-coordinate for labels (default: 1.05, above plot area)
            query_sequence: Optional query sequence for mismatch highlighting

        Raises:
            ValueError: If sequence/position length mismatch
        """
        if len(positions) != len(reference_sequence):
            raise ValueError(
                f"Position count ({len(positions)}) must match sequence length ({len(reference_sequence)})"
            )

        # Get base colors from theme
        base_colors = self.theme_manager.get_base_colors()

        # Prepare data
        text_data = {
            "x": [],
            "y": [],
            "text": [],
            "color": [],
            "is_mismatch": [],
        }

        for i, (pos, ref_base) in enumerate(
            zip(positions, reference_sequence, strict=True)
        ):
            ref_base_upper = ref_base.upper()
            base_color = base_colors.get(
                ref_base_upper, base_colors.get("N", "#808080")
            )

            # Check for mismatch
            is_mismatch = False
            if query_sequence and i < len(query_sequence):
                query_base = query_sequence[i].upper()
                is_mismatch = ref_base_upper != query_base and query_base != "N"

            text_data["x"].append(pos)
            text_data["y"].append(y_position)
            text_data["text"].append(ref_base_upper)
            text_data["color"].append(base_color)
            text_data["is_mismatch"].append(is_mismatch)

        # Split into match and mismatch
        match_data = {k: [] for k in ["x", "y", "text", "color"]}
        mismatch_data = {k: [] for k in ["x", "y", "text", "color"]}

        for i in range(len(text_data["x"])):
            target = mismatch_data if text_data["is_mismatch"][i] else match_data
            target["x"].append(text_data["x"][i])
            target["y"].append(text_data["y"][i])
            target["text"].append(text_data["text"][i])
            target["color"].append(text_data["color"][i])

        # Render matches
        if match_data["x"]:
            match_source = ColumnDataSource(data=match_data)
            fig.text(
                x="x",
                y="y",
                text="text",
                source=match_source,
                text_color="color",
                text_align="center",
                text_baseline="bottom",
                text_font_size="9pt",
            )

        # Render mismatches (bold)
        if mismatch_data["x"]:
            mismatch_source = ColumnDataSource(data=mismatch_data)
            fig.text(
                x="x",
                y="y",
                text="text",
                source=mismatch_source,
                text_color="color",
                text_align="center",
                text_baseline="bottom",
                text_font_size="10pt",
                text_font_style="bold",
            )
