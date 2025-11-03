"""
Tests for plot strategy pattern base classes and infrastructure
"""

import pytest
from bokeh.models.plots import Plot
from bokeh.plotting import figure

from squiggy.constants import Theme
from squiggy.plot_strategies.base import PlotStrategy


class ConcreteTestStrategy(PlotStrategy):
    """Concrete implementation of PlotStrategy for testing"""

    def create_plot(self, data, options):
        """Test implementation that creates a simple plot"""
        self.validate_data(data)

        # Create a simple figure
        fig = figure(width=800, height=400, title=data.get("title", "Test Plot"))
        fig.line([1, 2, 3], [1, 4, 9], color="blue")

        html = self._figure_to_html(fig)
        return html, fig

    def validate_data(self, data):
        """Test validation requiring 'signal' key"""
        if "signal" not in data:
            raise ValueError("Missing required data: signal")


class TestPlotStrategy:
    """Tests for PlotStrategy abstract base class"""

    def test_cannot_instantiate_abstract_class(self):
        """Test that PlotStrategy cannot be instantiated directly"""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            PlotStrategy(Theme.LIGHT)

    def test_concrete_strategy_instantiation(self):
        """Test that concrete strategy can be instantiated"""
        strategy = ConcreteTestStrategy(Theme.LIGHT)
        assert strategy.theme == Theme.LIGHT
        assert isinstance(strategy, PlotStrategy)

    def test_concrete_strategy_with_dark_theme(self):
        """Test concrete strategy with DARK theme"""
        strategy = ConcreteTestStrategy(Theme.DARK)
        assert strategy.theme == Theme.DARK

    def test_create_plot_returns_tuple(self):
        """Test that create_plot returns (html, figure) tuple"""
        strategy = ConcreteTestStrategy(Theme.LIGHT)
        data = {"signal": [1, 2, 3]}
        options = {}

        result = strategy.create_plot(data, options)

        assert isinstance(result, tuple)
        assert len(result) == 2

        html, fig = result
        assert isinstance(html, str)
        assert isinstance(fig, Plot)

    def test_create_plot_html_is_valid(self):
        """Test that generated HTML contains expected Bokeh elements"""
        strategy = ConcreteTestStrategy(Theme.LIGHT)
        data = {"signal": [1, 2, 3], "title": "Test Plot"}
        options = {}

        html, _ = strategy.create_plot(data, options)

        # Check for Bokeh HTML structure
        assert "<html" in html
        assert "<head>" in html
        assert "<body>" in html
        assert "BokehJS" in html or "bokeh" in html.lower()
        assert "Test Plot" in html

    def test_validate_data_raises_on_missing_required(self):
        """Test that validate_data raises ValueError for missing data"""
        strategy = ConcreteTestStrategy(Theme.LIGHT)
        data = {}  # Missing 'signal'

        with pytest.raises(ValueError, match="Missing required data: signal"):
            strategy.validate_data(data)

    def test_validate_data_passes_with_required(self):
        """Test that validate_data passes with required data"""
        strategy = ConcreteTestStrategy(Theme.LIGHT)
        data = {"signal": [1, 2, 3]}

        # Should not raise
        strategy.validate_data(data)

    def test_create_plot_calls_validate_data(self):
        """Test that create_plot validates data before plotting"""
        strategy = ConcreteTestStrategy(Theme.LIGHT)
        data = {}  # Missing 'signal'
        options = {}

        with pytest.raises(ValueError, match="Missing required data: signal"):
            strategy.create_plot(data, options)

    def test_figure_to_html_helper(self):
        """Test _figure_to_html helper method"""
        strategy = ConcreteTestStrategy(Theme.LIGHT)

        # Create a simple figure
        fig = figure(width=400, height=300, title="Helper Test")
        fig.circle([1, 2, 3], [1, 4, 9], radius=0.1)

        html = strategy._figure_to_html(fig)

        assert isinstance(html, str)
        assert "<html" in html
        assert "Helper Test" in html
        assert "bokeh" in html.lower()

    def test_multiple_strategies_independent(self):
        """Test that multiple strategy instances are independent"""
        strategy1 = ConcreteTestStrategy(Theme.LIGHT)
        strategy2 = ConcreteTestStrategy(Theme.DARK)

        assert strategy1.theme == Theme.LIGHT
        assert strategy2.theme == Theme.DARK
        assert strategy1.theme != strategy2.theme


class TestPlotStrategyInterface:
    """Tests for PlotStrategy interface contract"""

    def test_strategy_has_create_plot_method(self):
        """Test that strategies have create_plot method"""
        strategy = ConcreteTestStrategy(Theme.LIGHT)
        assert hasattr(strategy, "create_plot")
        assert callable(strategy.create_plot)

    def test_strategy_has_validate_data_method(self):
        """Test that strategies have validate_data method"""
        strategy = ConcreteTestStrategy(Theme.LIGHT)
        assert hasattr(strategy, "validate_data")
        assert callable(strategy.validate_data)

    def test_strategy_has_theme_attribute(self):
        """Test that strategies have theme attribute"""
        strategy = ConcreteTestStrategy(Theme.LIGHT)
        assert hasattr(strategy, "theme")
        assert strategy.theme in (Theme.LIGHT, Theme.DARK)

    def test_create_plot_signature(self):
        """Test that create_plot has correct signature"""
        import inspect

        strategy = ConcreteTestStrategy(Theme.LIGHT)
        sig = inspect.signature(strategy.create_plot)

        params = list(sig.parameters.keys())
        assert "data" in params
        assert "options" in params

    def test_validate_data_signature(self):
        """Test that validate_data has correct signature"""
        import inspect

        strategy = ConcreteTestStrategy(Theme.LIGHT)
        sig = inspect.signature(strategy.validate_data)

        params = list(sig.parameters.keys())
        assert "data" in params


class TestPlotStrategyDocumentation:
    """Tests for PlotStrategy documentation and examples"""

    def test_base_class_has_docstring(self):
        """Test that PlotStrategy has comprehensive docstring"""
        assert PlotStrategy.__doc__ is not None
        assert len(PlotStrategy.__doc__) > 100

    def test_create_plot_has_docstring(self):
        """Test that create_plot has detailed docstring"""
        assert PlotStrategy.create_plot.__doc__ is not None
        assert "Args:" in PlotStrategy.create_plot.__doc__
        assert "Returns:" in PlotStrategy.create_plot.__doc__

    def test_validate_data_has_docstring(self):
        """Test that validate_data has detailed docstring"""
        assert PlotStrategy.validate_data.__doc__ is not None
        assert "Args:" in PlotStrategy.validate_data.__doc__
        assert "Raises:" in PlotStrategy.validate_data.__doc__
