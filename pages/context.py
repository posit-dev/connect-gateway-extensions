"""Explicit shared services page servers may use without importing app.py."""

from dataclasses import dataclass
from typing import Callable

import pandas as pd
import plotly.graph_objects as go
from shiny import render


@dataclass(frozen=True)
class PageContext:
    flt: Callable[[], dict]
    where: Callable[..., str]
    cat: Callable[[], dict]
    est_cost: Callable[[], str]
    register_export: Callable[..., None]
    register_sql_export: Callable[..., None]
    est_cost_uncached: Callable[[], str]
    total_bars: Callable[..., go.Figure]
    measure_of: Callable[[dict], dict]
    formatted: Callable[[pd.DataFrame], pd.DataFrame]
    chart_labels: Callable[..., list]
    fitted_grid: Callable[..., render.DataGrid]
    priced_only: Callable[[], str]  # WHERE fragment leaving out unpriced models
