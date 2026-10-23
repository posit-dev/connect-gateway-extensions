"""Explicit shared services page servers may use without importing app.py."""

from dataclasses import dataclass
from typing import Callable

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
    fitted_grid: Callable[..., render.DataGrid]
