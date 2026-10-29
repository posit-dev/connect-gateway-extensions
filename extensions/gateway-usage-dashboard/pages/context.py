"""Explicit shared services page servers may use without importing app.py."""

from dataclasses import dataclass
from typing import Callable

from shiny import render

from filters import Filters


@dataclass(frozen=True)
class PageContext:
    flt: Callable[[], Filters]
    cat: Callable[[], dict]
    register_export: Callable[..., None]
    register_sql_export: Callable[..., None]
    fitted_grid: Callable[..., render.DataGrid]
