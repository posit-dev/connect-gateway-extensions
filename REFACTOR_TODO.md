# Code refactor checklist

Goal: make the dashboard easier to change without changing its pages, input/output IDs, URLs, bookmarks, downloads, or data semantics. Take one checked, tested slice at a time; keep `app:app` as the entrypoint.

- [x] Move pricing SQL and catalog-sensitive caches from `app.py` to `pricing.py`; test rates and refresh behavior.
- [x] Move filter SQL construction (`where`, metadata paths, quoting) to `filters.py`; test representative filter combinations and escaping.
- [x] Extract conversation grouping/cost-driver calculations to `conversations.py`, with data-frame tests.
- [x] Extract common shell, drawer, and shared UI components to `ui_shell.py` without changing the DOM/input IDs.
- [x] Establish a small, explicit page-server dependency interface in `pages/context.py`.
- [x] Move Adoption UI and outputs together to `pages/adoption.py`; preserve its CSV export and IDs.
- [x] Move Models UI and outputs to `pages/models.py` (including caching, latency, and errors).
- [x] Move Raw Data UI and outputs to `pages/raw_data.py`, keeping its SQL builder available to Principals and its uncapped export.
- [x] Move Overview UI and outputs to `pages/overview.py` (KPIs, origin/tool, provider, and model spend); the unpriced-model checks stay in `app.py` because the topbar and Conversations share them. `chart_labels` is now on `PageContext`.
- [x] Move Attribution and Conversations into page modules one at a time; preserve cross-page selection and exports.
  Attribution done: `pages/attribution.py` owns its UI, outputs and `attribution.csv` export; `server(input, ctx, open_profile)` runs after Principals so an outlier pick still opens its profile. It imports `TOKEN_COUNT` from `filters.py` and `GUID_LABEL_CHARS` from `table_format.py`.
  Conversations done: `pages/conversations.py` (imported as `conversations_page`) owns its UI, cost-driver SQL, turn-by-turn detail and `conversations.csv` export; the unpriced-model checks stay in `app.py` for the topbar and reach the page as `PageContext.priced_only`, with `fitted_grid` also added.
- [x] Align the existing Principals module with the page-module interface once it is proven elsewhere.
  `principals.py` now has `panel()` (returns its own `nav_panel`) and `server(input, ctx, records_sql)`, taking Raw Data's SQL builder and returning `open_profile`; `HIGH_PCT`, `MIN_BASELINE` and `higher_cost` moved to `conversations.py`, so `principals.py` imports it at the top with no cycle.
- [x] Finish with app smoke checks, tests, and a short architecture note in README.
  `shiny run app:app` against the mock Connect serves `/` (200) and a websocket session renders all 68 outputs with no errors, the same ids as the pre-refactor app; `test_app_entrypoint_serves_page_and_session` covers this. app.py had no dead imports left; README has an Architecture section.

For each slice: check `git diff --check`, run `.venv/bin/python -m unittest discover -s tests -v`, and avoid changing UI or behavior as part of the move.
