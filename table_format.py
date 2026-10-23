"""Labels shared by tables and charts: table column names and how much of a
guid a chart label keeps."""

import pandas as pd

# How many leading characters of a content guid a chart axis shows.
GUID_LABEL_CHARS = 8


def pretty_columns(df: pd.DataFrame) -> pd.DataFrame:
    nice = {"est_spend_usd": "estimated spend (usd)"}
    return df.rename(lambda x: nice.get(x, x.replace("_", " ")), axis="columns")
