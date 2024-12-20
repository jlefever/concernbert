import pandas as pd
import numpy as np


# Written by ChatGPT
def dataframe_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    col_format: str,
    wide_table: bool = False,
    decimal_places: int | None = None,
) -> str:
    """
    Converts a pandas DataFrame into a LaTeX table format following ACM guidelines.

    Parameters:
    - df: pandas DataFrame containing the table data.
    - caption: The caption to be used for the table.
    - label: The LaTeX label to reference the table.
    - col_format: A string representing the column formatting (e.g., 'lrrrr' for left-aligned and right-aligned columns).
    - wide_table: If True, uses the `table*` environment for wide tables, otherwise uses `table`.
    - decimal_places: Optional; if provided, truncates numerical values to the specified number of decimal places.

    Returns:
    - A string containing the LaTeX table.
    """
    # Determine the LaTeX table environment based on wide_table flag
    table_env = "table*" if wide_table else "table"

    # If decimal_places is provided, round numeric columns
    if decimal_places is not None:
        df = df.applymap(
            lambda x: round(x, decimal_places) if isinstance(x, (int, float)) else x
        ) # type: ignore

    # Start creating the LaTeX string
    latex_str = f"\\begin{{{table_env}}}\n"
    latex_str += f"  \\caption{{{caption}}}\n"
    latex_str += f"  \\label{{{label}}}\n"
    latex_str += f"  \\begin{{tabular}}{{{col_format}}}\n"
    latex_str += "    \\toprule\n"

    # Add the header row
    latex_str += "    " + " & ".join(df.columns) + " \\\\\n"
    latex_str += "    \\midrule\n"

    # Add the data rows
    for _, row in df.iterrows():
        latex_str += "    " + " & ".join(map(str, row)) + " \\\\\n"

    # Add the bottom rule and end the tabular environment
    latex_str += "    \\bottomrule\n"
    latex_str += "  \\end{tabular}\n"
    latex_str += f"\\end{{{table_env}}}\n"

    return latex_str
