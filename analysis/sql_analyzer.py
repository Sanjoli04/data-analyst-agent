# analysis/sql_analyzer.py
import duckdb
import re
import pandas as pd
from utils.plotting import create_plot # We will create this file next

def translate_sql_error(error_message):
    """Translates common SQL errors into user-friendly messages."""
    error_str = str(error_message).lower()
    if "binder error" in error_str and "strptime(date, string_literal)" in error_str:
        match = re.search(r"strptime\((.*?),\s*'.*?'\)", error_str)
        column = match.group(1) if match else "a date column"
        return (f"The SQL query failed because it tried to convert the column '{column}' to a date, but that column is already in a date format. "
                "The agent will attempt to fix this by removing the unnecessary conversion.")
    if "binder error" in error_str:
        return f"The SQL query failed because a column or function name is incorrect or does not exist. Please review the query."
    if "out of memory" in error_str:
        return "The SQL query failed because it required too much memory. This can happen with very large datasets. Try filtering the data further (e.g., by year or category)."
    return f"An SQL error occurred: {error_message}"

def run_sql_analysis(sql_config):
    """
    Executes DuckDB queries, now using human-readable explanations for results.
    """
    queries = sql_config.get('queries', {})
    explanations = sql_config.get('query_explanations', {})
    if not queries:
        return {'error': 'No queries were provided for SQL analysis.'}
        
    results = []
    plot_uri = None
    
    try:
        con = duckdb.connect(database=':memory:', read_only=False)
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("INSTALL parquet; LOAD parquet;")

        for key, query in queries.items():
            try:
                result_df = con.execute(query).fetchdf()
            except Exception as query_e:
                error_str = str(query_e).lower()
                if "binder error" in error_str and "strptime(date, string_literal)" in error_str:
                    print(f"INFO: Detected strptime error on a DATE column in query '{key}'. Rewriting query.")
                    fixed_query = re.sub(r"STRPTIME\(([^,]+?),\s*'.*?'\)", r"\1", query, flags=re.IGNORECASE)
                    print(f"DEBUG: Rewritten query: {fixed_query}")
                    try:
                        result_df = con.execute(fixed_query).fetchdf()
                    except Exception as retry_e:
                        results.append(f"{explanations.get(key, key)}: Failed. {translate_sql_error(retry_e)}")
                        continue
                else:
                    results.append(f"{explanations.get(key, key)}: Failed. {translate_sql_error(query_e)}")
                    continue

            # Process the successful result
            answer_prefix = explanations.get(key, f"Result for '{key}'")
            if key == "plot_data":
                if not result_df.empty and len(result_df.columns) >= 2:
                    x_col, y_col = result_df.columns[0], result_df.columns[1]
                    plot_result = create_plot(result_df, x_col, y_col, regression=True)
                    if isinstance(plot_result, str):
                        plot_uri = plot_result
                    else:
                        results.append(f"Plot generation failed: {plot_result.get('error')}")
                else:
                    results.append("The query for the plot returned no data.")
            else:
                if not result_df.empty:
                    if result_df.shape[0] == 1 and result_df.shape[1] == 1:
                         results.append(f"{answer_prefix}: {result_df.iloc[0,0]}")
                    else:
                         results.append(f"{answer_prefix}:\n{result_df.to_string()}")
                else:
                    results.append(f"{answer_prefix}: The query returned no results.")

    except Exception as e:
        return {'error': f"SQL analysis setup failed: {e}"}
    finally:
        if 'con' in locals():
            con.close()

    final_response = results
    if plot_uri:
        final_response.append(plot_uri)
        
    return {"answers": final_response}
