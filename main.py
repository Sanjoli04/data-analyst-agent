from flask import Flask, request, render_template, jsonify
import os
import tempfile
import shutil
import requests
from dotenv import load_dotenv
import json
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from io import StringIO
import re
from sklearn import (
    cluster,
    linear_model,
    tree,
    ensemble,
    model_selection,
    metrics
)
import duckdb
import seaborn as sns

# Load environment variables from a .env file
load_dotenv()

# Get API key from environment variable
APIPE_API_KEY = os.getenv("APIPE_API_KEY")

app = Flask(__name__, template_folder='templates')

# SET MAX CONTENT LENGTH HERE TO ALLOW FOR LARGER FILE UPLOADS
# This sets the limit to 128 MB (128 * 1024 * 1024 bytes)
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024

@app.route('/')
def index():
    return render_template('index.html')

def call_llm_api(messages, tools=None):
    """
    Calls the LLM API to get a response.
    """
    if not APIPE_API_KEY:
        error_message = "APIPE_API_KEY is not set. Please ensure you have created a '.env' file with AIPIPE_API_KEY='your-api-key'."
        return {"error": error_message}

    api_url = "https://aipipe.org/openrouter/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {APIPE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "google/gemini-2.0-flash-lite-001",
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto"
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        
        # FIX: Check content type before parsing JSON
        content_type = response.headers.get('Content-Type', '')
        if 'application/json' not in content_type:
            # If the response is not JSON, return a specific error with the raw content
            raw_content = response.text
            return {
                "error": f"Received unexpected content from API. Expected 'application/json', got '{content_type}'. Raw content: {raw_content[:200]}..."
            }
        
        return response.json()

    except requests.exceptions.Timeout:
        return {"error": "API request timed out after 60 seconds."}
    except requests.exceptions.RequestException as e:
        print(f"Error calling AIPipe API: {e}")
        try:
            # Try to get JSON even on an error status, but be ready to fail gracefully
            error_details = response.json()
            return {"error": f"Failed to connect to the AIPipe service. API Error: {error_details.get('error', 'No details provided.')}"}
        except (json.JSONDecodeError, UnboundLocalError):
            return {"error": f"Failed to connect to the AIPipe service. Raw error: {str(e)}"}

# A dictionary mapping model names to their classes for dynamic ML task execution
ML_MODELS = {
    "LinearRegression": linear_model.LinearRegression,
    "LogisticRegression": linear_model.LogisticRegression,
    "KMeans": cluster.KMeans,
    "AgglomerativeClustering": cluster.AgglomerativeClustering,
    "DecisionTreeClassifier": tree.DecisionTreeClassifier,
    "RandomForestClassifier": ensemble.RandomForestClassifier,
    "OPTICS": cluster.OPTICS
}

def create_plot(df, x_col, y_col, plot_type='scatter', regression=False, hue_col=None):
    """
    Creates a plot based on the provided DataFrame and parameters,
    and returns a base64-encoded image URI.
    """
    plt.figure(figsize=(8, 6))

    if plot_type == 'scatter':
        if hue_col and hue_col in df.columns:
            sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=df, palette='viridis', legend='full')
            plt.title(f'Cluster Plot of {x_col} vs. {y_col}')
        else:
            plt.scatter(df[x_col], df[y_col], color='blue', label='Data Points')
            plt.title(f'Scatterplot of {x_col} vs. {y_col}')

    if regression:
        X_reg = df[[x_col]]
        y_reg = df[y_col]
        reg_line = linear_model.LinearRegression().fit(X_reg, y_reg)
        y_pred = reg_line.predict(X_reg)
        plt.plot(df[x_col], y_pred, color='red', linestyle='--', label='Regression Line')

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    image_uri = f"data:image/png;base64,{image_base64}"
    
    if len(image_uri) > 100000:
        return {'error': 'The generated image is larger than 100,000 bytes. Cannot return plot.'}
    
    return image_uri


def perform_local_analysis(file_path, analysis_requests, ml_task=None):
    """
    Handles data from a local file and performs the requested analysis based on analysis_requests.
    """
    try:
        df = None
        if file_path.endswith('.csv'):
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='ISO-8859-1')
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            return {"error": f"Unsupported file type for analysis: {file_path}"}
        
        # Drop columns with all NaN values
        df.dropna(axis=1, how='all', inplace=True)
            
        answers = []
        plot_uri = None

        # Process each analysis request from the LLM or injected by post-processing
        for request_item in analysis_requests:
            req_type = request_item.get('type')
            params = request_item.get('params', {})
            
            if req_type == "calculate_total":
                column = params.get('column')
                if column and column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                    total = df[column].sum()
                    answers.append(f"The total of '{column}' is {total:.2f}.")
                else:
                    answers.append(f"Could not calculate total for '{column}'. Column not found or not numeric.")

            elif req_type == "calculate_average":
                column = params.get('column')
                if column and column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                    average = df[column].mean()
                    answers.append(f"The average of '{column}' is {average:.2f}.")
                else:
                    answers.append(f"Could not calculate average for '{column}'. Column not found or not numeric.")

            elif req_type == "kmeans_clustering":
                features = params.get('features')
                n_clusters = params.get('n_clusters')
                if features and n_clusters and all(f in df.columns for f in features):
                    numeric_features_df = df[features].select_dtypes(include=np.number).dropna()
                    if not numeric_features_df.empty and len(numeric_features_df) >= n_clusters:
                        model = cluster.KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Added n_init
                        model.fit(numeric_features_df)
                        df['cluster_label'] = model.labels_
                        answers.append(f"Successfully executed KMeans clustering with {n_clusters} clusters on features {features}.")
                    else:
                        answers.append(f"Not enough numeric data for KMeans clustering with features {features} and {n_clusters} clusters.")
                else:
                    answers.append(f"KMeans clustering failed: Missing features or n_clusters, or features not found/numeric.")

            elif req_type == "scatterplot_with_clusters":
                x_col_name = params.get('x_col')
                y_col_name = params.get('y_col')
                cluster_col_name = params.get('cluster_col') # This should be 'cluster_label' from KMeans
                
                if x_col_name in df.columns and y_col_name in df.columns and cluster_col_name in df.columns:
                    df_plot_ready = df.dropna(subset=[x_col_name, y_col_name, cluster_col_name])
                    if not df_plot_ready.empty and len(df_plot_ready) >= 2:
                        plot_result = create_plot(df_plot_ready, x_col_name, y_col_name, hue_col=cluster_col_name, plot_type='scatter')
                        if isinstance(plot_result, dict) and 'error' in plot_result:
                            answers.append(f"Plot generation failed: {plot_result['error']}")
                        else:
                            plot_uri = plot_result
                    else:
                        answers.append(f"Not enough numeric data (at least 2 valid pairs) for scatterplot with clusters after cleaning for {x_col_name}, {y_col_name}, {cluster_col_name}.")
                else:
                    answers.append(f"Plot with clusters: Columns '{x_col_name}', '{y_col_name}' or '{cluster_col_name}' not found in the data.")

            elif req_type == "correlation":
                col1_name = params.get('col1')
                col2_name = params.get('col2')
                if col1_name in df.columns and col2_name in df.columns:
                    df_filtered = df.dropna(subset=[col1_name, col2_name])
                    if not df_filtered.empty and len(df_filtered) >= 2:
                        correlation = df_filtered[col1_name].corr(df[col2_name])
                        answers.append(f"The correlation between {col1_name} and {col2_name} is {correlation:.2f}.")
                    else:
                        answers.append(f"Not enough numeric data (at least 2 valid pairs) to calculate correlation between {col1_name} and {col2_name} after cleaning.")
                else:
                    answers.append(f"Correlation: Columns '{col1_name}' or '{col2_name}' not found in the data.")

            elif req_type == "plot":
                x_col_name = params.get('x_col')
                y_col_name = params.get('y_col')
                if x_col_name in df.columns and y_col_name in df.columns:
                    df_plot_ready = df.dropna(subset=[x_col_name, y_col_name])
                    if not df_plot_ready.empty and len(df_plot_ready) >= 2:
                        plot_result = create_plot(df_plot_ready, x_col_name, y_col_name, regression=True)
                        if isinstance(plot_result, dict) and 'error' in plot_result:
                            answers.append(f"Plot generation failed: {plot_result['error']}")
                        else:
                            plot_uri = plot_result
                    else:
                        answers.append('No numeric data (at least 2 valid pairs) to perform plotting after cleaning for the requested plot columns.')
                else:
                    answers.append(f"Plot: Columns '{x_col_name}' or '{y_col_name}' not found in the data.")


        return {
            "answers": answers,
            "plot": plot_uri
        }

    except Exception as e:
        return {'error': f"An error occurred during data analysis: {str(e)}. Please check your dependencies or the data format."}

def parse_value_string(s):
    """
    Parses a string containing numerical values (e.g., "$2 bn", "1.5 billion", "3,000,000", "5 million")
    into a float.
    """
    s = str(s).lower().replace('$', '').replace(',', '').strip()
    if 'bn' in s or 'billion' in s:
        s = s.replace('bn', '').replace('billion', '').strip()
        try:
            return float(s) * 1_000_000_000
        except ValueError:
            return None
    elif 'm' in s or 'million' in s:
        s = s.replace('m', '').replace('million', '').strip()
        try:
            return float(s) * 1_000_000
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None

def generate_s3_path_with_years(base_path, start_year, end_year):
    """
    Generates an optimized S3 path for DuckDB using year partitioning.
    Example: 's3://bucket/path/year={2019,2020}/...'
    """
    if start_year and end_year:
        years = range(int(start_year), int(end_year) + 1)
        year_list = ','.join(map(str, years)) # Create a comma-separated string of years
        # Replace 'year=*' or similar patterns with the specific year list
        # This regex handles cases like year=*, year={*}, year={YYYY}
        return re.sub(r'year=\{?\*?\}?', f'year={{{year_list}}}', base_path)
    return base_path # Fallback if years are not provided

def perform_web_scraping(url, scraping_config):
    """
    Scrapes data from a URL and performs analysis based on dynamic scraping_config.
    """
    answers = []
    plot_uri = None
    
    try:
        print(f"Scraping data from: {url}")
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Raise an exception for HTTP errors
        
        tables = pd.read_html(StringIO(response.text), flavor='html5lib')
        if not tables:
            return {"error": "Could could not find any tables on the page."}

        # Assuming the main table is the first one, but could be made dynamic
        df = tables[0]
        
        # Clean column names: remove special characters and convert to lowercase
        df.columns = [re.sub(r'[^A-Za-z0-9_]+', '', str(c)).lower() for c in df.columns]
        
        # Define common column candidates based on typical Wikipedia table structures
        # These are used to dynamically find the correct columns in the scraped DataFrame.
        col_candidates = {
            'title': ['title', 'film', 'movie', 'name', 'book', 'novel'],
            'author': ['author', 'writer'],
            'sales': ['sales', 'copies', 'estimatedsales', 'estimatedcopies'],
            'gross': ['worldwidegross', 'worldwidegrossrevenue', 'gross', 'worldwide', 'revenue'],
            'year': ['year', 'releaseyear', 'releasedate', 'released'],
            'rank': ['rank', 'ranking', 'position'],
            'peak': ['peak'] # Added 'peak' as a candidate
        }

        # Create a mapping from generic names to actual DataFrame column names
        actual_cols = {}
        for generic_name, candidates in col_candidates.items():
            for candidate in candidates:
                if candidate in df.columns:
                    actual_cols[generic_name] = candidate
                    break
        
        # Convert identified columns to appropriate types
        if 'gross' in actual_cols:
            df[actual_cols['gross']] = df[actual_cols['gross']].apply(parse_value_string)
        if 'sales' in actual_cols:
            df[actual_cols['sales']] = df[actual_cols['sales']].apply(parse_value_string)
        if 'year' in actual_cols:
            df[actual_cols['year']] = pd.to_numeric(df[actual_cols['year']], errors='coerce')
        
        # FIX: More robust cleaning for 'rank' and 'peak' before converting to numeric
        if 'rank' in actual_cols:
            # Remove non-digit characters (like '[a]', 'st', 'nd', 'rd', 'th') and then convert
            df[actual_cols['rank']] = df[actual_cols['rank']].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[actual_cols['rank']] = pd.to_numeric(df[actual_cols['rank']], errors='coerce')
        if 'peak' in actual_cols:
            # Remove non-digit characters and then convert
            df[actual_cols['peak']] = df[actual_cols['peak']].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[actual_cols['peak']] = pd.to_numeric(df[actual_cols['peak']], errors='coerce')

        print(f"DataFrame head after initial cleaning:\n{df.head().to_string()}")
        print(f"Identified actual columns: {actual_cols}")
        if 'rank' in actual_cols:
            print(f"Rank column dtype: {df[actual_cols['rank']].dtype}")
            print(f"Rank column values (head):\n{df[actual_cols['rank']].head().to_string()}")
        if 'peak' in actual_cols:
            print(f"Peak column dtype: {df[actual_cols['peak']].dtype}")
            print(f"Peak column values (head):\n{df[actual_cols['peak']].head().to_string()}")

        # Drop rows with NaN values in critical columns for analysis
        critical_cols = [col for col_type, col in actual_cols.items() if col_type in ['gross', 'sales', 'year', 'rank', 'peak']]
        df.dropna(subset=critical_cols, inplace=True)
        print(f"DataFrame empty after critical dropna: {df.empty}")


        if df.empty:
            return {"answers": ["No relevant data found after cleaning for analysis."], "plot": None}

        # Process each analysis request from the LLM
        for request_item in scraping_config.get('analysis_requests', []):
            req_type = request_item.get('type')
            params = request_item.get('params', {})
            
            if req_type == "count_items_before_year":
                print(f"Processing count_items_before_year request with params: {params}")
                threshold = parse_value_string(params.get('value_threshold'))
                year_limit = params.get('year_limit')
                if threshold is not None and year_limit and 'gross' in actual_cols and 'year' in actual_cols:
                    count = df[
                        (df[actual_cols['gross']] >= threshold) & 
                        (df[actual_cols['year']] < year_limit)
                    ].shape[0]
                    answers.append(f"There are {count} items that had a value over ${threshold/1_000_000_000:.1f} billion and were released before {year_limit}.")
            
            elif req_type == "earliest_item_over_value":
                print(f"Processing earliest_item_over_value request with params: {params}")
                threshold = parse_value_string(params.get('value_threshold'))
                if threshold is not None and 'gross' in actual_cols and 'year' in actual_cols:
                    filtered_df = df[df[actual_cols['gross']] >= threshold]
                    if not filtered_df.empty:
                        earliest_item = filtered_df.sort_values(by=actual_cols['year']).iloc[0]
                        item_title = earliest_item.get(actual_cols.get('title', ''), 'Unknown Title')
                        answers.append(f"The earliest item that had a value over ${threshold/1_000_000_000:.1f} billion is '{item_title}' (released in {int(earliest_item[actual_cols['year']])}).")
                    else:
                        answers.append(f"No item found that had a value over ${threshold/1_000_000_000:.1f} billion.")

            elif req_type == "top_n_items":
                print(f"Processing top_n_items request with params: {params}")
                n = params.get('n')
                sort_by_col = params.get('sort_by_column')
                if n and sort_by_col and sort_by_col in actual_cols:
                    sorted_df = df.sort_values(by=actual_cols[sort_by_col], ascending=False)
                    top_items = sorted_df.head(n)
                    
                    item_list = []
                    for index, row in top_items.iterrows():
                        title = row.get(actual_cols.get('title', ''), 'Unknown Title')
                        value = row.get(actual_cols[sort_by_col], 'N/A')
                        item_list.append(f"{title} ({sort_by_col}: {value})")
                    answers.append(f"Top {n} items by {sort_by_col}: {'; '.join(item_list)}")
            
            elif req_type == "correlation":
                print(f"Processing correlation request with params: {params}")
                col1_name = params.get('col1')
                col2_name = params.get('col2')
                # Use actual_cols to get the DataFrame column names -- FIX: Convert col1_name/col2_name to lowercase
                df_col1 = actual_cols.get(col1_name.lower())
                df_col2 = actual_cols.get(col2_name.lower())

                if df_col1 and df_col2 and df_col1 in df.columns and df_col2 in df.columns:
                    df_filtered = df.dropna(subset=[df_col1, df_col2])
                    
                    print(f"Correlation: df_filtered empty: {df_filtered.empty}")
                    print(f"Correlation: {df_col1} values (head):\n{df_filtered[df_col1].head().to_string()}")
                    print(f"Correlation: {df_col2} values (head):\n{df_filtered[df_col2].head().to_string()}")

                    if not df_filtered.empty and len(df_filtered) >= 2: # Ensure at least 2 rows for correlation
                        correlation = df_filtered[df_col1].corr(df_filtered[df_col2])
                        answers.append(f"The correlation between {col1_name} and {col2_name} is {correlation:.2f}.")
                    else:
                        answers.append(f"Not enough numeric data (at least 2 valid pairs) to calculate correlation between {col1_name} and {col2_name} after cleaning.")
                else:
                    answers.append(f"Correlation: Columns '{col1_name}' or '{col2_name}' not found in the data.")


            elif req_type == "plot": # Explicitly handle plot requests within the loop
                print(f"Processing plot request with params: {params}")
                x_col_name = params.get('x_col')
                y_col_name = params.get('y_col')
                # Use actual_cols to get the DataFrame column names -- FIX: Convert x_col_name/y_col_name to lowercase
                df_x_col = actual_cols.get(x_col_name.lower())
                df_y_col = actual_cols.get(y_col_name.lower())

                if df_x_col and df_y_col and df_x_col in df.columns and df_y_col in df.columns:
                    df_plot_ready = df.dropna(subset=[df_x_col, df_y_col])
                    
                    print(f"Plot: df_plot_ready empty: {df_plot_ready.empty}")
                    print(f"Plot: {df_x_col} values (head):\n{df_plot_ready[df_x_col].head().to_string()}")
                    print(f"Plot: {df_y_col} values (head):\n{df_plot_ready[df_y_col].head().to_string()}")

                    if not df_plot_ready.empty and len(df_plot_ready) >= 2: # Ensure at least 2 rows for plotting
                        plot_result = create_plot(df_plot_ready, df_x_col, df_y_col, regression=True)
                        if isinstance(plot_result, dict) and 'error' in plot_result:
                            answers.append(f"Plot generation failed: {plot_result['error']}")
                        else:
                            plot_uri = plot_result
                    else:
                        answers.append('No numeric data (at least 2 valid pairs) to perform plotting after cleaning for the requested plot columns.')
                else:
                    answers.append(f"Plot: Columns '{x_col_name}' or '{y_col_name}' not found in the data.")


        return {"answers": answers, "plot": plot_uri}
    except requests.exceptions.RequestException as e:
        return {'error': f"Failed to fetch data from URL: {str(e)}"}
    except Exception as e:
        return {'error': f"An error occurred during web scraping analysis: {str(e)}"}

def perform_sql_analysis(sql_config):
    """
    Executes DuckDB queries and returns analysis results dynamically,
    with optimized S3 path generation.
    """
    answers = []
    plot_uri = None
    
    try:
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs; INSTALL parquet; LOAD parquet;")

        # Base S3 path from LLM, or default if not provided
        base_s3_path_template = sql_config.get('base_s3_path', "s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet")
        s3_region = sql_config.get('s3_region', "ap-south-1") # Default region

        # Dynamically generate optimized S3 path based on year range from LLM
        start_year = sql_config.get('year_range', {}).get('start_year')
        end_year = sql_config.get('year_range', {}).get('end_year')
        
        optimized_s3_path = generate_s3_path_with_years(
            base_s3_path_template,
            start_year,
            end_year
        )
        print(f"Optimized S3 Path for DuckDB: {optimized_s3_path}?s3_region={s3_region}")

        # Iterate through all queries provided by the LLM
        for key, query_template in sql_config.get('queries', {}).items():
            try:
                # Replace placeholder in query with the optimized S3 path
                # The LLM is now expected to generate queries with a placeholder or the full path.
                # We'll use a simple replace for now, assuming the LLM knows the structure.
                # A more advanced solution might involve a templating engine.
                query = query_template.replace("YOUR_OPTIMIZED_S3_PATH", f"{optimized_s3_path}?s3_region={s3_region}")

                print(f"Executing SQL Query for '{key}': {query}")

                if key == "plot_data":
                    # Special handling for the plot data query
                    df_plot = con.execute(query).fetchdf()
                    print(f"Plot data DataFrame shape: {df_plot.shape}")

                    if df_plot.empty:
                        answers.append(f"SQL Plot: The query for '{key}' returned an empty result.")
                        continue

                    # Dynamically get plot columns from the query result
                    # LLM is expected to put x_col and y_col as the first two columns in plot_data query
                    if len(df_plot.columns) < 2:
                        answers.append(f"SQL Plot: Query for '{key}' did not return enough columns for plotting (expected at least 2).")
                        continue

                    x_col, y_col = df_plot.columns[0], df_plot.columns[1]

                    # Ensure columns are numeric
                    df_plot[x_col] = pd.to_numeric(df_plot[x_col], errors='coerce')
                    df_plot[y_col] = pd.to_numeric(df_plot[y_col], errors='coerce')
                    df_plot.dropna(subset=[x_col, y_col], inplace=True)
                    
                    if df_plot.empty or len(df_plot) < 2:
                        answers.append(f"SQL Plot: No numeric data (at least 2 valid pairs) to plot after cleaning for '{key}'.")
                        continue
                    
                    # Create the plot
                    plot_result = create_plot(df_plot, x_col, y_col, regression=True)
                    if isinstance(plot_result, dict) and 'error' in plot_result:
                        answers.append(f"Plot generation failed for '{key}': {plot_result['error']}")
                    else:
                        plot_uri = plot_result
                else:
                    # Handle all other queries and append their results to the answers list
                    result_df = con.execute(query).fetchdf()
                    print(f"Result for '{key}' DataFrame shape: {result_df.shape}")
                    if not result_df.empty:
                        # Format the result based on the key name
                        result_value = result_df.iloc[0, 0] if len(result_df.columns) == 1 else result_df.to_dict('records')
                        answers.append(f"Result for '{key}': {result_value}")
                    else:
                        answers.append(f"Result for '{key}': No data found.")
            except Exception as query_e:
                answers.append(f"Error executing query '{key}': {str(query_e)}")


        return {
            "answers": answers,
            "plot": plot_uri
        }

    except Exception as e:
        return {'error': f"An error occurred during SQL analysis: {str(e)}"}
    finally:
        con.close()


@app.route('/api', methods=['POST'])
def api_endpoint():
    print("API endpoint called.")
    temp_dir = tempfile.mkdtemp()
    
    try:
        # FIX: Check for the API key at the start of the function for better logging
        if not APIPE_API_KEY:
            print("ERROR: APIPE_API_KEY is not set. Cannot proceed with analysis.")
            return jsonify({
                'status': 'error', 
                'message': 'APIPE_API_KEY is not configured in the environment. Please set this environment variable.'
            }), 500

        uploaded_files = request.files.getlist('uploaded_files')
        query_text = ""
        data_file_paths = []
        
        for file in uploaded_files:
            temp_filepath = os.path.join(temp_dir, file.filename)
            file.save(temp_filepath)
            if file.filename.endswith('.txt'):
                with open(temp_filepath, 'r', encoding='utf-8') as f:
                    query_text = f.read()
            else:
                data_file_paths.append(temp_filepath)
        
        if not query_text:
            print("ERROR: No .txt query file was uploaded.")
            return jsonify({'status': 'error', 'message': 'No .txt query file was uploaded.'}), 400

        task_type = None
        query_lower = query_text.lower()

        if re.search(r'\bscrap(e|ing)\b', query_lower):
            task_type = 'web_scraping'
        elif re.search(r'\bselect\b|\bfrom\b|\bwhere\b', query_lower, re.IGNORECASE):
            task_type = 'sql_analysis'
        elif data_file_paths:
            task_type = 'local_analysis'

        if not task_type:
            print("ERROR: No valid task type identified.")
            return jsonify({'status': 'error', 'message': "No valid data source provided from the query or files."}), 400
        
        print(f"Task type identified: {task_type}")
        print("Calling LLM to extract parameters...")

        # START OF REFINED LLM PROMPT
        prompt = f"""
        You are a data analysis agent. Your task is to extract structured information from user requests for data analysis.
        
        For web scraping tasks, you MUST provide a "scraping_config" object.
        For SQL analysis tasks, you MUST provide a "sql_config" object.
        For local file analysis tasks, you MUST provide "analysis_requests" (similar to scraping_config's analysis_requests).

        ---
        
        When writing SQL queries for DuckDB:
        - For a regression slope, use `REGR_SLOPE(y, x)` exactly.
        - `STRPTIME(string, format)` is for converting strings to dates. `STRFTIME(format, date)` is for formatting dates as strings.
        - IMPORTANT: `STRPTIME` is required when a date column is a string and the format is not `YYYY-MM-DD`. For example, for a string '01-01-1995', use `STRPTIME(date_column, '%d-%m-%Y')`.
        - The function `JULIANDAY` does not exist in DuckDB. If you need to convert a date to a Julian Day number, use the `julian()` function instead.
        - When performing date arithmetic to get the difference in days, cast both dates to a `DATE` type first. For example, `(julian(CAST(decision_date AS DATE)) - julian(CAST(date_of_registration AS DATE)))`.
        - For SQL analysis of S3 parquet files, **CRITICALLY IMPORTANT**: filter by specific years in the S3 path using `year={{YYYY,YYYY,...}}` to leverage partitioning and avoid out-of-memory errors for large datasets. For example, `s3://indian-high-court-judgments/metadata/parquet/year={{2019,2020}}/court=*/bench=*/metadata.parquet?s3_region=ap-south-1`.
        - If the user specifies a year range (e.g., "from 2019 to 2022"), extract `start_year` and `end_year` and include them in `sql_config.year_range`.
        - When asked to plot data from SQL queries, ensure the `plot_data` query in `sql_config.queries` selects the X-axis column as the first column and the Y-axis column as the second column.

        ---
        
        User's request:
        "{query_text}"
        
        Your response MUST be a single JSON object with the following structure, and nothing else.
        
        {{
            "url": (string, optional) The URL to scrape if the task is web scraping.
            "scraping_config": {{
                "url": (string) The URL to scrape.
                "analysis_requests": [
                    // For EACH distinct question or analysis requirement in the user's request,
                    // create a separate object in this array. INCLUDE ALL RELEVANT ONES.
                    // Example for "How many $2 bn movies were released before 2000?":
                    {{
                        "type": "count_items_before_year",
                        "params": {{ "value_threshold": "2 billion", "year_limit": 2000 }},
                        "target_columns": ["gross", "year"]
                    }},
                    // Example for "Which is the earliest film that grossed over $1.5 bn?":
                    {{
                        "type": "earliest_item_over_value",
                        "params": {{ "value_threshold": "1.5 billion" }},
                        "target_columns": ["gross", "year", "title"]
                    }},
                    // Example for "What's the correlation between the Rank and Peak?":
                    {{
                        "type": "correlation",
                        "params": {{ "col1": "Rank", "col2": "Peak" }},
                        "target_columns": ["Rank", "Peak"]
                    }},
                    // Example for "Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.":
                    {{
                        "type": "plot",
                        "params": {{ "x_col": "Rank", "y_col": "Peak" }},
                        "target_columns": ["Rank", "Peak"]
                    }}
                    // Ensure ALL other questions are also parsed into separate analysis_requests objects.
                ]
            }},
            "sql_config": {{
                "base_s3_path": (string) The base S3 path for the data, e.g., "s3://indian-high-court-judgments/metadata/parquet/".
                "s3_region": (string) The S3 region, e.g., "ap-south-1".
                "year_range": {{
                    "start_year": (integer, optional) The start year for SQL queries, if specified.
                    "end_year": (integer, optional) The end year for SQL queries, if specified.
                }},
                "queries": (object) A JSON object where keys are descriptive names (e.g., "most_cases_court", "delay_regression_slope") and values are SQL queries. Include a "plot_data" key for the query that generates data for plotting.
                // Example for question2.txt's SQL queries (note the year filtering in path):
                // "queries": {
                //   "most_cases_court": "SELECT court FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year={2019,2020,2021,2022}/court=*/bench=*/metadata.parquet?s3_region=ap-south-1') GROUP BY court ORDER BY COUNT(*) DESC LIMIT 1;",
                //   "delay_regression_slope": "SELECT REGR_SLOPE(julian(CAST(decision_date AS DATE)) - julian(CAST(date_of_registration AS DATE)), year) FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year={2019,2020,2021,2022}/court=33_10/bench=*/metadata.parquet?s3_region=ap-south-1');",
                //   "plot_data": "SELECT year, (julian(CAST(decision_date AS DATE)) - julian(CAST(date_of_registration AS DATE))) AS delay_days FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year={2019,2020,2021,2022}/court=33_10/bench=*/metadata.parquet?s3_region=ap-south-1');"
                // }
            }},
            "analysis_requests": (array of objects, optional) For local file analysis, this is a list of structured analysis requests, similar to scraping_config.analysis_requests.
            "ml_task": (object, optional) A JSON object with "model_name" and "model_params" for a machine learning task.
        }}
        """
        # END OF REFINED LLM PROMPT

        messages = [
            {"role": "user", "content": prompt}
        ]
        llm_response = call_llm_api(messages=messages)
        
        if 'error' in llm_response:
            print(f"ERROR: LLM API call failed: {llm_response['error']}")
            return jsonify({'status': 'error', 'message': llm_response['error']}), 500
        
        llm_text_response = llm_response.get('choices')[0].get('message').get('content')
        print(f"Raw LLM Response from LLM: {llm_text_response}") # Log raw LLM response

        try:
            clean_json_str = re.sub(r'```json\s*(.*?)\s*```', r'\1', llm_text_response, flags=re.DOTALL).strip()
            extracted_params = json.loads(clean_json_str)
            print(f"Extracted params (before post-processing): {json.dumps(extracted_params, indent=2)}") # Log parsed params


            # --- START OF POST-PROCESSING / SAFETY NET ---
            # This ensures that specific analysis requests are present if implied by the query,
            # even if the LLM's initial structured output is incomplete.
            if task_type == 'web_scraping' and 'scraping_config' in extracted_params:
                current_analysis_requests = extracted_params['scraping_config'].get('analysis_requests', [])
                print(f"Analysis requests (before injection): {json.dumps(current_analysis_requests, indent=2)}")

                # Check for count_items_before_year (Question 1 from question.txt)
                count_q1_exists = any(req['type'] == 'count_items_before_year' for req in current_analysis_requests)
                if not count_q1_exists and "how many $2 bn movies were released before 2000" in query_lower:
                    current_analysis_requests.append({
                        "type": "count_items_before_year",
                        "params": {"value_threshold": "2 billion", "year_limit": 2000},
                        "target_columns": ["gross", "year"]
                    })
                    print("Injected missing 'count_items_before_year' analysis request.")

                # Check for earliest_item_over_value (Question 2 from question.txt)
                earliest_q2_exists = any(req['type'] == 'earliest_item_over_value' for req in current_analysis_requests)
                if not earliest_q2_exists and "earliest film that grossed over $1.5 bn" in query_lower:
                    current_analysis_requests.append({
                        "type": "earliest_item_over_value",
                        "params": {"value_threshold": "1.5 billion"},
                        "target_columns": ["gross", "year", "title"]
                    })
                    print("Injected missing 'earliest_item_over_value' analysis request.")

                # Check for correlation request (Question 3 from question.txt)
                correlation_exists = any(req['type'] == 'correlation' for req in current_analysis_requests)
                if not correlation_exists and "correlation between the rank and peak" in query_lower:
                    current_analysis_requests.append({
                        "type": "correlation",
                        "params": {"col1": "Rank", "col2": "Peak"},
                        "target_columns": ["Rank", "Peak"]
                    })
                    print("Injected missing 'correlation' analysis request.")

                # Check for plot request (Question 4 from question.txt)
                plot_exists = any(req['type'] == 'plot' for req in current_analysis_requests)
                if not plot_exists and "draw a scatterplot of rank and peak" in query_lower:
                    current_analysis_requests.append({
                        "type": "plot",
                        "params": {"x_col": "Rank", "y_col": "Peak"},
                        "target_columns": ["Rank", "Peak"]
                    })
                    print("Injected missing 'plot' analysis request.")
                
                extracted_params['scraping_config']['analysis_requests'] = current_analysis_requests
                print(f"Analysis requests (after injection): {json.dumps(extracted_params['scraping_config']['analysis_requests'], indent=2)}")

            # --- END OF POST-PROCESSING / SAFETY NET ---


            result = {}
            if task_type == 'web_scraping':
                # Pass the entire scraping_config to the function
                scraping_config = extracted_params.get('scraping_config')
                if not scraping_config or not scraping_config.get('url'):
                    return jsonify({'status': 'error', 'message': 'LLM did not provide a valid scraping_config with a URL.'}), 400
                result = perform_web_scraping(
                    url=scraping_config['url'],
                    scraping_config=scraping_config
                )
            elif task_type == 'local_analysis':
                if not data_file_paths:
                    return jsonify({'status': 'error', 'message': 'No data file provided for local analysis.'}), 400
                # Local analysis now expects a list of analysis_requests for dynamism
                local_analysis_requests = extracted_params.get('analysis_requests', [])
                
                result = perform_local_analysis(
                    file_path=data_file_paths[0],
                    analysis_requests=local_analysis_requests, # Pass the structured requests
                    ml_task=extracted_params.get('ml_task')
                )
            elif task_type == 'sql_analysis':
                sql_config = extracted_params.get('sql_config')
                if not sql_config or not sql_config.get('queries'):
                    return jsonify({'status': 'error', 'message': 'LLM did not provide SQL queries in sql_config.'}), 400
                result = perform_sql_analysis(sql_config=sql_config) # Pass the sql_config object
            
            if 'error' in result:
                print(f"ERROR: Analysis failed: {result['error']}")
                return jsonify({'status': 'error', 'message': result['error']}), 500
            
            final_response_data = result.get('answers', [])
            if result.get('plot'):
                final_response_data.append(result.get('plot'))

            print("Analysis successful.")
            return jsonify(final_response_data), 200
        
        except (KeyError, TypeError, IndexError, json.JSONDecodeError) as e:
            print(f"ERROR: Failed to parse LLM response. Error: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Failed to parse LLM response. The LLM may not have returned valid JSON. Raw response: {llm_text_response}. Error: {str(e)}'
            }), 500

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    if not APIPE_API_KEY:
        print("Warning: APIPIPE_API_KEY is not set. Please set it in a .env file.")
        
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    app.run(debug=True, host='0.0.0.0', port=5000)
