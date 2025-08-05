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
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "API request timed out after 30 seconds."}
    except requests.exceptions.RequestException as e:
        print(f"Error calling AIPipe API: {e}")
        try:
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


def perform_local_analysis(file_path, questions, plot_columns, ml_task=None):
    """
    Handles data from a local file and performs the requested analysis.
    This function is now fully dynamic.
    """
    try:
        df = None
        if file_path.endswith('.csv'):
            try:
                # FIX: Try multiple encodings for CSV files
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

        if plot_columns and len(plot_columns) == 2:
            col1, col2 = plot_columns[0], plot_columns[1]

            if col1 not in df.columns or col2 not in df.columns:
                return {'error': f"Plot columns '{col1}' or '{col2}' not found in the data."}

            # General correlation and regression logic
            if any('correlation' in q.lower() for q in questions):
                correlation = df[col1].corr(df[col2])
                answers.append(f"The correlation between {col1} and {col2} is {correlation:.2f}.")
            
            if any('forecast' in q.lower() for q in questions):
                # Ensure columns are numeric for regression
                if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                    X = df[[col1]].dropna()
                    y = df[col2].dropna()
                    reg = linear_model.LinearRegression().fit(X, y)
                    forecasted_value = reg.predict([[df[col1].max() + 1]])[0]
                    answers.append(f"Based on a linear regression, the forecasted {col2} for the next value of {col1} is {forecasted_value:.2f}.")
                else:
                    answers.append(f"Cannot perform regression on non-numeric columns: {col1} and {col2}.")

        # Dynamic ML Model execution
        if ml_task and ml_task.get('model_name') in ML_MODELS:
            model_name = ml_task.get('model_name')
            model_params = ml_task.get('model_params', {})
            model_class = ML_MODELS[model_name]
            
            try:
                # Drop non-numeric columns for ML tasks and fill NaNs
                numeric_df = df.select_dtypes(include=np.number).dropna()
                
                if model_name in ['DecisionTreeClassifier', 'RandomForestClassifier', 'LogisticRegression']:
                    # Assuming a target column for classification - using the last numeric column
                    if len(numeric_df.columns) > 1:
                        target_col = numeric_df.columns[-1]
                        X = numeric_df.drop(columns=[target_col], errors='ignore')
                        y = numeric_df[target_col]
                        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
                        model = model_class(**model_params)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        accuracy = metrics.accuracy_score(y_test, y_pred)
                        answers.append(f"Successfully executed {model_name}. Accuracy: {accuracy:.2f}.")
                    else:
                        answers.append(f"Classification requires at least two numeric columns (features and target).")
                
                elif model_name in ["KMeans", "AgglomerativeClustering", "OPTICS"]:
                    if len(numeric_df.columns) >= 2:
                        X = numeric_df
                        model = model_class(**model_params)
                        model.fit(X)
                        df['cluster_label'] = model.labels_
                        answers.append(f"Successfully executed {model_name} with parameters: {model_params}. Found {df['cluster_label'].nunique()} clusters.")
                        
                        if plot_columns and len(plot_columns) == 2:
                            plot_uri = create_plot(df, plot_columns[0], plot_columns[1], hue_col='cluster_label')
                    else:
                        answers.append(f"Clustering requires at least two numeric columns.")
            except Exception as e:
                answers.append(f"ML model '{model_name}' failed to run: {str(e)}")
        
        # Generate final plot if plot columns were provided and no clustering plot was made
        if plot_uri is None and plot_columns and len(plot_columns) == 2:
            plot_uri = create_plot(df, plot_columns[0], plot_columns[1], regression=any('forecast' in q.lower() for q in questions))

        return {
            "answers": answers,
            "plot": plot_uri
        }

    except Exception as e:
        return {'error': f"An error occurred during data analysis: {str(e)}. Please check your dependencies or the data format."}

def perform_web_scraping(url, questions, plot_columns):
    """
    Scrapes data from a URL and performs a basic analysis.
    This function is made more generic to handle various tables.
    """
    try:
        print(f"Scraping data from: {url}")
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        
        tables = pd.read_html(StringIO(response.text), flavor='html5lib')
        if not tables:
            return {"error": "Could not find any tables on the page."}

        df = tables[0]
        
        # Clean column names and convert to lowercase for consistent lookup
        df.columns = [re.sub(r'[^A-Za-z0-9_]+', '', str(c)).lower() for c in df.columns]
        
        # Log the cleaned column names as requested by the user
        print(f"Available columns after cleaning and lowercasing: {df.columns.tolist()}")
        
        answers = []
        plot_uri = None

        if plot_columns and len(plot_columns) == 2:
            # Convert requested plot columns to lowercase to match the DataFrame
            col1, col2 = plot_columns[0].lower(), plot_columns[1].lower()

            if col1 not in df.columns or col2 not in df.columns:
                return {'error': f"Plot columns '{col1}' or '{col2}' not found in the scraped data. Available columns: {df.columns.tolist()}"}
            
            # Ensure columns are numeric for plotting and correlation
            df[col1] = pd.to_numeric(df[col1], errors='coerce')
            df[col2] = pd.to_numeric(df[col2], errors='coerce')
            df.dropna(subset=[col1, col2], inplace=True)

            if not df.empty:
                correlation = df[col1].corr(df[col2])
                answers.append(f"The correlation between {col1} and {col2} is {correlation:.2f}.")

                plot_uri = create_plot(df, col1, col2, regression=True)
            else:
                return {'error': 'No numeric data to perform analysis or plotting after cleaning.'}
        else:
            # Fallback analysis if plot columns are not specified
            numeric_cols = df.select_dtypes(include=np.number).columns
            if len(numeric_cols) >= 2:
                answers.append(f"Found {len(numeric_cols)} numeric columns. The average of '{numeric_cols[0]}' is {df[numeric_cols[0]].mean():.2f}.")
            
            
        return {"answers": answers, "plot": plot_uri}
    except Exception as e:
        return {'error': f"An error occurred during web scraping: {str(e)}"}

def perform_sql_analysis(queries):
    """
    Executes DuckDB queries and returns analysis results.
    """
    try:
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs; INSTALL parquet; LOAD parquet;")

        # Run query 1
        q1 = queries.get("q1")
        if not q1:
             return {'error': 'Query "q1" is missing from the LLM response.'}
        
        court_most_cases_df = con.execute(q1).fetchdf()
        court_most_cases = court_most_cases_df.iloc[0, 0] if not court_most_cases_df.empty else "N/A"
        
        # Run query 2 for regression slope
        q2 = queries.get("q2")
        if not q2:
             return {'error': 'Query "q2" is missing from the LLM response.'}

        slope_df = con.execute(q2).fetchdf()
        slope = slope_df.iloc[0, 0] if not slope_df.empty else np.nan
        
        # Run query for plot data
        plot_data_query = queries.get("plot_data")
        if not plot_data_query:
            return {'error': 'Plot data query is missing from the LLM response.'}

        df_plot = con.execute(plot_data_query).fetchdf()

        if df_plot.empty:
            return {'error': 'The plot data query returned an empty result.'}

        # Dynamically get plot columns from the query result
        x_col, y_col = df_plot.columns[0], df_plot.columns[1]

        # Ensure columns are numeric
        df_plot[x_col] = pd.to_numeric(df_plot[x_col], errors='coerce')
        df_plot[y_col] = pd.to_numeric(df_plot[y_col], errors='coerce')
        df_plot.dropna(subset=[x_col, y_col], inplace=True)
        
        if df_plot.empty:
             return {'error': 'No numeric data to plot after cleaning.'}

        # Create the plot
        plot_uri = create_plot(df_plot, x_col, y_col, regression=True)

        answers = [
            f"The high court with the most cases from 2019-2022 is {court_most_cases}.",
            f"The regression slope of delay by year is {slope:.2f}."
        ]
        
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
    temp_dir = tempfile.mkdtemp()
    
    try:
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
            return jsonify({'status': 'error', 'message': 'No .txt query file was uploaded.'}), 400

        task_type = None

        # Lowercased query for keyword checks
        query_lower = query_text.lower()

        # 👇 Priority: scraping > sql > ml > local
        if re.search(r'\bscrap(e|ing)\b', query_lower):
            task_type = 'web_scraping'
        elif re.search(r'\bselect\b|\bfrom\b|\bwhere\b', query_lower, re.IGNORECASE):
            task_type = 'sql_analysis'
        elif data_file_paths:
            # If a data file is present, assume local analysis
            task_type = 'local_analysis'

        if not task_type:
            return jsonify({'status': 'error', 'message': "No valid data source provided from the query or files."}), 400

        prompt = f"""
        When writing SQL queries for DuckDB:
        - For a regression slope, use `REGR_SLOPE(y, x)` exactly.
        - `STRPTIME(string, format)` is for converting strings to dates. `STRFTIME(format, date)` is for formatting dates as strings.
        - IMPORTANT: `STRPTIME` is required when a date column is a string and the format is not `YYYY-MM-DD`. For example, for a string '01-01-1995', use `STRPTIME(date_column, '%d-%m-%Y')`.
        - The function `JULIANDAY` does not exist in DuckDB. If you need to convert a date to a Julian Day number, use the `julian()` function instead.
        - When performing date arithmetic to get the difference in days, cast both dates to a `DATE` type first. For example, `(julian(CAST(decision_date AS DATE)) - julian(CAST(date_of_registration AS DATE)))`.

        Extract the following information from the user's request and return it as a single JSON object.
        
        User's request:
        "{query_text}"
        
        Your response must be a single JSON object with the following keys:
        - "url": (string, optional) The URL to scrape if the task is web scraping.
        - "queries": (object, optional) A JSON object with keys "q1", "q2", and "plot_data" for SQL queries, if the task is SQL analysis.
        - "questions": (array of strings) The questions to be answered.
        - "plot_columns": (array of strings) The two columns for the scatterplot, in the order [x, y].
        - "ml_task": (object, optional) A JSON object with "model_name" and "model_params" for a machine learning task.
        
        Do NOT include any other text or explanation in your response.
        """

        print("Sending prompt to LLM to extract parameters...")
        messages = [
            {"role": "user", "content": prompt}
        ]
        llm_response = call_llm_api(messages=messages)
        
        if 'error' in llm_response:
            return jsonify({'status': 'error', 'message': llm_response['error']}), 500
        
        try:
            llm_text_response = llm_response.get('choices')[0].get('message').get('content')
            clean_json_str = re.sub(r'```json\s*(.*?)\s*```', r'\1', llm_text_response, flags=re.DOTALL)
            extracted_params = json.loads(clean_json_str)

            result = {}
            if task_type == 'web_scraping':
                if not extracted_params.get('url'):
                    return jsonify({'status': 'error', 'message': 'LLM did not provide a URL for scraping.'}), 400
                result = perform_web_scraping(
                    url=extracted_params.get('url'),
                    questions=extracted_params.get('questions', []),
                    plot_columns=extracted_params.get('plot_columns', [])
                )
            elif task_type == 'local_analysis':
                if not data_file_paths:
                    return jsonify({'status': 'error', 'message': 'No data file provided for local analysis.'}), 400
                result = perform_local_analysis(
                    file_path=data_file_paths[0],
                    questions=extracted_params.get('questions', []),
                    plot_columns=extracted_params.get('plot_columns', []),
                    ml_task=extracted_params.get('ml_task')
                )
            elif task_type == 'sql_analysis':
                if not extracted_params.get('queries'):
                    return jsonify({'status': 'error', 'message': 'LLM did not provide SQL queries.'}), 400
                result = perform_sql_analysis(queries=extracted_params.get('queries'))
            
            if 'error' in result:
                return jsonify({'status': 'error', 'message': result['error']}), 500
            
            final_response_data = result.get('answers', [])
            if result.get('plot'):
                final_response_data.append(result.get('plot'))

            return jsonify(final_response_data), 200
        
        except (KeyError, TypeError, IndexError, json.JSONDecodeError) as e:
            return jsonify({
                'status': 'error',
                'message': f'Failed to parse LLM response. The LLM may not have returned valid JSON. Raw response: {llm_text_response}. Error: {str(e)}'
            }), 500

    except Exception as e:
        print(f"An error occurred: {e}")
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
