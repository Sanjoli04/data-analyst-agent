# analysis/local_analyzer.py (API Enabled)
import pandas as pd
import numpy as np
import os
import requests
import time

# --- Scikit-learn imports for tabular data ---
from sklearn import (
    cluster, linear_model, tree, ensemble,
    model_selection, metrics, preprocessing, impute
)

from utils.plotting import create_plot

# --- CONFIGURATIONS ---
# (Tabular ML model configs remain the same)
ML_MODELS = {
    "LinearRegression": linear_model.LinearRegression,
    "LogisticRegression": linear_model.LogisticRegression,
    "KMeans": cluster.KMeans,
    "DecisionTreeClassifier": tree.DecisionTreeClassifier,
    "RandomForestClassifier": ensemble.RandomForestClassifier,
}
EVALUATION_METRICS = {
    "accuracy": metrics.accuracy_score, "precision": metrics.precision_score,
    "recall": metrics.recall_score, "f1_score": metrics.f1_score,
    "r2_score": metrics.r2_score, "mse": metrics.mean_squared_error,
}

# ==============================================================================
# --- NEW HUGGING FACE API WORKFLOW ---
# ==============================================================================

def _analyze_image_with_api(image_paths, config):
    """
    Analyzes images using the Hugging Face Inference API.
    This is a lightweight, scalable alternative to local training.
    """
    HUGGING_FACE_API_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")
    if not HUGGING_FACE_API_TOKEN:
        return ["Error: HUGGING_FACE_API_TOKEN not found in environment variables."]

    model_id = config.get("model_id")
    if not model_id:
        return ["Error: No model_id provided in the image analysis configuration."]

    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_TOKEN}"}
    
    results = []
    
    # Analyze each uploaded image
    for image_path in image_paths:
        results.append(f"\n--- Analysis for: {os.path.basename(image_path)} ---")
        
        with open(image_path, "rb") as f:
            data = f.read()

        try:
            # Initial API call
            response = requests.post(API_URL, headers=headers, data=data, timeout=45)
            
            # Handle model loading time on Hugging Face (HTTP 503)
            if response.status_code == 503:
                estimated_time = response.json().get("estimated_time", 20)
                results.append(f"Model is loading, please wait ~{int(estimated_time)} seconds...")
                time.sleep(estimated_time)
                response = requests.post(API_URL, headers=headers, data=data, timeout=45)

            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            api_result = response.json()

            # Format the output nicely
            if isinstance(api_result, list): # For classification/detection
                for item in api_result:
                    label = item.get('label', 'N/A').title()
                    score = item.get('score')
                    box = item.get('box')
                    score_str = f" (Confidence: {score:.2%})" if score else ""
                    box_str = f" at location {box}" if box else ""
                    results.append(f"- Found: {label}{score_str}{box_str}")
            elif isinstance(api_result, dict): # For image-to-text
                caption = api_result.get('generated_text', 'No caption generated.')
                results.append(f"- Generated Caption: {caption}")

        except requests.exceptions.RequestException as e:
            results.append(f"API Request Failed: {e}. Check model ID and your API token.")
        except Exception as e:
            results.append(f"An unexpected error occurred: {e}")

    return results


# ==============================================================================
# --- EXISTING TABULAR DATA WORKFLOW (Unchanged) ---
# ==============================================================================

def _preprocess_data(df, steps, features, target):
    # ... (This function remains unchanged) ...
    df_processed = df.copy()
    if not steps: return df_processed.dropna(), features
    for step in steps:
        step_name = step.get("step"); columns = step.get("columns", []); strategy = step.get("strategy")
        if "all_numeric" in columns:
            columns.remove("all_numeric"); columns.extend(df_processed.select_dtypes(include=np.number).columns.tolist())
        if "all_categorical" in columns:
            columns.remove("all_categorical"); columns.extend(df_processed.select_dtypes(include=['object', 'category']).columns.tolist())
        columns = [col for col in list(set(columns)) if col != target]
        if step_name == "impute":
            imputer = impute.SimpleImputer(strategy=strategy); df_processed[columns] = imputer.fit_transform(df_processed[columns])
        elif step_name == "scale":
            scaler = preprocessing.StandardScaler() if strategy == "standard" else preprocessing.MinMaxScaler(); df_processed[columns] = scaler.fit_transform(df_processed[columns])
        elif step_name == "one_hot_encode":
            df_processed = pd.get_dummies(df_processed, columns=columns, drop_first=True)
            new_features = [col for col in df_processed.columns if any(c in col for c in columns)]
            features.extend(new_features); features = [f for f in features if f not in columns]
    return df_processed.dropna(), list(set(features))

def _train_and_evaluate_model(df, ml_task_config):
    # ... (This function remains unchanged) ...
    features = ml_task_config.get('features'); target = ml_task_config.get('target'); model_name = ml_task_config.get('model_name'); model_params = ml_task_config.get('model_params', {}); eval_config = ml_task_config.get("evaluation", {}); results = []
    df_processed, updated_features = _preprocess_data(df, ml_task_config.get("preprocessing_steps", []), features, target)
    if df_processed.empty: return ["ML task failed: No data available after preprocessing."]
    X = df_processed[updated_features]; y = df_processed[target]
    test_size = eval_config.get("test_size", 0.2); X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=42)
    results.append(f"Data split into {1-test_size:.0%} training and {test_size:.0%} testing sets.")
    model = ML_MODELS[model_name](**model_params); model.fit(X_train, y_train); results.append(f"Successfully trained {model_name} model.")
    y_pred = model.predict(X_test); eval_metrics = eval_config.get("metrics", [])
    if eval_metrics:
        results.append("\n--- Model Performance ---")
        for metric_name in eval_metrics:
            if metric_name in EVALUATION_METRICS:
                try: score = EVALUATION_METRICS[metric_name](y_test, y_pred); results.append(f"- {metric_name.replace('_', ' ').title()}: {score:.4f}")
                except Exception as e: results.append(f"- Could not calculate {metric_name}: {e}")
    return results

# ==============================================================================
# --- MAIN ORCHESTRATOR FUNCTION ---
# ==============================================================================

def perform_local_analysis(file_paths, analysis_requests, ml_task=None, image_analysis_task=None):
    """
    Main orchestrator. Detects file type and routes to the correct workflow.
    """
    results = []
    plot_uri = None
    if not file_paths: return {"error": "No data file path provided."}
        
    # --- ROUTING LOGIC: Check for image task ---
    is_image_task = any(fp.lower().endswith(('.png', '.jpg', '.jpeg')) for fp in file_paths) or image_analysis_task

    if is_image_task:
        if image_analysis_task:
            img_results = _analyze_image_with_api(file_paths, image_analysis_task)
            results.extend(img_results)
            return {"answers": results}
        else:
            return {"error": "Image files detected, but no image_analysis_task config provided."}

    # --- If not an image task, proceed with TABULAR workflow ---
    file_path = file_paths[0]
    try:
        if file_path.endswith('.csv'): df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        elif file_path.endswith(('.xls', '.xlsx')): df = pd.read_excel(file_path)
        elif file_path.endswith('.parquet'): df = pd.read_parquet(file_path)
        else: return {"error": f"Unsupported tabular file type: {file_path}"}
    except Exception as e: return {'error': f"Failed to read data file: {e}"}

    if ml_task:
        ml_results = _train_and_evaluate_model(df, ml_task)
        results.extend(ml_results)

    for req in analysis_requests:
        # ... (Simple analysis request loop remains unchanged) ...
        req_type = req.get('type'); params = req.get('params', {}); required_cols = [col for col in [params.get('column'), params.get('col1'), params.get('col2'), params.get('x_col'), params.get('y_col')] if col]
        if not all(col in df.columns for col in required_cols): results.append(f"Analysis '{req_type}' failed: Columns not found: {[c for c in required_cols if c not in df.columns]}."); continue
        if req_type == "describe": results.append(f"Data description:\n{df.describe().to_string()}")
        elif req_type == "count_missing_values": results.append(f"Total missing values: {df.isnull().sum().sum()}.")
        elif req_type == "calculate_average":
            column = params.get('column')
            if pd.api.types.is_numeric_dtype(df[column]): results.append(f"The average of '{column}' is {df[column].mean():.2f}.")
            else: results.append(f"Cannot average non-numeric column '{column}'.")
        elif req_type == "find_correlation":
            col1, col2 = params.get('col1'), params.get('col2')
            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                correlation = df[[col1, col2]].corr().iloc[0, 1]; results.append(f"Correlation between '{col1}' and '{col2}' is {correlation:.2f}.")
            else: results.append(f"Cannot correlate non-numeric columns.")
        elif req_type == "plot":
            plot_result = create_plot(df, params['x_col'], params['y_col'], regression=params.get('regression', False))
            if isinstance(plot_result, str): plot_uri = plot_result
            else: results.append(plot_result.get('error', 'Plotting failed.'))
    
    final_response = results
    if plot_uri: final_response.append(plot_uri)
    return {"answers": final_response}
