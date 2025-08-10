# main.py (Refactored)
from flask import Flask, request, render_template, jsonify
import os
import tempfile
import shutil
import json
import re
import traceback
import pandas as pd
from dotenv import load_dotenv

# --- Modular Imports ---
from prompts.prompt_manager import get_prompt
from analysis.sql_analyzer import run_sql_analysis
from analysis.web_scraper import perform_dynamic_web_scraping
from analysis.local_analyzer import perform_local_analysis
from utils.llm_api import call_llm_api

load_dotenv()
app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api', methods=['POST'])
def api_endpoint():
    if not os.getenv("APIPE_API_KEY"):
        return jsonify({'error': 'API key is not configured on the server.'}), 500

    temp_dir = tempfile.mkdtemp()
    try:
        uploaded_files = request.files.getlist('uploaded_files')
        query_text = ""
        data_file_paths = []
        file_info = {'names': [], 'columns': []}

        if not uploaded_files:
            return jsonify({'error': 'No files were uploaded.'}), 400

        for file in uploaded_files:
            temp_filepath = os.path.join(temp_dir, file.filename)
            file.save(temp_filepath)
            if file.filename.endswith('.txt') and not query_text:
                with open(temp_filepath, 'r', encoding='utf-8') as f:
                    query_text = f.read()
            else:
                data_file_paths.append(temp_filepath)
                file_info['names'].append(file.filename)
        
        if not query_text:
            return jsonify({'error': 'A query .txt file is required.'}), 400

        # --- Task Routing ---
        task_type = 'local_analysis'
        query_lower = query_text.lower()
        url_match = re.search(r'https?://[^\s]+', query_lower)
        
        if "scrape" in query_lower and url_match:
            task_type = 'web_scraping'
        elif any(keyword in query_lower for keyword in ['select ', ' from ', 'sql']):
            task_type = 'sql_analysis'
        
        result = {}

        # --- NEW DIRECT WORKFLOW FOR WEB SCRAPING ---
        if task_type == 'web_scraping':
            print("INFO: Starting Web Scraping Task")
            scraping_config = {"url": url_match.group(0), "objective": query_text}
            result = perform_dynamic_web_scraping(scraping_config)
        
        # --- STANDARD WORKFLOW FOR SQL AND LOCAL ANALYSIS ---
        else: # task_type is 'sql_analysis' or 'local_analysis'
            if task_type == 'local_analysis' and data_file_paths:
                try:
                    peek_path = data_file_paths[0]
                    if peek_path.endswith('.csv'):
                        df_peek = pd.read_csv(peek_path, nrows=1, on_bad_lines='skip')
                        file_info['columns'] = list(df_peek.columns)
                except Exception as e:
                    print(f"Could not peek into file for columns: {e}")

            prompt = get_prompt(task_type, query_text=query_text, file_info=file_info)
            messages = [{"role": "user", "content": prompt}]
            llm_response = call_llm_api(messages)
            
            if 'error' in llm_response:
                return jsonify({'error': f"LLM API Error: {llm_response['error']}"}), 500
            
            try:
                raw_content = llm_response['choices'][0]['message']['content']
                config_str = re.sub(r'```json\s*(.*?)\s*```', r'\1', raw_content, flags=re.DOTALL).strip()
                config = json.loads(config_str)
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                return jsonify({'error': f"Failed to parse LLM JSON response: {e}. Raw: {raw_content}"}), 500

            if task_type == 'sql_analysis':
                for name in file_info['names']:
                    for qk, qv in config['sql_config']['queries'].items():
                        config['sql_config']['queries'][qk] = qv.replace(f"'{name}'", f"read_csv_auto('{os.path.join(temp_dir, name)}')")
                result = run_sql_analysis(config['sql_config'])
            
            elif task_type == 'local_analysis':
                config_data = config['local_analysis_config']
                result = perform_local_analysis(
                    file_paths=data_file_paths,
                    analysis_requests=config_data.get('analysis_requests', []),
                    ml_task=config_data.get('ml_task'),
                    image_analysis_task=config_data.get('image_analysis_task')
                )
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
        
        final_response_data = result.get('answers', ["No textual answer produced."])
        return jsonify(final_response_data), 200

    except Exception as e:
        print(f"FATAL ERROR in /api endpoint: {e}")
        traceback.print_exc()
        return jsonify({'error': f'A critical server error occurred: {str(e)}'}), 500
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write("<html><body><h1>Data Analyst Agent</h1><p>Ready for analysis.</p></body></html>")
            
    app.run(debug=True, host='0.0.0.0', port=5001)
