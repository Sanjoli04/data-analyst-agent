# prompts/prompt_manager.py (Corrected with file_names)
from . import sql_prompts, local_analysis_prompts, network_analysis_prompts

def get_prompt(task_type, **kwargs):
    """
    Retrieves and formats the correct prompt based on the task type.
    """
    if task_type == 'sql_analysis':
        prompt_template = sql_prompts.SQL_ANALYSIS_PROMPT
        file_info = kwargs.get("file_info", {})
        # --- FIX: Pass both file_names and column_names into the prompt ---
        return prompt_template.format(
            query_text=kwargs.get("query_text", ""),
            file_names=', '.join(file_info.get('names', [])),
            column_names=', '.join(file_info.get('columns', []))
        )

    elif task_type == 'local_analysis':
        prompt_template = local_analysis_prompts.LOCAL_ANALYSIS_PROMPT
        file_details = "No data files uploaded."
        file_info = kwargs.get("file_info")
        if file_info and file_info.get('names'):
            file_details = f"""
            The user uploaded the following files:
            - File names: {', '.join(file_info['names'])}
            - File columns (first file): {', '.join(file_info['columns'])}
            """
        return prompt_template.format(
            query_text=kwargs.get("query_text", ""),
            file_details=file_details
        )
    
    elif task_type == 'network_analysis':
        prompt_template = network_analysis_prompts.NETWORK_ANALYSIS_PROMPT
        file_details = "No data files uploaded."
        file_info = kwargs.get("file_info")
        if file_info and file_info.get('names'):
            file_details = f"""
            The user uploaded the following files:
            - File names: {', '.join(file_info['names'])}
            - File columns (first file): {', '.join(file_info['columns'])}
            """
        return prompt_template.format(
            query_text=kwargs.get("query_text", ""),
            file_details=file_details
        )
            
    elif task_type == 'web_scraping':
        # This case is handled directly by the web_scraper module.
        return None
            
    else:
        raise ValueError(f"Unknown task type: {task_type}")
