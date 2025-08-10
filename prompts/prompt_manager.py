# prompts/prompt_manager.py (Updated)
from . import sql_prompts, local_analysis_prompts

# The web scraping prompt is now for code generation and is handled directly
# in the web_scraper module, so we remove it from here.

def get_prompt(task_type, **kwargs):
    """
    Retrieves and formats the correct prompt based on the task type.
    """
    if task_type == 'sql_analysis':
        prompt_template = sql_prompts.SQL_ANALYSIS_PROMPT
        return prompt_template.format(query_text=kwargs.get("query_text", ""))

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
    
    # We no longer need a specific prompt for web scraping here,
    # as the logic is now self-contained in the analyzer.
    elif task_type == 'web_scraping':
        # This case is now handled by perform_dynamic_web_scraping
        return None
            
    else:
        raise ValueError(f"Unknown task type: {task_type}")
