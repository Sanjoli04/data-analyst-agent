import pandas as pd
import traceback
import asyncio
import textwrap
import re
import codecs
from playwright.async_api import async_playwright, TimeoutError
from utils.llm_api import call_llm_api
# Import the new prompt variables
from prompts.web_scraping_prompts import PROMPT_HEADER, PROMPT_FOOTER

# This template is used to build the final, runnable script
SCRIPT_TEMPLATE = """
import asyncio
from playwright.async_api import async_playwright, TimeoutError
import pandas as pd
import time
import re
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import traceback
from scipy.stats import linregress

async def main():
    # Default to None, so we can check if it was ever assigned
    scraped_data = None
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
{logic_block}
            # --- FIX: Add a final check after the AI's code runs ---
            if scraped_data is None:
                print("Execution finished, but 'scraped_data' was not assigned.")
                scraped_data = ["Error: The AI's logic completed without producing a result. This could be due to no data matching its criteria."]

        except Exception as e:
            scraped_data = [f"A critical error occurred: {{repr(e)}}\\n{{traceback.format_exc()}}"]
        finally:
            await browser.close()
    return scraped_data
"""

# This function executes the generated script
async def _execute_scraping_code_async(code_string):
    sandbox = {}
    try:
        exec(code_string, sandbox)
        main_coroutine = sandbox.get('main')
        if not asyncio.iscoroutinefunction(main_coroutine):
            return {"error": "No valid async main() found in generated code"}
        return await main_coroutine()
    except Exception as e:
        print("--- ERROR EXECUTING SCRIPT ---")
        print(code_string)
        error_trace = traceback.format_exc()
        print(error_trace)
        return {"error": f"Execution of generated script failed: {e}", "traceback": error_trace}

# This function orchestrates the web scraping task
def perform_dynamic_web_scraping(config):
    url, objective = config.get("url"), config.get("objective")
    if not all([url, objective]):
        return {"error": "Missing 'url' or 'objective' in config"}

    # Construct the prompt safely in two parts
    formatted_footer = PROMPT_FOOTER.format(url=url, objective=objective)
    prompt = PROMPT_HEADER + formatted_footer
    
    llm_response = call_llm_api([{"role": "user", "content": prompt}])

    if 'error' in llm_response:
        return {"error": f"LLM API Error: {llm_response['error']}"}
    if not llm_response.get("choices") or not isinstance(llm_response.get("choices"), list) or len(llm_response["choices"]) == 0:
        return {"error": f"The LLM API returned a response with no valid 'choices'."}

    try:
        raw_content = llm_response['choices'][0]['message']['content']
        if '\\' in raw_content:
             raw_content = codecs.decode(raw_content, 'unicode_escape')

        logic_block = raw_content
        match = re.search(r"```(?:python)?\s*\n(.*)```", raw_content, re.DOTALL)
        if match:
            logic_block = match.group(1)
        
        indented_logic = textwrap.indent(textwrap.dedent(logic_block).strip(), ' ' * 12)
        full_script = SCRIPT_TEMPLATE.format(logic_block=indented_logic)

    except Exception as e:
        return {"error": f"An unexpected error occurred while preparing the script: {e}"}

    try:
        scraped_result = asyncio.run(_execute_scraping_code_async(full_script))

        if isinstance(scraped_result, dict) and 'error' in scraped_result:
            return scraped_result
        if isinstance(scraped_result, list):
            return {"answers": scraped_result}
        
        # This handles the case where the script returns a single non-list item
        if scraped_result is not None:
             return {"answers": [str(scraped_result)]}
        
        # This should now be caught by the new logic in the template, but as a fallback:
        return {"error": "Script finished with an unexpected None result."}
    except Exception as e:
        return {"error": f"A critical error occurred while running the async task: {e}"}
