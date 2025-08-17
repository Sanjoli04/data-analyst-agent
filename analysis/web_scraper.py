import asyncio
import textwrap
import traceback
from playwright.async_api import async_playwright, TimeoutError
from utils.llm_api import call_llm_api
from prompts.web_scraping_prompts import get_code_generation_prompt

async def execute_scraping_logic(page, logic_block: str):
    """Run the generated scraping code inside a controlled async environment."""
    # This function creates a temporary async function to execute the AI's code.
    # It ensures that the 'page' object is correctly passed and that any result is captured.
    # --- FIX: Removed 'scraped_data = None' to prevent conflict with the AI's logic. ---
    # The AI's script is now fully responsible for creating and assigning the 'scraped_data' variable.
    function_code = f"""
async def _ai_logic(page):
{logic_block}
    # We now check if the AI's script successfully created the variable at all.
    if 'scraped_data' not in locals():
        return ["Error: The AI's logic completed without producing a result variable."]
    return scraped_data
"""
    # A sandbox dictionary to hold the dynamically executed function.
    sandbox = {}
    try:
        # Execute the string as Python code, defining the _ai_logic function.
        exec(function_code, sandbox)
        # Call the newly defined function and return its result.
        return await sandbox['_ai_logic'](page)
    except Exception as e:
        # If the generated code has a syntax or runtime error, catch it.
        error_trace = traceback.format_exc()
        return [f"Execution of generated script failed: {e}\n{error_trace}"]

def perform_dynamic_web_scraping(config):
    """
    Orchestrates the entire web scraping process, from browser launch to completion.
    """
    url = config.get("url")
    objective = config.get("objective")
    if not all([url, objective]):
        return {"error": "Missing 'url' or 'objective' for web scraping."}

    # --- Step 1: Generate the prompt and get the AI's logic ---
    prompt = get_code_generation_prompt(url=url, objective=objective)
    messages = [{"role": "user", "content": prompt}]
    
    llm_response = call_llm_api(messages)
    if 'error' in llm_response:
        return {"error": f"LLM API Error: {llm_response['error']}"}

    try:
        # Extract the Python code block from the LLM's response.
        logic_block = llm_response['choices'][0]['message']['content']
        if '```python' in logic_block:
            logic_block = logic_block.split('```python')[1].split('```')[0]
        
        # Clean up the code block's indentation to prepare it for injection.
        dedented_logic = textwrap.dedent(logic_block).strip()
        indented_logic = textwrap.indent(dedented_logic, '    ')
    except (KeyError, IndexError) as e:
        return {"error": f"Failed to parse logic block from LLM response: {e}"}

    # --- Step 2: Define the main async workflow to run the browser ---
    async def run_browser_task():
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            print(f"Navigating to {url}...")
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            
            # Attempt to automatically handle common cookie/consent banners.
            consent_locator = page.locator(
                "button:has-text('Accept'), button:has-text('Agree'), "
                "button:has-text('OK'), button:has-text('Allow all'), "
                "button:has-text('Accept all')"
            )
            try:
                await consent_locator.first.click(timeout=3000)
                print("Consent banner handled.")
            except TimeoutError:
                print("No consent banner found, which is fine. Continuing.")
            
            print("Waiting for page to fully load...")
            await page.wait_for_load_state("networkidle", timeout=60000)
            print("Page is ready. Executing AI logic...")

            # Execute the AI's logic on the fully prepared page.
            result = await execute_scraping_logic(page, indented_logic)
            
            await browser.close()
            return {"answers": result}

    # --- Step 3: Run the async workflow and return the final result ---
    try:
        # This is the entry point that starts the async browser operations.
        return asyncio.run(run_browser_task())
    except Exception as e:
        traceback.print_exc()
        return {"error": f"A critical error occurred during the async scraping task: {e}"}
