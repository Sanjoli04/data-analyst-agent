# analysis/web_scraper.py (Refactored for BeautifulSoup)
import asyncio
import textwrap
import traceback
from playwright.async_api import async_playwright, TimeoutError
from utils.llm_api import call_llm_api
from prompts.web_scraping_prompts import get_code_generation_prompt

def execute_scraping_logic(html_content: str, logic_block: str):
    """
    Executes the AI-generated Python script (using BeautifulSoup) in a sandboxed environment.
    """
    # The sandbox is pre-populated with the necessary html_content variable.
    sandbox = {"html_content": html_content}
    try:
        # The AI's script is expected to define a 'scraped_data' variable in the sandbox.
        exec(logic_block, sandbox)
        return sandbox.get('scraped_data', ["Error: The AI's script completed without producing a result variable."])
    except Exception as e:
        error_trace = traceback.format_exc()
        return [f"Execution of generated script failed: {e}\n{error_trace}"]

def perform_dynamic_web_scraping(config):
    """
    Orchestrates the web scraping process:
    1. Fetches the page's static HTML using Playwright.
    2. Sends the HTML to the LLM to generate a BeautifulSoup parsing script.
    3. Executes the generated script to get the final analysis.
    """
    url = config.get("url")
    objective = config.get("objective")
    if not all([url, objective]):
        return {"error": "Missing 'url' or 'objective' for web scraping."}

    async def fetch_page_html():
        """Connects to the browser, gets the HTML, and closes."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                print(f"Navigating to {url}...")
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                
                # A short wait can help with pages that load content dynamically after DOM load.
                await page.wait_for_timeout(3000) 
                
                print("Page is ready. Fetching HTML content...")
                html_content = await page.content()
                
            finally:
                await browser.close()

            return html_content

    try:
        # --- Step 1: Get the page HTML ---
        html = asyncio.run(fetch_page_html())
        if not html:
            return {"error": "Failed to retrieve HTML content from the page."}

        # --- Step 2: Generate the prompt and get the AI's logic ---
        prompt = get_code_generation_prompt(url=url, objective=objective, html_content=html)
        messages = [{"role": "user", "content": prompt}]
        
        llm_response = call_llm_api(messages)
        if 'error' in llm_response:
            return {"error": f"LLM API Error: {llm_response['error']}"}

        logic_block = llm_response['choices'][0]['message']['content']
        if '```python' in logic_block:
            logic_block = logic_block.split('```python')[1].split('```')[0]
        
        # --- Step 3: Execute the AI's logic ---
        result = execute_scraping_logic(html, logic_block)
        return {"answers": result}

    except Exception as e:
        traceback.print_exc()
        return {"error": f"A critical error occurred during the scraping task: {e}"}
