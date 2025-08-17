# prompts/web_scraping_prompts.py (Final Resilient Version)

PROMPT_HEADER = """
You are a world-class Python developer specializing in web scraping and data analysis. Your primary function is to generate a Python script snippet that is generic and can handle any web scraping task.
**CRITICAL INSTRUCTION**: Do not write hardcoded logic. Your script must be adaptable and resilient, using the dynamic, await-based patterns demonstrated below. You must not use `asyncio.run()`.

Your mission is to write a script that accomplishes the user's objective by following the Masterclass Example.

--- MASTERCLASS EXAMPLE ---
Study this perfect script that scrapes a generic Wikipedia table. It is highly resilient and uses the correct asynchronous pattern. You must imitate its structure precisely.

```python
# --- Start of Gold Standard Example Snippet ---
from playwright.async_api import TimeoutError
import pandas as pd
import re
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

# Define the main async logic in a function.
async def scrape_and_analyze(page):
    # --- 1. INITIALIZATION ---
    scraped_data_list = []
    raw_data_for_debugging = []
    table_locator = None

    # --- 2. RESILIENT SCRAPING LOGIC ---
    try:
        possible_selectors = ["table.wikitable.sortable", "table.wikitable", "table[class*='wikitable']"]
        for selector in possible_selectors:
            try:
                table_locator = page.locator(selector).first
                await table_locator.wait_for(timeout=5000)
                print(f"INFO: Table found with selector: '{selector}'")
                break
            except TimeoutError:
                print(f"INFO: Selector '{selector}' did not find a table.")
                continue
                
        if not table_locator:
            raise TimeoutError("Could not find a suitable data table on the page after trying all selectors.")

        header_locators = await table_locator.locator("thead tr").last.locator("th").all()
        
        raw_headers = [re.sub(r'\\s+', ' ', (await th.text_content() or '')).strip() for th in header_locators]
        headers = [re.sub(r'[^0-9a-zA-Z_]', '', h.replace(' ', '_')).lower() for h in raw_headers]

        all_row_locators = await table_locator.locator("tbody tr").all()

        for row_locator in all_row_locators:
            cell_locators = await row_locator.locator("th, td").all()
            if not cell_locators or len(cell_locators) != len(headers):
                continue

            raw_texts = [(await cell.text_content() or '').strip() for cell in cell_locators]
            if not any(raw_texts):
                continue

            raw_data_for_debugging.append(" | ".join(raw_texts))
            row_data = {headers[i]: raw_texts[i] for i in range(len(headers))}
            scraped_data_list.append(row_data)

    except TimeoutError as e:
        return [f"Error: {e}"]
    except Exception as e:
        return [f"An unexpected error occurred during scraping: {str(e)}"]

    # --- 3. ANALYSIS & FINALIZATION ---
    if not scraped_data_list:
        return ["Error: No valid data rows could be parsed.", "RAW DATA PREVIEW:"] + raw_data_for_debugging[:5]
    
    df = pd.DataFrame(scraped_data_list)
    
    # --- FIX: Robust data cleaning and type conversion ---
    # Clean common reference links and then iterate through each column.
    df.replace(r'\\[[a-zA-Z0-9]+\\]', '', regex=True, inplace=True)
    for col in df.columns:
        # First, clean the string values in the column to remove non-numeric characters.
        df[col] = df[col].astype(str).str.replace(r'[^0-9.\\-]', '', regex=True)
        # Now, convert to numeric, coercing any remaining errors into Not-a-Number (NaN).
        df[col] = pd.to_numeric(df[col], errors='coerce')

    num_items = len(df)
    description = df.describe(include='all').to_string()
    
    image_answer = "No numeric data found to plot."
    try:
        # Dynamically select the first available numeric column for plotting.
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            plot_col = numeric_cols[0]
            df.dropna(subset=[plot_col], inplace=True)
            
            if not df.empty:
                plt.figure(figsize=(10, 6))
                sns.histplot(df[plot_col], bins=20, kde=True)
                plt.title(f'Distribution of {plot_col}')
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                image_answer = f"data:image/png;base64,{img_base64}"
                plt.close()
    except Exception as plot_e:
        image_answer = f"Could not generate plot: {plot_e}"

    return [
        "Success: Analysis complete.",
        f"Total items scraped: {num_items}",
        "Data Summary:",
        description,
        image_answer
    ]

# The final result is the execution of the async function.
scraped_data = await scrape_and_analyze(page)
# --- End of Gold Standard Example Snippet ---
```
"""

PROMPT_FOOTER = """
--- YOUR MISSION ---
- **URL:** {url}
- **OBJECTIVE:** {objective}

Now, using the exact same adaptable and resilient patterns from the Masterclass Example, write the scraping and analysis logic to fulfill the user's objective. Your final output MUST be assigned to the `scraped_data` variable by awaiting the main async function.
"""

def get_code_generation_prompt(url: str, objective: str) -> str:
    """
    Constructs the full, formatted prompt for the LLM by combining the header and footer.
    """
    full_prompt = PROMPT_HEADER + PROMPT_FOOTER.format(url=url, objective=objective)
    return full_prompt
