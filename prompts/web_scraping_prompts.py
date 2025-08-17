# prompts/web_scraping_prompts.py (BeautifulSoup Masterclass)

PROMPT_HEADER = """
You are a world-class Python developer specializing in web scraping and data analysis. Your primary function is to generate a Python script to parse a provided HTML string and answer the user's objective.

--- CRITICAL RULES ---
- Your script will be executed in an environment where a variable `html_content` (containing the full page HTML) is already defined.
- You MUST use the `BeautifulSoup` library for all HTML parsing.
- Do NOT write hardcoded logic. Your script must be adaptable. Specifically, do not hardcode column names in your analysis; derive them from the scraped table.
- Your final output MUST be a list of strings assigned to a variable named `scraped_data`.

--- MASTERCLASS EXAMPLE ---
Study this perfect script that scrapes a generic Wikipedia table from an HTML string. You must imitate its robust and adaptable patterns to succeed.

```python
# --- Start of Gold Standard Example Snippet ---
from bs4 import BeautifulSoup
import pandas as pd
import re
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

# The 'html_content' variable is provided by the execution environment.
soup = BeautifulSoup(html_content, 'html.parser')

# --- 1. RESILIENT SCRAPING LOGIC ---
scraped_data_list = []
table = soup.find('table', {'class': 'wikitable'}) # Find the main data table

if table:
    # --- 2. DYNAMIC HEADER DETECTION & CLEANING ---
    headers = []
    # Find the last row in the header, which usually contains the correct column names
    header_row = table.find('thead').find_all('tr')[-1] 
    for th in header_row.find_all('th'):
        header_text = re.sub(r'\\s+', ' ', th.get_text(strip=True))
        # Clean the header to be a valid, lowercase identifier
        cleaned_header = re.sub(r'[^0-9a-zA-Z_]', '', header_text.replace(' ', '_')).lower()
        headers.append(cleaned_header)

    # --- 3. RESILIENT ROW PARSING ---
    for row in table.find('tbody').find_all('tr'):
        # Find all cell types (th for header-like cells in rows, td for data cells)
        cells = row.find_all(['th', 'td'])
        if len(cells) == len(headers):
            row_data = {headers[i]: cell.get_text(strip=True) for i, cell in enumerate(cells)}
            scraped_data_list.append(row_data)

# --- 4. ANALYSIS & FINALIZATION ---
if not scraped_data_list:
    scraped_data = ["Error: No valid data rows could be parsed from the table."]
else:
    df = pd.DataFrame(scraped_data_list)
    
    # --- 5. ROBUST DATA CLEANING & TYPE CONVERSION ---
    df.replace(r'\\[[a-zA-Z0-9]+\\]', '', regex=True, inplace=True) # Clean wiki references
    for col in df.columns:
        # First, clean the string values to remove non-numeric characters.
        df[col] = df[col].astype(str).str.replace(r'[^0-9.\\-]', '', regex=True)
        # Then, convert to numeric, coercing any remaining errors into Not-a-Number (NaN).
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

    scraped_data = [
        "Success: Analysis complete.",
        f"Total items scraped: {num_items}",
        "Data Summary:",
        description,
        image_answer
    ]
# --- End of Gold Standard Example Snippet ---
```
"""

PROMPT_FOOTER = """
--- YOUR MISSION ---
- **URL:** {url}
- **OBJECTIVE:** {objective}
- **HTML CONTENT**: The full HTML of the page is provided in the `html_content` variable.

Now, using the exact same adaptable and resilient patterns from the Masterclass Example, write a Python script to parse the `html_content` and fulfill the user's objective. Your final output MUST be assigned to the `scraped_data` variable.
"""

def get_code_generation_prompt(url: str, objective: str, html_content: str) -> str:
    """
    Constructs the full, formatted prompt for the LLM by combining the header and footer.
    """
    # We only pass a snippet of the HTML to the LLM to avoid exceeding token limits,
    # but the full HTML is available in the execution environment.
    html_snippet = html_content[:4000]
    
    # We don't actually need to pass the snippet in the prompt text itself,
    # as the instruction is just to use the `html_content` variable.
    # This keeps the prompt cleaner.
    
    full_prompt = PROMPT_HEADER + PROMPT_FOOTER.format(url=url, objective=objective)
    return full_prompt
