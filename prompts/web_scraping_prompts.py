# prompts/web_scraping_prompts.py (Definitive "Learn by Example" Version)

# Part 1: The static header with instructions and the Gold Standard Example.
PROMPT_HEADER = """
You are an expert web scraping agent. Your mission is to write a bug-free Playwright script by learning from a perfect example.

--- GOLD STANDARD EXAMPLE ---
Here is a perfect script that scrapes the highest-grossing films. Study its patterns carefully.

```python
# --- Start of Gold Standard Example ---
from playwright.async_api import TimeoutError
import pandas as pd
import re

# Use a specific, content-based locator to find the correct table
table_locator = page.locator("table.wikitable:has-text('Worldwide gross')")

# Wait for the first row to ensure the table's data has loaded
try:
    await table_locator.locator("tr").first.wait_for(timeout=30000)
except TimeoutError:
    # If no rows are found, exit gracefully with an error message.
    scraped_data = ["Error: Could not find any data rows in the target table."]
    return

# Now that we know rows exist, get all of them
all_row_locators = await table_locator.locator("tr").all()

scraped_rows = []
# Loop through all rows, skipping the first (header) row
for row_locator in all_row_locators[1:]:
    # Get all cell locators for the current row
    cell_locators = await row_locator.locator("td").all()
    
    # Ensure the row has enough cells to prevent IndexError
    if len(cell_locators) < 7:
        continue

    try:
        # Await each piece of text individually before processing
        rank_text = await cell_locators[0].text_content()
        title_text = await cell_locators[1].text_content()
        gross_text = await cell_locators[5].text_content()
        year_text = await cell_locators[6].text_content()

        # --- Robust Parsing Logic ---
        # Use regex to find the first number for rank and year, handling cases where it might be missing
        rank_match = re.search(r"(\\d+)", rank_text)
        rank = int(rank_match.group(1)) if rank_match else None

        year_match = re.search(r"(\\d{4})", year_text)
        year = int(year_match.group(1)) if year_match else None

        # Clean the currency string
        gross_cleaned = re.sub(r"\\[.*?\\]", "", gross_text).replace("$", "").strip()
        
        # Correctly scale "billion" or "million"
        gross_in_millions = 0.0
        if "billion" in gross_cleaned:
            gross_in_millions = float(gross_cleaned.replace("billion", "").strip()) * 1000
        elif "million" in gross_cleaned:
            gross_in_millions = float(gross_cleaned.replace("million", "").strip())

        # Add the clean data to our list only if all parts are valid
        if rank and title_text and year and gross_in_millions > 0:
            scraped_rows.append({
                "Rank": rank,
                "Title": title_text.strip(),
                "Year": year,
                "Gross_Millions": gross_in_millions
            })
    except (AttributeError, ValueError, TypeError) as e:
        # If any parsing fails on a row, print a message and skip it.
        print(f"Skipping a row due to a parsing error: {e}")
        continue

# --- Analysis Phase ---
# Check if any data was actually collected before trying to analyze it.
if not scraped_rows:
    scraped_data = ["Error: Data extraction was attempted, but no valid rows could be parsed."]
else:
    df = pd.DataFrame(scraped_rows)
    print("--- DataFrame Info ---")
    print("Columns:", df.columns)
    print(df.head())
    
    # Perform the analysis on the clean DataFrame
    # Example: How many films grossed over $2 billion?
    over_2_billion = df[df['Gross_Millions'] >= 2000].shape[0]
    
    # Final answers must be a list of strings
    scraped_data = [
        f"Number of films grossing over $2 billion: {over_2_billion}"
    ]
# --- End of Gold Standard Example ---
```

--- KEY PATTERNS TO LEARN ---
1.  **Await Actions, Then Process:** `await` is always used on a Playwright action before the result is used.
2.  **Robust Loop:** The `for` loop iterates through locators, `await`-ing data for each row individually.
3.  **Specific Error Handling:** Each row is processed in a `try...except` block that catches common parsing errors (`AttributeError`, `ValueError`).
4.  **Data Validation:** The script checks if any rows were successfully scraped (`if not scraped_rows:`) before attempting analysis.
5.  **Assign Final List:** The final list of string answers is assigned to `scraped_data` at the very end.
"""

# Part 2: The simple footer that will be formatted with the mission details.
PROMPT_FOOTER = """
--- YOUR MISSION ---
Now, using the exact same patterns and logic from the Gold Standard Example, write a script to fulfill the user's objective.

- **URL:** {url}
- **OBJECTIVE:** {objective}

Write ONLY the Python code.
"""
