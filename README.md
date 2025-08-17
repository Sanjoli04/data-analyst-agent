# 💡 Data Analyst Agent

This is a powerful, AI-driven **data analysis agent** designed to streamline the process of sourcing, preparing, analyzing, and visualizing data. The agent uses a combination of natural language processing and robust data science libraries to execute complex analysis tasks on various data sources — including local files, web pages, and SQL databases.

---

## 🚀 Features

- **🧠 Multi-Modal Analysis**  
  Seamlessly handles analysis requests for data from multiple sources:
  - **Local Files**: Supports `.csv`, `.xlsx`, and `.parquet`.
  - **Web Scraping**: Scrapes tables from web pages for instant analysis.
  - **SQL Queries**: Executes `DuckDB` queries for fast, in-process querying.

- **🤖 Machine Learning Integration**  
  Supports ML tasks like clustering (KMeans) and regression directly from natural language prompts.

- **📊 Dynamic Visualization**  
  Generates insightful visualizations (scatterplots, regression lines, etc.).

- **🔍 AI-Powered Interface**  
  Uses an LLM to parse user queries and extract task parameters.

- **🚢 Easy Deployment**  
  Configured for deployment on platforms like **Render** or **Replit**.

---

## 🛠️ Getting Started

Follow these steps to set up and run the Data Analyst Agent locally.

### ✅ Prerequisites

- Python 3.8 or higher  
- `pip` (Python package installer)

### 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Sanjoli04/data-analyst-agent.git
cd data-analyst-agent

# Install required Python libraries
pip install -r requirements.txt
```
### 🔐 Setup Environment Variables
- Create a .env file in the root directory with your API key for the LLM service:
```env
GOOGLE_GEMINI_API_KEY="your-api-key-here"
```
### ▶️ Run the Application
- Start the Flask development server:
```python
python main.py
```
### 💻 Technologies Used
- Flask — Web framework

- Pandas — Data analysis

- DuckDB — In-process SQL engine

- Scikit-learn — Machine learning

- Matplotlib & Seaborn — Data visualization

- BeautifulSoup & Requests — Web scraping

- GSAP — Front-end animations

- Render/Cloud Run — Cloud deployment

### 📄 License
`This project is licensed under the MIT License.`

### ❤️ Credits
`Made with ❤️ by Sanjoli Vashisth`
