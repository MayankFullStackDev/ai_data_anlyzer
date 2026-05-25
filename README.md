# AI Data Analyzer

A premium, interactive, and AI-powered **Data Analyzer** web application built using Python, Streamlit, Pandas, Seaborn, and the Google Gemini API.

This application allows users to upload, clean, and visualize structured datasets (CSVs and Excel spreadsheets), and engage in a natural language chat to extract statistical insights, trends, patterns, and anomalies.

---

## 🌟 Key Features

1. **Upload & Parse Dashboard**: Secure upload support for CSV and Excel files. Computes visual metric cards for quick statistics (rows, columns, duplicate entries, missing values) and displays an interactive data preview with full schema profiles.
2. **Interactive Data Cleaning**: Interactive missing-value counts, options to impute (mean/median for numeric, mode for categorical), option to drop records with null values, and direct data export/download of the cleaned CSV.
3. **Advanced Summarization & Statistics**: Computes detailed numeric metrics including Mean, Median, Mode, Variance, and Standard Deviation.
4. **Dynamic Data Visualizations**: Elegant, dark-themed charts styled using Matplotlib and Seaborn, supporting:
   - **Bar Charts** (aggregated, sorted, and filtered lists)
   - **Line Charts** (sequential trend tracking)
   - **Histograms** (frequency distributions with adjustable bins and KDE curves)
   - **Correlation Heatmaps** (numeric relationship matrices)
5. **Gemini AI Integration**:
   - **Automated Insights**: Scans data structure and summary statistics to write immediate analytical reports detailing trends, patterns, and anomalies.
   - **Conversational Chatbot**: Powered by `gemini-1.5-flash` with full memory context, enabling users to ask freeform analytical questions about their spreadsheet.
6. **Robust Error Boundaries**: Gracefully handles API key validation, missing columns, corrupt spreadsheets, and mismatched file formats.

---

## 🛠️ Tech Stack

- **Frontend Interface**: Streamlit
- **Data Engineering**: Pandas, OpenPyXL (Excel integration)
- **Visualizations**: Seaborn, Matplotlib
- **AI Engine**: Google Gemini API (`google-generativeai` SDK)

---

## 📁 Directory Structure

```text
AI_App/
│
├── main.py            # Streamlit dashboard entry point and UI architecture
├── utils.py           # Core data loading, cleaning, and statistical metrics
├── ai_module.py       # Google Gemini API connector and native chat session handler
├── requirements.txt   # Third-party Python dependencies
└── README.md          # Setup and operation instructions
```

---







## 🚀 Getting Started (Local Setup)

Follow these simple steps to run the application locally on your system:

### 1. Prerequisites
Ensure you have **Python 3.9** or higher installed on your computer.

### 2. Clone or Navigate to the Directory
Open your terminal or PowerShell and change directory into the project folder:
```bash
cd C:\Users\hp\OneDrive\Documents\DATA-SCIENCE\AI_App
```

### 3. Create a Virtual Environment (Recommended)
Set up a clean virtual environment to manage dependencies:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
Install all required packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 5. Configure the Google Gemini API Key
Create a `.env` file in the root directory:
```bash
# Windows PowerShell
New-Item -Path . -Name ".env" -ItemType "file"
```

Open the `.env` file in a text editor and add your Gemini API key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```
> **Note**: You can get a free API Key from [Google AI Studio](https://aistudio.google.com/).
> *If you do not set this in the `.env` file, you can also enter the API key directly in the application's sidebar interface during runtime.*

### 6. Run the Application
Launch the Streamlit dev server:
```bash
streamlit run main.py
```
A browser tab will automatically open at `http://localhost:8501`. If it doesn't, copy and paste the URL from the command line interface.


# AI Data Analyzer

## Features
- Upload CSV files
- Data cleaning
- Data visualization
- AI analysis using Gemini

## Tech Stack
- Python
- Streamlit
- Pandas
- Google Gemini API

## How to run
pip install -r requirements.txt
streamlit run app.py