import pandas as pd
import numpy as np

def load_data(file_path_or_buffer, file_name: str) -> pd.DataFrame:
    """
    Safely loads CSV or Excel files. Returns a Pandas DataFrame.
    Raises ValueError with a user-friendly message on failure.
    """
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_path_or_buffer)
            return df
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path_or_buffer)
            return df
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel (.xlsx, .xls) file.")
    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        raise ValueError(f"Error reading the file: {str(e)}")

def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Computes basic metadata metrics about the dataset.
    """
    num_rows, num_cols = df.shape
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    
    dtypes_list = []
    for col in df.columns:
        dtypes_list.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "non_null": int(df[col].notna().sum()),
            "unique": int(df[col].nunique())
        })
        
    return {
        "rows": num_rows,
        "columns": num_cols,
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "dtypes": dtypes_list
    }

def get_missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies columns with missing values and returns count and percentage.
    """
    total_missing = df.isna().sum()
    percent_missing = (df.isna().sum() / len(df)) * 100
    
    report = pd.DataFrame({
        'Missing Count': total_missing,
        'Percentage (%)': percent_missing.round(2)
    })
    
    # Filter to show only columns that actually have missing values or show all
    return report

def clean_missing_values(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """
    Cleans missing values in the DataFrame according to the selected strategy.
    Strategies:
      - 'drop': Drop rows with any missing values.
      - 'mean': Impute numeric columns with mean, categorical/object with mode.
      - 'median': Impute numeric columns with median, categorical/object with mode.
      - 'mode': Impute all columns with their mode.
    """
    df_cleaned = df.copy()
    
    if strategy == 'drop':
        df_cleaned = df_cleaned.dropna()
    elif strategy in ['mean', 'median', 'mode']:
        for col in df_cleaned.columns:
            if df_cleaned[col].isna().sum() == 0:
                continue
                
            # Determine data type category
            is_numeric = pd.api.types.is_numeric_dtype(df_cleaned[col])
            
            # Find mode (safe fallback if empty)
            mode_series = df_cleaned[col].mode()
            mode_val = mode_series.iloc[0] if not mode_series.empty else np.nan
            
            if strategy == 'mean' and is_numeric:
                mean_val = df_cleaned[col].mean()
                # If mean_val is NaN (e.g. all NaNs), fallback to 0 or drop
                df_cleaned[col] = df_cleaned[col].fillna(mean_val if not pd.isna(mean_val) else 0)
            elif strategy == 'median' and is_numeric:
                median_val = df_cleaned[col].median()
                df_cleaned[col] = df_cleaned[col].fillna(median_val if not pd.isna(median_val) else 0)
            elif strategy == 'mode' or not is_numeric:
                # For categorical columns, or when mode strategy is selected, use mode
                if not pd.isna(mode_val):
                    df_cleaned[col] = df_cleaned[col].fillna(mode_val)
                else:
                    # Fallback if no mode is available (e.g., all NaN column)
                    df_cleaned[col] = df_cleaned[col].fillna("Missing")
                    
    return df_cleaned

def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes summary statistics for all numeric columns:
    Mean, Median, Mode, Variance, Standard Deviation, Min, Max.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return pd.DataFrame()
        
    stats_data = {}
    for col in numeric_cols:
        col_data = df[col].dropna()
        if col_data.empty:
            continue
            
        mean_val = col_data.mean()
        median_val = col_data.median()
        
        mode_series = col_data.mode()
        mode_val = mode_series.iloc[0] if not mode_series.empty else np.nan
        
        variance_val = col_data.var()
        std_val = col_data.std()
        min_val = col_data.min()
        max_val = col_data.max()
        
        stats_data[col] = {
            "Mean": round(mean_val, 4),
            "Median": round(median_val, 4),
            "Mode": round(mode_val, 4) if not pd.isna(mode_val) else "N/A",
            "Variance": round(variance_val, 4) if not pd.isna(variance_val) else 0.0,
            "Standard Deviation": round(std_val, 4) if not pd.isna(std_val) else 0.0,
            "Minimum": round(min_val, 4),
            "Maximum": round(max_val, 4)
        }
        
    return pd.DataFrame(stats_data).T

def generate_data_context_prompt(df: pd.DataFrame) -> str:
    """
    Generates a structured text summary of the dataset to be used as context for the AI.
    This provides rich context without sending the entire raw dataset.
    """
    rows, cols = df.shape
    
    # 1. Schema Info
    schema_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = df[col].isna().sum()
        unique_count = df[col].nunique()
        
        # Sample categories if the column has few unique values
        sample_str = ""
        if unique_count < 10:
            uniques = df[col].dropna().unique()
            sample_str = f" | Unique Values: {list(uniques)}"
        elif pd.api.types.is_numeric_dtype(df[col]):
            sample_str = f" | Range: [{df[col].min()}, {df[col].max()}]"
            
        schema_info.append(f"- Column: '{col}' ({dtype}) | Missing: {null_count} | Unique: {unique_count}{sample_str}")
    
    schema_str = "\n".join(schema_info)
    
    # 2. Descriptive Stats Summary
    numeric_stats = df.describe(include='all').to_string() if not df.empty else "No descriptive statistics."
    
    # 3. Data Preview (First 5 and Last 5 rows)
    head_preview = df.head(5).to_string()
    tail_preview = df.tail(5).to_string()
    
    # Assemble the context prompt
    context = f"""[DATASET INFORMATION]
- Dimensions: {rows} rows, {cols} columns
- Total Duplicated Rows: {df.duplicated().sum()}

[SCHEMA & COLUMNS PROFILE]
{schema_str}

[SUMMARY STATISTICS]
{numeric_stats}

[DATA PREVIEW - FIRST 5 ROWS]
{head_preview}

[DATA PREVIEW - LAST 5 ROWS]
{tail_preview}
"""
    return context
