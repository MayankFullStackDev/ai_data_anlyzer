import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Import our modular logic
import utils
import ai_module

# Load environment variables from .env
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Data Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Custom Styling (CSS) ---
st.markdown("""
<style>
    /* Import modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Elegant Sidebar Design */
    .css-1d391kg {
        background-color: #0F172A;
    }
    
    /* Premium Header styling */
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366F1 0%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    
    /* KPI Card styling */
    .metric-card {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #6366F1;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    
    .metric-warn {
        color: #F59E0B;
    }
    
    .metric-danger {
        color: #EF4444;
    }
    
    .metric-success {
        color: #10B981;
    }
    
    /* Subheaders */
    .section-header {
        font-size: 1.6rem;
        font-weight: 600;
        color: #F8FAFC;
        border-left: 4px solid #6366F1;
        padding-left: 12px;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Chat bubbles styling */
    .chat-bubble {
        padding: 12px 16px;
        border-radius: 16px;
        margin-bottom: 12px;
        max-width: 80%;
    }
    
    .chat-user {
        background-color: #6366F1;
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 2px;
    }
    
    .chat-bot {
        background-color: #334155;
        color: #F8FAFC;
        margin-right: auto;
        border-bottom-left-radius: 2px;
        border: 1px solid #475569;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv("GEMINI_API_KEY", "")
if 'api_key_valid' not in st.session_state:
    st.session_state.api_key_valid = None
if 'automated_insights' not in st.session_state:
    st.session_state.automated_insights = None

# --- API Key Validation Helper ---
def check_key_validity():
    if st.session_state.api_key:
        with st.spinner("Validating Gemini API Key..."):
            is_valid = ai_module.validate_api_key(st.session_state.api_key)
            st.session_state.api_key_valid = is_valid
    else:
        st.session_state.api_key_valid = None

# Trigger validation on startup if an environment variable key is present
if st.session_state.api_key and st.session_state.api_key_valid is None:
    check_key_validity()

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("<div style='text-align: center; padding: 10px;'><h2 style='color:#F8FAFC; margin-bottom:0;'>📊 AI Analyzer</h2><p style='color:#64748B; font-size:0.85rem; margin-top:2px;'>Powered by Google Gemini</p></div>", unsafe_allow_html=True)
    st.markdown("---")
    
    # 1. Navigation Panel
    st.subheader("🧭 Navigation")
    navigation = st.radio(
        "Go to page:",
        ["📊 Dashboard & Upload", "🧹 Clean & Statistics", "📈 Interactive Visualizations", "💬 AI Assistant & Chat"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # 2. Gemini API Configuration
    st.subheader("🔑 Gemini API Key")
    
    # Handle API Key Inputs (Env Variable vs Sidebar override)
    env_exists = bool(os.getenv("GEMINI_API_KEY"))
    key_input = st.text_input(
        "Enter Google Gemini API Key:",
        value=st.session_state.api_key,
        type="password",
        help="If not loaded automatically from a .env file, paste your API Key here. Get one from Google AI Studio.",
        placeholder="AIzaSy..."
    )
    
    # If the user inputted a different key, trigger re-validation
    if key_input != st.session_state.api_key:
        st.session_state.api_key = key_input
        st.session_state.api_key_valid = None
        check_key_validity()
        
    # Render key status notification badges
    if st.session_state.api_key_valid is True:
        st.success("🟢 API Key is Valid")
    elif st.session_state.api_key_valid is False:
        st.error("🔴 Invalid API Key. Please verify.")
    else:
        if env_exists:
            st.info("💡 Loading API Key from .env...")
        else:
            st.warning("⚠️ API Key not configured. Add a .env file or input here.")
            
    st.markdown("---")
    st.markdown("<div style='text-align:center; color:#475569; font-size:0.8rem;'>Antigravity Studio &copy; 2026</div>", unsafe_allow_html=True)

# --- Header Section ---
st.markdown("<h1 class='main-title'>AI Data Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Instantly parse, clean, visualize, and query your datasets using advanced semantic intelligence.</p>", unsafe_allow_html=True)

# --- Navigation Views ---

# SECTION 1: DASHBOARD & UPLOAD
if navigation == "📊 Dashboard & Upload":
    st.markdown("<div class='section-header'>Dataset Upload & Overview</div>", unsafe_allow_html=True)
    
    # Uploader Container
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV, Excel)", 
        type=["csv", "xlsx", "xls"],
        help="Supports files up to 200MB. Supported extensions: .csv, .xlsx, .xls"
    )
    
    if uploaded_file is not None:
        # Load the file if it's different from the loaded one
        if st.session_state.file_name != uploaded_file.name:
            try:
                with st.spinner("Loading dataset..."):
                    df = utils.load_data(uploaded_file, uploaded_file.name)
                    st.session_state.df = df
                    st.session_state.file_name = uploaded_file.name
                    st.session_state.cleaned_df = df.copy() # fallback
                    st.session_state.automated_insights = None # Reset insights
                    st.toast("🎉 Dataset loaded successfully!")
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.df = None
                st.session_state.file_name = None
                
    # If dataset is loaded, show metadata & preview
    if st.session_state.df is not None:
        df = st.session_state.df
        summary = utils.get_dataset_summary(df)
        
        # 1. Metric Display Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Rows Count</div>
                <div class='metric-value'>{summary['rows']:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Columns Count</div>
                <div class='metric-value'>{summary['columns']:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            missing_val = summary['missing_cells']
            color_class = "metric-success" if missing_val == 0 else "metric-danger" if missing_val > 50 else "metric-warn"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Missing Cells</div>
                <div class='metric-value {color_class}'>{missing_val:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            dups = summary['duplicate_rows']
            color_class = "metric-success" if dups == 0 else "metric-warn"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Duplicate Rows</div>
                <div class='metric-value {color_class}'>{dups:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown(" ")
        
        # 2. Interactive Dataset Preview
        st.subheader("📋 Dataset Preview (First 5 Rows)")
        st.dataframe(df.head(5), use_container_width=True)
        
        # 3. Structural Metadata Profile
        st.subheader("🧬 Column Structure & Metadata")
        dtypes_df = pd.DataFrame(summary['dtypes'])
        dtypes_df.columns = ["Column Name", "Data Type", "Non-Null Counts", "Unique Values"]
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)
        
    else:
        # Prompt landing page
        st.markdown("""
        <div style='background-color: #1E293B; padding: 40px; border-radius: 16px; border: 1px dashed #475569; text-align: center; margin-top:20px;'>
            <h3 style='color: #F8FAFC; margin-bottom: 12px;'>No Dataset Uploaded</h3>
            <p style='color: #94A3B8; max-width: 600px; margin: 0 auto 24px auto; line-height: 1.6;'>
                To start analyzing, upload your CSV or Excel spreadsheet using the uploader panel above.
                The app will automatically extract structure, parse missing entries, calculate summary statistics, 
                and enable Gemini-powered natural language conversation.
            </p>
            <div style='display: inline-block; background-color: #334155; color: #F8FAFC; padding: 10px 20px; border-radius: 8px; font-weight:600;'>
                Supports CSV, XLSX, XLS
            </div>
        </div>
        """, unsafe_allow_html=True)

# SECTION 2: CLEAN & STATISTICS
elif navigation == "🧹 Clean & Statistics":
    st.markdown("<div class='section-header'>Data Cleaning & Summary Statistics</div>", unsafe_allow_html=True)
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        col_clean, col_stats = st.columns([1, 1])
        
        # Left Panel: Data Cleaning & Missing values report
        with col_clean:
            st.subheader("🧹 Data Cleaning Workspace")
            
            missing_report = utils.get_missing_values_report(df)
            
            # Show missing report
            if missing_report['Missing Count'].sum() > 0:
                st.warning(f"⚠️ Your dataset contains {missing_report['Missing Count'].sum()} missing cells.")
                st.dataframe(missing_report[missing_report['Missing Count'] > 0], use_container_width=True)
                
                # Cleaning form
                st.markdown("### Choose Cleaning Strategy")
                clean_strategy = st.selectbox(
                    "Select Strategy:",
                    [
                        "Do nothing (Keep missing values)",
                        "Drop rows with missing values",
                        "Impute missing values (Mean for numeric, Mode for categorical)",
                        "Impute missing values (Median for numeric, Mode for categorical)",
                        "Impute all columns with Mode"
                    ]
                )
                
                if st.button("🧼 Apply Cleaning Strategy", use_container_width=True):
                    strategy_code = 'none'
                    if "Drop" in clean_strategy:
                        strategy_code = 'drop'
                    elif "Mean" in clean_strategy:
                        strategy_code = 'mean'
                    elif "Median" in clean_strategy:
                        strategy_code = 'median'
                    elif "Mode" in clean_strategy:
                        strategy_code = 'mode'
                        
                    if strategy_code != 'none':
                        with st.spinner("Cleaning dataset..."):
                            df_cleaned = utils.clean_missing_values(df, strategy_code)
                            st.session_state.df = df_cleaned
                            st.toast("✨ Cleaning applied and dataset updated!")
                            st.rerun()
            else:
                st.success("🎉 Excellent! Your dataset contains no missing values.")
                
            # Allow Downloading the current active dataset
            st.markdown("### Export Dataset")
            st.markdown("You can download the current state of your dataset as a CSV file.")
            
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Dataset (CSV)",
                data=csv_data,
                file_name=f"processed_{st.session_state.file_name or 'dataset.csv'}",
                mime="text/csv",
                use_container_width=True
            )
            
        # Right Panel: Descriptive Statistics
        with col_stats:
            st.subheader("📈 Summary Statistics")
            
            stats_df = utils.get_summary_statistics(df)
            
            if not stats_df.empty:
                st.markdown("Detailed summaries for numeric variables:")
                st.dataframe(stats_df, use_container_width=True)
                
                # Mini info guide
                with st.expander("📚 What do these statistics mean?"):
                    st.markdown("""
                    - **Mean**: The mathematical average value.
                    - **Median**: The exact center value (50th percentile) when sorted. Resistant to extreme outliers.
                    - **Mode**: The most frequently occurring value in the column.
                    - **Variance**: Measures the dispersion/spread of values from the mean.
                    - **Standard Deviation**: The average distance of values from the mean. High standard deviation indicates wider spread.
                    - **Minimum / Maximum**: The absolute range limits of the column values.
                    """)
            else:
                st.info("No numeric columns found in the dataset to calculate statistical aggregates.")
                
    else:
        st.warning("⚠️ Please upload a dataset first in the **Dashboard & Upload** section.")

# SECTION 3: INTERACTIVE VISUALIZATIONS
elif navigation == "📈 Interactive Visualizations":
    st.markdown("<div class='section-header'>Data Visualization Workspace</div>", unsafe_allow_html=True)
    
    if st.session_state.df is not None:
        df = st.session_state.df
        columns = df.columns.tolist()
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        # 1. Visualization selection
        chart_type = st.selectbox(
            "Select Chart Type:",
            ["Bar Chart", "Line Chart", "Histogram", "Correlation Heatmap"]
        )
        
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        plt.style.use('dark_background')  # Cohesive dark theme for plots
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#1E293B')
        
        # Render dynamic parameters based on chart type
        plot_ready = False
        
        if chart_type == "Bar Chart":
            col_x, col_y, col_c = st.columns(3)
            with col_x:
                x_col = st.selectbox("X-Axis (Categorical/Numeric):", columns)
            with col_y:
                y_col = st.selectbox("Y-Axis (Numeric preferred):", numeric_columns)
            with col_c:
                palette_choice = st.selectbox("Color Palette:", ["viridis", "magma", "rocket", "mako", "crest"])
                
            if x_col and y_col:
                with st.spinner("Generating Bar Chart..."):
                    # Aggregate duplicates if any for cleaner plotting
                    agg_df = df.groupby(x_col)[y_col].mean().reset_index().sort_values(by=y_col, ascending=False).head(15)
                    sns.barplot(data=agg_df, x=x_col, y=y_col, ax=ax, palette=palette_choice)
                    ax.set_title(f"Average of {y_col} by {x_col} (Top 15)", fontsize=12, color='#F8FAFC', pad=15)
                    ax.set_xlabel(x_col, fontsize=10, color='#94A3B8')
                    ax.set_ylabel(y_col, fontsize=10, color='#94A3B8')
                    plt.xticks(rotation=45, ha='right', color='#94A3B8')
                    plt.yticks(color='#94A3B8')
                    ax.grid(color='#334155', linestyle='--', linewidth=0.5)
                    plot_ready = True
                    
        elif chart_type == "Line Chart":
            col_x, col_y, col_c = st.columns(3)
            with col_x:
                x_col = st.selectbox("X-Axis (Ordered numeric/Time/Category):", columns)
            with col_y:
                y_col = st.selectbox("Y-Axis (Numeric):", numeric_columns)
            with col_c:
                line_color = st.color_picker("Line Color:", "#6366F1")
                
            if x_col and y_col:
                with st.spinner("Generating Line Chart..."):
                    # Sort by X-axis to make line chart sequential
                    sorted_df = df.sort_values(by=x_col)
                    sns.lineplot(data=sorted_df, x=x_col, y=y_col, ax=ax, color=line_color, marker='o', linewidth=2)
                    ax.set_title(f"{y_col} vs {x_col}", fontsize=12, color='#F8FAFC', pad=15)
                    ax.set_xlabel(x_col, fontsize=10, color='#94A3B8')
                    ax.set_ylabel(y_col, fontsize=10, color='#94A3B8')
                    plt.xticks(rotation=45, ha='right', color='#94A3B8')
                    plt.yticks(color='#94A3B8')
                    ax.grid(color='#334155', linestyle='--', linewidth=0.5)
                    plot_ready = True
                    
        elif chart_type == "Histogram":
            col_x, col_bins, col_color = st.columns(3)
            with col_x:
                x_col = st.selectbox("Select Numeric Column:", numeric_columns)
            with col_bins:
                bins_val = st.slider("Number of Bins:", min_value=5, max_value=100, value=30)
            with col_color:
                hist_color = st.color_picker("Histogram Color:", "#EC4899")
                
            if x_col:
                with st.spinner("Generating Histogram..."):
                    sns.histplot(data=df, x=x_col, bins=bins_val, ax=ax, color=hist_color, kde=True, edgecolor='#1E293B')
                    ax.set_title(f"Distribution of {x_col}", fontsize=12, color='#F8FAFC', pad=15)
                    ax.set_xlabel(x_col, fontsize=10, color='#94A3B8')
                    ax.set_ylabel("Frequency", fontsize=10, color='#94A3B8')
                    plt.xticks(color='#94A3B8')
                    plt.yticks(color='#94A3B8')
                    ax.grid(color='#334155', linestyle='--', linewidth=0.5)
                    plot_ready = True
                    
        elif chart_type == "Correlation Heatmap":
            if len(numeric_columns) >= 2:
                show_values = st.checkbox("Show Correlation Coefficients (Numbers)", value=True)
                with st.spinner("Generating Correlation Heatmap..."):
                    corr_matrix = df[numeric_columns].corr()
                    sns.heatmap(
                        corr_matrix, 
                        annot=show_values, 
                        cmap="coolwarm", 
                        vmin=-1, 
                        vmax=1, 
                        ax=ax, 
                        cbar_kws={'label': 'Correlation Coefficient'}
                    )
                    ax.set_title("Numeric Correlation Heatmap", fontsize=12, color='#F8FAFC', pad=15)
                    plt.xticks(rotation=45, ha='right', color='#94A3B8')
                    plt.yticks(color='#94A3B8')
                    plot_ready = True
            else:
                st.info("At least two numeric columns are required to draw a correlation heatmap matrix.")
                
        # Draw plot if ready
        if plot_ready:
            st.pyplot(fig)
            plt.close(fig) # Prevent memory leaks
            
    else:
        st.warning("⚠️ Please upload a dataset first in the **Dashboard & Upload** section.")

# SECTION 4: AI ASSISTANT & CHAT
elif navigation == "💬 AI Assistant & Chat":
    st.markdown("<div class='section-header'>Google Gemini AI Assistant</div>", unsafe_allow_html=True)
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Verify Key
        if not st.session_state.api_key or st.session_state.api_key_valid is not True:
            st.error("🔑 Google Gemini API Key required. Please enter a valid API Key in the sidebar input box to activate the AI panel.")
            
        else:
            # Layout divided: Automated Insights Accordion + Conversation
            col_insight, col_chat = st.columns([1, 2])
            
            with col_insight:
                st.subheader("⚡ Automated Insights")
                st.markdown("Let Gemini scan your schema and statistics profile to extract underlying trends, patterns, and anomalies.")
                
                # Check if insights already generated
                if st.session_state.automated_insights:
                    st.markdown(st.session_state.automated_insights)
                    if st.button("🔄 Regenerate Insights", use_container_width=True):
                        st.session_state.automated_insights = None
                        st.rerun()
                else:
                    if st.button("✨ Generate AI Insights Profile", use_container_width=True):
                        with st.spinner("Gemini is examining your statistics profile..."):
                            try:
                                insights = ai_module.generate_automated_insights(df, st.session_state.api_key)
                                st.session_state.automated_insights = insights
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to generate insights: {e}")
                                
            with col_chat:
                st.subheader("💬 Chat with your Dataset")
                st.markdown("Ask natural language questions like *'which variable has the highest standard deviation?'* or *'summarize the key relationships'*.")
                
                # Custom container for chat messages
                chat_container = st.container(height=400)
                
                with chat_container:
                    # Clear History Button
                    if len(st.session_state.chat_history) > 0:
                        col_space, col_clear = st.columns([4, 1])
                        with col_clear:
                            if st.button("🧹 Clear Chat"):
                                st.session_state.chat_history = []
                                st.rerun()
                                
                    # Welcome Message
                    if len(st.session_state.chat_history) == 0:
                        st.chat_message("assistant").write("Hello! I've loaded your dataset's summary profile. Ask me any analytical question, and I'll help you extract the answers!")
                        
                    # Display history messages
                    for msg in st.session_state.chat_history:
                        st.chat_message(msg["role"]).write(msg["content"])
                        
                # Chat Input Box
                user_input = st.chat_input("What would you like to know about your dataset?")
                
                if user_input:
                    # Immediately show user question in UI
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    chat_container.chat_message("user").write(user_input)
                    
                    # Call Gemini API
                    with st.spinner("Thinking..."):
                        try:
                            # Pass chat history (excluding the current user question which we've already appended)
                            reply = ai_module.chat_with_dataset(
                                df=df,
                                user_query=user_input,
                                chat_history=st.session_state.chat_history[:-1],
                                api_key=st.session_state.api_key
                            )
                            # Append bot reply and show in UI
                            st.session_state.chat_history.append({"role": "assistant", "content": reply})
                            chat_container.chat_message("assistant").write(reply)
                            st.rerun()
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                            
    else:
        st.warning("⚠️ Please upload a dataset first in the **Dashboard & Upload** section.")
