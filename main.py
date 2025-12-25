import streamlit as st
import pandas as pd
import google.generativeai as genai
import helper
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
import preprocessing
import os
from dotenv import load_dotenv

your_api_key = st.sidebar.text_input(
    "Enter your Gemini 2.5 API key",
    type="password"
)

genai.configure(api_key=your_api_key)

model = genai.GenerativeModel("gemini-2.5-flash")

# ---- CHAT SESSION INIT ----
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": ["You are a helpful data analyst. Explain datasets."]
            }
        ]
    )
if "tab3_initialized" not in st.session_state:
    st.session_state.show_duplicates = False
    st.session_state.tab3_initialized = True
if "show_duplicates" not in st.session_state:
    st.session_state.show_duplicates = False

if "duplicates_dropped" not in st.session_state:
    st.session_state.duplicates_dropped = False
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "chatbot_active" not in st.session_state:
    st.session_state.chatbot_active = False

if "remember_response" not in st.session_state:
    st.session_state.remember_response = model.start_chat(history=[])

if "key_findings" not in st.session_state:
    st.session_state.key_findings = None
def fill_zero(df,selected_column_name):
    df[selected_column_name]=df[selected_column_name].fillna(0)

    return df
def check(df, selected_radio, selected_column_name):
    if selected_radio == 'mean':
        df[selected_column_name] = df[selected_column_name].fillna(
            df[selected_column_name].mean()
        )
    elif selected_radio == 'median':
        df[selected_column_name] = df[selected_column_name].fillna(
            df[selected_column_name].median()
        )
    elif selected_radio == 'mode':
        df[selected_column_name] = df[selected_column_name].fillna(
            df[selected_column_name].mode()[0]
        )
    return df

# ---- SIDEBAR ----
st.sidebar.title("📂 Data Controls")
st.sidebar.caption("Upload your dataset, understand it visually, chat with it, and clean it easily.")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])


# ---- RESET SESSION STATE FOR NEW FILE ----
if uploaded_file:
    # Always reload df if a new file is uploaded
    if "uploaded_file_name" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.uploaded_file_name = uploaded_file.name

        # Clear previous file's session data
        st.session_state.messages = []
        st.session_state.key_findings = None
        st.session_state.chatbot_active = False
        st.session_state.show_missing = False
        st.session_state.show_duplicates = False

        if "chat" in st.session_state:
            del st.session_state.chat
        if "remember_response" in st.session_state:
            del st.session_state.remember_response

        # Reinitialize chat sessions
        st.session_state.chat = model.start_chat(
            history=[{"role": "user", "parts": ["You are a helpful data analyst. Explain datasets."]}]
        )
        st.session_state.remember_response = model.start_chat(history=[])

        # Load the new file into df
        st.session_state.df = pd.read_csv(uploaded_file)

    # Use df
    df = st.session_state.df
# ---- DATASET + AUTO EXPLANATION ----
if uploaded_file:
    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(uploaded_file)

    df = st.session_state.df

    with st.expander("📋 View Your Dataset"):
        st.caption("Preview of your uploaded data")
        st.dataframe(df)

    first_input = f"""
    Role: You are a data science teacher.
    Task: Explain the given dataset in a very simple and beginner-friendly way.
    Guidelines:
    Assume I have no prior knowledge of data analysis.
    Use easy words and short sentences.
    Explain what each column means and why it is important.
    Explain what each row represents.
    Give real-life examples so I can relate to the data.
    Do not rush. Teach step-by-step like a teacher.

    Format your answer as:
    What this dataset is about (overall idea)
    Explanation of each column (one by one)
    Example using 1–2 sample rows
    What we can learn from this dataset
    and mention at top as note in bold i am taking "some part of data only" not the entire datase
    Dataset preview:
    {df.head(5).to_string()}
    data types of columns:{df.dtypes}
    statistical summary:{df.describe}
    total no of missing values:{df.isnull().sum().sum()}
    missing values by column:{df.isnull().sum()}
    shape of the dataset:{df.shape}
    """

    if len(st.session_state.messages) == 0:
        assistant_reply_on_ds = st.session_state.chat.send_message(first_input)
        st.session_state.messages.append(("assistant", assistant_reply_on_ds.text))

    # CHAT BOT
    if st.sidebar.button("💬 Open Chatbot", use_container_width=True):
        st.session_state.chatbot_active = True


    if st.sidebar.button("🏠 Back to Dashboard", use_container_width=True):
        st.session_state.chatbot_active = False

    if st.session_state.chatbot_active:
        st.title("💬 Chat with Your Data")
        st.caption("Ask questions and get insights from your dataset in real-time")
        st.markdown("---")
        
        for role, msg in st.session_state.messages:
            with st.chat_message(role.lower()):
                st.markdown(msg)

        # ---- USER INPUT ----
        if uploaded_file:
            user_input = st.chat_input("Ask about your data...")
            if user_input:
                st.session_state.messages.append(("user", user_input))
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.spinner("Analyzing data..."):
                    response = st.session_state.chat.send_message(
                        f"""
                    You are a helpful data analysis assistant.

                    Task:
                    - Carefully analyze the dataset provided below.
                    - Answer the user's question using ONLY the information from this dataset.
                    - If the answer cannot be found directly from the dataset, clearly say so in simple words.
                    - Also if user asks question about anything beyond the dataset then clearly say I am trained to give answers from dataset only.

                    Guidelines:
                    - Explain things in very simple language.
                    - Assume the user is a beginner.
                    - Use short sentences and clear explanations.
                    - If numbers are involved, explain what they mean in real life.
                    - Avoid complex technical terms. If used, explain them simply.
                    - If helpful, give a small example from the dataset.

                    Dataset:
                    {df.head(5).to_string()}
                    data types of columns:{df.dtypes}
                    dataset shape:{df.shape}
                    {df.describe}
                    duplicated records:{df.duplicated()}
                    total no of duplicated records:{df.duplicated().sum()}
                    total no. missing values:
                    {df.isnull().sum().sum()}

                    User Question:
                    {user_input}
                    """
                    )

                st.session_state.messages.append(("assistant", response.text))
                with st.chat_message("assistant"):
                    st.markdown(response.text)

    else:
        tab1, tab2, tab3 = st.tabs(["📊 EDA", "💡 Key Findings", "🧹 Data Cleaning"])

        with tab1:
            # EDA
            st.header("📊 Exploratory Data Analysis (EDA)")
            st.caption("Basic structure, data quality, and column-level insights")
            st.markdown("<div style='margin-top: 30px'></div>", unsafe_allow_html=True)

            shape, total_missing_vals, total_duplicate_rows, perc_of_missing = helper.dataset_basic_info(df)

            st.subheader("📈 Dataset Overview")
            st.caption("Quick summary of your dataset's structure and quality")
            sub_col1, sub_col2, sub_col3, sub_col4 = st.columns(4)
            with sub_col1:
                st.subheader("📐 Shape of dataset:")
                st.subheader(str(shape))
            with sub_col2:
                st.subheader("❓ Sum of missing values:")
                st.subheader(total_missing_vals)
            with sub_col3:
                st.subheader("📉 Percentage of missing data:")
                st.subheader(round(perc_of_missing, 2))
            with sub_col4:
                st.subheader("🔄 Total Duplicate Rows:")
                st.subheader(total_duplicate_rows)

            column_names, describe = helper.ds_structure_and_stats(df)

            st.markdown("<div style='margin-top: 20px'></div>", unsafe_allow_html=True)
            st.subheader("🔍 Data Structure & Statistics")
            st.caption("Detailed breakdown of column types and statistical measures")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### 📝 Data Types of Columns")
                column_names = column_names.reset_index()
                column_names.columns = ['Column Name', 'Data type']
                st.dataframe(column_names)

            with col2:
                st.markdown("##### 📊 Basic Statistics")
                st.dataframe(describe)

            st.markdown("<div style='margin-top: 30px'></div>", unsafe_allow_html=True)
            st.subheader("❓ Missing Values Analysis")
            st.caption("Identify and visualize missing data patterns across columns")
            col_missing_vals, col_perc_missing_vals = helper.column_wise_vals(df)
            new_df = pd.DataFrame({"Missing Values": col_missing_vals, "Percentage of Missing Values": col_perc_missing_vals})
            st.dataframe(new_df)
            st.markdown("<div style='margin-top: 30px'></div>", unsafe_allow_html=True)
            fig, ax = plt.subplots()
            new_df.plot(kind='barh', ax=ax)
            st.pyplot(fig)

            st.markdown("<div style='margin-top: 30px'></div>", unsafe_allow_html=True)
            st.subheader("🎯 Unique Values Analysis")
            st.caption("Explore data diversity and cardinality per column")
            unique_df = helper.unique_vals(df)
            st.dataframe(unique_df)
            st.markdown("<div style='margin-top: 30px'></div>", unsafe_allow_html=True)
            st.bar_chart(unique_df.set_index("Column Name")["Unique Values Count"])

            st.markdown("---")
            st.markdown("<div style='margin-top: 30px'></div>", unsafe_allow_html=True)
            st.subheader("📄 Comprehensive Data Report")
            st.caption("Generate an in-depth automated analysis report")
            if st.button("🚀 Generate Detailed Report", use_container_width=True):
                profile = ProfileReport(
                    df,
                    title="Dataset Report",
                    minimal=True
                )
                st.components.v1.html(profile.to_html(), height=1000, scrolling=True)

        with tab2:
            # KEY FINDINGS
            st.header("💡 Key Findings & Insights")
            st.caption("AI-powered analysis highlighting patterns, anomalies, and actionable recommendations")
            st.markdown("---")
            
            if st.session_state.key_findings is None:
                prompt = f"""
            You are a senior data analyst.

            Rules:
            - Keep each point to 1–2 short lines.
            - Use bullet points only.
            - Focus on what matters most.

            Analyze this dataset and give insights:

            Dataset info:
            - Shape: {df.shape}
            - statistical summary:{df.describe}
            - data types of columns:{df.dtypes}
            - total no of missing values:{df.isnull().sum().sum()}
            - missing values:{df.isnull().sum()}
            - Columns: {list(df.columns)}
            - Sample rows:
            {df.head(2).to_string()}

            Provide:
            1. Top 3 unusual patterns or outliers
            2. Key data quality issues
            3. 2–3 relationships or correlations
            4. 3 actionable business insights
            5. One-line final recommendation
            """

                with st.spinner("🔍 Generating key findings..."):
                    response = st.session_state.remember_response.send_message(prompt)

                st.session_state.key_findings = response.text

            st.markdown(st.session_state.key_findings)

        with tab3:
            if "show_missing" not in st.session_state:
                st.session_state.show_missing = False

            st.header("🧹 Clean Your Data")
            st.caption("Handle missing values, remove duplicates, and prepare your dataset for analysis")

            st.markdown("---")
            st.subheader("❓ Handle Missing Values")
            st.caption("Fill or remove null values to improve data quality")
            if st.button("🔍 Analyze Missing Values", use_container_width=True):
                st.session_state.show_missing = True
            
            if st.session_state.show_missing:
                    
                    col_missing_vals,col_perc_missing_vals=helper.column_wise_vals(df)
                    column_names,describe=helper.ds_structure_and_stats(df)

                    col1,col2=st.columns(2)
                    with col1:
                        st.markdown("<div style='margin-top: 20px'></div>", unsafe_allow_html=True)
                        st.markdown("##### 📋 Missing Values Summary")
                        col_missing_vals=col_missing_vals.reset_index()
                        col_missing_vals.columns=['col_name','missing_vals']
                        column_names=column_names.reset_index()
                        column_names.columns=['col_name','data_type']

                        combined_df = pd.merge(column_names, col_missing_vals,on="col_name")

                        st.dataframe(combined_df)
                    with col2:
                        st.markdown("##### 📊 Visual Breakdown")
                        fig, ax=plt.subplots()

                        ax.barh(combined_df['col_name'],combined_df['missing_vals'])
                        ax.set_xlabel("Number of Missing Values")
                        ax.set_ylabel("Column Name")

                        # ✅ Title
                        ax.set_title("Column-wise Missing Values")

                        # Optional: improve readability
                        ax.grid(axis='x', linestyle='--', alpha=0.6)

                        st.pyplot(fig)

                    st.markdown("<div style='margin-top: 20px'></div>", unsafe_allow_html=True)
                    st.markdown("##### 🎯 Select Column to Clean")
                    selected_column_name = st.selectbox("Select a column", col_missing_vals['col_name'])

                    if selected_column_name:
                        st.caption(f"Showing all rows with missing values in '{selected_column_name}'")
                        filtered_df = df[df[selected_column_name].isnull()]
                        st.dataframe(filtered_df)

                        
                        st.markdown(
                            """
                            <style>
                            .stButton>button {
                                width: 100%;
                                height: 40px;
                            }
                            </style>
                            """,
                            unsafe_allow_html=True
                        )

                        st.markdown("##### ⚙️ Choose Fill Method")
                        selected_option=st.selectbox("Fill values by:",['mean median and mode','zero'])

                        if selected_option=='mean median and mode':

                            st.warning("⚠️ Mean & Median must have numerical values!")
                            selected_radio = st.radio("Select a method:", ['mean', 'median', 'mode'])

                            if st.button("✅ Fill Missing Values", use_container_width=True):
                                st.session_state.df = check(st.session_state.df, selected_radio, selected_column_name)

                                st.success(f"✅ Filled missing values using {selected_radio}!")
                                st.write("Updated Missing Values Count:")
                                st.write(st.session_state.df.isnull().sum())

                                # Convert DataFrame to CSV string
                                csv = st.session_state.df.to_csv(index=False)

                                # Download button
                                st.download_button(
                                    label="📥 Download Cleaned Dataset",
                                    data=csv,
                                    file_name="cleaned_data.csv",
                                    mime="text/csv"
                                )


                        if selected_option=='zero':
                            if st.button("✅ Fill Missing Values"):
                                st.session_state.df=fill_zero(df,selected_column_name)

                                st.success("✅ Filled missing values with zero!")
                                st.write("Updated Missing Values Count:")
                                st.write(st.session_state.df.isnull().sum())

                                    # Convert DataFrame to CSV string
                                csv = st.session_state.df.to_csv(index=False)

                                    # Download button
                                st.download_button(
                                        label="📥 Download Cleaned Dataset",
                                        data=csv,
                                        file_name="filled_zero.csv",
                                        mime="text/csv"
                                )
            
            st.markdown("---")
            
            st.subheader("🔄 Handle Duplicate Values")
            st.caption("Identify and remove duplicate rows from your dataset")
            if st.button("🔍 Find Duplicates", use_container_width=True):
                st.session_state.show_duplicates = True
                st.session_state.duplicates_dropped = False

                

            if st.session_state.show_duplicates is True:

                col1,col2=st.columns(2)
                with col1:
                    st.markdown("<div style='margin-top: 50px'></div>", unsafe_allow_html=True)
                    st.markdown("##### 📋 Duplicate Rows")
                    st.dataframe(st.session_state.df[st.session_state.df.duplicated()])
                    
                    
                
                with col2:
                    st.markdown("##### 📊 Duplicate Distribution")
                    duplicate_vals = st.session_state.df.duplicated().value_counts()

                    labels = ["Unique" if i == False else "Duplicate" for i in duplicate_vals.index]

                    fig, ax = plt.subplots(figsize=(3, 3))

                    ax.pie(
                        duplicate_vals.values,
                        labels=labels,
                        autopct="%1.1f%%",
                        startangle=90,
                        textprops={"fontsize": 9}
                    )

                    ax.set_title("Duplicate vs Unique Rows", fontsize=10)
                    ax.axis("equal")

                    st.pyplot(fig)
                duplicate_values=st.session_state.df.duplicated().sum()
                st.info(f"ℹ️ Found {duplicate_values} duplicated records")
                if duplicate_values==0:
                    pass
                else:
                    if st.button("🗑️ Drop Duplicates", use_container_width=True):
                        st.session_state.df=st.session_state.df.drop_duplicates().reset_index(drop=True)
                        st.success("✅ Duplicate records removed!")
                        st.dataframe(st.session_state.df)
                         # Store in session state so download works

                        csv = st.session_state.df.to_csv(index=False)
                # Download button
                        st.download_button(
                            label="📥 Download Cleaned Dataset",
                            data=csv,
                            file_name="removed_duplicates.csv",
                            mime="text/csv"
                        )

            st.markdown("---")
            st.subheader("🚨 Drop All Null Values")
            st.caption("Remove all rows containing any missing values (use with caution)")
            if st.checkbox("⚠️ I want to drop all null values"):
                if st.button("🗑️ Drop All Null Values", use_container_width=True):
                    st.session_state.df=st.session_state.df.dropna()
                    st.success("✅ Dropped all null values successfully!")

                    st.write("Updated Missing Values Count:")
                    st.write(st.session_state.df.isnull().sum())
                    csv = st.session_state.df.to_csv(index=False)
                # Download button
                    st.download_button(
                            label="📥 Download Cleaned Dataset",
                            data=csv,
                            file_name="removed_nulls.csv",
                            mime="text/csv"
                    )
                   
else:
    st.warning("⚠️ Please upload a CSV file to start analysis")
