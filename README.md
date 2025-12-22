# Chat With Your Data 📊💬  
### AI-Powered Data Understanding & Cleaning App using Streamlit + Gemini API

Understanding a dataset for the first time can be confusing.  
Rows, columns, numbers everywhere — and no clear explanation of what the data is actually saying.

This project solves that problem by turning raw data into a **conversation**.

Using **Google’s Gemini API** and **Streamlit**, this application helps beginners:
- Understand what a dataset is about
- Learn what each column means
- Ask questions directly to the data
- Perform EDA and clean data easily

---

## 🚀 Features

### 🧠 AI-Powered Dataset Explanation
- Automatically explains what the dataset is about
- Describes each column in simple, beginner-friendly language
- Uses real-life style explanations

### 💬 Chat With Your Data
- Ask questions in natural language
- Answers are generated **only from the dataset**
- Prevents hallucinations or out-of-scope answers

### 📊 Exploratory Data Analysis (EDA)
- Dataset shape (rows × columns)
- Missing values count & percentage
- Duplicate rows detection
- Data types and basic statistics
- Unique values analysis
- Visual charts for better understanding

### 🧹 Data Cleaning Tools
- Fill missing values using:
  - Mean
  - Median
  - Mode
  - Zero
- Remove duplicate rows
- Drop all null values (with caution)
- Download cleaned dataset as CSV

### 📄 Automated Data Report
- Generate a detailed profiling report using `ydata-profiling`

---

## 🧠 AI Used

- **Google Gemini API**
- Model: `gemini-2.5-flash`
- Used for:
  - Dataset explanation
  - Column-wise understanding
  - Chat-based Q&A strictly based on dataset

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- Matplotlib
- Google Generative AI (Gemini API)
- ydata-profiling

---
## 🎯 Why This Project?

While learning data science, I faced a lot of difficulty understanding datasets.
Sometimes the data is so confusing that even knowing the tools is not enough.

This project was built to:
Make data beginner-friendly
Reduce fear around raw datasets
Combine AI + Data Analysis in a practical way
