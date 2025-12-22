
def dataset_basic_info(df):
    shape=df.shape

    total_missing_vals=df.isnull().sum().sum()
    total_duplicate_rows=df.duplicated().sum()
    perc_of_missing = (df.isnull().sum().sum() / df.size) * 100

    return shape,total_missing_vals,total_duplicate_rows,perc_of_missing

def ds_structure_and_stats(df):
    column_names=df.dtypes
    describe=df.describe()

    return column_names,describe

def column_wise_vals(df):
    col_missing_vals=df.isnull().sum()
    col_perc_missing_vals=df.isnull().mean()*100

    return col_missing_vals,col_perc_missing_vals

def unique_vals(df):
    unique_counts = df.nunique().reset_index()
    unique_counts.columns = ["Column Name", "Unique Values Count"]

    return unique_counts


