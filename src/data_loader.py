import pandas as pd


def load_data(filepath):
    """Load dataset from CSV file."""
    df = pd.read_csv(filepath)
    return df


def check_df(dataframe, head=5):
    """Display basic information about the dataframe."""
    print(" SHAPE ".center(70, "#"))
    print(dataframe.shape)
    print(" INFO ".center(70, "#"))
    print(dataframe.info())
    print(" MEMORY USAGE ".center(70, "#"))
    print(f"{dataframe.memory_usage().sum() / (1024 ** 2):.2f} MB")
    print(" NUNIQUE ".center(70, "#"))
    print(dataframe.nunique())
    print(" MISSING VALUES ".center(70, "#"))
    print(dataframe.isnull().sum())
    print(" DUPLICATED VALUES ".center(70, "#"))
    print(dataframe.duplicated().sum())


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """Identify and categorize columns by type."""
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "object" or dataframe[col].dtypes.name == "category"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "object" and dataframe[col].dtypes.name != "category"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   (dataframe[col].dtypes == "object" or dataframe[col].dtypes.name == "category")]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "object" and dataframe[col].dtypes.name != "category"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car, num_but_cat


def convert_num_but_cat_to_object(dataframe, num_but_cat):
    """Convert numeric categorical columns to object type."""
    for col in num_but_cat:
        if col in dataframe.columns:
            dataframe[col] = dataframe[col].astype('object')
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")
    return dataframe
