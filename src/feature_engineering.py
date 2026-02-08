import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    """Identify highly correlated columns."""
    numeric_df = dataframe.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


def target_summary_with_cat(dataframe, target, categorical_col):
    """Show target mean by categorical groups."""
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")
