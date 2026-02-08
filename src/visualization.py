import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def num_summary(dataframe, numerical_col, plot=False):
    """Display numerical summary with optional histogram."""
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


def plot_correlation_heatmap(dataframe, num_cols):
    """Plot correlation heatmap for numerical columns."""
    corr = dataframe[num_cols].corr()
    sns.set(rc={'figure.figsize': (12, 12)})
    sns.heatmap(corr, cmap="RdBu")
    plt.show()


def plot_target_distribution(dataframe, target_col):
    """Plot target variable distribution."""
    dataframe[target_col].hist(bins=100)
    plt.show()
    
    np.log1p(dataframe[target_col]).hist(bins=50)
    plt.show()


def plot_importance(model, features, num=20, save=False):
    """Plot feature importance."""
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")
