import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_dist(df, cols, is_cat):
    fig_rows = len(is_cat)
    fig, axes = plt.subplots(fig_rows, 1, figsize=(5, 4 * fig_rows))
    axes = axes.flatten()
    for i in range(fig_rows):
        if is_cat[i]:
            sns.countplot(data=df, x=cols[i], ax=axes[i])
        else:
            sns.histplot(data=df, x=cols[i], ax=axes[i])
        if df[cols[i]].isna().any():
            print(f'{cols[i]} has {df[cols[i]].isna().sum() :.0f} missing values: {df[cols[i]].isna().sum()/df[cols[i]].isna().count() * 100 :.3f}%')
        else:
            print(f'{cols[i]} has no missing values')
    return fig

def print_outlier(df, cols):
    """Display number and percentage of outliers in each column.
        Outliers are observations outside [Q1 - 1.5IQR, Q3 + 1.5IQR]

    Args:
        df (DataFrame): DataFrame to investigate
    """
    for col in cols:
        IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
        lower_bound = df[col].quantile(0.25) - 1.5 * IQR
        if lower_bound < df[col].min():
            lower_bound = df[col].min()
        upper_bound = df[col].quantile(0.75) + 1.5 * IQR
        if upper_bound > df[col].max():
            upper_bound = df[col].max()
        num_outliers = df.loc[df[col]<lower_bound, col].count() + df.loc[df[col]>upper_bound, col].count()
        print(f'{col} has {num_outliers} outliers: {num_outliers / df[col].count() * 100 :.3f}%')
        print(f'{col} upper bound is {upper_bound}')
        print(f'{col} lower bound is {lower_bound}')

def separate_missing(df, col):
    return df.loc[df[col].isna()], df.loc[df[col].notna()]

