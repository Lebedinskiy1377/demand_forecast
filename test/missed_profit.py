from typing import Tuple

import numpy as np
import pandas as pd


def week_missed_profits(
        df: pd.DataFrame,
        sales_col: str,
        forecast_col: str,
        date_col: str = "day",
        price_col: str = "price",
) -> pd.DataFrame:
    """
    Calculates the missed profits every week for the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to calculate the missed profits for.
        (Must contain columns "sku_id", "date", "price", "sales" and "forecast")
    sales_col : str
        The column with the actual sales.
    forecast_col : str
        The column with the forecasted sales.
    price_col : str, optional
        The column with the price, by default "price".

    Returns
    -------
    pd.DataFrame
        The DataFrame with the missed profits.
        (Contains columns "day", "revenue", "missed_profits")
    """

    def calculate_miss(group: pd.Series):
        miss = group[forecast_col] - group[sales_col]
        miss[miss < 0] = 0
        miss *= group[price_col]
        return np.sum(miss)

    def calculate_income(group: pd.Series):
        inc = group[sales_col]
        inc *= group[price_col]
        return np.sum(inc)

    miss = df.groupby(pd.Grouper(key=date_col,
                                 freq='W')).apply(calculate_miss).reset_index().rename(columns={0: "missed_profits"})
    income = df.groupby(pd.Grouper(key=date_col,
                                   freq='W')).apply(calculate_income).reset_index().rename(columns={0: "revenue"})
    return pd.merge(income, miss, on=date_col)


def missed_profits_ci(
        df: pd.DataFrame,
        missed_profits_col: str,
        confidence_level: float = 0.95,
        n_bootstraps: int = 1000,
) -> Tuple[Tuple[float, Tuple[float, float]], Tuple[float, Tuple[float, float]]]:
    """
    Estimates the missed profits for the given DataFrame.
    Calculates average missed_profits per week and estimates
    the 95% confidence interval.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to calculate the missed_profits for.

    missed_profits_col : str
        The column with the missed_profits.

    confidence_level : float, optional
        The confidence level for the confidence interval, by default 0.95.

    n_bootstraps : int, optional
        The number of bootstrap samples to use for the confidence interval,
        by default 1000.

    Returns
    -------
    Tuple[Tuple[float, Tuple[float, float]], Tuple[float, Tuple[float, float]]]
        Returns a tuple of tuples, where the first tuple is the absolute average
        missed profits with its CI, and the second is the relative average missed
        profits with its CI.

    Example:
    -------
    ((1200000, (1100000, 1300000)), (0.5, (0.4, 0.6)))
    """
    missed_array = df[missed_profits_col]
    revenue = df["revenue"]
    bootstrap_miss = []
    alpha = 1 - confidence_level
    gamma = alpha / 2

    for _ in range(n_bootstraps):
        sample_miss = np.random.choice(missed_array, size=len(missed_array), replace=True)
        bootstrap_miss.append(np.mean(sample_miss))

    mean_miss = np.mean(bootstrap_miss)
    mean_revenue = np.mean(revenue)
    left_bound = np.quantile(bootstrap_miss, gamma)
    right_bound = np.quantile(bootstrap_miss, gamma + confidence_level)

    confidence_interval = (mean_miss, (left_bound, right_bound))
    normed_confidence_interval = (mean_miss / mean_revenue, (left_bound / mean_revenue, right_bound / mean_revenue))

    return (confidence_interval, normed_confidence_interval)
