from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from tqdm import tqdm


def split_train_test(
        df: pd.DataFrame,
        test_days: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train and test sets.

    The last `test_days` days are held out for testing.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        test_days (int): The number of days to include in the test set (default: 30).
            use ">=" sign for df_test

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
        A tuple containing the train and test DataFrames.
    """
    df.day = pd.to_datetime(df.day)
    date_threshhold = df.day.max() - pd.Timedelta(days=test_days)
    df_train = df[df.day < date_threshhold]
    df_test = df[df.day >= date_threshhold]
    return df_train, df_test


class MultiTargetModel:
    def __init__(
            self,
            features: List[str],
            horizons: List[int] = [7, 14, 21],
            quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """
        Parameters
        ----------
        features : List[str]
            List of features columns.
        horizons : List[int]
            List of horizons.
        quantiles : List[float]
            List of quantiles.

        Attributes
        ----------
        fitted_models_ : dict
            Dictionary with fitted models for each sku_id.
            Example:
            {
                sku_id_1: {
                    (quantile_1, horizon_1): model_1,
                    (quantile_1, horizon_2): model_2,
                    ...
                },
                sku_id_2: {
                    (quantile_1, horizon_1): model_3,
                    (quantile_1, horizon_2): model_4,
                    ...
                },
                ...
            }

        """
        self.quantiles = quantiles
        self.horizons = horizons
        self.sku_col = "sku_id"
        self.date_col = "day"
        self.features = features
        self.targets = [f"next_{horizon}d" for horizon in self.horizons]
        self.fitted_models_ = {}

    def fit(self, data: pd.DataFrame, verbose: bool = False) -> None:
        """Fit model on data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to fit on.
        verbose : bool, optional
            Whether to show progress bar, by default False
            Optional to implement, not used in grading.
        """
        data = data.dropna()
        data = data.sort_values(by=[self.date_col, self.sku_col])
        sku_ids = data[self.sku_col].unique()

        for sku_id in sku_ids:
            data_sku = data[data.sku_id == sku_id]
            models_for_cur_sku = {}
            for horizon in self.horizons:
                X = data_sku[self.features]
                y = data_sku[f"next_{horizon}d"]
                for q in self.quantiles:
                    quant_reg = QuantileRegressor(quantile=q, alpha=0)
                    quant_reg.fit(X, y)
                    models_for_cur_sku[(q, horizon)] = quant_reg
            self.fitted_models_[sku_id] = models_for_cur_sku

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict on data.

        Predict 0 values for a new sku_id.

        Parameters
        ----------
        data : pd.DataFrame
            Data to predict on.

        Returns
        -------
        pd.DataFrame
            Predictions.
        """
        data = data.dropna()
        data = data.sort_values(by=[self.date_col, self.sku_col])
        required_columns = [
            "sku_id", "day",
            "pred_7d_q10", "pred_7d_q50", "pred_7d_q90",
            "pred_14d_q10", "pred_14d_q50", "pred_14d_q90",
            "pred_21d_q10", "pred_21d_q50", "pred_21d_q90"
        ]

        res = pd.DataFrame(0, index=data.index, columns=required_columns)

        for sku_id in data[self.sku_col].unique():
            df_sku = data[data[self.sku_col] == sku_id]
            res.loc[df_sku.index, "sku_id"] = sku_id
            res.loc[df_sku.index, "day"] = df_sku["day"].dt.date
            for horizon in self.horizons:
                for quantile in self.quantiles:
                    target_col = f"pred_{horizon}d_q{int(quantile * 100)}"
                    if sku_id in self.fitted_models_:
                        model = self.fitted_models_[sku_id].get((quantile, horizon))
                        if model:
                            res.loc[df_sku.index, target_col] = model.predict(df_sku[self.features])
                        else:
                            res.loc[df_sku.index, target_col] = 0
                    else:
                        res.loc[df_sku.index, target_col] = 0

        res = res.reset_index(drop=True)
        res = res.sort_values(by=[self.sku_col, self.date_col])
        return res[required_columns]


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Calculate the quantile loss between the true and predicted values.

    The quantile loss measures the deviation between the true
        and predicted values at a specific quantile.

    Parameters
    ----------
    y_true : np.ndarray
        The true values.
    y_pred : np.ndarray
        The predicted values.
    quantile : float
        The quantile to calculate the loss for.

    Returns
    -------
    float
        The quantile loss.
    """
    error = y_true - y_pred
    loss = np.maximum(quantile * error, (quantile - 1) * error)
    return loss


def evaluate_model(
        df_true: pd.DataFrame,
        df_pred: pd.DataFrame,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        horizons: List[int] = [7, 14, 21],
) -> pd.DataFrame:
    """Evaluate model on data.

    Parameters
    ----------
    df_true : pd.DataFrame
        True values.
    df_pred : pd.DataFrame
        Predicted values.
    quantiles : List[float], optional
        Quantiles to evaluate on, by default [0.1, 0.5, 0.9].
    horizons : List[int], optional
        Horizons to evaluate on, by default [7, 14, 21].

    Returns
    -------
    pd.DataFrame
        Evaluation results.
    """
    losses = {}

    for quantile in quantiles:
        for horizon in horizons:
            true = df_true[f"next_{horizon}d"].values
            pred = df_pred[f"pred_{horizon}d_q{int(quantile * 100)}"].values
            loss = quantile_loss(true, pred, quantile)

            losses[(quantile, horizon)] = loss

    losses = pd.DataFrame(losses, index=["loss"]).T.reset_index()
    losses.columns = ["quantile", "horizon", "avg_quantile_loss"]  # type: ignore

    return losses


if __name__ == "__main__":
    model = MultiTargetModel(
        features=[
            "price",
            "qty",
            "qty_7d_avg",
            "qty_7d_q10",
            "qty_7d_q50",
            "qty_7d_q90",
            "qty_14d_avg",
            "qty_14d_q10",
            "qty_14d_q50",
            "qty_14d_q90",
            "qty_21d_avg",
            "qty_21d_q10",
            "qty_21d_q50",
            "qty_21d_q90",
        ],
        horizons=[7, 14, 21],
        quantiles=[0.1, 0.5, 0.9],
    )
    df = pd.read_csv("../../data/features.csv")
    df_train, df_test = split_train_test(df)
    model.fit(df_train, verbose=True)

    predictions = model.predict(df_test)
    print(predictions)
    # predictions.to_csv("data/pred.csv", index=False)
