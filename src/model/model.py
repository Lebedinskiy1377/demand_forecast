import time
from typing import List, Tuple

from joblib import Parallel, delayed
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
            Data to fit on.revised simplex
        verbose : bool, optional
            Whether to show progress bar, by default False
            Optional to implement, not used in grading.
        """
        data = data.dropna()
        data = data.sort_values(by=[self.date_col, self.sku_col])
        sku_ids = data[self.sku_col].unique()

        def fit_sku(sku_id):
            data_sku = data[data[self.sku_col] == sku_id]
            X = data_sku[self.features]

            models_for_cur_sku = {}
            for horizon in self.horizons:
                y = data_sku[f"next_{horizon}d"]
                for q in self.quantiles:
                    quant_reg = QuantileRegressor(quantile=q, alpha=0)
                    quant_reg.fit(X, y)
                    models_for_cur_sku[(q, horizon)] = quant_reg
            return sku_id, models_for_cur_sku

        results = Parallel(n_jobs=-1, verbose=verbose)(delayed(fit_sku)(sku_id) for sku_id in sku_ids)

        for sku_id, models_for_cur_sku in results:
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

        def predict_sku(sku_id):
            df_sku = data[data[self.sku_col] == sku_id]
            result = pd.DataFrame(0, index=df_sku.index, columns=required_columns)
            result["sku_id"] = sku_id
            result["day"] = df_sku[self.date_col].dt.date

            for horizon in self.horizons:
                for quantile in self.quantiles:
                    target_col = f"pred_{horizon}d_q{int(quantile * 100)}"
                    if sku_id in self.fitted_models_:
                        model = self.fitted_models_[sku_id].get((quantile, horizon))
                        if model:
                            result[target_col] = model.predict(df_sku[self.features])
                        else:
                            result[target_col] = 0
                    else:
                        result[target_col] = 0
            return result

        results = Parallel(n_jobs=-1)(delayed(predict_sku)(sku_id) for sku_id in data[self.sku_col].unique())

        res = pd.concat(results)
        res = res.reset_index(drop=True)
        res = res.sort_values(by=[self.sku_col, self.date_col])
        return res[required_columns]


if __name__ == "__main__":
    start = time.time()
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
    predictions.to_csv("../../data/pred.csv", index=False)
