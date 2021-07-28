import pandas as pd
import xgboost as xgb
from savvy_freight_eta.base_experiment import BaseExperiment
from savvy_freight_eta.constants import (
    TARGET,
    STARTING_DATE,
    ORIGIN,
    DESTINATION,
    ORDER_ID,
)
import savvy_freight_eta.features as ft

class SimpleXGBoost(BaseExperiment):
    def __init__(self, log_comet, target=1, n_splits=5, filter_outliers=False):
        """
        Very simple predictor in which we calculate
        the dataset wide mean and use that to predict each trip.
        This is simplest than the CFL baseline.

        Parameters
        ----------
        log_comet: bool
            Wheather to create a comet-ml experiment and log it.


        """
        name = "SimpleXGBoost"
        super(SimpleXGBoost, self).__init__(
            name, log_comet, target, n_splits, filter_outliers
        )

    def train(self, df_train):
        """
        Calculates the mean of the train dataset as `mean`.

        Parameters
        ----------
        df_train: pd.DataFrame
            Dataset used for training

        Returns
        -------
        predict: function(pd.DataFrame) -> pd.Series
            returns a function that accepts a pd.DataFrame
            containing the test data and returns a pd.Series
            with the predicton.

        """

        CATEGORICAL_COLUMNS = [ORIGIN, DESTINATION]

        df_train = ft.add_temporal_features(
            df_train.copy(), date_variable=STARTING_DATE
        )
        encoder = ft.get_target_encoding_features(
            df_train,
            categorical_variables=CATEGORICAL_COLUMNS,
            target_variable=TARGET,
        )

        transformed_columns = encoder.transform(df_train[CATEGORICAL_COLUMNS])
        df_train[CATEGORICAL_COLUMNS] = transformed_columns

        df_train = df_train.drop(columns=[ORDER_ID, STARTING_DATE])
        target = df_train.pop(TARGET)

        model = xgb.XGBRegressor(n_jobs=1).fit(df_train, target)

        def predict(df_test):
            """
            Predicts each value in test with a the mean
            of the train dataset.
            Args:
                df_test: pd.DataFrame
                Dataset used from testing

            Returns:
            y_pred: pd.Series
                A series with the prediction for each element in df_test.
                The index of this series and that of df_test must be the same.
            """
            df_test = ft.add_temporal_features(
                df_test.copy(), date_variable=STARTING_DATE
            )
            transformed_columns = encoder.transform(df_test[CATEGORICAL_COLUMNS])
            df_test[CATEGORICAL_COLUMNS] = transformed_columns
            df_test = df_test.drop(columns=[ORDER_ID, STARTING_DATE])

            y_pred = pd.Series(index=df_test.index, dtype=float)
            y_pred.iloc[:] = model.predict(df_test)

            return y_pred

        return predict


if __name__ == "__main__":
    exp = SimpleXGBoost(log_comet=False)
    res = exp.evaluate_prediction()
