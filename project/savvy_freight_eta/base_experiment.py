import matplotlib.pyplot as plt
from tqdm import tqdm

from comet_ml import Experiment
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from savvy_freight_eta.utils import positive_error_with_tolerance


from savvy_freight_eta.constants import (
    TARGET,
    ORIGIN,
    DESTINATION,
    COMET_PROJECT_NAME,
    COMET_WORKSPACE,
)

from savvy_freight_eta.data_preprocessing import load_dataset
# from savvy_freight_eta.keys import COMET_API_KEY


class BaseExperiment(object):

    """
    This class is the reference to implement
    experiments. It should provide a metric for reproducibility
    and ease of comparisson.
    """

    MIN_NUMBER_OBSERVATIONS = 0

    def __init__(
        self, name, log_comet, target=1, n_splits=5, filter_outliers=False
    ):
        """
        Base class to calculate experiments

        Parameters
        -----------
        target: int, default=5
            used to select which dataset to load.
            For more info read `load_dataset`.
        n_splits: int, default=5
            Number of splits into which to dive de dataset

        """
        # self._load_dataset()

        self.df = load_dataset(target, filter_outliers)
        self._create_splits(n_splits=n_splits)
        self.log_comet = log_comet
        self.target = target
        self.name = name

#         if self.log_comet:
#             experiment = Experiment(
#                 api_key=COMET_API_KEY,
#                 project_name=COMET_PROJECT_NAME,
#                 workspace=COMET_WORKSPACE,
#             )
#             experiment.set_name(name)
#             experiment.log_metric("train_hash", self.df_train_hash)
#             experiment.log_metric("test_hash", self.df_test_hash)

#             self.experiment = experiment

    def _create_splits(self, n_splits):
        """
        Divides the dataset into S splits
        which will be used for cross validation

        Paramters
        -----------
        n_splits: int,
            Number of splits to use to divde
            the dataset

        Returns
        --------
        splits: list of lists
            Each item of the list contains the indices
            of each split. These are divided in two, with the first
            group containing `n_splits - ` / `n_splits` of the total
            number of rows.

        """
        df = self.df

        df_test = df.sample(frac=0.1, random_state=42)
        df_train = df[~df.index.isin(df_test.index)].copy()

        encoder = LabelEncoder().fit(df_train[ORIGIN] + df_train[DESTINATION])
        labels = encoder.transform(df_train[ORIGIN] + df_train[DESTINATION])

        splits = list(
            StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=42
            ).split(df_train, labels)
        )

        self.df_test = df_test
        self.df_train = df_train
        self.splits = splits

    def get_train_data(self, split):
        """
        Returns a copy DataFrame with the default
        data that can be used for training

        Parameters
        ----------
        function : TODO

        Returns
        -------
        df_train: pd.DataFrame
            Dataset used for training

        """
        split = self.splits[split][0]
        return self.df_train.iloc[split].copy()

    def get_cv_test_data(self, split):
        """
        Returns a copy of the DataFrame used for testing

        Parameters
        ----------
        function : TODO

        Returns
        -------
        df_test: pd.DataFrame

        """
        split = self.splits[split][1]
        return self.df_train.iloc[split].copy().drop(columns=TARGET)

    def get_cv_test_target(self, split):
        """TODO: Docstring for get_test_tempalte.

        Parameters
        ----------

        Returns
        -------
        TODO

        """
        split = self.splits[split][1]
        return self.df_train.iloc[split][TARGET].copy()

    def get_test_data(self):
        return self.df_test

    def train(self, df_train, *kwargs):
        """TODO: Docstring for train.

        Parameters
        ----------
        df_train: pd.DataFrame
            Dataset used for training
        kwargs: possible extra parameters that the function might take

        Returns
        -------
        predict: function(pd.DataFrame) -> pd.Series
            returns a function that accepts a pd.DataFrame
            containing the test data and returns a pd.Series
            with the predicton.

        """
        pass

    def evaluate_prediction(self, *args, figures=False):
        """
        Evalutes the prediction on the test set with
        respect to given metrics. This should ensure
        reproducibility as the test set is always the same
        and the metrics too

        Parameters
        ----------

        Returns
        -------
        results: list of dict
            a list whose length is equal to the number
            of splits.
            each member of the list is a dictionary with
            the metrics calcualted for each split.
            Currently:
            `rmse`, `mae`, `fig_pred_plot`.

        """
        S = len(self.splits)
        results = []

        # Get results from cross validation
        for i in tqdm(range(S)):

            df_train = self.get_train_data(i)
            df_test = self.get_cv_test_data(i)
            y_real = self.get_cv_test_target(i)

            predict_i = self.train(df_train, *args)
            y_pred = predict_i(df_test)

            rmse = mean_squared_error(y_pred, y_real, squared=False)
            mae = mean_absolute_error(y_pred, y_real)
            pewt = positive_error_with_tolerance(y_pred, y_real)
            result = {"rmse": rmse, "mae": mae, "pewt": pewt}

            if figures:
                fig, ax = plt.subplots()
                y_real = y_real.sort_values(ascending=True)
                ax.plot(y_real.values, label="Real")
                ax.plot(y_pred[y_real.index].values, label="Prediction")
                ax.set_xlabel("Different Purchase Orders")
                ax.set_ylabel("Predicted TARGET (in days)")
                ax.legend()
                ax.set_title(
                    f"Predictions ordered by increasing errors in split {i}"
                )
                result["fig_pred_plot"] = fig

            results.append(result)

        # Get results from test data
        df_train = self.df_train
        df_test = self.df_test
        y_real = df_test.pop(TARGET)
        predict_test = self.train(df_train, *args)
        y_pred = predict_test(df_test)

        rmse = mean_squared_error(y_pred, y_real, squared=False)
        mae = mean_absolute_error(y_pred, y_real)
        pewt = positive_error_with_tolerance(y_pred, y_real)
        result = {
            "test_rmse": rmse,
            "test_mae": mae,
            "test_pewt": pewt,
        }
        if figures:

            fig, ax = plt.subplots()
            y_real = y_real.sort_values(ascending=True)
            ax.plot(y_real.values, label="Real")
            ax.plot(y_pred[y_real.index].values, label="Prediction")
            ax.set_xlabel("Different Purchase Orders")
            ax.set_ylabel("Predicted TARGET (in days)")
            ax.legend()
            ax.set_title(
                f"Predictions ordered by increasing errors in split {i}"
            )
            result["test_fig_pred_plot"] = fig

        results.append(result)

        return results
    
    def predict(self, *args, data):
        predictor = self.train(self.df_train, *args)
        return predictor(data)
