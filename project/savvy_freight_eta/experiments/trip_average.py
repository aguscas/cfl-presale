from savvy_freight_eta.base_experiment import BaseExperiment
from savvy_freight_eta.constants import ORIGIN, DESTINATION, TARGET


class TripAverage(BaseExperiment):
    def __init__(self, log_comet, target=1, n_splits=5, filter_outliers=False):
        """TODO: Docstring for __init__.

        Parameters
        ----------
        function : TODO

        Returns
        -------
        TODO

        """
        name = "TripAverage"
        super(TripAverage, self).__init__(
            name, log_comet, target, n_splits, filter_outliers
        )

    def train(self, df_train):
        """TODO: Docstring for train.

        Parameters
        ----------
        function : TODO

        Returns
        -------
        TODO

        """

        def predict(df_test):
            mean_all = df_train[TARGET].mean()
            mean_trip = df_train.groupby([ORIGIN, DESTINATION])[TARGET].mean()
            y_pred = df_test.join(mean_trip, on=[ORIGIN, DESTINATION])[
                TARGET
            ].fillna(mean_all)
            return y_pred

        return predict


if __name__ == "__main__":
    exp = TripAverage(log_comet=False)
    res = exp.evaluate_prediction()
