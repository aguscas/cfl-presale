import numpy as np
from category_encoders import TargetEncoder


def std_numpy(x):
    # Workaround for groupby agg
    # Using np.std directly returns nan when the group has one (or even two!)
    # samples
    # Redifining std to return np.std solves this issue
    return np.std(x)


def add_temporal_features(df, date_variable, min_date=None):

    df = df.copy()

    if min_date is None:
        min_date = df[date_variable].min()

    df.loc[:, "day"] = df[date_variable].dt.day
    df.loc[:, "day_of_year"] = df[date_variable].dt.dayofyear
    df.loc[:, "weekday"] = df[date_variable].dt.weekday
    # df.loc[:, "week"] = df[date_variable].dt.week
    df.loc[:, "week"] = (
        df[date_variable]
        .dt.isocalendar()["week"]
        .astype("int64")
        .rename("date")
    )
    df.loc[:, "month"] = df[date_variable].dt.month
    df.loc[:, "year"] = df[date_variable].dt.year
    df.loc[:, "hour"] = df[date_variable].dt.hour
    df.loc[:, "minute"] = df[date_variable].dt.minute
    df.loc[:, "n_days"] = (df[date_variable] - min_date).dt.days
    df.loc[:, "n_weeks"] = (df[date_variable] - min_date).dt.days / 7
    df.loc[:, "week_sin"] = np.sin((df["week"] - 1) * (2 * np.pi / 52))
    df.loc[:, "week_cos"] = np.cos((df["week"] - 1) * (2 * np.pi / 52))
    df.loc[:, "month_sin"] = np.sin((df["month"] - 1) * (2 * np.pi / 12))
    df.loc[:, "month_cos"] = np.cos((df["month"] - 1) * (2 * np.pi / 12))
    df.loc[:, "weekday_sin"] = np.sin((df["weekday"] - 1) * (2 * np.pi / 7))
    df.loc[:, "weekday_cos"] = np.cos((df["weekday"] - 1) * (2 * np.pi / 7))
    df.loc[:, "day_sin"] = np.sin((df["day"] - 1) * (2 * np.pi / 31))
    df.loc[:, "day_cos"] = np.cos((df["day"] - 1) * (2 * np.pi / 31))
    df.loc[:, "hour_sin"] = np.sin((df["hour"]) * (2 * np.pi / 24))
    df.loc[:, "hour_cos"] = np.cos((df["hour"]) * (2 * np.pi / 24))
    df.loc[:, "min_sin"] = np.sin((df["minute"] - 1) * (2 * np.pi / 60))
    df.loc[:, "min_cos"] = np.cos((df["minute"] - 1) * (2 * np.pi / 60))

    return df


def add_previous_peak(df, groupby_variables, date_variable, target_variable):

    columns = list(df.columns) + ["previous_peak"]

    df2 = df.copy()
    df2["day"] = df2[date_variable].dt.date
    df2["year"] = df2[date_variable].dt.year
    df2["month"] = df2[date_variable].dt.month
    df2["day"] = df2[date_variable].dt.day

    peak = df2.copy()
    peak["day"] = peak[date_variable].dt.date
    peak = (
        peak.groupby(groupby_variables + ["day"])[target_variable]
        .max()
        .reset_index()
    )

    avg = peak[target_variable].mean()

    peak["is_weekday"] = peak["day"].apply(lambda x: x.weekday() < 5)
    peak["previous_peak"] = peak.groupby(groupby_variables + ["is_weekday"])[
        target_variable
    ].shift(1)
    peak.head()

    peak["year"] = peak["day"].apply(lambda x: x.year)
    peak["month"] = peak["day"].apply(lambda x: x.month)
    peak["day"] = peak["day"].apply(lambda x: x.day)
    peak = peak.set_index(groupby_variables + ["year", "month", "day"])[
        "previous_peak"
    ]
    df2 = df2.set_index(groupby_variables + ["year", "month", "day"])

    df2 = df2.join(peak).fillna(avg).reset_index()
    df2 = df2[columns]
    return df2


def get_target_encoding_features(df, categorical_variables, target_variable):
    # Target encode categorical features
    target_enc = TargetEncoder(cols=categorical_variables)
    target_enc.fit(df[categorical_variables], df[target_variable])
    # Encode as float16, since values are scaled (careful, see np.finfo(np.float16))
    return target_enc
