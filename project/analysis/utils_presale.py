from math import isnan
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from datetime import timedelta
import seaborn as sns

from savvy_freight_eta.constants import (
    ORIGIN,
    DESTINATION,
    ORDER_ID,
    TARGET,
    STARTING_DATE,
)


def estimate_error(model, origin, destination):
    df_test = model.df_test
    
    df_test_origin_destination = df_test[(df_test[ORIGIN] == origin) & (df_test[DESTINATION] == destination)]

    if len(df_test_origin_destination)==0:
        print(f"No data was found from {origin} to {destination}, so the error will be estimated using the whole test set")
        df_test_origin_destination = df_test

    estimated_test_durations = model.predict(data=df_test_origin_destination)

    actual_durations = model.df.loc[df_test_origin_destination.index][TARGET]

    relative_error = (estimated_test_durations-actual_durations)/actual_durations

    rmse = mean_squared_error(relative_error, 0*estimated_test_durations, squared=False)
    mae = mean_absolute_error(relative_error, 0*estimated_test_durations)
    return rmse, mae

def estimate_all_errors(model):
    origin_destination_pairs = {}

    df_test = model.df_test

    for index, row in df_test.iterrows():
        try:
            origin_destination_pairs[row[ORIGIN]].update({row[DESTINATION]: None})
        except KeyError:
            origin_destination_pairs[row[ORIGIN]] = {row[DESTINATION]: None}


    for origin in origin_destination_pairs:
        for destination in origin_destination_pairs[origin]:

            rmse, mae = estimate_error(model, origin, destination)

            origin_destination_pairs[origin][destination] = {'rmse': rmse, 'mae': mae}
    
    return origin_destination_pairs

def plot_errors(origin_destination_pairs):
    origins = []
    destinations = []

    for origin in origin_destination_pairs:
        if origin not in origins:
            origins.append(origin)
        for destination in origin_destination_pairs[origin]:
            if destination not in destinations:
                destinations.append(destination)

    origins = np.array(origins)
    destinations = np.array(destinations)

    origin_destination_rmse = np.zeros((len(origins), len(destinations)))
    origin_destination_mae = np.zeros((len(origins), len(destinations)))
    max_rmse = 0
    max_mae = 0

    for n, origin in enumerate(origins):
        for m, destination in enumerate(destinations):
            try: 
                rmse, mae = origin_destination_pairs[origin][destination]['rmse'], origin_destination_pairs[origin][destination]['mae']
                max_rmse = max(max_rmse, rmse)
                max_mae = max(max_mae, mae)
            except KeyError:
                rmse, mae = None, None

            origin_destination_rmse[n][m] = rmse
            origin_destination_mae[n][m] = mae
            
    def sort_by_row_mean(array):
        mean_rows = []
        for row in array:

            not_none_row = [entry for entry in row if not isnan(entry)]
            mean_rows.append(np.mean(not_none_row))

        return np.argsort(mean_rows)[::-1]
    sort = sort_by_row_mean(origin_destination_rmse)
    origin_destination_rmse = origin_destination_rmse[sort]
    origin_destination_mae = origin_destination_mae[sort]
    origins = origins[sort]

    fig, (axes_rmse, axes_mae) = plt.subplots(ncols=1, nrows=2, figsize=(20, 50))

    axes_rmse.set_title('Relative rmse for origin destination pairs')
    axes_rmse.set_xlabel('destinations')
    axes_rmse.set_ylabel('origins')
    axes_rmse.set_xticklabels(destinations)
    axes_rmse.set_yticklabels(origins)
    axes_rmse.set_xticks(np.arange(len(destinations)))
    axes_rmse.set_yticks(np.arange(len(origins)))
    mappable_rmse = axes_rmse.imshow(origin_destination_rmse, interpolation='none', aspect='auto', vmin=0, vmax=max_rmse)
    fig.colorbar(mappable_rmse, ax=axes_rmse)

    axes_mae.set_title('Relative mae for origin destination pairs')
    axes_mae.set_xlabel('destinations')
    axes_mae.set_ylabel('origins')
    axes_mae.set_xticklabels(destinations)
    axes_mae.set_yticklabels(origins)
    axes_mae.set_xticks(np.arange(len(destinations)))
    axes_mae.set_yticks(np.arange(len(origins)))
    mappable_mae = axes_mae.imshow(origin_destination_mae, interpolation='none', aspect='auto', vmin=0, vmax=max_mae)
    fig.colorbar(mappable_mae, ax=axes_mae)

def predict_duration(model, origin, destination, starting_date):
    data = {
        ORIGIN: [origin], 
        ORDER_ID: [0],
        DESTINATION: [destination],
        STARTING_DATE: [pd.to_datetime(starting_date, infer_datetime_format=True)],
    }

    # Create DataFrame  
    df = pd.DataFrame(data)  

    estimation = model.predict(data=df)
    estimation = timedelta(days=float(estimation))
    time_of_arrival = estimation + df[STARTING_DATE][0]

    return estimation, time_of_arrival

def compare_models_frequency_errors(error_dict, error_type = 'rmse'):
    list_of_errors= []
    model_names = []

    for model in error_dict:
        model_names.append(model)
        errors_in_this_model = []
        for origin in error_dict[model]:
            for destination in error_dict[model][origin]:
                errors_in_this_model.append(error_dict[model][origin][destination][error_type])
        list_of_errors.append(errors_in_this_model)

    list_of_errors = np.array(list_of_errors)

    max_error = np.max(list_of_errors)

    fig, ax = plt.subplots(figsize=(20,6))

    ax.set_title(f'Relative {error_type} comparison')
    ax.set_xlabel('Relative error')
    ax.set_ylabel('Frequency')

    for error_model in list_of_errors:
        sns.distplot(error_model, hist=True, kde=True, hist_kws={'edgecolor':'black'}, kde_kws={'linewidth':2}, bins=30)
    ax.set_xticks(np.arange(0, max_error,0.5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.legend(model_names)
    plt.show()    