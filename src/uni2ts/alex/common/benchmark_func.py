import os

import numpy as np
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import TestData, split
from gluonts.ev.metrics import MAE, MAPE

from uni2ts.eval_util.evaluation import evaluate_model
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule


def moving_average(array: np.ndarray, window_size: int):
    window_size = window_size

    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []

    # Loop through the array t o
    # consider every window of size 3
    while i < len(array) - window_size + 1:
        # Calculate the average of current window
        window_average = round(np.sum(array[i : i + window_size]) / window_size, 2)

        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)

        # Shift window to right by one position
        i += 1

    moving_averages = np.array(moving_averages)

    return moving_averages


def load_pretrained(
    size: str,
    prediction_lenght: int,
    context_length: int,
    patch_size: int | str = "auto",
    num_samples: int = 100,
):
    moirai = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{size}")

    pretrained_model = MoiraiForecast(
        module=moirai,
        prediction_length=prediction_lenght,
        context_length=context_length,
        patch_size=patch_size,
        num_samples=num_samples,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )

    return pretrained_model


def get_model_data(
    model_folder: str,
    prediction_lenght: int,
    num_of_weeks: int,
    data: pd.DataFrame,
    patch_size: int | str = "auto",
    num_samples: int = 100,
):
    if num_of_weeks * 7 * 24 < 720:
        CTX = num_of_weeks * 7 * 24  # context length: any positive integer
    else:
        CTX = 720

    # load the model of a specific week or the pretrained one
    if model_folder != "pretrained":
        model_folder = model_folder + str(num_of_weeks)
        model = os.listdir(model_folder)[0]

        model_path = os.path.join(model_folder, model)

        model = MoiraiForecast.load_from_checkpoint(
            prediction_length=prediction_lenght,
            context_length=CTX,
            patch_size=patch_size,
            num_samples=num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            checkpoint_path=model_path,
        )
    else:
        model = load_pretrained(
            size="small",
            prediction_lenght=prediction_lenght,
            context_length=CTX,
            num_samples=num_samples,
        )

    ds = PandasDataset(data, target="Value", freq="H")

    # Split into train/test set
    _, test_template = split(
        ds, offset=-(data.shape[0] - num_of_weeks * (7 * 24))
    )  # assign last TEST time steps as test set

    NUM_WINDOWS = (data.shape[0] - num_of_weeks * (7 * 24)) // prediction_lenght

    # Construct rolling window evaluation
    test_data = test_template.generate_instances(
        prediction_length=prediction_lenght,  # number of time steps for each prediction
        windows=NUM_WINDOWS,  # number of windows in rolling window evaluation
        distance=prediction_lenght,  # number of time steps between each window - distance=PDT for non-overlapping windows
        max_history=CTX,
    )

    return model, test_data


def get_model_perf(
    model: MoiraiForecast,
    test_data,
    quantiles: list[float],
    step: int,
    metric: str = "MAE",
    batch_size: int = 24,
):
    metrics = []
    for perc in quantiles:
        if metric == "MAPE":
            eval_function = MAPE(str(perc))
        else:
            eval_function = MAE(str(perc))

        metric_eval = evaluate_model(
            model=model.create_predictor(batch_size=batch_size),
            test_data=test_data,
            metrics=[eval_function],
        )

        metrics.append(metric_eval)

    metrics = pd.concat(metrics, axis=1)
    metrics.index = pd.Series(f"Model_step_{step}")

    return metrics


def get_eval_foreasts(
    model: MoiraiForecast, test_data: TestData, batch_size: int = 168
):
    predictor = model.create_predictor(batch_size=batch_size)
    forecasts = predictor.predict(test_data.input)

    target_values = np.stack([target["target"] for target in list(test_data.label)])
    # shape NUM_WINDOWS, NUM_SAMPLES, PREDICTION_LENGTH
    # samples predictions at columns, i.e. [:, 0] forecasts for 1st time step, [:, 1] forecasts for 2nd time step,
    forecast_samples = np.stack([sample.samples for sample in list(forecasts)])

    return forecast_samples, target_values


def get_metrics_v2(
    model_folder: str, data: pd.DataFrame, weeks_lim: int
) -> dict[str, list[np.ndarray]]:
    """Get the forecasted distributions per time step and the target. Afterwards, compute
    absolute percentage errors of the mean, median, 97.5th and 2.5th percentile of each
    forecasted distributions for each model in the model_folder.

    Args:
        model_folder (str): Folder where the respective models (1 model per week of available data) are located.
        data (pd.DataFrame): The dataset based on which evaluation is going to be performed.
        weeks_lim (int): Limit the number of weeks, i.e. models, to compute metrics for.

    Returns:
        metrics: Dictionary consisting of 4 lists: mean of forecasted distributions, median, 97.5th percentile, 2.5th percentile.
                 Every item in a list corresponds to 1 model, so that the length of each the list equals the number of model evaluated.
    """
    metrics = {"mean": [], "median": [], "lower_0025": [], "upper_0975": []}
    for i in range(1, weeks_lim + 1):
        try:
            model, test_data = get_model_data(
                model_folder=model_folder,
                prediction_lenght=168,
                num_of_weeks=i,
                data=data,
                patch_size="auto",
                num_samples=500,
            )

            forecast_samples, target_values = get_eval_foreasts(model, test_data)

            # absolute error of the average forecast of each time step
            mean_error_ts = (
                np.mean(forecast_samples, axis=1).flatten() - target_values.flatten()
            ) / target_values.flatten()
            mean_error_ts = np.abs(mean_error_ts)

            # absolute error of the median forecast of each time step
            median_error_ts = (
                np.quantile(forecast_samples, 0.5, axis=1).flatten()
                - target_values.flatten()
            ) / target_values.flatten()
            median_error_ts = np.abs(median_error_ts)

            # absolute error of the 97.5th percentile forecast of each time step
            upper_error_ts = (
                np.quantile(forecast_samples, 0.975, axis=1).flatten()
                - target_values.flatten()
            ) / target_values.flatten()
            upper_error_ts = np.abs(upper_error_ts)

            # absolute error of the 2.5th percentile forecast of each time step
            lower_error_ts = (
                np.quantile(forecast_samples, 0.025, axis=1).flatten()
                - target_values.flatten()
            ) / target_values.flatten()
            lower_error_ts = np.abs(lower_error_ts)

            metrics["mean"].append(mean_error_ts)
            metrics["median"].append(median_error_ts)
            metrics["lower_0025"].append(lower_error_ts)
            metrics["upper_0975"].append(upper_error_ts)

        except:
            pass

    return metrics


def get_metrics(
    model_folder: str,
    prediction_lenght: int,
    data: pd.DataFrame,
    metric: str,
    quantiles: list[float],
    patch_size: int | str = "auto",
    num_samples: int = 100,
):
    models_perf = []
    for i in range(1, 53):
        try:
            num_of_weeks = i
            model, test_data = get_model_data(
                model_folder=model_folder,
                prediction_lenght=prediction_lenght,
                num_of_weeks=num_of_weeks,
                data=data,
                patch_size=patch_size,
                num_samples=num_samples,
            )

            model_metrics = get_model_perf(
                model,
                test_data=test_data,
                quantiles=quantiles,
                step=num_of_weeks,
                metric=metric,
            )

            models_perf.append(model_metrics)
        except:
            pass

    return models_perf
