import os
import numpy as np
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split, TestData
from gluonts.ev.metrics import MAE, MAPE
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.eval_util.evaluation import evaluate_model


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


def get_eval_foreasts(model: MoiraiForecast, test_data: TestData, batch_size: int = 168):
    predictor = model.create_predictor(batch_size=batch_size)
    forecasts = predictor.predict(test_data.input)
    
    target_values = np.stack([target['target'] for target in list(test_data.label)])
    # shape NUM_WINDOWS, NUM_SAMPLES, PREDICTION_LENGTH
    # samples predictions at columns, i.e. [:, 0] forecasts for 1st time step, [:, 1] forecasts for 2nd time step, 
    forecast_samples = np.stack([sample.samples for sample in list(forecasts)])
    
    return forecast_samples, target_values
    

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
