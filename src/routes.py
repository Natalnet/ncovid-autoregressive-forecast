from flask import Flask, jsonify, request

app = Flask(__name__)

import json
import autoregressive
from autoregressive import *

@app.route(
    "/api/v1/autoregressive/model_type/<modelType>/train/repo/<repo>/path/<path>/feature/<feature>/begin/<begin>/end/<end>/",
    methods=["POST"],
)
# modelTypes: exp_smoothing, arima, sarima
def train_autoregressive(modelType, repo, path, feature, begin, end):
    # get the metadata
    metadata_to_train = json.loads(request.form.get("metadata"))

    # instantiate handler object
    handler = autoregressive.AutoRegressive()

    cfg_func = getattr(autoregressive, f"{modelType}_configs")
    model_type_func = getattr(autoregressive, f"{modelType}_forecast")

    # create list of configs
    cfg_list = list()
    # cfg_list.append(exp_smoothing_configs())
    # cfg_list.append(exp_smoothing_configs())
    cfg_list.append(cfg_func())

    models = dict()
    # models['exp'] = exp_smoothing_forecast
    models['exp'] = model_type_func

    # get adequate data to train
    handler.get_data(repo, path, feature, metadata_to_train['mavg_window_size'], begin, end)

    # grid-search some configurations
    handler.grid_search(metadata_to_train['testSize'], metadata_to_train['inputWindowSize'], cfg_list, models)

    # retrain best models found by grid-search
    handler.retrain_best_models()

    # persist best instances. send metadata to requester
    # TODO: send binary to db
    metadata_instance = handler.instance_save()
    
    return jsonify(metadata_instance)

@app.route(
    "/api/v1/autoregressive/predict/model-instance/<modelInstance>",
    methods=["POST"],
)
def predict_autoregressive(modelInstance):
    # get the metadata
    metadata_instance = json.loads(request.form.get("metadata"))

    forecast_begin_date = metadata_instance['begin']
    forecast_end_date = metadata_instance['end']

    # instantiate handler object
    handler = autoregressive.AutoRegressive()

    # get forecast from instance
    forecast = handler.instance_forecast_by_period(forecast_begin_date, forecast_end_date, modelInstance)

    # format forecast to send back to requester
    response_json = jsonify(handler.predictions_to_weboutput(forecast, forecast_begin_date, forecast_end_date))

    return response_json

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")