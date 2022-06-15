from flask import Flask, jsonify, request

app = Flask(__name__)

import json
import exponential_smoothing

@app.route(
    "api/v1/exp-smoothing/train/repo/<repo>/path/<path>/feature/<feature>/begin/<begin>/end/<end>/",
    methods=["POST"],
)
def train_exp_smoothing(repo, path, feature, begin, end):
    # get the metadata
    metadata_to_train = json.loads(request.form.get("metadata"))

    # instantiate handler object
    handler_exp = exponential_smoothing.ExpSmoothing()

    # get adequate data to train
    handler_exp.get_data(repo, path, feature, metadata_to_train['mavg_window_size'], begin, end)

    # grid-search some configurations
    handler_exp.grid_search_exp(metadata_to_train['testSize'], metadata_to_train['inputWindowSize'])

    # retrain best models found by grid-search
    handler_exp.retrain_best_models()

    # persist best instances. send metadata to requester
    # TODO: send binary to db
    metadata_instance = handler_exp.instance_save()
    
    return metadata_instance

@app.route(
    "api/v1/exp-smoothing/predict/model-instance/<modelInstance>",
    methods=["POST"],
)
def predict_exp_smoothing(modelInstance):
    # get the metadata
    metadata_instance = json.loads(request.form.get("metadata"))

    forecast_begin_date = metadata_instance['begin']
    forecast_end_date = metadata_instance['end']

    # instantiate handler object
    handler_exp = exponential_smoothing.ExpSmoothing()

    # get forecast from instance
    forecast = handler_exp.instance_forecast_by_period(forecast_begin_date, forecast_end_date, modelInstance)

    # format forecast to send back to requester
    response_json = jsonify(handler_exp.predictions_to_weboutput(forecast, forecast_begin_date, forecast_end_date))

    return response_json

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")