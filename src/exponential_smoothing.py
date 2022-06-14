import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from train_all import *

class ExpSmoothing:

    def __init__(self) -> "ExpSmoothing":
        self.data = None
        self.input_window_size = None
        self.models = None
        self.scores_list = None
        self.model_instances = None
        self.begin_raw = None
        self.n_test = 28
        self.n_days = 7
        self.data_used_in_trainning = None

    # gets data from datamanager endpoint. returns data as csv
    def get_data(self, repo, path, feature, mavg_window_size, begin ,end):
        file_name = "".join(
                        f"http://ncovid.natalnet.br/datamanager/"
                        f"repo/{repo}/"
                        f"path/{path}/"
                        f"features/{feature}/"
                        f"window-size/{mavg_window_size}/"
                        f"begin/{begin}/"
                        f"end/{end}/as-csv"
                    )
        df = pd.read_csv(file_name, parse_dates=["date"], index_col="date")
        # TODO: solve this for multivariate case
        for x in feature.split(":")[1:]:
            data = df[x].values
        # store the first day available
        self.begin_raw = df.index[0]
        self.data = data
        return data

    # generate many configs and do a grid-search. returns the 3 best configs and its scores.
    def grid_search_exp(self, data_test_size_in_days, input_window_size):
        
        n_test = data_test_size_in_days
        n_days = input_window_size

        cfg_list = list()
        cfg_list.append(exp_smoothing_configs())

        models = dict()
        models['exp'] = exp_smoothing_forecast
                
        scores_list = list()

        for (name, func), cfg_list_ in zip(models.items(), cfg_list):
            scores = grid_search(func, self.data, cfg_list_, n_test, n_days)
            scores_list.append(scores[:3])
            print('done model '+name)
            print('3 best models are: ')
            # list top 3 configs
            for cfg, error in scores[:3]:
                print(cfg, error)
            print()
        
        self.scores_list = scores_list

        # update global variables
        self.input_window_size = input_window_size
        self.score_list = scores[:3]
        self.models = models
        return scores[:3]
    
    def retrain_best_models(self):
        # prepare data to retrain
        train = self.data[len(self.data) % self.input_window_size:]
        train = array(split(train, len(train)/self.input_window_size))

        self.data_used_in_trainning = to_series(train)

        models_forecast = list()

        for (name, func), scores_ in zip(self.models.items(), self.scores_list):
            for cfg_str, error in scores_:
                cfg = literal_eval(cfg_str)
                model = func(train, cfg, True)
                models_forecast.append((name, func, model, cfg_str, error))

        self.model_instances = models_forecast
    
    # returns the forecast of the best model trained
    def instance_forecast_ahead(self, days_ahead):
        return self.model_instances[0][2].predict(0, len(self.data_used_in_trainning)+days_ahead)

    # begin and end are dates in "%Y-%m-%d" format
    def instance_forecast_by_period(self, begin_forecast, end_forecast):
        # cast date string to datetime
        begin_raw_date = datetime.datetime.strptime(str(self.begin_raw.date()), "%Y-%m-%d")
        begin_forecast_date = datetime.datetime.strptime(begin_forecast, "%Y-%m-%d")
        end_forecast_date = datetime.datetime.strptime(end_forecast, "%Y-%m-%d")

        # find forecast indexes
        initial_forecast_index = begin_forecast_date - begin_raw_date
        forecast_period_in_days = end_forecast_date - begin_forecast_date
        final_forecast_index = initial_forecast_index + forecast_period_in_days
        
        # get forecast from best instance
        yhat = self.model_instances[0][2].predict(initial_forecast_index.days, final_forecast_index.days)
        return yhat