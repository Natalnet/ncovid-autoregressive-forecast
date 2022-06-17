import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from autoregressive_utils import *
import uuid, json

class AutoRegressive:

    def __init__(self) -> "AutoRegressive":
        self.data = None
        self.input_window_size = None
        self.n_test = 28
        self.n_days = 7
        self.model_type_dict = None
        self.scores_list = None
        self.model_instances = None
        self.begin_raw = None
        self.end_raw = None
        self.begin_training = None
        self.data_used_in_trainning = None
        self.instance_region = None
        self.save_instance_path = "../dbs/instances_object/"
        self.save_metadata_path = "../dbs/instances_metadata/"
        self.model_category = "autoregressive"

    # gets data from datamanager endpoint. returns data as csv
    def get_data(self, repo, path, feature, mavg_window_size, begin ,end):
        if begin == None:
            begin = "2020-01-01"
        if end == None:
            end = "2050-01-01"

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
        # store first and last day available
        self.begin_raw = df.index[0]
        self.end_raw = df.index[-1]
        self.data = data
        self.instance_region = path
        return data

    # generate many configs and do a grid-search. returns the 3 best configs and its scores.
    def grid_search(self, data_test_size_in_days, input_window_size, cfg_list, models):
        
        n_test = data_test_size_in_days
        n_days = input_window_size

        # cfg_list = list()
        # cfg_list.append(exp_smoothing_configs())
        # # cfg_list.append(arima_configs())

        # models = dict()
        # models['exp'] = exp_smoothing_forecast
        # # models['arima'] = arima_forecast
                
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
        self.model_type_dict = models
        return scores[:3]
    
    def retrain_best_models(self):
        # prepare data to retrain
        train = self.data[len(self.data) % self.input_window_size:]
        train = array(split(train, len(train)/self.input_window_size))

        self.data_used_in_trainning = to_series(train)
        leftover = len(self.data) - len(self.data_used_in_trainning)

        models_forecast = list()

        for (name, func), scores_ in zip(self.model_type_dict.items(), self.scores_list):
            for cfg_str, error in scores_:
                cfg = literal_eval(cfg_str)
                model = func(train, cfg, True)
                models_forecast.append((name, func, model, cfg_str, error))

        self.model_instances = models_forecast
        self.begin_training = datetime.datetime.strptime(str(self.begin_raw.date()), "%Y-%m-%d") + datetime.timedelta(days=leftover)
    
    # returns the forecast of the best model trained
    def instance_forecast_ahead(self, days_ahead):
        return self.model_instances[0][2].predict(0, len(self.data_used_in_trainning)+days_ahead)

    # begin and end are dates in "%Y-%m-%d" format
    def instance_forecast_by_period(self, begin_forecast, end_forecast, instance_id=None):
        if instance_id == None:
            instance_object = self.model_instances[0][2]
        else:
            instance_object = self.load_instance_from_local_metadata_filename(str(instance_id))
            f = open(self.save_metadata_path + str(instance_id) + '.json')
            instance_metadata = json.load(f)
        
        begin_raw = instance_metadata['data_begin_date']

        # cast date string to datetime
        begin_raw_date = datetime.datetime.strptime(str(begin_raw), "%Y-%m-%d")
        begin_forecast_date = datetime.datetime.strptime(begin_forecast, "%Y-%m-%d")
        end_forecast_date = datetime.datetime.strptime(end_forecast, "%Y-%m-%d")

        # find forecast indexes
        initial_forecast_index = begin_forecast_date - begin_raw_date
        forecast_period_in_days = end_forecast_date - begin_forecast_date
        final_forecast_index = initial_forecast_index + forecast_period_in_days
        
        # get forecast from best instance
        yhat = instance_object.predict(initial_forecast_index.days, final_forecast_index.days)
        return yhat

    # save best instances
    def instance_save(self):
        metadata = None
        for name, func, instance, cfg, score in self.model_instances:
            instance_uuid = str(uuid.uuid1())
            instance.save(self.save_instance_path+instance_uuid+".pkl")
            metadata = self.save_metadata(instance_uuid, cfg, score)
        return metadata

    def save_metadata(self, instance_uuid, cfg, score):
        metadata = {}
        metadata['instance_id'] = instance_uuid
        metadata['cfg'] = cfg
        metadata['score'] = score
        metadata['region'] = self.instance_region
        metadata['data_begin_date'] = str(self.begin_training.date())
        metadata['data_end_date'] = str(self.end_raw.date())
        metadata['date_of_training'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata['model_type'] = list(self.model_type_dict.keys())[0]
        metadata['model_category'] = self.model_category
        with open(
            self.save_metadata_path + instance_uuid + ".json", "w"
        ) as json_to_save:
            json.dump(metadata, json_to_save, indent=4)
        return metadata

    # load local instance from id
    def load_instance_from_id(self, instance_id):
        import pickle
        with open(self.save_instance_path+instance_id+".pkl", 'rb') as f:
            instance_object = pickle.load(f)
        return instance_object

    # load local instance from metadata filename
    def load_instance_from_local_metadata_filename(self, instance_id):
        print(instance_id)
        metadata = json.load(open(self.save_metadata_path+instance_id+'.json'))
        return self.load_instance_from_id(metadata['instance_id'])

    def predictions_to_weboutput(self, yhat, begin, end):
        period = pd.date_range(begin, end)
        returned_dictionary = list()
        for date, value in zip(period, yhat):
            returned_dictionary.append(
                {
                    "date": datetime.datetime.strftime(date, "%Y-%m-%d"),
                    "prediction": str(value),
                }
            )
        return returned_dictionary