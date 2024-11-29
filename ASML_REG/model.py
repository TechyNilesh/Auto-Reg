from river import base, metrics
from .search import PipelineSearch
import numpy as np
import random


class AutoStreamRegressor(base.Regressor):
    def __init__(
        self,
        config_dict=None,
        metric=metrics.RMSE(),
        exploration_window=1000,
        budget=10,
        ensemble_size=3,
        prediction_mode='ensemble', #best, ensemble
        feature_selection=True,
        verbose=False,
        seed=42
    ):
        self.validate_params(
            config_dict=config_dict,
            metric=metric,
            exploration_window=exploration_window,
            budget=budget,
            ensemble_size=ensemble_size,
            prediction_mode=prediction_mode,
            feature_selection=feature_selection,
            verbose=verbose,
            seed=seed
        )
        self.metric = metric
        self.exploration_window = exploration_window
        self.budget = budget
        self.config_dict = config_dict
        self.feature_selection = feature_selection
        self.timestep = 0
        self.verbose = verbose
        self.seed = seed
        
        self.current_score = None
        self.new_pipeline_probability= 0.5

        if self.seed is not None:
             random.seed(self.seed)
             np.random.seed(self.seed)

        self.pipe_search = PipelineSearch(config_dict=self.config_dict,
                                          budget=self.budget-1,
                                          feature_selection=self.feature_selection,
                                          seed=self.seed)
        
        self.pipeline_list = self.pipe_search._create_pipelines()
        self._metrics = [type(self.metric)() for _ in range(len(self.pipeline_list))]
        self._best_model_idx = np.random.randint(len(self.pipeline_list))
        self.best_model = self.pipeline_list[self._best_model_idx]
        self.prediction_mode = prediction_mode

        if self.prediction_mode == 'ensemble':
            self.ensemble_size = ensemble_size
            self.model_snapshots = [self.pipeline_list[np.random.randint(len(self.pipeline_list))] for _ in range(self.ensemble_size)]
            self.model_snapshots_metrics = [type(self.metric)() for _ in range(self.ensemble_size)]
    
    @staticmethod
    def validate_params(config_dict, metric, exploration_window, budget, ensemble_size, prediction_mode, feature_selection, verbose, seed):
        if prediction_mode not in ['best', 'ensemble']:
            raise ValueError("prediction_mode must be string and either 'best' or 'ensemble'")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("seed must be an integer or None")
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be a boolean")
        if not isinstance(feature_selection, bool):
            raise ValueError("feature_selection must be a boolean")
        if not isinstance(exploration_window, int):
            raise ValueError("exploration_window must be an integer")
        if not isinstance(budget, int):
            raise ValueError("budget must be an integer")
        if not isinstance(ensemble_size, int):
            raise ValueError("ensemble_size must be an integer")
        if not isinstance(metric, metrics.base.RegressionMetric):
            raise ValueError("metric must be a river metric")
        if config_dict is not None and not isinstance(config_dict, dict):
            raise ValueError("config_dict must be a dictionary or None")

    def reset_exploration(self):
        self._metrics = [type(self.metric)() for _ in range(len(self.pipeline_list))]
        self._best_model_idx = np.random.randint(len(self.pipeline_list))
        if self.prediction_mode == 'ensemble':
            self.model_snapshots_metrics = [type(self.metric)() for _ in range(len(self.model_snapshots))]

    def print_batch_info(self):
        print(f"Data Point: {self.timestep}")
        try:
            print(f"Best Pipeline: {self.best_model}")
            print(f"Best Preprocessor Hyper: {self.pipe_search._get_current_params(list(self.best_model.steps.values())[0])}")
            if len(list(self.best_model.steps.values())) == 3:
                print(f"Best Feature Hyper: {self.pipe_search._get_current_params(list(self.best_model.steps.values())[1])}")
            print(f"Best Model Hyper: {self.pipe_search._get_current_params(list(self.best_model.steps.values())[-1])}")
        except Exception:
            pass
        print("----------------------------------------------------------------------")

    def predict_one(self, x):
        if self.prediction_mode == 'best':
            try:
                return self.best_model.predict_one(x)
            except Exception:
                return 0.0
        else:
            predictions = []
            for reg in self.model_snapshots:
                try:
                    predictions.append(reg.predict_one(x))
                except Exception:
                    continue
            
            return np.mean(predictions) if predictions else 0.0

    def learn_one(self, x, y):
        for idx, _ in enumerate(self.pipeline_list):
            try:
                y_pred = self.pipeline_list[idx].predict_one(x)
                self._metrics[idx].update(y, y_pred)
                self.pipeline_list[idx].learn_one(x, y)
                if self._metrics[idx].is_better_than(self._metrics[self._best_model_idx]):
                    self._best_model_idx = idx
            except Exception:
                continue

        if self.prediction_mode == 'best':
            try:
                self.best_model.learn_one(x, y)
            except Exception:
                pass
        else:
            for idx, _ in enumerate(self.model_snapshots):
                try:
                    y_pred = self.model_snapshots[idx].predict_one(x)
                    self.model_snapshots_metrics[idx].update(y, y_pred)
                    self.model_snapshots[idx].learn_one(x, y)
                except Exception:
                    continue
        self.timestep += 1
        self._check_exploration_phase()

    def _check_exploration_phase(self):
        if self.timestep % self.exploration_window == 0:
            self.best_model = self.pipeline_list[self._best_model_idx]
            if self.prediction_mode == 'ensemble':
                if len(self.model_snapshots) >= self.ensemble_size:
                    if type(self.metric).__name__ in ['R2']:
                        worst_idx = np.argmin([m.get() for m in self.model_snapshots_metrics])
                    else:
                        worst_idx = np.argmax([m.get() for m in self.model_snapshots_metrics])
                    self.model_snapshots.pop(worst_idx)
                    self.model_snapshots_metrics.pop(worst_idx)
                self.model_snapshots.append(self.best_model)
                self.model_snapshots_metrics.append(type(self.metric)())

            if self.verbose:
                self.print_batch_info()

            self.pipeline_list = self.pipe_search.select_and_update_pipelines(self.best_model)
            self.reset_exploration()
            
    def reset(self):
        self.__init__()