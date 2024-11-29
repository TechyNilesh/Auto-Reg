from river import base, metrics
from .search import PipelineSearch
import numpy as np
import pandas as pd
import random
import copy


class AutoStreamRegressor(base.Regressor):
    def __init__(
        self,
        config_dict=None,
        metric=metrics.RMSE(),
        exploration_window=1000,
        budget=10,
        ensemble_size=5,
        prediction_mode='ensemble', #best, ensemble
        feature_selection=True,
        verbose=False,
        aggregation_method='mean', #mean, median
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
            aggregation_method=aggregation_method,
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
        self.aggregation_method = aggregation_method
        
        self.current_score = None
        self.alpha = 0.5
        self.lambda_ = 0.1 #np.round(np.random.uniform(0.1, 1.0),5)
        self.step_size = 0.1#np.random.uniform(0.01, 0.1)

        if self.seed is not None:
             random.seed(self.seed)
             np.random.seed(self.seed)

        self.pipe_search = PipelineSearch(config_dict=self.config_dict,
                                          budget=self.budget-1,
                                          feature_selection=self.feature_selection,
                                          seed=self.seed)
        
        self.init_pipeline_list = self.pipe_search._create_pipelines()
        self.pipeline_list = [self.init_pipeline_list[np.random.randint(len(self.init_pipeline_list))] for _ in range(self.budget)]
        self._metrics = [type(self.metric)() for _ in range(len(self.pipeline_list))]
        self._best_model_idx = np.random.randint(len(self.pipeline_list))
        self.best_model = self.pipeline_list[self._best_model_idx]
        self.prediction_mode = prediction_mode

        self.interpret_info = []

        if self.prediction_mode == 'ensemble':
            self.ensemble_size = ensemble_size
            self.model_snapshots = [self.pipeline_list[np.random.randint(len(self.pipeline_list))] for _ in range(self.ensemble_size)]
            self.model_snapshots_metrics = [type(self.metric)() for _ in range(self.ensemble_size)]
    
    @staticmethod
    def validate_params(config_dict, metric, exploration_window, budget, ensemble_size, prediction_mode, feature_selection, verbose, aggregation_method, seed):
        if prediction_mode not in ['best', 'ensemble']:
            raise ValueError("prediction_mode must be string and either 'best' or 'ensemble'")
        if aggregation_method not in ['mean', 'median']:
            raise ValueError("aggregation_method must be string and either 'mean' or 'median'")
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
        
        #self.step_size = np.random.uniform(0.01, 0.1)

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
    
    def update_interpret_info(self, metrics_dict, pipe, y_true, y_pred):

        del metrics_dict['classified instances']

        self.interpret_info.append({
            'instance': self.timestep,
            'pipeline': tuple(self.best_model.steps.keys()),
            'pre_hyper': self.pipe_search._get_current_params(list(pipe.steps.values())[0]),
            'feature_hyper': self.pipe_search._get_current_params(list(pipe.steps.values())[1]) if len(list(pipe.steps.values())) == 3 else None,
            'model_hyper': self.pipe_search._get_current_params(list(pipe.steps.values())[-1]),
            'y_true': y_true,
            'y_pred': y_pred,
            'search_probablity': self.lambda_,
        } | metrics_dict)

        #print("interpret updated successfully at timestep: {}".format(self.timestep))
    
    def get_interpret_info(self):
        return pd.DataFrame(self.interpret_info)

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
            
            if self.aggregation_method == 'median':
                return np.median(predictions) if predictions else 0.0
            
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
    
    def adjust_lambda(self):
        best_score = self._metrics[self._best_model_idx]

        if self.current_score is None:
            self.current_score = copy.deepcopy(best_score)
            if self.verbose:
                print("Initial run: setting current_score to best_score")
            return

        # Calculate performance improvement
        #self.step_size *= abs(self.current_score.get() - best_score.get())/1.0
        
        # Calculate performance improvement ratio
        #performance_ratio = abs(self.current_score.get() - best_score.get()) / (abs(self.current_score.get()) + 1e-9)
        #self.step_size = min(max(performance_ratio, 0.01), 0.1)  # Step size in range [0.01, 0.1]
        
        print("Step size:", self.step_size)

        if best_score.is_better_than(self.current_score):
            self.alpha = max(0, self.alpha - self.step_size)  # Decrease probability size
            self.lambda_ = max(0, self.lambda_ - self.step_size)
            if self.verbose:
                print("Decreasing lambda to:", self.lambda_)
        else:
            self.alpha = min(1, self.alpha + self.step_size)  # Increase probability size
            self.lambda_ = min(1, self.lambda_ + self.step_size)
            if self.verbose:
                print("Increasing lambda to:", self.lambda_)

        self.current_score = copy.deepcopy(best_score)
    
    # def adjust_new_pipeline_probability(self):
        
    #     best_score = self._metrics[self._best_model_idx]

    #     if self.current_score is None:
    #         self.current_score = copy.deepcopy(best_score)
    #         if self.verbose:
    #             print("Initial run: setting current_score to best_score")
    #         return

    #     # Check if the new best score is better (lower RMSE or higher R2) than the current best score
    #     if best_score.is_better_than(self.current_score):
    #         self.new_pipeline_probability = max(0, self.new_pipeline_probability - self.step_size)  # Decrease probability size
    #         self.lambda_ = max(0, self.lambda_ - self.step_size)
    #         if self.verbose:
    #             print("Decreasing probability to:", self.new_pipeline_probability)
    #     else:
    #         self.new_pipeline_probability = min(1, self.new_pipeline_probability + self.step_size)  # Increase probability size
    #         self.lambda_ = min(1, self.lambda_ + self.step_size)
    #         if self.verbose:
    #             print("Increasing probability to:", self.new_pipeline_probability)
        
    #     self.current_score = copy.deepcopy(best_score)

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
            
            self.adjust_lambda()
            self.pipeline_list = self.pipe_search.select_and_update_pipelines(self.best_model,
                                                                              lambda_ = self.lambda_,
                                                                              alpha = self.alpha)
            self.reset_exploration()
            
    def reset(self):
        self.__init__()