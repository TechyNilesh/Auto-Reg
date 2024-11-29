from .config import default_config_dict_reg as default_config
import random, math
import numpy as np
import river

class PipelineSearch:
    def __init__(self, config_dict=None,
                 feature_selection=True,
                 budget=10,
                 seed=42):
        self.config_dict = config_dict
        self.feature_selection = feature_selection
        self.pipeline_list = []
        self.budget = budget
        self.seed = seed
        self.random_pipeline_budget = math.ceil(self.budget/2)
        self.ardns_pipeline_budget = self.budget - self.random_pipeline_budget
        
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        if not config_dict:
            self.hyperparameters = default_config.get('hyperparameters', {})
            self.algorithms = [self._get_hyperparameters(algo, hyper_init="default") for algo in default_config.get('models', [])]
            self.preprocessing_steps = [self._get_hyperparameters(pre, hyper_init="default") for pre in default_config.get('preprocessors', [])]
            if self.feature_selection:
                self.feature_selection_methods = [self._get_hyperparameters(fs, hyper_init="default") for fs in default_config.get('features', [])]
        else:
            self.hyperparameters = config_dict.get('hyperparameters', {})
            self.algorithms = [self._get_hyperparameters(algo, hyper_init="default") for algo in config_dict.get('models', [])]
            self.preprocessing_steps = [self._get_hyperparameters(pre, hyper_init="default") for pre in config_dict.get('preprocessors', [])]
            if self.feature_selection:
                self.feature_selection_methods = [self._get_hyperparameters(fs, hyper_init="default") for fs in config_dict.get('features', [])]

    def _get_hyperparameters(self, model, hyper_init="random"):
        model_name = type(model).__name__
        model_hyper = self.hyperparameters[model_name]

        hyperparser = {}

        if hyper_init == "random":
            for key, values in model_hyper.items():
                hyperparser[key] = np.random.choice(values)
        else:
            default_hyper = model._get_params()
            for key, values in model_hyper.items():
                if default_hyper[key] in values:
                    hyperparser[key] = default_hyper[key]
                else:
                    hyperparser[key] = values[0]
        
        if river.__version__ >= '0.12.0':
            return model.clone(new_params=hyperparser)
        return model._set_params(hyperparser)
    
    def _create_pipelines(self):
        for preprocessing_step in self.preprocessing_steps:
            for model_instance in self.algorithms:
                # Create a pipeline without feature selection
                pipeline = preprocessing_step | model_instance
                self.pipeline_list.append(pipeline.clone())

                if self.feature_selection:
                    # Create a pipeline with feature selection
                    for feature_selection in self.feature_selection_methods:
                        full_pipeline = preprocessing_step | feature_selection | model_instance
                        self.pipeline_list.append(full_pipeline.clone())
        
        return self.pipeline_list
    
    def _initialize_random_pipeline(self,random_hyper=False):

        # Ensure that each component type is selected at least once
        random_algorithm = np.random.choice(self.algorithms)
        if len(self.preprocessing_steps)!=0:
            random_preprocessing = np.random.choice(self.preprocessing_steps)
        if self.feature_selection:
            if len(self.feature_selection_methods)!=0:
                random_feature_selection = np.random.choice(self.feature_selection_methods)
        
        if random_hyper:
            random_algorithm = self._get_hyperparameters(random_algorithm, hyper_init="random")
            random_preprocessing = self._get_hyperparameters(random_preprocessing, hyper_init="random")
            if self.feature_selection:
                random_feature_selection = self._get_hyperparameters(random_feature_selection, hyper_init="random")
        
        if self.feature_selection:
            new_choice = np.random.choice(["WITH_FEATURE", "WITHOUT_FEATURE"])
        else:
            new_choice = "WITHOUT_FEATURE"

        if new_choice == "WITH_FEATURE":
            random_feature_selection = np.random.choice(self.feature_selection_methods)
            return random_preprocessing | random_feature_selection | random_algorithm
        else:
            return random_preprocessing | random_algorithm
    
    def _get_current_params(self, model):
        model_name = type(model).__name__
        model_hyper = self.hyperparameters[model_name]
        current_hyper = {}
        for k, _ in model_hyper.items():
            current_hyper[k] = model._get_params()[k]
        return current_hyper

    def _ardns(self, current_value, values_list):

        nearby_option = np.random.choice(["same", "upper", "lower", "random"])

        if nearby_option == "same":
            return current_value
        elif nearby_option == "upper":
            return values_list[min(values_list.index(current_value) + 1, len(values_list) - 1)]
        elif nearby_option == "lower":
            return values_list[max(values_list.index(current_value) - 1, 0)]
        else:  # "random"
            return np.random.choice(values_list)

    def _suggest_nearby_hyperparameters(self, model):

        # Get the current hyperparameter values
        current_hyperparameters = self._get_current_params(model)

        # Get the user define hyperparameter values
        user_defined_search_space = self.hyperparameters[type(model).__name__]
        # print(user_defined_search_space)

        suggested_hyperparameters = {}

        for param_name, param_value in current_hyperparameters.items():
            if isinstance(param_value, int):
                # Integer hyperparameter
                new_value = self._ardns(param_value, user_defined_search_space[param_name])

                suggested_hyperparameters[param_name] = int(new_value)

            elif isinstance(param_value, float):
                # Float hyperparameter
                new_value = self._ardns(param_value, user_defined_search_space[param_name])

                suggested_hyperparameters[param_name] = float(new_value)

            elif isinstance(param_value, (str, bool)):
                # Categorical hyperparameter
                new_value = self._ardns(param_value, user_defined_search_space[param_name])

                suggested_hyperparameters[param_name] = new_value

            else:
                # Unsupported type (e.g., NoneType)
                # random value selected from search space
                suggested_hyperparameters[param_name] = np.random.choice(user_defined_search_space[param_name])

        return model._set_params(suggested_hyperparameters)
        # return model.clone(new_params=suggested_hyperparameters)
    
    def next_nerby(self, pipeline):

        next_nerby_pipeline = list(pipeline.steps.values())

        new_algorithm = self._suggest_nearby_hyperparameters(next_nerby_pipeline[-1])
        new_preprocessing = self._suggest_nearby_hyperparameters(next_nerby_pipeline[0])

        if len(next_nerby_pipeline) > 2:
            new_feature_selection = self._suggest_nearby_hyperparameters(next_nerby_pipeline[1])
            return new_preprocessing | new_feature_selection | new_algorithm
        else:
            return new_preprocessing | new_algorithm
    
    def select_and_update_pipelines(self, best_pipeline):
        
        pipeline = best_pipeline.clone()
        
        next_nerby_pipelines = []
        
        for _ in range(self.ardns_pipeline_budget):
            next_nerby_pipeline = self.next_nerby(pipeline)
            next_nerby_pipelines.append(next_nerby_pipeline)
            pipeline = next_nerby_pipeline.clone()
        
        new_pipelines = [self._initialize_random_pipeline(random_hyper=False).clone() for i in range(self.random_pipeline_budget)]
        self.pipeline_list = [best_pipeline.clone()] + next_nerby_pipelines + new_pipelines
        return self.pipeline_list