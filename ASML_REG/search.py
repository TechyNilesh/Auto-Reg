from .config import default_config_dict_reg as default_config
import random
import numpy as np
import river
from numpy.random import default_rng
from scipy.stats import truncnorm



class PipelineSearch:
    def __init__(self, config_dict=None,
                 feature_selection=True,
                 budget=10,
                 seed=42):
        self.config_dict = config_dict or default_config
        self.feature_selection = feature_selection
        self.pipeline_list = []
        self.budget = budget
        self.seed = seed
        
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.hyperparameters = self.config_dict.get('hyperparameters', {})
        self.algorithms = [self._get_hyperparameters(algo, hyper_init="default") for algo in self.config_dict.get('models', [])]
        self.preprocessing_steps = [self._get_hyperparameters(pre, hyper_init="default") for pre in self.config_dict.get('preprocessors', [])]
        
        if self.feature_selection:
            self.feature_selection_methods = [self._get_hyperparameters(fs, hyper_init="default") for fs in self.config_dict.get('features', [])]

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
                hyperparser[key] = default_hyper.get(key, values[0])
        
        if river.__version__ >= '0.12.0':
            return model.clone(new_params=hyperparser)
        return model._set_params(hyperparser)
    
    def _create_pipelines(self):
        for preprocessing_step in self.preprocessing_steps:
            for model_instance in self.algorithms:
                pipeline = preprocessing_step | model_instance
                self.pipeline_list.append(pipeline.clone())

                if self.feature_selection:
                    for feature_selection in self.feature_selection_methods:
                        full_pipeline = preprocessing_step | feature_selection | model_instance
                        self.pipeline_list.append(full_pipeline.clone())
        
        return self.pipeline_list
    
    def _initialize_random_pipeline(self, random_hyper=False):
        random_algorithm = np.random.choice(self.algorithms)
        random_preprocessing = np.random.choice(self.preprocessing_steps) if self.preprocessing_steps else None
        random_feature_selection = np.random.choice(self.feature_selection_methods) if self.feature_selection else None
        
        if random_hyper:
            random_algorithm = self._get_hyperparameters(random_algorithm, hyper_init="random")
            if random_preprocessing:
                random_preprocessing = self._get_hyperparameters(random_preprocessing, hyper_init="random")
            if self.feature_selection:
                random_feature_selection = self._get_hyperparameters(random_feature_selection, hyper_init="random")
        
        if self.feature_selection and random_feature_selection:
            return random_preprocessing | random_feature_selection | random_algorithm
        return random_preprocessing | random_algorithm
    
    def _get_current_params(self, model):
        model_name = type(model).__name__
        model_hyper = self.hyperparameters[model_name]
        current_hyper = {k: model._get_params()[k] for k in model_hyper.keys()}
        return current_hyper
    
    def _pwhs(self, current_value, options, lambda_=0.5):
        #print(f"current_value: {current_value}, options: {options}")
        if isinstance(current_value, (int, float)):
            lower_bound, upper_bound = min(options), max(options)
            
            mu = current_value
            
            sigma = np.std(options) * (1 + lambda_)

            #sigma = (upper_bound-lower_bound)*lambda_

            # Truncate the normal distribution to stay within bounds
            a_norm, b_norm = (lower_bound - mu) / sigma, (upper_bound - mu) / sigma

            try:
                # Sample a value from the truncated normal distribution
                new_value = truncnorm.rvs(a_norm, b_norm, loc=mu, scale=sigma)
                # Find the closest value in the options to the sampled value
                new_value = min(options, key=lambda x: abs(x - new_value))
            except ValueError:
                print(f"**ValueError: {a_norm}, {b_norm}, {mu}, {sigma}**")
                print("**Assigning random value**")
                new_value = np.random.choice(options)
            
            if isinstance(current_value, int):
                new_value = int(round(new_value))
        
        else: # Categorical or boolean hyperparameter and other types
            # Assign probabilities based on lambda_
            try:
                probabilities = [(1 - lambda_) if val == current_value else lambda_ / (len(options) - 1) for val in options]
                new_value = np.random.choice(options, p=probabilities)
            except:
                print(f"**Error: {current_value}, {options}, {lambda_}**")
                print("**Assigning random value**")
                new_value = np.random.choice(options)
        
        return new_value
        
    # def _pwhs(self, theta, H, lambda_=0.5):
        
    #     #rng = default_rng()

    #     if isinstance(theta, (int, float)):
    #         mu = np.mean(H)
    #         sigma = np.std(H)
    #         distance_from_mean = abs(theta - mu)
    #         adjusted_sigma = sigma * (1 + lambda_ * (distance_from_mean / mu))
    #         a, b = min(H), max(H)
    #         # Generate a truncated normal distribution
    #         truncated_value = np.random.normal(loc=mu, scale=adjusted_sigma)
    #         # Clip the value within the range [a, b]
    #         new_value = np.clip(truncated_value, a, b)
    #         if isinstance(theta, int):
    #             new_value = int(round(new_value))
    #     elif isinstance(theta, (str, bool)):
    #         probabilities = [(1 - lambda_) / (len(H) - 1) if val != theta else lambda_ for val in H]
    #         new_value = np.random.choice(H, p=probabilities)
    #     else:
    #         new_value = np.random.choice(H)
        
    #     return new_value
    # def _pwhs(self, theta, H, lambda_=0.5):
    #     if isinstance(theta, (int, float)):
    #         range_H = max(H) - min(H)
    #         sigma = lambda_ * range_H  # Scale sigma inversely to lambda
    #         mu = theta
    #         # Sample from Gaussian distribution
    #         theta_star = np.random.normal(mu, sigma)
    #         # Clip to valid range
    #         theta_star = np.clip(theta_star, min(H), max(H))
    #         if isinstance(theta, int):
    #             # Round to nearest integer if the current value is an integer
    #             theta_star = int(round(theta_star))
    #     elif isinstance(theta, str):
    #         d = len(H)
    #         if random.random() < lambda_:
    #             theta_star = theta  # Stick with current value
    #         else:
    #             H_prime = [x for x in H if x != theta]  # Remove current value
    #             theta_star = random.choice(H_prime)  # Uniform sampling from other values
    #     elif isinstance(theta, bool):
    #         p = (1 - lambda_) / 2  # Probability of flipping
    #         theta_star = not theta if random.random() < p else theta
    #     else:
    #         theta_star = random.choice(H)
        
    #     return theta_star

    def pwhs_main(self, model, lambda_=0.5):
        current_hyperparameters = self._get_current_params(model)
        user_defined_search_space = self.hyperparameters[type(model).__name__]
        suggested_hyperparameters = {
            param_name: self._pwhs(param_value, user_defined_search_space[param_name], lambda_)
            for param_name, param_value in current_hyperparameters.items()
        }

        if river.__version__ >= '0.12.0':
            return model.clone(new_params=suggested_hyperparameters)
        return model._set_params(suggested_hyperparameters)
    
    def next_nearby(self, pipeline, lambda_=0.5):
        next_nearby_pipeline = list(pipeline.steps.values())
        new_algorithm = self.pwhs_main(next_nearby_pipeline[-1], lambda_)
        new_preprocessing = self.pwhs_main(next_nearby_pipeline[0], lambda_)

        if len(next_nearby_pipeline) > 2:
            new_feature_selection = self.pwhs_main(next_nearby_pipeline[1], lambda_)
            return new_preprocessing | new_feature_selection | new_algorithm
        return new_preprocessing | new_algorithm
    
    def select_and_update_pipelines(self, best_pipeline, lambda_=0.1,alpha=0.5):
        print(f"lambda: {lambda_}")
        pipeline = best_pipeline.clone()
        next_nearby_pipelines = []
        random_pipeline_budget = np.random.binomial(self.budget, alpha)
        nearby_pipeline_budget = self.budget - random_pipeline_budget
        #lambda_ = max(min(new_pipeline_probability, 1), 0)
        
        for _ in range(nearby_pipeline_budget):
            next_nearby_pipeline = self.next_nearby(pipeline, lambda_=lambda_)
            next_nearby_pipelines.append(next_nearby_pipeline)
            pipeline = next_nearby_pipeline.clone()
        
        new_pipelines = [self._initialize_random_pipeline(random_hyper=False).clone() for _ in range(random_pipeline_budget)]
        self.pipeline_list = [best_pipeline.clone()] + next_nearby_pipelines + new_pipelines
        return self.pipeline_list
