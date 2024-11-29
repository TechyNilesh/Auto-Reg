#!pip install flaml[notebook,vw]==1.1.2
from flaml import AutoVW
from flaml.tune import loguniform, choice

import psutil
import time
import json
import random
import numpy as np

from capymoa.stream import stream_from_file
from capymoa.evaluation import RegressionEvaluator, RegressionWindowedEvaluator

import warnings
warnings.filterwarnings("ignore")

import argparse

def main(dataset_name:str,run_count:int=None,seed:int=42):
    
    #seed = random.randint(42,52)
    
    print(f"Loading dataset: {dataset_name}, Run Count: {run_count}, Random Seed:{seed}")

    stream = stream_from_file(f"RDatasets/{dataset_name}.arff")

    regressionEvaluator = RegressionEvaluator(schema=stream.get_schema())
    regressionWindowedEvaluator = RegressionWindowedEvaluator(schema=stream.get_schema(),window_size=1000)

    """ create an AutoVW instance for tuning namespace interactions and learning rate"""
    # set up the search space and init config
    search_space_nilr = {
        "interactions": AutoVW.AUTOMATIC,
        "learning_rate": loguniform(lower=2e-10, upper=1.0),
    }
    init_config_nilr = {"interactions": set(), "learning_rate": 0.5}
    # create an AutoVW instance
    autovw = AutoVW(
        max_live_model_num=5,
        search_space=search_space_nilr,
        init_config=init_config_nilr,
        random_seed=seed,
    )

    def to_vw_format(instance):
        res = f"{instance.y_value} |"
        for idx, value in enumerate(instance.x):
            res += f" {idx}:{value}"
        return res

    t=0
    times = []
    memories = []
    while stream.has_more_instances():
        instance = stream.next_instance()
        vw_instance = to_vw_format(instance)
        mem_before = psutil.Process().memory_info().rss # Recording Memory
        start = time.time()  # Recording Time
        try:
            prediction = autovw.predict(vw_instance)
        except Exception as e:
            print(f"Error while prediction: {e}")
            prediction = 0.0
        #print(f"y_true: {instance.y_value}, y_pred: {prediction}")
        regressionEvaluator.update(instance.y_value, prediction)
        regressionWindowedEvaluator.update(instance.y_value, prediction)
        try:
            autovw.learn(vw_instance)
        except Exception as e:
            print(f"Error while learning: {e}")
        end = time.time()
        mem_after = psutil.Process().memory_info().rss
        iteration_mem = mem_after - mem_before
        memories.append(iteration_mem)
        iteration_time = end - start
        times.append(iteration_time)
        t+=1
        if t%1000==0:
            print(f"Running Instance **{t}**")
            print(f"R2 score - {round(regressionEvaluator.R2(),3)}")
            print(f"RMSE score - {round(regressionEvaluator.RMSE(),3)}")
            print("-"*40)

    print("**Final Results**")
    print(f"R2 score - {round(regressionEvaluator.R2(),3)}")
    print(f"RMSE score - {round(regressionEvaluator.RMSE(),3)}")

    # saving results in dict
    save_record = {
        "model": 'CHACHA',
        "dataset": dataset_name,
        "regressionEvaluator": regressionEvaluator.metrics_dict(),
        "windows_scores": regressionWindowedEvaluator.metrics_per_window().to_dict(orient='list'),
        "time": times,
        "memory": memories
    }

    if run_count is not None:
        file_name = f"{save_record['model']}_{save_record['dataset']}_{run_count}.json"
    else:
        file_name = f"{save_record['model']}_{save_record['dataset']}.json"

    #To store the dictionary in a JSON file
    with open(f"TEMP/{file_name}", 'w') as json_file:  # change temp to  saved_results_json for final run
        json.dump(save_record, json_file)
    
    print(f"Results saved successfully in TEMP/{file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CHACHA on a dataset')
    parser.add_argument('--dataset', type=str, help='Dataset Name')
    parser.add_argument('--run_count', type=int, default=None, help='Run Count')
    parser.add_argument('--seed', type=int, default=42, help='Random Seed')
    args = parser.parse_args()
    main(dataset_name=args.dataset,
         run_count=args.run_count,
         seed=args.seed)