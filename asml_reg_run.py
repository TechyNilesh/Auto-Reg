from ASML_REG_BASE import AutoStreamRegressorBase
import psutil
import time
import json

from capymoa.stream import stream_from_file
from capymoa.evaluation import RegressionEvaluator, RegressionWindowedEvaluator

from river import metrics

import warnings
warnings.filterwarnings("ignore")

import argparse

def main(dataset_name:str,
         run_count:int=None,
         EW:int=1000,
         ES:int=3,
         B:int=10,
         PM:str='ensemble',
         seed:int=42):

    print(f"Loading dataset: {dataset_name}, Run Count: {run_count}, Random Seed:{seed}")
    print(f"Current Hyperparameters: EW - {EW}, ES - {ES}, B - {B}, PM - {PM}")
    
    stream = stream_from_file(f"RDatasets/{dataset_name}.arff")

    regressionEvaluator = RegressionEvaluator(schema=stream.get_schema())
    regressionWindowedEvaluator = RegressionWindowedEvaluator(schema=stream.get_schema(),window_size=1000)

    ASR = AutoStreamRegressorBase(config_dict=None, #config_dict
        exploration_window=EW, # Window Size
        prediction_mode=PM, #change 'best' or 'ensemble' if you want best model prediction 
        budget=B,# How many pipelines run concurrently
        ensemble_size=ES, # Ensemble size 
        metric=metrics.RMSE(), # Online metrics
        feature_selection = True,
        verbose=False,
        seed=seed) # Random/Fixed seed

    t=0
    times = []
    memories = []
    while stream.has_more_instances():
        instance = stream.next_instance()
        x = dict(enumerate(instance.x))
        mem_before = psutil.Process().memory_info().rss # Recording Memory
        start = time.time()  # Recording Time
        prediction = ASR.predict_one(x)
        #print(f"y_true: {instance.y_value}, y_pred: {prediction}")
        regressionEvaluator.update(instance.y_value, prediction)
        regressionWindowedEvaluator.update(instance.y_value, prediction)
        ASR.learn_one(x, instance.y_value)
        end = time.time()
        mem_after = psutil.Process().memory_info().rss
        iteration_mem = mem_after - mem_before
        memories.append(iteration_mem)
        iteration_time = end - start
        times.append(iteration_time)
        t+=1
        #print(f"Running Instance....{t}",end='\r')
        if t%1000==0:
            print(f"Running Instance **{t}**")
            print(f"R2 score - {round(regressionEvaluator.R2(),5)}")
            print(f"RMSE score - {round(regressionEvaluator.RMSE(),5)}")
            print("-"*40)

    print("**Final Results**")
    print(f"R2 score - {round(regressionEvaluator.R2(),3)}")
    print(f"RMSE score - {round(regressionEvaluator.RMSE(),3)}")
    
    # saving results in dict
    save_record = {
        "model": f'ASML_REG_BASE_BEST' if PM == 'best' else f'ASML_REG_BASE',
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
    parser = argparse.ArgumentParser(description="AutoStreamML Regression Script")
    parser.add_argument('--dataset', type=str, help='Dataset Name', required=True)
    parser.add_argument('--run_count', type=int, default=None, help='Run Count', required=False)
    parser.add_argument('--exploration_window', type=int, default=1000, help='Exploration Window Size', required=False)
    parser.add_argument('--ensemble_size', type=int, default=5, help='Ensemble Size', required=False)
    parser.add_argument('--budget', type=int, default=10, help='Budget', required=False)
    parser.add_argument('--prediction_mode', type=str,default='ensemble', help='Prediction Mode', required=False)
    parser.add_argument('--seed', type=int, default=42, help='Random Seed', required=False)
    args = parser.parse_args()
    main(dataset_name=args.dataset,
         run_count=args.run_count,
         EW=args.exploration_window,
         ES=args.ensemble_size,
         B=args.budget,
         PM=args.prediction_mode,
         seed=args.seed)