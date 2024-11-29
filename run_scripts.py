import concurrent.futures
import subprocess
import random
import argparse

# List of dataset names
dataset_name_list = [
    # 'ailerons',
    # 'elevators',
    'fried',
    # 'hyperA',
    'FriedmanGsg',
    'FriedmanGra',
    'FriedmanLea',
    'kin8nm',
    'abalone',
    'bike',
    'House8L',
    'MetroTraffic',
    'cpu_activity',
    'white_wine',
]

models = [
    # 'chacha',
    # 'soknl',
    # 'arf',
    # 'hat',
    # 'eaml',
    'asml',
    # 'asml_base',
    # 'asml_base_best',
    # 'asml_med',
    # 'asml_best',
    # 'asml_moa'
]

run_counts = range(27, 101)

def run_script(dataset_name, model_name, run_count=None, random_seed=False):
    """
    Function to execute a given script for a specified dataset name, model name, and run count.
    """
    if random_seed:
        seed = random.randint(1, 101)
    else:
        seed = 42

    print(f"Running {model_name.upper()} on {dataset_name} dataset with {run_count} runs and seed {seed}")

    command = ['python']
    if model_name == 'asml':
        command += ['asml_reg_run.py', '--dataset', dataset_name]
    elif model_name == 'asml_med':
        command += ['asml_reg_run.py', '--dataset', dataset_name, '--aggregation_method', 'median']
    elif model_name == 'asml_best':
        command += ['asml_reg_run.py', '--dataset', dataset_name, '--prediction_mode', 'best']
    elif model_name == 'asml_base':
        command += ['asml_reg_base_run.py', '--dataset', dataset_name]
    elif model_name == 'asml_base_best':
        command += ['asml_reg_base_run.py', '--dataset', dataset_name, '--prediction_mode', 'best']
    elif model_name == 'eaml':
        command += ['eaml_reg_run.py', '--dataset', dataset_name]
    elif model_name == 'chacha':
        command += ['chacha_run.py', '--dataset', dataset_name]
    elif model_name == 'soknl':
        command += ['soknl_run.py', '--dataset', dataset_name]
    elif model_name == 'arf':
        command += ['arf_reg_run.py', '--dataset', dataset_name]
    elif model_name == 'hat':
        command += ['hat_reg_run.py', '--dataset', dataset_name]
    elif model_name == 'asml_moa':
        command += ['asml_moa_reg_run.py', '--dataset', dataset_name]
    else:
        print('Invalid model name')
        return

    if run_count is not None:
        command += ['--run_count', str(run_count)]
    
    command += ['--seed', str(seed)]

    # Execute the command using Popen and print output in real-time
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    for line in process.stdout:
        print(line, end='')  # Print each line of stdout in real-time

    process.wait()  # Wait for the process to complete

    # Check if there were any errors and print them
    if process.returncode != 0:
        print(f"Dataset {dataset_name}: Error\n{process.stderr.read()}")


def run_all_datasets_for_model(model):
    print(f"*** Running all datasets for {model.upper()} model ***")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(run_script, dataset_name_list, [model] * len(dataset_name_list))
        executor.shutdown(wait=True)

def run_all_models_for_dataset(dataset):
    print(f"*** Running all models for {dataset.upper()} dataset ***")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(run_script, [dataset] * len(models), models)
        executor.shutdown(wait=True)

def run_all_models_and_datasets():
    print("*** Running all models and datasets concurrently ***")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tasks = [(dataset, model) for dataset in dataset_name_list for model in models]
        executor.map(run_script, [task[0] for task in tasks], [task[1] for task in tasks])
        executor.shutdown(wait=True)

def run_all_with_random():
    print("*** Running all models and datasets with multiple 10 runs concurrently ***")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tasks = [(dataset, model, run)
                 for run in run_counts for dataset in dataset_name_list for model in models]
        executor.map(run_script, [task[0] for task in tasks], [
                     task[1] for task in tasks], [task[2] for task in tasks], [True] * len(tasks))
        executor.shutdown(wait=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run models on datasets.")
    parser.add_argument('--mode', choices=['dataset', 'model', 'all', 'random'], required=True, help="Choose the mode to run: dataset, model, all, random")
    parser.add_argument('--name', help="Name of the dataset or model to run in specific mode")

    args = parser.parse_args()

    if args.mode == 'dataset':
        if not args.name:
            print("Please provide the name of the dataset using --name")
        else:
            run_all_models_for_dataset(args.name)
    elif args.mode == 'model':
        if not args.name:
            print("Please provide the name of the model using --name")
        else:
            run_all_datasets_for_model(args.name)
    elif args.mode == 'all':
        run_all_models_and_datasets()
    elif args.mode == 'random':
        run_all_with_random()
