# AUTO-REG

A Dynamic AutoML Framework for Streaming Regression

### Data Streams Regression Benchmarking Dataset

The dataset files can be found in the repository folder `RDatasets`.

1. **Ailerons**: A synthetic dataset with 40 features and 13,750 instances, designed to simulate the control of an F16 aircraft.
2. **Elevators**: A synthetic dataset including 18 features and 16,599 instances, representing data related to the control of an elevator system.
3. **Fried**: A synthetic dataset with 10 features and 40,768 instances, generated using a non-linear formula to test the robustness of regression algorithms.
4. **Friedman GRA**: A synthetic dataset with 10 features and 100,000 instances, designed to simulate Global Recurring Abrupt drift.
5. **Friedman GSG**: A synthetic dataset with 10 features and 100,000 instances, designed to simulate Global and Slow Gradual drift.
6. **Friedman LEA**: A synthetic dataset with 10 features and 100,000 instances, designed to simulate Local Expanding Abrupt drift.
7. **Hyper (A)**: A synthetic dataset with 10 features and 500,000 instances, generating a hyperplane in a \(d\)-dimensional space to assess drift detection ability.
8. **Abalone**: A real-world dataset with 8 features and 4,977 instances, used to predict the age of abalones based on physical measurements.
9. **Bike**: A real-world dataset with 12 features and 17,379 instances, providing data on bike-sharing rentals.
10. **Cpu Activity**: A real-world dataset with 22 features and 8,192 instances, representing measurements of CPU performance.
11. **House8L**: A real-world dataset with 8 features and 22,784 instances, related to house prices.
12. **Kin8nm**: A real-world dataset with 9 features and 8,192 instances, related to forward kinematics of an 8-link robot arm.
13. **Metro Traffic**: A real-world dataset with 7 features and 48,204 instances, providing traffic data for a metropolitan area.
14. **White Wine**: A real-world dataset with 11 features and 4,898 instances, containing chemical properties of white wine to predict its quality.

These datasets ensure a comprehensive evaluation of regression algorithms in diverse scenarios, combining both synthetic and real-world data.

### Running Experiments

To execute the benchmarking experiments, we use a Python script that orchestrates the execution of multiple models on various datasets. The script utilizes concurrent execution to run these experiments efficiently.

1. **Install Requirements**: Before running the experiments, ensure all required dependencies are installed by using `pip` to install the `requirements.txt` file. This can be done with the following command:
   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset and Model Lists**: 
   - **Datasets**: The list of datasets includes `ailerons`, `elevators`, `fried`, `hyperA`, `FriedmanGsg`, `FriedmanGra`, `FriedmanLea`, `kin8nm`, `abalone`, `bike`, `House8L`, `MetroTraffic`, `cpu_activity`, and `white_wine`.
   - **Models**: The list of models includes `auto_reg`, `asml_reg`, `chacha`, `soknl`, `arf`, `hat`, and `eaml`.

3. **Execution Modes**: 
   - **Single Model on All Datasets**: runs a specific model on all datasets concurrently.
   - **All Models on a Single Dataset**: runs all models on a specific dataset concurrently.
   - **All Models on All Datasets**: runs all models on all datasets concurrently.
   - **Randomized Runs**: runs all models on all datasets with multiple runs (10 times) using random seeds.

4. **Command-Line Interface**: The script uses `argparse` to provide a command-line interface. The user can specify the mode (`dataset`, `model`, `all`, or `random`) and optionally the name of the dataset or model to run specific experiments.

### Example Commands

- To run all models on a specific dataset:
  ```bash
  python run_scripts.py --mode dataset --name ailerons
  ```

- To run a specific model on all datasets:
  ```bash
  python run_scripts.py --mode model --name auto_reg
  ```

- To run all models on all datasets:
  ```bash
  python run_scripts.py --mode all
  ```

- To run all models on all datasets with multiple runs and random seeds:
  ```bash
  python run_scripts.py --mode random
  ```

This setup ensures comprehensive and efficient execution of benchmarking experiments across different models and datasets.

