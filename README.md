# Treasure Hunt RL Project

## Overview

This project implements a reinforcement learning (RL) Agent using the Q-Learning algorithm to navigate a 10x10 grid-based environment. The agent learns to reach a treasure while avoiding dynamic obstacles (monsters). The project employs the **Stable Baselines3** library for RL and adheres to object-oriented programming principles for clean and modular design.

The Agent was trained on multiple sets of diffrent hyperparameters and its performance was judged on the basis of the following parameters

- Average reward
- Success rate 
- Average Steps

The best results obtained after multiple tests episodes on models trained across diffrent hyperparameters were as follows

| Metric           | Value |
|-------------------|-------|
| **Average Reward** | 13.54 |
| **Success Rate**   | 76%   |
| **Average Steps**  | 8.84  |


## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual Environment (recommended)

### Steps

1. Clone the repository:
    ```bash
    git clone git@github.com:arbaazali872/TreasureHuntRL.git
    cd TreasureHuntRL
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Create directories for saving logs and models:
    ```bash
    mkdir models logs metrices     
    ```

5. Set up environment variables:
    - Populate the `config.env` file in the root directory with your desired training parameters

## Running the Project

### Using `model_handler.py`
The `model_handler.py` file serves as the entry point for both training and testing the RL agent. The mode is selected by setting the `MODE` environment variable in the `config.env` file. If no mode is selected, the script will both train and test the model.

1. **Training the Agent**:
    - Set `MODE=train` in the `config.env` file.
    - Run the script:
      ```bash
      python model_handler.py
      ```
    - Trained models are saved in the `models/` directory.

2. **Testing the Agent**:
    - Set `MODE=test` in the `config.env` file.
    - Run the script:
      ```bash
      python model_handler.py
      ```
    - Performance metrics and logs are saved in the `logs/` directory.

### Using `hyperparameter_tester.py`
Modify the hyperparameter values in the `hyperparameters.json` file before running the script to train with different configurations.

- Run the script:
    
    ```bash
    python hyperparameter_tester.py
    ```

- Performance metrics are saved in the `metrices/` directory.

**To read the whole project report containing training parameters and agent's performances [Click Here](https://drive.google.com/file/d/1z55SWv3yn1JfWjG1lUMa26UsZvUG55ns/view?usp=sharing)**

**The documentation of the code can be found in the docs folder in the root directory of the project**