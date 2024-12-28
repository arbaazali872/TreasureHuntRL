# Treasure Hunt RL Project

## Overview

This project implements a reinforcement learning (RL) solution using the Q-Learning algorithm to navigate a 10x10 grid-based environment. The agent learns to reach a treasure while avoiding dynamic obstacles (monsters). The project employs the **Stable Baselines3** library for RL and adheres to object-oriented programming principles for clean and modular design.

---

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
    ```python
    import os
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    ```

5. Set up environment variables:
    - Populate the `config.env` file in the root directory with your desired parameters
---

## Running the Project

### Using `model_handler.py`
The `model_handler.py` file serves as the entry point for both training and testing the RL agent. The mode is selected by setting the `MODE` environment variable in the `.env` file. If no mode is selected, the script will both train and test the model.

1. **Training the Agent**:
    - Set `MODE=train` in the `.env` file.
    - Run the script:
      ```bash
      python model_handler.py
      ```
    - Trained models are saved in the `models/` directory.

2. **Testing the Agent**:
    - Set `MODE=test` in the `.env` file.
    - Run the script:
      ```bash
      python model_handler.py
      ```
    - Performance metrics and logs are saved in the `logs/` directory.

3. **Customizing Hyperparameters**:
    - Modify the hyperparameter values in the `.env` file before running the script to train with different configurations.

---

