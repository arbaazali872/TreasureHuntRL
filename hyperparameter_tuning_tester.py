from logger_setup import setup_logger
from game_env import GameEnvironment
from stable_baselines3 import DQN
import os
import json


class HyperParamTester:
    """
    A class for testing multiple hyperparameter configurations for a DQN model 
    in a custom game environment.

       Attributes:
        hyperparameter_configs (list): List of hyperparameter configurations.
        total_timesteps (int): Total number of timesteps for training.
        total_episodes (int): Total number of episodes for testing.
        metrics (dict): Dictionary to store metrics for each configuration.
        env (GameEnvironment): The game environment instance.
        logger (Logger): Logger for logging training and testing activities.

    Class Attributes:
        metrics_dir (str): Directory for saving metrics.
        logs_dir (str): Directory for saving logs.
        models_dir (str): Directory for saving trained models.
        metrics_path (str): Path for saving the metrics as a JSON file.
    """
    metrics_dir = "metrices"
    logs_dir = "logs"
    models_dir = "models"
    metrics_path = os.path.join(metrics_dir, "training_metrics.json")

    def __init__(self, hyperparameter_configs, total_timesteps=500000, total_episodes=100):
        """
        Initialize the HyperParamTester class.

        Args:
            hyperparameter_configs (list): List of hyperparameter configurations.
            total_timesteps (int): Total timesteps for training. Defaults to 500000.
            total_episodes (int): Total episodes for testing. Defaults to 100.
        """
        self.logger = setup_logger('HyperParamTester', os.getenv('hyper_param_tester_log_path', f'{self.logs_dir}/hyper_param_tester.log'))
        self.env = GameEnvironment()
        self.hyperparameter_configs = hyperparameter_configs
        self.total_episodes = total_episodes
        self.total_timesteps = total_timesteps
        self.metrics = {"config_id": [], "total_reward": [], "success_rate": [], "average_episode_length": []}

    def train_model(self, config, config_count):
        """
        Train the DQN model with the given hyperparameter configuration.

        Args:
            config (dict): Dictionary of hyperparameters for the model.
            config_count (int): Index of the configuration being trained.

        Returns:
            DQN: The trained DQN model.
        """
        # Initialize the RL model with the provided hyperparameters
        model = DQN(
            "MlpPolicy",
            self.env,
            learning_rate=config["learning_rate"],
            gamma=config["gamma"],
            exploration_initial_eps=config["exploration_initial_eps"],
            exploration_final_eps=config["exploration_final_eps"],
            exploration_fraction=config["exploration_fraction"],
            buffer_size=config["buffer_size"],
            batch_size=config["batch_size"],
            target_update_interval=config["target_update_interval"],
            verbose=1
        )

        self.logger.info(f"Training model {config_count + 1}...")
        model.learn(total_timesteps=self.total_timesteps)
        model_file_name = os.path.join(self.models_dir, f"dqn_treasure_hunter_config_{config_count + 1}.zip")
        
        # Save the trained model
        model.save(model_file_name)
        self.logger.info(f"Model saved to {model_file_name}")
        return model


    def test_model(self, model, config_count):
        """
        Test the trained model and record metrics.

        Args:
            model (DQN): The trained DQN model.
            config_count (int): Index of the configuration being tested.

        Returns:
            None
        """
        self.logger.info(f"Starting testing with configuration {config_count + 1}")
        total_reward = 0
        success_count = 0
        sum_episode_length = 0

        for episode in range(self.total_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                action, _states = model.predict(state)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_length += 1

            total_reward += episode_reward
            sum_episode_length += episode_length
            if reward > 0:  # Success if the agent reached the treasure
                success_count += 1

        avg_reward = total_reward / self.total_episodes
        avg_episode_length = sum_episode_length / self.total_episodes
        success_rate = success_count / self.total_episodes

        self.logger.info(f"Configuration {config_count + 1}: Average Reward: {avg_reward}, Success Rate: {success_rate}, Average Episode Length: {avg_episode_length}")

        # Save metrics
        self.metrics["config_id"].append(config_count + 1)
        self.metrics["total_reward"].append(avg_reward)
        self.metrics["success_rate"].append(success_rate)
        self.metrics["average_episode_length"].append(avg_episode_length)


    def save_metrices(self):
        """
        Save the metrics dictionary to a JSON file.
        """
        try:
            with open(self.metrics_path, "w") as f:
                json.dump(self.metrics, f)
            self.logger.info(f"Metrics saved to {self.metrics_path}")
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")
            raise

    def run(self):
        """
        Execute the hyperparameter tuning workflow.

        For each hyperparameter configuration:
        - Train the model.
        - Test the model.
        - Record the results.
        """
        for config_count, config in enumerate(self.hyperparameter_configs):
            self.logger.info(f"Starting training with configuration {config_count + 1}: {config}")
            trained_model = self.train_model(config, config_count)
            self.test_model(trained_model, config_count)
        self.save_metrices()


if __name__ == "__main__":
    """
    Entry point for running the hyperparameter tester.

    Reads hyperparameter configurations from a JSON file and runs training and testing.
    """
    try:
        with open("hyperparameters.json", "r") as f:
            hyperparameter_configs = json.load(f)
        test = HyperParamTester(
            hyperparameter_configs,
            total_timesteps=int(os.getenv('hyper_param_tester_total_timesteps', 500000)),
            total_episodes=int(os.getenv('hyper_param_tester_total_episodes', 100))
        )
        test.run()
    except FileNotFoundError as e:
        print(f"Error: Hyperparameters file not found: {e}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in hyperparameters file: {e}")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
