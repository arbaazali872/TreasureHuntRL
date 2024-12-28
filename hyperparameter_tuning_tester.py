from logger_setup import setup_logger
from game_env import GameEnvironment
from stable_baselines3 import DQN
import os
import json

# Determine the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
metrics_dir = os.path.join(current_dir, "metrices")
logs_dir = os.path.join(current_dir, "logs")
models_dir = os.path.join(current_dir, "models")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

class HyperParamTester():
    def __init__(self,hyperparameter_configs,total_timesteps=500000,total_episodes=100):
        """
        Initialize the HyperParamTester class.

        Args:
            hyperparameter_configs (list): List of hyperparameter configurations.
            total_timesteps (int): Total training timesteps.
            total_episodes (int): Total episodes for testing.
        """
        self.logger = setup_logger('HyperParamTester', os.getenv('hyper_param_tester_log_path', 'logs/hyper_param_tester.log'))
        self.env = GameEnvironment()
        self.hyperparameter_configs = hyperparameter_configs
        # logs_dir = os.path.join(current_dir, "logs")
        self.models_dir = os.path.join(current_dir, "models")
        self.metrics_path = os.path.join(metrics_dir, "training_metrics.json")
        self.total_episodes = total_episodes
        self.total_timesteps=total_timesteps
        self.metrics = {"config_id": [], "total_reward": [], "success_rate": [], "average_episode_length": []}


    def train_model(self,config,config_count):
        """Train the model with the specified hyperparameters."""
        
        # Initialize the RL model with current hyperparameters
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

        # Train the model
        self.logger.info(f"Training model {config_count + 1}...")
        model.learn(total_timesteps=self.total_timesteps)
        model_file_name = os.path.join(self.models_dir, f"dqn_treasure_hunter_config_{config_count + 1}.zip")
        # Save the trained model
        model.save(model_file_name)
        self.logger.info(f"Model saved to {model_file_name}")
        return model
    
    
    def test_model(self,model,config_count):
        """Test the trained model and log metrics."""
        # Test the trained model
        self.logger.info(f"Starting testing with configuration {config_count + 1}")
        state = self.env.reset()
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

        # Log metrics for this configuration
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
        try:
            with open(self.metrics_path, "w") as f:
                json.dump(self.metrics, f)
            self.logger.info(f"Metrics saved to {self.metrics_path}")
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")
            raise

    def run(self):
        """Run training and testing for all configurations."""
        for config_count, config in enumerate(self.hyperparameter_configs):
            self.logger.info(f"Starting training with configuration {config_count + 1}: {config}")
            trained_model = self.train_model(config,config_count)       
            self.test_model(trained_model,config_count) 
        self.save_metrices()
        

if __name__=="__main__":
    try:
        with open("hyperparameters.json", "r") as f:
            hyperparameter_configs = json.load(f)
        test = HyperParamTester(hyperparameter_configs, total_timesteps=os.getenv('hyper_param_tester_total_timesteps'), total_episodes=os.getenv('hyper_param_tester_total_episodes'))
        test.run()
    except FileNotFoundError as e:
        print(f"Error: Hyperparameters file not found: {e}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in hyperparameters file: {e}")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")



