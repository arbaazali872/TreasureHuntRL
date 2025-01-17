from logger_setup import setup_logger
from game_env import GameEnvironment
from stable_baselines3 import DQN
from dotenv import load_dotenv
import numpy as np
import os

class ModelHandler:
    """
    A handler class for managing the training and testing of a DQN model 
    in the custom GameEnvironment.

    Attributes:
        env (GameEnvironment): The game environment for the agent.
        model_path (str): Path to save or load the DQN model.
        logger (Logger): Logger for logging training and testing activities.
    """
    def __init__(self, config_path="config.env"):
        """
        Initialize the model handler with the environment and logger.

        Args:
            config_path (str): Path to the environment variable configuration file.
        """
        load_dotenv(config_path)
        self.env = GameEnvironment()
        self.model_path = os.getenv('model_path', 'models/dqn_model.zip')  # Default path
        self.logger = setup_logger('Test', os.getenv('test_log_path', 'logs/test.log'))

    def setup_model(self):
        """
        Set up the DQN model using hyperparameters loaded from environment variables.

        Returns:
            DQN: The initialized DQN model with specified hyperparameters.
        """
        return DQN(
            "MlpPolicy",
            self.env,
            learning_rate=float(os.getenv('learning_rate', 0.0005)),
            gamma=float(os.getenv('gamma', 0.95)),
            exploration_initial_eps=float(os.getenv('exploration_initial_eps', 1.0)),
            exploration_final_eps=float(os.getenv('exploration_final_eps', 0.1)),
            exploration_fraction=float(os.getenv('exploration_fraction', 0.4)),
            buffer_size=int(os.getenv('buffer_size', 200000)),
            batch_size=int(os.getenv('batch_size', 64)),
            train_freq=int(os.getenv('train_freq', 1)),
            target_update_interval=int(os.getenv('target_update_interval', 5000)),
            verbose=int(os.getenv('verbose', 1)),
        )

    def train_model(self, total_timesteps=None):
        """
        Train the DQN model and save it to the specified path.

        Args:
            total_timesteps (int, optional): The number of timesteps to train the model.
                                             If not provided, it defaults to the value 
                                             specified in the environment variables.
        """
        if not total_timesteps:
            total_timesteps = int(os.getenv('total_timesteps', 1000))
        model = self.setup_model()
        self.logger.info("Starting model training...")
        model.learn(total_timesteps=total_timesteps)
        model.save(self.model_path)
        self.logger.info(f"Model training complete and saved at {self.model_path}")

    def test_model(self):
        """
        Test the trained DQN model by running it in the environment.

        Logs the agent's actions, rewards, and positions at each step, and reports 
        the total reward accumulated during the episode.

        If the model file does not exist, logs an error and skips testing.
        """
        if not os.path.exists(self.model_path):
            self.logger.error(f"Model file not found at {self.model_path}. Please train the model first.")
            return

        try:
            model = DQN.load(self.model_path)
        except Exception as e:
            self.logger.error(f"Failed to load the model: {e}")
            return
        state = self.env.reset()
        total_reward = 0
        done = False
        step = 0
        while not done:
            step += 1
            action, _states = model.predict(state)  # Get the action from the trained model
            
            self.logger.info(f"Step {step}:")
            self.logger.info(f"Agent's current position: {self.env.agent_pos}")
            self.logger.info(f"Monster positions: {self.env.monster_positions}")

            # Perform the action and get the new state
            next_state, reward, done, info = self.env.step(action)

            # Accumulate total reward
            total_reward += reward
            self.logger.info(f"Action: {action}, Reward: {reward}, Done: {done}")

            # Print the new positions and distance to monsters
            self.logger.info(f"Agent's new position: {self.env.agent_pos}")
            for monster_pos in self.env.monster_positions:
                dist = np.linalg.norm(self.env.agent_pos - monster_pos)
                self.logger.info(f"Distance to monster at {monster_pos}: {dist}")
            
            # Check if the game is over
            if done:
                if reward > 0:
                    self.logger.info("Agent reached the treasure!")
                else:
                    self.logger.info("Agent lost to a monster!")
                break

        # Print the total reward accumulated in the episode
        self.logger.info(f"Total reward for the episode: {total_reward}")

if __name__ == "__main__":
    """
    Entry point for the script.

    Mode can be specified using the 'mode' environment variable:
        - 'train': Only trains the model.
        - 'test': Only tests the model.
        - Default: Both trains and tests the model.
    """
    handler = ModelHandler()
    mode = os.getenv('mode')

    if mode == 'train':
        handler.train_model()
    elif mode == 'test':
        handler.test_model()
    else:
        print("No/invalid mode selected, script will both train and test the model now")
        handler.train_model()
        handler.test_model()