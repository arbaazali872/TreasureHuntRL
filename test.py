from logger_setup import setup_logger
from game_env import GameEnvironment
from stable_baselines3 import DQN
from dotenv import load_dotenv
import numpy as np
import os

class test_game():

    def __init__(self, config_path="config.env"):
        # Load environment variables
        load_dotenv(config_path)
        self.env = GameEnvironment()
        self.model_path = os.getenv('model_path', 'dqn_model.zip')  # Default path
        self.logger = setup_logger('Test', os.getenv('test_log_path', 'test.log'))

    def setup_model_params(self):
        """Set up model parameters from environment variables."""
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
        """Train the model."""
        if not total_timesteps:
            total_timesteps = int(os.getenv('total_timesteps', 500000))
        model = self.setup_model_params()
        self.logger.info("Training the model...")
        model.learn(total_timesteps=total_timesteps)
        model.save(self.model_path)
        self.logger.info(f"Model training complete and saved at {self.model_path}")

    def test_model(self):
        """Test the trained model."""

        model = DQN.load(self.model_path)

        # Reset the environment and print the initial state
        state = self.env.reset()
        total_reward = 0
        # Test the trained RL model
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
    test = test_game()
    test.train_model()
    test.test_model()
        