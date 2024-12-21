from logger_setup import setup_logger
from game_env import GameEnvironment
import numpy as np
from stable_baselines3 import DQN

# Set up the logger for testing
logger = setup_logger('Test', 'test.log')

# Initialize the environment
env = GameEnvironment()

# Initialize the RL model
# model = DQN('MlpPolicy', env, verbose=1)
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.0005,  # Slightly increased learning rate
    gamma=0.95,  # Slightly reduced discount factor to prioritize short-term rewards
    exploration_initial_eps=1.0,  # Start with full exploration
    exploration_final_eps=0.1,  # End with more exploration than before
    exploration_fraction=0.4,  # Slow down exploration decay
    buffer_size=200000,  # Increase replay buffer size
    batch_size=64,  # Increase batch size for more stable updates
    train_freq=1,  # Update after every step
    target_update_interval=5000,  # Update target network more frequently
    verbose=1,  # Print training information
)

# Train the model
logger.info("Training the model...")
model.learn(total_timesteps=100000)  # Increased total timesteps
model.save("dqn_treasure_hunter_tuned")
logger.info("Model training complete and saved.")

# Load the trained RL model
model = DQN.load("dqn_treasure_hunter_tuned")

# Reset the environment and print the initial state
state = env.reset()
total_reward = 0

'''
# Test random moves
actions = [0, 1, 2, 3]  # Up, Down, Left, Right
for i in range(20):  # Simulate 20 steps
    action = env.action_space.sample()  # Random action
    print(f"Step {i+1}:")
'''
# Test the trained RL model
done = False
step = 0
while not done:
    step += 1
    action, _states = model.predict(state)  # Get the action from the trained model
    
    logger.info(f"Step {step}:")
    logger.info(f"Agent's current position: {env.agent_pos}")
    logger.info(f"Monster positions: {env.monster_positions}")

    # Perform the action and get the new state
    next_state, reward, done, info = env.step(action)

    # Accumulate total reward
    total_reward += reward
    logger.info(f"Action: {action}, Reward: {reward}, Done: {done}")

    # Print the new positions and distance to monsters
    logger.info(f"Agent's new position: {env.agent_pos}")
    for monster_pos in env.monster_positions:
        dist = np.linalg.norm(env.agent_pos - monster_pos)
        logger.info(f"Distance to monster at {monster_pos}: {dist}")
    
    # Check if the game is over
    if done:
        if reward > 0:
            logger.info("Agent reached the treasure!")
        else:
            logger.info("Agent lost to a monster!")
        break

# Print the total reward accumulated in the episode
logger.info(f"Total reward for the episode: {total_reward}")
