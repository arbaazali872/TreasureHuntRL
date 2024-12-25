from game_env import GameEnvironment  # Import your environment
import numpy as np
from stable_baselines3 import DQN  # Import the DQN model
from evaluate import evaluate_model
log_name = "stage_1c"

# Initialize the environment
env = GameEnvironment()

# Initialize the model (DQN here, but you can experiment with others like PPO or A2C)
model = DQN('MlpPolicy', env, verbose=1)

# Train the model (this will take some time)
model.learn(total_timesteps=10000)

# Save the trained model
model.save(f"dqn_treasure_hunter_{log_name}")

# Load the trained RL model
model = DQN.load(f"dqn_treasure_hunter_{log_name}")

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
    
    print(f"Step {step}:")
    print(f"Agent's current position: {env.agent_pos}")
    print(f"Monster positions: {env.monster_positions}")

    # Perform the action and get the new state
    next_state, reward, done, info = env.step(action)

    # Accumulate total reward
    total_reward += reward
    
    # Print the next state, reward, and done status after action
    print(f"Action: {action}, State: {next_state}, Reward: {reward}, Done: {done}")

    # Print the new positions and distance to monsters
    print(f"Agent's new position: {env.agent_pos}")
    for monster_pos in env.monster_positions:
        dist = np.linalg.norm(env.agent_pos - monster_pos)
        print(f"Distance to monster at {monster_pos}: {dist}")
    
    # Check if the game is over
    if done:
        if reward > 0:
            print("Agent reached the treasure!")
        else:
            print("Agent lost to a monster!")
        break

# Print the total reward accumulated in the episode
print(f"Total reward for the episode: {total_reward}")

# Evaluate the trained model
metrics = evaluate_model(model, env, num_episodes=100, log_file=f"evaluation_{log_name}.json")

# Print the evaluation results
print("Evaluation Results:")
print(metrics)