from game_env import GameEnvironment  # Import your environment
import numpy as np
# Initialize the environment
env = GameEnvironment()

# Reset the environment and print the initial state
# state = env.reset()
# print("Initial State (Agent, Treasure):", state)

# Test random moves
actions = [0, 1, 2, 3]  # Up, Down, Left, Right
for i in range(20):  # Simulate 20 steps
    action = env.action_space.sample()  # Random action
    print(f"Step {i+1}:")
    
    # Print agent's initial position and monsters' positions before action
    print(f"Agent's current position: {env.agent_pos}")
    print(f"Monster positions: {env.monster_positions}")
    
    # Perform the action and get the new state
    next_state, reward, done, info = env.step(action)
    
    # Print the next state, reward, and done status after action
    print(f"Action: {action}, State: {next_state}, Reward: {reward}, Done: {done}")
    
    # Print the new positions and distance to monsters
    print(f"Agent's new position: {env.agent_pos}")
    for monster_pos in env.monster_positions:
        dist = np.linalg.norm(env.agent_pos - monster_pos)
        print(f"Distance to monster at {monster_pos}: {dist}")
    
    # Check if the game is over
    if done:
        print("Game Over!")
        break

# Test if the game ends correctly when adjacent to a monster
