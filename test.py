from game_env import GameEnvironment  # Import your environment

# Initialize the environment
env = GameEnvironment()

# Reset the environment and print the initial state
state = env.reset()
print("Initial State (Agent, Treasure):", state)

# Test some random moves
actions = [0, 1, 2, 3]  # Up, Down, Left, Right
for i in range(10):  # Simulate 10 steps
    action = env.action_space.sample()  # Random action
    next_state, reward, done, info = env.step(action)
    print(f"Step {i+1}: Action: {action}, State: {next_state}, Reward: {reward}, Done: {done}")
    
    if done:
        print("Game Over!")
        break
