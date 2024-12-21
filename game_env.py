import gym
from gym import spaces
import numpy as np

class GameEnvironment(gym.Env):
    def __init__(self):
        super(GameEnvironment, self).__init__()
        
        # Grid dimensions
        self.grid_size = 10
        
        # Define action space (up, down, left, right)
        self.action_space = spaces.Discrete(4)
        
        # Observation space (agent's position + treasure + monsters)
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(3,), dtype=np.int32
        )
        
        # Initialize game state
        self.reset()

    def reset(self):
        """Resets the game environment."""
        self.agent_pos = np.random.randint(0, self.grid_size, size=(2,))
        self.treasure_pos = np.random.randint(0, self.grid_size, size=(2,))
        self.monster_positions = self._generate_monsters()
        
        print(f"Agent starting position: {self.agent_pos}")
        print(f"Treasure position: {self.treasure_pos}")
        print(f"Monster positions: {self.monster_positions}")
        
        return self._get_state()

    def step(self, action):
        """Executes a move based on the action."""
        self._move_agent(action)
        self._move_monsters()  # New: Move monsters after the agent moves
        reward, done = self._check_game_state()
        return self._get_state(), reward, done, {}

    def _move_agent(self, action):
        """Moves the agent based on the chosen action."""
        old_pos = self.agent_pos.copy()
        if action == 0:  # Up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # Down
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        elif action == 2:  # Left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3:  # Right
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        
        print(f"Agent moved from {old_pos} to {self.agent_pos} with action {action}")

    def _move_monsters(self):
        """Moves each monster in a random direction."""
        for i, monster_pos in enumerate(self.monster_positions):
            direction = np.random.choice([0, 1, 2, 3])  # Randomly choose a direction
            old_monster_pos = monster_pos.copy()

            # Move monsters in the same grid as the agent
            if direction == 0:  # Up
                monster_pos[1] = max(0, monster_pos[1] - 1)
            elif direction == 1:  # Down
                monster_pos[1] = min(self.grid_size - 1, monster_pos[1] + 1)
            elif direction == 2:  # Left
                monster_pos[0] = max(0, monster_pos[0] - 1)
            elif direction == 3:  # Right
                monster_pos[0] = min(self.grid_size - 1, monster_pos[0] + 1)

            print(f"Monster {i+1} moved from {old_monster_pos} to {monster_pos}")

    def _check_game_state(self):
        """Checks if the game is won or lost."""
        if np.array_equal(self.agent_pos, self.treasure_pos):
            print("Agent reached the treasure!")
            return 10, True  # Win
        elif self._is_adjacent_to_monster():
            print("Agent is adjacent to a monster!")
            return -10, True  # Lose
        return -1, False  # Step penalty

    def _is_adjacent_to_monster(self):
        """Checks if the agent is adjacent to any monster."""
        for monster_pos in self.monster_positions:
            dist = np.linalg.norm(self.agent_pos - monster_pos)
            print(f"Distance to monster at {monster_pos}: {dist}")
            if dist <= 1:  # Adjacent or on the same position
                return True
        return False

    def _generate_monsters(self):
        """Generates monster positions."""
        num_monsters = np.random.randint(1, 5)  # Random number of monsters
        print("number of monsters: ",num_monsters)
        return [np.random.randint(0, self.grid_size, size=(2,)) for _ in range(num_monsters)]

    def _get_state(self):
        """Returns the current state."""
        return np.concatenate((self.agent_pos, self.treasure_pos))