import gym
from gym import spaces
import numpy as np
from logger_setup import setup_logger

logger = setup_logger('GameEnvironment', 'game_env.log')
class GameEnvironment(gym.Env):
    def __init__(self):
        super(GameEnvironment, self).__init__()
        logger.info("Initializing the game environment.")
        # Grid dimensions
        self.grid_size = 10
        
        # Define action space (up, down, left, right)
        self.action_space = spaces.Discrete(4)
        
        # Observation space (agent's position + treasure + 3 monsters)
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(10,), dtype=np.int32
        )
        
        # Initialize game state
        self.reset()

    def reset(self):
        """Resets the game environment."""
        self.agent_pos = np.random.randint(0, self.grid_size, size=(2,))
        self.treasure_pos = np.random.randint(0, self.grid_size, size=(2,))
        self.monster_positions = self._generate_monsters()
        
        # Initialize proximity tracking
        self.previous_distance_to_treasure = np.linalg.norm(self.agent_pos - self.treasure_pos)

        logger.info(f"Agent starting position: {self.agent_pos}")
        logger.info(f"Treasure position: {self.treasure_pos}")
        logger.info(f"Monster positions: {self.monster_positions}")
        
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
        
        logger.debug(f"Agent moved from {old_pos} to {self.agent_pos} with action {action}")

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

            logger.debug(f"Monster {i+1} moved from {old_monster_pos} to {monster_pos}")

    # def _check_game_state(self):
    def _check_game_state(self):
        """Checks if the game is won or lost and returns the appropriate reward."""
        if np.array_equal(self.agent_pos, self.treasure_pos):
            logger.info("Agent reached the treasure!")
            return 10, True  # Win
        elif self._is_adjacent_to_monster():
            logger.info("Agent is adjacent to a monster!")
            return -20, True  # Increased penalty for losing
        else:
            # Introduce proximity-based rewards and penalties
            distance_to_treasure = np.linalg.norm(self.agent_pos - self.treasure_pos)
            reward = -1  # Step penalty
            # Reward for moving closer to the treasure
            if distance_to_treasure < self.previous_distance_to_treasure:
                reward += 1
            self.previous_distance_to_treasure = distance_to_treasure
            return reward, False

    def _is_adjacent_to_monster(self):
        """Checks if the agent is adjacent to any monster."""
        for monster_pos in self.monster_positions:
            dist = np.linalg.norm(self.agent_pos - monster_pos)
            logger.debug(f"Distance to monster at {monster_pos}: {dist}")
            if dist <= 1:  # Adjacent or on the same position
                return True
        return False

    def _generate_monsters(self):
        """Generates exactly 3 monster positions."""
        num_monsters = 3  # Restrict number of monsters to 3
        logger.info(f"Number of monsters: {num_monsters}")
        return [np.random.randint(0, self.grid_size, size=(2,)) for _ in range(num_monsters)]

    def _get_state(self):
        """Returns the current state: agent position, treasure position, and 3 monster positions."""
        return np.concatenate((self.agent_pos, self.treasure_pos, *self.monster_positions))
