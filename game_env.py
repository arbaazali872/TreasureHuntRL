import gym
from gym import spaces
import numpy as np
from logger_setup import setup_logger

logger = setup_logger('GameEnvironment', 'game_env_3rd_tuned_reward1.log')

class GameEnvironment(gym.Env):
    def __init__(self, grid_size=10, min_agent_treasure_distance=5, min_agent_monster_distance=3):
        super(GameEnvironment, self).__init__()
        logger.info("Initializing the game environment.")

        # Grid dimensions
        self.grid_size = grid_size

        # Distance constraints
        self.min_agent_treasure_distance = min_agent_treasure_distance
        self.min_agent_monster_distance = min_agent_monster_distance

        # Define action space (up, down, left, right)
        self.action_space = spaces.Discrete(4)

        # Observation space (agent's position + treasure + 3 monsters)
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(10,), dtype=np.int32
        )

        # Initialize game state
        # self.reset()

    def reset(self):
        """Resets the game environment."""
        while True:
            # Generate agent and treasure positions
            self.agent_pos = self._generate_unique_position([])
            self.treasure_pos = self._generate_unique_position([self.agent_pos])

            # Check if the agent and treasure are far enough (using Manhattan distance)
            distance = self._manhattan_distance(self.agent_pos, self.treasure_pos)
            if distance >= self.min_agent_treasure_distance:
                break

        # Generate distinct positions for monsters
        self.monster_positions = self._generate_monsters()

        # Initialize proximity tracking
        self.previous_distance_to_treasure = self._manhattan_distance(self.agent_pos, self.treasure_pos)

        logger.info(f"Agent starting position: {self.agent_pos}")
        logger.info(f"Treasure position: {self.treasure_pos}")
        logger.info(f"Monster positions: {self.monster_positions}")

        return self._get_state()

    def step(self, action):
        """Executes a move based on the action."""
        old_pos = self.agent_pos.copy()  # Store the agent's position before moving
        self._move_agent(action)
        self._move_monsters()
        
        # Penalize stagnation (agent staying in the same position)
        reward, done = self._check_game_state()
        if np.array_equal(self.agent_pos, old_pos):  # Check if the agent's position hasn't changed
            reward -= 2  # Penalize staying in place
            logger.debug("Agent stayed in the same position, applying stagnation penalty.")
        
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

    def _check_game_state(self):
        """Checks if the game is won or lost and returns the appropriate reward."""
        old_pos = self.agent_pos.copy()
        if np.array_equal(self.agent_pos, self.treasure_pos):
            logger.info("Agent reached the treasure!")
            return 50, True  # Larger reward for winning
        elif self._is_adjacent_to_monster():
            logger.info("Agent is adjacent to a monster!")
            return -20, True  # Penalty for losing
        else:
            # Calculate proximity-based rewards and penalties
            distance_to_treasure = self._manhattan_distance(self.agent_pos, self.treasure_pos)
            reward = -1  # Step penalty

            # Reward or penalize based on distance to the treasure
            if distance_to_treasure < self.previous_distance_to_treasure:
                reward += self.previous_distance_to_treasure - distance_to_treasure  # Reward for getting closer
            elif distance_to_treasure > self.previous_distance_to_treasure:
                reward -= distance_to_treasure - self.previous_distance_to_treasure  # Penalty for getting farther
            
            # Penalize proximity to monsters
            for monster_pos in self.monster_positions:
                distance_to_monster = self._manhattan_distance(self.agent_pos, monster_pos)
                if distance_to_monster < 3:  # Within danger zone
                    reward -= (3 - distance_to_monster) * 2  # Larger penalty for being closer

            # Penalize stagnant behavior
            if np.array_equal(self.agent_pos, old_pos):  # Check if the position didn't change
                reward -= 2  # Penalize staying in place

            # Update proximity tracking
            self.previous_distance_to_treasure = distance_to_treasure
            return reward, False


    def _is_adjacent_to_monster(self):
        """Checks if the agent is adjacent to any monster."""
        for monster_pos in self.monster_positions:
            dist = self._manhattan_distance(self.agent_pos, monster_pos)
            logger.debug(f"Distance to monster at {monster_pos}: {dist}")
            if dist <= 1:  # Adjacent or on the same position
                return True
        return False

    def _generate_monsters(self):
        """Generates exactly 3 monster positions ensuring no overlap and minimum Manhattan distance from the agent."""
        num_monsters = 3
        logger.info(f"Number of monsters: {num_monsters}")
        positions = []
        for _ in range(num_monsters):
            while True:
                position = self._generate_unique_position(positions + [self.agent_pos, self.treasure_pos])
                # Ensure the monster is at least `min_agent_monster_distance` Manhattan units away from the agent
                if self._manhattan_distance(position, self.agent_pos) >= self.min_agent_monster_distance:
                    positions.append(position)
                    break
                else:
                    logger.debug(f"Retrying position for monster to meet Manhattan distance constraint.")
        return positions

    def _generate_unique_position(self, occupied_positions):
        """Generates a unique position on the grid that does not overlap with any positions in the occupied_positions list."""
        while True:
            position = np.random.randint(0, self.grid_size, size=(2,))
            if not any(np.array_equal(position, occupied) for occupied in occupied_positions):
                return position

    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_state(self):
        """Returns the current state: agent position, treasure position, and 3 monster positions."""
        return np.concatenate((self.agent_pos, self.treasure_pos, *self.monster_positions))
