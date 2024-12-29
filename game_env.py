from logger_setup import setup_logger
from gym import spaces
import numpy as np
import gym
import os

# Constants with default values
WIN_REWARD = int(os.getenv("WIN_REWARD", 50))
LOSE_PENALTY = int(os.getenv("LOSE_PENALTY", -20))
STEP_PENALTY = int(os.getenv("STEP_PENALTY", -1))
DANGER_ZONE_DISTANCE = int(os.getenv("DANGER_ZONE_DISTANCE", 3))
STAGNATION_PENALTY = int(os.getenv("STAGNATION_PENALTY", -2))

print(f"WIN_REWARD: {WIN_REWARD}, LOSE_PENALTY: {LOSE_PENALTY}, STEP_PENALTY: {STEP_PENALTY}, "
      f"DANGER_ZONE_DISTANCE: {DANGER_ZONE_DISTANCE}, STAGNATION_PENALTY: {STAGNATION_PENALTY}")


# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

class GameEnvironment(gym.Env):
    def __init__(self, grid_size = 10, min_agent_treasure_distance = 5, min_agent_monster_distance = 3):
        """
        Initialize the game environment.
        Args:
            grid_size (int): Size of the grid.
            min_agent_treasure_distance (int): Minimum Manhattan distance between agent and treasure.
            min_agent_monster_distance (int): Minimum Manhattan distance between agent and monsters.
        """
        super(GameEnvironment, self).__init__()
        self.logger = setup_logger('game_env', os.getenv('game_env_log_path', 'logs/game_env.log'))
        self.logger.info("Initializing the game environment.")
        self.grid_size = grid_size
        self.min_agent_treasure_distance = min_agent_treasure_distance
        self.min_agent_monster_distance = min_agent_monster_distance

        # Define action space (up, down, left, right)
        self.action_space = spaces.Discrete(4)

        # Observation space (agent's position + treasure + 3 monsters)
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(10,), dtype=np.int32
        )

    def reset(self):
        """
        Reset the game environment to its initial state.

        - Randomly places the agent, treasure, and monsters on the grid.
        - Ensures the agent is at a sufficient distance from the treasure and monsters.

        Returns:
            np.ndarray: The initial state of the environment.
        """
        while True:
            self.agent_pos = self._generate_unique_position([])
            self.treasure_pos = self._generate_unique_position([self.agent_pos])

            # Check if the agent and treasure are far enough (using Manhattan distance)
            distance = self._manhattan_distance(self.agent_pos, self.treasure_pos)
            if distance >= self.min_agent_treasure_distance:
                break
        self.monster_positions = self._generate_monsters()
        self.previous_distance_to_treasure = self._manhattan_distance(self.agent_pos, self.treasure_pos)

        self.logger.info(f"Agent starting position: {self.agent_pos}")
        self.logger.info(f"Treasure position: {self.treasure_pos}")
        self.logger.info(f"Monster positions: {self.monster_positions}")

        return self._get_state()

    def step(self, action):
        """
        Perform an action in the environment and update the state.

        Args:
            action (int): The action to perform (0=up, 1=down, 2=left, 3=right).

        Returns:
            tuple:
                - np.ndarray: The updated state of the environment.
                - float: The reward for the action.
                - bool: Whether the episode has ended (True/False).
                - dict: Additional information (empty in this case).
        """
        old_pos = self.agent_pos.copy()  # Store the agent's position before moving
        self._move_agent(action)
        self._move_monsters()
        reward, done = self._evaluate_game_state(old_pos)
        return self._get_state(), reward, done, {}

    def _move_agent(self, action):
        """Move the agent based on the action."""
        old_pos = self.agent_pos.copy()
        if action == 0:  # Up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # Down
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        elif action == 2:  # Left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3:  # Right
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        self.logger.debug(f"Agent moved from {old_pos} to {self.agent_pos}.")

    def _move_monsters(self):
        """Move all monsters in random directions."""
        for i, monster_pos in enumerate(self.monster_positions):
            direction = np.random.choice([0, 1, 2, 3])
            if direction == 0:
                monster_pos[1] = max(0, monster_pos[1] - 1)
            elif direction == 1:
                monster_pos[1] = min(self.grid_size - 1, monster_pos[1] + 1)
            elif direction == 2:
                monster_pos[0] = max(0, monster_pos[0] - 1)
            elif direction == 3:
                monster_pos[0] = min(self.grid_size - 1, monster_pos[0] + 1)

    def _evaluate_game_state(self, old_pos):
        """
        Evaluate the current state of the game to check if the episode ends.

        Args:
            old_pos (np.ndarray): The agent's position before taking the current action.

        Returns:
            tuple:
                - float: The reward for the current state.
                - bool: Whether the episode has ended (True/False).
        """
        if np.array_equal(self.agent_pos, self.treasure_pos):
            self.logger.info("Agent reached the treasure!")
            return WIN_REWARD, True
        if self._is_adjacent_to_monster():
            self.logger.info("Agent encountered a monster!")
            return LOSE_PENALTY, True
        reward = self._compute_reward(old_pos)
        return reward, False

    def _compute_reward(self, old_pos):
        """
        Calculate the reward for the agent's current step.

        - Rewards the agent for moving closer to the treasure.
        - Penalizes the agent for moving farther or staying stagnant.
        - Additional penalties are applied for proximity to monsters.

        Args:
            old_pos (np.ndarray): The agent's position before the current move.

        Returns:
            float: The calculated reward for the step.
        """
        reward = STEP_PENALTY
        distance_to_treasure = self._manhattan_distance(self.agent_pos, self.treasure_pos)

        # Reward or penalize distance to treasure
        if distance_to_treasure < self.previous_distance_to_treasure:
            reward += self.previous_distance_to_treasure - distance_to_treasure
        elif distance_to_treasure > self.previous_distance_to_treasure:
            reward -= distance_to_treasure - self.previous_distance_to_treasure

        # Penalize proximity to monsters
        for monster_pos in self.monster_positions:
            distance_to_monster = self._manhattan_distance(self.agent_pos, monster_pos)
            if distance_to_monster < DANGER_ZONE_DISTANCE:
                reward -= (DANGER_ZONE_DISTANCE - distance_to_monster) * 2

        # Penalize stagnant behavior
        if np.array_equal(self.agent_pos, old_pos):
            reward += STAGNATION_PENALTY

        self.previous_distance_to_treasure = distance_to_treasure
        return reward

    def _is_adjacent_to_monster(self):
        """Check if the agent is adjacent to any monster."""
        for monster_pos in self.monster_positions:
            dist = self._manhattan_distance(self.agent_pos, monster_pos)
            self.logger.debug(f"Distance to monster at {monster_pos}: {dist}")
            if dist <= 1:  # Adjacent or on the same position
                return True
        return False
    def _generate_monsters(self):
        """
        Generate unique positions for monsters on the grid.

        - Ensures monsters are sufficiently far from the agent and each other.
        - Places 3 monsters by default.

        Returns:
            list[np.ndarray]: List of positions for all monsters.
        """
        positions = []
        for _ in range(3):  # Three monsters
            while True:
                pos = self._generate_unique_position(positions + [self.agent_pos, self.treasure_pos])
                if self._manhattan_distance(pos, self.agent_pos) >= self.min_agent_monster_distance:
                    positions.append(pos)
                    break
        return positions

    def _generate_unique_position(self, occupied_positions: list[np.ndarray]) -> np.ndarray:
        """
        Generate a unique grid position that is not occupied.

        Args:
            occupied_positions (list[np.ndarray]): List of positions already occupied on the grid.

        Returns:
            np.ndarray: A unique position on the grid.
        """
        while True:
            position = np.random.randint(0, self.grid_size, size=(2,))
            if not any(np.array_equal(position, occupied) for occupied in occupied_positions):
                return position

    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_state(self):
        """
        Retrieve the current state of the environment.

        - Combines the agent's position, treasure position, and monster positions into a single array.

        Returns:
            np.ndarray: The concatenated state representation of the environment.
        """
        return np.concatenate((self.agent_pos, self.treasure_pos, *self.monster_positions))
