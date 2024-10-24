# Danny
from functools import cached_property
import numpy as np
from pyboy.utils import WindowEvent
from pyboy_environment.environments.pokemon.pokemon_environment import PokemonEnvironment
from pyboy_environment.environments.pokemon import pokemon_constants as pkc

class PokemonBrock(PokemonEnvironment):
    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 0,
        headless: bool = False,
    ) -> None:

        # Initialize the initial and last positions to None
        self.initial_position = None
        self.last_position = None

        # Failure counter for when an action doesn’t move the agent
        self.failure_count = 0
        self.max_failures = 5  # Threshold for discouraging stuck actions

        self.max_distance = 0

        # Valid actions for the environment
        valid_actions: list[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        release_button: list[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START,
        ]

        # Initialize the parent environment
        super().__init__(
            act_freq=act_freq,
            task="brock",
            init_name="has_pokedex.state",
            emulation_speed=emulation_speed,
            valid_actions=valid_actions,
            release_button=release_button,
            headless=headless,
        )
    
    def reset(self) -> np.ndarray:

        # Call the parent class reset (if needed)
        super().reset()

        # Reset internal tracking variables
        self.initial_position = None
        self.last_position = None
        self.failure_count = 0
        self.max_distance = 0

        # Return the initial state for the agent
        return self._get_state()


    def _get_state(self) -> np.ndarray:
        # Get the current game stats
        game_stats = self._generate_game_stats()

        # Save the initial position if not set yet
        if self.initial_position is None:
            self.initial_position = (game_stats["location"]["x"], game_stats["location"]["y"])

        return [game_stats["badges"]]

    def _calculate_reward(self, new_state: dict) -> float:
        """
        Calculates reward based on two tasks:
        1. Traveling far (reward only when a new record distance is reached).
        2. Leveling up Pokémon in the party.
        """
        battle_reward = 0
        exploration_reward = 0
        # --------------------- Traveling Reward Calculation ---------------------
        # Get the agent's current position
        current_position = (new_state["location"]["x"], new_state["location"]["y"])

        # Set the last position if this is the first step
        if self.last_position is None:
            self.last_position = current_position

        # Calculate the distance from the initial position
        distance_from_initial = np.linalg.norm(np.array(current_position) - np.array(self.initial_position))

        # If there is no movement from the last position
        if np.linalg.norm(np.array(current_position) - np.array(self.last_position)) == 0:
            self.failure_count += 1

            # Apply a larger penalty if the agent has been stuck for too long
            if self.failure_count >= self.max_failures:
                exploration_reward -= 2  # Bigger penalty for repeated failure
        else:
            # Check if the agent has reached a new record distance
            if distance_from_initial > self.max_distance:
                # Reward for reaching a new record distance
                exploration_reward += 10  # Assign a reward for reaching a new record
                self.max_distance = distance_from_initial  # Update the record distance
                self.failure_count = 0  # Reset failure count since the agent moved
            else:
                exploration_reward -= 1  # penalty for no progress

        # Update the last position to the current position
        self.last_position = current_position

        # # --------------------- Battling Reward Calculation ---------------------
        # Get the current party levels
        current_levels = self._read_party_level()

        # Initialize the previous party levels if not already set
        # Start with a total level of 5 distributed among the team
        if "party_levels" not in self.prior_game_stats:
            self.prior_game_stats["party_levels"] = [6,0,0,0,0,0] # e.g., 5 Pokémon with level 1, one empty slot
        
        previous_levels = self.prior_game_stats["party_levels"]

        # Calculate the total level of the current and previous Pokémon party
        total_current_level = sum(current_levels)
        total_previous_level = sum(previous_levels)

        # Only reward if the total level of the team has increased
        if total_current_level > total_previous_level:
            # Reward based on the overall team level increase (10 points per level increase)
            battle_reward += (total_current_level - total_previous_level) * 10 

        # Update the prior_game_stats to store the current party levels for the next step
        self.prior_game_stats["party_levels"] = current_levels

        #To encourage seeing pokemon.
        battle_reward += self._seen_reward(new_state) * 100
        #To encourage defeating pokemon.
        battle_reward += self._xp_reward(new_state) * 10

        # print (exploration_reward)
        # print (battle_reward)
        reward = exploration_reward + battle_reward

        return reward

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if the agent beats the first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here, e.g., running out of Pokéballs or max step count
        return self.steps >= 1000
