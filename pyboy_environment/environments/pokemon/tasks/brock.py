from functools import cached_property

import numpy as np
from pyboy.utils import WindowEvent

from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)
from pyboy_environment.environments.pokemon import pokemon_constants as pkc


class PokemonBrock(PokemonEnvironment):
    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 1,
        headless: bool = False,
    ) -> None:

        valid_actions: list[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            # WindowEvent.PRESS_BUTTON_START,
        ]

        release_button: list[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            # WindowEvent.RELEASE_BUTTON_START,
        ]

        super().__init__(
            act_freq=act_freq,
            task="brock",
            init_name="has_pokedex.state",
            emulation_speed=emulation_speed,
            valid_actions=valid_actions,
            release_button=release_button,
            headless=headless,
        )

        # Previous position buffer, initialize with starting position
        self.position_buffer = [self._get_location()] * 10

        # Map_id sequence
        self.map_sequence = [40, 0, 12, 1, 13, 50, 51, 47, 3]

        # Track the initial position and maximum distance
        self.initial_position = None
        self.max_distance = 0
        self.current_map_ticks = 0

        # Counter for consecutive ticks in the same position
        self.consecutive_ticks_in_same_position = 0

    def _get_state(self) -> np.ndarray:
        # State includes badges, map_id, player location (x, y), battle status, health, and level of the first Pokémon
        game_stats = self._generate_game_stats()

        state_array = [
            game_stats["location"]["x"],
            game_stats["location"]["y"],
            game_stats["location"]["map_id"],
            game_stats["in_battle"],
            # game_stats["hp"]["current"][0] if game_stats["party_size"] > 0 else 0,  # Check hp of pokemon
            # game_stats["levels"][0] if game_stats["party_size"] > 0 else 0,         # Check level of pokemon
        ]
        return state_array


    def _calculate_reward(self, new_state: dict[str, any]) -> float:
        reward = 0

        # Punish agent for inputting invalid button
        if self.current_button not in self.valid_actions and self.current_button not in self.release_button:
            reward -= 5

        # Get in_battle bool and pokemon health and level
        in_battle = new_state["in_battle"]
        # pokemon_health = new_state["hp"]["current"][0] if new_state["party_size"] > 0 else 0
        # pokemon_level = new_state["levels"][0] if new_state["party_size"] > 0 else 0

        # # Get previous Pokémon health and level
        # previous_health = self.prior_game_stats["hp"]["current"][0] if self.prior_game_stats["party_size"] > 0 else 0
        # previous_level = self.prior_game_stats["levels"][0] if self.prior_game_stats["party_size"] > 0 else 0

        # Reward player for battling trainer, punish battling wild pokemon
        # if self._read_m(0xD057) == 1:
        #     reward -= 0.5
        # elif self._read_m(0xD057) == 2:
        #     reward += 1

        # Reward agent for being in battle (its exploiting this and stalling)
        # if in_battle != 0:
        #     reward += 1

        # Motivate Agent to go to grass
        reward += self._grass_reward(new_state) * 2

        # Reward agent for finding and defeating pokemon
        reward += self._seen_reward(new_state) * 1000
        reward += self._xp_reward(new_state) * 10000

        # # Evaluate rewards for battling
        # if in_battle:
        #     # Penalize if the Pokémon's health is low (e.g., below 20% of max health)
        #     max_health = new_state["hp"]["max"][0] if new_state["party_size"] > 0 else 1
        #     if max_health > 0 and pokemon_health / max_health < 0.2:
        #         reward -= 20  # Penalty for battling with low health

        # # Reward the player if the Pokémon levels up
        # if pokemon_level > previous_level:
        #     reward += 100  # Significant reward for leveling up
        

        # Get current and previous map IDs
        previous_map_id = self.prior_game_stats.get("location", {}).get("map_id", 40)  # Default to Oak's lab (map_id=40)
        current_map_id = new_state["location"]["map_id"]

        # Handle map transitions and reset the initial position and max distance
        if current_map_id != previous_map_id:
            # Entering a new location, update the initial position and reset max distance
            self.initial_position = (new_state["location"]["x"], new_state["location"]["y"])
            self.max_distance = 0

        # Handle map transitions for rewards
        if previous_map_id in self.map_sequence and current_map_id in self.map_sequence:
            prev_index = self.map_sequence.index(previous_map_id)
            curr_index = self.map_sequence.index(current_map_id)

            if curr_index == prev_index + 1:
                reward += 100  # Reward for progressing to the next map in the sequence
                self.position_buffer.clear()
            elif curr_index == prev_index - 1:
                reward -= 250  # Penalty for going back to the previous map in the sequence
            else:
                reward -= 0.5 # Penalty for staying in the same map for too long
            
            # self.current_map_ticks = 0
        
        
        # Penalty for staying in the same map ID for too long, wait for 10 ticks before it starts punishing
        # if current_map_id == previous_map_id:
        #     self.current_map_ticks += 1
        # if self.current_map_ticks == 10:
        #     reward -= 0.5

        # Reward or penalty based on movement in position
        current_position = (new_state["location"]["x"], new_state["location"]["y"])
        if current_position not in self.position_buffer:
            reward += 20  # Reward for moving to a new position
        else:
            reward -= 2.5  # Penalty for revisiting a previous position

        # Penalize the agent for standing in the same position for 5 ticks
        if len(self.position_buffer) > 0 and current_position == self.position_buffer[-1]:
            self.consecutive_ticks_in_same_position += 1
            if self.consecutive_ticks_in_same_position >= 5:
                reward -= 5  # Apply penalty for staying idle too long
        else:
            self.consecutive_ticks_in_same_position = 0  # Reset the counter if agent moves


        # Update the position buffer (remove the oldest and add the new one)
        # if len(self.position_buffer) >= 200:
        #     self.position_buffer.pop(0)  # Remove the oldest position if buffer is full
        self.position_buffer.append(current_position)  # Add the new position

        # Reward for reaching a new maximum distance from the initial position
        if self.initial_position is not None:
            distance_from_initial = np.linalg.norm(np.array(current_position) - np.array(self.initial_position))
            if distance_from_initial > self.max_distance:
                reward += 50  # Reward for reaching a new record distance
                self.max_distance = distance_from_initial  # Update the maximum distance

        return reward

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here

        # Maybe if we run out of pokeballs...? or a max step count

        if self.steps >= 1000:
            self.position_buffer.clear()  # Clear the position buffer when truncation occurs
            self.initial_position = None
            self.max_distance = 0
            return True
        else:
            return False
