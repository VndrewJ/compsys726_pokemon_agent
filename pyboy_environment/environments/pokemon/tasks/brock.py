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
        emulation_speed: int = 0,
        headless: bool = False,
    ) -> None:

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

        super().__init__(
            act_freq=act_freq,
            task="brock",
            init_name="has_pokedex.state",
            emulation_speed=emulation_speed,
            valid_actions=valid_actions,
            release_button=release_button,
            headless=headless,
        )

    def _get_state(self) -> np.ndarray:
        # State includes map_id to track location changes
        game_stats = self._generate_game_stats()
        location = self._get_location()
        return [game_stats["badges"], location["map_id"]]


    def _calculate_reward(self, new_state: dict[str, any]) -> float:
        # Reward +10 for leaving Oak's lab (map_id=40) and entering Pallet Town (map_id=0)
        # Penalize -10 for going back into Oak's lab from Pallet Town
        previous_map_id = self.prior_game_stats.get("location", {}).get("map_id", 40)  # Default to 40 (Oak's lab)
        current_map_id = new_state["location"]["map_id"]
        
        reward = 0
        if previous_map_id == 40 and current_map_id == 0:
            reward += 10  # Reward for leaving Oak's lab and entering Pallet Town
        elif previous_map_id == 0 and current_map_id == 40:
            reward -= 10  # Penalty for going back into Oak's lab

        return reward
        #return new_state["badges"] - self.prior_game_stats["badges"]

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here

        # Maybe if we run out of pokeballs...? or a max step count
        return self.steps >= 1000
