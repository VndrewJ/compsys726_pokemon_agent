# Jormickey
from functools import cached_property
import numpy as np
from pyboy.utils import WindowEvent
from pyboy_environment.environments.pokemon.pokemon_environment import PokemonEnvironment
from pyboy_environment.environments.pokemon import pokemon_constants as pkc

class PokemonBrock(PokemonEnvironment):
    OPTIMAL_PATH = [
        40,  # OAKS_LAB
        0,   # PALLET_TOWN
        12,  # ROUTE_1
        1,   # VIRIDIAN_CITY
        33,  # ROUTE_22 (optional for training)
        13,  # ROUTE_2
        51,  # VIRIDIAN_FOREST
        2,   # PEWTER_CITY
        54,  # PEWTER_GYM
    ]

    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 0,
        headless: bool = False,
    ) -> None:
        # Initialize tracking variables
        self.max_progression = 0
        self.current_state = None
        self.prior_state = None
        
        valid_actions: list[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            #WindowEvent.PRESS_BUTTON_START,
        ]
        release_button: list[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            #WindowEvent.RELEASE_BUTTON_START,
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

    def _get_progression_index(self, map_id: int) -> int:
        """Returns how far along the optimal path the agent is"""
        try:
            return self.OPTIMAL_PATH.index(map_id)
        except ValueError:
            return -1

    def _get_state(self) -> np.ndarray:
        game_stats = self._generate_game_stats()
        
        # Get current location info
        location = game_stats["location"]
        map_id = location["map_id"]
        x_pos = location["x"]
        y_pos = location["y"]
        
        # Get party info
        party_size = game_stats["party_size"]
        first_pokemon_level = game_stats["levels"][0] if party_size > 0 else 0
        first_pokemon_hp = game_stats["hp"]["current"][0] if party_size > 0 else 0
        first_pokemon_max_hp = game_stats["hp"]["max"][0] if party_size > 0 else 0
        
        # Calculate progression
        current_progression = self._get_progression_index(map_id)
        
        # Update state tracking
        self.prior_state = self.current_state
        self.current_state = {
            "map_id": map_id,
            "x": x_pos,
            "y": y_pos,
            "progression": current_progression
        }
        
        # Print current state info
        print(f"\nCurrent Position - Map: {map_id}, X: {x_pos}, Y: {y_pos}")
        if self.prior_state:
            y_change = y_pos - self.prior_state["y"]
            print(f"Y Movement: {y_change}")
        
        # Create the state array
        state = np.array([
            current_progression,   # Where we are in the optimal path (-1 if off path)
            self.max_progression,  # Furthest we've gotten on optimal path
            map_id,               # Current map ID
            x_pos,                # X position
            y_pos,                # Y position
            first_pokemon_level,  # Level of first Pokemon
            first_pokemon_hp,     # Current HP of first Pokemon
            first_pokemon_max_hp, # Max HP of first Pokemon
            1 if self._is_grass_tile() else 0,  # Whether in grass
            game_stats["badges"], # Number of badges
            party_size,          # Number of Pokemon in party
            sum(game_stats["levels"]),  # Total levels of all Pokemon
        ])
        
        return state

    def _calculate_reward(self, new_state: dict) -> float:
        reward = 0.0
        
        if self.prior_state is None:
            return 0.0  # No reward for first state
            
        print("\nCalculating Reward:")
        print(f"Current Map: {self.current_state['map_id']} (Progression: {self.current_state['progression']})")
        print(f"Previous Map: {self.prior_state['map_id']} (Progression: {self.prior_state['progression']})")
        
        # Get Y movement
        y_change = self.current_state["y"] - self.prior_state["y"]
        print(f"Y Change: {y_change}")
        
        # Movement rewards - reversed in Oak's Lab
        if self.current_state["map_id"] == 40:  # In Oak's Lab
            if y_change > 0:  # Moving south (toward exit)
                reward += 1.0
                print("Moving south toward exit in Oak's Lab! +1.0")
            elif y_change < 0:  # Moving north (away from exit)
                reward -= 1.0
                print("Moving north away from exit in Oak's Lab! -1.0")
        else:  # Outside Oak's Lab
            if y_change < 0:  # Moving north
                reward += 2.0
                print("Moving north! +2.0")
            elif y_change > 0:  # Moving south
                reward -= 2.0
                print("Moving south! -2.0")
        
        # Progression rewards
        if self.current_state['progression'] > self.max_progression:
            reward += 25.0
            self.max_progression = self.current_state['progression']
            print(f"New area reached! +25.0")
        elif self.current_state['progression'] >= 0:
            reward += 0.1
            print("On correct path +0.1")
        else:
            reward -= 1.0
            print("Off optimal path! -1.0")
        
        # Add small reward for being near exit in Oak's Lab
        if self.current_state["map_id"] == 40:
            target_y = 9  # Assuming this is near the exit
            distance_to_exit = abs(self.current_state["y"] - target_y)
            if distance_to_exit < 2:  # If close to exit
                reward += 0.5
                print("Near exit in Oak's Lab! +0.5")
        
        # Other rewards
        if self._badges_reward(new_state) > 0:
            reward += 100.0
            print("Badge obtained! +100.0")
        
        if self._is_grass_tile() and self.current_state['progression'] >= self.OPTIMAL_PATH.index(1):
            reward += 0.2
            print("In grass tile +0.2")
        
        print(f"Total Reward: {reward}\n")
        return reward

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        if self.steps >= 1000:
            return True
        if game_stats["party_size"] > 0 and sum(game_stats["hp"]["current"]) == 0:
            return True
        return False