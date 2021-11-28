import copy
import random

from rl_agent.luxai2021.game.constants import Constants

class Reward():
    def __init__(self, game, team) -> None:
        super().__init__()
        
        self.stats = None
        self.stats_last_game = None

        self.game = game
        self.team = team
        self.max_reward = 100
    def game_start(self):
        """
        This funciton is called at the start of each game. Use this to
        reset and initialize per game. Note that self.team may have
        been changed since last game. The game map has been created
        and starting units placed.

        Args:
            game ([type]): Game.
        """
        self.last_generated_fuel = self.game.stats["teamStats"][self.team]["fuelGenerated"]
        self.last_resources_collected = copy.deepcopy(self.game.stats["teamStats"][self.team]["resourcesCollected"])
        if self.stats != None:
            self.stats_last_game =  self.stats
        self.stats = {
            "rew/r_total": 0,
            "rew/r_wood": 0,
            "rew/r_coal": 0,
            "rew/r_uranium": 0,
            "rew/r_research": 0,
            "rew/r_city_tiles_end": 0,
            "rew/r_fuel_collected":0,
            "rew/r_units":0,
            "rew/r_city_tiles":0,
            "game/turns": 0,
            "game/research": 0,
            "game/unit_count": 0,
            "game/cart_count": 0,
            "game/city_count": 0,
            "game/city_tiles": 0,
            "game/wood_rate_mined": 0,
            "game/coal_rate_mined": 0,
            "game/uranium_rate_mined": 0,
            "rew/r_carts": 0,
        }
        self.is_last_turn = False

        # Calculate starting map resources
        type_map = {
            Constants.RESOURCE_TYPES.WOOD: "WOOD",
            Constants.RESOURCE_TYPES.COAL: "COAL",
            Constants.RESOURCE_TYPES.URANIUM: "URANIUM",
        }

        self.fuel_collected_last = 0
        self.fuel_start = {}
        self.fuel_last = {}
        for type, type_upper in type_map.items():
            self.fuel_start[type] = 0
            self.fuel_last[type] = 0
            for c in self.game.map.resources_by_type[type]:
                self.fuel_start[type] += c.resource.amount * self.game.configs["parameters"]["RESOURCE_TO_FUEL_RATE"][type_upper]

        self.research_last = 0
        self.units_last = 0
        self.city_tiles_last = 0
        self.carts_last = 0

    def get_reward(self):
        """
        Returns the reward function for this step of the game.
        """

        # Get some basic stats
        unit_count = len(self.game.state["teamStates"][self.team % 2]["units"])
        cart_count = 0
        for id, u in self.game.state["teamStates"][self.team % 2]["units"].items():
            if u.type == Constants.UNIT_TYPES.CART:
                cart_count += 1

        unit_count_opponent = len(self.game.state["teamStates"][(self.team + 1) % 2]["units"])
        research = min(self.game.state["teamStates"][self.team]["researchPoints"], 200.0) # Cap research points at 200
        city_count = 0
        city_count_opponent = 0
        city_tile_count = 0
        city_tile_count_opponent = 0
        for city in self.game.cities.values():
            if city.team == self.team:
                city_count += 1
            else:
                city_count_opponent += 1

            for cell in city.city_cells:
                if city.team == self.team:
                    city_tile_count += 1
                else:
                    city_tile_count_opponent += 1
        
        # Basic stats
        self.stats["game/research"] = research
        self.stats["game/city_tiles"] = city_tile_count
        self.stats["game/city_count"] = city_count
        self.stats["game/unit_count"] = unit_count
        self.stats["game/cart_count"] = cart_count
        self.stats["game/turns"] = self.game.state["turn"]

        rewards = {}

        # Give up to 1.0 reward for each resource based on % of total mined.
        type_map = {
            Constants.RESOURCE_TYPES.WOOD: "WOOD",
            Constants.RESOURCE_TYPES.COAL: "COAL",
            Constants.RESOURCE_TYPES.URANIUM: "URANIUM",
        }
        fuel_now = {}

        # Calc max reward
        # Max reward of 12.5 + 0.06 * #cities + 0.1*#units at this point
        self.max_reward = 12.5 + 0.06 * self.city_tiles_last + 0.1 * self.units_last + 0.2 * (self.city_tiles_last + self.units_last)

        # Max reward of 3 from this
        for type, type_upper in type_map.items():
            fuel_now = self.game.stats["teamStats"][self.team]["resourcesCollected"][type] * self.game.configs["parameters"]["RESOURCE_TO_FUEL_RATE"][type_upper]
            rewards["rew/r_%s" % type] = (fuel_now - self.fuel_last[type]) / self.fuel_start[type]
            self.stats["game/%s_rate_mined" % type] = fuel_now / self.fuel_start[type]
            self.fuel_last[type] = fuel_now
        
        # Give more incentive for coal and uranium
        # Max reward of 1 + 2 + 4  = 7 at this point
        rewards["rew/r_%s" % Constants.RESOURCE_TYPES.COAL] *= 2
        rewards["rew/r_%s" % Constants.RESOURCE_TYPES.URANIUM] *= 4
        
        # Give a reward based on amount of fuel collected. 1.0 reward for each 20K fuel gathered.
        # Lets assume that we will not collect more than 100k of fuel per turn. So max pts for this turn is 5.
        # Max reward of 7 + 5 = 12 at this point. 
        fuel_collected = self.game.stats["teamStats"][self.team]["fuelGenerated"]
        rewards["rew/r_fuel_collected"] = ( (fuel_collected - self.fuel_collected_last) / 20000 )
        self.fuel_collected_last = fuel_collected

        # Give a reward for unit creation/death. 0.05 reward per unit.
        # Can create at most 1 unit per city. Max reward for this is 0.05 * # cities
        # Max reward of 12 + 0.05 * #cities at this point
        rewards["rew/r_units"] = (unit_count - self.units_last) * 0.05
        self.units_last = unit_count

        # Give a reward for cart creation/death. 0.01 reward per cart
        # Can create at most 1 cart per city. Max reward for this is 0.01 * # cities
        # Max reward of 12 + 0.06 * #cities at this point
        rewards["rew/r_carts"] = (cart_count - self.carts_last) * 0.01

        # Tiny reward for research to help. Up to 0.5 reward for this.
        # Max reward of 12.5 + 0.06 * #cities + 0.1*#units at this point
        rewards["rew/r_research"] = (research - self.research_last) / (200 * 2)
        self.research_last = research

        # Give a reward for unit creation/death. 0.1 reward per city.
        # Can create at most 1 per unit. Max reward for this is 0.1 * # units
        # Max reward of 12 + 0.06 * #cities + 0.1*#units at this point
        rewards["rew/r_city_tiles"] = (city_tile_count - self.city_tiles_last) * 0.1

        # Give a reward for total number of cities to incentivize maintaining cities. 
        # Max reward = 0.2 * (city_tiles_last + unit_count) -> maintain all old cities, each unit builds a new city
        rewards["rew/r_city_tiles_end"] = city_tile_count * 0.2
        self.city_tiles_last = city_tile_count

        # Update the stats and total reward
        reward = 0
        for name, value in rewards.items():
            self.stats[name] += value
            reward += value
        self.stats["rew/r_total"] += reward

        # Print the final game stats sometimes
        if random.random() <= 0.15:
            stats_string = []
            for key, value in self.stats.items():
                stats_string.append("%s=%.2f" % (key, value))
            print(",".join(stats_string))

        assert reward <= self.max_reward
        scaled_reward = reward/self.max_reward
        return scaled_reward, self.max_reward