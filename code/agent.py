import math, sys

from simple.lux.game import Game
from simple.lux.game_map import Cell, RESOURCE_TYPES
from simple.lux.constants import Constants
from simple.lux.game_constants import GAME_CONSTANTS
from simple.lux import annotate

from imitation_learning.agent import agent as agent_imit
from working_title.agent import agent as agent_wt
from rl_ait.agent import agent as agent_ait
from simple.agent import agent as agent_simple

DIRECTIONS = Constants.DIRECTIONS
game_state = None


def agent(observation, configuration):
    global game_state

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])
    
    actions = []

    ### AI Code goes down here! ### 
    # player = game_state.players[observation.player]
    # opponent = game_state.players[(observation.player + 1) % 2]
    # width, height = game_state.map.width, game_state.map.height

    actions = agent_imit(observation, configuration)    
    return actions
