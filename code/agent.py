import math, sys
import numpy as np

from reward import Reward

from rl_agent.luxai2021.game.game import Game
from rl_agent.luxai2021.game.constants import Constants
from luxai2021.game.constants import LuxMatchConfigs_Default

from imitation_learning.agent import agent as agent_imit
from working_title.agent import agent as agent_wt
# from rl_ait.agent import agent as agent_ait
# from simple.agent import agent as agent_simple

experts = [agent_imit, agent_wt]
max_reward = 1000

DIRECTIONS = Constants.DIRECTIONS
game_state = None

algos = ["EXP3", "A-EXP4"]
algo = "EXP3" # params: gamma
params = {} # dict for parameters for online learning algo
assert algo in algos

meta_bot = None
rew = None

class OnlineLearner():
    def __init__(self, experts, algo, params=None):
        self.experts = experts
        self.n_experts = len(experts)
        self.rewards = []
        self.total_rewards = []
        self.played_arm = -1
        self.algo = algo
        self.params = params
        self.initialize()


    def initialize(self):

        if algo == "EXP3":
            self.L = np.zeros(self.n_experts)
            try:
                self.gamma = self.params['gamma']
            except:
                self.gamma = np.sqrt(np.log(self.n_experts)/(self.n_experts * 360))

    def run(self, observation):
        if self.algo == "EXP3":
            self.P = np.exp(self.gamma * self.L)/sum(np.exp(self.gamma * self.L))
        # choose expert to play
        i_chosen_expert = np.random.choice(self.n_experts, p=self.P)
        self.played_arm = i_chosen_expert

        # get actions from chosen expert
        for i, exp in enumerate(self.experts):
            if i == i_chosen_expert:
                actions = exp(observation, None, calc_actions=True)
            # non-chosen experts are played to update their game states, but actions are not computed
            else:
                _ = exp(observation, None, calc_actions=False)

        return(actions)

    def update(self, reward):

        if self.algo == "EXP3":
            # update importance sampling
            # update Q
            self.L = np.array([(L_i + 1 - (1-reward)/self.P[arm]) if arm == self.played_arm else (L_i + 1) for arm, L_i in enumerate(self.L)])

def agent(observation, configuration):
    global game_state
    global algo
    global meta_bot
    global rew
    global params
    configuration = None

    if observation["step"] == 0:
        config = LuxMatchConfigs_Default
        config['height'] = int(observation["updates"][1].split()[0])
        config['width'] = int(observation["updates"][1].split()[1])
        game_state = Game(config)
        game_state.reset(observation["updates"][2:]) # .reset() calls .process_updates()
        rew = Reward(game=game_state, team=observation.player)
        rew.game_start()

        meta_bot = OnlineLearner(experts, algo, params)

    else:
        game_state.process_updates(observation["updates"])
        meta_bot.update(rew.get_reward())


    ### AI Code goes down here! ### 
    # player = observation.player    
    actions = meta_bot.run(observation)
    return actions
