import math, sys, os
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
expert_names = ['imitation_learning', 'working_title']
max_reward = 1000

DIRECTIONS = Constants.DIRECTIONS
game_state = None

algos = ["EXP3", "EXP3++"]
algo = "EXP3" 
params = {} # dict for parameters for online learning algo
"""Params:
EXP3: gamma
EXP3++: c
"""

assert algo in algos

meta_bot = None
rew = None

def contains_nan(arr):
    return True if sum(np.isnan(arr)) > 0 else False

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
        if self.algo == "EXP3":
            self.L = np.zeros(self.n_experts)
            try:
                self.gamma = self.params['gamma']
            except:
                self.gamma = np.sqrt(np.log(self.n_experts)/(self.n_experts * 360))
        elif self.algo == "EXP3++":
            self.L = np.zeros(self.n_experts)
            self.t = 1
            try: # used in computing xi
                self.c = self.params['c']
            except:
                self.c = 18
        else:
            raise ValueError(f"Algorithm {self.algo} not recognized, expected something in {algos}")


    def xi(self, a):
        """Used for EXP3++"""
        delta_hat = min(1, (self.L[a] - min(self.L))/self.t)
        return(self.c * np.log(self.t ** 2)/(self.t * delta_hat ** 2))

    def eta(self):
        return(0.5 * np.sqrt(np.log(self.n_experts))/(self.t * self.n_experts))

    def run(self, observation):
        if self.algo == "EXP3":
            self.P = np.exp(self.gamma * self.L)/sum(np.exp(self.gamma * self.L))

        elif self.algo == "EXP3++":
            eps = [min([1/(2 * self.n_experts), 0.5*np.sqrt(np.log(self.n_experts)/(self.t * self.n_experts)), self.xi(a)]) for a in range(self.n_experts)]
            assert not contains_nan(eps)
            rho = [np.exp(self.eta() * self.L[a]) for a in range(self.n_experts)]
            assert not contains_nan(rho)
            rho /= sum(rho)
            assert not contains_nan(rho)
            sum_eps = sum(eps)

            self.P = [(1-sum_eps) * rho[a] + eps[a] for a in range(self.n_experts)]
            assert not contains_nan(self.P)
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

        elif self.algo == "EXP3++":
            loss = (reward - max_reward)/max_reward
            self.L[self.played_arm] += loss/self.P[self.played_arm]
            assert not contains_nan(self.L)
            self.t += 1

def agent(observation, configuration):
    global game_state
    global algo
    global meta_bot
    global rew
    global params
    global max_reward
    global expert_names
    configuration = None

    if observation["step"] == 0:
        config = LuxMatchConfigs_Default
        config['height'] = int(observation["updates"][1].split()[0])
        config['width'] = int(observation["updates"][1].split()[1])
        game_state = Game(config)
        game_state.reset(observation["updates"][2:]) # .reset() calls .process_updates()
        rew = Reward(game=game_state, team=observation.player, max_reward=max_reward)
        rew.game_start()

        meta_bot = OnlineLearner(experts, algo, params)

        os.remove('chosen_agent.log')

    else:
        game_state.process_updates(observation["updates"])
        reward = rew.get_reward()
        meta_bot.update(reward)

        with open('chosen_agent.log', 'a') as f:
            f.write(expert_names[meta_bot.played_arm] + ',' + str(reward*max_reward) + ',' + str(reward) + '\n')

    ### AI Code goes down here! ### 
    # player = observation.player    
    actions = meta_bot.run(observation)

    return actions
