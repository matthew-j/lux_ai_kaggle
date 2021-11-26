import math, sys, os
import numpy as np

from reward import Reward

from rl_agent.luxai2021.game.game import Game
from rl_agent.luxai2021.game.constants import Constants
from rl_agent.luxai2021.game.constants import LuxMatchConfigs_Default

from imitation_learning.agent import agent as agent_imit
from working_title.agent import agent as agent_wt

from kaban.agent_tb import get_action
from kaban.agent_tb import agent_tb as agent_kaban_tb
from kaban.agent_dr import agent_dr as agent_kaban_dr
from kaban.agent_rl import agent_rl as agent_kaban_rl


# log stuff: sys.stdout.write(<some string text here>)

# experts = [agent_imit, agent_wt, agent_kaban_tb, agent_kaban_dr, agent_kaban_rl]
# expert_names = ['imitation_learning', 'working_title', 'imitation_toad_brigade', 'imitation_dr', 'imitation_rl']

# for exp4stochastic and nexp
experts = [agent_kaban_tb, agent_kaban_dr, agent_kaban_rl]
expert_names = ['imitation_toad_brigade', 'imitation_dr', 'imitation_rl']

max_reward = 500
log = True

algos = ["EXP3", "EXP3++", "EXP3Light", "EXP4", "EXP4Stochastic", "NEXP"]
algo = "EXP4Stochastic" 
params = {} # dict for parameters for online learning algo
"""Params:
EXP3: gamma
EXP3++: c
EXP3Light: max_loss
EXP4: gamma, eta (higher means less exploration)
"""

assert algo in algos
if algo in ['EXP4Stochastic', 'NEXP']:
    experts = [agent_kaban_tb, agent_kaban_dr, agent_kaban_rl]
    expert_names = ['imitation_toad_brigade', 'imitation_dr', 'imitation_rl']

meta_bot = None
rew = None
DIRECTIONS = Constants.DIRECTIONS
game_state = None

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

    def sample_and_play(self, observation):
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

class EXP3(OnlineLearner):
    def __init__(self, experts, algo, params=None):
        super().__init__(experts, algo, params)
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
    def run(self, observation):
        self.P = np.exp(self.gamma * self.L)/sum(np.exp(self.gamma * self.L))
        actions = self.sample_and_play(observation)
        return(actions)

    def update(self, reward):
        self.L = np.array([(L_i + 1 - (1-reward)/self.P[arm]) if arm == self.played_arm else (L_i + 1) for arm, L_i in enumerate(self.L)])
    
class EXP3PP(OnlineLearner):
    def __init__(self, experts, algo, params=None):
        super().__init__(experts, algo, params)


    def initialize(self):
        self.L = np.zeros(self.n_experts)
        self.t = 1
        try: # used in computing xi
            self.c = self.params['c']
        except:
            self.c = 18

    def xi(self, a):
        delta_hat = min(1, (self.L[a] - min(self.L))/self.t)
        return(self.c * np.log(self.t ** 2)/(self.t * delta_hat ** 2))

    def eta(self):
        return(0.5 * np.sqrt(np.log(self.n_experts))/(self.t * self.n_experts))

    def run(self, observation):
        eps = [min([1/(2 * self.n_experts), 0.5*np.sqrt(np.log(self.n_experts)/(self.t * self.n_experts)), self.xi(a)]) for a in range(self.n_experts)]
        assert not contains_nan(eps)
        rho = [np.exp(self.eta() * self.L[a]) for a in range(self.n_experts)]
        assert not contains_nan(rho)
        rho /= sum(rho)
        assert not contains_nan(rho)
        sum_eps = sum(eps)

        self.P = [(1-sum_eps) * rho[a] + eps[a] for a in range(self.n_experts)]
        assert not contains_nan(self.P)

        actions = self.sample_and_play(observation)
        return(actions)
    
    def update(self, reward):
        loss = 1 - reward
        self.L[self.played_arm] += loss/self.P[self.played_arm]
        assert not contains_nan(self.L)
        self.t += 1

class EXP3Light(OnlineLearner):
    def __init__(self, experts, algo, params=None):
        super().__init__(experts, algo, params)

    def initialize(self):
        global max_reward
        try:
            self.max_loss = self.params['max_loss']
        except:
            self.max_loss = max_reward
        self.r = 0
        self.L_squiggle = np.zeros(self.n_experts)
    
    def run(self, observation):
        eta = np.sqrt(2*(np.log(self.n_experts) + self.n_experts * np.log(360)/(self.n_experts * 4 ** self.r)))
        self.P = np.exp(-1 * eta * self.L_squiggle / self.max_loss)
        self.P /= sum(self.P)
        
        actions = self.sample_and_play(observation)
        return(actions)
    def update(self, reward):
        loss = 1 - reward
        self.L_squiggle[self.played_arm] += loss/self.P[self.played_arm]
        min_L_squiggle = min(self.L_squiggle)
        if min_L_squiggle/self.max_loss > 4 ** self.r:
            self.r = np.ceil(np.log(min_L_squiggle/self.max_loss)/np.log(4)) # log_4(.)

class EXP4(OnlineLearner):
    def __init__(self, experts, algo, params=None):
        super().__init__(experts, algo, params)

    def initialize(self):
        self.eta = 0.01 # np.sqrt(2*np.log(self.n_experts)/(360 * N_ARMS))
        self.gamma = 0
        self.Q = [1/self.n_experts] * self.n_experts
    def run(self, observation):
        i_chosen_expert = np.random.choice(self.n_experts, p=self.Q)
        self.played_arm = i_chosen_expert

        self.all_actions = []
        # get actions from chosen expert
        for i, exp in enumerate(self.experts):
            self.all_actions.append(exp(observation, None, calc_actions=True))

        return(self.all_actions[self.played_arm])
        
    def update(self, reward):
        """It is difficult to model the distribution across arms since the number of potential
        actions increases very quickly. Therefore, we make certain assumptions in the update
        computations.
        
        1) We treat each expert as picking a deterministic action at each time step. In other words,
        if an agent is chosen, it plays the action that it picked with probability 1.

        2) Since there are so many potential actions (multiple different choices for many different
        units), it is statistically unlikely that two agents pick the exact same choices for all units.
        Therefore, we assume the probability that an action is chosen is the same as the probability of
        choosing the agent that chose that action. In other words, P[i] == Q[j] if agent j chose action i.
        
        3) Due to assumption 1, each row in the matrix E (which describes each agent and their probability 
        distributions of action choices) has a single 1 and the rest of the values are 0. 

        For the selected action/agent, X_hat is 1 - (1 - scaled_reward)/(Q[selected_agent] + gamma). For 
        all other actions, the value is 1.

        For X_squiggle, the element corresponding to the selected agent has value 
        1 - (1 - scaled_reward)/(Q[selected_agent] + gamma) since we are assuming each agent picks
        their action with probability 1. For non-selected agents, the value is 0. 
        """

        X_squiggle = np.zeros(self.n_experts)
        X_squiggle[self.played_arm] = 1 - (1 - reward)/(self.Q[self.played_arm] + self.gamma)

        self.Q = np.exp(self.eta * X_squiggle) * self.Q
        self.Q /= sum(self.Q)
##
##  Exp4 stochastic expects all experts to be from the Kaban folder
##  Since each has the same logic for cities, only consider probabilities on units
class EXP4Stochastic(OnlineLearner):
    def __init__(self, experts, algo, params=None):
        super().__init__(experts, algo, params)

    def softmax(self, nums):
        return (np.exp(nums) / sum(np.exp(nums))).tolist()

    def initialize(self):
        self.eta = 0.001 # np.sqrt(2*np.log(self.n_experts)/(360 * N_ARMS))
        self.gamma = 0
        self.Q = np.array([1/self.n_experts] * self.n_experts)

    def run(self, observation):
        self.unit_actions = {}
        self.played_unit_actions = []
        units = []
        final_actions = []
        
        # Gather distributions from each expert for each unit
        for i, exp in enumerate(self.experts):
            units, probabilities, city_actions = exp(observation, None, calc_actions=True, stochastic_actions=True)
            if i == 0:
                units = units
                final_actions.extend(city_actions)
            for j, unit in enumerate(units):
                if j not in self.unit_actions:
                    self.unit_actions[j] = [self.softmax(probabilities[j])]
                else:
                    self.unit_actions[j].append(self.softmax(probabilities[j]))
        
        # Sample each distribution to get actions
        for key, value in self.unit_actions.items():
            expert_actions_dist = np.array(value).transpose()
            action_dist = np.matmul(expert_actions_dist, self.Q.transpose()).tolist()
            self.played_unit_actions.append(np.random.choice(np.arange(len(action_dist)), p=self.softmax(action_dist)))
        
        # Turn each action into command
        dest = []
        for i, unit in enumerate(units):
            policy = [1] * 5
            policy[self.played_unit_actions[i]] *= 100
            action, pos = get_action(policy, unit, dest)
            final_actions.append(action)
            dest.append(pos)

        return final_actions
        
    def update(self, reward):
        self.eta = 0.001 # np.sqrt(2*np.log(self.n_experts)/(360 * N_ARMS))
        self.gamma = 0
        expert_probabilities = []
        pt_j = 0

        # Get expert probabilities of choosing what happened
        for i, exp in enumerate(self.experts):
            prob = 1
            j = 0
            for key, value in self.unit_actions.items():
                prob *= value[i][self.played_unit_actions[j]]
                j += 1
            expert_probabilities.append([prob, 1-prob])
        
        #calc pt_j
        for i in range(len(expert_probabilities)):
            pt_j += expert_probabilities[i][0] * self.Q[i]
        
        # estimate reward
        reward_vec_np = 1 - np.array([(1 - reward) / (pt_j + self.gamma), 0])
        expert_probabilities_np = np.array(expert_probabilities)
        expert_rewards = np.matmul(expert_probabilities_np, reward_vec_np)
        
        self.Q = np.exp(self.eta * expert_rewards) * self.Q
        self.Q /= sum(self.Q)
##
##  NEXP stochastic expects all experts to be from the Kaban folder
##  Since each has the same logic for cities, only consider probabilities on units
class NEXP(OnlineLearner):
    def __init__(self, experts, algo, params=None):
        super().__init__(experts, algo, params)
        self.alpha = .1
        self.c_threshold = 0.1

    def softmax(self, nums):
        return (np.exp(nums) / sum(np.exp(nums))).tolist()

    def initialize(self):
        self.eta = 0.001 # np.sqrt(2*np.log(self.n_experts)/(360 * N_ARMS))
        self.gamma = 0
        self.Q = np.array([1/self.n_experts] * self.n_experts)

    def LP_Mix_Solve(self, p_est, pmin):
        c = 1
        c_last = -40

        while c - c_last > self.c_threshold:
            A_zero = []
            A_one = []
            for i in range(len(pmin)):
                if pmin[i] >= c * p_est[i]:
                    A_zero.append(i)
                else:
                    A_one.append(i)

            numerator = 1 - sum([pmin[i] for i in A_zero])
            denominator = sum([p_est[i] for i in A_one])
            c_last = c
            c = numerator / denominator

        return [max(pmin[i], c * p_est[i]) for i in range(len(pmin))]

    def run(self, observation):
        self.unit_actions = {}
        self.played_unit_actions = []
        self.p_a = 1
        units = []
        final_actions = []

        # Gather distributions from each expert for each unit
        for i, exp in enumerate(self.experts):
            units, probabilities, city_actions = exp(observation, None, calc_actions=True, stochastic_actions=True)
            if i == 0:
                units = units
                final_actions.extend(city_actions)
            for j, unit in enumerate(units):
                if j not in self.unit_actions:
                    self.unit_actions[j] = [self.softmax(probabilities[j])]
                else:
                    self.unit_actions[j].append(self.softmax(probabilities[j]))

        # Sample each distribution to get actions
        for key, value in self.unit_actions.items():
            expert_actions_dist = np.array(value).transpose()
            action_dist = np.matmul(expert_actions_dist, self.Q.transpose()).tolist()

            pmin = [self.alpha * max(expert_actions_dist.tolist()[i]) for i in range(5)]
            p_final = self.LP_Mix_Solve(action_dist, pmin)
            self.played_unit_actions.append(np.random.choice(np.arange(len(action_dist)), p=p_final))
            self.p_a *= p_final[self.played_unit_actions[-1]] 

        # Turn each action into command
        dest = []
        for i, unit in enumerate(units):
            policy = [1] * 5
            policy[self.played_unit_actions[i]] *= 100
            action, pos = get_action(policy, unit, dest)
            final_actions.append(action)
            dest.append(pos)

        return final_actions

    def update(self, reward):
        self.gamma = 0
        expert_probabilities = []
        pt_j = 0

        # Get expert probabilities of choosing what happened
        for i, exp in enumerate(self.experts):
            prob = 1
            j = 0
            for key, value in self.unit_actions.items():
                prob *= value[i][self.played_unit_actions[j]]
                j += 1
            expert_probabilities.append(prob)

        reward_vec = [reward * expert_probabilities[i] / self.p_a for i in range(len(self.experts))]
        self.Q = np.array([self.Q[i] * np.exp(self.alpha * reward_vec[i]) for i in range(len(self.experts))])
        self.Q /= sum(self.Q)

def agent(observation, configuration):
    global game_state
    global algo
    global meta_bot
    global rew
    global params
    global max_reward
    global expert_names
    global log
    configuration = None

    if observation["step"] == 0:
        config = LuxMatchConfigs_Default
        config['height'] = int(observation["updates"][1].split()[0])
        config['width'] = int(observation["updates"][1].split()[1])
        game_state = Game(config)
        game_state.reset(observation["updates"][2:]) # .reset() calls .process_updates()
        rew = Reward(game=game_state, team=observation.player, max_reward=max_reward)
        rew.game_start()

        meta_bot = globals()[algo](experts, algo, params) # create object from class specified by algo

        if log:
            os.remove('chosen_agent.log')
            with open('chosen_agent.log', 'a') as f:
                f.write(algo + '\n' + ','.join(expert_names) + '\n')

    else:
        game_state.process_updates(observation["updates"])
        reward = rew.get_reward()
        meta_bot.update(reward)

        if log:
            with open('chosen_agent.log', 'a') as f:
                f.write(expert_names[meta_bot.played_arm] + ',' + str(reward*max_reward) + ',' + str(reward) + '\n')

    ### AI Code goes down here! ### 
    # player = observation.player    
    actions = meta_bot.run(observation)

    return actions
