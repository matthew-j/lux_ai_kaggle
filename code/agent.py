import math, sys, os
import numpy as np

from reward import Reward

from rl_agent.luxai2021.game.game import Game
from rl_agent.luxai2021.game.constants import Constants
from rl_agent.luxai2021.game.constants import LuxMatchConfigs_Default

from working_title.agent import agent as agent_wt

from kaban.agent_tb import get_action
from kaban.agent_tb import agent_tb as agent_kaban_tb
from kaban.agent_dr import agent_dr as agent_kaban_dr
from kaban.agent_rl import agent_rl as agent_kaban_rl


# log stuff: sys.stdout.write(<some string text here>)

# if using exp4stochastic and nexp, code automatically keeps only the kaban agents
# experts = [agent_wt, agent_kaban_tb, agent_kaban_dr, agent_kaban_rl]
# expert_names = ['working_title', 'imitation_tb', 'imitation_dr', 'imitation_rl']

experts = [agent_kaban_tb, agent_kaban_dr]
expert_names = [ 'imitation_tb', 'imitation_dr']


log = False if os.path.exists('/kaggle_simulations') else True

algos = ["EXP3", "EXP3PP", "EXP3Light", "EXP4", "EXP4Stochastic", "NEXP"]
algo = "EXP4"
params = {} # dict for parameters for online learning algo
"""Params:
EXP3: gamma
EXP3++: c
EXP3Light: max_loss
EXP4: gamma, eta (higher means less exploration)
"""

assert algo in algos
# if algo in ['EXP4Stochastic', 'NEXP']:
#     experts = [agent_kaban_tb, agent_kaban_dr, agent_kaban_rl]
#     expert_names = ['imitation_tb', 'imitation_dr', 'imitation_rl']

# Setting global variables
meta_bot = None
rew = None
DIRECTIONS = Constants.DIRECTIONS
game_state = None

def contains_nan(arr):
    """Determine if an array has nans"""
    return True if sum(np.isnan(arr)) > 0 else False

class OnlineLearner():
    def __init__(self, experts, params=None):
        self.experts = experts
        self.n_experts = len(experts)
        self.rewards = []
        self.total_rewards = []
        self.played_arm = -1
        self.params = params
        self.initialize()

    def sample_and_play(self, observation):
        """Takes self.P as the probability distribution for experts. Automatically samples from self.P and chooses an
        expert, plays that expert, and returns the actions recommended by the expert"""

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
    """Plays EXP3 algorithm, treating agents as arms."""
    def __init__(self, experts, params=None):
        super().__init__(experts, params)
    def initialize(self):
        self.L = np.zeros(self.n_experts)
        try:
            self.gamma = self.params['gamma']
        except:
            self.gamma = np.sqrt(np.log(self.n_experts)/(self.n_experts * 360))

    def run(self, observation):
        self.P = np.exp(self.gamma * self.L)/sum(np.exp(self.gamma * self.L))
        actions = self.sample_and_play(observation)
        return(actions)

    def update(self, reward):
        self.L = np.array([(L_i + 1 - (1-reward)/self.P[arm]) if arm == self.played_arm else (L_i + 1) for arm, L_i in enumerate(self.L)])
    
class EXP3PP(OnlineLearner):
    """Plays EXP3++ algorithm, treating agents as arms."""
    def __init__(self, experts, params=None):
        super().__init__(experts, params)

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
    """Plays EXP3Light algorithm with agents as arms."""
    def __init__(self, experts, params=None):
        super().__init__(experts, params)

    def initialize(self):
        try:
            self.max_loss = self.params['max_loss']
        except:
            self.max_loss = 1
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
    """Plays EXP4 with agents as experts. Treats expert actions as deterministic, i.e. the recommended action is played definitely."""
    def __init__(self, experts, params=None):
        super().__init__(experts, params)

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
        X_squiggle = np.zeros(self.n_experts)
        X_squiggle[self.played_arm] = 1 - (1 - reward)/(self.Q[self.played_arm] + self.gamma)

        self.Q = np.exp(self.eta * X_squiggle) * self.Q
        self.Q /= sum(self.Q)
##
##  Exp4 stochastic expects all experts to be from the Kaban folder
##  Since each has the same logic for cities, only consider probabilities on units
class EXP4Stochastic(OnlineLearner):
    """Plays EXP4 with agents as experts. Treats expert actions as a probability distribution over actions and the action that
    is played is sampled from that distribution."""
    def __init__(self, experts, params=None):
        super().__init__(experts, params)

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
            # Divide by sum to ensure probability distribution actually sums to 1
            cur_sum = sum(action_dist)
            action_dist = [x / cur_sum for x in action_dist]
            self.played_unit_actions.append(np.random.choice(np.arange(len(action_dist)), p=action_dist))
        
        # Turn each action into command
        dest = []
        for i, unit in enumerate(units):
            policy = [1] * 5
            policy[self.played_unit_actions[i]] *= 100
            action, pos = get_action(policy, unit, dest)
            final_actions.append(action)
            dest.append(pos)

        # for logging purposes only: EXP4Stoch doesn't actually "choose" an expert
        # report expert with highest probability as the arm that is "played"
        self.played_arm = np.argmax(self.Q)

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
    """Plays the NEXP algorithm with agents as experts"""
    def __init__(self, experts, params=None):
        super().__init__(experts, params)
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
            # Scale to ensure that it sums to 1
            cur_sum = sum(p_final)
            p_final = [x / cur_sum for x in p_final]
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

        # for logging purposes only: NEXP doesn't actually "choose" an expert
        
        self.played_arm = np.argmax(self.Q)

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
    """The function that is played by the Kaggle game simulation. Takes an observation (game state) and configuration (not used)
    as inputs and outputs the actions for that game step."""
    global game_state
    global algo
    global meta_bot
    global rew
    global params
    global expert_names
    global log
    configuration = None

    if observation["step"] == 0:
        config = LuxMatchConfigs_Default
        config['height'] = int(observation["updates"][1].split()[0])
        config['width'] = int(observation["updates"][1].split()[1])
        game_state = Game(config)
        game_state.reset(observation["updates"][2:]) # .reset() calls .process_updates()
        rew = Reward(game=game_state, team=observation.player)
        rew.game_start()

        meta_bot = globals()[algo](experts, params) # create object from class specified by algo

        if log:
            os.remove('chosen_agent.log')
            with open('chosen_agent.log', 'a') as f:
                f.write(algo + '\n' + ','.join(expert_names) + '\n')

    else:
        game_state.process_updates(observation["updates"])
        reward, max_reward = rew.get_reward()
        meta_bot.update(reward)

        if log:
            with open('chosen_agent.log', 'a') as f:
                f.write(expert_names[meta_bot.played_arm] + ',' + str(reward*max_reward) + ',' + str(reward) + '\n')

    actions = meta_bot.run(observation)

    return actions
