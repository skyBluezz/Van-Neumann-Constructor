from time import time
import json, os
from argparse import Namespace
import gym
from slimevolleygym import multiagent_rollout as rollout
from slimevolleygym import BaselinePolicy
from neat_src import Neat
from neat_src.dataGatherer import DataGatherer
from neat_src.utils import *

global neat 
global baseline_policy
global env
global tournament_length
global p

def initialize(config_path):
    # Load hyperparameters
    with open(config_path) as data_file: 
        hyp = json.load(data_file)
        global p
        p = Namespace(**hyp)
    if not os.path.exists(p.logdir):
        os.makedirs(p.logdir)

    data = DataGatherer(p.saveout_path, hyp)

    # Globalize for visibility during multi-processing
    global neat
    neat = Neat(hyp)

    global env
    env = gym.make("SlimeVolley-v0")
    env.survival_bonus = p.survival_bonus
    env.seed(612)

    global baseline_policy
    baseline_policy = BaselinePolicy()

    global tournament_length
    tournament_length = p.tournament_length

    return data, p

def eval(ind):
  ind_reward = 0
  for tourn in range(tournament_length):
    score, length, distance, jumping = rollout(env, ind, baseline_policy)
    ind_reward += score + length*p.survival_bonus + distance*p.distance_bonus - jumping*p.jumping_bonus
  return ind_reward

def finalEval(data, neat, baseline_policy, p):
  """Runs final evaluation of the best individual against baseline, over a 100 game tournament
  Finally, renders a game between your player and the baseline. 
  """
  gatherData(data,neat,savePop=True)
  best_ind = neat.pop[argsort([ind.fitness for ind in neat.pop])[-1]]
  neat_wins = 0
  baseline_wins = 0
  for tourn in range(100):
    score, length, distance, jumping = rollout(env, best_ind, baseline_policy, render_mode=False)
    # if score is positive, it means ind won.
    print(score, length*p.survival_bonus, distance*p.distance_bonus, jumping*p.jumping_bonus)
    if score == 0: 
      continue
    if score < 0:
      baseline_wins += 1
    if score > 0:
      neat_wins += 1
  print(f"Baseline wins are {baseline_wins}")
  print(f"Neat wins are {neat_wins}")
  rollout(env, best_ind, baseline_policy, render_mode=True)

def gladiatorBattle(neat1, neat2, p):
  """Takes two neat populations to compete in a gladiator round-robin against each other. 
  The top half of each colony are concatenated, forming a single strong population. 

  Args:
    neat1   - (Neat)       - Neat object containing population 1
    neat2   - (Neat)       - Neat object containing population 2
    p       - (Namespace)  - Algorithm hyperparameters (see config.json)
  
  Todo:
    Ensure the final neat object properly handles its innov attributes after conjoining from 
    two colonies.
  """
  neat1.clearFitness()
  neat2.clearFitness()
  popsize = len(neat1.pop)
  assert popsize == len(neat2.pop)

  # Each ind plays every ind in the other population
  for ind1 in neat1.pop:
    for ind2 in neat2.pop: 
      for _ in range(p.gladiator_battle_length):
        score, _, _, _ = rollout(env, ind1, ind2, render_mode=False)
        ind1.fitness += score
        ind2.fitness -= score

  neat = Neat(vars(p))

  # Combine the best half from each population
  neat.pop = neat1.pop[argsort([ind.fitness for ind in neat1.pop])][:popsize // 2] + \
             neat2.pop[argsort([ind.fitness for ind in neat2.pop])][:popsize // 2]
  
  return neat