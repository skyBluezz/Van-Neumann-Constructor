
import os
import pdb
from tqdm import tqdm
import json
import numpy as np
import gym
import slimevolleygym
import slimevolleygym.mlp as mlp
from slimevolleygym.mlp import Model
from slimevolleygym import multiagent_rollout as rollout
from slimevolleygym import BaselinePolicy
from neat_src1 import Neat
from neat_src1.dataGatherer import DataGatherer
from neat_src1.utils import *
from sky_src.generateShapes import generate_circle_dataset, plot_dataset, generate_xor_dataset
from sky_src.engine import predict, train, eval, plot_results

# Load hyperparameters
with open('config.json') as data_file: 
  hyp = json.load(data_file)

saveout_path = "latestrun"
data = DataGatherer(saveout_path, hyp)

# Settings
random_seed = 612
save_freq = 1000
Ndatapoints = 200
Nepochs = 60
evolve_period = 9

# Log results
logdir = "ga_selfplay"
if not os.path.exists(logdir):
  os.makedirs(logdir)

global neat
neat = Neat(hyp)

inputs, targets = generate_circle_dataset(Ndatapoints)
# inputs, targets = generate_xor_dataset(Ndatapoints)
losses = []
accuracy = []
for ep in tqdm(range(Nepochs)):

  if ep < evolve_period: 
    neat.ask()
    aveloss = train(neat, inputs, targets, lr=None, backprop=False)
    print(f"Average weight shape is {np.mean([ind.wMat.shape for ind in neat.pop])}")
    neat.tell([])
  else:
    lr = hyp['learning_rate'] / (1 + (ep-evolve_period) // 20)
    aveloss = train(neat, inputs, targets, lr, backprop=True)  
  
  losses.append(aveloss)
  if ep//10 and not ep%10:
    best_ind = neat.pop[argsort([ind.fitness for ind in neat.pop])[-1]]
    preds, score, logits = eval(best_ind, inputs, targets)
    accuracy.append(score/len(inputs))

plot_results(inputs, targets, preds, losses, circle=True)
pdb.set_trace()

best_ind = neat.pop[argsort([ind.fitness for ind in neat.pop])[-1]]
eval(best_ind, inputs, targets)
