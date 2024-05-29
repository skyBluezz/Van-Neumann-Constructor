import pdb
import multiprocessing
from time import time
import numpy as np
from slimevolleygym import multiagent_rollout as rollout
from neat_src.utils import *
from train_utils import initialize, eval, finalEval, gladiatorBattle
from train_utils import neat, baseline_policy, env, p

# Initialize for visiblity for multi-processing
data, p = initialize("config.json")

neat.ask()
print(f"Loading from checkpoint..")
savedpop = loadPop(p.loadpath)
neat.pop = savedpop
  
for epoch in range(p.Nepochs):
  neat.ask()
  start = time()
  start = time()
  pool = multiprocessing.Pool(p.num_threads)
  rewards = pool.map(eval, neat.pop)
  pool.close()
  pool.join()

  rewards = np.array(rewards)
  neat.tell(rewards)
  print(f"EPOCH {epoch}, {time()-start:0.2f} s, ETA = {((p.Nepochs - epoch)*(time()-start) / 3600):0.2f} hrs")

# Evaluate against baseline and render a gif
finalEval(data, neat, baseline_policy, p)
pdb.set_trace()
