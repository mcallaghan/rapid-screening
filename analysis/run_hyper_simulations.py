import pandas as pd
import numpy as np
import os, sys
import argparse
from functools import partial
import math
from scipy.stats import hypergeom

parser = argparse.ArgumentParser(description="run ML systematic review scenarios")
parser.add_argument('iterations', type=int)
parser.add_argument('-p', type=int, default=0)
parser.add_argument('-mpi', type=int, default=0)
args = parser.parse_args()

N_T = 50000
r_tot = 4150
r_seen = 3750

def get_recall(r_seen, r_tot):
    return r_seen/r_tot

get_recall(r_seen,r_tot)

# Lets start with a situation where we have seen half the documents, and ~85% of relevant documents
N_ML = 25000
r_ML = 3750
N_s = N_T - N_ML
K = r_tot - r_ML

# Returning to our numerical example, let's set the target recall to .95 and get K_hat
tau_target=.95
def get_K_hat(r_seen,tau_target,r_ML):
    return(math.floor(r_seen/tau_target - r_ML+1))

vec_get_K_hat = np.vectorize(get_K_hat)

## We can define a function for simulating drawing from this sample
def draw_sample(x,sample,r_ML,tau_target=0.95):
    vec_get_K_hat = np.vectorize(get_K_hat)
    sample = np.random.choice(sample, len(sample), replace=False) # each time we shuffle the sample
    r_seen = r_ML+sample.cumsum() 
    tau = r_seen/r_tot
    K_hat = vec_get_K_hat(r_seen,tau_target,r_ML)
    # work out cumulative probabilites for the each value of k (sample.cumsum()) each value of K_hat, and each value of n (np.arange(1,N_s+1))
    p = hypergeom.cdf(sample.cumsum(), N_s, K_hat, np.arange(1,N_s+1))
    target_it = np.argmax(tau>=tau_target) # The first iteration where we reach the target
    target_achievement = {"tau":tau[target_it], "i":target_it,"p":p[target_it], "K_hat":K_hat[target_it]}
    H0_it = np.argmax(p<0.05)
    H0_rejection = {"tau":tau[H0_it], "i":H0_it,"p":p[H0_it], "K_hat":K_hat[H0_it]}
    return p, tau, H0_rejection, target_achievement

iterations = args.iterations


starting_targets = [.175, .375, .575, 0.775, .875, .975]
r_ML_range = [round(x*r_tot) for x in starting_targets]
tau_range = []
p_range = []
tau_array_range = []
one_shot_p_range = []

results_df = pd.DataFrame()

for r_ml in r_ML_range:
    K_i = r_tot - r_ml
    sample = np.zeros(N_s)
    sample[:K_i] = 1
    y, tau, h, t = zip(*[draw_sample(x, sample=sample,r_ML=r_ml) for x in range(iterations)])
    H0_rejection_taus = np.array([x['tau'] for x in h])
    p_range.append(y)
    tau_array_range.append(tau)
    tau_range.append(H0_rejection_taus)  
    print(r_ml)
    failure = sum(H0_rejection_taus<.95)/iterations
    print(failure)
    
    one_shot_p = []
    for i, x in enumerate(y):
        t = tau[i]
        one_shot_it = np.random.randint(1,len(x))
        if x[one_shot_it]<0.05:
            if t[one_shot_it] >= tau_target:
                one_shot_p.append(0) # failure
            else:
                one_shot_p.append(1)
        else:
            one_shot_p.append(np.nan)

    one_shot_p = np.array(one_shot_p)
    one_shot_p_range.append(one_shot_p)
    
    results_df = results_df.append(pd.DataFrame.from_dict({
        "H0_rejection_tau": H0_rejection_taus,
        "one_shot_p": one_shot_p,
        "starting_recall": [r_ml/r_tot]*one_shot_p.shape[0]
    }))
    
if args.mpi:
    results_df.to_csv(f'../results/results_{rank}.csv', index=False)
else:
    results_df.to_csv('../results/results.csv', index=False)
    
