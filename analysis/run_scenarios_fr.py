import pandas as pd
import numpy as np
import rapid_review as rr
import os
import matplotlib.pyplot as plt
import importlib
from sklearn.svm import SVC, OneClassSVM
import argparse

parser = argparse.ArgumentParser(description="run ML systematic review scenarios")
parser.add_argument('iterations', type=int)
args = parser.parse_args()

datasets = ["Hall","Kitchenham","Radjenovic","Wahono"]

dfs = []
cols = ["Abstract", "label"]
for d in datasets:
    df = pd.read_csv(f'../data/fastread/{d}.csv', encoding="ISO-8859-1")[cols]
    df.loc[df['label']=="no", "relevant"] = 0
    df.loc[df['label']=="yes", "relevant"] = 1
    df["review"] = d
    dfs.append(df)
    
frdf = pd.concat(dfs)

models = [
    SVC(kernel='linear',class_weight='balanced',probability=True)
]
iterations = args.iterations
results = []
for name, group in frdf.groupby('review'):
    df = group.dropna().reset_index(drop=True)
    df['x'] = df['Abstract']
    for s in [200, 500]:
        ss = rr.ScreenScenario(
            df, models, s, [50, 100, 200], name
        )
        for i in range(iterations):
            print(i)
            r = ss.screen(i, True)
            if r is not None:
                results.append(r)

results_df = pd.DataFrame.from_dict(results)
results_df.to_csv('../results/results_fr.csv', index=False)
