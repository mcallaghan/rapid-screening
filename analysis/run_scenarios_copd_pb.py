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

importlib.reload(rr)

dfs = []
for path in [("ProtonBeam","all"),("COPD","copd")]:
    document_index = rr.parse_pb_xml(f'../data/{path[0]}/{path[1]}.xml')
    document_index = document_index.drop_duplicates()
    document_index['rec-number'] = document_index['rec-number'].astype(int)
    
    relevant_index = pd.read_csv(
    f'../data/{path[0]}/relevant.txt',header=None,
    names=["rec-number"])
    
    relevant_index['relevant'] = 1
    
    df = pd.merge(
        document_index,
        relevant_index,
        how="left"
    )
    
    df['review'] = path[0]
    dfs.append(df)
    
df = pd.concat(dfs)
df['relevant'] = df['relevant'].fillna(0)

importlib.reload(rr)

results = []
iterations = args.iterations

df['x'] = df['ab']

models = [
    SVC(kernel='linear',class_weight='balanced',probability=True)
]

for name, group in df.groupby('review'):
    group = group.dropna().reset_index(drop=True)
    if group.shape[0] < 2000:
        continue
    for s in [200, 500]:
        ss = rr.ScreenScenario(
            group, models, s, [50, 100, 200], name
        )
        for i in range(iterations):
            print(i)
            r = ss.screen(i, True)
            if r is not None:
                results.append(r)

results_df = pd.DataFrame.from_dict(results)
results_df.to_csv('../results/results_pb_copd.csv', index=False)
