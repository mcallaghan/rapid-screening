import pandas as pd
import numpy as np
import rapid_review as rr
import os
import matplotlib.pyplot as plt
import importlib
from functools import partial
from sklearn.svm import SVC, OneClassSVM
import argparse

parser = argparse.ArgumentParser(description="run ML systematic review scenarios")
parser.add_argument('iterations', type=int)
parser.add_argument('-p', type=int, default=0)
args = parser.parse_args()

datasets = ["Hall","Kitchenham","Radjenovic","Wahono"]

bad_abstract = '<div style="font-variant: small-caps; font-size: .9em;">First Page of the Article</div><img class="img-abs-container" style="width: 95%; border: 1px solid #808080;" src="/xploreAssets/images/absImages/01618458.png" border="0">'


dfs = []
cols = ["Document Title","Abstract", "label"]
for d in datasets:
    df = pd.read_csv(f'../data/fastread/{d}.csv', encoding="ISO-8859-1")[cols]
    df.loc[df['label']=="no", "relevant"] = 0
    df.loc[df['label']=="yes", "relevant"] = 1
    df.loc[df['Abstract']==bad_abstract, "Abstract"] = np.NaN
    df["review"] = d
    dfs.append(df)
    
frdf = pd.concat(dfs)

frdf['x'] = frdf['Abstract'].str.cat(frdf['Document Title'], sep=" ")

models = [
    SVC(kernel='linear',class_weight='balanced',probability=True)
]
iterations = args.iterations
results = []
for name, group in frdf.groupby('review'):
    df = group.dropna().reset_index(drop=True)
    for s in [200, 500]:
        ss = rr.ScreenScenario(
            df, models, s, [50, 100, 200], name
        )
        if args.p:
            from multiprocessing import Pool
            def simulate_screening_parallel(i,ss):
                return ss.screen(i, True)
            with Pool(args.p) as pool:
                results.append(pool.map(partial(simulate_screening_parallel, ss=ss), list(range(iterations))))
            print(results)
            break
        else:
            for i in range(iterations):
                print(i)
                r = ss.screen(i, True)
                if r is not None:
                    results.append(r)

results_df = pd.DataFrame.from_dict(results)
results_df.to_csv('../results/results_fr.csv', index=False)
