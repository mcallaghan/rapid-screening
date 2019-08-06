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

##########################################
## Pull the document metadata from the xml files from the Pubmed API
document_index = None

for fpath in os.listdir('../data/'):
    if "cohen_all" in fpath:
        ndf = rr.parse_pmxml(f'../data/{fpath}')
        if document_index is None:
            document_index = ndf
        else:
            document_index = pd.concat([document_index,ndf])
        
document_index = document_index.drop_duplicates()
print(document_index.shape)
document_index.head()

#############################
## Load the cohen database of SRs
#https://dmice.ohsu.edu/cohenaa/systematic-drug-class-review-data.html

cohen_db = pd.read_csv(
    '../data/epc-ir.clean.tsv',
    sep='\t',header=None,
    names=["review","EID","PMID","ab_relevant","fulltext_relevant"]
)

cohen_db['relevant'] = np.where(cohen_db['ab_relevant']=="I",1,0)
cohen_db = cohen_db[["review","PMID","relevant"]]

cohen_db.head()

models = [
    SVC(kernel='linear',class_weight='balanced',probability=True)
]
iterations = args.iterations

results = []
rs_results = []
paths = []
for name, group in cohen_db.groupby('review'):
    df = pd.merge(
        group,
        document_index,
    )
    if df.shape[0] > 1000000:
        continue
    df['x'] = df['mesh']
    df = df[df['x'].str.contains('\w')]
    df = df.dropna().reset_index(drop=True)
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
results_df.to_csv('../results/results.csv', index=False)
