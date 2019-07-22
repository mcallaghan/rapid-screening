import pandas as pd
import numpy as np
import rapid_review as rr
import os
import matplotlib.pyplot as plt
import importlib
from sklearn.svm import SVC, OneClassSVM

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
    names=["review","EID","PMID","relevant","fulltext_relevant"]
)

cohen_db['relevant'] = np.where(cohen_db['relevant']=="I",1,0)
cohen_db = cohen_db[["review","PMID","relevant"]]

cohen_db.head()

models = [
    SVC(kernel='linear',C=5,probability=True)
]
iterations = 10

results = []
rs_results = []
paths = []
for name, group in cohen_db.groupby('review'):
    df = pd.merge(
        group,
        document_index,
    )
    df = df.dropna().reset_index(drop=True)
    for s in [200, 500]:
        ss = rr.ScreenScenario(
            df, models, s, 50, name
        )
        for i in range(iterations):
            print(i)
            r = ss.screen(i, True) 
            if r is not None:
                rs_results.append(r)
            r = ss.screen(i)
            if r is not None:
                results.append(r)
            paths.append({
                "dataset": name,
                "work_path": ss.work_track,
                "recall_path": ss.recall_track,
                "estimated_recall_path": ss.estimated_recall_path
            })

paths = [x for x in paths if x is not None]
path_df = pd.DataFrame.from_dict(paths)
results_df = pd.DataFrame.from_dict(results)
rs_results_df = pd.DataFrame.from_dict(rs_results)
rs_results_df.to_csv('../results/rs_results.csv', index=False)
results_df.to_csv('../results/results.csv', index=False)
paths_df.to_csv('../results/paths.csv', index=False)
