import pandas as pd
import xml.etree.ElementTree as ET
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords as sw
import string
punct = set(string.punctuation)
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
stopwords = set(sw.words('english'))
import random
import scipy.stats as st
import numpy as np
import time, sys
from IPython.display import clear_output
import math

def lemmatize(token, tag):
    tag = {
        'N': wn.NOUN,
        'V': wn.VERB,
        'R': wn.ADV,
        'J': wn.ADJ
    }.get(tag[0], wn.NOUN)
    return WordNetLemmatizer().lemmatize(token, tag)

def tokenize(X):
    for sent in sent_tokenize(X):
        for token, tag in pos_tag(wordpunct_tokenize(sent)):
            token = token.lower().strip()
            if token in stopwords:
                continue
            if all(char in punct for char in token):
                continue
            if len(token) < 3:
                continue
            if all(char in string.digits for char in token):
                continue
            lemma = lemmatize(token,tag)
            yield lemma

def ci_ac(X, n, a):
    def get_k(a):
        a = (1-(1-a)/2)
        return st.norm.ppf(a)

    k = get_k(a)

    X_tilde = X + k**2/2
    n_tilde = n + k**2
    p_tilde = X_tilde / n_tilde
    q_tilde = 1-p_tilde

    ci = k* np.sqrt(p_tilde*q_tilde/n_tilde) 

    return p_tilde, ci

class ScreenScenario:
    '''
    A class to record and advance the state of ml-aided document screening
    '''
    def __init__(self, df, models, sample, irrelevant_heuristic, dataset):
        self.df = df
        self.dataset = dataset
        self.models = models
        self.N = df.shape[0]
        self.r_docs = df[df["relevant"]==1].shape[0]
        self.p = self.r_docs/self.N
        self.bir = None
        self.bir_upperbound = None
        self.df['seen'] = 0
        self.seen_docs = 0
        self.r_seen = 0
        self.r_predicted = None
        self.r_predicted_upperbound = None
        self.s = sample
        self.irrelevant_heuristic = irrelevant_heuristic # how many irrelevant do we have to see in a row in order to stop
        self.results = []
        self.iteration_size = 50
        self.recall_track = []
        self.work_track = []
        self.ratings = []
        self.recall = None
        self.recall_pf = None
        self.recall_bir = None
        self.recall_bir_ci = None
        self.wss95_bir = None # Work saved at 95\% recall
        self.wss95_bir_ci = None
        self.wss95_pf = None # Work saved at 95\% recall with perfect knowledge
        self.wss95_ih = None
        self.recall_ih = None
        self.unseen_p = None
        
        self.X = TfidfVectorizer(
            ngram_range=(1,2),
            min_df=5, max_df=0.6, strip_accents='unicode', 
            max_features=10000,
            use_idf=1,
            smooth_idf=1, sublinear_tf=1,
            stop_words="english",tokenizer=tokenize
        ).fit_transform(df['ab'])

    def reset(self):
        self.df['seen'] = 0
        self.seen_docs = 0
        self.ratings = []
        self.recall_track = []
        self.work_track = []
        self.wss95_bir = None
        self.wss95_bir_ci = None
        self.wss95_pf = None
        self.wss95_ih = None
        self.wss95_rs = None

    def screen(self, i, rs=False):
        s = self.s
        self.iteration = i
        self.reset()
        ## Do the random sample
        if s > self.df.shape[0]*0.5:
            print(f"skipping sample {s}, as it is more than 50% of the data")
            return
        sids = random.sample(list(self.df.index), s)
        self.df.loc[sids,'seen'] = 1
        self.seen_docs = s
        self.r_seen = self.df.query('seen==1 & relevant==1').shape[0]
        self.bir = self.r_seen/self.seen_docs
        self.r_predicted = round(self.bir*self.N)
        bir, ci = ci_ac(self.r_seen, self.seen_docs, 0.95)
        self.bir_upperbound = bir + ci
        self.r_predicted_upperbound = round(self.bir_upperbound*self.N)

        # Do some machine learning
        for clf in self.models:
            learning = True
            while learning:
                clear_output(wait=True)
                print(f"Dataset: {self.dataset}, iteration {self.iteration}.  {self.seen_docs} out of {self.N} documents seen")
                index = self.df.query('seen==1').index
                unseen_index = self.df.query('seen==0').index
                if len(unseen_index) == 0:
                    learning=False
                    break
                x = self.X[index]
                y = self.df.query('seen==1')['relevant']
                if len(set(y))<2:
                    y_pred = [random.random() for x in unseen_index]
                else:
                    clf.fit(x,y)
                    y_pred = clf.predict_proba(self.X[unseen_index])[:,1]
                    if rs:
                        if max(y_pred) < 0.5:
                            r = self.sample_threshold()
                            return r
                # These are the next documents
                next_index = unseen_index[(-y_pred).argsort()[:self.iteration_size]]
                for i in next_index:
                    self.df.loc[i,'seen'] = 1
                    self.ratings.append(self.df.loc[i,'relevant'])
                    self.seen_docs = self.df.query('seen==1').shape[0]
                    self.r_seen = self.df.query('seen==1 & relevant==1').shape[0]
                    self.recall_track.append(self.r_seen / self.r_docs)
                    self.work_track.append(self.seen_docs / self.N)
                    last_ratings = self.ratings[-self.irrelevant_heuristic:]
                    self.recall = self.r_seen / self.r_docs
                    if len(last_ratings) == self.irrelevant_heuristic and np.sum(last_ratings)==0:
                        if self.wss95_ih is None:
                            self.wss95_ih = 1 - self.seen_docs / self.N
                            self.recall_ih = self.r_seen / self.r_docs
                    if self.r_seen > self.r_predicted_upperbound and self.wss95_bir_ci is None:
                        self.wss95_bir_ci = 1 - self.seen_docs / self.N
                        self.recall_bir_ci = self.r_seen / self.r_docs
                    if self.r_seen > self.r_docs*0.95 and self.wss95_pf is None:
                        self.wss95_pf = 1 - self.seen_docs / self.N
                        self.recall_pf = self.r_seen / self.r_docs
                    if self.r_seen > self.r_predicted and self.wss95_bir is None:
                        self.wss95_bir = 1 - self.seen_docs / self.N
                        self.recall_bir = self.r_seen / self.r_docs
            if self.wss95_bir is None:
                self.wss95_bir = 0
                self.recall_bir = 1
            if self.wss95_bir_ci is None:
                self.wss95_bir_ci = 0
                self.recall_bir_ci = 1
            if self.wss95_ih is None:
                self.wss95_ih = 0
                self.recall_ih = 1

        result = {
            "dataset": self.dataset,
            "s": self.s,
            "iteration": self.iteration,
            "wss95_bir_ci": self.wss95_bir_ci,
            "recall_bir_ci": self.recall_bir_ci,
            "wss95_pf": self.wss95_pf,
            "recall_pf": self.recall,
            "wss95_bir": self.wss95_bir,
            "recall_bir": self.recall_bir,
            "wss95_ih": self.wss95_ih,
            "recall_ih": self.recall_ih
        }
        return result

    def sample_threshold(self):
        unseen_index = list(self.df.query('seen==0').index)
        random.shuffle(unseen_index)
        self.random_start_work = self.seen_docs / self.N
        self.random_start_recall = self.r_seen / self.r_docs
        X = 0
        for j, i in enumerate(unseen_index):
            self.df.loc[i,'seen'] = 1
            self.ratings.append(self.df.loc[i,'relevant'])
            X += self.df.loc[i, 'relevant']
            self.seen_docs = self.df.query('seen==1').shape[0]
            self.r_seen = self.df.query('seen==1 & relevant==1').shape[0]
            self.recall_track.append(self.r_seen / self.r_docs)
            self.work_track.append(self.seen_docs / self.N)            

            p_tilde, ci = ci_ac(X, j+1, 0.95)
            self.n_remaining = self.N - self.seen_docs
            self.estimated_r_docs = math.floor((p_tilde+ci)*self.n_remaining) + self.r_seen
            self.estimated_p_ub = self.estimated_r_docs / self.N
            self.estimated_missed = round((p_tilde+ci)*self.n_remaining)
            self.estimated_recall_min = (self.estimated_r_docs - self.estimated_missed) / self.estimated_r_docs

            #print(self.estimated_missed, self.estimated_r_docs)
            #self.estimated_recall_min = 1 - (p_tilde+ci)/self.estimated_p_ub * self.n_remaining / self.N
            self.recall = self.r_seen / self.r_docs
            if self.recall > 0.95 and self.wss95_pf is None:
                self.recall_pf = self.recall
                self.wss95_pf = 1 - self.seen_docs / self.N
            
            if self.estimated_recall_min > 0.95 and self.wss95_rs is None:
                self.wss95_rs = 1 - self.seen_docs / self.N
                self.recall_rs = self.r_seen / self.r_docs


        result = {
            "dataset": self.dataset,
            "s": self.s,
            "iteration": self.iteration,
            "wss95_rs": self.wss95_rs,
            "recall_rs": self.recall_rs,
            "wss95_pf": self.wss95_pf,
            "recall_pf": self.recall_pf,
            "random_start_work": self.random_start_work,
            "random_start_recall": self.random_start_recall
        }
        return result 
            
    def __str__(self):
        return f"a screening scenario with {self.N} documents"


def get_field(a, f):
    try:
        return list(a.iter(f))[0].text
    except:
        return None
    
def parse_pmxml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    docs = []
    for a in root.iter('PubmedArticle'):
        docs.append({
            "ab": get_field(a, 'AbstractText'),
            "PMID": get_field(a, 'PMID'),
            "ti": get_field(a, 'ArticleTitle')
        })
    df = pd.DataFrame.from_dict(docs)
    df['PMID'] = pd.to_numeric(df['PMID'])
    return df
    
