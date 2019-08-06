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
import copy
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPClassifier
from scipy.stats import hypergeom

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
        self.iteration_size = 20
        self.iterations = 0
        self.recall_track = []
        self.work_track = []
        self.ratings = []
        self.random_recall_track = []
        self.random_work_track = []
        self.estimated_recall_path = []
        self.recall = None
        self.recall_pf = None
        self.recall_bir = None
        self.recall_bir_ci = None
        self.wss95_bir = None # Work saved at 95\% recall
        self.wss95_bir_ci = None
        self.wss95_pf = None # Work saved at 95\% recall with perfect knowledge
        self.unseen_p = None
        
        self.X = TfidfVectorizer(
            ngram_range=(1,1),
            min_df=2, max_df=0.9, strip_accents='unicode', 
            max_features=10000,
            use_idf=1,
            smooth_idf=1, sublinear_tf=1,
            #stop_words="english",tokenizer=tokenize
        ).fit_transform(df['x'])

        clf = IsolationForest()
        clf.fit(self.X)
        self.df['outlying'] = clf.predict(self.X)

    def reset(self):
        self.df['seen'] = 0
        self.seen_docs = 0
        self.ratings = []
        self.recall_track = []
        self.work_track = []
        self.random_recall_track = []
        self.random_work_track = []
        self.estimated_recall_path = []
        self.estimated_p_path = []
        self.hyper_hypo_path = []
        self.prob_recall_path = []
        self.max_min_recall_path= []
        self.wss95_bir = None
        self.wss95_bir_ci = None
        self.wss95_pf = None
        self.wss95_ih = None
        self.wss95_rs = None
        self.wss95_nrs = None
        self.wss95_hyper = None
        self.max_prob_recall = 0
        for ih in self.irrelevant_heuristic:
            setattr(self, f'wss95_ih_{ih}', None)

    def screen(self, i, rs=False, nrs=True):
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

        outliers = False
        if outliers:
            self.df['seen'] = 0
            sids = self.df.sort_values('outlying').index[:s]
            self.df.loc[sids,'seen'] = 1
            self.seen_docs = s
            self.r_seen = self.df.query('seen==1 & relevant==1').shape[0]

        # Do some machine learning
        for clf in self.models:
            learning = True
            while learning:
                self.iterations +=1
                clear_output(wait=True)
                print(f"Dataset: {self.dataset}, iteration {self.iteration}.  {self.seen_docs} out of {self.N} documents seen ({self.seen_docs/self.N:.0%}) - recall: {self.get_recall():.2%}, probable recall: {self.max_prob_recall:.2%}")
                index = self.df.query('seen==1').index
                unseen_index = self.df.query('seen==0').index
                if len(unseen_index) == 0:
                    learning=False
                    break
                x = self.X[index]
                y = self.df.loc[index,'relevant']
                y = self.df.query('seen==1')['relevant']
                n_last = self.iteration_size*4
                if n_last > 0.05*self.N:
                    n_last = int(round(0.05*self.N))
                last_iteration = self.ratings[-n_last:]
                last_iteration_relevance = np.sum(last_iteration)/len(last_iteration)
                if len(set(y))<2: # if we have a single class - just keep sampling
                    y_pred = np.array([random.random() for x in unseen_index])
                    next_index = unseen_index[(-y_pred).argsort()[:self.iteration_size]]
                else:
                    clf.fit(x,y)
                    y_pred = clf.predict_proba(self.X[unseen_index])[:,1]
                    if rs and self.iterations > 2 and self.random_work_track == []:
                        if (max(y_pred) < 0.2 and self.max_prob_recall > 0.95) or self.seen_docs/self.N > 0.9:
                            self.last_iteration_relevance=last_iteration_relevance
                            tdf = copy.deepcopy(self.df)
                            r = self.sample_threshold()
                            
                            self.df = tdf
                    # These are the next documents
                    next_index = unseen_index[(-y_pred).argsort()[:self.iteration_size]]

                for i in next_index:
                    self.df.loc[i,'seen'] = 1
                    self.ratings.append(self.df.loc[i,'relevant'])
                    self.seen_docs = self.df.query('seen==1').shape[0]
                    self.r_seen = self.df.query('seen==1 & relevant==1').shape[0]
                    
                    self.recall_track.append(self.get_recall())
                    self.work_track.append(self.seen_docs / self.N)
                    for ih in self.irrelevant_heuristic:
                        last_ratings = self.ratings[-ih:]
                        if len(last_ratings) == ih and np.sum(last_ratings)==0:
                            if getattr(self, f'wss95_ih_{ih}') is None:
                                setattr(self, f'wss95_ih_{ih}', 1 - self.seen_docs / self.N)
                                setattr(self, f'recall_ih_{ih}', self.get_recall())
                    if self.r_seen > self.r_predicted_upperbound and self.wss95_bir_ci is None:
                        self.wss95_bir_ci = 1 - self.seen_docs / self.N
                        self.recall_bir_ci = self.get_recall()
                    if self.get_recall() > 0.95 and self.wss95_pf is None:
                        self.wss95_pf = 1 - self.seen_docs / self.N
                        self.recall_pf = self.get_recall()
                    if self.r_seen > self.r_predicted and self.wss95_bir is None:
                        self.wss95_bir = 1 - self.seen_docs / self.N
                        self.recall_bir = self.get_recall()
                    X = 0
                    max_min_recall = 0
                    self.max_prob_recall = 0
                    if self.wss95_nrs is None and nrs is True:
                        for n, j in enumerate(self.ratings[::-1]):
                            X+=j
                            if n < 20:
                                continue
                            p_tilde, ci = ci_ac(X, n+1, 0.95)
                            estimated_r_docs = (p_tilde+ci)*(self.N-self.seen_docs) + self.r_seen
                            estimated_recall_min = self.r_seen / estimated_r_docs
                            if estimated_recall_min > max_min_recall:
                                max_min_recall = estimated_recall_min
                            p_tilde, ci = ci_ac(X, n+1, 0.33)
                            prob_r_docs = (p_tilde+ci)*(self.N-self.seen_docs) + self.r_seen
                            prob_recall_min = self.r_seen / prob_r_docs
                            if prob_recall_min > self.max_prob_recall:
                                self.max_prob_recall = prob_recall_min                            
                            if n > 100 and estimated_recall_min < 0.5:
                                break
                            if n > 500 and estimated_recall_min < 0.8:
                                break
                            if n > 1000 and estimated_recall_min < 0.9:
                                break
                        
                        if max_min_recall > 0.95 and self.wss95_nrs is None:
                            self.wss95_nrs = 1 - self.seen_docs / self.N
                            self.recall_nrs = self.get_recall()
                        self.prob_recall_path.append(self.max_prob_recall)
                        self.max_min_recall_path.append(max_min_recall)
                    
            if self.wss95_nrs is None:
                self.wss95_nrs = 0
                self.recall_nrs = 1
            if self.wss95_bir is None:
                self.wss95_bir = 0
                self.recall_bir = 1
            if self.wss95_bir_ci is None:
                self.wss95_bir_ci = 0
                self.recall_bir_ci = 1
            
            for ih in self.irrelevant_heuristic:
                if getattr(self, f'wss95_ih_{ih}') is None:
                    setattr(self, f'wss95_ih_{ih}', 0)
                    setattr(self, f'recall_ih_{ih}', 1)

        ignore_fields = ["df", "X", "unseen_p"]
                    
        result = {k: v for k, v in self.__dict__.items() if k not in ignore_fields}
        return result

    def get_recall(self):
        return self.r_seen / self.r_docs
    
    def sample_threshold(self):
        unseen_index = list(self.df.query('seen==0').index)
        random.shuffle(unseen_index)
        unseen_index = random.sample(unseen_index, len(unseen_index))
        
        self.seen_docs = self.df.query('seen==1').shape[0]
        self.r_seen = self.df.query('seen==1 & relevant==1').shape[0]
        
        self.random_start_work = self.seen_docs / self.N
        self.n_remaining_pre_sample = self.N - self.seen_docs
        self.r_seen_pre_sample = self.r_seen
        self.random_start_recall = self.get_recall()
        self.estimated_recall_path = []
        self.hyper_linterval_path = []
        self.hyper_uinterval_path = []
        self.X_sample_path = []
        X = 0

        print(f"Dataset: {self.dataset}, iteration {self.iteration}.  {self.seen_docs} out of {self.N} documents seen ({self.seen_docs/self.N:.0%}) - recall: {self.get_recall():.0%} - switching to random sampling")
        
        for j, i in enumerate(unseen_index):
            self.df.loc[i,'seen'] = 1
            #self.ratings.append(self.df.loc[i,'relevant'])
            X += self.df.loc[i, 'relevant']
            self.X_sample = X
            self.n_sample = j+1
            self.seen_docs = self.df.query('seen==1').shape[0]
            self.r_seen = self.df.query('seen==1 & relevant==1').shape[0]
            
            self.random_recall_track.append(self.get_recall())
            self.random_work_track.append(self.seen_docs / self.N)            

            p_tilde, ci = ci_ac(X, j+1, 0.95)
            self.n_remaining = self.N - self.seen_docs
            self.estimated_r_docs = (p_tilde+ci)*self.n_remaining + self.r_seen
            self.estimated_p_ub = self.estimated_r_docs / self.N            
            self.estimated_recall_min = self.r_seen / self.estimated_r_docs

            self.estimated_p_path.append(p_tilde+ci)
            self.estimated_recall_path.append(self.estimated_recall_min)

            self.hypothetical_95 = math.ceil(self.r_seen / 0.95 + 0.01) - self.r_seen + X
            self.hyper_prob_pmf = hypergeom.pmf(X, j+1+self.n_remaining, self.hypothetical_95, j+1)
            self.hyper_prob = hypergeom.cdf(X, j+1+self.n_remaining, self.hypothetical_95, j+1)
            self.hyper_interval = hypergeom.interval(0.95, j+1+self.n_remaining, self.hypothetical_95, j+1)

            self.hyper_uinterval_path.append(self.hyper_interval[1])
            self.hyper_linterval_path.append(self.hyper_interval[0])
            self.X_sample_path.append(self.X_sample)
            
            #self.hyper_prob_diff =  hypergeom.pmf(X, j+1+self.n_remaining, self.hypothetical_95+1, j+1)
            #self.hyper_hypo_path.append(self.hyper_prob)
            #print(self.estimated_missed, self.estimated_r_docs)
            #self.estimated_recall_min = 1 - (p_tilde+ci)/self.estimated_p_ub * self.n_remaining / self.N
            #if self.hyper_prob < 0.05 and self.wss95_hyper is None:# and self.hyper_prob_diff < self.hyper_prob:
 

            if X < self.hyper_interval[0] and self.wss95_hyper is None:
                self.recall_hyper = self.get_recall()
                self.wss95_hyper = 1 - self.seen_docs / self.N
            
            if self.get_recall() > 0.95 and self.wss95_pf is None:
                self.recall_pf = self.get_recall()
                self.wss95_pf = 1 - self.seen_docs / self.N
            
            if self.estimated_recall_min > 0.95 and self.wss95_rs is None:
                self.wss95_rs = 1 - self.seen_docs / self.N
                self.recall_rs = self.get_recall()
                
                

        return  
            
    def __str__(self):
        return f"a screening scenario with {self.N} documents"


def get_field(a, f):
    try:
        f = list(a.iter(f))[0]
        if f and f.text is None:
            try:
                return list(f.iter('style'))[0].text
            except:
                pass
        return f.text
    except:
        return None

def get_mesh_headings(a):
    ms = []
    for m in a.iter('MeshHeading'):
        d = m.find('DescriptorName')
        if d is not None:
            ms.append(f"MESHHEAD{d.attrib['UI']}")
    return " "+ " ".join(ms)
                      
def parse_pmxml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    docs = []
    for a in root.iter('PubmedArticle'):
        docs.append({
            "ab": get_field(a, 'AbstractText'),
            "PMID": get_field(a, 'PMID'),
            "ti": get_field(a, 'ArticleTitle'),
            "mesh": get_mesh_headings(a)
        })
    df = pd.DataFrame.from_dict(docs)
    df['PMID'] = pd.to_numeric(df['PMID'])
    return df

def parse_pb_xml(path,):
    tree = ET.parse(path)
    root = tree.getroot()
    docs = []
    for a in root.iter('record'):
        docs.append({
            "ab": get_field(a, 'abstract'),
            "rec-number": get_field(a, 'rec-number'),
            "ti": get_field(a, 'title')
        })
    df = pd.DataFrame.from_dict(docs)
    return df
