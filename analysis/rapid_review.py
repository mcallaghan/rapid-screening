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

class ScreenScenario:
    '''
    A class to record and advance the state of ml-aided document screening
    '''
    def __init__(self, df, models, samples):
        self.df = df
        self.models = models
        self.N = df.shape[0]
        self.r_docs = df[df["relevant"]==1].shape[0]
        self.p = self.r_docs/self.N
        self.bir = None
        self.bir_upperbound = None
        self.df['seen'] = 0
        self.seen_docs = 0
        self.r_seen = 0
        self.samples = samples
        self.results = []
        self.iteration_size = 20
        self.recall = 0
        
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

    def screen_bir(self):
        for s in self.samples:
            self.reset()
            ## Do the random sample
            if s > self.df.shape[0]*0.5:
                print(f"skipping sample {s}, as it is more than 50% of the data")
                break
            sids = random.sample(list(self.df.index), s)
            self.df.loc[sids,'seen'] = 1
            self.seen_docs = s
            self.r_seen = self.df.query('seen==1 & relevant==1').shape[0]
            self.bir = self.r_seen/self.seen_docs
            self.r_predicted = round(self.bir*self.N)

            ## Do some machine learning
            for clf in self.models:
                learning = True
                while learning:
                    index = self.df.query('seen==1').index
                    unseen_index = self.df.query('seen==0').index
                    if len(unseen_index) == 0:
                        learning=False
                        break
                    x = self.X[index]
                    y = self.df.query('seen==1')['relevant']
                    clf.fit(x,y)
                    y_pred = clf.predict_proba(self.X[unseen_index])[:,1]

                    # These are the next documents
                    next_index = unseen_index[(-y_pred).argsort()[:self.iteration_size]]
                    self.df.loc[next_index,'seen'] = 1
                    self.seen_docs = self.df.query('seen==1').shape[0]
                    self.r_seen = self.df.query('seen==1 & relevant==1').shape[0]
                    if self.r_seen > self.r_predicted:
                        learning = False
                        break
                self.wss95 = 1 - self.seen_docs / self.N
                self.recall = self.r_seen / self.r_docs
                
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
    
