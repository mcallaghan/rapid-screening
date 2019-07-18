import pandas as pd
import xml.etree.ElementTree as ET

class Screen:
    '''
    A class to record and advance the state of ml-aided document screening
    '''
    def __init__(self, N):
        self.N = N


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
            "pmid": get_field(a, 'PMID'),
            "ti": get_field(a, 'ArticleTitle')
        })
    df = pd.DataFrame.from_dict(docs)
    return df
    
