import requests
import time
from itertools import zip_longest
def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

pmids = []
with open('data/epc-ir.clean.tsv','r') as f:
    for l in f:
        pmids.append(l.split('\t')[2])

pmids = set(pmids)

for i, group in enumerate(grouper(200, pmids)):
    group = [x for x in group if x is not None]
    if os.path.exists(f'data/cohen_all_{i}.xml'):
        continue
    with open(f'data/cohen_all_{i}.xml','w') as f:
        pm_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id='
        pm_url = pm_url+",".join(group)
        result = requests.post(pm_url)
        f.write(str(result.content, "utf-8"))
        time.sleep(5)
