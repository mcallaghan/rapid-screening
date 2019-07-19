#!/usr/bin/python3

import requests
from bs4 import BeautifulSoup


def parse_ncbi(url):
    result = requests.get(url)
    c = result.content
    pmids = []
    for row in soup.find('table').find_all('tr'):
        if not row.find_all('td'):
            continue
        if row.find('td').attrs['colspan']=='2':
            continue
        if row.find_all('td')[1].string:
            pmids.append(row.find_all('td')[1].string)

    pm_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id='
    pm_url = pm_url+",".join(pmids)

    result = requests.post(pm_url)
    return result

result = parse_ncbi('https://www.ncbi.nlm.nih.gov/books/NBK44537/')
with open('data/pbr_included.xml','w') as f:
    f.write(str(result.content, "utf-8"))

result = parse_ncbi('https://www.ncbi.nlm.nih.gov/books/NBK44542/')
with open('data/pbr_excluded.xml','w') as f:
    f.write(str(result.content, "utf-8"))

