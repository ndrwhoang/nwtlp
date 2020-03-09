# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 14:29:09 2020

Scraping text from The Newton Project
Main repository: http://www.newtonproject.ox.ac.uk/texts/newtons-works/all

@author: hoang
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd

def newton_scraping():
    url = "http://www.newtonproject.ox.ac.uk/texts/newtons-works/all"
    r = requests.get(url)
    #print(r.content)
    
    soup = BeautifulSoup(r.content, 'html5lib')
    #print(soup.prettify())
    
    # Article features
    text_id = []
    title = []
    metadata = []
    source = []
    full_text = []
    
    # Scraping
    text_div = soup.findAll("td", {"class": "record"})
    
    for i_text in text_div:
        text_id.append(i_text.findAll("p", {"class": "metadataContent"})[1].text)
        title.append(i_text.find("p", {"class": "title"}).text)
        metadata.append(i_text.find("p", {"class": "metadata"}).text)
        source.append(i_text.find("a")["href"])
    
    print(title)
    '''
    for i_source in source:
        url_subtext = str("http://www.newtonproject.ox.ac.uk" + i_source)
        try:    
            r = requests.get(url_subtext)
            soup = BeautifulSoup(r.content, "html5lib")
            for j_soup in soup.findAll("div", {"id": "tei"}):
                full_text.append(j_soup.get_text(strip=True))
        except requests.exceptions.ConnectionError:   # Somce links go to another site
            url_subtext = str(i_source)
            r = requests.get(url_subtext)
            soup = BeautifulSoup(r.content, "html5lib")
            for j_soup in soup.findAll("div", {"id": "tei"}):
                full_text.append(j_soup.get_text(strip=True))
    
    
    df = pd.DataFrame(list(zip(text_id, title, metadata, source, full_text)), columns=["id", "title", "metadata", "source", "full_text"])
    
    
    return df
    '''
    
newton_scraping()

