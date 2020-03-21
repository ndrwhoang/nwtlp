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
import re

page_list = []
j = 1

for i in range(0,67):
    a = "http://www.newtonproject.ox.ac.uk/texts/newtons-works/all?n=25&sr=" + str(j) + "&name=1&tr=1&sort=date&order=asc"
    page_list.append(a)
    j += 25
    
def newton_scraping(url="http://www.newtonproject.ox.ac.uk/texts/newtons-works/all"):
    r = requests.get(url)
    #print(r.content)
    
    soup = BeautifulSoup(r.content, 'html5lib')
    #print(soup.prettify())
    
    # Article features
    source = []
    full_text = []
    
    # Scraping
    text_div = soup.findAll("td", {"class": "record"})
    
    for i_text in text_div:
        hyper = i_text.find_all('a', href = re.compile('norm'))
        for j in hyper:
            source.append(j.get('href'))
            
    print(len(source))
    
    for i_source in source:
        print(i_source)
        url_subtext = str("http://www.newtonproject.ox.ac.uk" + i_source)
        try:    
            r = requests.get(url_subtext)
            soup = BeautifulSoup(r.content, "html5lib")
            paragraphs = soup.find('div', {'id' : 'tei'}).find_all('p')
            a = " ".join([paragraph.text for paragraph in paragraphs])
            full_text.append(a)
        except requests.exceptions.ConnectionError:   # Somce links go to another site
            url_subtext = str(i_source)
            r = requests.get(url_subtext)
            soup = BeautifulSoup(r.content, "html5lib")
            paragraphs = soup.find_all('div')
            a = " ".join([paragraph.text for paragraph in paragraphs])
            full_text.append(a)   
    # Create a dataframe
    print(len(full_text))
    page_df = pd.DataFrame({'source' : source,
                       "full_text" : full_text
                       })
    
    
    return page_df

def get_newton_text():
    df = pd.DataFrame()
    
    for i in page_list:
        page_df = newton_scraping(url = i)
        df.append(page_df)
        
    df['text_id'] = str(df.source).split("/")[-1]    
    
    return df
    
df = get_newton_text()

