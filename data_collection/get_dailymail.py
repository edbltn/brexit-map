import pandas as pd
from scrape_api import * 
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import requests

def safe_get(url):
    i = 0
    while True:
        # Rate limit
        sleep(0.5)
        try:
            page = requests.get(url)
            return page
        except:
            i += 1
            if i % 10 == 0:
                print(f'{url} failed 10 times.')
                sleep(60)
                print('Trying again')

columns = ['article_url', 'body', 'headline', 'publication_date', 'section', 'source']
all_articles_df = pd.DataFrame(columns = columns)

articles_df = pd.read_csv(f'../data/all_brexit_articles.csv')

for i, row in articles_df.iterrows():
    source = row['source']

    if source == 'daily-mail':
        url = row['article_url']
        print(url)
        pub_date = row['publication_date']
    
        page = safe_get(url)
        text = re.sub(r'</?br>', ' ', page.text)
        text = re.sub(r'</?em>', ' ', text)
        text = re.sub(r'</?strong>', ' ', text)
        document = parse_html(text, source, url)
        document['body'] = row['body']
        document['article_url'] = url
        document['publication_date'] = row['publication_date']
    else:
        document = {}
        document['article_url'] = row['article_url']
        document['body'] = row['body']
        document['headline'] = row['headline']
        document['publication_date'] = row['publication_date']
        document['section'] = row['section']
        document['source'] = row['source']

    if len(document['body']) >=  10:
        all_articles_df.loc[len(all_articles_df)] = document

all_articles_df.to_csv(f'../data/all_brexit_articles_new.csv')
