import pandas as pd
from scrape_api import * 
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

def safe_get(driver, url):
    i = 0
    while True:
        # Rate limit
        sleep(0.5)
        try:
            driver.get(url)
            delay = 3 # seconds
            element_present = EC.presence_of_element_located((By.ID, 'main_content'))
            WebDriverWait(driver, timeout).until(element_present)
        except TimeoutException:
            print("Loading took too much time!")
        except:
            i += 1
            if i % 10 == 0:
                print(f'{url} failed 10 times.')
                sleep(60)
                print('Trying again')
        page = driver.page_source
        return page

user='edbltn@gmail.com'
driver = webdriver.Chrome('../chromedriver')
driver.get('https://secure.telegraph.co.uk/secure/login/')

raise
elem = driver.find_element_by_name('identifier')
elem.send_keys(user)
elem = driver.find_element_by_name('auth_key')
elem.send_keys(pwd)
elem.send_keys(Keys.RETURN)

columns = ['article_url', 'body', 'headline', 'publication_date', 'section', 'source']
all_articles_df = pd.DataFrame(columns = columns)

for week in ['0122', '0129', '0205', '0212', '0219', '0305', '0312', '0319', '0326', '0402']:
    articles_df = pd.read_csv(f'../data/2019{week}_brexit_articles.csv')

    for i, row in articles_df.iterrows():
        source = row['source']

        if source == 'the-telegraph':
            url = row['article_url']
            print(url)
            pub_date = row['publication_date']
        
            page = safe_get(driver, url)
            text = re.sub(r'</?br>', ' ', page)
            text = re.sub(r'</?em>', ' ', text)
            text = re.sub(r'</?strong>', ' ', text)
            document = parse_html(text, source, url)
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

all_articles_df.to_csv(f'../data/all_brexit_articles.csv')
