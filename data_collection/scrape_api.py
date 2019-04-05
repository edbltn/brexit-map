from time import gmtime, strftime, sleep
from difflib import ndiff
import requests as req
import pandas as pd
import itertools
import datetime
from progressbar import ProgressBar
import re
from newsapi import NewsApiClient
from guardianapi import GuardianApiClient
from bs4 import BeautifulSoup

def scrape_all_articles(sources, end_date, time_delta):

    all_documents_df = pd.DataFrame(columns=['source', 'headline', 'body', 'publication_date', 'section'])
    
    days = [end_date - datetime.timedelta(days=i) for i in reversed(range(time_delta))]
   
    queries = itertools.product(days,sources)
    bar = ProgressBar(max_value = len(days)*len(sources))
    i = 0
    for day, source in queries:
        documents = scrape_articles(source, day)
        for document in documents:
            all_documents_df = all_documents_df.append(document, ignore_index=True)
        i+=1
        bar.update(i)

    return all_documents_df

def scrape_articles(source, day):
    documents = []
    total = 20
    page = 1

    exclude_domains = ['hotair.com', 'www.ft.com', 'seekingalpha.com',
                        'www.businessinsider.com', 'www.investmentwatchblog.com',
                        'www.seattlepi.com', 'www.thisismoney.co.uk',
                        'www.mirror.co.uk', 'www.indpenedent.co.uk',
                        'www.investopedia.com', 'www.freemalaysiatoday.com',
                        'new.sky.com', 'www.fxstreet.com', 'www.wsj.com',
                        'www.cnbc.com', 'finance.yahoo.com', 'www.pbs.org'
                        'www.dailymail.co.uk', 'www.bbc.co.uk', 'foreingpolicy.com',
                        'www.nytimes.com']


    domains_dict = {'bbc-news': 'bbc.co.uk',
                    'the-telegraph': 'telegraph.co.uk',
                    'daily-mail': 'dailymail.co.uk'}

    while page * 20 <= total and page <= 5:
        if source == 'the-guardian-uk':
            response = guardianapi.get_everything(q='Brexit',
                                                  from_param=day.strftime('%Y-%m-%d'),
                                                  to=day.strftime('%Y-%m-%d'),
                                                  sort_by='relevance',
                                                  page=page)
            articles = response['response']['results']
            total = response['response']['total']
        else:
            response = newsapi.get_everything(q='Brexit',
                                              sources=source,
                                              from_param=day.strftime('%Y-%m-%d'),
                                              to=day.strftime('%Y-%m-%d'),
                                              language='en',
                                              sort_by='relevancy',
                                              page=page,
                                              domains=domains_dict[source])
            articles = response['articles']
            total = response['totalResults']

        documents.append(get_documents(articles, source))
        page += 1

    return documents

def get_documents(articles, source):
    documents = []
    for article in articles:
        if source == 'the-guardian-uk':
            url = article['webUrl']
            pub_date = article['webPublicationDate'] 
        else:
            url = article['url']
            pub_date = article['publishedAt']
        
        page = safe_get(url)
        text = re.sub(r'</?br>', ' ', page.text)
        text = re.sub(r'</?em>', ' ', text)
        text = re.sub(r'</?strong>', ' ', text)
        document = parse_html(text, source, url)
        document['article_url'] = url
        document['publication_date'] = datetime.datetime.strptime(pub_date, "%Y-%m-%dT%H:%M:%SZ")

        if len(document['body']) >=  10:
            documents.append(document)

        print(url)

    return documents

def safe_get(url):
    i = 0
    while True:
        # Rate limit
        sleep(0.5)
        try:
            page = req.get(url)
            return page
        except:
            i += 1
            if i % 10 == 0:
                print(f'{url} failed 10 times.')
                sleep(60)
                print('Trying again')

# TODO: eliminate common mentions of news source, URL
def parse_html(text, source, url):
    # TODO: Deal more graciously with single component articles
    soup = BeautifulSoup(text, features="html.parser")
    if source == 'bbc-news':
        result = parse_soup(soup, 'h1', ['story-body__inner', 'vxp-media__summary',
            'story-body', 'lx-stream-post-body'])
        section = get_bbc_section(url)
        result['body'] = result['body'].replace('BBC', '<ORG>')
    elif source == 'breitbart-news':
    # TODO: See how to deal with embedded tweets
        result = parse_soup(soup, 'h1', ['entry-content'])
    elif source == 'associated-press':
    # TODO: See how to clean ends of articles
        result = parse_soup(soup, 'h1', ['Article'])
    elif source == 'rt':
        result = parse_soup(soup, 'h1', ['article__text'])
    elif source == 'reuters':
        result = parse_soup(soup, 'h1', ['StandardArticleBody_body'])
    elif source == 'the-guardian-uk':
        result = parse_soup(soup, 'h1', ['content__article-body', 'dropdown__content', 
            'content__main', ['podcast', 'gs-container']],
                skip_names = ['Best of the rest', 'Top comment', 'Top tweet', 'Join the debate – email'], 
                recursive=True, tweet_name='Tweet')
        section = get_guardian_section(url)
        result['body'] = result['body'].replace('Guardian', '<ORG>')
        result['body'] = re.sub('\S*theguardian.com|\S*gu.com\S*', '<URL>', result['body'])
    elif source == 'the-telegraph':
    # TODO: Circumvent paywall
        result = parse_soup(soup, 'h1', ['article__content'], recursive=True)
        section = get_telegraph_section(url)
        result['body'] = result['body'].replace('Telegraph', '<ORG>')
    elif source == 'daily-mail':
        result = parse_soup(soup, 'h2',  ['articleBody'], recursive=False, class_name='itemprop')
        section = get_dailymail_section(url)
        result['body'] = result['body'].replace('Daily Mail', '<ORG>')
    else:
        print(source)
        raise
    result['section'] = section
    result['source'] = source

    result['body'] = re.sub(' +', ' ', result['body'])
    return result

def parse_soup(soup, 
        headline_name, 
        body_names,
        skip_names = [],
        recursive=False, 
        class_name='class', 
        tweet_name='tweet'):

    paragraphs = []
    h1 = soup.find_all(headline_name)[0]
    headline = h1.text
    for div in soup.find_all('div'):
        div_class = div.get(class_name)
        if not div_class:
            continue
        elif div_class[0] in body_names or div_class in body_names:
            for paragraph in div.find_all(recursive = recursive):
                if paragraph.text in skip_names or '@theguardian.com' in paragraph.text:
                    break

                # Define conditions for appending paragraph
                is_paragraph = paragraph.name == u'p'
                is_list = paragraph.name == u'ul'
                is_good_list = is_list and paragraph.get(class_name) == 'mol-bullets-with-font'
                is_bad_list = not paragraph.get(class_name) and is_list
                is_not_bad_list = not is_bad_list
                is_not_info = not paragraph.find_all('i')
                p_class = paragraph.get(class_name)
                if p_class:
                    is_not_tweet = p_class[0].startswith(tweet_name)
                else:
                    is_not_tweet = True
             
                # Combine conditions
                if (is_paragraph or is_good_list) and is_not_info and \
                        is_not_tweet and is_not_bad_list:
                    paragraphs.append(paragraph.text)
#                else:
#                    print(paragraph.find_all('i'))

            # BBC specific
            for paragraph in div.find_all('span', 'media-caption__text'):
                paragraphs.append(paragraph.text)
        else:
            continue
    
    document = '\n'.join(paragraphs)
    return {'headline': headline, 'body': document}

def get_bbc_section(url):
    m = re.match(r'https://www.bbc.co.uk/\w+/((live|av|sport)/)*((\w|-)+)[-/]\d+', url)
    if m:
        return m.group(3)
    else:
        return 'none'

def get_guardian_section(url):
    m = re.match(r'https://www.theguardian.com/(\w+)/.*', url)
    if m:
        return m.group(1)
    else:
        return 'none'

def get_telegraph_section(url):
    m = re.match(r'https://www.telegraph.co.uk/([^/]+)/.*', url)
    if m:
        return m.group(1)
    else:
        return 'none'

def get_dailymail_section(url):
    m = re.match(r'https://www.dailymail.co.uk/([^/]+)/.*', url)
    if m:
        return m.group(1)
    else:
        return 'none'

if __name__ == '__main__':
    newsapi = NewsApiClient(api_key='14c21d448d24414ca12577e0e4f39fe9')
    guardianapi = GuardianApiClient(api_key='bfab1f02-49da-4317-a6a4-5bb945841c98')

    sources = ['the-guardian-uk', 'bbc-news', 'the-telegraph', 'daily-mail']
    end_date = datetime.date.today() - datetime.timedelta(days=28)

    all_documents_df = scrape_all_articles(sources, end_date=end_date, time_delta=4)

    end_date_str = end_date.strftime('%Y%m%d')
    all_documents_df.to_csv(f'../data/{end_date_str}_brexit_articles.csv',
                            encoding='utf-8',
                            index=False)
