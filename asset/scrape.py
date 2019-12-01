import re
import requests
from bs4 import BeautifulSoup
import time
import os
import pandas as pd

os.getcwd()
# create function to scrape data
# 2 arguments, hotel names and page number
def getData(path, num,page):
    # create a empty list
    data = []
    # access the webpage as Chrome
    my_headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36'}

    for k in range(0, page+5, 5):
        # give the url of the page
        if k == 0 :
            url = 'https://www.tripadvisor.com/Hotel_Review-'+ num +'-Reviews-'+ path +'-New_York_City_New_York.html'
        else: 
            url = 'https://www.tripadvisor.com/Hotel_Review-' + num + '-Reviews-or'+str(k)+'-'+ path +'-New_York_City_New_York.html#REVIEWS'
        # initialize src False
        src = False

        # try to scrape 5 times
        for i in range(1,6):
            try:
                # get url content
                response = requests.get(url, headers = my_headers)
                # get html content
                src = response.content
                break
            except:
                print('failed attempt #', i)
                # wait 2 secs before next time
                time.sleep(2)

        # if we could not get the page
        if not src:
            print('Could not get page', url)
        else:
            print('Successfully get the page', url)
        
        soup = BeautifulSoup(src.decode('ascii', 'ignore'),'lxml')

        # find div with class hotels-community-tab-common-Card__card--ihfZB hotels-community-tab-common-Card__section--4r93H
        divs = soup.findAll('div', {'class':re.compile('hotels-community-tab-common-Card__card--ihfZB hotels-community-tab-common-Card__section--4r93H')})
        
        for div in divs:
            # find div with class hotels-review-list-parts-SingleReview__mainCol--2XgHm
            reviews = div.find('div', {'class':re.compile('hotels-review-list-parts-SingleReview__mainCol--2XgHm')})
            # find div with class
            infos = div.find('div', {'class':re.compile('social-member-event-MemberEventOnObjectBlock__member_event_block--1Kusx')})

            # 
            if infos:
                contributions = infos.find('span', {'class':re.compile('social-member-MemberHeaderStats__stat_item--34E1r')})
                span = contributions.find('span', {'class':re.compile('social-member-MemberHeaderStats__bold--3z3qh')})
                contribution_text = span.text.strip()
            else:
                print('Cannot find contribution and helpful votes')
            
            if reviews:
                q = reviews.find('q', {'class':re.compile('hotels-review-list-parts-ExpandableReview__reviewText--3oMkH')})
                review = q.find('span')
                review_text = review.text.strip()

                rates = reviews.find('span', {'class':re.compile('ui_bubble_rating')})
                rate = rates.attrs['class'][1]
            else: 
                print('Cannot find rates and reviews')

            data.append([contribution_text, rate, review_text])

    with open(path+'.txt', 'w', encoding='utf-8') as f:
        for text in data:
            f.write(text[0] + '\t' + text[1] + '\t' + text[2] +'\n')

path = 'Park_Lane_Hotel'
num = 'g60763-d93579'              
if __name__ == "__main__":
    getData(path=path, num=num,page=100)
    pass











