import re
import string
import csv
from string import punctuation
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

os.getcwd()

# def textProcess(text_path, text_name, word_name):
def textProcess(text_path, text_name, word_name):
    ## open the scrape file
    output_data = pd.read_csv(text_path, delimiter='\t', header=None)
    output_data.columns = ["rating","review"]

    ## extract rates numbers  
    for i in range(len(output_data)):
        output_data.iloc[i]['rating'] = output_data.iloc[i]['rating'].replace('bubble_', '')

    # ## drop lower rates
    # output_data['rating'] = output_data['rating'].apply(lambda x: np.nan if x in ['10','20','30'] else x)
    # output_data = output_data.dropna()

    ## remove punctuation
    for i in range(len(output_data)):
        output_data.iloc[i]['review'] = ''.join(text for text in output_data.iloc[i]['review'] if text not in punctuation)

    ## convert text to lowercase
    for i in range(len(output_data)):
        output_data.iloc[i]['review'] = ' '.join([text.lower() for text in nltk.word_tokenize(output_data.iloc[i]['review'])])

    ## stop words removal
    stopword = stopwords.words('english')
    for i in range(len(output_data)):
        output_data.iloc[i]['review'] = ' '.join([word for word in nltk.word_tokenize(output_data.iloc[i]['review']) if word not in stopword])

    ## lemmatizer
    word_lemmatizer = WordNetLemmatizer()
    for i in range(len(output_data)):
        output_data.iloc[i]['review'] = ' '.join([word_lemmatizer.lemmatize(word) for word in nltk.word_tokenize(output_data.iloc[i]['review'])])

    """ create text file """
    with open(text_name, 'w') as f:
        for i in range(len(output_data)):
            f.write(output_data.iloc[i]['rating'] + '\t' + output_data.iloc[i]['review'] + '\n')

    """ count word and create word count file """
    word_count = {}
    for i in range(len(output_data)):
        for word in nltk.word_tokenize(output_data.iloc[i]['review']):
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1

    with open(word_name,'w') as f:
        fieldnames = ['word', 'frequency']
        word = csv.DictWriter(f,fieldnames=fieldnames)
        word.writeheader()
        for w in sorted(word_count, key=word_count.get, reverse=True):
            word.writerow({'word': w, 'frequency': word_count[w]})

if __name__ == '__main__':
    textProcess('merge.txt', 'data.txt', 'word_count.csv')
    pass

