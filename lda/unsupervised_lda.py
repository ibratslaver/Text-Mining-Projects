import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pip._vendor.distlib.compat import raw_input
import gensim
import nltk
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pyLDAvis.gensim
import re
from Database_Utility import *

data = pd.read_excel('rmp_top_1000_reviews.xlsx')
data = data[['review_id', 'professor_id', 'review_text']]
# convert review text to lowercase
data['review_text'] = data.review_text.astype(str).str.lower()
   
# create a new column that creates sentence tokens 
data['tokenized_text'] = data['review_text'].apply(sent_tokenize)
# data.to_csv('word_token_test_sent.csv')
review_text = data['tokenized_text'].values.tolist()
    
stop_words = stopwords.words('english')  
  
big_list = []  
  
for ind in data.index:
    filtered_list = []
    review = data['review_text'][ind].lower()
    word_token = word_tokenize(review)
    for w in word_token:
        if w not in stop_words:
            filtered_list.append(w)
    big_list.append(filtered_list)
  
# Tokenization and Stopword removal       
for review in review_text:
    filtered_list = []
    for sentence in review:
        p=""
        for w in sentence.split():
            if w not in stop_words:
                filtered_list.append(w)
    big_list.append(filtered_list)


dictionary = corpora.Dictionary(big_list)  
corpus = [dictionary.doc2bow(sentence) for sentence in big_list]          
    
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5)

for i in range(0, lda.num_topics):
    print(lda.print_topic(i))
 