from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
#from Database_Utility import *
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 


stopwords = stopwords.words('english')
newStopWordsClass = ['good','great','intro','free','basic','short','understanding','Great']
stopwords.extend(newStopWordsClass)

df = pd.read_excel('GA Feedback Categorized.xlsx', sheet_name = "Class")

df['Tokenized_Feedback'] = df['feedback'].apply(word_tokenize)


# #VERSION 1 CLASS : Only Manual Filtering with the stopwords

sentence_text = []

for ind in df.index:
    sentence = df['Tokenized_Feedback'][ind]
    sentence = [word for word in sentence if word not in ('!','','?',', ', '.',',','(','--',')')]
    sentence = [word for word in sentence if word not in stopwords]
    sentence = ' '.join(word for word in sentence)
    sentence_text.append(sentence)

combined = ' '.join(sentence_text)
#print(combined)

cloud = WordCloud(background_color = "white").generate(combined)
plt.imshow(cloud)
plt.show()


#VERSION 2: Only Manual Filtering with the stopwords

# df['Tokenized_Feedback'] = df['feedback'].str.lower().apply(word_tokenize)

# sentence_text = []

# for ind in df.index:
#     sentence = df['Tokenized_Feedback'][ind]
#     sentence = [word for word in sentence if word not in ('!','','?',', ', '.',',','(','--',')')]
#     sentence = [word for word in sentence if word not in stopwords]
#     tagged = nltk.pos_tag(sentence)
#     included_tags = ['JJ']
#     tagged = [tag for tag in tagged if tag[1] in included_tags]
#     sentence = ' '.join(tag[0] for tag in tagged)
#     sentence_text.append(sentence)

# combined = ' '.join(sentence_text)

cloud = WordCloud(background_color = "white").generate(combined)
plt.imshow(cloud)
#plt.show()




