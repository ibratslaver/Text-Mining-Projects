from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
#from Database_Utility import *
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
import os

stopwords = stopwords.words('english')

# newStopWords = ['instructor', 'Instructor', 'Igor', 'teacher','data', 'analytics', 'well', 'great', 'job', 'kept', 'really', 'everyone', 'time', 'class', 'examples', 'questions','us','understand','easy']
# stopwords.extend(newStopWords)

df = pd.read_excel('GA Feedback Categorized.xlsx', sheet_name = "Instructor")

df['Tokenized_Feedback'] = df['feedback'].apply(word_tokenize)


# CODE TO BREAK FEEDBACK INTO SENTENCES

# for Feedback_ID,Feedback,Date,Recommend in zip(df.ID,df.Tokenized_Feedback, df.Date, df.Recommend):
#     for Sentence in Feedback:
#         Sentence = Sentence.replace("'","''")
#         Feedback_ID = str(Feedback_ID)
#         Recommend = str(Recommend)
#         insert_ga_feedback_sentences(Feedback_ID, Sentence, Date, Recommend)

# #VERSION 1: Only Manual Filtering with the stopwords

# sentence_text = []

# for ind in df.index:
#     sentence = df['Tokenized_Feedback'][ind]
#     sentence = [word for word in sentence if word not in ('!','','?',', ', '.',',','(','--',')')]
#     sentence = [word for word in sentence if word not in stopwords]
#     sentence = ' '.join(word for word in sentence)
#     #print(sentence)
#     sentence_text.append(sentence)

# combined = ' '.join(sentence_text)

# cloud = WordCloud(background_color = "white").generate(combined)
# plt.imshow(cloud)
# plt.show()


#VERSION 2: Only Manual Filtering with the stopwords

# sentence_text = []

# for ind in df.index:
#     sentence = df['Tokenized_Feedback'][ind]
#     sentence = [word for word in sentence if word not in ('!','','?',', ', '.',',','(','--',')')]
#     sentence = [word for word in sentence if word not in stopwords]
#     tagged = nltk.pos_tag(sentence)
#     print(tagged)
#     included_tags = ['JJ','VBG']
#     tagged = [tag for tag in tagged if tag[1] in included_tags]
#     sentence = ' '.join(tag[0] for tag in tagged)
#     sentence_text.append(sentence)

# combined = ' '.join(sentence_text)
# print(combined)

# cloud = WordCloud(background_color = "white").generate(combined)
# plt.imshow(cloud)
# plt.show()







