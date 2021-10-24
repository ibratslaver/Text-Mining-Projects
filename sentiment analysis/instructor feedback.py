import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#from Database_Utility import *
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from afinn import Afinn 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np

analyzer = SentimentIntensityAnalyzer()

sentiment = analyzer.polarity_scores("I LOVE this ")
print(sentiment)

df = pd.read_excel("Consolidated Feedback.xlsx")

df = df[pd.notnull(df.Feedback)]

#Convert to lowercase
df['Feedback'] = df.Feedback.astype(str).str.lower()

print(df['Feedback'])

#Word Tokenization
df['Feedback_Tokenized'] = df['Feedback'].apply(word_tokenize)

df.to_csv("TOOKENIZE.csv")

stop_words = stopwords.words('english')

big_list = []  
  
for ind in df.index:
    filtered_list = []
    review = df['Feedback_Tokenized'][ind]
    review = [word for word in review if word.isalnum()] 
    review = [word for word in review if word.isnumeric() == False]

    for w in review:
        if w not in stop_words:
            filtered_list.append(w)
    big_list.append(filtered_list)




def listToString(s):  
    str1 = " " 
    return (str1.join(s))

df1 = pd.DataFrame(zip(big_list), columns = ['Feedback'])

#df1.to_csv("new_df.csv")

df1['Feedback']= df1['Feedback'].apply(listToString)




def vaderize(df, textfield):
    '''Compute the Vader polarity scores for a textfield.
    Returns scores and original dataframe.'''

    #initiate sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    #calculate sentiments
    print('Estimating polarity scores for %d cases.' % len(df))
    sentiment = df1[textfield].fillna("").apply(analyzer.polarity_scores)

    # convert to dataframe
    sdf = pd.DataFrame(sentiment.tolist()).add_prefix('vader_')

    # merge with original dataframes
    df_combined = pd.concat([df1, sdf], axis=1)
    return df_combined

df_vaderized = vaderize(df1, 'Feedback')

#df_vaderized = df_vaderized[pd.notnull(df.Feedback)]


#Histograms for each of the sentiments
plt.hist(df_vaderized['vader_pos'],color='green')
plt.title("Positive Sentiment")
plt.show()

plt.hist(df_vaderized['vader_compound'],color='blue')
plt.title("Compound Sentiment")
plt.show()

plt.hist(df_vaderized['vader_neg'],color='red')
plt.title("Negative Sentiment")
plt.show()
