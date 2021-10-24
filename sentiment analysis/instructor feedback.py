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


#Option 1, this still creates a stacked bar chart but straight from the group by
# df_vaderized.groupby(df_vaderized['Date']).vader_compound.mean().plot(kind='bar', color='green')
# df_vaderized.groupby(df_vaderized['Date']).vader_pos.mean().plot(kind='bar', color='red')
# df_vaderized.groupby(df_vaderized['Date']).vader_neg.mean().plot(kind='bar',color='blue')

# #Option 2: Turn the series into a dataframe
# compound_series = df_vaderized.groupby(df_vaderized['Date']).vader_compound.mean()
# pos_series = df_vaderized.groupby(df_vaderized['Date']).vader_pos.mean()
# neg_series = df_vaderized.groupby(df_vaderized['Date']).vader_neg.mean()

# df = pd.DataFrame({'Date':compound_series.index, 'compound_mean':compound_series.values})
# df1 = pd.DataFrame({'Date':pos_series.index, 'pos_mean':pos_series.values})
# df2 = pd.DataFrame({'Date':neg_series.index, 'neg_mean':neg_series.values})

# df['Date'] = df['Date'].dt.strftime('%Y-%m')
# df1['Date'] = df1['Date'].dt.strftime('%Y-%m')
# df2['Date'] = df2['Date'].dt.strftime('%Y-%m')

# #Create bar chart
# x = df.Date
# y = df.compound_mean

# x1 = df1.Date
# y1 = df1.pos_mean

# x2 = df2.Date
# y2 = df2.neg_mean

# N = 9
# width = .35
# ind = np.arange(N) 

# print(ind)

# plt.bar(ind, y, width, label = 'Compound')
# plt.bar(ind + width, y1, width, label = 'Pos',)
# plt.bar(ind + width*2, y2, width, label = 'Neg')

# plt.xticks(ind + width / 2, df.Date)
# plt.legend(loc='best')

# plt.show()

