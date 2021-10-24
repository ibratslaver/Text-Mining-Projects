from Database_Utility import *
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize


df = pd.read_excel('Consolidated Feedback.xlsx')

df['Tokenized_Feedback'] = df['Feedback'].apply(str).apply(sent_tokenize)

#print(df['Tokenized_Feedback'])

for Feedback_ID,Feedback,Date,Recommend in zip(df.ID,df.Tokenized_Feedback, df.Date, df.Recommend):
    for Sentence in Feedback:
        Sentence = Sentence.replace("'","''")
        Feedback_ID = str(Feedback_ID)
        Recommend = str(Recommend)
        insert_ga_feedback_sentences(Feedback_ID, Sentence, Date, Recommend)


     
