import pandas as pd
import nltk
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import RegexpTokenizer
import matplotlib.cm as cm

df = pd.read_excel(r"C:\Users\Igor\Desktop\RMP Dataset ADM.xlsx", sheet_name="data")

#convert text column to all lowercase
df['review_text'] = df['review_text'].str.lower()

#Word Tokenization
df['review_text_tokenized'] = df['review_text'].apply(word_tokenize)

stop_words = stopwords.words('english')
newStopWordsClass = ['class','teacher']
stop_words.extend(newStopWordsClass)

df = df.head(1)

big_list = []  
  
for ind in df.index:
    filtered_list = []
    review = df['review_text_tokenized'][ind]
    review = [word for word in review if word.isalnum()] 
    review = [word for word in review if word.isnumeric() == False]
    print(review)

    for w in review:
        if w not in stop_words:
            filtered_list.append(w)
    big_list.append(filtered_list)

print(big_list)

def listToString(s):  
    str1 = " " 
    return (str1.join(s))

#convert list to dataframe
df1 = pd.DataFrame(zip(big_list), columns = ['review_text'])

#convert list to string
df1['review_text_clean']= df1['review_text'].apply(listToString)


#delete review_text column
df1 = df1.drop('review_text', axis=1)

df1_combined = pd.concat([df, df1], axis=1)

def vaderize(df, textfield):
    '''Compute the Vader polarity scores for a textfield.
    Returns scores and original dataframe.'''

    #initiate sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    #calculate sentiments
    print('Estimating polarity scores for %d cases.' % len(df))
    sentiment = df1_combined[textfield].apply(analyzer.polarity_scores)

    # convert to dataframe
    sdf = pd.DataFrame(sentiment.tolist()).add_prefix('vader_')

    # merge with original dataframes
    df_combined = pd.concat([df1_combined, sdf], axis=1)
    return df_combined

df_vaderized = vaderize(df1_combined, 'review_text_clean')

#remove no comments
df_vaderized = df_vaderized[df_vaderized['review_text'] != 'No Comments']

#create year column
df_vaderized['review_year'] = df_vaderized['review_date'].dt.year

# #line charts for each professor by department
# for (groupname,subdf) in df_vaderized.groupby('department'):
    subdf = pd.pivot_table(subdf, values="vader_compound", index='review_year', columns='professor_name')
    subdf.plot(title=groupname, subplots=True)
    plt.ylabel("vader_compound")
    plt.savefig(groupname +  " - " +  "Sentiment by Professor")

# #histogram for each professor seperate
for (groupname, department), subdf in df_vaderized.groupby(['professor_name','department']):
    subdf.hist(column="vader_compound")
    plt.title(groupname + ' - ' + department)
    plt.ylabel('frequency')
    plt.xlabel('vader_compound')

# #histogram for each department, seperate
for (groupname, subdf) in df_vaderized.groupby(['department']):
    subdf.hist(column="vader_compound")
    plt.title(groupname)
    plt.ylabel('frequency')
    plt.xlabel('vader_compound')
    plt.savefig(groupname + " - " + "Histogram by Department")

# #scatterplot for each professor

# # #set colors
colors = ['blue','red','green','purple','orange','black']
color_index = 0

# for color, (groupname, subdf) in zip(colors,df_vaderized.groupby('professor_name')):
    plt.scatter(x=subdf['vader_pos'], y = subdf['vader_neg'], color=colors[color_index])
    plt.title(groupname)
    color_index+=1

# #scatter plot for each department
for color, (groupname, subdf) in zip(colors,df_vaderized.groupby('department')):
        for (profname, subsubdf) in subdf.groupby('professor_name'):
            plt.scatter(subsubdf['vader_pos'], subsubdf['vader_neg'],c=subsubdf['professor_name'])
            plt.label(profname)
            plt.title(groupname)
            plt.show()


# #WORD CLOUD - SARAH WILEY, HISTORY, 2009, LOW, N = 20
df_sarah_wiley = df_vaderized[(df_vaderized['review_year']==2009) & (df_vaderized['professor_name'] =='Sarah Wiley')]
combined = []

# for ind in df_sarah_wiley.index:
    sentence = df_sarah_wiley['review_text_clean'][ind]
    combined.append(sentence)
   
# combined = ' '.join(combined)

# cloud = WordCloud(background_color = "white").generate(combined)
plt.imshow(cloud)
plt.show()


# #WORD CLOUD - ARLENE CASIPLE, MATH, 2008, LOW, N=23
df_arlene_caspile = df_vaderized[(df_vaderized['review_year']==2008) & (df_vaderized['professor_name'] =='Arlene Casiple')]

combined = []

# for ind in df_arlene_caspile.index:
    sentence = df_arlene_caspile['review_text_clean'][ind]
    combined.append(sentence)
   
combined = ' '.join(combined)

# cloud = WordCloud(background_color = "white").generate(combined)
plt.imshow(cloud)
plt.show()


# #WORD CLOUD - SEAN DAVIS, COMPUTER SCIENCE, 2008, HIGH, N=8
df_sean_davis = df_vaderized[(df_vaderized['review_year']==2008) & (df_vaderized['professor_name'] =='Sean Davis')]

combined = []

for ind in df_sean_davis.index:
    sentence = df_sean_davis['review_text_clean'][ind]
    combined.append(sentence)
   
combined = ' '.join(combined)

cloud = WordCloud(background_color = "white").generate(combined)
plt.imshow(cloud)
plt.show()

# #WORD CLOUD - KIM LUTTON, SOCIOLOGY, 2011, HIGH, N=18
df_kim_luton = df_vaderized[(df_vaderized['review_year']==2011) & (df_vaderized['professor_name'] =='Kim Luton')]

combined = []

for ind in df_kim_luton.index:
    sentence = df_kim_luton['review_text_clean'][ind]
    combined.append(sentence)
   
combined = ' '.join(combined)

cloud = WordCloud(background_color = "white").generate(combined)
plt.imshow(cloud)
plt.show()


# # #WORD CLOUD - Steven Scarborough, Math, 2010, HIGH, N=15
df_steve_scar = df_vaderized[(df_vaderized['review_year']==2010) & (df_vaderized['professor_name'] =='Stephen Scarborough')]

combined = []

# for ind in df_steve_scar.index:
    sentence = df_steve_scar['review_text_clean'][ind]
    combined.append(sentence)
   
combined = ' '.join(combined)

cloud = WordCloud(background_color = "white").generate(combined)
plt.imshow(cloud)
plt.show()


# #WORD CLOUD - Steven Scarborough, Math, 2010, HIGH, N=15
df_steve_scar = df_vaderized[(df_vaderized['review_year']==2010) & (df_vaderized['professor_name'] =='Stephen Scarborough')]

combined = []

# for ind in df_steve_scar.index:
    sentence = df_steve_scar['review_text_clean'][ind]
    combined.append(sentence)
   
combined = ' '.join(combined)

# cloud = WordCloud(background_color = "white").generate(combined)
plt.imshow(cloud)
plt.show()

color_map = {'a': 'r', 'b': 'b', 'c': 'y'}
ax = plt.subplot()
x, y = df.cpu, df.wait
colors = df.category.map(color_map)
colors = ['r','r','r','y','y','b']

plt.scatter(x, y, c = df.time)
plt.show()


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

