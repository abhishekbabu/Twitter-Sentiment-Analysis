import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import warnings
from nltk.stem.porter import *
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Importing tweet data
train  = pd.read_csv('train_E6oV3lV.csv')
test = pd.read_csv('test_tweets_anuFYb8.csv')

# Combining training and test data for feature engineering and cleaning up input
combined = train.append(test,ignore_index=True)

# Function to remove character patterns from tweets
def remove_pattern(input_txt, pattern):
	res = re.findall(pattern, input_txt)
	for each in res:
		input_txt = re.sub(each,'',input_txt)

	return input_txt

# Removing user handles (words that start with '@') from tweets
combined['clean_tweet'] = np.vectorize(remove_pattern)(combined['tweet'], "@[\w]*")

# Removing all special characters except alphabetic characters and hashtags
combined['clean_tweet'] = combined['clean_tweet'].str.replace("[^a-zA-Z#]", " ")

# Removing all words of length less than or equal to 3
combined['clean_tweet'] = combined['clean_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

# Splitting each tweet into individual words
tokenized_tweet = combined['clean_tweet'].apply(lambda x: x.split())

# Removing suffixes from words using Natural Language Tool Kit's stemmer - significantly increases runtime
# stemmer = PorterStemmer()
# tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(each) for each in x])

# Joining tokens back into tweets
for each in range(len(tokenized_tweet)):
	tokenized_tweet[each] = ' '.join(tokenized_tweet[each])
combined['clean_tweet'] = tokenized_tweet

## Wordcloud of all words
# all_words = ' '.join([text for text in combined['clean_tweet']])
# wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
# plt.figure(figsize=(10, 7))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis('off')
# plt.show()

## Wordcloud of neutral words
# normal_words =' '.join([text for text in combined['clean_tweet'][combined['label'] == 0]])
# wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
# plt.figure(figsize=(10, 7))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis('off')
# plt.show()

## Wordcloud of negative words
# negative_words = ' '.join([text for text in combined['clean_tweet'][combined['label'] == 1]])
# wordcloud = WordCloud(width=800, height=500,
# random_state=21, max_font_size=110).generate(negative_words)
# plt.figure(figsize=(10, 7))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis('off')
# plt.show()

def hashtag_extract(x):
	hashtags = []
	for each in x:
		ht = re.findall(r"#(\w+)", each)
		hashtags.append(ht)

	return hashtags

# Extracting hashtags from neutral and negative tweets
HT_regular = hashtag_extract(combined['clean_tweet'][combined['label'] == 0])
HT_negative = hashtag_extract(combined['clean_tweet'][combined['label'] == 1])

# Unnesting lists of hastags
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])

## Bar plot of top 10 neutral hashtags
# a = nltk.FreqDist(HT_regular)
# d = pd.DataFrame({'Hashtag': list(a.keys()),
#                   'Count': list(a.values())})
# d = d.nlargest(columns="Count", n = 10) 
# plt.figure(figsize=(16,5))
# ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
# ax.set(ylabel = 'Count')
# plt.show()

## Bar plot of top 10 negative hashtags
# b = nltk.FreqDist(HT_negative)
# e = pd.DataFrame({'Hashtag': list(b.keys()),
# 				  'Count': list(b.values())})
# e = e.nlargest(columns="Count", n = 10)   
# plt.figure(figsize=(16,5))
# ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
# ax.set(ylabel = 'Count')
# plt.show()

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(combined['clean_tweet'])

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(combined['clean_tweet'])



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_bow = bow[:31962,:]
test_bow = bow[31962:,:]

xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)

lreg = LogisticRegression(tol=0.00001, max_iter=10000)
lreg.fit(xtrain_bow, ytrain)

prediction = lreg.predict_proba(xvalid_bow)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)

print(f1_score(yvalid, prediction_int))


test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['id','label','tweet']]
submission.to_csv('sub_lreg_bow.csv', index=False)