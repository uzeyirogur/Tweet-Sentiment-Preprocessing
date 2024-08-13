import nltk 
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt 
import random 
import numpy as np 

#nltk.download("twitter_samples")

#apt positive ant negative tweets
all_positive_tweets = twitter_samples.strings("positive_tweets.json")
all_negative_tweets = twitter_samples.strings("negative_tweets.json")

print('Number of positive tweets: ', len(all_positive_tweets))
print('Number of negative tweets: ', len(all_negative_tweets))

print('\nThe type of all_positive_tweets is: ', type(all_positive_tweets))
print('The type of a tweet entry is: ', type(all_negative_tweets[0]))

#Plot
fig = plt.figure(figsize=(5,5))
labels = "Positive","Negative"
sizes = [len(all_positive_tweets),len(all_negative_tweets)]
plt.pie(sizes,labels=labels,autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.axis('equal')  
plt.show()

print('\033[92m' + all_positive_tweets[random.randint(0,5000)])
print('\033[91m' + all_negative_tweets[random.randint(0,5000)])

tweet = all_positive_tweets[2277]
print(tweet)

#nltk.download('stopwords')

import re                                  # library for regular expression operations
import string                              # for string operations

from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings


tweet2 = re.sub(r"^RT[\s]+","",tweet)
tweet2 = re.sub(r'https?://[^\s\n\r]+', '', tweet2)
tweet2 = re.sub(r'#', '', tweet2)

#Tokinezer dizeleri boşluklar ve sekmeler olmadan tek tek kelimelere bölme işlemi
tokenizer = TweetTokenizer(preserve_case=False,strip_handles=True,reduce_len=True)   
"""
#preserve_case = Metindeki küçük büyük harf duyarı false yapılırsa hepsi küçük harfe döner
#strip_handles = Metindeki kullanıcı isimlerinin (örneğin @kullanici) kaldırılmasını belirler.
#reduce_len = Bu parametre, metinde ardışık karakterlerin uzunluğunu azaltmayı belirler. coooool---> cool olarak çevirir
# 
"""
tweet_tokens = tokenizer.tokenize(tweet2)

stopwords_english = stopwords.words('english') 
tweets_clean = []
for word in tweet_tokens :
    if (word not in stopwords_english and word not in string.punctuation) :
        tweets_clean.append(word)
    
#Porter Stemmer 
stemmer = PorterStemmer()
tweets_stem = []
for word in tweets_clean :
    stem_word = stemmer.stem(word)
    tweets_stem.append(stem_word)

#HER YERDE KULLANABİLİRSİN SAKLAAAAA
from utils import process_tweet # Import the process_tweet function
from utils import build_freqs
# choose the same tweet
tweet = all_positive_tweets[2277]
print(tweet)
# call the imported function
tweets_stem = process_tweet(tweet); # Preprocess a given tweet
print('preprocessed tweet:')
print(tweets_stem) # Print the result


    
