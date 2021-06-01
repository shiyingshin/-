import snscrape.modules.twitter as sntwitter
import pandas as pd
import re

##数据收集##
tweets_list3 = []

#用TwitterSearchScraper，收集数据，存到tweetslist3
# since - until 2021.5月的信息!
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('#StopAsianHate since:2021-05-01 until:2021-05-31').get_items()):
    if i>10000:
        break
    tweets_list3.append([tweet.date, tweet.id, tweet.content, tweet.username])
    
#形成df
tweets_df3= pd.DataFrame(tweets_list3, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])

tweets_list3

list=[]

for i in range(len(tweets_list3)):
    list.append(tweets_list3[i][2])
T = list

##数据整理##
Feb=(" ").join(T)
Feb
F= re.sub(r'\n', "", Feb)
F = re.sub(r'[^\w\s#@+:/.]', '', F)
F = re.sub(r'https://[a-zA-Z0-9\.\/?_=&]+', '', F)
F = re.sub(r'\W*\b\w{1,3}\b', '', F)
F=F.lower()
#词性标注
from nltk.tokenize import TweetTokenizer
t = TweetTokenizer()
F=t.tokenize(F)

from nltk.stem import WordNetLemmatizer
n=WordNetLemmatizer()
F = [n.lemmatize(w) for w in F]
print(F)

#前20个单词频率分析
import pandas as pd
pd.Series(F).value_counts().head(20)
#词云图
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt

from os import path
FONT_PATH = '/WINDOWS/FONTS/H2GTRE.TTF'

cloud=(" ").join(F)
cloud = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', cloud)


wordcloud = WordCloud(width = 500, height = 500 ,font_path = FONT_PATH).generate(cloud)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='sinc')
plt.axis("off")
plt.show()
plt.close()

#舆情分析
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentences):
    score = analyser.polarity_scores(sentences)
    print("{}".format(str(score)))
    
print(sentiment_analyzer_scores(cloud))

#语权分布分析
import pandas as pd # dataframes
import langid 
from matplotlib.pyplot import plot

from nltk.tokenize import sent_tokenize
F= sent_tokenize(feb)

df =pd.DataFrame(F, columns=['text'])
df

ids_langid = df['text'].apply(langid.classify)

langs = ids_langid.apply(lambda tuple: tuple[0])

print("Number of tagged languages (estimated):")
print(len(langs.unique()))

print("Percent of tweets in English (estimated):")
print((sum(langs=="en")/len(langs))*100)

langs_df = pd.DataFrame(langs)


langs_count = langs_df.text.value_counts()


langs_count.plot.bar(figsize=(20,10), fontsize=20)

#使用100次以上的单词分析
print("Languages with more than 100 tweets in our dataset:")
print(langs_count[langs_count > 100])

print("")

print("Percent of our dataset in these languages:")
print((sum(langs_count[langs_count > 100])/len(langs)) * 100)