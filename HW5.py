import spacy
import string
import en_core_web_lg
from newsapi.newsapi_client import NewsApiClient
import pickle
import pandas as pd
from spacy.lang.en import punctuation
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nlp_eng = en_core_web_lg.load()
newsapi = NewsApiClient (api_key='753eb8f70f404f9e8a1a0cb4cce69b2d')
articles = []

for i in range(1,6):
   temp = newsapi.get_everything(q='coronavirus', language='en', 
                                 from_param='2021-02-25', to='2021-03-23',
                                 sort_by='relevancy', page=i)
   articles.append(temp)
# print(articles)

filename = 'articlesCOVID.pck1'
pickle.dump(articles, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))

filepath = 'D:/CPP/Junior Year/Spring 2021/CS 4650/Homework/HW5/articlesCOVID.pck1'
pickle.dump(loaded_model, open(filepath, 'wb'))

dados = []

for i, article in enumerate(articles):
   for x in article['articles']:
      title = x['title']
      date = x['publishedAt']
      description = x['description']
      content = x['content']
      dados.append({'title':title, 'date':date,
                    'desc':description, 'content':content})

df = pd.DataFrame(dados)
df = df.dropna()
df.head()
pickle.dump(df, open(filename, 'wb'))
# print(df)

def get_keywords_eng(content):
   doc = nlp_eng(content)
   result = []
   pos_tag = ["VERB","NOUN","PROPN"]
   for token in doc:
       if (token.text in nlp_eng.Defaults.stop_words or token.text in string.punctuation):
          continue
       if (token.pos_ in pos_tag):
          result.append(token.text)
   return result


results = []
for content in df.content.values:
    results.append([('#' + x[0]) 
    for x in Counter(get_keywords_eng(content)).most_common(5)])
df['keywords'] = results

text = str(results)
print(text)
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()



