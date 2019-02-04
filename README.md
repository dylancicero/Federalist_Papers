# Authorship Unknown? A Textual Analysis of the Federalist Papers

##### This analysis seeks to predict the true authors of the Federalist Papers where authorship is currenty contested.  Methods involve scraping the Federalist Papers from the web, consolidating them by authorship, evaluating the frequency of tokens by author, conducting principle components & linear discriminant analysis to generate predictions, and evaluating the confidence of results.  Results suggest with a high degree of confidence that all 11 Federalist Papers with currently contested authorship were in fact writted by James Madison. Below find my full set of code and commentary.

##### The first part of this analysis was written in Python:

```Python3
import requests
from os import getcwd
from os import chdir
from os import listdir as ls
import pandas as pd
from bs4 import BeautifulSoup
import nltk
import numpy
import csv


path = getcwd()
print(path)
chdir('/Users/dylancicero/Desktop/Data_Analysis/Federalist_Papers')
path = getcwd()
print(path)
ls()
```
Create list of urls for each federalist paper:
```Python3
url_list = []
for number in range(1,86):
    url = "http://avalon.law.yale.edu/18th_century/fed"
    if number < 10:
        url = url + "0" + str(number) + ".asp"
    else:
        url = url + str(number) + ".asp"
    url_list.append(url)
url_list
```
Get list of authors:
```Python3
r = requests.get("https://www.congress.gov/resources/display/content/The+Federalist+Papers")
s = BeautifulSoup(r.text)
table = s.find("table", {"class": "confluenceTable"})
rows = table.findAll("tr")
for row in rows[1:]:
    author.append(row.findAll("td")[2].text)

df = pd.DataFrame(author)
df.to_csv('authors.csv')
```
Iterate through url_list, saving the content of each federalist paper in a .txt file:
```Python3
headers = []
for i in range(85):
    url = url_list[i]
    r = requests.get(url)
    s = BeautifulSoup(r.text)
    paragraphs = [p.text for p in s.findAll("p")]
    paragraphs
    if i < 9:
        f = open("0{}.txt".format(i+1), "w")
        for paragraph in paragraphs:
            f.write(paragraph + "\n")
        f.close()
    else:
        f = open("{}.txt".format(i+1), "w")
        for paragraph in paragraphs:
            f.write(paragraph + "\n")
        f.close()
```
Merge papers for each author:
```Python3
hamilton = ""
madison = ""
jay = ""
chdir("/Users/dylancicero/Desktop/Data_Analysis/Federalist_Papers")
docnames = [f for f in os.listdir() if f[-4:]==".txt"]
docnames.sort()

for i in range(85):
    with open(docnames[i], 'r') as myfile:
        text = myfile.read().replace('\n', '')
        if author[i] == "Hamilton":
            hamilton = hamilton + text
        if author[i] == "Madison":
            madison = madison + text
        if author[i] == "Jay":
            jay = jay + text
```
Remove occurences of meaningless text strings from each set:
```Python3
hamilton.replace("To the People of the State of New York:","")
hamilton.replace("Publius","")
madison.replace("To the People of the State of New York:","")
madison.replace("Publius","")
jay.replace("To the People of the State of New York:","")
jay.replace("Publius","")
```
Define function to draw next word from frequency distribution:
```Python3
def draw_word(distrn):
    words = list(distrn)
    freqs = [freq for w, freq in distrn.items()]
    total = sum(freqs)
    probs = [freq/total for freq in freqs]
    return numpy.random.choice(words, p=probs)
```
Define function to generate third-order Gauss Markov frequency distribution for a given text and write an 100-word passage based on random selection of the next word from the frequency distribution:
```Python3
def generate_with_trigrams(text, word=None, num=100):
    tokens = nltk.tokenize.word_tokenize(text)
    trigrams = nltk.trigrams(tokens)
    condition_pairs = (((w0, w1), w2) for w0, w1, w2 in trigrams)
    cfdist = nltk.ConditionalFreqDist(condition_pairs)
    if word is None:
        prev = draw_word(nltk.FreqDist(tokens))
        word = draw_word(nltk.ConditionalFreqDist(nltk.bigrams(tokens))[prev])
    elif len(word.split()) == 1:
        prev = word
        word = draw_word(nltk.ConditionalFreqDist(nltk.bigrams(tokens))[prev])
        # will give an error if this pair doesn't show up in the text
    else:
        prev, word = word.split()[:2]
    print(prev, end=' ')
    for i in range(1, num):
        print(word, end=' ')
        prev, word = word, draw_word(cfdist[(prev, word)])

generate_with_trigrams(hamilton, "The")
generate_with_trigrams(madison, "The")
generate_with_trigrams(jay, "The")
```
Extract the tokens' relative frequencies and export to .csv:
```Python3
N = len(docnames)
print(N)
docs = [None]*N
dict_list = []
for i in range(N):
    with open(docnames[i], 'r') as f:
        docs[i] = f.read()
        tokens = nltk.tokenize.word_tokenize(docs[i])
        total = len(tokens)
        table = nltk.FreqDist(tokens)
        temp_dict = {}
        for w,freq in dict(table).items():
            prob = freq/total
            temp_dict[w]=prob
        dict_list.append(temp_dict)

df = pd.DataFrame(dict_list)
chdir('/Users/dylancicero/Desktop/Data_Analysis/Federalist_Papers')
path = getcwd()
df.to_csv('federalist_frequencies.csv')
```
