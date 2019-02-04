# Authorship Unknown? A Textual Analysis of the Federalist Papers

### This analysis seeks to predict the true authors of the Federalist Papers where authorship is currenty contested.  Methods involve scraping the Federalist Papers from the web, consolidating them by authorship, evaluating the frequency of tokens by author, conducting principle components & linear discriminant analysis to generate predictions, and evaluating the confidence of results.  Results suggest with a high degree of confidence that all 11 Federalist Papers with currently contested authorship were in fact writted by James Madison. Below find my full set of code and commentary.

#### The first part of this analysis is written in Python:

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

#### The second part of this analysis is written in R.
Read in .csv of frequencies:
```R
# Read in the generated csv of token frequencies for each of the 85 Federalist Papers
# Convert the 'na' values in data matrix to 0
# Display the first 6 rows and columns
x <- read.csv('/Users/dylancicero/Desktop/Data_Analysis/Federalist_Papers/federalist_frequencies.csv', row.names=1)
x[is.na(x)] <- 0
x[1:6, 1:6]
# xbar is a vector of the mean of frequencies for a given token across all papers
xbar <- apply(x, 2, mean)
# iterate across each row, subtracting the mean for a given token from the frequency value for a given paper
# (not necessary, but centers the axes around data)
for(i in 1:nrow(x)) {
  x[i, ] <- x[i, ] - xbar
}
```
Read in authors:
```R
# Read in csv of authors.  Transform it into a vector listing the authors in order from first to last paper.
a <- read.csv('/Users/dylancicero/Desktop/Data_Analysis/Federalist_Papers/authors.csv')$X0
# Create a variable "single", a vector of boolean values in order from first to last paper, 
# where TRUE means that the paper had single authorship.
single <- a %in% c("Hamilton", "Madison", "Jay")
# Create a new matrix 'y', extracting the papers (rows) from matrix 'x' for which there is a known single author
y <- x[single, ]
# Develop factors for the three single authors, to be used later in analytic charts
b <- as.factor(as.character(a[single]))
```
Conduct principle components analysis and plot the results:
```R
# PCA (note: prior to conducting pca, units of features should be normalized.  All units here are of the same type- relative
# frequency- and thus scaling in this case has the undesired effect of reducing overall variance between data points)
p <- prcomp(y)
# 'prot' is an output of pca.  Multiplying the original matrix y by a given number of columns in prot 
# outputs a matrix where the original features are replaced by the same number ^^ of principle components.
# This represents a rotation of the original axes of the dataframe, whereby the first principle component
# maximizes variance across all data points, and the second principle component maximizes variance 
# across all datapoints in a direction orthagonal to the first.
prot <- (p$rotation)

# Plot the second principle component against the first, colored by authors.
plot(as.matrix(y) %*% prot[, 1:2], col=b, main = "Principle Components Analysis")
legend("topleft", levels(b), text.col=1:length(levels(b)),
       fill=1:length(levels(b)))
mtext("(means subtracted)", side = 3, adj = 0.5, line = 0.4)
```

![rplot1](https://user-images.githubusercontent.com/43111524/52185457-cfe2f200-27ed-11e9-8497-a84776723205.png)
The first two principle components are able to distinguish the authors with a little overlap. But we could do better...

Conduct linear discriminant analysis:
```R
# LDA
library(MASS)
# The below (2) lines (commented out) demonstrate attempts to run lda on the full data set as well as on the first 20
# features of the full data set.  Lda is unable to operate in these cases due to the sparsity of the data set.
# absent <- which(apply(y, 2, var)==0)
# y[, c(10,15, 34,35,39,56)] # test
# x[, c(10,15, 34,35,39,56)] # test
# z <- y[, -absent]
# l <- lda(z, grouping=b)
# l <- lda(z[, 1:20], grouping=b)
# In order to run lda, we first perform pca, keeping only the first 40 principle components.
w <- as.matrix(y) %*% p$rotation[, 1:40]
# The resultant matrix 'w' is neither sparse nor too large.
l <- lda(w, grouping=b)
```
Evaluate results of LDA.  LDA rotates the axes of the data such that the known groups will by optimally seperated. This could be misleading if a random labeling of the data partitions the data as well as the true labeling of the data.  To quickly contrast true and random labeling of the data, make a null lineup:
```R
par(mfrow = c(3, 3))
answer <- sample(1:9, 1)
for (i in 1:9) {
  if (i == answer) {
    # LDA for the original data
    plot(as.matrix(w) %*% l$scaling[, 1:2], col=b)
  } else {
    # LDA with labels reassigned at random
    B <- sample(b, length(b))
    L <- lda(w, grouping=B)
    plot(as.matrix(w) %*% L$scaling[, 1:2], col=B)
  }
}
par(mfrow = c(1, 1))
# Show the true plot alone
plot(as.matrix(w) %*% l$scaling[, 1:2], col=b)
legend("topright", levels(b), text.col=1:length(b),
       fill=1:length(levels(b)))
```
![rplot](https://user-images.githubusercontent.com/43111524/52185569-cdcd6300-27ee-11e9-9bda-f45acc278a2e.png)
It is obvious that plot #8 stands out from the rest.

![rplot01](https://user-images.githubusercontent.com/43111524/52185572-d3c34400-27ee-11e9-96a0-5430ff889c22.png)
The true plot displayed alone

I based the LDA above off the first 40 principle components.  This raises an interesting question... What proportion of the variance is captured by each additional principle component?
```R
# Create a vector of the variance for each principle component
ps <- p$sdev^2
# Create another vector of same length and populate it with the sum of its variance and all prior pcs
pss <- rep(NA, length(ps))
pss[1] <- ps[1]
pss
for(i in 2:length(ps)) {
  pss[i] <- pss[i-1] + ps[i]
}
#
plot(1:length(ps), (pss/pss[length(pss)] * 100), 
     main = "Percent of Total Variance Captured by Principle Components", 
     xlab = "ith Principle Component", ylab = "Percent of Total Variance Captured", type = "b")
```


