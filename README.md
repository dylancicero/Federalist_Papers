## Authorship Unknown? A Textual Analysis of the Federalist Papers

This analysis seeks to predict the true authors of the Federalist Papers where authorship is currently contested.  Methods involve scraping the Federalist Papers from the web, consolidating them by authorship, evaluating the frequency of tokens by author, conducting principle components & linear discriminant analysis to generate predictions, and evaluating the confidence of results.  Results suggest with a high degree of confidence that all 11 Federalist Papers with currently contested authorship were in fact writted by James Madison. Below find my full set of code and commentary.

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

<img width="1051" alt="screen shot 2019-02-23 at 3 16 02 pm" src="https://user-images.githubusercontent.com/43111524/53291213-f0afbf00-377d-11e9-8f46-f073c09ff23c.png">
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
<img width="1052" alt="screen shot 2019-02-23 at 3 18 39 pm" src="https://user-images.githubusercontent.com/43111524/53291238-4b491b00-377e-11e9-8692-0cfb2ccdbe49.png">
It is obvious that plot #6 stands out from the rest.

<img width="1052" alt="screen shot 2019-02-23 at 3 24 44 pm" src="https://user-images.githubusercontent.com/43111524/53291315-2ef9ae00-377f-11e9-8667-edfde31fb8ab.png">
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
![rplot02](https://user-images.githubusercontent.com/43111524/52185617-50eeb900-27ef-11e9-9bc2-12ff2309f8c8.png)
40 principle components capture around 85% of the total variance, but 60 principle components capture around 95%...

Trial LDA based off of 60 principle components:
```R
w <- as.matrix(y) %*% p$rotation[, 1:60]
l <- lda(w, grouping=b)
plot(as.matrix(w) %*% l$scaling[, 1:2], col=b)
legend("topright", levels(b), text.col=1:length(b),
       fill=1:length(levels(b)))
```
<img width="1052" alt="screen shot 2019-02-23 at 3 26 18 pm" src="https://user-images.githubusercontent.com/43111524/53291327-f9a19000-377f-11e9-8276-6e9d24210865.png">
Clusters are much tighter here!  However, test cases (see below) perform better using only 40 principle components. The 40 pcp method will be evaluated from here on out. 

Predicting true authorship of joint or unknown papers:
```R
# add documents that are attributed to both Hamilton and Madison
both <- which(a == "Hamilton and Madison")
zb <- x[both, ]
wb <- as.matrix(zb) %*% p$rotation[, 1:60]
points(as.matrix(wb) %*% l$scaling[, 1:2], col=4)
# add documents of unknown authorship
unknown <- which(a == "Hamilton or Madison")
zu <- x[unknown, ]
wu <- as.matrix(zu) %*% p$rotation[, 1:60]
text(wu %*% l$scaling[, 1:2], labels=rownames(wu), col=5)
### Discussed in class (predict function)
p <- points(wu %*% l$scaling, col=5, pch="x", cex=2)
points(wu %*% l$scaling, col=predict(l,wu)$class, cex=3)
```

<img width="1052" alt="screen shot 2019-02-23 at 3 28 44 pm" src="https://user-images.githubusercontent.com/43111524/53291355-b810e500-377f-11e9-9761-522b4cc1f0ca.png">
All 11 papers with unknown authorship are predicted by this methodology to have been written by James Madison

Develop cross-validation method to assess the performance of this approach.  Cross-validation evaluates the method for prediction on omitted cases with known authorship:
```R
# Create a vector 'leave', comprised of 5 random paper samples of known authorship
leave <- sample(1:nrow(y), 5)
# The remaining papers become the training set
train <- y[-leave, ]
# Remove 'leave' authors from vector of authorship
b.train <- b[-leave]
# Perform pca on the training set
p <- prcomp(train)
# Perform lda on the training set
l <- lda(as.matrix(train) %*% p$rotation[, 1:40], grouping=b.train)
M <- p$rotation[, 1:40] %*% l$scaling[, 1:2]
train.proj <- as.matrix(train) %*% M
# Plot the resultant lda results (training set)
plot(train.proj, col=b.train)
# Create a test matrix, and multiply it by the output rotation of the pca/lda analysis
test.proj <- as.matrix(y[leave, ]) %*% M
# Plot the points of the test set colored by true authorship
points(test.proj, col=b[leave], cex=2, pch="x")
legend("topright", levels(b.train), text.col=1:length(b.train),
       fill=1:length(levels(b.train)))
# Plot the points of the test set colored by predicted authorship
points(test.proj, col=predict(l, as.matrix(y[leave, ]) %*% p$rotation[, 1:40])$class, cex=3)
```
<img width="1052" alt="screen shot 2019-02-23 at 3 30 00 pm" src="https://user-images.githubusercontent.com/43111524/53291366-e8f11a00-377f-11e9-9791-e384ac98a44b.png">
All ommitted cases were assigned by the methodology to their correct class

Implement the cross-validation method on numerous test cases.
```R
for (i in 1:100) {
  total <- total + 5
  leave <- sample(1:nrow(y), 5)
  train <- y[-leave, ]
  b.train <- b[-leave]
  p <- prcomp(train)
  l <- lda(as.matrix(train) %*% p$rotation[, 1:60], grouping=b.train)
  predicted_authors <- predict(l, as.matrix(y[leave, ]) %*% p$rotation[, 1:60])$class
  actual_authors <- b[leave]
  for (j in 1:length(predicted_authors)) {
    if (predicted_authors[j] == actual_authors[j]) {
      correct <- correct + 1
    }
  }
}
# 'Confidence' is the proportion of test cases that were correctly classified by the model
confidence <- correct/total
confidence
```
99.2% of test cases were correctly classified by the model 


In the final step, I assess the generalizability of the cross-validation results to classification of documents with unknown authorship.  Here, I evaluate the Mahalanobis Distance of the predicted results for papers with unknown authorship against the Mahalanobis Distance of the predicted results for test cases.  If the difference in average Mahalanobis Distances is small, we could assume that the methodology is valid with high confidence:
```R
# DETERMINE THE MEAN VECTOR AND COVARIANCE MATRIX SIGMA FROM EACH  OF THE SAMPLE DISTRIBUTIONS
# Isolate the 2D data points (LD1, LD2) for each author
Hamilton <- a %in% c("Hamilton")
Madison <- a %in% c("Madison")
Jay <- a %in% c("Jay")
y_ham <- x[Hamilton, ]
y_mad <- x[Madison, ]
y_jay <- x[Jay, ]
w_ham <- as.matrix(y_ham) %*% p$rotation[, 1:40]
w_mad <- as.matrix(y_mad) %*% p$rotation[, 1:40]
w_jay <- as.matrix(y_jay) %*% p$rotation[, 1:40]
hamDist <- as.matrix(w_ham) %*% l$scaling[, 1:2]
madDist <- as.matrix(w_mad) %*% l$scaling[, 1:2]
jayDist <- as.matrix(w_jay) %*% l$scaling[, 1:2]
dim(hamDist)
dim(madDist)
dim(jayDist)
# Calculate the mean vector for each author
for (i in 1:2) {
  ham_mean <- c(mean(hamDist[,1]), mean(hamDist[,2]))
  mad_mean <- c(mean(madDist[,1]), mean(madDist[,2]))
  jay_mean <- c(mean(jayDist[,1]), mean(jayDist[,2]))
}
# Calculate the covariance matrix for each author
ham_cov <- cov(hamDist)
mad_cov <- cov(madDist)
jay_cov <- cov(jayDist)
# Determine Mahalanobis Distance of papers with unknown authorship from their predicted clusters. 
unknown <- which(a == "Hamilton or Madison")
zu <- x[unknown, ]
wu <- as.matrix(zu) %*% p$rotation[, 1:40]
wuDist <- as.matrix(wu) %*% l$scaling[, 1:2]
wuDist
Distances <- c()
for (i in 1:nrow(wuDist)) {
  Distances <- append(Distances,t(wuDist[i,]-mad_mean)%*%solve(mad_cov)%*%(wuDist[i,]-mad_mean))
}
# 'Distances' is a vector of squared Mahalanobis distances of points with unknown authorship 
# away from their predicted clusters. (Here, they are all predicted to be part of Madison distribution)
Distances



# Determine average Mahalanobis Distance of test cases (20 iterations) away from their actual cluster
Distances_Mad <- c()
Distances_Ham <- c()
Distances_Jay <- c()
for (i in 1:20) {
  total <- total + 5
  leave <- sample(1:nrow(y), 5)
  leave1 <- y[leave,]
  train <- y[-leave, ]
  b.train <- b[-leave]
  p <- prcomp(train)
  l <- lda(as.matrix(train) %*% p$rotation[, 1:40], grouping=b.train)
  w1 <- as.matrix(train) %*% p$rotation[, 1:40]
  w2 <- as.matrix(leave1) %*% p$rotation[, 1:40]
  ldata <- as.matrix(w1) %*% l$scaling[, 1:2]
  ldata2 <- as.matrix(w2) %*% l$scaling[, 1:2]
  Hamilton <- b.train %in% c("Hamilton")
  Madison <- b.train %in% c("Madison")
  Jay <- b.train %in% c("Jay")
  # Recalculate distributions for each author based on new sample set
  hamDist <- ldata[Hamilton,]
  madDist <- ldata[Madison,]
  jayDist <- ldata[Jay,]
  # Calculate the mean vector for each author
  for (i in 1:2) {
    ham_mean <- c(mean(hamDist[,1]), mean(hamDist[,2]))
    mad_mean <- c(mean(madDist[,1]), mean(madDist[,2]))
    jay_mean <- c(mean(jayDist[,1]), mean(jayDist[,2]))
  }
  # Calculate the covariance matrix for each author
  ham_cov <- cov(hamDist)
  mad_cov <- cov(madDist)
  jay_cov <- cov(jayDist)
  actual_authors <- b[leave]
  for (j in 1:length(actual_authors)) {
    if (actual_authors[j] == "Hamilton") {
      for (k in 1:nrow(ldata2)) {
        Distances_Ham <- append(Distances_Ham,t(ldata2[k,]-ham_mean)%*%solve(ham_cov)%*%(ldata2[k,]-ham_mean))
      }
    }
    if (actual_authors[j] == "Madison") {
      for (k in 1:nrow(ldata2)) {
        Distances_Mad <- append(Distances_Mad,t(ldata2[k,]-mad_mean)%*%solve(mad_cov)%*%(ldata2[k,]-mad_mean))
      }
    }
    if (actual_authors[j] == "Jay") {
      for (k in 1:nrow(ldata2)) {
        DistancesJay <- append(DistancesJay,t(ldata2[k,]-jay_mean)%*%solve(jay_cov)%*%(ldata2[k,]-jay_mean))
      }
    }
  }
}


# 'Distances_Ham,' 'Distances_Mad,' and 'Distances_Jay' are vectors of the squared Mahalanobis distances of test 
# points with known authorship away from their actual clusters. 
# These test poihts were ommitted in the creation of cluster mean and covariance paramters.


mean(Distances)
mean(Distances_Ham)
mean(Distances_Mad)
mean(DistancesJay)
```
The mean of squared Mahalanobis Distances for the papers with unknown authorship (Distances_Ham = 36.8; Distances_Mad = 62.7; Distances_Jay = 3099.7)- at least for the Hamilton and Madison distributions, are quite close to the mean of squared Mahalanobis Distances for the test set (7.4).  This gives me a high degree of confidence that the cross-validation results are generalizable, and that the PCA/LDA classification scheme is robust.

#### Conclusion
All 11 Federalist Papers with contested authorship were almost certainly written by James Madison!



