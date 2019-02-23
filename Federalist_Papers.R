#### Visualizing the Federalist Papers ####



# READ IN FREQUENCIES
# Read in the generated csv of token frequencies for each of the 85 Federalist Papers
# Convert the 'na' values in data matrix to 0
# Display the first 6 rows and columns
x <- read.csv('/Users/dylancicero/Desktop/Data_Analysis/Project1/Federalist_Papers/federalist_frequencies.csv', row.names=1)
x[is.na(x)] <- 0
x[1:6, 1:6]
# xbar is a vector of the mean of frequencies for a given token across all papers
xbar <- apply(x, 2, mean)
# iterate across each row, subtracting the mean for a given token from the frequency value for a given paper
# (not necessary, but centers the axes around data)
for(i in 1:nrow(x)) {
  x[i, ] <- x[i, ] - xbar
}



# READ IN AUTHORS
# Read in csv of authors.  Transform it into a vector listing the authors in order from first to last paper.
a <- read.csv('/Users/dylancicero/Desktop/Data_Analysis/Project1/Federalist_Papers/authors.csv')$X0
# Create a variable "single", a vector of boolean values in order from first to last paper, 
# where TRUE means that the paper had single authorship.
single <- a %in% c("Hamilton", "Madison", "Jay")
# Create a new matrix 'y', extracting the papers (rows) from matrix 'x' for which there is a known single author
y <- x[single, ]
# Develop factors for the three single authors, to be used later in analytic charts
b <- as.factor(as.character(a[single]))



# CONDUCT PRINCIPLE COMPONENT ANALYSIS
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
legend("bottomleft", levels(b), text.col=1:length(levels(b)),
       fill=1:length(levels(b)))
mtext("(means subtracted)", side = 3, adj = 0.5, line = 0.4)

# The first two principle components are able to distinguish the authors with a little overlap
# But we could do better...



# WHAT PROPORTION OF THE VARIANCE IS CAPTURED BY EACH ADDITIONAL PRINCIPLE COMPONENT?
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


# CONDUCT LINEAR DISCRIMINANT ANALYSIS
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



# EVALUATE RESULTS OF LDA
# LDA rotates the axes of the data such that the known groups will by optimally seperated.
# This could be misleading if a random labeling of the data partitions the data as well as the true labeling of the data.
# To quickly contrast true and random labeling of the data, make a null lineup.
par(mfrow = c(3, 3), oma = c(0,0,2,0))
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
mtext("Linear Discriminant Analysis: Null Lineup", outer = TRUE, cex = 1.5)

par(mfrow = c(1, 1))
# It is obvious that one plot stands out from the rest.
# Show the true plot alone
plot(as.matrix(w) %*% l$scaling[, 1:2], col=b, 
     main = "Predicting Authorship of Unknown Papers", ylim = c(-7,7))
legend("topright", levels(b), text.col=1:length(b),
       fill=1:length(levels(b)))

dim(p$rotation)

# TRIAL LDA WITH 60 PRINCIPLE COMPONENTS USED (VS 40)
w <- as.matrix(y) %*% p$rotation[, 1:60]
l <- lda(w, grouping=b)
plot(as.matrix(w) %*% l$scaling[, 1:2], col=b,main = "Linear Discriminant Analysis")
legend("topright", levels(b), text.col=1:length(b),
       fill=1:length(levels(b)))
mtext("(based off 60 principle components)", side = 3, adj = 0.5, line = 0.4)
# Clusters are much tighter here!  But test cases perform worse...



# PREDICTING TRUE AUTHORSHIP OF JOINT OR UNKNOWN PAPERS
# add documents that are attributed to both Hamilton and Madison
both <- which(a == "Hamilton and Madison")
zb <- x[both, ]
wb <- as.matrix(zb) %*% p$rotation[, 1:40]
points(as.matrix(wb) %*% l$scaling[, 1:2], col=4)
# add documents of unknown authorship
unknown <- which(a == "Hamilton or Madison")
zu <- x[unknown, ]
wu <- as.matrix(zu) %*% p$rotation[, 1:40]
text(wu %*% l$scaling[, 1:2], labels=rownames(wu), col=5)
### Discussed in class (predict function)
p <- points(wu %*% l$scaling, col=5, pch="x", cex=2)
points(wu %*% l$scaling, col=predict(l,wu)$class, cex=3)



# DEVELOP CROSS-VALIDATION METHOD TO ASSESS THE PERFORMANCE OF THE APPROACH
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
plot(train.proj, col=b.train, ylim = c(-10,10), main = "Evaluating Methodology via Cross-Validation")
# Create a test matrix, and multiply it by the output rotation of the pca/lda analysis
test.proj <- as.matrix(y[leave, ]) %*% M
# Plot the points of the test set colored by true authorship
points(test.proj, col=b[leave], cex=2, pch="x")
legend("topright", levels(b.train), text.col=1:length(b.train),
       fill=1:length(levels(b.train)))
# Plot the points of the test set colored by predicted authorship
points(test.proj, col=predict(l, as.matrix(y[leave, ]) %*% p$rotation[, 1:40])$class, cex=3)



# IMPLEMENT CROSS-VALIDATION METHOD ON NUMEROUS TEST CASES
correct <- 0
total <- 0
# Iterate over 100 trials, each time removing a random sample of known authors from the training set,
# performing pca/lda, and summing the number of test cases that are correctly predicted by the model.
for (i in 1:100) {
  total <- total + 5
  leave <- sample(1:nrow(y), 5)
  train <- y[-leave, ]
  b.train <- b[-leave]
  p <- prcomp(train)
  l <- lda(as.matrix(train) %*% p$rotation[, 1:40], grouping=b.train)
  predicted_authors <- predict(l, as.matrix(y[leave, ]) %*% p$rotation[, 1:40])$class
  actual_authors <- b[leave]
  for (j in 1:length(predicted_authors)) {
    if (predicted_authors[j] == actual_authors[j]) {
      correct <- correct + 1
    }
  }
}
# 'Confidence' is the proportion of test cases that were correctly classified by the model
confidence <- correct/total
print(confidence)
confidence


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



# ASSESS THE GENERALIZABILTY OF THE CROSS-VALIDATION RESULTS TO CLASSIFICATION OF DOCUMENTS WITH UNKNOWN AUTHORSHIP
# HERE, I EVALUATE THE MAHALANOBIS DISTANCE OF THE PREDICTED RESULTS FOR PAPERS WITH UNKNOWN AUTHORSHIP
# AGAINST THE MAHALANOBIS DISTANCE OF THE PREDICTED RESULTS FOR TEST CASES
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

# The mean of squared Mahalanobis Distances for the papers with unknown authorship is actually lower than the mean
# of squared Mahalanobis Distances for the test set (likely due to higher sample size of test set), giving me a high
# degree of confidence that the cross-validation results are generalizable, and that the PCA/LDA classification scheme
# is robust.






