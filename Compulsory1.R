# Compulsory 1 - SL

# Packages
install.packages("knitr") # probably already installed
install.packages("rmarkdown") # probably already installed
install.packages("ggplot2") # plotting with ggplot2
install.packages("dplyr") # for data cleaning and preparation
install.packages("tidyr") # also data preparation
install.packages("carData") # dataset
install.packages("class") # for KNN
install.packages("pROC") # calculate roc
install.packages("plotROC") # plot roc
install.packages("ggmosaic") # mosaic plot
install.packages('caret')

# libraries
library(MASS)
library(class)
library(pROC)
library(plotROC)
library(caret)
########## Problem 01
id <- "1X_8OKcoYbng1XvYFDirxjEWr7LtpNr1m" # google file ID
values <- dget(sprintf("https://docs.google.com/uc?id=%s&export=download", id))

X <- values$X
dim(X)

x0 <- values$x0
dim(x0)

beta <- values$beta
dim(beta)

sigma <- values$sigma
sigma

# e)
library(ggplot2)
# Squared bias function
bias <- function(lambda, X, x0, beta) {
  p <- ncol(X)
  value <- ( t(x0) %*% beta - t(x0)%*% solve(t(X)%*%X+lambda*diag(p))%*%t(X)%*%X%*%beta )^2
  return(value)
}
lambdas <- seq(0, 2, length.out = 500)
BIAS <- rep(NA, length(lambdas))
for (i in seq_along(lambdas)) BIAS[i] <- bias(lambdas[i], X, x0, beta)
dfBias <- data.frame(lambdas = lambdas, bias = BIAS)
ggplot(dfBias, aes(x = lambdas, y = bias)) +
  geom_line(color = "hotpink") +
  xlab(expression(lambda)) +
  ylab(expression(bias^2))

# f)
# Variance function
variance <- function(lambda, X, x0, sigma) {
  p <- ncol(X)
  inv <- solve(t(X) %*% X + lambda * diag(p))
  value <- sigma^2 * (t(x0) %*% inv %*% t(X) %*% X %*% inv %*% x0)
  return(value)
}
lambdas <- seq(0, 2, length.out = 500)
VAR <- rep(NA, length(lambdas))
for (i in seq_along(lambdas)) VAR[i] <- variance(lambdas[i], X, x0, sigma)
dfVar <- data.frame(lambdas = lambdas, var = VAR)
ggplot(dfVar, aes(x = lambdas, y = var)) +
  geom_line(color = "gold") +
  xlab(expression(lambda)) +
  ylab("variance")

# g)
# Expected MSE
exp_mse <- BIAS + VAR + sigma^2
lambdas[which.min(exp_mse)] # gives optimal value of lambda for this problem

dfexp_mse <- data.frame(lambdas = lambdas, bias = BIAS, var = VAR, exp_MSE = exp_mse)
dfexp_mse
ggplot(dfexp_mse, aes(x = lambdas, y = exp_MSE))+
  geom_line(aes(x=lambdas, y=exp_MSE), dfexp_mse, color = "red")+
  geom_line(aes(x=lambdas, y=bias), dfexp_mse, color="hotpink") +
  geom_line(aes(x=lambdas, y=var), dfexp_mse, color = "gold") +
  geom_line(aes(x=lambdas, y=rep(sigma^2, 500)),dfexp_mse, color="green") +
  geom_vline(xintercept = lambdas[which.min(exp_mse)])

########## Problem 03
bigfoot_original <- readr::read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-09-13/bigfoot.csv")
library(dplyr)

# Prepare the data:
bigfoot <- bigfoot_original %>%
  # Select the relevant covariates:
  dplyr::select(classification, observed, longitude, latitude, visibility) %>%
  # Remove observations of class C (these are second- or third hand accounts):
  dplyr::filter(classification != "Class C") %>%
  # Turn into 0/1, 1 = Class A, 0 = Class B:
  dplyr::mutate(class = ifelse(classification == "Class A", 1, 0)) %>%
  # Create new indicator variables for some words from the description:
  dplyr::mutate(fur = grepl("fur", observed),
                howl = grepl("howl", observed),
                saw = grepl("saw", observed),
                heard = grepl("heard", observed)) %>%
  # Remove unnecessary variables:
  dplyr::select(-c("classification", "observed")) %>%
  # Remove any rows that contain missing values:
  tidyr::drop_na()

set.seed(2023)
# 70% of the sample size for training set
training_set_size <- floor(0.7 * nrow(bigfoot))
train_ind <- sample(seq_len(nrow(bigfoot)), size = training_set_size)
train <- bigfoot[train_ind, ]
test <- bigfoot[-train_ind, ]

# a (Logistic regression)
# i
glm.BigFoot = glm(class ~ longitude + latitude + visibility + fur + howl + saw + heard, data=train, family="binomial")
summary(glm.BigFoot)
glm.probs_BigFoot = predict(glm.BigFoot, newdata = test, type = "response")
glm.preds_BigFoot = ifelse(glm.probs_BigFoot > 0.5, "1", "0")
confusionMatrix(as.factor(glm.preds_BigFoot),reference=as.factor(test$class),positive="1")
# Results is in logit -> convert to odd ratio just exponent it:
# NB. To convert logit to prob -> exp(logit)/(1+exp(logit))
exp(coef(glm.BigFoot)) 
# Classified as class=1: 299 
# Fraction of correct answer = (323+299)/912=0.68201

# ii
glm.BigFootNoSaw = glm(class ~ longitude + latitude + visibility + fur + howl + heard, data=train, family="binomial")
summary(glm.BigFootNoSaw)
glm.probs_BigFootNoSaw = predict(glm.BigFootNoSaw, newdata = test, type = "response")
glm.preds_BigFootNoSaw = ifelse(glm.probs_BigFootNoSaw > 0.5, "1", "0")
confusionMatrix(as.factor(glm.preds_BigFootNoSaw),reference=as.factor(test$class),positive="1")
exp(coef(glm.BigFoot))
# Classified as class=1: 310 
# Fraction of correct answer = (297+310)/912=0.66557

# b (QDA)
# i
qda.BigFoot = qda(class ~ longitude + latitude + visibility + fur + howl + saw + heard, data=train)
summary(qda.BigFoot)
qda.BigFoot_pred = predict(qda.BigFoot, newdata = test)$class
qda.BigFoot_prob = predict(qda.BigFoot, newdata = test)$posterior
confusionMatrix(as.factor(qda.BigFoot_pred),reference=as.factor(test$class),positive="1")
# Classified as class=1: 389

# ii
# 

# c (KNN)
# i
knnMod <- knn(train = train, test = test, cl = train$class , k=25, prob=TRUE)
#table(knnMod, test$class)
#knnMod.BigFoot_prob = attributes(knnMod)$prob
knnMod.BigFoot_prob = ifelse(knnMod == 0,
                             1 - attributes(knnMod)$prob,
                             attributes(knnMod)$prob)
confusionMatrix(as.factor(knnMod),reference=as.factor(test$class),positive="1")

# ii
#

# d
# i
# ii
print("Confusion matrix, sensitivity and specificity of Logistic Regression:")
confusionMatrix(as.factor(glm.preds_BigFoot),reference=as.factor(test$class),positive="1")
print("Confusion matrix, sensitivity and specificity of QDA:")
confusionMatrix(as.factor(qda.BigFoot_pred),reference=as.factor(test$class),positive="1")
print("Confusion matrix, sensitivity and specificity of Knn (K=25):")
confusionMatrix(as.factor(knnMod),reference=as.factor(test$class),positive="1")
# iii
# ROC curve
glmroc = roc(response= test$class, predictor = glm.probs_BigFoot, direction="<", levels=c(0,1))
qdaroc = roc(response= test$class, predictor = qda.BigFoot_prob[,2], direction="<", levels=c(0,1))
knnroc = roc(response= test$class, predictor = knnMod.BigFoot_prob, direction="<", levels=c(0,1))
print("ROC curve")
data = data.frame(Class = test$class, glm = glm.probs_BigFoot, qda = qda.BigFoot_prob[,2], knn = knnMod.BigFoot_prob)
data_log = melt_roc(data, "Class", c("glm", "qda", "knn"))
ggplot(data_log, aes(d = D, m = M, color = name)) + geom_roc(n.cuts = F) + xlab("1-Specificity") + ylab("sensitivity")
print("AUC of Logistic Regression:")
auc(glmroc)
print("AUC of QDA:")
auc(qdaroc)
print("AUC of Knn:")
auc(knnroc)




