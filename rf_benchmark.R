# makes the random forest submission

library(randomForest)

train <- read.csv("F:/Coursera/DataScience/Titanic/trainClean.csv", header=FALSE)
test <- read.csv("F:/Coursera/DataScience/Titanic/testCleanNoLabels.csv", header=FALSE)

labels <- as.factor(train[,1])
train <- train[,-1]

rf <- randomForest(train, labels, xtest=test, ntree=100)
predictions <- levels(labels)[rf$test$predicted]

write(predictions, file="F:/Coursera/DataScience/Titanic/rf_titanic_pred.csv", ncolumns=1) 
