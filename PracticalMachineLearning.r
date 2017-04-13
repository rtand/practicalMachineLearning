#load  libs
library(tidyverse)
library(caret)
library(randomForest)
library(e1071)
library(kernlab)
library(doParallel)

#cl <- makeCluster(detectCores())
cl <- makeCluster(3)

registerDoParallel(cl)

#get project data
url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/"
naStrings <- c("''","NA","#DIV/0!","")
train<-read.csv(sprintf("%spml-training.csv", url), na.strings = naStrings)
test<-read.csv(sprintf("%spml-testing.csv", url), na.strings = naStrings)

#subset the training data
inTrain <- createDataPartition(y=train$classe, p=0.6, list=FALSE)
training <- train[inTrain, ]
testing <- training[-inTrain, ]

#removing row_id, artifact of the csv file
training <- select(training, -X, -raw_timestamp_part_1, -raw_timestamp_part_2, -cvtd_timestamp)

#removing near zero var
nearZeroVars <- nearZeroVar(training, saveMetrics=TRUE)
nearZeroVars$feature <- rownames(nearZeroVars) 
nearZeroVars <- nearZeroVars %>% filter(nzv==FALSE)
training <- training[nearZeroVars$feature]

#removing columns with over 60% NA
lowNA <- sapply(training, function(x) sum(is.na(x)/nrow(training))) %>% as.data.frame()
colnames(lowNA)=c("percentNA")
lowNA$feature <- rownames(lowNA)
lowNA<- filter(lowNA, percentNA<0.60)
training <- training[lowNA$feature]

#propogating transfomations to other sets
testing <- testing[colnames(training)]
test <- test[colnames(training[,-55])]

#Using Random Forest
rf_model<-train(classe~.,data=training,method="rf",
                trControl=trainControl(method="repeatedcv",number=5),
                prox=TRUE,allowParallel=TRUE,
                preProc=c("center", "scale"))
print(rf_model)
print(rf_model$finalModel)

testing$prediction<-predict(rf_model, newdata = select(testing, -classe))
testing<-mutate(testing, ifelse(classe==prediction,1,0))

## Run it on the
testPrediction<-predict(rf_model, newdata = test)
