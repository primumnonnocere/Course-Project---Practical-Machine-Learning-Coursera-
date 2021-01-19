---
title: "Course Project - Practical Machine Learning (Coursera)"
date: "1/10/2021"
output: html_document
editor_options: 
  chunk_output_type: console
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

```{r packages, include=FALSE}
library(foreach); library(parallel);library(doParallel);library(tidyverse);library(caret);library(data.table);library(corrplot);library(rattle);library(gbm)
```
## Course Project - Practical Machine Learning (Coursera)

The goal of this course project is to demonstrate the use of developed skills from the "Practical Machine Learning" course (Part of the Data Science Specialization offered by JHU through Coursera) in a practical data assessment. The dataset to be analyzed was assess by Velloso et al (2013), which assessed six individuals engaged in a series of weight lifting exercises (WLE), with several actigraphy and motor axis variables constituting the breakdown of these WLE. This accelerometer data was collected from four different locations on the participants; at the forearm, arm, belt, and on the barbell being used in the WLE. These barbell exercises were conducted correctly and incorrectly in five different ways. The dataset, as well as further details related to its collection process and the assessed variables, can be found at the following website: <http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>  

For the purposes of this project, we were asked to use machine learning (ML) techniques in analyzing both a training and testing set, the latter of which is to be submitted in quiz format. My strategy in assessing the training dataset was to construct a primary training and testing subset, including the use of cross-validation techniques (CV), across three primary types of potential models; random forest (rf), decision-tree, and a boosted tree variant (gbm in this case). Next, I constructed prediction models for the designated "testing" subject in order to assess predictive respective accuracies of each model, selected the best performing of the three, and finally used this model to assess the bonafide "testing" set that was to be reported in the quiz. Below is the code which accomplished these aforementioned tasks. 

Although not shown in the code, the following packages are required in order to complete the following commands as shown below. Please include "install.package()" and "library()" commands for each of the follow packages: **foreach, parallel, doParallel, tidyverse, caret, data.table, corrplot, rattle, gbm, and ranger**.


### Download Dataset
Here, we are just downloading the dataset, checking the dimensions, making sure that all variables (besides the outcome variable, **classe**, which will be assigned as a "factor" class) are numeric/integers, and finally make a basic split between the training (comprising 70% of the data) and the testing subsets (comprising the final 30%).  

```{r download_dataset, echo=T}
training_dl <- download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "pml-training.csv",mode="wb")
testing_dl <- download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "pml-testing.csv",mode="wb")
data_training <- fread("pml-training.csv", header = T, na.strings=c("","NA"))
data_testing <- fread("pml-testing.csv", header = T, na.strings=c("","NA")) 
dim(data_training)
dim(data_testing)
str(data_training$classe)
data_training$classe <- as.factor(data_training$classe)
dt <- as.data.frame(data_training)
inTrain <- createDataPartition(y=dt$classe,p=0.7, list=FALSE)
training <- data_training[inTrain,]
testing <- data_training[-inTrain,]
```

### Data cleaning, Correlation analysis
Next, ML techniques require that we have a tidy dataset comprised of full columns and rows. To assess our dataset in its crude state, we construct a table to see if there are any variables with considerable amounts of missing values. 
```{r Data Cleaning pt.1}
percent_na <- colMeans(is.na(training))

table(percent_na)
qplot(percent_na)
```

From the above table, it is clear that there is a bimodal distribution in column percentages of missing data, with many comprised almost completely with missing data, and a few remainders with almost no missing data. These are the variables we want to preserve and conduct ML on, which is what we accomplished in the subsequent code.  

```{r Data Cleaning pt.2}
training_cut <-filter(subset(training,select=colMeans(is.na(training))<0.75))
```
Next, we decided to screen for variables with virtually zero variance in their values, as these will likely not be useful to include in our ML models.
```{r nzv}
nsv <- nearZeroVar(training_cut,saveMetrics=T)
nsv_remove <- nsv %>% filter(nzv==T)
nsv_remove
training_cut2 <- training_cut %>% select(-new_window)
```
From the above assessment, only one variable (**new_window**) met this threshold and was consequently removed from the dataset.  

Finally, we need to remove all time and identifier variables, as these will not be analyzable by our ML models. Columns 1-5 meet this qualification, and are therefore removed from the ML dataset.
```{r ml dataset cleaning}
table(sapply(training_cut2, class))
training_cut3 <- training_cut2[, -(1:5)]
dim(training_cut3)
table(sapply(training_cut3, class))
```
With the above table command, we confirm that all rows in the above dataset appear to be the correct type, which suggests that this dataset can now be assessed with ML. From the dim command, we see that our original dataset with 160 variables has now been reduced to 54.  

Next, we want to take a brief exploratory look at the correlations between all of these variables, just to see if there is anything particularly interesting that might emerge. 

```{r correlation}
covariate_matrix <- training_cut3 %>% select(-classe) %>% cor()
corrplot(covariate_matrix,type="lower", tl.cex= .4, tl.srt=45)
```

From my eye, nothing really pops out immediately, so I then proceed to the standard process of ML.

Below are the models for each of the respective types: random forest, decision tree, and tree boosting. Each model is repeated with minimal changes across validation techniques and additional default values in order to focus on their comparative performances regarding prediction accuracy.

### ML - random forest (rf)
**NOTE** For the rf calculations below, I deviated somewhat from the course in that I sought out a more modern R package that could leverage the use of parallel processing for the random forest computation in order to speed up analysis time; the package I ended up using is titled "ranger" (described as method="ranger" in the command below). Note that the base "rf" method which comes with the caret package will arrive at the same accuracy and prediction values (which, although not shown, was repeated on my dataset in order to confirm this assertion.)

```{r trainingtest 1_rf}
set.seed(11111)
rf_fit <- train(classe ~., method="ranger", data=training_cut3, trControl=trainControl(method="cv", number=3, verboseIter=FALSE))
confusionMatrix(testing$classe, predict(rf_fit,testing, predict.all=TRUE))
```

### ML - decision-tree (rpart)
```{r trainingtest 2_tree}
tree_fit <- train(classe ~ .,method="rpart",data=training_cut3,trControl=trainControl(method="cv", number=3, verboseIter=FALSE))

fancyRpartPlot(tree_fit$finalModel)
confusionMatrix(testing$classe, predict(tree_fit,testing))
```

### ML - tree boosted (gbm)
```{r trainingtest 3_gbm}
gbm_fit <- train(classe ~ .,method="gbm",data=training_cut3,trControl=trainControl(method="cv", number=3, verboseIter=FALSE), verbose=FALSE)
confusionMatrix(testing$classe, predict(gbm_fit,testing))
```

### Assessment of ML model types
From the reported accuracy values of these models, it  **random forest** is the best performing with 99.58% accuracy (although the gbm model performed almost as well at 98.93%; meanwhile, the decision tree ML model performed much worse at only 49.36% accuracy). Therefore, we will use the rf model in assessing the final testing set so as to predict the "classe" for each row.

### Expected out-of-sample error (selected rf model)
Given our selection of the rf model, we can anticipate that the expected out-of-sample error would be the difference between 1 and our accuracy, 0.9958; 1-0.9958 =0.0042. Given that there are only 20 samples in the test set used for the quiz, it is quite likely that we will get all (or nearly all) of the predictions correct.

### Quiz Submission - Test set prediction
```{r quiz prediction}
prediction_quiz <- predict(rf_fit, newdata=data_testing)

classe<-as.matrix(prediction_quiz)
prediction_quiz <- as.factor(prediction_quiz)

quiz_data <- data_testing %>% select(1) 
df_quiz <- data.frame(quiz_data, classe)
df_quiz
```

Above are the predictions for each of the datapoints in the original test dataset which were used for answering the requested quiz questions.