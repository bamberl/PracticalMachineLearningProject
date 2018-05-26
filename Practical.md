---
title: "Practical machine learning course project"
output: github_document  
        
---
#Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict . They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. 

Since the outcome variable is a "factor" I will treat the problem as one of "classification"i.e of predicting whether a movement is A,B,C,D or E. Therefore, I  will use the caret methods of decision trees ("rpart"") and random forests ("rf"). In the random forrest model I will use cross validation with k fold sampling to reduce overfitting.


```
## Warning: package 'dplyr' was built under R version 3.4.3
```

```
## 
## Attaching package: 'dplyr'
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```
## Warning: package 'caret' was built under R version 3.4.4
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Want to understand how all the pieces fit together? Buy the
## ggplot2 book: http://ggplot2.org/book/
```

```
## Warning: package 'rattle' was built under R version 3.4.4
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.1.0 Copyright (c) 2006-2017 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```
## Loading required package: foreach
```

```
## foreach: simple, scalable parallel programming from Revolution Analytics
## Use Revolution R for scalability, fault tolerance and more.
## http://www.revolutionanalytics.com
```

```
## Loading required package: iterators
```

```
## Warning: package 'randomForest' was built under R version 3.4.4
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:rattle':
## 
##     importance
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```
## The following object is masked from 'package:dplyr':
## 
##     combine
```

```
## Warning: package 'ggraph' was built under R version 3.4.4
```

```
## Warning: package 'igraph' was built under R version 3.4.4
```

```
## 
## Attaching package: 'igraph'
```

```
## The following objects are masked from 'package:dplyr':
## 
##     as_data_frame, groups, union
```

```
## The following objects are masked from 'package:stats':
## 
##     decompose, spectrum
```

```
## The following object is masked from 'package:base':
## 
##     union
```


#1. Load raw data


```r
setwd("C:/Users/lb858473/Desktop")
rawtraining<- read.csv("pml-training.csv", , na.strings=c("", "NA"))
rawtraining$cvtd_timestamp<- as.POSIXct(rawtraining$cvtd_timestamp, format = "%d/%m/%Y %H:%M")
rawtesting<- read.csv("pml-testing.csv")
```

#2. Exploratory data analysis

From the exploratory data analysis I will make a number of key assumptions about the predictor variables that should be included in the model.  

##2.1 Treatment of NAs and Blanks

The raw data seems to contain a large number of descriptive statistics ("avg", "min", "max") that would have a lot of blanks("") and NAs. These will be removed from the training set to leave only the readings from the equipment. 


```r
training<- rawtraining[, colSums(is.na(rawtraining)) == 0] 
testing<- rawtesting[, colSums(is.na(rawtraining)) == 0] 
```


##2.2 Time series check

Though the data does contain "date" stamps for the observations, there is no code book and time related variables cannot be easily interpreted (there is only one timestamp per user). We will therefore assume that the data is not a time series and we will remove all variables that seem time- related. 


```r
training<- select(training, -raw_timestamp_part_1, -raw_timestamp_part_2,-cvtd_timestamp, -num_window, -new_window)
testing<- select(testing, -raw_timestamp_part_1, -raw_timestamp_part_2,-cvtd_timestamp, -num_window, -new_window)
```

##2.3 User_name analysis

An exploratory analysis of readings by user, shows that user could be significant. The yaw of the belt varies significantly by user whether the motion is performed correctly or not. Charles and Euricos readings are between +100 and -20. Carlitos readings are between -7. Therefore user_name cannot be excluded as a predictor


```r
ggplot(training,aes(x=X,y=yaw_belt, group=classe, color=user_name)) + 
        geom_point() + 
        facet_grid(~ classe) +
        theme_classic() 
```

![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5-1.png)

##2.4 Data visualisation

Even with the reduced number of variables (160>53) the data is highly dimensional and all the plots cannot be presented in one feature plot. Therefore we need to be able to chunk the predictor variables into smaller feturePlots. Below is a sample plot of four variables that seem to have some predictive value.   


```r
featurePlot(x = training[, c(3,5,8,55)], 
            y = training$classe, 
            plot = "pairs")
```

![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-6-1.png)

For example, it looks like "pitch_belt"  and "total_accel_belt" are predictive of classe "E"" and "gyros_belt_z" is predictive of classe= "D". However it is very difficult to visually select all the appropriate predictors. So I will now use two machine learning algorithms to develop better models.  

Note: The index (X) does not have any predictive value, it was used purely for graphing purposes, so it will now be removed from the training and test sets.


```r
training<- select(training, -X)
testing<- select(testing, -X)
```

#3.Cross validation

Cross validaton can be used to reduce overfitting and thereby to reduce "out of sample" errors.The number of observations are realtively large and I will split the training set into "Training" and test in order  to test the acccuracy of the predicton.


```r
set.seed(123)
cv <- createDataPartition(y=training$classe, p=0.6, list=FALSE)
training <- training[cv, ] 
trainingTest <- training[-cv, ]
```


#4. Preprocessing

In order to remove outliers and to reduce the "noise" in the data I will standardise the training, test and validation data sets. 


```r
preObj<-preProcess(training, c("center","scale"), verbose = TRUE)
```

```
## Calculating 52 means for centering
## Calculating 52 standard deviations for scaling
```

```r
scaledtraining<- predict(preObj, training)
scaledtrainingTest<- predict(preObj, trainingTest)
scaledTest<- predict(preObj, testing)
```

#5. Select model 

##5.1 Use decision tree with no cross validation


```r
set.seed(123)
modelFitDt<- train(classe~.,data=scaledtraining,method="rpart")
print(modelFitDt$finalModel)
```

```
## n= 11776 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 11776 8428 A (0.28 0.19 0.17 0.16 0.18)  
##    2) roll_belt< 1.051305 10776 7434 A (0.31 0.21 0.19 0.18 0.11)  
##      4) pitch_forearm< -1.585983 956    7 A (0.99 0.0073 0 0 0) *
##      5) pitch_forearm>=-1.585983 9820 7427 A (0.24 0.23 0.21 0.2 0.12)  
##       10) yaw_belt>=1.899936 478   47 A (0.9 0.046 0 0.042 0.01) *
##       11) yaw_belt< 1.899936 9342 7092 B (0.21 0.24 0.22 0.2 0.12)  
##         22) magnet_dumbbell_z< -1.000801 1104  447 A (0.6 0.28 0.043 0.058 0.023) *
##         23) magnet_dumbbell_z>=-1.000801 8238 6231 C (0.16 0.24 0.24 0.22 0.14)  
##           46) pitch_belt< -1.93658 491   75 B (0.014 0.85 0.084 0.026 0.029) *
##           47) pitch_belt>=-1.93658 7747 5781 C (0.17 0.2 0.25 0.24 0.15)  
##             94) magnet_dumbbell_y< 0.2077132 3437 2046 C (0.19 0.12 0.4 0.16 0.12) *
##             95) magnet_dumbbell_y>=0.2077132 4310 3032 D (0.15 0.26 0.13 0.3 0.16) *
##    3) roll_belt>=1.051305 1000    6 E (0.006 0 0 0 0.99) *
```

```r
fancyRpartPlot(modelFitDt$finalModel)
```

![plot of chunk unnamed-chunk-10](figure/unnamed-chunk-10-1.png)

```r
dtFitPredict <- predict(object = modelFitDt, newdata = scaledtraining)
confusionMatrix(dtFitPredict, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2037  340   47   84   30
##          B    7  416   41   13   14
##          C  668  401 1391  555  422
##          D  630 1122  575 1278  705
##          E    6    0    0    0  994
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5194          
##                  95% CI : (0.5103, 0.5284)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4023          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.6084  0.18254   0.6772   0.6622  0.45912
## Specificity            0.9406  0.99210   0.7895   0.6921  0.99938
## Pos Pred Value         0.8026  0.84725   0.4047   0.2965  0.99400
## Neg Pred Value         0.8581  0.83491   0.9205   0.9127  0.89133
## Prevalence             0.2843  0.19353   0.1744   0.1639  0.18385
## Detection Rate         0.1730  0.03533   0.1181   0.1085  0.08441
## Detection Prevalence   0.2155  0.04169   0.2919   0.3660  0.08492
## Balanced Accuracy      0.7745  0.58732   0.7334   0.6771  0.72925
```

The decision tree has only gives an accuracy of 51% and the algorithm is performing poorly 

##5.2 Use random forest

Random forest models are one of the more accurate methods used in Machine Learning projects and I will test to see if they have a better predictive value for the data set under consideration.  

###5.2.1 Configure parallel processing

The random forest methods require a lot of computer power and it is often necessary to use parralel processing


```r
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
```

###5.2.2 Configure trainControl object

The most critical arguments for the trainControl function are the resampling method, in this case "cross validation" is used and the "number"" that specifies the quantity of folds for k-fold cross-validation.


```r
fitControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE)
```

###5.2.3 Train random forrest model


```r
set.seed(123)
modfitRf<- train(classe~., method="rf", data=scaledtraining, trControl = fitControl)
```

###5.2.4 De-register parallel processing cluster


```r
stopCluster(cluster)
registerDoSEQ()
```

#5.2.5 Evaluate in sample model accuracy


```r
print(modfitRf)
```

```
## Random Forest 
## 
## 11776 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 9421, 9420, 9421, 9420, 9422 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9877713  0.9845281
##   29    0.9888756  0.9859268
##   57    0.9832711  0.9788381
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 29.
```

Using the random forrest model with cros validation has greatly increased the  in sample model accuracy is very good at 97.8%

#5.2.6 Evalute out of sample model accuracy

A disdvantage of the random forrest method is that it is prone to overfitting, and we would expect the out of sample model accuracy to be lower than the in sample model accuracy.


```r
rfFitPredict <- predict(object = modfitRf, newdata = scaledtrainingTest)
confusionMatrix(rfFitPredict, scaledtrainingTest$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1344    0    0    0    0
##          B    0  908    0    0    0
##          C    0    0  836    0    0
##          D    0    0    0  752    0
##          E    0    0    0    0  870
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9992, 1)
##     No Information Rate : 0.2854     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2854   0.1928   0.1775   0.1597   0.1847
## Detection Rate         0.2854   0.1928   0.1775   0.1597   0.1847
## Detection Prevalence   0.2854   0.1928   0.1775   0.1597   0.1847
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

However, when the test data is used to predict the model accuracy, the result is 100% out of sample model accuracy.

#Conclusion



With high dimension data it is very difficult to visually identify what all the predictors are and therefore to develop a highly accurate model. 

Machine learning offers a number of models that offer higher accuracy. decision trees identify multiple predictors but the accuracy is not good. Using cross validation to boost the modelling to create random forrests greatly improve the accuracy of the model. But the processing speeds is slow (and unusable) if parralel processing is not used.  
