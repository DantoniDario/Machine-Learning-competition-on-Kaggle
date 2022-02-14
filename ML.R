# STACKING METHOD APPLICATION (FIRST TRY)

# Packages imported
library(caret)
library(tidyverse)
library(e1071)
library(randomForest)
library(keras)
library(tensorflow)
library(xgboost)
library(plyr)
library(glmnet)
library(Matrix)
Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "1")

# Import datasets
dataset <- read.csv("~/Desktop/maban-2020-mlcompetition/train.csv", header=TRUE)
test <- read.csv("~/Desktop/maban-2020-mlcompetition/test.csv", header=TRUE)

# Check dataset structure
str(dataset)
str(test)

# Clean
# Missing values on categorical variables will be treated as a new category "na"
dataset[,c(2:5,12,17)] <- lapply(dataset[,c(2:5,12,17)], factor)
test[,c(3:6,13)] <- lapply(test[,c(3:6,13)], factor)

# Split the X's component with the y
x <- dataset[,1:16]
y <- dataset[,17]
x_test <- test[,2:17]

# I will use 5 folds cross-validation to tune my models
control <- trainControl(method='cv', number=5)

set.seed(123)

############################# BASE LEARNERS #############################

# Random Forest
tunegrid_rf <- expand.grid(.mtry = seq(sqrt(ncol(x))-2, sqrt(ncol(x))+2))
rf_model <- train(y~., 
                  data=dataset, 
                  method='rf', 
                  metric='Accuracy', 
                  tuneGrid=tunegrid_rf, 
                  trControl=control)

# If you want details about the CV process, use the two codes below
print(rf_model)
plot(rf_model)

# Train all the data (without CV) with the selected tuning parameters
rf_final_model <- train(y~., 
                        data=dataset, 
                        method='rf', 
                        metric='Accuracy', 
                        tuneGrid= expand.grid(.mtry = rf_model$bestTune$mtry))

pred_rf = predict(rf_final_model, x)


# Extrem Gradient Boosting
tunegrid_xgb <- expand.grid(nrounds = 460,
                            max_depth = 7,
                            eta = 0.03,
                            gamma = 1.5,
                            colsample_bytree = 1,
                            min_child_weight = 0,
                            subsample = 0.75)
xgb_model <- train(y~., 
                   data=dataset, 
                   method='xgbTree', 
                   metric='Accuracy', 
                   tuneGrid=tunegrid_xgb, 
                   trControl=control)

# Train all the data (without CV) with the selected tuning parameters
xgb_final_model <- train(y~., 
                         data=dataset, 
                         method='xgbTree', 
                         metric='Accuracy', 
                         tuneGrid= expand.grid(nrounds = xgb_model$bestTune$nrounds,
                                               max_depth = xgb_model$bestTune$max_depth,
                                               eta = xgb_model$bestTune$eta,
                                               gamma = xgb_model$bestTune$gamma,
                                               colsample_bytree = xgb_model$bestTune$colsample_bytree,
                                               min_child_weight = xgb_model$bestTune$min_child_weight,
                                               subsample = xgb_model$bestTune$subsample))

pred_xgb = predict(xgb_final_model, x)


# Neural network
remove(nn_model)
nn_model <- keras_model_sequential() 
y_Keras <- to_categorical(y, 2)
dummy <- dummyVars(" ~ .", data=x)
final_df <- data.frame(predict(dummy, newdata=x))
x_Keras <- as.matrix(final_df)
x_test_Keras <- as.matrix(data.frame(predict(dummy, newdata=x_test)))

nn_model %>% 
  layer_dense(units = 120, activation = 'relu', input_shape = c(37), kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dropout(rate = 0.25) %>%  
  layer_dense(units = 80, activation = 'relu',  kernel_regularizer = regularizer_l2(l = 0.001)) %>%  
  layer_dropout(rate = 0.4) %>%  
  layer_dense(units = 60, activation = 'relu',  kernel_regularizer = regularizer_l2(l = 0.001)) %>%  
  layer_dropout(rate = 0.4) %>%  
  layer_dense(units = 2, activation = 'softmax')

nn_model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = 'accuracy'
)
history <- nn_model %>% fit(x_Keras, y_Keras,   epochs = 200, batch_size = 100,   validation_split = 0.2,  verbose = 0)
plot(history)

pred_nn = nn_model %>% predict_classes(x_Keras)
pred_nn = as.factor(pred_nn)

############################# META LEARNER #############################

# Stacking models together
predDF <- data.frame(RF = pred_rf, XGB = pred_xgb, NN = pred_nn, real_answer = y)

# alpha = 0 stands for Ridge (alpha = 1 for Lasso)
tunegrid_lmRidge <- expand.grid(alpha = 0, lambda = 10^seq(-3, 3, length = 100))

lmRidge_model <- train(real_answer~., 
                       data=predDF, 
                       method='glmnet', 
                       metric='Accuracy', 
                       tuneGrid=tunegrid_lmRidge, 
                       trControl=control)

# Train all the data (without CV) with the selected tuning parameters
lmRidge_final_model <- train(real_answer~., 
                             data=predDF, 
                             method='glmnet', 
                             metric='Accuracy', 
                             tuneGrid= expand.grid(alpha = 0, lambda = lmRidge_model$bestTune$lambda))

# Pass the test dataset in the base learners
rf_pred_test = predict(rf_final_model, x_test)
xgb_pred_test = predict(xgb_final_model, x_test)
nn_pred_test = nn_model %>% predict_classes(x_test_Keras)
nn_pred_test = as.factor(nn_pred_test)

# Then in the meta learner
predDF_test <- data.frame(RF =  rf_pred_test, XGB =  xgb_pred_test, NN = nn_pred_test)

# Final predicted y
pred_lmRidge = predict(lmRidge_final_model, predDF_test)
lmRidge_final_model$finalModel$beta

# Save it in a csv file
write.csv(data.frame(ID=1:4263, y=pred_lmRidge), file='stacking.csv', row.names=FALSE)

