library(caTools)
library(randomForest)
library(ggplot2)


df_train <- read.csv('train.csv')
df_test <- read.csv('test.csv')
df_test$Survived <- NA
df <- rbind(df_train,df_test)

summary(df)

lapply(df,class)
View(df)

##Check the structure
str(df)

###is there any Missing obesrvation
colSums(is.na(df))

####Empty data
colSums(df=='')

###Imputing Embarked
table(df$Embarked)
df$Embarked[df$Embarked == ''] <- 'S'

FACTOR_VARIABLES = c("Survived","Pclass","Sex","Embarked")
df[, FACTOR_VARIABLES] <- as.data.frame(sapply(df[, FACTOR_VARIABLES], as.factor))
str(df)

full_titanic <- df[1:891,]
###Visualize P class which is the best proxy for Rich and Poor  

ggplot(full_titanic[1:891,],aes(x = Pclass,fill=factor(Survived))) +
  geom_bar() +
  ggtitle("Pclass v/s Survival Rate")+
  xlab("Pclass") +
  ylab("Total Count") +
  labs(fill = "Survived")  

# Visualize the 3-way relationship of sex, pclass, and survival
ggplot(full_titanic[1:891,], aes(x = Sex, fill = Survived)) +
  geom_bar() +
  facet_wrap(~Pclass) + 
  ggtitle("3D view of sex, pclass, and survival") +
  xlab("Sex") +
  ylab("Total Count") +
  labs(fill = "Survived")

## Age Visualisation
df[,"age_cat"] <- as.factor(ifelse(df$Age < 10 , 'baby' , ifelse(df$Age > 20 , 'adult','teen')))
table(df$age_cat)
df$age_cat[is.na(df$age_cat)] <- 'adult'

full_titanic <- df[1:891,]
ggplot(full_titanic[1:891,],aes(x = age_cat,fill=factor(Survived))) +
  geom_bar() +
  ggtitle("AGE CATEGORY v/s Survival Rate")+
  xlab("Age") +
  ylab("Total Count") +
  labs(fill = "Survived")  

#### 3 way age visualisation with sex
ggplot(full_titanic[1:891,], aes(x = Sex, fill = Survived)) +
  geom_bar() +
  facet_wrap(~age_cat) + 
  ggtitle("3D view of sex, pclass, and survival") +
  xlab("Sex") +
  ylab("Total Count") +
  labs(fill = "Survived")


## Calculating family size
df$FamilySize <-df$SibSp + df$Parch + 1

df$FamilySized[df$FamilySize == 1]   <- 'Single'
df$FamilySized[df$FamilySize < 5 & df$FamilySize >= 2]   <- 'Small'
df$FamilySized[df$FamilySize >= 5]   <- 'Big'

df$FamilySized=as.factor(df$FamilySized)
full_titanic <- df[1:891,]

###Lets Visualize the Survival rate by Family size 
ggplot(full_titanic[1:891,],aes(x = FamilySized,fill=factor(Survived))) +
  geom_bar() +
  ggtitle("Family Size V/S Survival Rate") +
  xlab("FamilySize") +
  ylab("Total Count") +
  labs(fill = "Survived")


###is there any association between Survial rate and where he get into the Ship.   
ggplot(full_titanic[1:891,],aes(x = Embarked,fill=factor(Survived))) +
  geom_bar() +
  ggtitle("Embarked vs Survival") +
  xlab("Embarked") +
  ylab("Total Count") +
  labs(fill = "Survived")

##Lets further divide the grpah by Pclass
ggplot(full_titanic[1:891,], aes(x = Embarked, fill = Survived)) +
  geom_bar() +
  facet_wrap(~Pclass) + 
  ggtitle("Pclass vs Embarked vs survival") +
  xlab("Embarked") +
  ylab("Total Count") +
  labs(fill = "Survived")


##### DATA PREPARATION

train_total <- df[1:891, c("Pclass", "age_cat","Sex","Embarked","FamilySized","Survived")]
str(train_total)

split_indices <- sample(nrow(train_total), nrow(train_total)*0.8, replace = F)
train <- train_total[split_indices,]
test <- train_total[-split_indices,]

test_total <- df[892:1309, c("Pclass", "age_cat","Sex","Embarked","FamilySized","Survived")]

####### CREATING MODELS #############

## DECISION TREE

# Classification Trees
library(rpart)
library(rpart.plot)
library(caret)

#1 build tree model- default hyperparameters
tree.model <- rpart(Survived ~ .,                     # formula
                    data = train,                   # training data
                    method = "class")               # classification or regression

# display decision tree
prp(tree.model)

# make predictions on the test set
tree.predict <- predict(tree.model, test, type = "class")

# evaluate the results
confusionMatrix(test$Survived, tree.predict)  # 0.8045

############ DECISION TREE END ########################

############ RANDOM FOREST START ######################
## Build the random forest
library(randomForest)
set.seed(71)
data.rf <- randomForest(Survived ~ ., data=train, proximity=FALSE,
                        ntree=1000, mtry=5, do.trace=TRUE, na.action=na.omit)
data.rf
varImpPlot(data.rf) ### to show the importance of variable in model
testPred <- predict(data.rf, newdata=test)
table(testPred, test$Survived)

############ RANDOM FOREST END ######################

############ SVM ######################
library(kernlab)
#Using Linear Kernel
Model_linear <- ksvm(Survived~ ., data = train, kernel = "vanilladot")
Eval_linear<- predict(Model_linear, test)

#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear,test$Survived) ## Accuracy 0.81


##Using Radial Kernel
trainControl <- trainControl(method="cv", number=5)


# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.

set.seed(7)
grid <- expand.grid(.sigma=c(0.015 ,0.025, 0.05), .C=c(0.1,0.5,1,2) )


#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit.svm <- train(Survived~., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

print(fit.svm)


# Constructing final model with sigma(gamma) = 0.015 and Cost(c) =1.0
Final_model <- ksvm(Survived~., data=train, scale = FALSE, gamma=0.015,cost=1.0 ,kernel = "rbfdot")
Eval_Final_model<- predict(Final_model, test)

confusionMatrix(Eval_Final_model,test$Survived)# accuracy 87%

################### SVM END ###############################

###### SVM ###### FINAL DATA #######

last_data <- predict(Final_model, test_total)
id <- df$PassengerId[892:1309]
Submission <- as.data.frame(cbind(df$PassengerId[892:1309],last_data))
Submission$Survived[which(Submission$last_data == '1')] <- '0'
Submission$Survived[which(Submission$last_data == '2')] <- '1'
write.csv(Submission[,c('V1','Survived')],'gender_submission.csv')


####################################


############### LR START ##################################

contrasts(train$Sex)

log.mod <- glm(Survived ~ ., family = binomial(link=logit), 
               data = train)
###Check the summary
summary(log.mod)

confint(log.mod)

## Training accuracy
train.probs <- predict(log.mod, data=train,type =  "response")
table(train$Survived,train.probs>0.5) # Accuracy 80%

###Logistic regression predicted train data with accuracy rate of 0.83 
test.probs <- predict(log.mod, newdata=test,type =  "response")
table(test$Survived,test.probs>0.5)


#df[,'cabin_type'] <- df$Cabin

df$cabin_type <- ifelse(grepl("A", df$Cabin), "A", 
                        ifelse(grepl("B", df$Cabin), "B", 
                               ifelse(grepl("C", df$Cabin), "C", 
                                      ifelse(grepl("D", df$Cabin), "D",
                                             ifelse(grepl("E", df$Cabin), "E", "Z"
                                              )))))


