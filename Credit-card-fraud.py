# ML-Project-6-Credit-Card-Fraud-Detection

#Import Libraries

import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from collections import Counter

#Read Dataset

import pandas as pd
df = pd.read_csv("/content/drive/MyDrive/card_transdata.csv", on_bad_lines="skip")
df

#Count Number of Fraud and Non-fraud Records

fraud_count = df['fraud'].value_counts()[1]
nonfraud_count = df['fraud'].value_counts()[0]

print(f"Number of Fraud Records: {fraud_count}")
print(f"Number of Non-Fraud Records: {nonfraud_count}")

#Determine the count of null values

total_null_values = df.isna().sum()
print("Total Null Values:", total_null_values)

#Determine the count of duplicate records

total_duplicates = df.duplicated().sum()
print("Total Duplicates:", total_duplicates)

#Drop duplicate records

df.drop_duplicates(inplace=True)
print(df)

df

#Descriptive Statistics

description=df.describe()
print(description)

import matplotlib.pyplot as plt
import pandas as pd

df.drop('fraud', axis=1).hist(figsize=(10, 8), bins=20)
plt.suptitle('Distribution of Numerical Features')
plt.show()

#Count of Fraud/Non-Fraud Records

import matplotlib.pyplot as plt

class_counts = df['fraud'].value_counts()
print("Class Distribution:")
print(class_counts)
class_counts = df['fraud'].value_counts()

# Create a bar plot
plt.figure(figsize=(6, 4))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
plt.title('Class Distribution of Fraud')
plt.xlabel('Fraud Class')
plt.ylabel('Count')
plt.show()

# Count each unique value in the 'fraud' column
df['fraud'].value_counts()

# Create a Pie Chart
df['fraud'].value_counts().plot.pie(autopct='%1.1f%%')

#The class distribution of the 'fraud' target variable is unbalanced, and it will be necessary to implement balancing techniques to ensure accurate classification.

#Variable selection

df

Columns_decimal=['distance_from_home','distance_from_last_transaction','ratio_to_median_purchase_price']

Columns_binary=['repeat_retailer','used_chip','used_pin_number','online_order']

fig, axes = plt.subplots(1, 3, sharex=False, figsize=(20,10))
position_matrix = [0,1,2]

count_var = 0

for feature in Columns_decimal:
    sns.kdeplot(ax=axes[count_var],data=df, x=feature,hue='fraud')
    axes[position_matrix[count_var]].set_title(feature)
    axes[position_matrix[count_var]].set_xlim(-100, 100)
    count_var += 1

sns.pairplot(data=df[Columns_decimal], height=6, aspect=1)

Columns_binary.pop()

fig, axes = plt.subplots(2, 2, sharex=False, figsize =(10,10))
sns.countplot(ax=axes[0,0],x="repeat_retailer", data=df, hue='fraud')
sns.countplot(ax=axes[0,1],x="used_chip", data=df, hue='fraud')
sns.countplot(ax=axes[1,0],x="used_pin_number", data=df, hue='fraud')
sns.countplot(ax=axes[1,1],x="online_order", data=df, hue='fraud')

fig, axes = plt.subplots(1, 3, sharex=False, figsize =(10,10))
sns.kdeplot(ax=axes[0],data=df,x="distance_from_home",hue="fraud")
sns.kdeplot(ax=axes[1],data=df,x="distance_from_last_transaction",hue="fraud")
sns.kdeplot(ax=axes[2],data=df,x="ratio_to_median_purchase_price",hue="fraud")

df.repeat_retailer.value_counts(1).plot.barh()

df.used_chip.value_counts(1).plot.barh()

df.used_pin_number.value_counts(1).plot.barh()

df.online_order.value_counts(1).plot.barh()

#TARGET ANALYSIS

#Now, let's exam our target, the FRAUD column. We are going to try to understand which variables are related to customer churn. This information is very useful, because we can use it to predict fraud transactions.

 df.fraud.value_counts(1).plot.barh()
 plt.title('Fraud Distribution', size = 18)
 plt.text(-0.1, 0.815, df.fraud.value_counts(1)[0])
 plt.text(0.9, 0.22, df.fraud.value_counts(1)[1])
 plt.ylim(0, 0.88)
 plt.text(1.6, 0.05, "We have a highly imbalanced dataset!", color='r')

#The main challenge in fraud detection is the extreme class imbalance in the data which makes it difficult for many classification algorithms to effectively separate the two classes. Only 0.087% of transactions are labeled as fradulent in this dataset.

# Drawing a pie plot to display the distribution of each categorical column
plt.figure(figsize = (16, 12))
for i, col in enumerate(Columns_binary):
    plt.subplot(1, 4, i+1)
    plt.pie(x = df[col].value_counts().values,
            labels = df[col].value_counts().index,
            autopct = '%1.1f%%')
    plt.xlabel(col, weight = 'bold')
plt.show()

#Binary features VS Target feature FRAUD:

fig, axes = plt.subplots(2, 2, figsize=(16, 8))
_ = sns.countplot(data=df, x='repeat_retailer', hue='fraud', ax=axes[0][0])
_ = sns.countplot(data=df, x='used_chip', hue='fraud', ax=axes[0][1])
_ = sns.countplot(data=df, x='used_pin_number', hue='fraud', ax=axes[1][0])
_ = sns.countplot(data=df, x='online_order', hue='fraud', ax=axes[1][1])
_ = plt.tight_layout()

#It looks like 'online_order' is the most common types of fraud. It actually makes sense cause if the website you're shopping online is fraudulent (the website was hacked, etc) your card number will be exposed on Internet. Let's take a closer look at this probability:

plt.figure(figsize = (15,12))

plt.subplot(2,2,1)
sns.countplot(x = 'repeat_retailer', hue= 'online_order', palette='Set2', data = df)

plt.subplot(2,2,2)
sns.countplot(x= 'used_chip', hue= 'online_order', palette='Set2', data = df)

plt.subplot(2,2,3)
sns.countplot(x = 'used_pin_number', hue= 'online_order', palette='Set2', data = df)

plt.subplot(2,2,4)
sns.countplot(x = 'fraud', hue= 'online_order', palette='Set2', data = df)

plt.figure(figsize = (15,12))

plt.subplot(2,2,1)
sns.countplot(x = 'online_order', hue= 'repeat_retailer', palette='Set2', data = df)

plt.subplot(2,2,2)
sns.countplot(x = 'used_chip', hue= 'repeat_retailer', palette='Set2', data = df)

plt.subplot(2,2,3)
sns.countplot(x = 'used_pin_number', hue= 'repeat_retailer', palette='Set2', data = df)

plt.subplot(2,2,4)
sns.countplot(x = 'fraud', hue= 'repeat_retailer', palette='Set2', data = df)

numerical_columns = ["distance_from_home", "distance_from_last_transaction", "ratio_to_median_purchase_price"]
for column in numerical_columns:
    plt.figure()
    plot = df[column]
    sns.histplot(plot, bins=10, kde=True)
    plt.show()

correlation_matrix = df.corr()

# Create a correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, center=0, linewidths=0.5)

plt.title('Correlation Heatmap of All Columns')
plt.show()

# full correlation table
df.corr().style.background_gradient(cmap='viridis')

ratio_to_median_purchase_price (ratio of purchased price transaction to median purchase price.) has a correlation of 0.46 with the target variable.

df.groupby('online_order')['fraud'].mean().sort_values()

df

df1=df.dropna(subset=['ratio_to_median_purchase_price', 'repeat_retailer','used_chip','used_pin_number','online_order','fraud'])

df1

# Selected variables
X = df1[['ratio_to_median_purchase_price', 'distance_from_home', 'distance_from_last_transaction', 'online_order', 'used_pin_number']]
y = df1['fraud']

Split train and test data

# Divide the data into training and testing sets for the model.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

Standard Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
Modeling

def cross_validation_score(estimator_name, estimator, X_train, y_train, score = 'recall', n = 5):

    '''This function is to validate the model across multiple stratified splits'''

    # Create a StratifiedKFold object with number of folds
    folds = StratifiedKFold(n_splits = n, shuffle = True, random_state = 42)

    validate = cross_val_score(estimator, X_train, y_train, scoring = score, cv = folds)

    print(f'Cross Validation Scores of {estimator_name}: {validate} \n')
    print(f'Mean of Scores for {estimator_name}: {validate.mean()} \n')
    print(f'Standard Deviation of Scores for {estimator_name}: {validate.std()}')

def hyperparameter_tunning(estimator, X_train, y_train, param_grid, score = 'recall', n = 5):

    '''This function is used to find the best set of hyperparameters for the model to optimize its performance'''

    # Perform grid search
    grid_search = GridSearchCV(estimator = estimator,
                               param_grid = param_grid,
                               cv = n,
                               scoring = score )

    # Fit the data
    grid_search.fit(X_train,y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Print the best parameters and score
    print(f'Best parameters: {best_params} \n')
    print(f'Best score: {best_score}')

    # best estimator
    best_estimator = grid_search.best_estimator_

    return best_estimator

def precision_recall_trade_off(model,X_test,y_test):

    '''this function is to plot the precision-recall curve then
    printing the thresholds that achieves the highest recall'''

    y_proba = model.predict_proba(X_test)
    precision ,recall ,threshold = precision_recall_curve(y_test,y_proba[:,1])
    p_r_t = pd.DataFrame({'Threshold':threshold,'Precision':precision[:-1],'Recall':recall[:-1]})
    fig = px.line(
        p_r_t,
        x='Recall',
        y='Precision',
        title='Precision-Recall Curve',
        width=700,height=500,
        hover_data=['Threshold']
    )
    fig.show()
    print(p_r_t[ (p_r_t['Recall']==1)].tail(10))

def model_evaluation(model, X_test, y_test, color = 'Reds'):

    ''' This function is used to evaluate the model through on classification report  and confusion matrix'''

    # classification report
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0), '\n ')

    #confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), cmap = color, annot = True)
    plt.xlabel('Predicted', size = 12, weight = 'bold')
    plt.ylabel('Actual', size = 12, weight = 'bold')
    plt.title('Confusion Matrix', weight = 'bold')
    plt.show()

def roc_auc_curve_score(model, X_test, y_test):

    '''This function plots the roc-auc curve and calculate the model ROC-AUC score '''

    # y predicted value
    #y_pred = model.predict(X_test)

    # y predicted probability
    y_proba = model.predict_proba(X_test)

    # ROC_AUC Score
    score = roc_auc_score(y_test, y_proba[:, 1])

    # ROC Curve
    fpr ,tpr ,thresholds = roc_curve(y_test, y_proba[:, 1])

    plt.figure(figsize = (8, 5))
    plt.plot(fpr, tpr, label = 'ROC_AUC Score (area = %0.2f)' % score)
    plt.plot([0, 1], [0, 1],'b--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.1])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver operating characteristic (ROC)')
    plt.legend(loc = "lower right")
    plt.show()
    
Random Forest Classifier

# Apply model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(criterion='entropy')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Training Accurecy
model.score(X_train, y_train)

Random Forest Evaluation

Print precision, recall, and F1-score
print(f"\nAccuracy: {accuracy}") print("Precision:", precision) print("Recall:", recall) print("F1 Score:", f1_score)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
from sklearn.metrics import accuracy_score
y_pred =model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Extract precision, recall, and F1-score from the classification report
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

# Print precision, recall, and F1-score
print(f"\nAccuracy: {accuracy_score}")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

Random Forest Model Overfitting/Underfitting

print('Training set score: {:.4f}'.format(model.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(model.score(X_test, y_test)))

Random Forest Cross Validation

from sklearn.model_selection import StratifiedKFold
cross_validation_score('Random Forest', model, X_train, y_train, n = 5)

from sklearn.metrics import auc, roc_curve
roc_auc_curve_score(model, X_test, y_test)

Verify model

# Verify model with a classification report
print(classification_report(y_test, y_pred))
  
# Verify model with a confusion matrix
confusion_matrix(y_test, y_pred)

Decision Tree Model Training

from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

dt_model = DecisionTreeClassifier(random_state=42,min_samples_leaf=10, max_depth=4,criterion="entropy")

# Train the model on the training data
dt_model.fit(X_train,y_train)

# Make predictions on the test set
Y_pred = dt_model.predict(X_test)

# Training Accurecy
dt_model.score(X_train, y_train)

Decision Tree Model Plotting

# Visualize the decision tree
plt.figure(figsize=(25, 20))
plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=['Non-Fraud', 'Fraud'], rounded=True, fontsize=10)
plt.show()

Decision Tree Model Evaluation

# Evaluate the model
from sklearn.metrics import precision_recall_fscore_support
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)
confusion_matrix_result = confusion_matrix(y_test, y_pred)

# Print the results
print("\nClassification Report:")
print(classification_report_result)
print("\nConfusion Matrix:")
print(confusion_matrix_result)

# Extract precision, recall, and F1-score from the classification report
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

# Print precision, recall, and F1-score
print(f"\nAccuracy: {accuracy}")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

from sklearn.metrics import auc, roc_curve
roc_auc_curve_score(dt_model, X_test, y_test)

Decision Tree Model Overfitting/Underfitting

print('Training set score: {:.4f}'.format(dt_model.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(dt_model.score(X_test,y_test)))

Decision Tree Model Confusion Matrix Plotting

!pip  install scikit-plot

import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(y_test, y_pred)

Decision Tree Model Cross Validation

from sklearn.model_selection import cross_validate, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Specify the number of folds
k_folds = 10

# Create a KFold object with the desired number of folds
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Define the scoring metrics
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']

# Perform cross-validation and get scores for each metric
cv_results = cross_validate(dt_model, X_train, y_train, cv=kf, scoring=scoring_metrics, return_train_score=False)

# Print the results
for metric in scoring_metrics:
    print(f'{metric.capitalize()}: {np.mean(cv_results[f"test_{metric}"])}')

cross_validation_score('Decision Tree', dt_model, X_train, y_train, n = 10)

Gradient Boosting Model

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

Gradient Boosting Model Training and Evaluation

gb_model.fit(X_train, y_train)

pred = gb_model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

# Extract precision, recall, and F1-score from the classification report
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, pred, average='weighted')

# Print precision, recall, and F1-score
print(f"\nAccuracy: {accuracy}")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

Gradient Boosting Model Overfitting/Underfitting

print('Training set score: {:.4f}'.format(gb_model.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(gb_model.score(X_test, y_test)))

Gradient Boosting Model Confusion Matrix Plotting

import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(4, 2))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=gb_model.classes_, yticklabels=gb_model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

Gradient Boosting Model Cross Validation

# Perform cross-validation and get scores for each metric
cv_results = cross_validate(gb_model, X_train, y_train, cv=kf, scoring=scoring_metrics, return_train_score=False)

for metric in scoring_metrics:
    print(f'{metric.capitalize()}: {np.mean(cv_results[f"test_{metric}"])}')

from sklearn.metrics import auc, roc_curve
roc_auc_curve_score(gb_model,X_test,y_test)

Logistic Regression Model Training

from sklearn.linear_model import LogisticRegression

logistic_regressor = LogisticRegression(max_iter=500)

logistic_regressor.fit(X_train, y_train)
L_pred = logistic_regressor.predict(X_test)

# Training Accurecy
from sklearn.linear_model import LogisticRegression
logistic_regressor.score(X_train, y_train)

Logistic Regression Model Evaluation

# Evaluate the model
accuracy = accuracy_score(y_test, L_pred)
classification_report_result = classification_report(y_test, L_pred)
confusion_matrix_result = confusion_matrix(y_test, L_pred)

# Print the results
print("\nClassification Report:")
print(classification_report_result)
print("\nConfusion Matrix:")
print(confusion_matrix_result)

# Extract precision, recall, and F1-score from the classification report
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, L_pred, average='weighted')

# Print precision, recall, and F1-score
print(f"\nAccuracy: {accuracy}")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

model_evaluation(logistic_regressor, X_test, y_test)

from sklearn.metrics import auc, roc_curve
roc_auc_curve_score(logistic_regressor,X_test,y_test)

Logistic Regression Overfitting/Underfitting

print('Training set score: {:.4f}'.format(logistic_regressor.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logistic_regressor.score(X_test, y_test)))

Logistic Regression Model Confusion Matrix Plotting

import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(y_test, y_pred)

Logistic Regression Model Cross Validation

# Perform cross-validation and get scores for each metric
cv_results = cross_validate(logistic_regressor, X_train, y_train, cv=kf, scoring=scoring_metrics, return_train_score=False)

for metric in scoring_metrics:
    print(f'{metric.capitalize()}: {np.mean(cv_results[f"test_{metric}"])}')

# Training Accurecy
logistic_regressor.score(X_train, y_train)

from sklearn.metrics import auc, roc_curve
roc_auc_curve_score(logistic_regressor, X_test, y_test)

KNN Model Training

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Define the parameter grid with different values of k
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}

# Initialize the KNeighborsClassifier
knnmodel = KNeighborsClassifier()

# Initialize GridSearchCV with the model and parameter grid
grid_search = GridSearchCV(knnmodel, param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_k = grid_search.best_params_['n_neighbors']

# Initialize the model with the best value of k
best_knnmodel = KNeighborsClassifier(n_neighbors=best_k)

best_knnmodel.fit(X_train, y_train)

Y_pred = best_knnmodel.predict(X_test)

KNN Model Evaluation

# Make predictions
y_pred = best_knnmodel.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)
confusion_matrix_result = confusion_matrix(y_test, y_pred)

# Print the results
print("\nClassification Report:")
print(classification_report_result)
print("\nConfusion Matrix:")
print(confusion_matrix_result)

# Extract precision, recall, and F1-score from the classification report
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

# Print precision, recall, and F1-score
print(f"\nAccuracy: {accuracy}")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

KNN Model Overfitting/Underfitting

print('Training set score: {:.4f}'.format(best_knnmodel.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(best_knnmodel.score(X_test, y_test)))

KNN Model Confusion Matrix Plotting

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

KNN Model Cross Validation

# Perform cross-validation and get scores for each metric
cv_results = cross_validate(best_knnmodel, X_train, y_train, cv=kf, scoring=scoring_metrics, return_train_score=False)

for metric in scoring_metrics:
    print(f'{metric.capitalize()}: {np.mean(cv_results[f"test_{metric}"])}')

Naive Bayes Model

from sklearn.naive_bayes import GaussianNB
naive = GaussianNB()
naive.fit(X_train, y_train)

# Training Accurecy
from sklearn.naive_bayes import GaussianNB
naive.score(X_train, y_train)

Naive Bayes Evaluation

nb_pred = naive.predict(X_test)

print("Classification Report:\n", classification_report(y_test, nb_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, nb_pred))

# Extract precision, recall, and F1-score from the classification report
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, nb_pred,average='weighted')

# Print precision, recall, and F1-score
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

Naive Bayes Overfitting/Underfitting

print('Training set score: {:.4f}'.format(naive.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(naive.score(X_test, y_test)))

Naive Bayes Cross Validation

# Perform cross-validation and get scores for each metric
cv_results = cross_validate(naive, X_train, y_train, cv=kf, scoring=scoring_metrics, return_train_score=False)

for metric in scoring_metrics:
    print(f'{metric.capitalize()}: {np.mean(cv_results[f"test_{metric}"])}')

roc_auc_curve_score(naive , X_test, y_test)

Extra Trees

# Extra Trees
from sklearn.ensemble import ExtraTreesClassifier

model2 = ExtraTreesClassifier(random_state=123)
model2.fit(X_train, y_train)
y_train_hat = model2.predict(X_train)
y_test_hat = model2.predict(X_test)

print(model2)
print('Train performance')
print('-------------------------------------------------------')
print(classification_report(y_train, y_train_hat))

print('Test performance')
print('-------------------------------------------------------')
print(classification_report(y_test, y_test_hat))

print('Roc_auc score')
print('-------------------------------------------------------')
print(roc_auc_score(y_test, y_test_hat))
print('')
print('Confusion matrix')
print('-------------------------------------------------------')
print(confusion_matrix(y_test, y_test_hat))

XGB classifier

from xgboost import XGBClassifier
model3 = XGBClassifier(random_state=123)
model3.fit(X_train, y_train)
y_pred = model3.predict(X_test)

print(model3)
print('Train performance')
print('-------------------------------------------------------')
print(classification_report(y_test, y_pred))

print('Test performance')
print('-------------------------------------------------------')
print(classification_report(y_test, y_pred))

print('Roc_auc score')
print('-------------------------------------------------------')
print(roc_auc_score(y_test, y_pred))
print('')

print('Confusion matrix')
print('-------------------------------------------------------')
print(confusion_matrix(y_test, y_pred))

model_evaluation(model3,X_train,y_train,'Greens')

from sklearn.metrics import auc, roc_curve
roc_auc_curve_score(model3,X_test,y_test)

SVC Model

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

svm_model=SVC()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model.fit(X_train_scaled,y_train)
SVC_train_preds = svm_model.predict(X_train)

pred = svm_model.predict(X_test_scaled)
print("Accuracy Score: ", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print("Classification Report:\n", classification_report(y_test, pred))

# Random forest , Decision tree and Gradient boosting have definitely the best performance from all models we've tried. All I did was scaling features. Our dataset is under extreme class imbalance so we must do something.

val_predictions = model.predict(X_test)
print(val_predictions)
preds = np.around(val_predictions)

predictions

Results=pd.DataFrame({'Actual':y_test,'Predictions':pred})
Results.head(5)

print(accuracy_score(y_test,y_pred))

print("Actual\t Predicted")
for i in range(50):
    x=y_test.iloc[i]
    y=y_pred[i]
    print(x,"\t",y)

cm=confusion_matrix(y_test,y_pred)
print(cm)

import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.title('Confusion Matrix - Test Data')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.clf()
plt.imshow(cm,interpolation='nearest',cmap=plt.cm.Wistia)
classNames = ['0','1']
plt.title('Confusion Matrix-Test Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(2)
plt.xticks(tick_marks,classNames,rotation=45)
plt.yticks(tick_marks,classNames)
s = [['TN','FP'],['FN','TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i,str(s[i][j])+"= "+str(cm[i][j]))
plt.show()

print(classification_report(y_test,y_pred))

nt=cm[0][0]
pf=cm[0][1]
nf=cm[1][0]
pt=cm[1][1]

recall=pt/(pt+nf)
print('Recall=',recall)

precision=pt/(pt+pf)
print("Precision=",precision)

specificity = nt /  (nt + pf)
print("Specificity = ", specificity)

accuracy = ( pt + nt ) / ( pt + nt + pf + nf)
print("accuracy =" , accuracy)

Gather and Interpret Results

Conclusion Data: outliers are more than 30% of the data, so removing them will badly affect models Models: The most accurate models are GradientBoostingClassifier, RandomForestClassifier and decisin tree bold text

Conclusion dropping outliers will lead to losing important information all the features except for repeat_retailer have a moderate to strong correlation with the target. Decision Tree ,XGBoost and Random Forest achieved great results on both training and test sets (nearly 100% F1 score,100% Recall, 100% ROC-AUC score).

