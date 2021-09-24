#!/usr/bin/env python
# coding: utf-8

# # Assignment 1 -- Supervised learning

# ### To run this code you need
# * Python 3.x
# 
# ### and the following packages
# * numpy
# * pandas
# * matplotlib
# * scikitlearn
# * xgboost
# 
# ### Dataset urls
# * [Dataset 1](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset)
# * [Dataset 2](https://archive.ics.uci.edu/ml/datasets/adult)
# ------

# # Imports and utility functions

# In[243]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit


# In[244]:


def train_and_get_stats(clf, X_train, Y_train, X_test, Y_test):
    start = time.time()
    clf.fit(X_train, Y_train)
    end = time.time()
    training_time = end - start
    
    start = time.time()
    train_acc = clf.score(X_train, Y_train)
    end = time.time()
    train_score_time = end - start
    
    start = time.time()
    test_acc = clf.score(X_test, Y_test)
    end = time.time()
    test_score_time = end - start
    
    cv_scores_train = cross_val_score(clf, X_train, Y_train, cv=5)
    cv_scores_test = cross_val_score(clf, X_test, Y_test, cv=5)
    
    return {
        'train_time': training_time,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_score_time': train_score_time,
        'test_score_time': test_score_time,
        'cv_scores_train': cv_scores_train,
        'cv_scores_test': cv_scores_test
    }


def plot_training_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, fig_size=(10, 10), train_sizes=np.linspace(.1, 1.0, 10)):
    plt.figure(figsize=fig_size)
    plt.title(title)
    if ylim is not None:
        plt.set_ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ =         learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    
    # Plot learning curve
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Test score")
    plt.legend(loc="best")
    

def plot_validation_curve(clf, param_name, param_range, X_train, Y_train, title='Validation Curve', cv=5, n_jobs=1):
    train_scores, test_scores = validation_curve(
        clf, 
        X_train, 
        Y_train, 
        param_name=param_name, 
        param_range=param_range, 
        cv=cv,
        n_jobs=n_jobs, 
        scoring = "accuracy"
    )

    plt.figure(figsize=(10, 10))
    plt.title(title, fontsize=20)
    plt.xlabel(param_name, fontsize=15)
    plt.ylabel("Score", fontsize=15)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Test score",
                 color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()


# # Dataset 1 - Salary class, Data preprocessing

# In[287]:


data = pd.read_csv('data/adult.data')
test = pd.read_csv('data/adult.test')


# In[348]:


data


# In[290]:


# get discrete class columns from data, and one hot encode
discrete_classes = ['workclass',
                    'education', 
                    'sex',
                    'marital-status',
                    'occupation',
                    'relationship', 
                    'native-country',
                    'race', 
                    'salary-class']
encoded_train = pd.get_dummies(data[discrete_classes])
encoded_test = pd.get_dummies(test[discrete_classes])

# drop old non-encoded columns from data, and add encoded data
data.drop(columns=discrete_classes, inplace=True)
data = pd.concat([data, encoded_train], axis=1)
test.drop(columns=discrete_classes, inplace=True)
test = pd.concat([test, encoded_test], axis=1)

# drop extra output column as 'salary <= 50k' -> 0, and 'salary >50k' -> 1
data.drop(columns=['salary-class_ <=50K'], inplace=True)
data.rename(columns={'salary-class_ >50K': 'salary-class'}, inplace=True)
test.drop(columns=['salary-class_ <=50K'], inplace=True)
test.rename(columns={'salary-class_ >50K': 'salary-class'}, inplace=True)


# In[291]:


x_keys = set(data.keys()) - set(['salary-class']) & set(test.keys()) - set(['salary-class'])
y_keys = set(['salary-class'])

X_train = data[x_keys]
X_test = test[x_keys]

Y_train = data[y_keys].to_numpy().ravel()
Y_test = test[y_keys].to_numpy().ravel()


# # Experiments

# ## Decision Tree classifier

# In[266]:


dt_classifier_unpruned = DecisionTreeClassifier(max_depth=None, criterion='gini', splitter='best', random_state=42)
dt1_Unpruned_stats = train_and_get_stats(dt_classifier_unpruned, X_train, Y_train, X_test, Y_test)
print(dt1_Unpruned_stats)

dt_classifier = DecisionTreeClassifier(max_depth=10, criterion='gini', splitter='best', random_state=42)
dt1_stats = train_and_get_stats(dt_classifier, X_train, Y_train, X_test, Y_test)
print(dt1_stats)

plot_training_curve(
    DecisionTreeClassifier(max_depth=10, criterion='gini', splitter='best', random_state=42),
    "Decision Tree Classifier -- Salary Class (Experiment 1)",
    X_train,
    Y_train,
    cv=5,
    n_jobs=18,
    fig_size=(10, 10)
)
plot_validation_curve(
    DecisionTreeClassifier(max_depth = 10, criterion='gini', splitter='best', random_state=42),
    'max_depth',
    list(range(1, 21)),
    X_train,
    Y_train,
    "Decision Tree Classifier, Max Depth varied -- Salary Class (Experiment 1)",
     cv=5,
    n_jobs=18
)


# In[ ]:


print(round(dt1_stats['train_acc'], 4) * 100)
print(round(dt1_stats['test_acc'], 4) * 100)

train_sizes = np.linspace(.1, 1.0, 10)
plt.figure(figsize=(10, 10))
plt.title("Decision Tree Classifier -- Salary Class (Experiment 1)")
plt.xlabel("Training examples")
plt.ylabel("Score")
train_sizes, train_scores, test_scores, fit_times, _ =     learning_curve(dt_classifier, X_train, Y_train, cv=5, n_jobs=18,
                   train_sizes=train_sizes,
                   return_times=True)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
fit_times_mean = np.mean(fit_times, axis=1)
fit_times_std = np.std(fit_times, axis=1)

# Plot learning curve
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Test score")

train_sizes, train_scores, test_scores, fit_times, _ =     learning_curve(dt_classifier_unpruned, X_train, Y_train, cv=5, n_jobs=18,
                   train_sizes=train_sizes,
                   return_times=True)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
fit_times_mean = np.mean(fit_times, axis=1)
fit_times_std = np.std(fit_times, axis=1)
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="b")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="y")
plt.plot(train_sizes, train_scores_mean, 'o-', color="b",
             label="Training score (Unpruned)")
plt.plot(train_sizes, test_scores_mean, 'o-', color="y",
             label="Test score (Unpruned)")

plt.legend(loc="best")
plt.show()


# ## Support Vector Machine classifier

# In[11]:


#
# Moved below because this is slow
#


# ## K Nearest Neighbors classifier

# In[276]:


knn_classifier = KNeighborsClassifier(n_neighbors=25, weights='distance', n_jobs=24)
knn1_stats = train_and_get_stats(knn_classifier, X_train, Y_train, X_test, Y_test)

print(knn1_stats)
plot_training_curve(
    KNeighborsClassifier(n_neighbors=5, weights='uniform'),
    "KNN Classifier -- Salary Class (Experiment 1)",
    X_train,
    Y_train,
    cv=5,
    n_jobs=10,
    fig_size=(10, 10)
)
plot_validation_curve(
    KNeighborsClassifier(n_jobs=10),
    'n_neighbors',
    list(range(1, 50)),
    X_train,
    Y_train,
    "KNN Classifier, N-Neighbors varied -- Salary Class (Experiment 1)",
     cv=5,
    n_jobs=18
)


# ## Neural Network classifier

# In[312]:


nn_classifier = MLPClassifier(hidden_layer_sizes=(25), activation='relu', solver='adam', alpha=0.0001, random_state=1)
nn1_stats = train_and_get_stats(nn_classifier, X_train, Y_train, X_test, Y_test)
print(nn1_stats)

plot_training_curve(
    MLPClassifier(hidden_layer_sizes=(10), activation='relu', solver='adam', alpha=0.0001, random_state=1, learning_rate='adaptive'),
    "Nerual Network Classifier -- Salary Class (Experiment 1)",
    X_train,
    Y_train,
    cv=5,
    n_jobs=18,
    fig_size=(10, 10)
)


# ## Boosted Decision Trees classifier

# In[339]:


clf = xgb.XGBClassifier(n_estimators=50, max_depth=5, random_state=42)
clf1_stats = train_and_get_stats(clf, X_train, Y_train, X_test, Y_test)

print(clf1_stats)
plot_training_curve(
    xgb.XGBClassifier(n_estimators=50, max_depth=5, random_state=42),
    "Boosted Decision Tree Classifier -- Salary Class (Experiment 1)",
    X_train,
    Y_train,
    cv=5,
    n_jobs=18,
    fig_size=(10, 10)
)


# # Dataset 2 -- Credit defaults, Data Preprocessing

# In[349]:


data2 = pd.read_csv('data/UCI_Credit_Card.csv')
target_column = ['default.payment.next.month']


# In[350]:


data2


# In[351]:


data2.loc[(data2.SEX == 1), 'SEX'] = 'male'
data2.loc[(data2.SEX == 2), 'SEX'] = 'female'

data2.drop(index=data2.index[data2['EDUCATION'] == 0], inplace=True)
data2.loc[(data2.EDUCATION == 1), 'EDUCATION'] = 'graduate school'
data2.loc[(data2.EDUCATION == 2), 'EDUCATION'] = 'university'
data2.loc[(data2.EDUCATION == 3), 'EDUCATION'] = 'high school'
data2.loc[(data2.EDUCATION == 4), 'EDUCATION'] = 'others'
data2.loc[(data2.EDUCATION == 5), 'EDUCATION'] = 'unknown'
data2.loc[(data2.EDUCATION == 6), 'EDUCATION'] = 'unknown'

data2.drop(index=data2.index[data2['MARRIAGE'] == 0], inplace=True)
data2.loc[(data2.MARRIAGE == 1), 'MARRIAGE'] = 'married'
data2.loc[(data2.MARRIAGE == 1), 'MARRIAGE'] = 'married'
data2.loc[(data2.MARRIAGE == 2), 'MARRIAGE'] = 'single'
data2.loc[(data2.MARRIAGE == 3), 'MARRIAGE'] = 'others'

data2.drop(columns=['ID'], inplace=True)

discerete_columns = [
    'SEX',
    'EDUCATION',
    'MARRIAGE'
]
encoded_train2 = pd.get_dummies(data2[discerete_columns])

# drop old non-encoded columns from data, and add encoded data
data2.drop(columns=discerete_columns, inplace=True)
data2 = pd.concat([data2, encoded_train2], axis=1)

# # Worse performance, when trying to use feature engineering
# for i in range(1, 7):
#     data2[f'PAY_RATIO{i}'] = (data2[f'PAY_AMT1']/data2[f'BILL_AMT1'])
# data2[data2.filter(regex="PAY_RATIO").columns] = data2.filter(regex="PAY_RATIO").fillna(0)
# data2.replace([np.inf, -np.inf], 0.0, inplace=True)
# data2.drop(columns=data2.filter(regex="([BILL|PAY]_AMT.*\d)").columns, inplace=True)

# Drop columns which are causing accuracy to drop
data2.drop(columns=data2.filter(regex="(SEX_)").columns, inplace=True)
data2.drop(columns=data2.filter(regex="(MARRIAGE_)").columns, inplace=True)
data2.drop(columns=data2.filter(regex="(LIMIT_BAL)").columns, inplace=True)
data2.drop(columns=['AGE'], inplace=True)


# In[329]:


x_keys2 = set(data2.keys()) - set(target_column) & set(data2.keys()) - set(target_column)
y_keys2 = set(target_column)

first_split = data2.sample(frac=0.6,random_state=200)
second_split = data2.drop(first_split.index)

X_train2 = first_split[x_keys2]
Y_train2 = first_split[y_keys2].to_numpy().ravel()

X_test2 = second_split[x_keys2]
Y_test2 = second_split[y_keys2].to_numpy().ravel()


# ## Decision Tree classifier 2

# In[342]:


dt_classifier2_unpruned = DecisionTreeClassifier(max_depth=None, criterion='gini', splitter='best', random_state=42)
dt2_Unpruned_stats = train_and_get_stats(dt_classifier2_unpruned, X_train2, Y_train2, X_test2, Y_test2)
print(dt2_Unpruned_stats)

dt_classifier2 = DecisionTreeClassifier(max_depth = 5, criterion='gini', splitter='best', random_state=42)
dt2_stats = train_and_get_stats(dt_classifier2, X_train2, Y_train2, X_test2, Y_test2)
print(dt2_stats)

plot_training_curve(
    DecisionTreeClassifier(max_depth=5, criterion='gini', splitter='best', random_state=42),
    "Decision Tree Classifier -- Credit Default (Experiment 2)",
    X_train2,
    Y_train2,
    cv=5,
    n_jobs=18,
    fig_size=(10, 10)
)
plot_validation_curve(
    DecisionTreeClassifier(max_depth=5, criterion='gini', splitter='best', random_state=42),
    'max_depth',
    np.linspace(1, 20, 20),
    X_train2,
    Y_train2,
    "Decision Tree Classifier, Max Depth varied -- Credit Default (Experiment 2)",
     cv=5,
    n_jobs=18
)


# In[ ]:


train_sizes = np.linspace(.1, 1.0, 10)
plt.figure(figsize=(10, 10))
plt.title("Decision Tree Classifier -- Credit Default (Experiment 2)")
plt.xlabel("Training examples")
plt.ylabel("Score")
train_sizes, train_scores, test_scores, fit_times, _ =     learning_curve(dt_classifier2, X_train2, Y_train2, cv=5, n_jobs=18,
                   train_sizes=train_sizes,
                   return_times=True)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
fit_times_mean = np.mean(fit_times, axis=1)
fit_times_std = np.std(fit_times, axis=1)

# Plot learning curve
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Test score")

train_sizes, train_scores, test_scores, fit_times, _ =     learning_curve(dt_classifier2_unpruned, X_train2, Y_train2, cv=5, n_jobs=18,
                   train_sizes=train_sizes,
                   return_times=True)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
fit_times_mean = np.mean(fit_times, axis=1)
fit_times_std = np.std(fit_times, axis=1)
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="b")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="y")
plt.plot(train_sizes, train_scores_mean, 'o-', color="b",
             label="Training score (Unpruned)")
plt.plot(train_sizes, test_scores_mean, 'o-', color="y",
             label="Test score (Unpruned)")

plt.legend(loc=(0.65, 0.6))
plt.show()


# ## Support Vector Machine classifier

# In[21]:


#
# Moved below
#


# ## K Nearest Neighbors classifier

# In[275]:


knn_classifier2 = KNeighborsClassifier(n_neighbors=20, weights='uniform', n_jobs=24)
knn2_stats = train_and_get_stats(knn_classifier2, X_train2, Y_train2, X_test2, Y_test2)

print(knn2_stats)
plot_training_curve(
    KNeighborsClassifier(n_neighbors=20, weights='uniform'),
    "KNN Classifier -- Credit Default (Experiment 2)",
    X_train2,
    Y_train2,
    cv=5,
    n_jobs=18,
    fig_size=(10, 10)
)
plot_validation_curve(
    KNeighborsClassifier(weights='uniform', n_jobs=20),
    'n_neighbors',
    list(range(1, 50)),
    X_train2,
    Y_train2,
    "KNN Classifier, N-Neighbors varied -- Credit Default (Experiment 2)",
     cv=5,
    n_jobs=18
)


# ## Neural Network classifier

# In[352]:


nn_classifier2 = MLPClassifier(hidden_layer_sizes=(100),  alpha=0.001, activation='relu', random_state=1337)
nn2_stats = train_and_get_stats(nn_classifier2, X_train2, Y_train2, X_test2, Y_test2)

print(nn2_stats)
plot_training_curve(
    MLPClassifier(hidden_layer_sizes=(100),  alpha=0.001, activation='relu', random_state=1337),
    "Neural Network Classifier -- Credit Default (Experiment 2)",
    X_train2,
    Y_train2,
    cv=5,
    n_jobs=18,
    fig_size=(10, 10)
)


# ## Boosted Decision Trees classifier

# In[338]:


clf2 = xgb.XGBClassifier(n_estimators=50, max_depth=5)
clf2_stats = train_and_get_stats(clf2, X_train2, Y_train2, X_test2, Y_test2)

print(clf2_stats)
plot_training_curve(
    xgb.XGBClassifier(n_estimators=50, max_depth=5),
    "Boosted Decision Tree Classifier -- Credit Default (Experiment 2)",
    X_train2,
    Y_train2,
    cv=5,
    n_jobs=18,
    fig_size=(10, 10)
)


# # SVCs because they are slow

# ## SVC Experiment 1

# In[320]:


svc = LinearSVC(random_state=42, max_iter=10000)
svc1_stats = train_and_get_stats(svc, X_train, Y_train, X_test, Y_test)

print(svc1_stats)
plot_training_curve(
    LinearSVC(random_state=42, max_iter=10000),
    "Support Vector Classifier -- Salary Class (Experiment 1)",
    X_train,
    Y_train,
    cv=5,
    n_jobs=18,
    fig_size=(10, 10)
)


# ## SVC Experiment 2

# In[321]:


svc2 = LinearSVC(random_state=42, max_iter=10000)
svc2_stats = train_and_get_stats(svc2, X_train2, Y_train2, X_test2, Y_test2)

print(svc2_stats)
plot_training_curve(
    LinearSVC(random_state=42, max_iter=10000),
    "Support Vector Classifier -- Credit Default (Experiment 2)",
    X_train2,
    Y_train2,
    cv=5,
    n_jobs=18,
    fig_size=(10, 10)
)


# --------
# # Compare all algos after training them all

# In[347]:


model_stats = [ 
    dt1_Unpruned_stats,
    dt1_stats,
    dt2_Unpruned_stats,
    dt2_stats,
    knn1_stats,
    knn2_stats,
    nn1_stats,
    nn2_stats,
    clf1_stats,
    clf2_stats,
    svc1_stats,
    svc2_stats
]
exp1 = [ 
    dt1_Unpruned_stats,
    dt1_stats,
    knn1_stats,
    nn1_stats,
    clf1_stats,
    svc1_stats,
]
exp2 = [ 
    dt2_Unpruned_stats,
    dt2_stats,
    knn2_stats,
    nn2_stats,
    clf2_stats,
    svc2_stats
]

objects = ('Decision Tree\n(Unpruned)', 'Decision Tree\n(Pruned)','KNN', 'Neural Network', 'Boosted DT', 'SVC')
y_pos = np.arange(len(objects))

###########################################

performance_exp1 = []
performance_exp2 = []
for (item1, item2) in zip(exp1, exp2):
    performance_exp1.append(round(item1['train_acc'], 4))
    performance_exp2.append(round(item2['train_acc'], 4))

plt.figure(figsize=(15, 15))
plt.ylim(0.2, 1.0)
plt.bar(y_pos-0.2, performance_exp1, 0.4, align='center', alpha=0.8, label="Experiment 1")
plt.bar(y_pos+0.2, performance_exp2, 0.4, align='center', alpha=0.8, label="Experiment 2")
plt.legend()
plt.xticks(y_pos, objects, fontsize=13)
plt.ylabel('Accuracy (Ratio)', fontsize=13)
plt.title('Model Accuracy -- On training set')
for index, value in enumerate(zip(performance_exp1, performance_exp2)):
    plt.text(index-0.38, value[0]-0.015, str(value[0]), fontsize=14)
    plt.text(index+0.02, value[1]-0.015, str(value[1]), fontsize=14)
plt.show()

###########################################

performance_exp1 = []
performance_exp2 = []
for (item1, item2) in zip(exp1, exp2):
    performance_exp1.append(round(item1['test_acc'], 4))
    performance_exp2.append(round(item2['test_acc'], 4))

plt.figure(figsize=(15, 15))
plt.ylim(0.2, 0.9)
plt.bar(y_pos-0.2, performance_exp1, 0.4, align='center', alpha=0.8, label="Experiment 1")
plt.bar(y_pos+0.2, performance_exp2, 0.4, align='center', alpha=0.8, label="Experiment 2")
plt.legend()
plt.xticks(y_pos, objects, fontsize=13)
plt.ylabel('Accuracy (Ratio)', fontsize=13)
plt.title('Model Accuracy -- On testing set')
for index, value in enumerate(zip(performance_exp1, performance_exp2)):
    plt.text(index-0.38, value[0]+0.005, str(value[0]), fontsize=14)
    plt.text(index+0.02, value[1]+0.005, str(value[1]), fontsize=14)
plt.show()

###########################################

performance_exp1 = []
performance_exp2 = []
for (item1, item2) in zip(exp1, exp2):
    performance_exp1.append(round(item1['train_time'], 2))
    performance_exp2.append(round(item2['train_time'], 2))

plt.figure(figsize=(15, 15))
lim=3
plt.xlim(0, lim)
plt.barh(y_pos-0.2, performance_exp1, 0.4, align='center', alpha=0.8, label="Experiment 1")
plt.barh(y_pos+0.2, performance_exp2, 0.4, align='center', alpha=0.8, label="Experiment 2")
plt.legend()
plt.yticks(y_pos, objects, fontsize=13)
plt.xlabel('train_time (seconds)', fontsize=13)
plt.title('Model Training time')
for index, value in enumerate(zip(performance_exp1, performance_exp2)):
    plt.text(min(value[0], lim)+0.02, index-0.2, str(value[0]), fontsize=14)
    plt.text(min(value[1], lim)+0.02, index+0.2, str(value[1]), fontsize=14)
plt.show()

###########################################

performance_exp1 = []
performance_exp2 = []
for (item1, item2) in zip(exp1, exp2):
    performance_exp1.append(round(item1['test_score_time'], 4))
    performance_exp2.append(round(item2['test_score_time'], 4))

lim=0.05
plt.figure(figsize=(15, 15))
plt.xlim(0, lim)
plt.barh(y_pos-0.2, performance_exp1, 0.4, align='center', alpha=0.8, label="Experiment 1")
plt.barh(y_pos+0.2, performance_exp2, 0.4, align='center', alpha=0.8, label="Experiment 2")
plt.legend()
plt.yticks(y_pos, objects, fontsize=13)
plt.xlabel('prediction_time (seconds)', fontsize=13)
plt.title('Model Prediction time')
for index, value in enumerate(zip(performance_exp1, performance_exp2)):
    plt.text(min(value[0], lim)+0.0003, index-0.2, str(value[0]), fontsize=14)
    plt.text(min(value[1], lim)+0.0003, index+0.2, str(value[1]), fontsize=14)
plt.show()

