
# coding: utf-8

# In[6]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
# get_ipython().magic(u'matplotlib inline')
pd.set_option('display.max_columns',500)
pd.set_option('display.max_rows',500)
pd.set_option('display.max_info_columns', 500)


# In[7]:

df_import = pd.read_csv('../data/train.csv')
df_import_test = pd.read_csv('../data/test.csv')


# ## Exploratory Data Analysis

# In[8]:

df_import.head(10)


# In[9]:

df_import.v3.unique(), df_import.v30.unique(), df_import.v31.unique()


# In[10]:

len(df_import.v22.unique()), len(df_import.v24.unique())


# In[11]:

df_import.info()


# In[12]:

df_import.describe()


# In[13]:

df = df_import.copy()
df_test = df_import_test.copy()


# In[14]:

len(df_test)


# ## Feature creation

# In[15]:

colnames = ['v3','v22','v24','v30','v31','v47','v52','v56','v66','v71','v74','v75','v79','v91','v107','v110','v112','v113','v125']


# In[16]:

for colname in colnames:
    df[colname] = pd.Categorical(df[colname]).codes + 2
    df_test[colname] = pd.Categorical(df_test[colname]).codes + 2


# In[17]:

df.fillna(0, inplace=True)


# In[18]:

df_test.fillna(0, inplace=True)


# In[19]:

# df.head(10)


# In[20]:

# len(df_test)


# In[21]:

# df_train['v14sq'] = df.v14**2
df['v12sq'] = df.v12**2
df_test['v12sq'] = df_test.v12**2
df.drop(['ID','v12'], axis=1, inplace=True)
df_test.drop('v12', axis=1, inplace=True)
df_train = df.copy()
# df_test  = clean_data(test)


# In[22]:

y = df_train.pop('target').values
X = df_train.values
X_test = df_test.values


print 'Random Forest'

rf = RandomForestClassifier(40, n_jobs=-1)
rf.fit(X,y)


# In[24]:

feat_rank = np.argsort(rf.feature_importances_)[::-1]
# feat_rank


# In[25]:

# len(df_train.columns)


# In[26]:

# df_train.columns[feat_rank][:25]


# In[27]:

# df_features = pd.DataFrame(rf.feature_importances_,df_train.columns, columns = ['feature_value'])
# df_features.sort_values('feature_value', ascending=False)


# In[ ]:

# scores = np.zeros((feat_rank.shape[0],2))
# for i in range(1,feat_rank.shape[0]+1):
#     features = [df_train.columns[feat_rank][x] for x in range(i)]
#     scores[i-1:] = (i,(cross_val_score(rf, df[features], df['target'], cv=3)).mean())
# scores


# In[ ]:

# plt.plot(scores[:,:1],scores[:,1:2])


# In[28]:

# len(df_train.columns)


# In[29]:

features = [df_train.columns[feat_rank][x] for x in range(130)]
# features[:10]


# In[30]:

X = df_train[features].values


# In[31]:

rf.fit(X,y)


# In[32]:

print cross_val_score(rf, X, y, cv=5, scoring='log_loss').mean()


# In[33]:

print X.shape


# In[35]:

def create_submission(model, train, test, features, filename):

#     model.fit(train[features].values, train['target'].values)
    predictions = model.predict_proba(test[features])[:,1:]

    submission = pd.DataFrame({
        "ID": test["ID"],
        "PredictedProb": predictions.flatten()
    })

    submission.to_csv(filename, index=False)


# In[36]:

create_submission(rf, df, df_test, features, "../submissions/rf_submission.csv")


# In[34]:

# feat_rank = np.argsort(rf.feature_importances_)[::-1]
# feat_rank
# df_features = pd.DataFrame(rf.feature_importances_,df[features].columns, columns = ['feature_value'])
# df_features.sort_values('feature_value', ascending=False)


# ## Hypertune Parameters

# In[37]:

# build a classifier
clf = RandomForestClassifier()


# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


print 'Random Search'

# specify parameters and distributions to sample from
param_dist = {"max_depth": [100, None],
              "max_features": sp_randint(20, 130),
              "min_samples_split": sp_randint(10, 100),
              "min_samples_leaf": sp_randint(10, 100),
              "bootstrap": [True, False],
              'n_estimators': [40, 60, 100],
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 30
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, n_jobs=-1, scoring='log_loss')

start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)


# In[ ]:

create_submission(random_search.best_estimator_, df, df_test, features, "../submissions/rfrand_submission.csv")


print 'Grid Search'

# use a full grid over all parameters
param_grid = {'max_depth': [300, 600, 900, None],
              'max_features': ['sqrt', 'log2', None],
              'min_samples_split': [4, 12, 20, 50],
              'min_samples_leaf': [30, 50, 70],
              'bootstrap': [True],
              'n_estimators': [100, 200],
              "criterion": ["gini", "entropy"]
             }

# run grid search
rf_grid = GridSearchCV(clf, param_grid=param_grid, n_jobs=-1, scoring='log_loss')
start = time()
rf_grid.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(rf_grid.grid_scores_)))
report(rf_grid.grid_scores_)


# In[ ]:

rf_grid.best_estimator_


# In[ ]:

create_submission(rf_grid.best_estimator_, df, df_test, features, "../submissions/rfgrid_submission.csv")


# ```
# Mean validation score: 0.781 (std: 0.001)
# Parameters: {'bootstrap': False, 'min_samples_leaf': 2, 'n_estimators': 50, 'min_samples_split': 7, 'criterion': 'gini', 'max_features': 4, 'max_depth': None}
# ```

print 'AdaBoost'

# In[ ]:

ada = AdaBoostClassifier(n_estimators=200)
ada.fit(df[features].values, df['target'].values)


# In[ ]:

param_grid = {'learning_rate': [1, 0.5, 0.1, 0.05, 0.01, 0.001]}

abc = AdaBoostClassifier(n_estimators=200)
abc_grid = GridSearchCV(abc, param_grid, n_jobs=-1, scoring='log_loss').fit(X,y)

abc_grid.best_params_


# In[ ]:

create_submission(abc_grid.best_estimator_, df, df_test, features, "../submissions/abc_submission.csv")


# ## SVM

print 'SVM'

pipeline = Pipeline([('scaler', StandardScaler()),
                     ('svc', SVC())])
pipeline.fit(X, y)


# In[ ]:

parameters = {'kernel':['linear','rbf'],
              'C':np.linspace(.001,10,5),'degree':np.linspace(0,10,5)}

svm_grid = GridSearchCV(estimator=pipeline.steps[1][1],
                    param_grid=parameters, scoring='log_loss', cv=5)


# In[ ]:

X = pipeline.steps[0][1].fit_transform(X)


# In[ ]:

svm_grid.fit(X,y)


# In[ ]:

svm_grid.grid_scores_, svm_grid.best_params_


# In[ ]:

create_submission(svm_grid.best_estimator_, df, df_test, features, "../submissions/svm_submission.csv")


# In[ ]:
