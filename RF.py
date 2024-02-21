import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
import time

start = time.time()

df = pd.read_csv('IOT.csv')

print(df.type.value_counts())

labelencoder = LabelEncoder()
df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
print(df.type.value_counts())

X = df.drop(['ts', 'label', 'type'], axis=1).values
y = df.iloc[:, -2].values.reshape(-1, 1)
y = np.ravel(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1337, stratify=y)

# print(pd.Series(y_train).value_counts())

transfer = StandardScaler()
train_data = transfer.fit_transform(X_train.reshape(-1, 1))
test_data = transfer.transform(X_test.reshape(-1, 1))

# Random Forest training and prediction
model = RandomForestClassifier(random_state=1337)

scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
scores = cross_validate(model, X, y, scoring=scoring, cv=4, n_jobs=-1)

print('scores：', scores)
print("fit_time:  %0.3f " % (scores['fit_time'].mean()))
print("score_time:  %0.3f " % (scores['score_time'].mean()))

print("Accuracy (Testing):  %0.4f " % (scores['test_accuracy'].mean()))
print("Precision (Testing):  %0.4f " % (scores['test_precision_macro'].mean()))
print("Recall (Testing):  %0.4f " % (scores['test_recall_macro'].mean()))
print("F1-Score (Testing):  %0.4f " % (scores['test_f1_macro'].mean()))

end = time.time()
print("Time taken {}".format(end - start))
