import numpy as np
import scipy.io as sio
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import joblib #store trainning model

#---------------------------open data from matlab---------------------
mat_file_name = 'spamTrain.mat'
data = sio.loadmat(mat_file_name)
X = data['X']
y = data['y']

mat_file_name = 'spamTest.mat'
data = sio.loadmat(mat_file_name)
Xtest = data['Xtest']
ytest = data['ytest']

#---------------------------building a model -------------------------
# Set the parameter candidates
parameter_candidates = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

# Create a classifier with the parameter candidates
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)

# Train the classifier on training data
clf.fit(X, np.ravel(y))


# -----------------------Print out the results-----------------------
print('Best score for training data:', clf.best_score_)
print('Best `C`:',clf.best_estimator_.C)
print('Best kernel:',clf.best_estimator_.kernel)
print('Best `gamma`:',clf.best_estimator_.gamma)

print('Accuracy is ',clf.score(Xtest,np.ravel(ytest)))

joblib_file = "joblib_model.pkl"
joblib.dump(clf, joblib_file)
