from processEmail import processEmail
import numpy as np
import scipy.io as sio # use matlab's .mat documents
import pickle

def email_features_vector(word_indices):
    """
    reture a feature vector
    """
    features = np.zeros(1899)
    for idx, value in enumerate(word_indices):
        features[value] = 1
    features = np.expand_dims(features, axis=0)
    return features

#load model
clf = pickle.load(open('./model/svm_model.pkl','rb'))

#load input file
root = './input/'
fname = input('Enter file name :')
try:
    fh = open(root+fname+'.txt',"r", encoding="utf-8")
    print('-'*30)
except:
    print('Error opening file')
    quit()

email_contents = fh.read()
word_indices = processEmail(email_contents)
features = email_features_vector(word_indices)



#predict
X =  features
result = clf.predict(X)

# result
if result[0] == 0:
    print('\n[Result]: NOT spam.')
else:
    print('\n[Result]: Spam mail!')
