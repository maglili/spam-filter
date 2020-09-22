from processEmail import processEmail
from emailFeatures import emailFeatures
import numpy as np
import scipy.io as sio # use matlab's .mat documents
import joblib #store trainning model

# -----------------------Load from file---------------------------------
clf = joblib.load('joblib_model.pkl')

fname = input('Enter file name :')
try:
    fh = open(fname,"r",encoding="utf-8")
except:
    print('Error opening file')
    quit()
#-----------------------mail processing----------------------------------
email_contents = fh.read()
word_indices = processEmail(email_contents)
features = emailFeatures(word_indices)

#------------------------------print info---------------------------------
print('\n============precessing info============\n')
print('Length of feature vector: {}\n'.format(len(features)))
sum = 0
for i in features:
    if i > 0 :
        sum = sum + 1
print('Number of non-zero entries: {}\n'.format(sum))

#-------------------------predict----------------------------------------
#change list "features" to array "X".
X =  np.array(features)
#svm predict
#use reshape to aviod error
result = clf.predict(X.reshape(1,-1))

print('\n============Result============\n')
if result[0] == 0:
    print('NOT spam.')
else:
    print('Spam mail !')
