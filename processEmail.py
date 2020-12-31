import re
from nltk.stem import PorterStemmer

def getVocabList():
    fh = open('./training_data/vocab.txt')
    vocabList=[]

    for row in fh:
        line_split = row.split()
        vocabList.append(line_split[1])
    return vocabList

def processEmail(email_contents):
    vocabList = getVocabList() #Load Vocabulary
    word_indices = [] #Init return value
    print('[Originoal Email]:\n')
    print(email_contents)
    print('='*30)
    #----------------------preprocess email-----------------------
    #text to lower case
    email_contents = email_contents.lower()

    #strip all HTML
    email_contents = re.sub(r'<[^<>]+>',' ',email_contents)

    #handle number
    email_contents = re.sub(r'[0-9]+','number',email_contents)

    #handle URLS
    email_contents = re.sub(r'(http|https)://[^\s]*','httpaddr',email_contents)

    #handle email address
    email_contents = re.sub(r'[^\s]+@[^\s]+','emailaddr',email_contents)

    #handle $ sign
    email_contents = re.sub(r'[$]+','dollar',email_contents)

    #---------------------------tokennize email--------------------------
    print('[Processed Email]:\n')
    email_split = email_contents.split()

    #handel non-character
    count = 0
    for word in email_split:
        if  (not word.isalpha()) and (not word.isnumeric() ):
            for char in word:
                if  not char.isalpha() :
                    word = word.replace(char,'')
                    email_split[count] = word
        count = count + 1

    #word stemming
    count = 0
    porter = PorterStemmer()
    for word in email_split:
        email_split[count] = porter.stem(word)
        count = count + 1

    #renew email content
    email_contents = ' '.join(email_split)
    email_split = email_contents.split()

    #-----------------------mapping word_indices-------------------
    for i in range(len(email_split)):
        for j in range(len(vocabList)):
            if email_split[i] == vocabList[j]:
                word_indices.append(j)

    print(email_contents)
    print('='*30)
    #print('\n========= word_indices =============\n')
    #print(word_indices)
    return word_indices
