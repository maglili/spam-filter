def emailFeatures(word_indices):
    #total number of words in the dictionarty
    n = 1899
    x=[]
    #init return array
    for i in range(1899):
        x.append(0)

    for i in range(len(word_indices)):
        x[word_indices[i]] = 1

    return x
