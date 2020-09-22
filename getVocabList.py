def getVocabList():
    fid = open('vocab.txt')
    vocabList=[]

    for line in fid:
        line_split = line.split()
        vocabList.append(line_split[1])

    return vocabList
