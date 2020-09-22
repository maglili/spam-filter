SpamFilter using sk-learn SVM  by Python.
This code is reference to Machine learning course taught by Andrew Ng on coursera.

=============================================================================================================
程式需要import下列module:
1. Numpy  
2. scipy  
3. Joblib    
4. re           
5. nltk
6. sklearn  

=============================================================================================================
spamFilter 資料夾應包含5個python程式:
1. emailFeatures.py
2. getVocabList.py
3. precessEmail.py
4. main.py
5. trainning.py

spamFilter 資料夾應包含2個.mat檔案:
1. spamTest.mat
2. spamTrain.mat

spamFilter 資料夾應包含1個.pkl檔案:
1. joblib_model.pkl

spamFilter 資料夾應包含vocab.txt檔案。

spamFilter 資料夾預設會有數個(.txt)文字檔，此文字檔即代表email內容供測試，使用者也可自行加入(.txt)檔案。

=============================================================================================================
使用方法:

1. 執行main.py
2. 輸入欲測試的郵件(.txt)檔案
3. 程式將會跑出預測結果

=============================================================================================================
原理:

1.訓練模型
在此程式中使用scikit-learn的svm套件，
一開始執行main.py後，main.py會呼叫joblib_model.pkl檔案，
此檔案即是由trainning.py所訓練出來的模型。

trainning,py是透過呼叫spamTest.mat與spamTrain.mat這兩個檔案，
這兩個檔案即是我們的資料集合，將資料丟入svm進行運算後，即得到模型。

2.資料型態
我們的資料是.mat檔，在spamTest.mat中，包含了兩個矩陣 X、y。
	X矩陣的大小是4000*1899
	y矩陣的大小事4000*1
代表這個資料集合共有4000筆資料，而每筆資料的特徵數為1899個。

特徵數1899是被選定的，常見於垃圾郵件中的文字。
這1899個字可以在vocab.txt中所看到。

3.預處理
在precessEmail.py中，作法是將email內容進行預處理，去除標點符號、特殊符號、html標籤、URL，並將文字做stemmimg。
我們先利用getVocabList()將vocab.txt的內容轉換成list "vocabList"。

然後在將處理完的email內容轉換成list "word_indices"，把email中的字依照vocabList的順序(vocab.txt的編號)進行映射。

4.特徵
由emailFeatures.py 建立一個長度1899的list "x"，並將word_indices傳入emailFeatures.py。
有出現在email中的字，將該位址之旗標設為1，其他皆為零。
這樣透過emailFeatures.py跑出來的list "x"，我們就能知道1899個字中，那些有出現。

5.預測
經由映射的方式，我們訓練出來的模型將會更準確。
