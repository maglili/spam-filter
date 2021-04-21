# spam-filter

## Introductione

本程式使用 sklearn 所提供的 SVM 訓練。
訓練資料來自 Andrew Ng 的 Coursera 線上課程 [Machine Learning](https://www.coursera.org/learn/machine-learning)。

## How-to-use

本程式僅能判斷.txt檔，請將欲判斷的郵件內容存成.txt後，丟入 input/ 資料匣。

之後在 cmd 輸入:
```cmd
python main.py 
```
之後再輸出剛剛所儲存的檔案名(不用加.txt)，即可進行預測。

## Detail

本 repo 含有3個程式檔:
1. main: 主程式
2. processEmail: 將 email 前處理
3. training: 訓練 svm

一開始 svm 是透過 training 先訓練好好，
若想重新訓練or更換其他模型，
請修改training，training 會將訓練好的模型存在 model/ 這個資料匣中。

主程式 main 會從資料匣 model/ 呼叫已經訓練好的模型，
然後會再呼叫 processEmail.py 中的函式，
此函式會對 email 內容進行前處理，
主要是將email的文字轉成文字庫(請見training_data/vocab.txt)中的代號，
最後會回傳一個list，
此 list 就是文字在 vocab.txt 的 index。 

main 會對回傳的list進行處理，轉換成1899維的向量X，
將此向量丟入訓練好的 svm 後，即可得到預測結果 y_pred。




