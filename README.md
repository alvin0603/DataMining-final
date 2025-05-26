# DataMining-final
## 目標
將 RNA-Seq 基因數據分成 Known(分類)和 Unknown(分群) 訓練資料只有三種病 但測試資料會有五種病

## 實驗數據：
`Thetacompare` : 對比 θ（分類門檻）在 0.5 和 0.9 下的結果 -> 證實會大程度影響最終結果


`PCAcompare` : 測試降維到不同維度(原始為兩萬多)的結果比較 -> 可討論不同種分類方法所適用的維度


`best_result_with_kFold` : 經過交叉驗證後選出的最佳 θ(0.95) -> 準確率可達0.94 -> 不見得好，我是覺得只是這個theta剛好配這個 dataset 而已


`svdd` : 在分類前多加一層 SVDD 把明顯不屬於已知分佈的 sample 過濾掉，再接著用原始的分類分群方法 -> 效果不好，反而會過濾掉表現穩定的以知 label 數據


`Prediction Vs. One-vs-Rest Rejection` : 

預設的 SVM 是用 **margin-based** -> 分類時兩個最大機率 label 相減 -> 用這個機率決定是否分類 (**看相對性**)  

**One-vs-rest rejection** -> 另一種選擇機制，最大機率大於某個門檻才分類，否則分群 -> 對自己的label信心比較高 (**看絕對性**)-> 在一樣SVM+kmeans的情況下和同參數下(θ=0.9)，可以提高5%準確率


## Note
*1. 程式碼不用去跑 也不用每段trace過 我寫的很亂(主要都是以印數據出來就好為主)加上還要裝tensorflow有的沒的 -> 直接去根據原理和實驗結果理解在做什麼就好 如果真的要去跑看看的話 用command跑(可以設定argument)*

*2. confusion matrix照著做出來就好(報告) 看不懂要表達啥也沒差 我上台和在報告裡會解釋*

*3. baseline不用管他 會這樣取是因為原本要先把baseline做出來再用深度學習提高準確率 但失敗了 所以不用特別提那些是baseline*

*4. 數據圖很亂只是給你們看而已 不管是confusion matrix還是accuracy都不要用截圖 做成表格放到報告裡*

*5. 除了基本原理方法外 實驗數據的部分做海報/書面時如果有不知道怎麼解釋的地方把架構寫好空著留給我寫就好*

Adding other comments here in yourself...
