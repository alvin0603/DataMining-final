# DataMining-final
## 目標
將 RNA-Seq 基因數據分成 Known(分類)和 Unknown(分群)

## 實驗數據：
`Thetacompare` : 對比 θ（分類門檻）在 0.5 和 0.9 下的結果 -> 證實會大程度影響最終結果
`PCAcompare` : 測試降維到不同維度(原始為兩萬多)的結果比較 -> 可討論不同種分類方法所適用的維度
`best_result_with_kFold` : 經過交叉驗證後選出的最佳 θ(0.95) -> 準確率可達0.94
`svdd` : 在分類前加一層 SVDD 把明顯不屬於已知分佈的 sample 過濾掉，再接著用原始的分類分群方法 -> 效果不好，反而會過濾掉表現穩定的以知 label 數據
`Prediction Vs. One-vs-Rest Rejection` : 
預設的 SVM 是用 **margin-based** -> 分類時兩個最大機率 label 相減 -> 用這個機率決定是否分類 (**看相對性**)  
**One-vs-rest rejection** -> 另一種選擇機制，最大機率大於某個門檻才分類，否則分群 -> 對自己的label信心比較高 -> 在一樣SVM+kmeans的情況下和同參數下(θ=0.9)，可以提高5%準確率



confusion 
baseline
trace
數據圖
