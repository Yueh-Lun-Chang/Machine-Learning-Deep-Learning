# Machine Learning & Deep Learning | 機器學習 & 深度學習     
Notes & Eamples
## 基本概念

#### 一、專家系統 : 人類可理解的確切公式                                                           
#### 二、機器學習 : 基於經驗，從輸入、輸出找出公式 ( 傳統統計 )                                       
- 擅長處理結構化資料              
- 資料量至少千筆以上，並依問題難度調整                
- 幫我們從數字中找出映射 ( Mapping )          
- 模型 : 基於統計        
  * 淺層，Output形式有限 => 深度學習               

#### 三、深度學習 : 模擬大腦 ( 神經網路 )，專注在圖像、語言領域 ( 抽象性 )                                      
- 擅長處理非結構化資料
- 資料量至少萬筆以上，並依問題難度調整
- 翻譯 ( ex.語意 )
- Input通常為高維度的數值，Output通常為相對低維的結果
- 模型 : 黑盒子

##### (一) 知名學派                 
  * 連結學派: 神經網路的串聯，代表人物: Geoffrey Hinton               
  * 貝葉斯學派: 基於機率，代表人物: Michael Jordan                 
  * 類比學派: 衡量誰跟誰比較像，可與神經網路結合，類似分群概念
  
##### (二) 機器學習 vs 深度學習
![GITHUB](https://i.imgur.com/sz6MJcI.jpg "機器學習 vs 深度學習")

## 監督式學習、非監督式學習、強化學習
#### 一、監督式學習 ( Supervised Learning ) : 必須有標籤 ( 答案 ) * 沒有教的機器就不會                     
- 離散: 分類 ( Classification ) 
- 連續: 迴歸 ( Regression )
#### 二、非監督式學習 ( Unsupervised Learning ) : 沒有標籤 ( 答案 )；情形1: 資料太多；情形2: 不知道                  
- 分群 ( Clustering )
#### 三、半監督式學習 : 前半給定少量標籤；後半自動產生大量且精準的標籤
#### 四、強化學習 ( Reinforcement Learing ) : 制約，在意"行為控制"的過程
- 機器人、Game

## Perceptron ( 感知器 )       
#### 活化函數 ( 激活函數，Activation Function ) : 將原來要變為線性迴歸的神經元，加入非線性的概念
- 選擇要件: 
  * 通常計算開銷要夠小 ( 太大: log、exp、tanh)
  * 盡可能覆蓋值域
- 重要的活化函數種類:
  * Sigmoid ( S函數 ) : 範圍為0~1；類似人類大腦，一次微分為常態分配；缺點為易有梯度消失問題、不是零中心而是以0.5為中心、計算開銷大     
  * Softmax : 範圍為0~1
  * ReLU : 最常用的活化函數，以0為中心，計算開銷小；缺點為轉折處 ( 點 ) 無法微分，梯度為常數，把那些小於0的特徵去掉
  * Leaky ReLU : 特性為給小於0的值微小斜度，不管數值正負都適用；缺點為轉折處 ( 點 ) 無法微分
  * Exponential ReLU : 為Leaky ReLU變體；缺點為計算開銷大
  * tanh :  範圍為-1~1；特性為以0為中心；常用時機為像素類問題、表徵學習
  * swish : 特性為遞減再遞增 ( 以往的活化函數均為單調遞增或單調遞減 )
  * SELU : 運用不動點定理，將散開的梯度資型收回，形成自我調節的ReLU
  
### 萬能建模定理 ( Universal approximation theorem )                      
#### 前饋神經網路若具有線性輸出層，和至少一層具任一種擠壓性質的活化函數隱藏層，只要給予網路足夠的隱藏神經元，它便可以任意精度來近似任何從一個有限維度空間到另一個有限維度空間的映射函數。簡單的說: 神經網路可以擬合各種神奇的函數                     

- 衍伸問題: 過擬合，必須採取許多措施防止 ( ex.數據增強: 樣本永遠不重複 )                  
- 推翻此定理: resnet

## CNN ( Convolutional Neural Networks，卷積神經網路)         
![GITHUB](https://i.imgur.com/sz5djYr.jpg "CNN-concept")
![GITHUB](https://i.imgur.com/VCrpmmg.jpg "CNN")

## Transfer Learning ( 遷移學習 ) 
#### 使用與問題主題相關、別人已訓練好的卷積模型，再加上自己搭建的MLP。          
- tensorflow.keras.applications裡有許多著名模型                       
![GITHUB](https://i.imgur.com/OrhBvi1.jpg "Transfer Learning")
#### 一、優點
- 資料不需要太多: 因為資料主要是拿來訓練MLP，因此約百~千筆就足夠
- 訓練時間減少: 不用太強的運算能力
#### 二、步驟
- step1: 固定CNN參數
- step2: 訓練MLP
- step3: 打開CNN，完整微調

## Examples
### Machine Learning 機器學習
- [Classification](https://github.com/Yueh-Lun-Chang/Machine-Learning-Deep-Learning/blob/master/Classification_review.ipynb)
- [Cluster](https://github.com/Yueh-Lun-Chang/Machine-Learning-Deep-Learning/blob/master/Cluster_review.ipynb)
- [Regression](https://github.com/Yueh-Lun-Chang/Machine-Learning-Deep-Learning/blob/master/Regression_review.ipynb)
- [Linear_regression_House_price](https://github.com/Yueh-Lun-Chang/Machine-Learning-Deep-Learning/blob/master/Linear_regression_House_price.ipynb)
- [Naive_bayes_Poem](https://github.com/Yueh-Lun-Chang/Machine-Learning-Deep-Learning/blob/master/Naive_bayes_Poem_review.ipynb)
- [Randomforest_regressor_House_price](https://github.com/Yueh-Lun-Chang/Machine-Learning-Deep-Learning/blob/master/Randomforest_regressor_House_price_review.ipynb)
- [Randomforest_classifier_KNN_Titanic](https://github.com/Yueh-Lun-Chang/Machine-Learning-Deep-Learning/blob/master/Randomforest_classifier_KNN_Titanic_review.ipynb)

### Deep Learning 深度學習
- [MLP_mnist](https://github.com/Yueh-Lun-Chang/Machine-Learning-Deep-Learning/blob/master/MLP_mnist.ipynb)
- [MLP_fashion_mnist](https://github.com/Yueh-Lun-Chang/Machine-Learning-Deep-Learning/blob/master/MLP_fashion_mnist.ipynb)
- [CNN_cifar10](https://github.com/Yueh-Lun-Chang/Machine-Learning-Deep-Learning/blob/master/CNN_cifar10.ipynb)
- [Transfer_Learning_dog_cat](https://github.com/Yueh-Lun-Chang/Machine-Learning-Deep-Learning/blob/master/Transfer_Learning_dog_cat.ipynb)
- [NLP_Embedding](https://github.com/Yueh-Lun-Chang/Machine-Learning-Deep-Learning/blob/master/NLP_Embedding.ipynb)
