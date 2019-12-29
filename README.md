# word2vec_skipgram  
  记一次自己手写加训练基于负采样的skip-gram的Word2vec。

## 数据获取
  从wki上获取了原始中文xml格式的数据，大约1.82G。  

## 文本转换
  通过官方链接提供的文本抽取工具WikiExtractor.py，共抽取了1082567条词条。  
  
## 训练样本抽取
  由于数据量对于自己的模型太大，无法训练。便根据pku_sim_test.txt文件中的词频，概率抽取了140M的数据。  
  最终数据量如下：  
  句子总数：514078  
  总词数：21370318  
  （unique）词数：829062  
  
## 基于负采样的skip-gram

### 建立词典
  词频词典、word2id字典、id2word字典  

### 下采样
![sub_sampling](https://github.com/zyDotwei/word2vec_skipgram/blob/master/image/sub_sampling.png)

### Negative Sampling
![neg](https://github.com/zyDotwei/word2vec_skipgram/blob/master/image/neg.png)

## 实验结果

### 可视化
![caodai](https://github.com/zyDotwei/word2vec_skipgram/blob/master/image/chaodai.png)  

### 前10相似短语
![computer](https://github.com/zyDotwei/word2vec_skipgram/blob/master/image/computer.png)  

## 说明
* word2vec中的路径“./data/wiki_zh_mini.txt”为分词后的语料文档。（由于太大[限制100M]，无法上传）  
* 所有的输出都放在的output文件中。  
* 中间文件下采样后的数据、根据窗口采样后的数据也都保存下来了。  
* 使用了多GPU进行训练。

## 存在的问题
* 效率很低，只适合小语料（玩）。    
* 相似度存在负值，不知道为什么（虽然应该有），但gensim却没有。  


