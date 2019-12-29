# word2vec_skipgram  
  记一次自己手写加训练基于负采样的skip-gram的Word2vec。

## 数据获取
  从wki上获取了原始中文xml格式的数据，大约1.82G.

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


### Negative Sampling

## 实验结果

### 可视化

### 前10相似短语



