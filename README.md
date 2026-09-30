# Word2Vec 编程练习

这是自然语言处理课程的 Word2Vec 编程练习。目标不是重新实现一套 Word2Vec，而是完成一次完整的“理解原理 → 运行代码 → 修改参数 → 观察结果”的实验。

## 1. 实验做了什么

本实验使用 **Gensim** 的 Word2Vec 实现，在其自带的 **Lee 英文新闻语料**（300 篇新闻）上训练词向量，并比较：

- CBOW 与 Skip-gram；
- 词向量维度 50 / 100；
- 上下文窗口 2 / 5 / 10；
- 训练轮数 5 / 50。

训练后使用 WordSim-353 中当前词表可以覆盖的词对做简单评价。这里的相关系数越大，说明模型给出的词语相似程度与人工评分越一致。

## 2. 主要结果

| 编号 | 模型 | 维度 | window | epochs | WordSim-353 相关系数 | 平均训练时间/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E1 | CBOW | 100 | 5 | 50 | 0.193 | 2.47 |
| E2 | Skip-gram | 100 | 5 | 50 | 0.240 | 10.38 |
| E3 | Skip-gram | 50 | 5 | 50 | 0.212 | 11.78 |
| E4 | Skip-gram | 100 | 2 | 50 | 0.166 | 5.60 |
| E5 | Skip-gram | 100 | 10 | 50 | 0.207 | 18.31 |
| E6 | Skip-gram | 100 | 5 | 5 | -0.218 | 1.04 |

这组小语料实验中，E2 的结果较好；将训练轮数从 50 减到 5 后，效果明显下降。扩大窗口也会增加训练时间。由于语料较小，这些结果只用于课程实验中的参数观察，不代表 Word2Vec 在大型语料上的一般结论。

## 3. 运行方法

~~~bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
python train_word2vec.py
~~~

训练结束后会生成：

- `results.csv`：本次运行的实验结果；
- `word2vec_e2.model`：E2 配置训练得到的模型。

查询某个词的近邻：

~~~bash
python demo.py --word afghanistan
~~~

## 4. 关键参数

- `vector_size`：每个词用多少维向量表示；
- `window`：训练时观察目标词前后多大的上下文范围；
- `min_count=5`：出现次数少于 5 的词不加入词表；
- `sg=0`：CBOW；`sg=1`：Skip-gram；
- `epochs`：完整遍历训练语料的次数。

## 5. 一个实际遇到的问题

经典示例 `king - man + woman ≈ queen` 在本实验中没有成功。检查语料后发现 `queen` 只出现 4 次，被 `min_count=5` 过滤，因此根本不在候选词表中。这个现象也说明：代码能运行并不意味着小语料一定能学习到理想的语义关系。

## 6. 文件说明

- `train_word2vec.py`：训练、参数对照和 WordSim-353 评价；
- `demo.py`：加载模型并查询相似词；
- `results.csv`：已完成实验的汇总结果；
- `requirements.txt`：Python 依赖。

实验报告中的结果来自实际运行记录。
