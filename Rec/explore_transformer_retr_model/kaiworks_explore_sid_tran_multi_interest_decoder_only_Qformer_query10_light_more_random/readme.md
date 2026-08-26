
- 多兴趣生成式模型，coarse_interest_model输入多个query token输出用户的多兴趣token，将输出的兴趣token输入到fine_item_model中，自回归的输出sid0,sid1,sid2
    - 训练时，将ground truth的sid序列进行sum pooling与多个interest token计算余弦相似度，只计算余弦相似度最高的interest token输入到fine_item_model的loss(修改为随机选择，作为对比实验)
    - 推理时，将coarse_interest_model输出的兴趣token作为输入，自回归的生成sid序列（从而大幅度降低beam search的时间复杂度）

- coarse_interest_model使用qformer结构
- fine_item_model使用decoder only结构
- 多个fine_item_model参数share，训练推理串行

- coarse_interest_model=4, fine_item_model=1, query token numb=10，beamsize=50，推理产出10*50=500

- input是 sid0,sid1+8192,sid2+8192+8192,label没有加上8192，就是sid0,sid1,sid2。在训练的时候输出的dense层是0-8192的范围，所以可以直接进行softmax。而在推理时输出的dense层是0-8192的范围，但是在保存结果时会加上对应的offset，所以推理得到的是sid0,sid1+8192,sid2+8192+8192
