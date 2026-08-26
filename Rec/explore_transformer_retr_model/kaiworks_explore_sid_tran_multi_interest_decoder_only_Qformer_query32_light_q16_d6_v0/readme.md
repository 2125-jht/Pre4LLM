
- 多兴趣生成式模型，coarse_interest_model输入多个query token输出用户的多兴趣token，将输出的兴趣token输入到fine_item_model中，自回归的输出sid0,sid1,sid2
    - 训练时，将ground truth的sid序列进行sum pooling与多个interest token计算余弦相似度，只计算余弦相似度最高的interest token输入到fine_item_model的loss
    - 推理时，将coarse_interest_model输出的兴趣token作为输入，自回归的生成sid序列（从而大幅度降低beam search的时间复杂度）

- coarse_interest_model使用qformer结构
- fine_item_model使用decoder only结构
- 多个fine_item_model参数share，训练推理串行

- coarse_interest_model=5, fine_item_model=2, query token numb=5，beamsize=128，推理产出5*128
