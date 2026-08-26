
todo

- 加sid特征
- 构建decoder only
    - 统计sid的有效数量，和筛选后数量

- 这里的sid list是否要选择的时候对齐profile？

- 验证scaling raw
    - 包括层数
    - 和dim
    - 以及改变结构 用moe 可以先尝试把ffn的hidden dim改大

- 求long short的相似度的时候，取sid的时候，要记得mask为0的 然后再取第一级求emb
- long short用1000之后可以改一下去掉传profile的sid了，因为只用了colossus
- 统一为colossus后 train那里就可以不执行两次click的编码了


optimize

- mask改为int8
- 输入的用于辅助筛选的特征list改成int32
- 训练的时候不用decoder输入不用传全部，最后一个不会求loss
- softmax_cross_entropy_with_logits 改成 v2 -> 改成 sparse_softmax_cross_entropy_with_logits
- 重构decoder only infer 改成第一次全部前向存储kv 而不是逐step
    - 前面的序列可以不用存储了 只留下kv cache？
- 查看infer的dense内容和type
- 查看gender内容
- 训练侧这里计算probs为了看指标 然后计算loss的时候又会算一遍 这里可以只用一次吗 softmax传算好的probs
- moe的地方改为只在训练时候算负载均衡loss 加一个逻辑判断（会影响图吗？还是判断后也算loss 直接写0）
- hard neg如果确定只打压sid0之后，可以把neg样本的前向传播改成只有第一级前向，避免多余的计算（不过这样的话看不到后面几级neg的probs和recall）
- mask的地方的repeat可以改成广播的形式
- 优化transpose 看起来很耗时 不要用太多transpose 换成其他的？或者减少来回变换？用view？
- 最后一次解码后不用再gather cache了
- 优化gather 把gather_nd换成gather?

tips

- decoder only 检查dsl传的colossus内容的顺序
- decoder only padding位置思考
