

decoder-only的结构 64his+256dim+4layers

- 选用colossus中最近的64个item的sid

- 筛选3s以上有交互or7s以上的序列

- 训练改为都作为训练样本求loss反向传播



todo

- 先进model再进行筛选逻辑和解码逻辑变为3级token（这里要拼接上label的3级）

- 解码的时候predlayer共用的问题，怎么共用更好做

- 打出总的loss，和最后三级的loss之和，以及pred各项和其他对齐，便于对比

- 实现kvcache在infer侧

- 看下是否要重构结构，对齐有kvcache那一版？

