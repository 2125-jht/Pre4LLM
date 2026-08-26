
256his 512dim 4layers

- based on v3_mask_512_no_enc

- 保持去掉enc

- 实现hard gsu，在生成第2级 第3级时挑选序列为sid0相同的

- 去掉layer norm


TODO

- dec改成moe 2/2结构

- 对于user static info，把mlp的隐藏层由2dim改为4dim

- 要不要在一开始只筛选有sid的item，还是说一开始的也不影响，这考虑到是否要加sid的多级特征

- 这里选的时候 userinfo是始终为true吗

- 选了之后 还是需要全部计算然后mask 还是可以直接gather一下 只算有效的 那batch怎么处理呢
