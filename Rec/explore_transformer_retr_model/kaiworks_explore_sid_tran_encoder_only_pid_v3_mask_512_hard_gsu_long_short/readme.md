
256his 512dim 4layers

- based on v3_mask_512_hard_gsu

- 保持去掉layer norm

- 实现长短期，用label的sid0的emb，和action list的sid0的emb的pooling，求相似度，然后选长or短

- 对于生成sid1/sid2的时候选择在colossus里面同sid0的item


TODO

- dec改成moe 2/2结构

- 对于user static info，把mlp的隐藏层由2dim改为4dim

- 要不要在一开始只筛选有sid的item，还是说一开始的也不影响，这考虑到是否要加sid的多级特征

- 这里选的时候 userinfo是始终为true吗

- 选了之后 还是需要全部计算然后mask 还是可以直接gather一下 只算有效的 那batch怎么处理呢


Tips

- 这里就更得选一下了先gather或者怎么做选择一下了 不然需要太长了？但是是cross attention好像还行 不占用太多？

- 这里share id，infer的流程不要写错了，记得写成share的

- 目前的实现 为了能并行 把输入都concat到一起的 在序列维度上 然后要mask来取用什么

- infer这里 为了能不改cache 先模拟一次前向得到encoder kv cache，然后在分别走long和short，那其实直接走也是一样 cache先存下来，然后后面拼接

- 后面可以优化，长短期用的是不同的mask，拼到一起一起前向？或者改 cache，让short可以用long的cache，可以先long然后清掉self，只用cross kv

- 选择哪些过mlp也可以优化，目前是colossus 1000都要过mlp

- 现在要和1000个求attention然后mask，改成可以直接mask或者gather再求的形式

- 现在是的最终输入的emb是profile由远到近，colossus是由近到远，可以后面改成profile也由近到远，这样就可以不用一开始reverse了

- mlp的输入和输出都置0然后传进enc 会不会更好

- 还可以优化，只对能选出来的colossus fea做mlp，然后再scatter回去，后面再选？这样会快吗？选了之后shape不固定了？还是要padding？padding 0会加快速度吗？

- train先第一级，然后二三级并行forward不方便实现cache后的并行，所以还是step的三级