

用这个多兴趣模型进行改成encoder 最后加一个token 做召回

- token拼在最后 然后encoder输出的之后取最后这个token用于召回

- photo侧没有mlp，直接sum pooling特征embedding
