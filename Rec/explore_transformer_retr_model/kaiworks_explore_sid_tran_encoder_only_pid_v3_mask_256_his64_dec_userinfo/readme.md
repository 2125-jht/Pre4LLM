
64his 256dim

- based on v3_mask_256_his64

- 在bos处加上user info

- 训练的时候dec输入不传全部，传最后一个之前的，因为最后一个不会求loss

- enc输入加上位置编码

- enc输入加上token type emb

- enc和dec的输入emb都进行ln和dropout


todo

- user_static_emb 已经经过 MLP 投到同维度，但为了数值尺度更稳，建议在拼接前做 LayerNorm 或者加一个可学习的缩放（gate）：

``` python
user_static_emb = tf.layers.layer_norm(user_static_emb)
或 gate = tf.sigmoid(tf.layers.dense(user_static_emb, self._dim)); user_static_emb *= gate
```

- 如果发现收敛慢，给 user token 一个单独的可学习尺度参数或残差门（上面提到的 gate）通常能改善。

