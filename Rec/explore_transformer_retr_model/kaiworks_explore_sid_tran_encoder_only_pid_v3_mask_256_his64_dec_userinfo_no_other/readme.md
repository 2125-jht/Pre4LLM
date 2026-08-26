
64his 256dim

- based on v3_mask_256_his64_dec_userinfo

- 去掉下面
<!-- - enc输入加上位置编码

- enc输入加上token type emb

- enc和dec的输入emb都进行ln和dropout -->

- 相当于对齐 v3_mask_256_his64 然后在bos处加上user info
