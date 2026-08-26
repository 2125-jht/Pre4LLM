
256his 256dim

- based on v3_mask_256

- 进行了select_size的选择，并筛选了sid有效的item

- 在enc侧对于item加上三级sid的特征，sid的三级特征先相加，再和之前的特征concat，然后过mlp

- 对于user static info，把mlp的隐藏层由2dim改为了4dim
