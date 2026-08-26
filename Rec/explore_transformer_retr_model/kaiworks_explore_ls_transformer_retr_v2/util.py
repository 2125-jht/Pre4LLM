import tensorflow as tf

def truncate_colossus_feature_v2(all_param_dict, feature_emb_size_dict, dense_feature_dict, size=768):
    sparse_fea_name = [
        "user_colossus_pid_list",
        "user_colossus_aid_list",
        "user_colossus_channel_list"
    ]

    dense_fea_name = [
        "colossus_play_time_list",
        "colossus_label_list",
        "colossus_duration_list",
        "colossus_channel_list"
    ]

    batch = tf.shape(all_param_dict["user_colossus_pid_list"])[0]
    length = tf.shape(all_param_dict["user_colossus_pid_list"])[1]
    dim = tf.shape(all_param_dict["user_colossus_pid_list"])[2]

    emb_end = feature_emb_size_dict["user_colossus_pid_list"] #[batch]
    emb_start = emb_end - size
    emb_start = tf.where(emb_start >= 0, emb_start, tf.zeros_like(emb_start))

    index_range = tf.expand_dims(tf.range(size), axis=0)
    index_range = tf.tile(index_range, [batch, 1])
    indices = index_range + tf.expand_dims(emb_start, axis=1)

    batch_indices = tf.expand_dims(tf.range(batch), axis=-1)  # [batch, 1]
    batch_indices = tf.tile(batch_indices, [1, size]) 

    flat_indices = batch_indices * length + indices

    flat_indices_reshaped = tf.reshape(flat_indices, [-1]) #[batch * size]

    for fea in sparse_fea_name:
        fea_flat = tf.reshape(all_param_dict[fea], [-1, dim]) #[batch * len, dim]
        all_param_dict["truncate_" + fea] = tf.reshape(tf.gather(fea_flat, flat_indices_reshaped), [batch, size, -1])
        feature_emb_size_dict["truncate_" + fea] = size

    truncate_dense_feature = {}
    for fea in dense_fea_name:
        fea_flat = tf.reshape(dense_feature_dict[fea], [-1, 1]) #[batch * len, 1]
        truncate_dense_feature["truncate_" + fea] = tf.reshape(tf.gather(fea_flat, flat_indices_reshaped), [batch, -1]) #[batch, size]

    return truncate_dense_feature

def processColossusFeature(config, all_param_dict, feature_emb_size_dict, history_size=768):
    user_colossus_fea_names = [
        "user_colossus_pid_list",
        "user_colossus_aid_list",
        "user_colossus_channel_list"
    ]   

    user_colossus_dense_feas = [
        "colossus_play_time_list",
        "colossus_label_list",
        "colossus_duration_list",
        "colossus_channel_list"
    ]

    dense_feature_dict = {}

    for fea in user_colossus_dense_feas:
        dense_fea = config.get_dense_fea(fea, dim=1000, dtype=tf.int64)
        dense_feature_dict[fea] = dense_fea

    truncate_dense_feature = truncate_colossus_feature_v2(all_param_dict, feature_emb_size_dict, dense_feature_dict, history_size)
    return truncate_dense_feature


def getLast(a, n):
    # a: [bs, len]
    def process_row(row):
        # 获取非零元素及其索引
        non_zero_indices = tf.where(tf.not_equal(row, 0))[:, 0]
        non_zero_values = tf.gather(row, non_zero_indices)

        # 取最后 n 个非零值（顺序保持）
        last_n_values = non_zero_values[-n:]

        # 计算需要补多少个 0
        pad_len = tf.maximum(0, n - tf.shape(last_n_values)[0])
        padded = tf.pad(last_n_values, [[pad_len, 0]])  # 前面补零

        return padded  # shape: [n]

    result = tf.map_fn(process_row, a, dtype=tf.int32)
    return result  # shape: [bs, n]


def processInput(input_tensor, vocab_size=[8192, 8192, 8192]):
    # originInput [bs, len]
    # 拆分操作
    a = tf.bitwise.right_shift(input_tensor, 30) # 右移 24 位，取出 a
    b = tf.bitwise.right_shift(input_tensor, 15)  # 右移 16 位，取出 b
    c = input_tensor                          # 直接取出 c

    # 使用 bitwise_and 进行掩码操作
    a = tf.bitwise.bitwise_and(a, 0x7FFF)  # 取出 a
    b = tf.bitwise.bitwise_and(b, 0x7FFF) + vocab_size[0]  # 取出 b
    c = tf.bitwise.bitwise_and(c, 0x7FFF) + vocab_size[0] + vocab_size[1]  # 取出 c

    output_tensor = tf.stack([a, b, c], axis=-1)  # 形状为 [batch, len, 4]
    output_tensor = tf.reshape(output_tensor, [-1, 3 * tf.shape(input_tensor)[1]])  # 
    output_tensor = tf.cast(output_tensor, tf.int32)   
    return output_tensor

def processLabel(input_tensor):
    a = tf.bitwise.right_shift(input_tensor, 30) # 右移 24 位，取出 a
    b = tf.bitwise.right_shift(input_tensor, 15)  # 右移 16 位，取出 b
    c = input_tensor                          # 直接取出 c

    # 使用 bitwise_and 进行掩码操作
    a = tf.bitwise.bitwise_and(a, 0x7FFF)  # 取出 a
    b = tf.bitwise.bitwise_and(b, 0x7FFF)  # 取出 b
    c = tf.bitwise.bitwise_and(c, 0x7FFF)  # 取出 c

    output_tensor = tf.stack([a, b, c], axis=-1)  # 形状为 [batch, len, 4]
    output_tensor = tf.reshape(output_tensor, [-1, 3 * tf.shape(input_tensor)[1]])  # 
    output_tensor = tf.cast(output_tensor, tf.int32)

    return output_tensor

def processOutputV2(input_tensor):
    ### input: [batch, 4*dim]
    input_tensor = tf.cast(input_tensor, tf.int64)
    x0 = input_tensor[:, :, 0]
    x1 = input_tensor[:, :, 1]
    x2 = input_tensor[:, :, 2]
    output = tf.bitwise.left_shift(x0, 26) + tf.bitwise.left_shift(x1, 13) + x2
    return output