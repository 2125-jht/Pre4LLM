import tensorflow as tf


# def getLast(a, n):
#     # a: [bs, len]
#     def process_row(row):
#         # 获取非零元素及其索引
#         non_zero_indices = tf.where(tf.not_equal(row, 0))[:, 0]
#         non_zero_values = tf.gather(row, non_zero_indices)

#         # 取最后 n 个非零值（顺序保持）
#         last_n_values = non_zero_values[-n:]

#         # 计算需要补多少个 0
#         pad_len = tf.maximum(0, n - tf.shape(last_n_values)[0])
#         padded = tf.pad(last_n_values, [[pad_len, 0]])  # 前面补零

#         return padded  # shape: [n]

#     result = tf.map_fn(process_row, a, dtype=tf.int32)
#     return result  # shape: [bs, n]

def getLast(a, n):
    """
    取每行最后 n 个 **非 -1** 的元素，不足 n 个时在左侧补 -1。

    参数
    ----
    a : tf.Tensor, shape = [bs, len], int32  
        序列张量，-1 表示 padding。
    n : int  
        需要保留的有效元素个数。

    返回
    ----
    tf.Tensor, shape = [bs, n], int32  
        每行长度为 n；左侧用 -1 填充。
    """
    def process_row(row):
        # 1) 找到非 0 的位置
        valid_idx   = tf.where(tf.not_equal(row, 0))[:, 0]   # [?]
        valid_vals  = tf.gather(row, valid_idx)               # [?]

        # 2) 取最后 n 个
        last_n_vals = valid_vals[-n:]                         # [<=n]

        # 3) 如不足 n 个，左侧补 -1
        pad_len = tf.maximum(0, n - tf.shape(last_n_vals)[0])
        padded  = tf.pad(last_n_vals,
                         paddings=[[pad_len, 0]],
                         constant_values=-1)                  # [n]

        return padded

    return tf.map_fn(process_row, a, dtype=tf.int64)          # [bs, n]


# def processInput(input_tensor, vocab_size=[8192, 8192, 8192]):
#     # originInput [bs, len]
#     # 拆分操作
#     a = tf.bitwise.right_shift(input_tensor, 30) # 右移 30 位，取出 a
#     b = tf.bitwise.right_shift(input_tensor, 15)  # 右移 15 位，取出 b
#     c = input_tensor                          # 直接取出 c

#     # 使用 bitwise_and 进行掩码操作
#     a = tf.bitwise.bitwise_and(a, 0x7FFF)  # 取出 a
#     b = tf.bitwise.bitwise_and(b, 0x7FFF) + vocab_size[0]  # 取出 b
#     c = tf.bitwise.bitwise_and(c, 0x7FFF) + vocab_size[0] + vocab_size[1]  # 取出 c

#     output_tensor = tf.stack([a, b, c], axis=-1)  # 形状为 [batch, len, 3]
#     output_tensor = tf.reshape(output_tensor, [-1, 3 * tf.shape(input_tensor)[1]])  # 
#     output_tensor = tf.cast(output_tensor, tf.int32)   
#     return output_tensor


def processInput(input_tensor,
                 vocab_size=(8192, 8192, 8192),
                 pad_val=-1):
    """
    把 64-bit sid 切成 3 段并映射到同一 embedding 索引：
        a ∈ [0,     8191]                 (高 15 位)
        b ∈ [8192,  16383]  = + 8192      (中 15 位)
        c ∈ [16384, 24575]  = + 16384     (低 15 位)

    其中 -1 视为 padding，结果置为 [-1,-1,-1]。

    Parameters
    ----------
    input_tensor : tf.Tensor(int64)   shape = [B, L]
    vocab_size   : tuple(int)         (8192, 8192, 8192)
    pad_val      : int               padding 标记

    Returns
    -------
    tf.Tensor(int32)  shape = [B, L*3]
    """
    # ---------- 1) mask ----------
    pad_mask  = tf.equal(input_tensor, pad_val)     # [B,L] bool
    pad_mask3 = tf.expand_dims(pad_mask, -1)        # [B,L,1]
    pad_mask3 = tf.tile(pad_mask3, [1, 1, 3])       # [B,L,3]  <─★ 扩维

    a = tf.bitwise.right_shift(input_tensor, 30) # 右移 30 位，取出 a
    b = tf.bitwise.right_shift(input_tensor, 15)  # 右移 15 位，取出 b
    c = input_tensor                          # 直接取出 c

    # 使用 bitwise_and 进行掩码操作
    a = tf.bitwise.bitwise_and(a, 0x7FFF)  # 取出 a
    b = tf.bitwise.bitwise_and(b, 0x7FFF) + vocab_size[0]  # 取出 b
    c = tf.bitwise.bitwise_and(c, 0x7FFF) + vocab_size[0] + vocab_size[1]  # 取出 c

    out = tf.stack([a, b, c], axis=-1)                   # [B,L,3]

    # ---------- 3) padding → -1 ----------
    pad_val_i64 = tf.cast(pad_val, input_tensor.dtype)
    out = tf.where(pad_mask3,
                tf.fill(tf.shape(out), pad_val_i64),   # [B,L,3]
                out)                                   # [B,L,3]

    out = tf.reshape(out, [tf.shape(input_tensor)[0], -1])
    return tf.cast(out, tf.int32)


def processLabel(input_tensor):
    a = tf.bitwise.right_shift(input_tensor, 30) # 右移 30 位，取出 a
    b = tf.bitwise.right_shift(input_tensor, 15)  # 右移 15 位，取出 b
    c = input_tensor                          # 直接取出 c

    # 使用 bitwise_and 进行掩码操作
    a = tf.bitwise.bitwise_and(a, 0x7FFF)  # 取出 a
    b = tf.bitwise.bitwise_and(b, 0x7FFF)  # 取出 b
    c = tf.bitwise.bitwise_and(c, 0x7FFF)  # 取出 c

    output_tensor = tf.stack([a, b, c], axis=-1)  # 形状为 [batch, len, 3]
    output_tensor = tf.reshape(output_tensor, [-1, 3 * tf.shape(input_tensor)[1]])  # 
    output_tensor = tf.cast(output_tensor, tf.int32)

    return output_tensor

def processOutputV2(input_tensor):
    ### input: [batch, 4*dim]
    input_tensor = tf.cast(input_tensor, tf.int64)
    x0 = input_tensor[:, :, 0]
    x1 = input_tensor[:, :, 1] - 8192
    x2 = input_tensor[:, :, 2] - 8192*2
    output = tf.bitwise.left_shift(x0, 26) + tf.bitwise.left_shift(x1, 13) + x2
    return output

def getActionList(user_colossus_label_list):
    # ---- 1. 每个 bit 的权值 ----
    LIKE                  = 1 << 0   # 0000 0001
    FOLLOW                = 1 << 1   # 0000 0010
    FORWARD               = 1 << 2   # 0000 0100
    HATE                  = 1 << 3   # 0000 1000
    COMMENT               = 1 << 4   # 0001 0000
    COM_INTENT            = 1 << 5   # 0010 0000
    ENTER_PROFILE         = 1 << 6   # 0100 0000
    COM_INTENT_V2         = 1 << 7   # 1000 0000
    COMMENT_STAY_TIME_VAL = 1 << 8   # 1 0000 0000

    # ---- 2. 解码出 0/1 标签 ----
    def bit_to_bool(x, mask):
        """返回和输入同 shape 的 bool 张量，指示该 bit 是否为 1。"""
        # 确保输入是整数类型，bitwise操作不支持浮点数
        # x = tf.cast(x, tf.int64)
        # mask = tf.cast(mask, tf.int64)
        return tf.not_equal(tf.bitwise.bitwise_and(x, mask), 0)

    like                  = bit_to_bool(user_colossus_label_list, LIKE)
    follow                = bit_to_bool(user_colossus_label_list, FOLLOW)
    forward               = bit_to_bool(user_colossus_label_list, FORWARD)
    hate                  = bit_to_bool(user_colossus_label_list, HATE)
    comment               = bit_to_bool(user_colossus_label_list, COMMENT)
    comm_intent           = bit_to_bool(user_colossus_label_list, COM_INTENT)
    enter_profile         = bit_to_bool(user_colossus_label_list, ENTER_PROFILE)
    comm_intent_v2        = bit_to_bool(user_colossus_label_list, COM_INTENT_V2)
    comment_stay_time_val = bit_to_bool(user_colossus_label_list, COMMENT_STAY_TIME_VAL)

    # ---- 3. 计算 user_colossus_action_list ----
    # 规则：like / follow / forward / comment / enter_profile 其中之一为 1 → action=1
    user_colossus_action_list = tf.cast(
        like | follow | forward | comment | enter_profile,
        tf.int64                       # 结果 0/1
    )

    # 现在：
    # - like, follow, … comment_stay_time_val  都是 [B,1000] 的 bool张量
    # - user_colossus_action_list 同样 shape，0/1 表示是否命中“用户有交互”
    
    return user_colossus_action_list