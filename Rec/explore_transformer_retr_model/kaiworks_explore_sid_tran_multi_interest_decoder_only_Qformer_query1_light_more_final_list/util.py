import tensorflow as tf


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


def processListLabel(input_tensor):
    a = tf.bitwise.right_shift(input_tensor, 30) # 右移 30 位，取出 a
    a = tf.bitwise.bitwise_and(a, 0x7FFF)  # 取出 a
    output_tensor = tf.cast(a, tf.int32)
    return output_tensor


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
    x1 = input_tensor[:, :, 1] - 8192
    x2 = input_tensor[:, :, 2] - 8192*2
    output = tf.bitwise.left_shift(x0, 26) + tf.bitwise.left_shift(x1, 13) + x2
    return output
