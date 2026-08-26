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


def processInput(input_tensor, vocab_size=[16, 32, 32, 32]):
    # originInput [bs, len]
    # 拆分操作
    a = tf.bitwise.right_shift(input_tensor, 24) # 右移 24 位，取出 a
    b = tf.bitwise.right_shift(input_tensor, 16)  # 右移 16 位，取出 b
    c = tf.bitwise.right_shift(input_tensor, 8) # 右移 8 位，取出 c
    d = input_tensor                          # 直接取出 d

    # 使用 bitwise_and 进行掩码操作
    a = tf.bitwise.bitwise_and(a, 0xFF)  # 取出 a
    b = tf.bitwise.bitwise_and(b, 0xFF) + vocab_size[0]  # 取出 b
    c = tf.bitwise.bitwise_and(c, 0xFF) + vocab_size[0] + vocab_size[1]  # 取出 c
    d = tf.bitwise.bitwise_and(d, 0xFF) + vocab_size[0] + vocab_size[1] + vocab_size[2]     # 取出 d

    output_tensor = tf.stack([a, b, c, d], axis=-1)  # 形状为 [batch, len, 4]
    output_tensor = tf.reshape(output_tensor, [-1, 4 * tf.shape(input_tensor)[1]])  # 

    return output_tensor

def processLabel(input_tensor):
    a = tf.bitwise.right_shift(input_tensor, 24) # 右移 24 位，取出 a
    b = tf.bitwise.right_shift(input_tensor, 16)  # 右移 16 位，取出 b
    c = tf.bitwise.right_shift(input_tensor, 8) # 右移 8 位，取出 c
    d = input_tensor                          # 直接取出 d

    # 使用 bitwise_and 进行掩码操作
    a = tf.bitwise.bitwise_and(a, 0xFF)  # 取出 a
    b = tf.bitwise.bitwise_and(b, 0xFF)  # 取出 b
    c = tf.bitwise.bitwise_and(c, 0xFF)  # 取出 c
    d = tf.bitwise.bitwise_and(d, 0xFF)  # 取出 d

    output_tensor = tf.stack([a, b, c, d], axis=-1)  # 形状为 [batch, len, 4]
    output_tensor = tf.reshape(output_tensor, [-1, 4 * tf.shape(input_tensor)[1]])  # 

    return output_tensor

def processOutput(input_tensor):
    ### input: [batch, dim, 4]
    x0 = input_tensor[:, :, 0]
    x1 = input_tensor[:, :, 1]
    x2 = input_tensor[:, :, 2]
    x3 = input_tensor[:, :, 3]

    output = tf.bitwise.left_shift(x0, 24) + tf.bitwise.left_shift(x1, 16) + tf.bitwise.left_shift(x2, 8) + x3
    return output

def processOutputV2(input_tensor):
    ### input: [batch, 4*dim]
    x0 = input_tensor[:, :, 0]
    x1 = input_tensor[:, :, 1]
    x2 = input_tensor[:, :, 2]

    output = tf.bitwise.left_shift(x0, 16) + tf.bitwise.left_shift(x1, 8) + x2
    return output


def mlp(name, net, hidden_units, output_unit=None, activation=tf.nn.relu):
  scope = name + '_mlp'
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      for i, k in enumerate(hidden_units):
          layer_name = scope + '_{}'.format(i)
          net = tf.layers.dense(net, k, activation=activation, name=layer_name)
      if output_unit != None:
          net = tf.layers.dense(net, output_unit, activation=None, name=scope + '_final')
  return net