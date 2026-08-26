import tensorflow as tf
import sys

transform_functions = {
    "x": lambda x, name: x,  # 线性函数
    "3x": lambda x, name: 3 * x,  # 线性函数
    "0.3x": lambda x, name: 0.3 * x,  # 线性函数
    "power3-0.5": lambda x, name: tf.math.pow(x * 0.5, 3, name=name),
    "power3-0.7": lambda x, name: tf.math.pow(x * 0.7, 3, name=name),
    # 上凸函数
    "log": lambda x, name: tf.math.log(5.0 * x + 1.0),
    "sqrt0.3": lambda x, name: tf.math.pow(5.0 * x, 0.3),
    "sqrt0.7": lambda x, name: tf.math.pow(5.0 * x, 0.7),
}

def transform_queues(input_data, transform_functions):
    """
    :param input_data: xtr_sparse, [-1, 1] * queue_num
    :return: [-1, func_num, queue_num]
    """
    input_data = tf.where(tf.is_nan(input_data), tf.zeros_like(input_data), input_data)  # queue_num个[b, 10] => 16*b*10
    input_data = tf.clip_by_value(input_data, 1e-6, 1.0)  # pxtr <= 1.0
    func_num = len(transform_functions)
    with tf.variable_scope("pxtr_trans_funcs", reuse=tf.AUTO_REUSE):
        transformed_queue = [func(input_data, name=name) for name, func in transform_functions.items()]  # [queue_num, b, 10] * func_num
        queues = tf.concat(transformed_queue, axis=-1)  # [queue_num, b, func_num*10]
        queues = tf.transpose(queues, perm=[1, 2, 0])  # [b, func_num*10, queue_num]
    return queues


def auto_ensemble(queues, weight_input, queue_num, transform_functions, name):
    """
    :param queues: [-1, func_num, queue_num]
    :return: [bs, queue_num]
    """
    func_num = len(transform_functions)
    with tf.variable_scope("pxtr_bias_gate_" + name, reuse=tf.AUTO_REUSE):
        # [-1,10, dim] -> [-1,10, func_num * queue_num]
        bias_gate_hidden = tf.layers.dense(weight_input, 256,
                                           activation=tf.nn.relu,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           bias_initializer=tf.truncated_normal_initializer(stddev=0.01))
        bias_gate = tf.layers.dense(bias_gate_hidden, func_num * queue_num ,
                                    activation=tf.nn.tanh,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())  # [bs, 10, func_num * queue_num] b*10*128
        queue_gate = tf.get_variable(name="pxtr_weight_" + name, shape=[10, func_num * queue_num], initializer=tf.contrib.layers.xavier_initializer()) #10*128
        gate = tf.reshape(bias_gate + queue_gate, [-1, 10, func_num, queue_num]) #b*10*8*16

    with tf.variable_scope("pxtr_weight_sum" + name, reuse=tf.AUTO_REUSE):
        queues = tf.reshape(queues, [-1, 10, func_num, queue_num])
        output_layer = tf.reduce_sum(queues * gate, axis=-2, name="queue_sum_" + name)  # [-1, 10, 16]
        logit = tf.layers.dense(output_layer, 1, activation=None, name="pxtr_logit_" + name)# [-1, 10, 1]
        print("cying_logit:", logit)
    return output_layer, logit

class RevisitModel:
    def __init__(self, user_param_dict, item_param_dict, extra_param_dict):
        self._user_param_dict = user_param_dict
        self._item_param_dict = item_param_dict
        self._extra_param_dict = extra_param_dict
        self.user_attr_names = [
            "device_id","user_id","user_gender", "user_age_segment"
        ]
        self.item_attr_names = [
            # "uTab",  # 只发现页
           "photo_id_list","author_id_list","hetu_cluster_id_list", "hetu_level_one_tag_list","hetu_level_two_tag_list","hetu_level_three_tag_list"
        ]
        self.pxtr_attr_names = [
            "pctr_list",
            "pltr_list",
            "pwtr_list",
            "plvtr_list",
            "pcmtr_list",
            "pcmef_list",
            "pptr_list",
            # "psvtr_list",

            "emp_ctr_list",
            "emp_ltr_list",
            "emp_wtr_list",
            "emp_lvtr_list"
        ]

        self.pxtr_index_attr_names = [
            "pctr_index_list",
            "pltr_index_list",
            "pwtr_index_list",
            "pvtr_index_list",
            "plvtr_index_list"
        ]

    def model(self):
        print_ops = []
        user_parameters_dict = self._user_param_dict
        item_parameters_dict = self._item_param_dict
        extra_param_dict = self._extra_param_dict
        user_feas = tf.concat([user_parameters_dict[attr] for attr in self.user_attr_names], axis=1) #b*72
        item_feas = tf.concat([item_parameters_dict[attr] for attr in self.item_attr_names], axis=1) #b*800
        dim = tf.shape(item_feas)[-1]//10
        item_feas = tf.reshape(item_feas, [-1, 40, 10])
        # print_ops.append(tf.print(f"[cying] item_fea ", item_feas, output_stream=sys.stdout))
        item_embs = tf.transpose(item_feas, perm=[0, 2, 1])
        # print_ops.append(tf.print(f"[cying] item_embs ", item_embs, output_stream=sys.stdout))
        user_embs = tf.tile(tf.expand_dims(user_feas, 1), [1, 10, 1])
        # remote_photo_emb = extra_param_dict["photo_id_emb"]
        # print_ops.append(tf.print(f"====> remote_photo_emb", remote_photo_emb, summarize=5, output_stream=sys.stdout))
        model_input = tf.concat([user_embs, item_embs], axis=-1) #b*10*152

        with tf.variable_scope("mlp", reuse=tf.AUTO_REUSE):
            h1 = tf.layers.dense(model_input, 128, activation=tf.nn.relu, name="h1", reuse=tf.AUTO_REUSE)
            h2 = tf.layers.dense(h1, 64, activation=tf.nn.relu, name="h2", reuse=tf.AUTO_REUSE)
            h3 = tf.layers.dense(h2, 32, activation=tf.nn.relu, name="h3", reuse=tf.AUTO_REUSE) #b*10*32
            # auto_ensemble
            xtr_index_dense = [tf.cast(tf.cast(extra_param_dict.get(attr_name), tf.float32)/420.0, tf.float32)
                               for attr_name in self.pxtr_index_attr_names] # 5 [b*10]
            xtr_dense = [extra_param_dict.get(attr_name) for attr_name in self.pxtr_attr_names] #11 [b*10]
            xtr_dense += xtr_index_dense #16 [b*10]

            # h3 = tf.Print(h3, [
            #     "model_input", model_input, "end",
            # ] + sum([[attr_name, extra_param_dict.get(attr_name), 'end'] for attr_name in self.pxtr_attr_names],[])
            # , message="hjz_model_print_s1:", summarize=-1)
        
            queues = transform_queues(xtr_dense, transform_functions) #b*80*16
            ae_out_layer, _ = auto_ensemble(queues, tf.stop_gradient(h3), len(xtr_dense), transform_functions, "auto_ens") # [b,10,16]
            print(f"====> model, ae_out_layer: {ae_out_layer}")

            h3_ext = tf.concat([h3, ae_out_layer], axis=-1) #[b, 10, 48]
            ltr = tf.layers.dense(h3_ext, 1, activation=tf.nn.sigmoid, name="ltr", reuse=tf.AUTO_REUSE)
            s_h3 = tf.reshape(tf.concat([h3_ext, ltr],axis=-1), [-1, 490])
            # tf.summary.histogram('s_h3', s_h3)
            ltr = tf.reshape(ltr, [-1, 10])
            pred_inner_time = tf.layers.dense(s_h3, 1, activation=tf.nn.sigmoid, name="pi", reuse=tf.AUTO_REUSE)
            # pred_out_time = tf.layers.dense(s_h3, 1, activation=tf.nn.sigmoid, name="po", reuse=tf.AUTO_REUSE)
            # pred_vv = tf.layers.dense(s_h3, 1, activation=tf.nn.sigmoid, name="pv", reuse=tf.AUTO_REUSE)
            pred_revisit = tf.layers.dense(s_h3, 1, activation=tf.nn.sigmoid, name="pr", reuse=tf.AUTO_REUSE)
        return ltr, pred_inner_time, pred_revisit, print_ops
