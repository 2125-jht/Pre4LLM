# -*- coding: UTF8 -*-
import kuiba_op_v2, kuiba_pybind, kuiba_utils
import tensorflow as tf
from enum import Enum, unique
import math
from tensorflow.python.framework.ops import Tensor

ctr_loss_name = "ctr_n"
beta_ctr_loss_name = "beta_n"
ltr_loss_name = "ltr_n"
lvtr_loss_name = "lvtr_n"
pvtr_loss_name = "pvtr_n"
cmtr_loss_name = "cmtr_n"
live_pvtr_loss_name = "live_pvtr_n"
ua_cross_loss_name = 'ua_cross_n'
ctr_time_loss_name = 'ctr_time_n'
live_bias_loss_name = "live_bias_n"
xtr_loss_name = "xtr_n"

user_click_list_loss_name = "user_click_list_n"
user_commentPhotoAuthor_list_loss_name = "user_commentPhotoAuthor_list_n"


class ModelBase(object):
    def __init__(self):
        self.loss_name_list = None
        self.merge_list = []

    def get_loss_name_list(self):
        assert self.loss_name_list, "must be set loss_name_list"
        return self.loss_name_list

    def merge_all(self):
        # tf.summary.merge_all https://www.jianshu.com/p/13841c62041f ,如果需要自己实现
        if len(self.merge_list) == 0:
            return None
        else:
            print("self.merge_all: merge_list_size={}".format(str(len(self.merge_list))))
            return tf.summary.merge(self.merge_list)

    def variable_summaries(self, name, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        if kuiba_utils.dryrun_to_get_variables:
            # 如果不做判断会报错，报错的问题主要是在dry_run模式的时候走summaries逻辑
            # 猜想未证实：怀疑可能是summary在tf里面是一个全局类在维护，走了两次之后存在两个相同summary,其中一个需要填充dry时候的占位符
            return
        with tf.variable_scope(name, reuse=kuiba_utils.reuse_variables()):
            print("self.variable_summaries: {}".format(name))
            mean = tf.reduce_mean(var)
            self.merge_list.append(tf.summary.scalar('mean', mean))
            self.merge_list.append(tf.summary.scalar('stddev', tf.sqrt(tf.reduce_mean(tf.square(var - mean)))))
            self.merge_list.append(tf.summary.scalar('max', tf.reduce_max(var)))
            self.merge_list.append(tf.summary.scalar('min', tf.reduce_min(var)))
            self.merge_list.append(tf.summary.histogram('histogram', var))

    def pre_layers(self, input, name):
        name = name + "_pre_layers"
        with tf.variable_scope(name, reuse=kuiba_utils.reuse_variables()):
            if kuiba_utils.train_mode():
                tf.summary.histogram(name + "_input", input)
            layers_size = [512, 256, 128]
            output = input
            for i in range(len(layers_size)):
                output = tf.layers.dense(output, layers_size[i], name=name + "_{}".format(i), activation=tf.nn.relu)
            return output

    def post_layers(self, input, name):
        with tf.variable_scope("{}_post_layers".format(name), reuse=kuiba_utils.reuse_variables()):
            output = tf.layers.dense(input, 128, activation=tf.nn.relu)
            output = tf.layers.dense(output, 128, activation=tf.nn.relu)
            output = tf.layers.dense(output, 1, activation=tf.nn.sigmoid)
            return output

    def attention_layer(self, querys, keys, is_nn=True):
        """
            queries:     [Batchsize, 1, embedding_size]
            keys:        [Batchsize, max_seq_len, embedding_size]  max_seq_len is the number of keys(e.g. number of creativeid for each sample)
            keys_id:     [Batchsize, max_seq_len]
        """
        keys_length = tf.shape(keys)[1]  # join_limit
        embedding_size = querys.get_shape().as_list()[-1]  # dim size
        # keys = tf.reshape(keys, shape=[-1, keys_length, embedding_size])

        outputs = None
        if not is_nn:
            querys = tf.tile(tf.reshape(querys, [-1, embedding_size]), [1, keys_length])
            querys = tf.reshape(querys, [-1, keys_length, embedding_size])
            all_logit_pre = tf.multiply(querys, keys,
                                        name="key_mul_query")  # [-1,keys_length,dim]
            all_logit = tf.reduce_sum(all_logit_pre, axis=2, name="all_logit_pre_reduce_sum")  # [-1,keys_length]
            all_logit = tf.divide(all_logit, math.sqrt(embedding_size))  # [-1,keys_length]
            all_logit = tf.reshape(all_logit, [-1, 1, keys_length])  # [-1,1,keys_length]
            outputs = all_logit  # [-1,1,keys_length]
        else:
            querys = tf.reshape(tf.tile(querys, [1, keys_length, 1]), shape=[-1, keys_length, embedding_size])
            net = tf.concat([keys, keys - querys, querys, keys * querys], axis=-1)
            for units in [32, 16]:
                net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
            att_wgt = tf.layers.dense(net, units=1, activation=tf.sigmoid)  # shape(batch_size, keys_length, 1)
            outputs = tf.reshape(att_wgt, shape=[-1, 1, keys_length],
                                 name="weight")  # shape(batch_size, 1, keys_length)

        scores = tf.nn.softmax(outputs, axis=-1)
        outputs = tf.matmul(scores, keys)  # (batch_size, 1, embedding_size)
        outputs = tf.reduce_sum(outputs, 1, name="attention_embedding")  # (batch_size, embedding_size)
        return outputs, scores

    def ensemble_exprt(self, gate, experts, name):
        # gate [-1,1,expert_num]
        # experts  [[-1,128]]
        name = "ensemble_{}".format(name)
        with tf.variable_scope(name, reuse=kuiba_utils.reuse_variables()):
            experts = tf.concat(experts, axis=1)
            experts = tf.reshape(experts, shape=[-1, len(self.loss_name_list), 128])  # [-1,expert_num,dim]
            outputs = tf.matmul(gate, experts)  # (batch_size, 1, embedding_size)
            outputs = tf.reduce_sum(outputs, 1)  # (batch_size, embedding_size)
            outputs = self.post_layers(outputs, name)
            return outputs

    def attention_handler(self, query, key_list, dim_size, gate_name):
        with tf.variable_scope("gate_{}".format(gate_name), reuse=kuiba_utils.reuse_variables()):
            keysize = len(key_list)
            keys_matrix_shape = [-1, keysize, dim_size]
            query_shape = [-1, 1, dim_size]
            old_query = query
            query = tf.reshape(query, shape=query_shape)
            print("keys:" + str(key_list))
            keys = tf.concat(key_list, axis=1)
            keys = tf.reshape(keys, shape=keys_matrix_shape)
            att, _ = self.attention_layer(query, keys)
            print("att : " + str(att))
            print("old_query:" + str(old_query))

            # return tf.concat([att, old_query], axis=1)
            return att


@unique
class ModelOutputType(Enum):
    # 模型的输出经过了sigmoid，对应tf.losses.log_loss
    predictions = 0
    # 模型的输出没有经过sigmoid，对应tf.losses.sigmoid_cross_entropy。二分类推荐用这种
    logits = 1
    # 模型的输出已经是计算好的loss
    loss = 2


class ModelOut:
    def __init__(self, predictions):
        # 默认使用 logloss(predictions) 方式
        self.predictions = predictions
        self.model_output_type = ModelOutputType.predictions
        self.loss_function = None
        self.logits = None
        self.push_metric_tensor = None
        self.push_tensor_op_list = None
        self.push_metric_list = []
        self.push_tensor_function_list = []
        self.group_name = None
        self.item_name = None

    def set_loss_function(self, loss_function, is_use_in_train):
        # 需要在model.py 里面 自定义loss function,设置使用自定义的flag
        # loss_function(logits, predictions, weights, labels_tensor)
        if is_use_in_train:
            self.model_output_type = ModelOutputType.loss
        self.loss_function = loss_function

    def get_loss(self, weights, labels_tensor):
        # train.py里面调用，传递 样本sample weight
        assert self.loss_function is not None, 'loss_function is None,need apply set_loss_function'
        assert self.logits is not None or self.predictions is not None, 'logits and predictions at least have one'
        return self.loss_function(self.logits, self.predictions, weights, labels_tensor)

    def set_logits(self, logits, is_use_in_train):
        # 二分类实验推荐直接使用logit
        if is_use_in_train:
            self.model_output_type = ModelOutputType.logits
        self.logits = logits

    @staticmethod
    def push_metric(tag, dim, group_key, item_key, pred, label, loss):
        assert kuiba_utils.train_mode()
        group_key = tf.reshape(group_key, [-1])
        group_key = tf.cast(group_key, tf.int64)
        item_key = tf.reshape(item_key, [-1])
        item_key = tf.cast(item_key, tf.int64)
        pred = tf.reshape(pred, [-1])
        label = tf.reshape(label, [-1])
        loss = tf.reshape(loss, [-1])
        print(
            "model_debug:push_metric:tag={},dim={},group_key={},item_key={},pred={},label={},loss={}".
                format(str(tag), str(dim), str(group_key), str(item_key), str(pred), str(label), str(loss)))
        return kuiba_op_v2.push_metric_op(
            tag=tag, dim=dim, group_key=group_key, item_key=item_key, pred=pred, label=label, loss=loss)

    def set_group_name(self, group_name):
        self.group_name = group_name

    def set_item_name(self, item_name):
        self.item_name = item_name

    def set_loss_name(self, loss_name):
        self.loss_name = loss_name

    def set_push_metric_tensor(self, loss_name, batch_id, label, loss):
        if len(self.push_metric_list) == 0:
            # 没有填充list
            return False
        node_list = [self.group_name, self.item_name]
        assert self.group_name
        assert self.item_name
        assert kuiba_utils.train_mode()
        print(
            "model_debug:set_push_metric_tensor:loss_function_name={},batch_id={},num_parameters={},parameter_names={}".
                format(str(loss_name), str(batch_id), str(len(node_list)), str(node_list)))
        sparse_key = kuiba_op_v2.pull_sparse_key_op(
            loss_function_name=loss_name, batch_id=batch_id, num_parameters=len(node_list),
            parameter_names=node_list)
        group_key = sparse_key[0]
        item_key = sparse_key[1]
        push_metric_list = []
        for fun in self.push_metric_list:
            ops = fun(loss_name, group_key, item_key, label, loss)
            print(
                "model_debug:set_push_tensor_fun={},isinstance(ops, list)={}".format(
                    str(ops), str(isinstance(ops, list))
                )
            )
            if isinstance(ops, list):
                push_metric_list.extend(ops)
            else:
                push_metric_list.append(ops)
        self.push_metric_tensor = tf.group(*push_metric_list)
        return True

    def _metric_element(self, loss_name, group_key, item_key, label, loss):
        ## demo
        act_xtr = self.predictions
        act_label = label
        act_loss = loss
        return ModelOut.push_metric(
            tag="%s-xtr" % loss_name, dim=1, item_key=item_key,
            group_key=group_key, pred=act_xtr, label=act_label, loss=act_loss)

    def set_metric_function_list(self, fun):
        # fun(loss_name, group_key, item_key, label, loss) demo function _metric_element
        #   input
        #   ouput kuiba_op_v2.push_metric_op
        self.push_metric_list.append(fun)

    def set_push_tensor_function_list(self, fun):
        # fun(loss_name, group_key, item_key, label, loss) demo function _metric_element
        #   input
        #   ouput kuiba_op_v2.push_metric_op
        self.push_tensor_function_list.append(fun)

    @staticmethod
    def push_to_btq(loss_name: str, node: str, embedding: Tensor, tensor_name: str, batch_id, qpath, qshard,
              qtype="ann_retr"):
        # qtype = "ann_retr"
        # qpath = ann的btq名字
        # qshard = ann shard 数量
        # tensor_name = ann 的 bucket name
        sparse_key = kuiba_op_v2.pull_sparse_key_op(
            loss_function_name=loss_name, batch_id=batch_id, num_parameters=1,
            parameter_names=[node])
        return kuiba_op_v2.dump_sparse_tensor_op(
            qtype=qtype,
            qpath=qpath,
            qshard=qshard,
            dim=embedding.shape[-1],
            tensor_name=tensor_name,
            keys=sparse_key[0],
            values=embedding)

    def set_push_tensor_op(self, loss_name, batch_id):
        if len(self.push_tensor_function_list) == 0:
            # 没有填充list
            return False
        assert kuiba_utils.train_mode()
        print(
            "model_debug:set_push_tensor_op:loss_function_name={},batch_id={}".
                format(str(loss_name), str(batch_id)))
        push_tensor_op_list = []
        for fun in self.push_tensor_function_list:
            ops = fun(loss_name, batch_id)
            print(
                "model_debug:set_push_tensor_op={},isinstance(ops, list)={}".format(
                    str(ops), str(isinstance(ops, list))
                )
            )
            if isinstance(ops, list):
                push_tensor_op_list.extend(ops)
            else:
                push_tensor_op_list.append(ops)
        self.push_tensor_op_list = tf.group(*push_tensor_op_list)
        return True

