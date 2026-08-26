from numpy import dtype
import tensorflow as tf
from feature_attr_extract import *
from modules_ import *


class FountainDeepLtrMultiTaskModel:
    def __init__(self, loss_names, parameters_dict, cand_size, training, print_ops, extra_param_dict= None):
        self._parameters_dict = parameters_dict
        self._cand_size = cand_size
        self._training = training
        self.print_ops = print_ops
        self.loss_names = loss_names
        self.position_embeddings = tf.get_variable(
            name='position_embeddings', 
            shape=[cand_size + 1, 64], 
            initializer=tf.random_normal_initializer()
        )

    def _get_shared_features(self, input_dicts) -> tuple:
        with tf.variable_scope("share_bottom", reuse=tf.AUTO_REUSE):
            user_embs     = tf.concat([input_dicts[k] for k in input_dicts if k in user_fea_names], axis=-1)
            user_embs     = tf.tile(tf.expand_dims(user_embs, axis=1), [1,self._cand_size,1]) if self._training else user_embs
            photo_embs    = tf.concat([input_dicts[k] for k in input_dicts if k in photo_fea_names], axis=-1)
            source_embs   = tf.concat([input_dicts[k] for k in input_dicts if k in source_fea_names], axis=-1)
            source_embs   = tf.tile(tf.expand_dims(source_embs, axis=1), [1,self._cand_size,1]) if self._training else source_embs
            explore_embs = tf.tile(tf.expand_dims(explore_embs, axis=1),[1,self._cand_size,1]) if self._training else explore_embs
            common_embs   = tf.concat([user_embs, photo_embs, source_embs], axis=-1)
            if not self._training:
                '''
                    infer 时需要注意实际请求的 batch size = items 长度, 但计算图 batch 的 shape 由 uni_predict_fused 中 executor_batchsizes 决定,
                    如600。为了通过计算图编译, 需要 rashape (1, -1, dim), 实际 -1 为请求端发送 items 长度
                '''
                emb_dim = common_embs.shape[-1]
                common_embs = tf.reshape(common_embs, [1, -1, emb_dim])
            return common_embs


    def model(self, index_label=None) -> tuple:
        with tf.variable_scope("act_ltr_model", reuse=tf.AUTO_REUSE):
            input_dicts = self._parameters_dict
            com_embs = self._get_shared_features(input_dicts) # (?, cand_size, 512)
            print("com_embs:", com_embs)
            
            if self._training:
                index_label = tf.cast(index_label, tf.int32) # 0: padding
                position_ids = index_label # (?, cand_size)
            else:
                batch_size = tf.shape(input_dicts["photo_id"])[0] # 特征实际输入 shape (cand_size, dim)
                # position_ids = tf.concat([tf.range(self._cand_size, dtype=tf.int32) + 1, tf.zeros([batch_size - self._cand_size,], dtype=tf.int32)], axis=-1) # (bs)
                position_ids = tf.cond(batch_size >= self._cand_size,
                                       lambda: tf.concat([tf.range(self._cand_size, dtype=tf.int32) + 1, tf.zeros([batch_size - self._cand_size,], dtype=tf.int32)], axis=-1),
                                       lambda: tf.range(batch_size, dtype=tf.int32) + 1)
                position_ids = tf.expand_dims(position_ids, 0) # (1, bs)
            position_embeddings = tf.nn.embedding_lookup(self.position_embeddings, position_ids) # (?, cand_size, dim)
            hidden_states = tf.concat([com_embs, position_embeddings], axis=-1)
            # hidden_states = com_embs
            hidden_states = tf.layers.dense(hidden_states, 256, activation=tf.nn.leaky_relu)
            encoder_layer = EncoderLayer("prm_enc_layer_0", 256, num_heads=8, dim_in=256, dropout_rate=0.1)
            hidden_states = encoder_layer.forward(hidden_states, training=self._training, causal_mask=False)
            encoder_layer = EncoderLayer("prm_enc_layer_1", 256, num_heads=8, dim_in=256, dropout_rate=0.1)
            hidden_states = encoder_layer.forward(hidden_states, training=self._training, causal_mask=False) # (?, cand_size, 256)
            com_embs = tf.concat([com_embs, hidden_states], axis=-1) # (?, cand_size, 512)
            print("com_embs:", com_embs)
            
            # 接一个 PLE 网络输出多目标
            ple_layer = PLE(self.loss_names, shared_key="vtr", cgc_layers = 1, task_expert_num=1, shared_expert_num=4,
                                expert_tower_dim = [256,128], gate_tower_dim = [256,128], print_ops = self.print_ops)
            input_feature_dict = {x: com_embs for x in self.loss_names}
            output_fea_dict = ple_layer(input_feature_dict)  # (?,candidates_size,64)
            print(f"ple_output vtr: {output_fea_dict['vtr']}")
            # 输出接 attention 平衡各个目标
            output_list = []
            key_output_list = []
            for key in output_fea_dict.keys():
                key_output_list.append(key)
                output_list.append(output_fea_dict[key])
            output_list = tf.stack(output_list, axis=2) # (?,candidates_size,n,64)
            output_list = output_attention("output_cross_attention", output_list, output_list, 64, values=output_list, need_initialize_values=False)
            output_list = tf.split(output_list, len(output_fea_dict), axis=2)
            for j in range(len(output_fea_dict)):
                output_fea_dict[key_output_list[j]] = tf.squeeze(output_list[j], axis=2)
            output_dict = {}
            for loss_name, output in output_fea_dict.items():
                with tf.variable_scope(f"output_mlp_{loss_name}", reuse=tf.AUTO_REUSE):
                    output  = tf.layers.dense(output, 128, activation=tf.nn.leaky_relu)
                    output  = tf.layers.dense(output, 64, activation=tf.nn.leaky_relu)
                    if loss_name == "wtd_level":
                        output = tf.layers.dense(output, 1, activation=None)  # (?,candidates_size,1)
                    else:
                        output = tf.layers.dense(output, 1, activation=tf.nn.sigmoid)  # (?,candidates_size,1)
                    output = tf.reshape(output, [-1, self._cand_size]) # (?, cand_size)
                    if self._training:
                        output = tf.reshape(output, [-1, self._cand_size]) # (?, cand_size)
                    else:
                        output = tf.reshape(output, [-1, 1]) # (cand_size, 1)
                output_dict[loss_name] = output
            return output_dict

