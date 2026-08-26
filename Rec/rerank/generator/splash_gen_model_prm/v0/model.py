import tensorflow as tf
from feature_attr_extract import *
from modules_ import *

Tensor = tf.Tensor


@tf.custom_gradient
def swish(x):
    sigx = tf.nn.sigmoid(x)
    y = x * sigx

    def grad(dy):
        return dy * (y + (1. - y) * sigx)

    return y, grad

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

    def _scaled_dot_product_attention(self,
                                      Q: Tensor,
                                      K: Tensor,
                                      V: Tensor,
                                      scope="scaled_dot_product_attention") -> Tensor:
        #Q (B, dq, da)
        #K (B, dk, da)
        #V (B, dk, da)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            d_k = Q.get_shape().as_list()[-1]

            # dot product
            outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (B, dq, dk)

            # scale
            outputs /= d_k ** 0.5

            # softmax
            outputs = tf.nn.softmax(outputs)

            # weighted sum (context vectors)
            outputs = tf.matmul(outputs, V)  # (B, dq, da)

            return outputs

    def _single_head_atten(self,
                           queries: Tensor,
                           keys: Tensor,
                           values: Tensor,
                           atten_num: int,
                           scope="single_head_atten") -> Tensor:
        atten_dim = queries.get_shape().as_list()[-1]
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Linear projections
            Q = tf.reshape(tf.layers.dense(queries, atten_dim * atten_num, use_bias=True), [-1, atten_dim, atten_num])
            K = tf.reshape(tf.layers.dense(keys, atten_dim * atten_num, use_bias=True), [-1, atten_dim, atten_num])
            V = tf.reshape(tf.layers.dense(values, atten_dim * atten_num, use_bias=True), [-1, atten_dim, atten_num])

            # Attention
            outputs = self._scaled_dot_product_attention(Q, K, V) # (B, atten_dim, atten_num)
            outputs = tf.squeeze(tf.layers.dense(outputs, 1, use_bias=True), axis=[2]) # (B, atten_dim)

            # Residual
            outputs = outputs + queries

            return outputs

    # def get_loss_name(self) -> list:
    #     return self.loss_name_list
    def _mmoe_gate_unit(self,
                        inputs: Tensor,
                        expers: Tensor,
                        gate_id: int,
                        use_bias=True,
                        scope_prefix="mmoe_gate_") -> Tensor:
        # inputs (B, d_inputs)
        # expers (B, d_expert, num_expert)
        d_expert, num_expert = expers.get_shape().as_list()[-2:]
        with tf.variable_scope(scope_prefix + str(gate_id), reuse=tf.AUTO_REUSE):
            gate_weights = tf.layers.dense(inputs, num_expert, activation=None, use_bias=use_bias)
            gate_weights = tf.nn.softmax(gate_weights, axis=-1)
            gate_weights = tf.expand_dims(gate_weights, axis=1)  # (B, 1, num_expert)
            outputs = tf.reduce_sum(expers * tf.tile(gate_weights, [1, d_expert, 1]), axis=2, keepdims=False)

            return outputs  # (B, d_expert)

    def _mmoe_expert_unit(self,
                            inputs: Tensor,
                            num_expert: int,
                            dim_expert: int,
                            activation=tf.nn.relu,
                            use_bias=True,
                            scope="mmoe_expert_unit") -> Tensor:
        # 2-dim inputs (B, )
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            mmoe_experts = tf.layers.dense(inputs, dim_expert * num_expert, activation=None, use_bias=use_bias) # (?, cand_size, dim_expert * num_expert)
            mmoe_shape = [-1, self._cand_size, dim_expert, num_expert] if self._training else [-1, dim_expert, num_expert]
            mmoe_experts = tf.reshape(mmoe_experts, shape=mmoe_shape)

            return activation(mmoe_experts) # (B, dim_expert, num_expert)

    def _get_shared_features(self, input_dicts) -> tuple:
        with tf.variable_scope("share_bottom", reuse=tf.AUTO_REUSE):
            user_embs     = tf.concat([input_dicts[k] for k in input_dicts if k in user_fea_names], axis=-1)
            user_embs     = tf.tile(tf.expand_dims(user_embs, axis=1), [1,self._cand_size,1]) if self._training else user_embs
            photo_embs    = tf.concat([input_dicts[k] for k in input_dicts if k in photo_fea_names], axis=-1)
            source_embs   = tf.concat([input_dicts[k] for k in input_dicts if k in source_fea_names], axis=-1)
            source_embs   = tf.tile(tf.expand_dims(source_embs, axis=1), [1,self._cand_size,1]) if self._training else source_embs
            explore_embs  = tf.concat([input_dicts[k] for k in input_dicts if k in explore_profile_fea_names], axis=-1)
            explore_embs  = tf.reduce_mean(explore_embs, axis=1)
            explore_embs = tf.tile(tf.expand_dims(explore_embs, axis=1),[1,self._cand_size,1]) if self._training else explore_embs
            fountain_embs = tf.concat([input_dicts[k] for k in input_dicts if k in fountain_profile_fea_names], axis=-1)
            fountain_embs  = tf.reduce_mean(fountain_embs, axis=1)
            fountain_embs = tf.tile(tf.expand_dims(fountain_embs, axis=1), [1,self._cand_size,1]) if self._training else fountain_embs

            playtime_embs = tf.concat([input_dicts[k] for k in input_dicts if k in playtime_fea_names], axis=-1)
            playtime_embs  = tf.reduce_mean(playtime_embs, axis=1) # (?, dim)
            uid_emb = tf.tile(tf.expand_dims(input_dicts["user_id"], axis=1), [1,self._cand_size,1]) if self._training else input_dicts["user_id"]
            playtime_embs = tf.tile(tf.expand_dims(playtime_embs, axis=1), [1,self._cand_size,1]) if self._training else playtime_embs
            print("photo_id_v2", input_dicts["photo_id_v2"])
            playtime_embs = tf.concat([input_dicts["photo_id_v2"], uid_emb, playtime_embs], axis=-1) # train:(?, cand_size, dim),infer:(?, dim)
            common_embs   = tf.concat([user_embs, photo_embs, explore_embs, fountain_embs, playtime_embs, source_embs], axis=-1)
            # common_embs   = tf.concat([user_embs, photo_embs, fountain_embs, playtime_embs, source_embs], axis=-1)
            photo_weights = tf.layers.dense(tf.concat([photo_embs, user_embs], axis=-1), 256, activation=tf.nn.sigmoid)

            common_embs   = tf.layers.dense(common_embs, 512, activation=tf.nn.leaky_relu)
            common_embs   = tf.layers.dense(common_embs, 256, activation=tf.nn.leaky_relu)
            common_embs   = tf.layers.dense(common_embs, 256, activation=tf.nn.leaky_relu) * photo_weights # (?, cand_size, 256)
            if not self._training:
                '''
                    infer 时需要注意实际请求的 batch size = items 长度, 但计算图 batch 的 shape 由 uni_predict_fused 中 executor_batchsizes 决定,
                    如600。为了通过计算图编译, 需要 rashape (1, -1, dim), 实际 -1 为请求端发送 items 长度
                '''
                emb_dim = common_embs.shape[-1]
                common_embs = tf.reshape(common_embs, [1, -1, emb_dim])
            return common_embs

    def _l2r_head(self, com_embs, com_experts, input_dicts) -> Tensor:
        with tf.variable_scope("l2r_head", reuse=tf.AUTO_REUSE):
            local_experts = tf.layers.dense(com_embs, 128, activation=tf.nn.leaky_relu)
            local_experts = tf.expand_dims(local_experts, axis=2)
            com_experts = tf.concat([com_experts, local_experts], axis=2)

            gate_embs = tf.concat([input_dicts[k] for k in input_dicts if (k in fountain_profile_fea_names or k in explore_profile_fea_names)], axis=1)
            gate_embs = tf.concat([gate_embs, com_embs], axis=1)
            l2r_embs  = self._mmoe_gate_unit(gate_embs, com_experts, gate_id=0, use_bias=False)
            l2r_embs  = tf.layers.dense(l2r_embs, 128, activation=tf.nn.leaky_relu)
            l2r_embs  = tf.layers.dense(l2r_embs, 128, activation=tf.nn.leaky_relu)
            l2r_embs  = tf.layers.dense(l2r_embs, 64, activation=tf.nn.leaky_relu)
            l2r_preds = tf.layers.dense(l2r_embs, 1, activation=tf.nn.sigmoid)

            return l2r_preds

    def _vtr_head(self, com_embs, com_experts, input_dicts) -> Tensor:
        with tf.variable_scope("vtr_head", reuse=tf.AUTO_REUSE):
            local_experts = tf.layers.dense(com_embs, 128, activation=tf.nn.leaky_relu)
            local_experts = tf.expand_dims(local_experts, axis=2)
            com_experts = tf.concat([com_experts, local_experts], axis=2)

            gate_embs      = tf.concat([input_dicts[k] for k in input_dicts if k in playtime_fea_names], axis=1)
            gate_embs      = tf.concat([gate_embs, com_embs], axis=1)
            vtr_embs       = self._mmoe_gate_unit(gate_embs, com_experts, gate_id=1, use_bias=False)
            vtr_embs       = tf.layers.dense(vtr_embs, 128, activation=tf.nn.leaky_relu)
            vtr_embs       = tf.layers.dense(vtr_embs, 128, activation=tf.nn.leaky_relu)
            vtr_embs       = tf.layers.dense(vtr_embs, 64, activation=None)
           
            duration_embs  = input_dicts["photo_duration_s"]
            duration_embs  = tf.layers.dense(duration_embs, 128, activation=tf.nn.leaky_relu)
            duration_embs  = tf.layers.dense(duration_embs, 128, activation=tf.nn.leaky_relu)
            duration_atten = tf.layers.dense(duration_embs, 64, activation=tf.nn.leaky_relu)
            duration_embs  = tf.layers.dense(duration_embs, 64, activation=tf.nn.leaky_relu)
            duration_atten = self._single_head_atten(vtr_embs, vtr_embs, duration_atten, atten_num=8, scope="duration_atten_block")
            vtr_preds      = tf.reduce_sum(vtr_embs * duration_atten, axis=1, keepdims=True)
            return tf.nn.sigmoid(vtr_preds)

    def _ctr_head(self, com_embs, com_experts, input_dicts) -> Tensor:
        with tf.variable_scope("ctr_head", reuse=tf.AUTO_REUSE):
            local_experts = tf.layers.dense(com_embs, 128, activation=tf.nn.leaky_relu)
            local_experts = tf.expand_dims(local_experts, axis=2)
            com_experts = tf.concat([com_experts, local_experts], axis=2)

            gate_embs = tf.concat([input_dicts[k] for k in input_dicts if k in playtime_fea_names], axis=1)
            gate_embs = tf.concat([gate_embs, com_embs], axis=1)
            ctr_embs  = self._mmoe_gate_unit(gate_embs, com_experts, gate_id=2, use_bias=False)

            ctr_embs  = tf.layers.dense(ctr_embs, 128, activation=tf.nn.leaky_relu)
            ctr_embs  = tf.layers.dense(ctr_embs, 128, activation=tf.nn.leaky_relu)
            ctr_embs  = tf.layers.dense(ctr_embs, 64, activation=tf.nn.leaky_relu)
            ctr_preds = tf.layers.dense(ctr_embs, 1, activation=tf.nn.sigmoid)
            return ctr_preds

    def _single_ctr_head(self, com_embs, com_experts, input_dicts) -> Tensor:
        with tf.variable_scope("ctr_head", reuse=tf.AUTO_REUSE):
            ctr_embs  = tf.layers.dense(com_embs, 128, activation=tf.nn.leaky_relu)
            ctr_embs  = tf.layers.dense(ctr_embs, 128, activation=tf.nn.leaky_relu)
            ctr_embs  = tf.layers.dense(ctr_embs, 64, activation=tf.nn.leaky_relu)
            # ctr_preds = tf.layers.dense(ctr_embs, 1, activation=tf.nn.sigmoid)
            ctr_preds = tf.layers.dense(ctr_embs, 1)
            ctr_preds= tf.reshape(ctr_preds, [-1, self._cand_size])
            return ctr_preds

    def _next_head(self, com_embs, com_experts, input_dicts) -> Tensor:
        with tf.variable_scope("next_head", reuse=tf.AUTO_REUSE):
            local_experts = tf.layers.dense(com_embs, 128, activation=tf.nn.leaky_relu)
            local_experts = tf.expand_dims(local_experts, axis=2)
            com_experts = tf.concat([com_experts, local_experts], axis=2)

            next_embs = self._mmoe_gate_unit(com_embs, com_experts, gate_id=3, use_bias=False)
            next_embs = tf.layers.dense(next_embs, 128, activation=tf.nn.leaky_relu)
            next_embs  = tf.layers.dense(next_embs, 128, activation=tf.nn.leaky_relu)
            next_embs  = tf.layers.dense(next_embs, 64, activation=tf.nn.leaky_relu)
            next_preds = tf.layers.dense(next_embs, 1, activation=tf.nn.sigmoid)
            return next_preds

    def _finish_head(self, com_embs, com_experts, input_dicts) -> Tensor:
        with tf.variable_scope("finish_head", reuse=tf.AUTO_REUSE):
            local_experts = tf.layers.dense(com_embs, 128, activation=tf.nn.leaky_relu)
            local_experts = tf.expand_dims(local_experts, axis=2)
            com_experts = tf.concat([com_experts, local_experts], axis=2)

            finish_embs = self._mmoe_gate_unit(com_embs, com_experts, gate_id=4, use_bias=False)
            finish_embs = tf.layers.dense(finish_embs, 128, activation=tf.nn.leaky_relu)
            finish_embs  = tf.layers.dense(finish_embs, 128, activation=tf.nn.leaky_relu)
            finish_embs  = tf.layers.dense(finish_embs, 64, activation=tf.nn.leaky_relu)
            finish_embs = tf.layers.dense(finish_embs, 1, activation=None)
            return finish_embs

    def model(self, index_label=None) -> tuple:
        with tf.variable_scope("act_ltr_model", reuse=tf.AUTO_REUSE):
            input_dicts = self._parameters_dict
            com_embs = self._get_shared_features(input_dicts)
            print("com_embs:", com_embs)
            
            # if self._training:
            #     index_label = tf.cast(index_label, tf.int32) # 0: padding
            #     position_ids = index_label # (?, cand_size)
            # else:
            #     position_ids = tf.range(self._cand_size, dtype=tf.int32) + 1
            #     position_ids = tf.expand_dims(position_ids, 0)
            #     position_ids = tf.tile(position_ids, [tf.shape(com_embs)[0], 1]) # (?, cand_size)
            # position_embeddings = tf.nn.embedding_lookup(self.position_embeddings, position_ids) # (?, cand_size, dim)

            # hidden_states = tf.concat([com_embs, position_embeddings], axis=-1)
            hidden_states = com_embs
            hidden_states = tf.layers.dense(hidden_states, 256, activation=tf.nn.leaky_relu)
            print("hidden_states:", hidden_states)
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
                    output = tf.layers.dense(output, 1, activation=tf.nn.sigmoid)  # (?,candidates_size,1)
                    output = tf.reshape(output, [-1, self._cand_size]) # (?, cand_size)
                output_dict[loss_name] = output
            return output_dict

