import tensorflow as tf
from feature_attr_extract import *
from modules_ import *
import sys

user_fea_names = [
  "user_id",
  "user_device_id",
  "user_age_segment",
  "user_gender",
  "user_city_id",
  "user_client_id",
  "user_level",
  "user_active_days",
  "user_emp_ctr",
  "user_emp_ltr",
  "user_emp_wtr",
  "user_emp_lvtr",
  "user_request_province_id",
  "user_request_city_id",
  "user_request_poi_type",
  "user_region_type",
]

explore_profile_fea_names = [
  "user_realtime_click_list",
  "user_realtime_like_list",
  "user_realtime_follow_list",
  "user_realtime_forward_list",
]


photo_fea_names = [
  "photo_id",
  "photo_author_id",
  "photo_author_gender",
  "photo_author_age_segment",
  "photo_province_id",
  "photo_city_id",
  "photo_mod",
  "photo_upload_type",
  "photo_music",
  "photo_hetu_tag_level1_list",
  "photo_hetu_tag_level2_list",
  "photo_hetu_tag_level5_list",
  "photo_duration_ms",

  "photo_emp_ctr",
  "photo_emp_ltr",
  "photo_emp_wtr",
  "photo_emp_lvtr",
  "photo_emp_svtr",
  "context_pctr",
  "context_pltr",
  "context_pwtr",
  "context_pftr",
  "context_plvtr",
  "context_psvtr",
  "context_pvtr",
  "context_pptr",
  "context_pcmtr",
  "context_pwtd",
  "context_cascade_pctr",
  "context_cascade_pltr",
  "context_cascade_pwtr",
  "context_cascade_plvtr",
  "context_cascade_psvtr"
]

def layer_norm(x, epsilon=1e-6):
    with tf.variable_scope("layer_norm", reuse=tf.AUTO_REUSE):
        gamma = tf.get_variable("gamma", [x.get_shape()[-1]], initializer=tf.ones_initializer()) 
        beta = tf.get_variable("beta", [x.get_shape()[-1]], initializer=tf.zeros_initializer())

        mean, variance = tf.nn.moments(x, axes=[-1], keep_dims=True)
        normalized = (x - mean) / tf.sqrt(variance + epsilon) 
        output = gamma * normalized + beta 
    return output


def multi_head_attention(queries, keys, values, num_heads, dropout_rate, training=False):
        def split_heads(x, num_heads):
            batch_size = tf.shape(x)[0]
            depth = x.get_shape().as_list()[-1] // num_heads
            reshaped = tf.reshape(x, [batch_size, -1, num_heads, depth])
            return tf.transpose(reshaped, [0, 2, 1, 3])

        def scaled_dot_product_attention(Q, K, V):
            matmul_qk = tf.matmul(Q, K, transpose_b=True) #(?,4,?,?)
            dk = tf.cast(tf.shape(K)[-1], tf.float32) #last dimension of K:8
            scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
            output = tf.matmul(attention_weights, V)
            return output, attention_weights
        
        training = tf.constant(training, dtype=tf.bool)
        with tf.variable_scope("multi_head_attention", reuse=tf.AUTO_REUSE):
            depth = queries.get_shape().as_list()[-1] #last dimension of queries,32
            Q = tf.layers.dense(queries, depth, use_bias=False) #a linear layer to project the queries to the desired depth
            K = tf.layers.dense(keys, depth, use_bias=False) #(?,60,32)
            V = tf.layers.dense(values, depth, use_bias=False) #(?,60,32)

            Q = split_heads(Q, num_heads) #(?,4,?,8)
            K = split_heads(K, num_heads)
            V = split_heads(V, num_heads)

            scaled_attention, attention_weights = scaled_dot_product_attention(Q, K, V)
            scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3]) #(?,?,4,8)

            concat_attention = tf.reshape(scaled_attention, [tf.shape(queries)[0], -1, depth]) #(?,?,32)
            output = tf.layers.dense(concat_attention, depth) #(?,?,32)

            output = tf.cond(training, lambda: tf.nn.dropout(output, rate=dropout_rate), lambda: output) #(?,?,32)
        return output
    
def feed_forward_network(dim, hidden_dim, dropout_rate, training=False):
    def ffn(x, training=training):
        training = tf.constant(training, dtype=tf.bool)
        with tf.variable_scope(f"feed_forward_network", reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(x, hidden_dim, activation=tf.nn.relu)
            x = tf.layers.dense(x, dim)
            # x = tf.nn.dropout(x, rate=dropout_rate)
            x = tf.cond(training, lambda: tf.nn.dropout(x, rate=dropout_rate), lambda: x)
            return x
    return ffn

class TransformerLayer:
    def __init__(self, name, dim, num_heads, hidden_dim, dropout_rate, training=False):
        super(TransformerLayer, self).__init__()
        self.name = name
        self.dim = dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        self.mha = multi_head_attention
        self.ffn = feed_forward_network(dim, hidden_dim, dropout_rate)
        
    def forward(self, x, training):
        with tf.variable_scope(f"{self.name}", reuse=tf.AUTO_REUSE):
            attn_output = self.mha(x, x, x, self.num_heads, self.dropout_rate, training=training) #self-attention
            out1 = layer_norm(x + attn_output) #residual connection, layer normalization
            
            ffn_output = self.ffn(out1, training=training) #feed forward network
            out2 = layer_norm(out1 + ffn_output) #residual connection, layer normalization
        
        return out2

class PositionLayer:
    def __init__(self, name, dim, num_heads, hidden_dim, dropout_rate, training=False):
        super(PositionLayer, self).__init__()
        self.name = name
        self.dim = dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        self.self_attention = multi_head_attention
        self.cross_attention = multi_head_attention
        self.ffn = feed_forward_network(dim, hidden_dim, dropout_rate)
        
    def forward(self, x, enc_output, training=False):
        with tf.variable_scope(f"{self.name}", reuse=tf.AUTO_REUSE):
            cross_attn_output = self.cross_attention(x, enc_output, enc_output, self.num_heads, self.dropout_rate, training=training) #cross-attention,combines information from the encoder
            out1 = layer_norm(x + cross_attn_output) #residual connection, layer normalization
            attn_output = self.self_attention(out1, out1, out1, self.num_heads, self.dropout_rate, training=training) #self-attention
            out2 = layer_norm(x + attn_output) #residual connection, layer normalization
        
            ffn_output = self.ffn(out2, training=training)
            out3 = layer_norm(out2 + ffn_output)
        
        return out3
    
class StackedTransformerModel():
    def __init__(self, num_layers, dim, num_heads, hidden_dim, dropout_rate, k, training=False):
        super(StackedTransformerModel, self).__init__()
        self.num_layers = num_layers
        self.k = 6
        self.dim = dim
        self.position = tf.get_variable('s', shape=[self.k, self.dim], initializer=tf.random_normal_initializer())
        self.layers = [TransformerLayer(f"transformer_layer_{i}", dim, num_heads, hidden_dim, dropout_rate, training=training) for i in range(num_layers)]
        self.position_layers = [PositionLayer(f"position_layer_{i}", dim, num_heads, hidden_dim, dropout_rate, training=training) for i in range(num_layers)]
        
    def forward(self, hidden_states, training): #hidden_states is [batch_size,60,32]
        pos_embedding = tf.reshape(self.position, [1, self.k, self.dim]) #[1,k,dim] [1,6,32]
        pos_embedding = tf.tile(pos_embedding, [tf.shape(hidden_states)[0], 1, 1]) #[batch_size,k,dim] [batch_size,6,32]
        for i in range(self.num_layers): #3 layers
            hidden_states = self.layers[i].forward(hidden_states, training=training) #(?,60,32)
            pos_embedding = self.position_layers[i].forward(pos_embedding, hidden_states, training=training) #(?,6,32)
        return hidden_states, pos_embedding #hidden_states is [batch_size,k,dim], pos_embedding is [batch_size,k,dim]


class FountainDeepLtrMultiTaskModel:
    def __init__(self, parameters_dict, label_value_dict, extra_param_dict= None):
        self._parameters_dict = parameters_dict
        self._label_value_dict = label_value_dict #label, model内未使用
        self.position_embeddings = tf.get_variable(
            name='position_embeddings', 
            shape=[6, 32], 
            initializer=tf.random_normal_initializer()
        )
        # dat model vertex position embeddings
        self.num_vertex = 24
        self.vertex_position_embeddings = tf.get_variable(
            name='vertex_position_embeddings', 
            shape=[self.num_vertex, 32],  #24 vertex/positions
            initializer=tf.random_normal_initializer() 
        )
        # transition_matrix
        self.Wq = tf.get_variable(name='wq',shape=[32,32],initializer=tf.random_normal_initializer())
        self.Wk = tf.get_variable(name='wk',shape=[32,32],initializer=tf.random_normal_initializer())
       
    
    def _mlp_layer(self,
                  scope_name,
                  hidden_states: Tensor,
                  hidden_units: list,
                  activation=tf.nn.relu) -> Tensor:
        with tf.variable_scope(f"{scope_name}_mlp_layer", reuse=tf.AUTO_REUSE):
            for i, hidden_unit in enumerate(hidden_units):
                hidden_states = tf.layers.dense(hidden_states, hidden_unit, activation=activation, use_bias=True)
        return hidden_states
    
    def _get_shared_features(self, input_dicts) -> tuple:
        with tf.variable_scope("share_bottom", reuse=tf.AUTO_REUSE):
            list_dim = 60
            print("user_id embeding shape", input_dicts['user_id'].shape)
            user_embs = tf.concat([input_dicts[k] for k in input_dicts if k in user_fea_names], axis=-1)
            user_embs = tf.tile(tf.expand_dims(user_embs, axis=1),[1,list_dim,1])
            print("user_embs shape ", user_embs.shape)
            print("photo_id ",input_dicts['photo_id'].shape)
            photo_embs    = tf.concat([input_dicts[k] for k in input_dicts if k in photo_fea_names], axis=2)
            print("photo_embs shape ", photo_embs.shape)
            explore_embs  = tf.concat([input_dicts[k] for k in input_dicts if k in explore_profile_fea_names], axis=2)
            explore_embs  = tf.reduce_mean(explore_embs, axis=1)
            explore_embs = tf.tile(tf.expand_dims(explore_embs, axis=1),[1,list_dim,1])

            print("explore_embs shape ", photo_embs.shape)
            common_embs   = tf.concat([user_embs, photo_embs, explore_embs], axis=-1)
            print("common_embs shape", common_embs.shape)
            
            # yqy v2: shuffle item order within a batch
            # batch_size = tf.shape(common_embs)[0]
            # random_indices = tf.argsort(tf.random.uniform((batch_size, common_embs.shape[1])), axis=1)
            # shuffle_common_embs = tf.gather(common_embs, random_indices[:, :, tf.newaxis], axis=1, batch_dims=1)
            # shuffle_common_embs = tf.gather(common_embs, random_indices, batch_dims=1)

            # print("shuffle_common_embs shape",shuffle_common_embs.shape) #(?,?,532)
            # use shuffle common_embs or not
            # common_embs = shuffle_common_embs
            return common_embs
        
    def _contrastive_loss(self, score_matrix, margin=0.8, seqlen=6): #update margin
       gold_score = tf.linalg.diag_part(score_matrix) # 对角线上的元素
       gold_score = tf.expand_dims(gold_score, axis=2)  # 将对角线元素扩展为 (?,1,1) 的形状

       difference_matrix = gold_score - score_matrix # 计算矩阵中每个元素与对角线元素的差值
       loss_matrix = margin - difference_matrix # 计算损失矩阵
       loss_matrix = tf.nn.relu(loss_matrix) # 将所有负值置为0

       base_mask = tf.ones((seqlen, seqlen)) - tf.linalg.diag(tf.ones(seqlen)) # 对角线为0，其余为1
       base_mask = tf.expand_dims(base_mask, axis=0) # 将掩码矩阵扩展一维，使其形状变为 (1, seqlen, seqlen)
       base_mask = tf.tile(base_mask,[tf.shape(score_matrix)[0],1,1]) # 将掩码复制到与得分矩阵相同的batch大小

       cl_loss = tf.reduce_mean(loss_matrix*base_mask) # 计算平均损失
       
       return cl_loss
    
    def _similarity_matrix(self, score_matrix, margin=0.0, seqlen=6):
       base_mask = tf.ones((seqlen, seqlen)) - tf.linalg.diag(tf.ones(seqlen)) # 对角线为0，其余为1
       base_mask = tf.expand_dims(base_mask, axis=0) # 将掩码矩阵扩展一维，使其形状变为 (1, seqlen, seqlen)
       base_mask = tf.tile(base_mask,[tf.shape(score_matrix)[0],1,1]) # 将掩码复制到与得分矩阵相同的batch大小 (?,6,6)
       score_matrix = score_matrix*base_mask # remove diagonal elements(self score) 相似度矩阵中不再包含自相似度信息
       
       return score_matrix
    
    # extract links function 
    def transition_matrix(self, vertex_states, num_vertex):
        Q = tf.matmul(vertex_states,self.Wq) #(?,?,32)
        K = tf.matmul(vertex_states,self.Wk) #(?,?,32)
        # 2.matrix E = softmax(QK^T/sqrt(d))
        d = tf.cast(tf.shape(vertex_states)[-1],tf.float32) #hidden size:32 
        logits = tf.matmul(Q,K,transpose_b=True) / tf.sqrt(d) #(?,60,60)
        E = tf.nn.softmax(logits,axis=-1) #(?,60,60)
        # 3.to ensure there's no cycle in graph, apply lower triangular masking
        # lower_triangular_mask = tf.linalg.band_part(tf.ones([num_vertex, num_vertex]),0,-1) - tf.eye(num_vertex) #下三角和对角线为0
        # masked_E = E * lower_triangular_mask #(?,60,60)
        
        # # 在log空间计算softmax
        # log_E = tf.nn.log_softmax(logits, axis=-1)  # (batch_size, seq_len, seq_len)
        # # 4. 下三角mask (在log空间中,用-inf替代0)
        # lower_triangular_mask = tf.linalg.band_part(tf.ones([num_vertex, num_vertex]), 0, -1) - tf.eye(num_vertex)
        # # 扩展掩码矩阵到3维以匹配batch
        # lower_triangular_mask = tf.expand_dims(lower_triangular_mask, 0)  # [1, 24, 24]
        # lower_triangular_mask = tf.tile(lower_triangular_mask, [tf.shape(log_E)[0], 1, 1])  # [batch_size, 24, 24]
        # log_masked_E = tf.where(
        #     lower_triangular_mask > 0,
        #     log_E,
        #     tf.fill(tf.shape(log_E), float('-inf'))
        # )  # (batch_size, seq_len, seq_len)
        
        # log_softmax
        log_E = tf.nn.log_softmax(logits, axis=-1)
        return log_E

    def directed_acyclic_decoder(self,hidden_states):
        with tf.variable_scope("dat_decoder", reuse=tf.AUTO_REUSE):
            # vertex states
            vertex_position_embeddings = tf.tile(tf.expand_dims(self.vertex_position_embeddings, axis=0),[tf.shape(hidden_states)[0],1,1]) #(?,num_vertex,32)
            n_layers = 3
            num_vertex=self.num_vertex
            for i in range(n_layers):
                # self_attention
                pe_self_atten=multi_head_attention(vertex_position_embeddings,vertex_position_embeddings,vertex_position_embeddings,4,0.1,training=True) #(?,num_vertex,32)
                # cross_attention
                pe_cross_atten=multi_head_attention(pe_self_atten,hidden_states,hidden_states,4,0.1,training=True) #(?,num_vertex,32)
                # ffn
                pe_ffn=feed_forward_network(32,128,0.1,training=True)(pe_cross_atten,training=True) #(?,num_vertex,32)
                vertex_position_embeddings = pe_ffn
            vertex_states = vertex_position_embeddings #(?,num_vertex,32)
            
            # transition probability 
            transition_matrix = self.transition_matrix(vertex_states,num_vertex) #(?,num_vertex,num_vertex)
            
            # token distribution
            vertex_states = tf.transpose(vertex_states,perm=[0,2,1]) #(?,32,num_vertex)
            token_distribution = tf.matmul(hidden_states,vertex_states) #(?,60,32)*(?,32,num_vertex)=(?,60,num_vertex)
            token_distribution = tf.transpose(token_distribution,perm=[0,2,1]) #(?,num_vertex,60)
            return transition_matrix, token_distribution
        
    
    def model(self) -> tuple:
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            input_dicts = self._parameters_dict
            common_embs = self._get_shared_features(input_dicts)
            hidden_states = self._mlp_layer("mlp_layer_1", common_embs, [64,32])

            dim = 32
            num_heads = 4
            hidden_dim = 128
            dropout_rate = 0.1
            sequence_length = 100
            num_layers = 3
            k = 6
            model = StackedTransformerModel(num_layers, dim, num_heads, hidden_dim, dropout_rate, k)
            hidden_states, pos_embedding = model.forward(hidden_states, training=True) #
            print("hidden_states shape ", hidden_states.shape) #(?,60,32),item_embedding
            print("pos_embedding shape ", pos_embedding.shape) #(?,6,32)
            
            pos_embedding_trans = tf.transpose(pos_embedding,  perm=[0, 2, 1]) #(?,32,6)
            #matching score
            predict = tf.matmul(hidden_states, pos_embedding_trans) #hidden * pos (?,60,6)
            print("predict shape", predict.shape) #(?,60,6)

            predict = tf.nn.softmax(predict, axis=1) 

            norm_rep = pos_embedding / tf.norm(pos_embedding, axis=2, keepdims=True) # normalize pos_embedding
            print("norm_rep shape", norm_rep.shape)
            cosine_scores_rep = tf.matmul(norm_rep, tf.transpose(norm_rep, perm=[0, 2, 1])) #(?,6,6)
            print("cosine_scores shape", cosine_scores_rep.shape)
            sim_rep = self._similarity_matrix(cosine_scores_rep) #(?,6,6)

            cl_loss_pad = self._contrastive_loss(cosine_scores_rep) #position embeddings对比学习

            norm_outputs = hidden_states / tf.norm(hidden_states, axis=2, keepdims=True) # normalize hidden_states
            cosine_scores_outputs = tf.matmul(norm_outputs, tf.transpose(norm_outputs, perm=[0, 2, 1]))
            print("cosine_scores_outputs shape", cosine_scores_outputs.shape) #(?,60,60)
            cl_loss_outputs = self._contrastive_loss(cosine_scores_outputs, seqlen=60) #item_embedding对比学习
            sim_outputs = self._similarity_matrix(cosine_scores_outputs, seqlen=60)
            cl_loss = cl_loss_pad+cl_loss_outputs #item_embedding对比学习+position embeddings对比学习

            predict = tf.transpose(predict,  perm=[0, 2, 1]) #(?,6,60)

            #dat decoder
            transition_matrix, token_distribution = self.directed_acyclic_decoder(hidden_states)
            
            return predict, cl_loss, sim_rep, hidden_states, transition_matrix, token_distribution, cl_loss_outputs