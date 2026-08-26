import os
import sys
import json
import logging
import argparse
import tensorflow as tf
from model_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict'], dest='mode', default='train')
parser.add_argument('--dryrun', dest='dryrun', const=True, default=False, nargs='?')
parser.add_argument('--with_kai', default=False)
# parser.add_argument('--with_kai', default=True)
parser.add_argument('--text', default=False)
parser.add_argument('--tower', choices=None, dest='tower', default='False')
parser.add_argument('--with_kai_v2', default=True) #False True 
# parser.add_argument('--with_kai_v2', default=False) #False True 
args = parser.parse_known_args()[0]
is_training = args.mode == "train"
LIST_SIZE = 2
print_ops = []
print("is training: ", is_training)
print("args.with_kai: ", args.with_kai)
print("args.with_kai_v2: ", args.with_kai_v2)

# 目前这段逻辑功能未知
if args.with_kai_v2:
    import kai.tensorflow as config
    import tensorflow.compat.v1 as tf
    default_param_attr = config.nn.ParamAttr(initializer=config.nn.UniformInitializer(0.0001),
                                             access_method=config.nn.ProbabilityAccess(100.0),
                                             recycle_method=config.nn.UnseendaysRecycle(delete_after_unseen_days=30, delete_threshold=0.1, allow_dynamic_delete=True))
    config.nn.set_default_param_attr(default_param_attr)
else:
    import tensorflow as tf
    from mio_tensorflow.config import MioConfig
    if not args.dryrun and not args.with_kai:
        import mio_tensorflow.patch as mio_tensorflow_patch
        mio_tensorflow_patch.apply()
    
    logging.basicConfig()
    base_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), './base.yaml')
    config = MioConfig.from_base_yaml(base_config,
                                      clear_embeddings=True,
                                      clear_params=True, #False,
                                      dryrun=args.dryrun,
                                      label_with_kv=True,
                                      grad_no_scale=False,
                                      with_kai=args.with_kai,
                                  predict=(args.mode != "train"))


config_from_kuiba = json.load(open("./kai_kuiba_config.json"))
all_dense_input_dict = {}
use_dragonfly_io = True
view_list_dim = [8,8]
ltr_view_sg_list = [False, False, False, False, False, False, False]
view_list_length = 50

def new_sized_embedding(name, dim, expand, slots, common):
  if args.mode == 'predict' or args.with_kai:
    return config.new_embedding(name, dim=dim, expand=expand, slots=slots, common=common)
  # kai2 单个值 expand=None
  elif expand == 1:
    expand = None
  x = config.new_embedding(name, dim=dim, expand=expand, common=common, slots=slots)
  return x

def get_sparse_input(feature_name, dim, slot_id, expand, common):
  feature_name = "KAI_" + feature_name
  if args.mode == 'predict':
    pass
  else:
    common = False

  sparse_input = new_sized_embedding(feature_name, dim, expand, slot_id, common)
  print ("new_embedding " + feature_name + ", dim:" + str(dim) + ", slot:" + str(slot_id))
  return sparse_input

def get_dense_input(name, dim=1, default_value=0.0):
  if name in all_dense_input_dict.keys():
    return all_dense_input_dict[name]
  print ("get_label:" + name, dim)

  sign_feature_dim = config_from_kuiba["sign_feature_dim"]
  if name in sign_feature_dim.keys():
    dim = sign_feature_dim[name]
  if use_dragonfly_io and name.startswith("KAI_"):
    if name[4:] in sign_feature_dim.keys():
      dim = sign_feature_dim[name[4:]]

  dense_input = config.get_extra_param(name, size=dim, default_value=default_value)
  all_dense_input_dict[name] = dense_input
  return dense_input

def get_kuiba_parameter_dim(name):
  sign_feature_dim = config_from_kuiba["sign_feature_dim"]
  return sign_feature_dim[name]

def get_kuiba_loss_relative(loss, parameters_dict: dict, with_label_dict, with_label_value):
  sign_feature_slot = config_from_kuiba["sign_feature_slot"]
  sign_feature_dim = config_from_kuiba["sign_feature_dim"]
  sign_feature_expand = config_from_kuiba["sign_feature_expand"]
  sign_feature_is_common = config_from_kuiba["sign_feature_is_common"]
  loss_config = config_from_kuiba["loss_functions"][loss]

  for input_name in loss_config["all_inputs"]:
    if input_name in parameters_dict: # 已存在
      continue
    if input_name in loss_config["sparse_inputs"]:
      sparse_input_name = input_name
      dim = sign_feature_dim[sparse_input_name]
      slot_id = sign_feature_slot[sparse_input_name]
      expand = sign_feature_expand[sparse_input_name]
      common = sign_feature_is_common[sparse_input_name]
      sparse_input = get_sparse_input(sparse_input_name, dim, slot_id, expand, common)
      parameters_dict[sparse_input_name] = sparse_input
    else:
      assert(input_name in loss_config["dense_inputs"])
      dense_input_name = input_name
      if use_dragonfly_io:
        parameters_dict[dense_input_name] = get_dense_input("KAI_" + dense_input_name)
      else:
        parameters_dict[dense_input_name] = get_dense_input(dense_input_name)

  label_dict = {}
  label_value_dict = {}
  for loss_name,config in config_from_kuiba["loss_functions"].items():
    output_dim = 1
    if "output_dim" in config.keys():
      output_dim = config["output_dim"]
    if (with_label_dict):
      label_dict[loss_name] = get_dense_input(loss_name + "_label", dim=output_dim)
    if (with_label_value):
      label_value_dict[loss_name] = get_dense_input(loss_name + "_label_value")

  return label_dict, label_value_dict

def sum_loss_tensor_dict(loss_dict):
  sum_loss = None
  for key,loss in loss_dict.items():
    if (sum_loss == None):
      sum_loss = loss
    else:
      sum_loss += loss
  return sum_loss

def get_parameter_names_by_loss_name(loss_name):
  loss_config = config_from_kuiba["loss_functions"][loss_name];
  input_set = {}
  all_inputs = []
  for input_name in loss_config["all_inputs"]:
    if input_name in input_set.keys():
      continue
    input_set[input_name] = 1
    all_inputs.append(input_name)
  return all_inputs

def transformer(loss, query, data, n, nh = 1, dim = 256, mask = None):
    scope = loss + '_transformer'
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        batch_size = tf.shape(data)[0]
        a = self_attention(scope, query, data, n, nh, dim, mask = mask)
        a = dense_layer(a, dim, scope + "_proj")
        output = norm(data + a, scope + '_ln_1')
        m = mlp(scope + "_mlp", output, dim * 4)
        output = norm(output + m, scope + '_ln_2')
    return output

def get_pos_embedding(name, pos_index, batch_size, dim=64):
  with tf.variable_scope(name+"pos_embedding", reuse=tf.AUTO_REUSE):
    embedding_param = tf.get_variable(name="pos_embedding", shape=[6, dim])
    ids = tf.ones([batch_size, 1], name="id_{}".format(pos_index), dtype=tf.int32) * pos_index
    embedding = tf.nn.embedding_lookup(ids=ids, params=embedding_param)
    embedding = tf.reshape(embedding, shape=[-1, dim])
    return embedding

user_base_feature_names = ["uId", "dId", "uBasicAge", "uGender","uCityId",]
user_view_list_feature_names = ["uRealHatePids", "uRealClickPids", "uRealLikePids", "uRealFollowPids", "uRealForwardPids",
        "uMidClickPids", "uMidLikePids", "uMidFollowPids", "uMidCommentPids", "uMidPlayEffectivePids", "uMidPlayLongPids",
        "uMidPlayShortPids", "uMidPlayViewHetu1", "uMidPlayViewHetu2",  "fountainClickPids", "fountainLikePids", "fountainFollowPids",
        "fountainLongviewPids", "fountainEffviewPids",]

def poslabel_model(parameters_dict, targets, loss_name):
  true_size = int(loss_name[-1])
  print("true_size:", true_size)

  parameters_name = get_parameter_names_by_loss_name(loss_name)
  parameters = [parameters_dict[name] for name in parameters_name]

  user_fea_num = 5
  view_list_num = 19

  with tf.variable_scope("splash_evaluator", reuse=tf.AUTO_REUSE):
    # 用户基础特征
    user = tf.concat(parameters[:user_fea_num], axis = 1) # (?, num * dim)
    # 用户行为特征
    real_neg_pids = parameters_dict["uRealHatePids"]
    real_act_pids = parameters[user_fea_num+1:user_fea_num+5]
    mid_act_pids = parameters[user_fea_num+5:user_fea_num+9]

    rown = tf.shape(user)[0]
    mid_play_ev_pids = tf.reshape(parameters_dict["uMidPlayEffectivePids"], (rown, 1, 32))
    mid_play_lv_pids = tf.reshape(parameters_dict["uMidPlayLongPids"], (rown, 1, 32))
    mid_play_sv_pids = tf.reshape(parameters_dict["uMidPlayShortPids"], (rown, 1, 32))

    cl_net_hetu1 =  tf.reshape(parameters_dict["uMidPlayViewHetu1"], (rown, 50, 8))
    cl_net_hetu2 = tf.reshape(parameters_dict["uMidPlayViewHetu2"], (rown, 50, 8))
    cl_input = [cl_net_hetu1, cl_net_hetu2]

    fountain_act_pids = parameters[user_fea_num + view_list_num-5:user_fea_num + view_list_num-2]
    fountain_play_pids = parameters[user_fea_num + view_list_num-2:user_fea_num + view_list_num]

    # 上下文
    # page = parameters_dict["page"]
    # real_size = parameters_dict["real_size"]
    # print_ops.append(tf.print(f"xxx page ", page, summarize = 8, output_stream=sys.stdout))
    # print_ops.append(tf.print(f"xxx real_size ", real_size, summarize = 8, output_stream=sys.stdout))

    # parameters = parameters[user_fea_num + view_list_num : ]
    # user = tf.stop_gradient(user)
    data0 = []

    data = []
    list_mask = []
    l2r_labels = []
    l2r_weights = []
    next_labels = []
    next_weights = []
    play_weights = []
    ltr_pids = []
    pos_embeddings = []
    
    interact_pxtrs = []
    watchtime_pxtrs = []
    play_rate = []
    hetus = []
    pvtrs = []
    durations = []
    for i in range(LIST_SIZE):
      photo_interact_pxtr_names = [f"pPltr_idx{i}", f"pPwtr_idx{i}", f"pPftr_idx{i}", f"pPhtr_idx{i}", f"pPptr_idx{i}", f"pPcmtr_idx{i}",
        f"pPctr_idx{i}", f"pMcPctr_idx{i}", f"pMcPwtr_idx{i}", f"pMcPltr_idx{i}"]
      photo_watchtime_pxtr_names = [f"pPlvtr_idx{i}", f"pPsvtr_idx{i}", f"pPvtr_idx{i}",
        f"pPwtd_idx{i}", f"pMcPlvtr_idx{i}", f"pMcPsvtr_idx{i}", f"pPcmef_idx{i}", f"pPepstr_idx{i}"]
      photo_emp_xtr_names = [f"pEmpCtr_idx{i}", f"pEmpLtr_idx{i}", f"pEmpWtr_idx{i}", f"pEmpFtr_idx{i}", f"pEmpCmtr_idx{i}", f"pEmpHtr_idx{i}", f"pEmpPtr_idx{i}",
        f"avg_watchtime_idx{i}", f"pContentLevel_idx{i}",]
      photo_base_feature_names = [f"pId_idx{i}", f"aId_idx{i}",
        f"pDurationMs_idx{i}", f"pUploadType_idx{i}", f"pCityId_idx{i}", f"pProvinceId_idx{i}", f"pTag_idx{i}", f"pMusic_idx{i}", f"pAuthorGender_idx{i}",
        f"pAgeHour_idx{i}", f"pMmuImgClusterV1_idx{i}", f"pMmuImgClusterV3_idx{i}", f"pMmuContentId_idx{i}", f"pOcrCoverTextWordCount_idx{i}", f"pMusicComboId_idx{i}",
        f"pHetuTagLevel1Id_idx{i}", f"pHetuTagLevel2Id_idx{i}"]
      ltr_pid = tf.concat([parameters_dict[f"pId_idx{i}"] if i < true_size else tf.zeros_like(parameters_dict[f"pId_idx0"]),
                           parameters_dict[f"aId_idx{i}"] if i < true_size else tf.zeros_like(parameters_dict[f"aId_idx0"])], -1) # 
      ltr_pids.append(ltr_pid)
      durations.append(parameters_dict[f"pDurationMs_idx{i}"]) if i < true_size else tf.zeros_like(parameters_dict[f"pDurationMs_idx{i}"])

      l2r_weights.append(parameters_dict[f"slide_neg_weight_idx{i}"] if i < true_size else tf.zeros_like(parameters_dict[f"slide_neg_weight_idx0"]))
      l2r_labels.append(parameters_dict[f"slide_wtd_label_idx{i}"] if i < true_size else tf.zeros_like(parameters_dict[f"slide_wtd_label_idx0"]))
      play_weights.append(parameters_dict[f"slide_play_weight_idx{i}"] if i < true_size else tf.zeros_like(parameters_dict[f"slide_play_weight_idx0"]))
      play_rate.append(parameters_dict[f"slide_play_rate_idx{i}"] if i < true_size else tf.zeros_like(parameters_dict[f"slide_play_rate_idx0"]))
      next_weights.append(parameters_dict[f"slide_next_weight_idx{i}"] if i < true_size else tf.zeros_like(parameters_dict[f"slide_next_weight_idx0"]))
      next_labels.append(parameters_dict[f"slide_next_label_idx{i}"] if i < true_size else tf.zeros_like(parameters_dict[f"slide_next_label_idx0"]))
      mask = tf.ones_like(parameters_dict[f"slide_next_label_idx0"]) if i < true_size else tf.zeros_like(parameters_dict[f"slide_next_label_idx0"])
      list_mask.append(mask)

      pvtrs.append(parameters_dict[f"pvtr_idx{i}"]) if i < true_size else tf.zeros_like(parameters_dict[f"pvtr_idx0"])
      interact_pxtr = tf.concat([parameters_dict[x] if i < true_size else tf.zeros_like(parameters_dict[x]) for x in photo_interact_pxtr_names], 1)
      watchtime_pxtr = tf.concat([parameters_dict[x] if i < true_size else tf.zeros_like(parameters_dict[x]) for x in photo_watchtime_pxtr_names], 1)
      interact_pxtrs.append(interact_pxtr)
      watchtime_pxtrs.append(watchtime_pxtr)
      photo_base_features = tf.concat([parameters_dict[x] if i < true_size else tf.zeros_like(parameters_dict[x]) for x in photo_base_feature_names], 1)
      photo_emp_features = tf.concat([parameters_dict[x] if i < true_size else tf.zeros_like(parameters_dict[x]) for x in photo_emp_xtr_names], 1)
      data0.append(tf.concat([photo_base_features, photo_emp_features], 1))
      hetus.append(tf.concat([parameters_dict[f"pHetuTagLevel1Id_idx{i}"] if i < true_size else tf.zeros_like(parameters_dict[f"pHetuTagLevel1Id_idx{i}"]),
                              parameters_dict[f"pHetuTagLevel2Id_idx{i}"] if i < true_size else tf.zeros_like(parameters_dict[f"pHetuTagLevel2Id_idx{i}"])], 1))
      if (args.mode == 'train'):
          pos_emb = get_pos_embedding('pos', i, rown, dim=64)
      elif (args.mode == 'predict'):
          pos_emb = get_pos_embedding('pos', 0, rown, dim=64)
      pos_embeddings.append(pos_emb)
    # 升维操作
    ltr_pid_emb = tf.stack(ltr_pids, axis = 1) # (?, list_size, dim)
    datas = tf.stack(data0, axis = 1)
    interact_pxtrs = tf.stack(interact_pxtrs, axis = 1)
    watchtime_pxtrs = tf.stack(watchtime_pxtrs, axis = 1)
    hetu_emb = tf.stack(hetus, axis = 1)
    pos_embeddings = tf.stack(pos_embeddings, axis = 1)
    # print_ops.append(tf.print(f"ltr_pid_emb ", ltr_pid_emb, summarize = 8, output_stream=sys.stdout))
    # print_ops.append(tf.print(f"datas ", datas, summarize = 8, output_stream=sys.stdout))

    #用户list处理
    real_act_slot, mid_act_slot, foutain_act_slot  = tf.split(fc_layer('user_act_slot_dnn', user, [32, 3]),[1,1,1], 1)
    # aa = tf.multiply(tf.stack(real_act_pids, axis = 1), real_act_slot)
    # bb = real_act_slot * tf.stack(real_act_pids, axis = 1)
    
    user_act_embs = tf.concat([tf.stack(real_act_pids, axis = 1) * tf.reshape(real_act_slot, [-1,1,1]),
                              tf.stack(mid_act_pids, axis = 1) * tf.reshape(mid_act_slot,[-1,1,1]),
                              tf.stack(fountain_act_pids, axis = 1) * tf.reshape(foutain_act_slot, [-1,1,1])], axis = 1)
    user_ev_slot, user_lv_slot, user_sv_slot, fountain_lv_slot,  fountain_ev_slot= tf.split(
      fc_layer('user_play_slot_dnn', user, [32, 5]),[1,1,1,1,1], 1)
    mid_play_emb = tf.concat([mid_play_ev_pids * tf.reshape(user_ev_slot, [-1,1,1]) , 
                              mid_play_lv_pids * tf.reshape(user_lv_slot, [-1,1,1]),
                              mid_play_sv_pids * tf.reshape(user_sv_slot, [-1,1,1]),
                              tf.reshape(fountain_play_pids[0], [-1, 1, 32]) * tf.reshape(fountain_lv_slot, [-1,1,1]),
                              tf.reshape(fountain_play_pids[1], [-1, 1, 32]) * tf.reshape(fountain_ev_slot, [-1,1,1])], axis =1)
    user_ems = tf.tile(tf.expand_dims(user, 1), [1, LIST_SIZE, 1])
    pid_att = self_attention("pid_attention", ltr_pid_emb, ltr_pid_emb, LIST_SIZE, 1, 64, mask = None) # (?, list_size, 64)
    short_net_query = tf.concat([user_ems, ltr_pid_emb, hetu_emb, pid_att], axis=2) # (?, list_size, 284)
    print("short_net_query ", short_net_query)
    # print_ops.append(tf.print(f"short_net_query ", short_net_query, summarize = 8, output_stream=sys.stdout))
    query_dim = 64
    short_net_query = fc_layer('ltr_user_short', short_net_query, [query_dim]) # (?, list_size, 64)
    short_net_query2 = fc_layer('ltr_user_short2', short_net_query, [32]) # (?, list_size, 32)
    # user + 全局 item 信息 对河图做 attention
    view_net_output = short_net("ltr_view_list", short_net_query, cl_input, view_list_length, view_list_dim,
                                query_dim = query_dim, nh = 1, dim = 64, stop_gradient_list = ltr_view_sg_list,
                                action_item_size = 64, att_emb_size = 64) # (?, list_size, 64)
    print("view_net_output ", view_net_output)
    # user + 全局 item 信息 对用户 act 行为序列做 attention
    pos_output = short_net("pos_list", short_net_query2, [user_act_embs], 11, [32],
                          query_dim = 32, nh = 1, dim = 32, stop_gradient_list = ltr_view_sg_list,
                          action_item_size = 32, att_emb_size = 32) # (?, list_size, 32)
    # user + 全局 item 信息 对用户 play 行为序列做 attention
    play_output = short_net("play_list", short_net_query2, [mid_play_emb], 5, [32],
                            query_dim = 32, nh = 1, dim = 32, stop_gradient_list = ltr_view_sg_list,
                            action_item_size = 32, att_emb_size = 32) # (?, list_size, 32)
    # user + 全局 item 信息 对用户 hate 行为序列做 attention
    neg_output = short_net("neg_list", short_net_query2, [real_neg_pids], 1, [32],
                          query_dim = 32, nh = 1, dim = 32, stop_gradient_list = ltr_view_sg_list,
                          action_item_size = 32, att_emb_size = 32) # (?, list_size, 32)
    gate_fea =  tf.concat([user_ems, view_net_output, pos_output, neg_output, play_output, datas, interact_pxtrs, watchtime_pxtrs], axis=-1) # (?, list_size, dim)
    # print_ops.append(tf.print(f"gate_fea ", gate_fea, summarize = 8, output_stream=sys.stdout))
    data = fc_layer('l2r_dnn', gate_fea, [256, 128, 64]) # (?, list_size, 64)
    data1 = gate_tower('vtr_dnn1', interact_pxtrs, [64], short_net_query) # ppnet
    data2 = gate_tower('vtr_dnn2', watchtime_pxtrs, [64], short_net_query)
    gate = gate_layer2('vtr_gate', short_net_query, 64)
    data_pxtr = data1 * gate + data2 * (1 - gate) # 互动和时长信息融合
    pxtr_wt = fc_layer('pxtr_wt', tf.reshape(data_pxtr, [-1, LIST_SIZE * 64]), [256, 128, 64, LIST_SIZE])
    avg_data_pxtr = tf.reshape(tf.reduce_mean(tf.reshape(pxtr_wt, [-1, LIST_SIZE, 1]) * data_pxtr, axis=1), [-1, 1, 64])
    # 融合上下文均值信息
    data_pxtr = tf.concat([data_pxtr,avg_data_pxtr], axis=1) # (?, list_size + 1, 64)
    print("data_pxtr ", data_pxtr)

    data = tf.add(data, pos_embeddings) # (?, list_size, 64)
    print("data ", data)
    data_wt = fc_layer('data_wt', tf.reshape(data, [-1, LIST_SIZE * 64]), [256, 128, 64, LIST_SIZE]) # (?, list_size)
    print("data_wt ", data_wt)
    list_emb = tf.reshape(tf.reduce_mean(tf.reshape(data_wt, [-1, LIST_SIZE, 1]) * gate_tower('list_emb', data, [64], user_ems),axis=1),[-1,1,64]) # (?, 1, 64)
    data = tf.concat([data, list_emb],axis=1) # (?, list_size + 1, 64)
    global_output = transformer("ltr_trf_1", data, data, LIST_SIZE + 1, nh=1, dim=64, mask=None) # (?, list_size + 1, 64)
    print("global_output ", global_output)
    atten_output = tf.reshape(tf.concat([global_output, data_pxtr],axis=-1), (rown, (LIST_SIZE + 1) * 128)) # (?, dim)
    print("atten_output ", atten_output)
    # print_ops.append(tf.print(f"atten_output ", atten_output, summarize = 8, output_stream=sys.stdout))
    atten_outputs = tf.split(atten_output, [128] * (LIST_SIZE + 1), 1) # [(?, 128)] * (list_size + 1)
    
    # 接一个 PLE 网络输出 ctr 和 l2r 目标
    photo_ple_layer = PLE(["next", "wtd", "play"], shared_key="wtd", cgc_layers = 1, task_expert_num=1, shared_expert_num=4,
                      expert_tower_dim = [128,128], gate_tower_dim = [64,32])
    output_next_scores = []
    output_l2r_scores = []
    output_play_scores = []
    output_list_play_scores = 0.0
    for i in range(LIST_SIZE) : 
      input_feature_dict = {"next": atten_outputs[i], "wtd": atten_outputs[i], "play": atten_outputs[i]}
      output_fea_dict = photo_ple_layer(input_feature_dict)
      output_list = []
      key_output_list = []
      for key in output_fea_dict.keys():
        key_output_list.append(key)
        output_list.append(output_fea_dict[key])
        # tf.summary.histogram(key, output_fea_dict[key], family="loss_begin")
      output_list = tf.stack(output_list, axis=1) # (?, 3, 128)
      # print_ops.append(tf.print(f"output_list_stack_{i} ", output_list, summarize = 8, output_stream=sys.stdout))
      output_list = self_attention("output_cross_attention", output_list, output_list, len(output_fea_dict), 1, 128, mask = None)
      # print_ops.append(tf.print(f"atten_output_{i} ", output_list, summarize = 8, output_stream=sys.stdout))
      output_list = tf.split(output_list, len(output_fea_dict), 1)
      for j in range(len(output_fea_dict)):
        key = key_output_list[j]
        values = tf.reshape(output_list[j],[-1, 128])
        # tf.summary.histogram(key, values, family="loss_after")
        output_hidden = fc_layer(key + 'ltr_proj', values, [64, 32])
        output_hidden = fc_layer(key + 'ltr_out_none', output_hidden, [32], activation = None)
        if key == "wtd":
          output_pos = fc_layer(key + 'ltr_out_pos', output_hidden, [1], activation = tf.nn.sigmoid)
          output_l2r_scores.append(output_pos)
        elif key == "next":
          output_pos = fc_layer(key + 'ltr_out_pos', output_hidden, [1], activation = tf.nn.sigmoid)
          output_next_scores.append(output_pos)
        elif key == "play":
          output_pos = fc_layer(key + 'ltr_out_pos', output_hidden, [1], activation = tf.nn.sigmoid)
          output_play_scores.append(output_pos)
      
  # list model
    output_list_play_score = fc_layer('list_play', atten_outputs[LIST_SIZE], [64, 32])
    output_list_play_score = fc_layer('list_play_out', output_list_play_score, [1], activation = None)
    
    losses = []
    output_dict = {}
    if args.mode == 'train':
      # if true_size == 4:
      #   print_ops.append(tf.print(f"list_mask", list_mask, summarize = 8, output_stream=sys.stdout))
      # next 单点模型
      for i in range(LIST_SIZE):
        next_labels[i]
        output_next_scores[i]
        next_weights[i]
        next_loss = tf.losses.log_loss(labels=next_labels[i], predictions=output_next_scores[i],
                                      weights = next_weights[i] * list_mask[i], reduction=tf.losses.Reduction.SUM)
        tf.summary.scalar('next_loss_' + str(i), next_loss)
        targets.append((loss_name + '_next_loss_' + str(i), output_next_scores[i], next_labels[i], list_mask[i], "auc"))
        # print_ops.append(tf.print(f"output_next_scores{i}", tf.reshape(output_next_scores[i], [-1]), summarize = 8, output_stream=sys.stdout))
        losses.append(next_loss)

      # l2r 单点模型
      for i in range(LIST_SIZE):
        l2r_loss = tf.losses.log_loss(labels=l2r_labels[i], predictions=output_l2r_scores[i],
                                      # weights = l2r_weights[i] * list_mask[i],
                                      weights = list_mask[i],
                                      reduction=tf.losses.Reduction.SUM)
        tf.summary.scalar('l2r_loss_' + str(i), l2r_loss)
        targets.append((loss_name + '_l2r_loss_' + str(i), output_l2r_scores[i], l2r_labels[i], list_mask[i], "auc"))
        # print_ops.append(tf.print(f"output_l2r_scores{i}", tf.reshape(output_l2r_scores[i], [-1]), summarize = 8, output_stream=sys.stdout))
        losses.append(l2r_loss)

      # play 结构
      # avg_play_weight = sum(play_weights)/6
      for i in range(LIST_SIZE):
        play_loss = tf.losses.log_loss(labels=play_rate[i], predictions=output_play_scores[i],
                                      weights = list_mask[i], reduction=tf.losses.Reduction.SUM)
        tf.summary.scalar('play_loss_' + str(i), play_loss)
        targets.append((loss_name + '_play_loss_' + str(i), output_play_scores[i], play_rate[i], list_mask[i], "linear_regression"))
        # print_ops.append(tf.print(f"play_rate{i}", play_rate[i], summarize = 8, output_stream=sys.stdout))
        losses.append(play_loss)

      # list loss
      # avg_play = tf.reshape(tf.reduce_mean(tf.stack(play_weights, axis = 1), axis=1), [-1,1])
      # list_play_loss = tf.losses.huber_loss(labels=avg_play, predictions = output_list_play_score, reduction=tf.losses.Reduction.SUM, delta = 1.0)
      # tf.summary.scalar('list_play_loss', list_play_loss)
      # losses.append(list_play_loss)

    # list 模型
    output_dict['loss'] = losses
    output_dict['mask'] = [list_mask[i] for i in range(LIST_SIZE)] \
      + [list_mask[i] for i in range(LIST_SIZE)] \
      + [list_mask[i] for i in range(LIST_SIZE)] \
      # +[output_list_play_score]
    output_dict['preds'] = [output_next_scores[i] for i in range(LIST_SIZE)] \
      + [output_l2r_scores[i] for i in range(LIST_SIZE)] \
      + [output_play_scores[i] for i in range(LIST_SIZE)] \
      # +[output_list_play_score]
    output_dict['labels'] = [next_labels[i] for i in range(LIST_SIZE)] \
      + [l2r_labels[i] for i in range(LIST_SIZE)] \
      + [play_rate[i] for i in range(LIST_SIZE)] \
      # + [avg_play]
    output_dict['q_names'] = [f"next" + '_idx' + str(i) for i in range(LIST_SIZE)] \
      + [f"l2r" + '_idx' + str(i) for i in range(LIST_SIZE)] \
      + [f"play" + '_idx' + str(i) for i in range(LIST_SIZE)] \
      # + [f"list_play"]
    print("output_dict ", output_dict)
    if (args.mode == 'predict'):
      for i in range(3 * LIST_SIZE + 1):
        if i < LIST_SIZE:
          output = output_next_scores[i]
          out_name = 'next'
          output = tf.identity(output, "{}".format(out_name + str(i)))
          output_dict[out_name + str(i)] = output
        elif i < LIST_SIZE * 2:
          output = output_l2r_scores[i - LIST_SIZE]
          out_name = 'pos'
          output = tf.identity(output, "{}".format(out_name + str(i - LIST_SIZE)))
          output_dict[out_name + str(i - LIST_SIZE)] = output
        elif i < LIST_SIZE * 3:
          output = output_play_scores[i - LIST_SIZE * 2]
          out_name = 'play'
          output = tf.identity(output, "{}".format(out_name + str(i - LIST_SIZE * 2)))
          output_dict[out_name + str(i - LIST_SIZE * 2)] = output
        # else :
        #   output = output_list_play_score
        #   out_name = 'list'
        #   output = tf.identity(output, "{}".format(out_name + str(i - LIST_SIZE * 3)))
        #   output_dict[out_name + str(i - LIST_SIZE * 3)] = output
    return output_dict

loss_model_dict = {
  # "slide_l2r_4" : poslabel_model,
  # "slide_l2r_3" : poslabel_model,
  "slide_l2r_2" : poslabel_model,
  # "slide_l2r_1" : poslabel_model,
}
parameters_dict = {}
if args.mode == 'train':
  targets = []
  loss_tensor_dict = {}
  dense_loss_tensor_dict = {}

  # q_names_set = set()
  for loss_name, model in loss_model_dict.items():
    label_dict, label_value_dict = get_kuiba_loss_relative(loss_name, parameters_dict, True, True)
    xtr_output = model(parameters_dict, targets, loss_name)
    preds = xtr_output['preds']
    q_names = xtr_output['q_names']
    loss_tensor = xtr_output['loss']
    target_n = 3
    for i, pred in enumerate(preds):
      loss_tensor_dict[q_names[i]] = loss_tensor[i]
      dense_loss_tensor_dict[q_names[i]] = loss_tensor[i]


  sparse_loss=sum_loss_tensor_dict(loss_tensor_dict)
  dense_loss=sum_loss_tensor_dict(dense_loss_tensor_dict)
  print("print_ops ", print_ops)
  with tf.control_dependencies(print_ops):
    logits = tf.reduce_sum(preds[0], axis=-1)
    logits = tf.expand_dims(logits, axis=-1)
    zero = tf.zeros_like(logits)
    one = tf.ones_like(logits)
    print("zero shape", zero.shape)
    print("one shape", one.shape)
    targets.append(('recall', logits, zero, one, 'linear_regression'))

  if args.with_kai:
      config.dump_kai_training_config('./training/conf', targets, loss=None, text=args.text,
                                      dense_loss=dense_loss,
                                      sparse_loss=sparse_loss)
  elif args.with_kai_v2:
      config.set_feature_score_attr("slide_play_rate_idx0", data_source_name="train")
      sparse_optimizer = config.optimizer.Adam(0.0005)
      dense_optimizer = config.optimizer.Adam(0.001)
      sparse_optimizer.minimize(sparse_loss, var_list=config.get_collection(config.GraphKeys.EMBEDDING_INPUT))
      dense_optimizer.minimize(dense_loss, var_list=config.get_collection(config.GraphKeys.TRAINABLE_VARIABLES))
      config.build_model(optimizer=[sparse_optimizer, dense_optimizer], metrics=targets)

else:
  targets = []
  for loss_name, model in loss_model_dict.items():
    label_dict, label_value_dict = get_kuiba_loss_relative(loss_name, parameters_dict, False, False)

    xtr_output= model(parameters_dict, targets, loss_name)
    for (k, v) in xtr_output.items():
      if k.find('pos') != -1:
        targets.append((k, v))
      if k.find('next') != -1:
        targets.append((k, v))
      if k.find('play') != -1:
        targets.append((k, v))
      if k.find('list') != -1:
        targets.append((k, v))
  print("targets ", targets)
  q_names, preds = zip(*targets)
  config.dump_predict_config('./predict', targets, input_type=3, extra_preds=q_names)
