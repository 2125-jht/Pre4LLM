#!/bin/python
# -*- coding: UTF8 -*-
from __future__ import print_function
import os
import json
import sys
from feature import gen_params


reader_name_suffix = "v1"
#reader_name = "kuiba_tf_hot_simple_ltr_model_hua" # + reader_name_suffix
reader_name = 'kai_explore_rerank_list_v1'
model_queue = 'kai_explore_rerank_list_v1'
btq_shard = 6
ps_shard = 6
listSize = 10
# 24

user_basic = [
    "uId", "dId",
    "uBasicAge", "uGender", "uCityId", "uProvinceId",
    "uExpClick", "uExpLike", "uExpFollow", "uExpLongView", "uExpWatchTime",
    "uRequestHour", "uRequestWeekday",
    "uClickPids", "uLikePids", "uFollowPids", "uHatePids", "uFollowAids", "uLikeAids",
    "uPlayActionLabel10m", "uPlayActionLabel30m", "uPlayActionLabel1h",
    "user_view_list_pids",
    "user_view_list_aids",
    "user_view_list_ev",
    "user_view_list_lv",
    "user_view_list_sv",
    "user_view_list_hetu1",
    "user_view_list_hetu2"
    # "uLongTermHetuLevel1topN", "uLongTermHetuLevel2topN", "uLongTermHetuLevel3topN"
]

context_fea = ["maxPctr_context", "maxPltr_context", "maxPwtr_context", "maxPftr_context", "maxPcmtr_context", "maxPcmef_context",
              "maxPptr_context", "maxPepstr_context", "maxPvtr_context", "maxPcltr_context", "maxPfetr_context", "maxPfeff_context",
              "maxPfrscore1_context", "maxPfrscore2_context", "maxPlvtr_context", "maxPsvtr_context", 
              "avgPctr_context", "avgPltr_context", "avgPwtr_context", "avgPftr_context", "avgPcmtr_context", "avgPcmef_context",
              "avgPptr_context", "avgPepstr_context", "avgPvtr_context", "avgPcltr_context", "avgPfetr_context", "avgPfeff_context",
              "avgPfrscore1_context", "avgPfrscore2_context", "avgPlvtr_context", "avgPsvtr_context",
              "avg_duration_context", "hetu_level_one_count", "hetu_level_two_count", "0_9s_duration_photo_count",
              "9_15s_duration_photo_count", "15_20s_duration_photo_count", "20_58s_duration_photo_count", "gt_58s_duration_photo_count"]
# 24
photo_xtr = [
  "pPctr", "pPltr", "pPwtr", "pPftr", "pPhtr", "pPptr", "pPcmtr", "pPcmef",
  "pPfetr", "pPfountainEff", "pMcPctr", "pMcPwtr", "pMcPltr", "pMcPlvtr", "pMcPsvtr", "pPvtr", "pPlvtr", "pPsvtr", "pPfrScore1", "pPfrScore2",
]

# 24
photo_basic = [
     "pId", "aId",
    "pHotLiving", "pDurationMs", "pUploadType", "pCityId", "pProvinceId",
    "pTag", "pMusic", "pAuthorGender", "pAgeHour",
    "pMmuImgClusterV1", "pMmuImgClusterV3", "pMmuContentId", "pOcrCoverTextWordCount", "pMusicComboId",
    #"exp_show_low", "exp_show_high", "exp_click_low", "exp_click_high", "exp_like_low", "exp_like_high", "exp_follow_low", "exp_follow_high",
    #"exp_hate", "exp_report", "fans_count_low", "fans_count_high", 
    "pEmpCtr", "pEmpLtr", "pEmpWtr", "pEmpFtr", "pEmpCmtr", "pEmpHtr", "pEmpPtr", "avg_watchtime",
    "pHetuTagLevel1Id", "pHetuTagLevel2Id", "pContentLevel",
    # "pHetuTagLevel3Id", "pHetuTagLevel5Id","pHetuTagId",
    # "pShortStatShowHetu1100n", "pShortStatShowHetu11000n", "pShortStatClickHetu1100n",
    # "pShortStatClickHetu11000n", "pShortStatShowHetu2100n", "pShortStatShowHetu21000n",
    # "pShortStatClickHetu2100n", "pShortStatClickHetu21000n", "pShortStatShowHetu3100n",
    # "pShortStatShowHetu31000n", "pShortStatClickHetu3100n", "pShortStatClickHetu31000n",
    # "pShortStatShowHetu5100n", "pShortStatShowHetu51000n", "pShortStatClickHetu5100n",
    # "pShortStatClickHetu51000n", "pShortStatShowHetuTag100n", "pShortStatShowHetuTag1000n",
    # "pShortStatClickHetuTag100n", "pShortStatClickHetuTag1000n"
]

ltr_photo_label = [
  "l2r_label",
  "l2r_weight",
]

list_label = [
  'list_label',
  'list_weight'
]

realshow = []
realshow_aid = []
realshow_tag = []
realshow_play = []

name = 'realshow_'
for i in range(30) :
  realshow.append(name + str(i))
  realshow_aid.append(name + 'aid_' + str(i))
  # realshow_tag.append(name + 'tag_' + str(i))
  # realshow_play.append(name + 'play_' + str(i))

click = []
click_aid = []
click_tag = []
click_play = []

name = 'click_'
for i in range(30) :
  click.append(name + str(i))
  click_aid.append(name + 'aid_' + str(i))
  click_tag.append(name + 'tag_' + str(i))
  click_play.append(name + 'play_' + str(i))

def gen_l2r_list_fea(n, prefix = 'l2r_') :
  photo_side = []
  for i in range(n) :
    for x in photo_xtr :
      photo_side.append(prefix + x + '_idx' + str(i))
    for x in photo_basic :
      photo_side.append(prefix + x + '_idx' + str(i))
    for x in ltr_photo_label :
      photo_side.append(x + '_idx' + str(i))
  for x in context_fea:
    photo_side.append(prefix + x)
  for x in list_label:
    photo_side.append(x)
  return photo_side

def gen_ctr_user(prefix = 'l2r_'):
  user_side = []
  for x in user_basic:
    user_side.append(prefix + x)
  return user_side

def gen_list_fea(prefix = 'list_'):
  user_side = []
  for x in user_basic:
    user_side.append(prefix + x)
  return user_side

input_l2r = [gen_ctr_user() + gen_l2r_list_fea(i) for i in range(11)]
input_list_all = input_l2r[10]

def gen_loss():
  loss = {}
  pos = 'maskPos'
  neg = 'maskNeg'
  for i in [10]:
    l2r_inputs = ["param." + p for p in input_l2r[i]]
    labels = {neg: {}, pos : {"sample_rate": 1.0, "expired_output": [1.0]}}
    loss.update({"l2r_"+str(i) : {"type": "LogLoss", "inputs": l2r_inputs, "labels": labels, "auc_uid": "uId"}})
    loss.update({"ctr_"+str(i) : {"type": "LogLoss", "inputs": l2r_inputs, "labels": labels, "auc_uid": "uId"}})
  # list_pos = 'PureClickListPos'
  # list_neg = 'PureClickListNeg'
  # for i in [10]:
  #   l2r_inputs = ["param." + p for p in input_l2r[i]]
  #   labels = {list_neg: {}, list_pos : {"sample_rate": 1.0, "expired_output": [1.0]}}
  #   loss.update({"context_ltr_1" : {"type": "LogLoss", "inputs": l2r_inputs, "labels": labels, "auc_uid": "uId"}})
    

  # for i in [6]:
  #   pos = 'pListSlidePosLabel_' + str(i)
  #   neg = 'pListSlideNegLabel_' + str(i)
  #   l2r_inputs = ["param." + p for p in input_slide_l2r[i]]
  #   labels = {neg: {}, pos : {"sample_rate": 1.0, "expired_output": [1.0]}}
  #   loss.update({"slide_ctr_"+str(i) : {"type": "LogLoss", "inputs": l2r_inputs, "labels": labels, "auc_uid": "uId"}})

  # for i in [2]:
  #   pos = 'pListSlidePosLabel_' + str(i)
  #   neg = 'pListSlideNegLabel_' + str(i)
  #   l2r_inputs = ["param." + p for p in input_slide_l2r[i]]
  #   labels = {neg: {}, pos : {"sample_rate": 1.0, "expired_output": [1.0]}}
  #   loss.update({"slide_detail_ctr_"+str(i) : {"type": "LogLoss", "inputs": l2r_inputs, "labels": labels, "auc_uid": "uId"}})

  return loss

print('user_fea_num = {}'.format(len(user_basic)))
print('photo_fea_num = {}'.format(len(photo_basic + photo_xtr)))
print('context_fea_num = {}'.format(len(context_fea)))

print('photo_dim = {}'.format(8 * len(photo_basic + photo_xtr) + 24 * 2))
print('user_dim = {}'.format(32*4 + 4*5 + 5*8 + 32*6 + 24*5 + 4 * 8 + 8*3))


#metric_name = ["count", "loss@real_loss", "rate", "xgauc", "xauc", "auc"]
#def gen_metric() :
#  metric = {}
#  for i in range(10):
#    metric.update({"ctr_idx" + str(i) : metric_name})
#    metric.update({"l2r_idx" + str(i) : metric_name})
#  return metric
################################################################################################

# 生成重加载的slot_id
ps_config = {
    # shm 的目录
    "shm_dir": "/dev/shm/ps",
    # ps 内存上限
    "ps_memory": (1 << 30) * 200,
    # ps 参数数量上限
    "ps_capacity": (1 << 30),
    # ps 分成几个 shard
    "shard_num": ps_shard,
    "sample_reader": {
        "type_name": "BTQueueSampleReader",
        "group": "hdfs_realtime_log",
        "begin_time": "20200103200000",
        # 对应模型的名字, 作为一个训练任务的唯一标识
        "reader_name": reader_name,
        # "reader_name_suffix": reader_name_suffix,
        "shard_num": ps_shard,  # ps 分成几个 shard
        "buffer_size": 8192,
        "queues": [],
    },
    "eval": {
        "eval_window_seconds": 300,
        "eval_window_samples": 0,
    },
    "disable_rename_model_queue": True,
    "network": {
        "shard_num": ps_shard,
        "model_queue": {
            "queue_group": "reco_mem_common_model",
            "queue_name": model_queue,
            # predict server 的 shard 数，可以跟 ps shard 不一致（准确说是btq的shard数）
            "shard_num": btq_shard,
            # "queue_handler": "infra_btq",
            # "batch_update_period_seconds": 3600 * 2,
            # "warmup": True
            "disable_rename": True,
        },
        "checkpoint": {
            "path": "viewfs:///home/reco_4/data/krp/xuwei09/fountain_rerank_listlabel_wtd/",
            "disable_rename": True,
             "save_interval_seconds" : 43200,
             "part_key_num" : 10000000,
            # "reserve_days" : 7,
        },
        "updater_type": "ada_momentum_updater",
        "type": "TFNetwork",
        # PS(参数服务器) 配置
        "parameters": gen_params(),
            # "online_push_limit": 0,
            # model params 以下特征均有有效值，注释掉的可以继续配置样本解析并使用
            ####################### user attr  #######################
            # user基础画像
        # 网络 layer 配置
        "layers": {
            "default_batch_num": 1,
            "default_batch_decay": 0.9999,
            "default_move_length": 0.001,
            "default_initial_lr": 0.000001,
            "default_mom_decay_rate": 0.99,
            "default_ada_decay_rate": 0.9999,
        },
        # 损失函数配置
        "loss_functions": gen_loss(),
    },
    # "model_loader": {
    #    "slots": [],  # 根据param自动设置
    #    "producer": "kuiba",
    #    "consumer": "ps",
    #    "user": "reco",
    #    "threads": 24,
    #    "shard_num": 6,
    #    "load_network_enable": False,  # 不加载network参数时要删除hdfs上的done_file，或者更改reader_name
    #    "model_path": "viewfs:///home/reco/shaohua/kuiba_tf_hot_simple_ltr_model_sh/checkpoints/1646611200"
    #    #"model_path": "viewfs:///home/reco/shaohua/kuiba_tf_hot_simple_ltr_model_swh/checkpoints/1635048000"
    # },
    "online_update": True,
    "__SERVER_NAME": "paas_ps_server",
    "__SERVER_PART": "default_part",
    "__SERVER_SHARD": "zw",
    "server_config": {
    },
    "kess_config": {
    },
    "status": "training",
}

# 生成重加载的slot_id
slot_names = []
for item in ps_config['network']['loss_functions']:
    slot_names += ps_config['network']['loss_functions'][item]['inputs']
slot_ids = [attr['key_type'] for name in set(slot_names)
            for attr in ps_config['network']['parameters'][name[6:]]['attrs']
            if attr['key_type'] < 1000]
slots = set(sorted(slot_ids))
#ps_config['model_loader']['slots'] = list(slots) 

# 删除无用的attr配置
unused_attrs = [attr_name for attr_name in ps_config['network']['parameters'].keys()
                if not attr_name.startswith("default") and attr_name not in input_list_all]
for attr_name in unused_attrs:
    del ps_config['network']['parameters'][attr_name]


def print_feature_size(side_name, feature_list):
    print('%s feature:\t%d\t dim sum:\t%d' %
          (side_name, len(feature_list), sum([ps_config['network']['parameters'][attr].get('dim')
                                              or ps_config['network']['parameters']['default_dim']
                                              for attr in feature_list])), file=sys.stderr)
    print('\n'.join(feature_list))


#print_feature_size("ltr user", user_basic)
#print_feature_size("ltr photo", photo_side)
#print_feature_size("ltr TOTAL", input_list_all)

# print_feature_size("ctr user", ctr_user_side)
# print_feature_size("ctr photo", ctr_photo_side)
# print_feature_size("ctr TOTAL", input_list_ctr)


learner_config = {
    # 读者类型
    "reader_type": "kafka",
    # 对应 ps server SampleReader 的 reader_name, 是模型训练的唯一标识
    "reader_name": reader_name,
    # "reader_name_suffix": reader_name_suffix,
    # ps shard 数
    "shard_num": ps_shard,
    # 读取线程
    "reading_threads": 12,
    # 训练线程
    "learner_threads": 12,
    # mini batch: 一次计算多少样本
    "mini_batch_size": 256,
    # network 的梯度 多少个 mini_batch 后更新回参数服务器
    "merge_size": 16,
    "fetch_sample": {
        "pass_size": 1024,
        "read_size": 128,
        #"filter": {
        #    "type": "simple",
        #    "attr": "pBadXtr",
        #    "value_set": [
        #    0
        #    ]
        #},
        "compress_sample": True,
    },
    "kafka": {
        "cluster": ["bjlt-reco2"],
        "offset_ms_ago": 12 * 3600 * 1000,
        "pass_size": 2560,
        "read_size": 256,
        "reader_name": reader_name,
        #"topic": "reco_produce_model_joint_log",
        "topic": "reco_hot_context_rank_joint_log",
    },
    "bt_queue": {
        "pass_size": 1024,
        "read_size": 4096,
        "group": "",
        "queue_name": "",
        "reader_name": "",
        "begin_time": "",
        "compress_sample": True,
    },
    "hdfs": {
        "pass_size": 1024,
        "read_size": 4096,
        "version": 21,
        "common_prefix": "",
        "data_path": [""],
    },
    # 自行改动
    "__SERVER_NAME": "paas_learner_server",
    "__SERVER_PART": "default_part",
    "__SERVER_SHARD": "default_shard",
    "eval_interval_seconds": 300,
}

config = {'krp_tf_learner_server': learner_config,
          'krp_ps_server': ps_config}

with open('dynamic_json_config.json', 'w') as fw:
    fw.write(json.dumps(config, indent=2))

# with open(r"./learner_config/dynamic_json_config.json", "w") as fw:
#     fw.write(json.dumps(learner_config, indent=2))

# with open(r"./ps_config/dynamic_json_config.json", "w") as fw:
#     fw.write(json.dumps(ps_config, indent=2))

# os.system("cp server_static.flags dynamic_json_config.json gen_config.py train.py "
#   