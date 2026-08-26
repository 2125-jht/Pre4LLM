#!/bin/python
# -*- coding: UTF8 -*-
from __future__ import print_function
import os
import json
import sys
from feature import gen_params


reader_name_suffix = "v1"
reader_name = 'kuiba_tf_fountain_rerank'
model_queue = 'reco_hot_multi_interest_new_model'
btq_shard = 6
ps_shard = 12
listSize = 10
# 24

# 用户侧特征
slide_user_basic = [ # 5+12+10
  "uId", "dId", "uBasicAge", "uGender", "uCityId", 
  "uRealHatePids",
  "uRealClickPids", "uRealLikePids","uRealFollowPids","uRealForwardPids",
  "uMidClickPids", "uMidLikePids", "uMidFollowPids","uMidCommentPids", 
  # "uMidClickAids", "uMidLikeAids","uMidFollowAids","uMidCommentAids"
  "uMidPlayEffectivePids", "uMidPlayLongPids","uMidPlayShortPids",
  # "uMidPlayViewPids", "uMidPlayViewAids", "uMidPlayViewEvLabels",  "uMidPlayViewLvLabels", "uMidPlayViewSvLabels",
  "uMidPlayViewHetu1","uMidPlayViewHetu2", 
  "fountainClickPids","fountainLikePids", "fountainFollowPids",
  "fountainLongviewPids", "fountainEffviewPids",
  
    # 增加内流的hate
  #    "uRequestHour", "uRequestWeekday",
  # "uProvinceId",没有数据
  # "uExpClick", "uExpLike", "uExpFollow", "uExpLongView",训练没有值，预测可以有值
  #"uPlayActionLabel10m", "uPlayActionLabel30m", "uPlayActionLabel5m",
  #"uPlayActionLabel1m", "uPlayActionLabel5m", "uPlayActionLabel10m", "uPlayActionLabel30m", "uPlayActionLabel1h",
]

# pid侧特征
slide_photo_xtr = [
  "pPltr", "pPwtr", "pPftr", "pPhtr", "pPptr", "pPcmtr", "pPctr", "pMcPctr", "pMcPwtr", "pMcPltr",
  "pPlvtr", "pPsvtr", "pPvtr","pPwtd","pMcPlvtr", "pMcPsvtr","pPcmef", "pPepstr",
  # 'pPctr_buck','pPltr_buck','pPwtr_buck','pPftr_buck','pPptr_buck','pPcmtr_buck','pPepstr_buck','pPcmef_buck','pMcPctr_buck','pMcPltr_buck','pMcPwtr_buck',
  # 'pPvtr_buck', 'pPlvtr_buck','pPsvtr_buck','pMcPlvtr_buck','pMcPsvtr_buck',
  # "pctr", "pltr", "pwtr", "pftr", "phtr", "pptr", "pcmtr", "pepstr",  "pcmef",  "mcPctr",  "mcPwtr", "mcPltr", 
  # "pvtr", "plvtr", "psvtr", "mcPlvtr", "mcPsvtr", 
]


slide_photo_xtr_num = [ 
  # "pltr", "pwtr", "pftr", "phtr", "pptr", "pcmtr", "mcPwtr", "mcPltr",  "pcmef", "pepstr", 
  # "pvtr", "plvtr", "psvtr", "mcPlvtr", "mcPsvtr", "pctr", "mcPctr",
  "pvtr"
]

# 24
slide_photo_basic = [
    "pId", "aId", "pDurationMs", "pUploadType", "pCityId", "pProvinceId",
    "pTag", "pMusic", "pAuthorGender", "pAgeHour",
    "pMmuImgClusterV1", "pMmuImgClusterV3", "pMmuContentId", "pOcrCoverTextWordCount", "pMusicComboId",
    #"exp_show_low", "exp_show_high", "exp_click_low", "exp_click_high", "exp_like_low", "exp_like_high", "exp_follow_low", "exp_follow_high",
    #"exp_hate", "exp_report", "fans_count_low", "fans_count_high", 
    "pEmpCtr", "pEmpLtr", "pEmpWtr", "pEmpFtr", "pEmpCmtr", "pEmpHtr", "pEmpPtr", "avg_watchtime","pContentLevel",
    "pHetuTagLevel1Id", "pHetuTagLevel2Id",
]
# label特征
slide_ctr_label = [
  "slide_neg_weight",
  "slide_wtd_label",
  "slide_play_weight",
  "slide_play_rate"
]

slide_next_label = [
  "slide_next_weight",
  "slide_next_label",
]

ctr_label = [  #没有使用
  "l2r_label",
  "l2r_weight",
]

realshow = []
realshow_aid = []
realshow_tag = []
realshow_play = []

name = 'realshow_'
for i in range(30) :
  realshow.append(name + str(i))
  realshow_aid.append(name + 'aid_' + str(i))
  realshow_tag.append(name + 'tag_' + str(i))
  realshow_play.append(name + 'play_' + str(i))

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

atten = realshow + realshow_aid + realshow_tag + realshow_play
atten2 = click + click_aid + click_tag + click_play

# 拼接pid侧特征和label
def gen_slide_l2r_list_fea(n, prefix = '') :
  photo_side = []
  for i in range(n) :
    for x in slide_photo_xtr_num :
      photo_side.append(prefix + x + '_idx' + str(i))
    for x in slide_photo_xtr :
      photo_side.append(prefix + x + '_idx' + str(i))
    for x in slide_photo_basic :
      photo_side.append(prefix + x + '_idx' + str(i))
    for x in slide_ctr_label :
      photo_side.append(x + '_idx' + str(i))
    for x in slide_next_label :
      photo_side.append(x + '_idx' + str(i))
  return photo_side

def gen_l2r_list_fea(n, prefix = '') :
  photo_side = []
  for i in range(n) :
    for x in slide_photo_basic :
      photo_side.append(prefix + x + '_idx' + str(i))
    for x in ctr_label :
      photo_side.append(x + '_idx' + str(i))
  return photo_side

def gen_slide_l2r_next_fea(n, prefix = '') :
  photo_side = []
  for i in range(n) :
    for x in slide_photo_xtr :
      photo_side.append(prefix + x + '_idx' + str(i))
    for x in slide_photo_basic :
      photo_side.append(prefix + x + '_idx' + str(i))
    for x in slide_next_label :
      photo_side.append(x + '_idx' + str(i))
  return photo_side

# input_ctr = user_basic + photo_basic
#input_slide_l2r = slide_user_basic + slide_photo_basic + photo_xtr
#input_slide_l2r = [slide_user_basic + gen_slide_l2r_list_fea(i) for i in range(7)]
input_slide_l2r = [slide_user_basic + gen_slide_l2r_list_fea(i) for i in range(7)]
input_slide_next = [slide_user_basic + gen_slide_l2r_next_fea(i) for i in range(7)]
input_l2r = [slide_user_basic + gen_l2r_list_fea(i) for i in range(11)]
input_list_all = input_slide_l2r[6] + input_l2r[10] + input_slide_next[6]


def gen_loss():
  loss = {}
  # for i in [10]:
  #   neg = 'pListNegLabel_' + str(i)
  #   pos = 'pListPosLabel_' + str(i)
  #   l2r_inputs = ["param." + p for p in input_l2r[i]]
  #   labels = {neg: {"sample_rate": 0.5}, pos : {"sample_rate": 0.5, "expired_output": [1.0]}}
  #   loss.update({"ctr_"+str(i) : {"type": "LogLoss", "inputs": l2r_inputs, "labels": labels, "auc_uid": "uId"}})

  for i in [6]:
    neg = 'pListSlideNegLabel_' + str(i)
    pos = 'pListSlidePosLabel_' + str(i)
    l2r_inputs = ["param." + p for p in input_slide_l2r[i]]
    # next_inputs = ["param." + p for p in input_slide_next[i]]
    labels = {neg: {}, pos : {"sample_rate": 1.0, "expired_output": [1.0]}}
    loss.update({"slide_l2r_" + str(i) : {"type": "LogLoss", "inputs": l2r_inputs, "labels": labels, "auc_uid": "uId" }})
    # loss.update({"slide_next_" + str(i) : {"type": "LogLoss", "inputs": next_inputs, "labels": labels, "auc_uid": "uId" }})
    # loss.update({"slide_ctr_" + str(i) : {"type": "LogLoss", "inputs": l2r_inputs, "labels": labels, "auc_uid": "uId" }})
    #loss.update({"slide_vtr_" + str(i) : {"type": "LogLoss", "inputs": l2r_inputs, "labels": labels, "auc_uid": "uId" }})

  return loss

print('slide_user_fea_num = {}'.format(len(slide_user_basic)))
print('photo_fea_num = {}'.format(len(slide_photo_basic + slide_photo_xtr+slide_photo_xtr_num)))
print('photo_basic_num = {}'.format(len(slide_photo_basic)))

# print('photo_dim = {}'.format(16 * len(photo_basic + photo_xtr) + 16 * 2))
print('user_dim = {}'.format(32 * 8 + 8 * 13))


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
            "path": "viewfs:///home/reco/shaohua/kuiba_tf_hot_simple_ltr_model_sh/",
            "disable_rename": False,
             "save_interval_seconds" : 43200,
             "part_key_num" : 6000000,
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
    #"model_loader": {
    #    "slots": [38, 34, 397, 398],  # 根据param自动设置
    #    "producer": "kuiba",
    #    "consumer": "ps",
    #    "user": "reco",
    #    "threads": 24,
    #    "shard_num": ps_shard,
    #    "load_network_enable": False,  # 不加载network参数时要删除hdfs上的done_file，或者更改reader_name
    #    #"model_path": "viewfs:///home/reco/songwenhao/kuiba_tf_hot_simple_ltr_model_sh/checkpoints/1634868000"
    #    "model_path": "/home/reco/shaohua/kuiba_tf_hot_simple_ltr_model_sh2/checkpoints/1639872000"
    #    #"model_path": "viewfs:/home/reco_4/data/krp/liuhu/krp_kuaishou_hot_krp_tf_training_server_fountain-rr-lh-new/checkpoints/1637366400"
    #},
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

#with open(r"./learner_config/dynamic_json_config.json", "w") as fw:
#    fw.write(json.dumps(learner_config, indent=2))
#
#with open(r"./ps_config/dynamic_json_config.json", "w") as fw:
#    fw.write(json.dumps(ps_config, indent=2))

# os.system("cp server_static.flags dynamic_json_config.json gen_config.py train.py "
#           "/Users/neo/projects/ks/ks/serving_online/krp_reco_photo_collect_model/v1")
