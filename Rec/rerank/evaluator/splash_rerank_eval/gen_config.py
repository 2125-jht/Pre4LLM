#!/bin/python
# -*- coding: UTF8 -*-
from __future__ import print_function
import os
import json
import sys
from feature import gen_params

# 用户侧特征
slide_user_basic = [ # 5+12+10
  "uId", "dId", "uBasicAge", "uGender", "uCityId", 
  "uRealHatePids",
  "uRealClickPids", "uRealLikePids","uRealFollowPids","uRealForwardPids",
  "uMidClickPids", "uMidLikePids", "uMidFollowPids","uMidCommentPids", 
  "uMidPlayEffectivePids", "uMidPlayLongPids","uMidPlayShortPids",
  "uMidPlayViewHetu1","uMidPlayViewHetu2", 
  "fountainClickPids","fountainLikePids", "fountainFollowPids",
  "fountainLongviewPids", "fountainEffviewPids",
  # 上下文特征
  "page",
  "pListSlideNegLabel_1",
  "pListSlideNegLabel_2",
  "pListSlideNegLabel_3",
  "pListSlideNegLabel_4",
  "pListSlideNegLabel_5",
  "pListSlideNegLabel_6",
]

# pid侧特征 分桶后的值
slide_photo_xtr = [
  "pPltr", "pPwtr", "pPftr", "pPhtr", "pPptr", "pPcmtr", "pPctr", "pMcPctr", "pMcPwtr", "pMcPltr",
  "pPlvtr", "pPsvtr", "pPvtr","pPwtd","pMcPlvtr", "pMcPsvtr","pPcmef", "pPepstr",
]

# pid侧特征 原始值
slide_photo_xtr_num = [
  "pvtr"
]

# 24
slide_photo_basic = [
    "pId", "aId", "pDurationMs", "pUploadType", "pCityId", "pProvinceId",
    "pTag", "pMusic", "pAuthorGender", "pAgeHour",
    "pMmuImgClusterV1", "pMmuImgClusterV3", "pMmuContentId", "pOcrCoverTextWordCount", "pMusicComboId",
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

input_slide_l2r = [slide_user_basic + gen_slide_l2r_list_fea(i) for i in range(7)]
input_slide_next = [slide_user_basic + gen_slide_l2r_next_fea(i) for i in range(7)]
input_l2r = [slide_user_basic + gen_l2r_list_fea(i) for i in range(11)]
input_list_all = input_slide_l2r[6] + input_l2r[10] + input_slide_next[6]


def gen_loss():
  loss = {}
  for i in [1, 2, 3, 4]: # list size
    neg = 'pListSlideNegLabel_' + str(i)
    pos = 'pListSlidePosLabel_' + str(i)
    l2r_inputs = ["param." + p for p in input_slide_l2r[i]]
    # next_inputs = ["param." + p for p in input_slide_next[i]]
    labels = {neg: {}, pos : {"sample_rate": 1.0, "expired_output": [1.0]}}
    loss.update({"slide_l2r_" + str(i) : {"type": "LogLoss", "inputs": l2r_inputs, "labels": labels, "auc_uid": "uId" }})

  return loss

print('slide_user_fea_num = {}'.format(len(slide_user_basic)))
print('photo_fea_num = {}'.format(len(slide_photo_basic + slide_photo_xtr+slide_photo_xtr_num)))
print('photo_basic_num = {}'.format(len(slide_photo_basic)))

# print('photo_dim = {}'.format(16 * len(photo_basic + photo_xtr) + 16 * 2))
print('user_dim = {}'.format(32 * 8 + 8 * 13))

################################################################################################

# 生成重加载的slot_id
ps_config = {
    "network": {
        "type": "TFNetwork",
        # PS(参数服务器) 配置
        "parameters": gen_params(),
        # 损失函数配置
        "loss_functions": gen_loss(),
    },
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



config = {'krp_ps_server': ps_config}

with open('dynamic_json_config.json', 'w') as fw:
  fw.write(json.dumps(config, indent=2))
