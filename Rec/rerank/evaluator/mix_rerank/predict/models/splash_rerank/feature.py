#!/bin/python
# -*- coding: UTF8 -*-
from __future__ import print_function
import os
import json
import sys

ID_DIM = 16
CATE_DIM = 16
LIST_JOIN_LIMIT = 32
global_slot = 500
default_dim = 16
listSize = 10

default_param = {
    "default_online_push_limit": 5,
    # type 默认是 1:EMBEDDING_PARAMETER, 其它类型有 2:LR_PARAMETER 3:THIRD_PATRY_DATA
    "default_type": 1,
    "default_dim": default_dim,
    "default_batch_num": 1,
    "default_batch_decay": 0.99,
    "default_move_length": 0.01,

    # expire_second 默认是 -1, 不过期
    "default_expire_second": -1,
    "default_decay_rate": 0.9999,
    "default_initial_lr": 0.05,
    "default_initial_g2sum": 3,
}

user_param = {
  "uId": {"attrs": [{"attr": ["uId"], "key_type": 38, "converter": "id"}], "dim": 32},
  "dId": {"attrs": [{"attr": ["dId"], "key_type": 34, "converter": "id"}], "dim": 32},
  "uCityId": {"attrs": [{"attr": ["uCityId"], "key_type": 53, "converter": "id"}], "dim": 4},
  "uProvinceId": {"attrs": [{"attr": ["uRequstProvinceId"], "key_type": 54, "converter": "id"}], "dim": 4},
  "uIsDouYin": {"attrs": [{"attr": ["uIsDouYin"], "key_type": 55, "converter": "id"}], "dim": 4},
  "uClickPids": {"attrs": [{"attr": ["uRealtimeClickList"], "key_type": 113, "converter": "list"}], "dim": ID_DIM, "expire_second": 604800},
  "uLikePids": {"attrs": [{"attr": ["uRealtimeLikeList"], "key_type": 114, "converter": "list"}], "dim": ID_DIM, "expire_second": 604800},
  "uFollowPids": {"attrs": [{"attr": ["uRealtimeFollowList"], "key_type": 115, "converter": "list"}], "dim": ID_DIM, "expire_second": 604800},
  "uHatePids": {"attrs": [{"attr": ["uRealtimeNegativeList"], "key_type": 116, "converter": "list"}], "dim": ID_DIM, "expire_second": 604800},
  "uFollowAids": {"attrs": [{"attr": ["uFollowPhotoAuthorList"], "key_type": 117, "converter": "list"}], "dim": ID_DIM, "expire_second": 604800},
  "uLikeAids": {"attrs": [{"attr": ["uLikePhotoAuthorList"], "key_type": 118, "converter": "list"}], "dim": ID_DIM, "expire_second": 604800},

  #l2r_ 特征值 = std::max(std::min(numerator / (denominator + smooth), max_val), min_value) * buckets

  "uExpClick": {"attrs": [{"attr": ["uExpClick"], "key_type": 58, "converter": "discrete", "converter_args": "10,1,10000000,1"}], "dim": 8},
  "uExpLike": {"attrs": [{"attr": ["uExpLike"], "key_type": 59, "converter": "discrete", "converter_args": "1,1,10000000,1"}], "dim": 8},
  "uExpFollow": {"attrs": [{"attr": ["uExpFollow"], "key_type": 60, "converter": "discrete", "converter_args": "1,1,10000,1"}], "dim": 8},
  "uExpLongView": {"attrs": [{"attr": ["uExpLongView"], "key_type": 61, "converter": "discrete", "converter_args": "1,1,10000,1"}], "dim": 8},
  "uExpWatchTime": {"attrs": [{"attr": ["uExpWatchTime"], "key_type": 62, "converter": "discrete", "converter_args": "1000,1,10000000,1"}], "dim": 8},
  "fountainClickPids": {"attrs": [{"attr": ["featureFountainProfileClickPidList"], "key_type": 63, "converter": "list"}], "dim": 24, "expire_second": 604800},
  "fountainLikePids": {"attrs": [{"attr": ["featureFountainProfileLikePidList"], "key_type": 64, "converter": "list"}], "dim": 24, "expire_second": 604800},
  "fountainFollowAids": {"attrs": [{"attr": ["featureFountainProfileFollowAidList"], "key_type": 65, "converter": "list"}], "dim": 24, "expire_second": 604800},
  "fountainLongviewPids": {"attrs": [{"attr": ["featureFountainProfileLongViewPidList"], "key_type": 66, "converter": "list"}], "dim": 24, "expire_second": 604800},
  "fountainEffviewPids": {"attrs": [{"attr": ["featureFountainProfileEffViewPidList"], "key_type": 67, "converter": "list"}], "dim": 24, "expire_second": 604800},

  "uBasicAge": {"attrs": [{"attr": ["uBasicAge"], "key_type": 68, "converter": "id"}], "dim": 4, },
  "uGender": {"attrs": [{"attr": ["uBasicGender"], "key_type": 69, "converter": "id"}], "dim": 4, },
}


def getDiscreateFea(name, attr, converter_args, dim, slot, expire) :
  return {loss_name + name: {"attrs": [{"key_type": slot, "converter": "discrete", "converter_args": converter_args, "attr": [attr], }, ], "dim": dim, "expire_second": expire},}

def getIdFea(name, attr, dim, slot, expire, converter) :
  return {loss_name + name: {"attrs": [{"attr": [attr], "key_type": slot, "converter": converter}], "dim": dim, "expire_second": expire},}

def getNumericFea(name, attr, slot) :
   return {loss_name + name: {"attrs": [{"attr": [attr], "key_type": slot, "converter": "numeric"}], "dim" : 1, },}

photo_param = {}


def genDiscreateFea(attr, convert, name = None, dim = None, slot = None, expire = None) :
  global global_slot
  if slot is None:
    slot = global_slot
    global_slot += 1
  if name is None:
    name = attr
  if dim is None:
    dim = default_dim
  if expire is None:
    expire = -1
  for i in range(listSize) :
    photo_param.update(getDiscreateFea(name + '_idx' + str(i), attr + '_idx' + str(i), convert, dim, slot, expire))
  photo_param.update(getDiscreateFea(name, attr, convert, dim, slot, expire))
    
def genIdFea(attr, name = None, dim = None, slot = None, expire = None, converter = "id") :
  global global_slot
  if slot is None:
    slot = global_slot
    global_slot += 1
  if name is None:
    name = attr
  if dim is None:
    dim = default_dim
  if expire is None:
    expire = -1
  for i in range(listSize) :
    photo_param.update(getIdFea(name + '_idx' + str(i), attr + '_idx' + str(i), dim, slot, expire, converter))
  photo_param.update(getIdFea(name, attr, dim, slot, expire, converter))

def genNumericFea(attr, name = None, slot = None) :
  global global_slot
  if slot is None:
    slot = global_slot
    global_slot += 1
  if name is None:
    name = attr
  for i in range(listSize) :
    photo_param.update(getNumericFea(name + '_idx' + str(i), attr + '_idx' + str(i), slot))
  photo_param.update(getNumericFea(name, attr, slot))

def genPhotoFea(loss) :
  # "denominator,smooth,max_val,buckets,min_val,expr"
  # 特征值 = std::max(std::min(numerator / (denominator + smooth), max_val), min_value) * buckets
  global loss_name
  loss_name = loss
  genDiscreateFea('pPctr', "1,0,1,10000,-1", dim = 16)
  genDiscreateFea('pPltr', "0.5,0,1,2000,-1")
  genDiscreateFea('pPptr', "0.5,0,1,2000,-1")
  genDiscreateFea('pPwtr', "0.2,0,1,2000,-1")
  genDiscreateFea('pPcmtr', "0.2,0,1,2000,-1")
  genDiscreateFea('pPftr', "0.2,0,1,2000,-1")
  genDiscreateFea('pPhtr', "0.1,0,1,2000,-1")
  genDiscreateFea('pPvtr', "0.1,0,1,2000,-1")
  genDiscreateFea('pPcmef', "0.2,0,1,2000,-1")
  genDiscreateFea('pPlvtr', "1,0,1,2000,-1")
  genDiscreateFea('pPsvtr', "1,0,1,2000,-1")
  genDiscreateFea('pPfrScore1', "1,0,1,2000")
  genDiscreateFea('pPfrScore2', "1,0,120,2000")
  genDiscreateFea('pPepstr', "1,0,1,2000,-1")

  genDiscreateFea('pMcPctr', "1,0,1,10000,-1", dim = 16)
  genDiscreateFea('pMcPltr', "0.5,0,1,2000,-1")
  genDiscreateFea('pMcPptr', "0.5,0,1,2000,-1")
  genDiscreateFea('pMcPwtr', "0.2,0,1,2000,-1")
  genDiscreateFea('pMcPcmtr', "0.2,0,1,2000,-1")
  genDiscreateFea('pMcPdtr', "0.2,0,1,2000,-1")
  genDiscreateFea('pMcPcltr', "0.2,0,1,2000,-1")
  genDiscreateFea('pMcPlvtr', "1,0,1,2000,-1")
  genDiscreateFea('pMcPsvtr', "1,0,1,2000,-1")
  genDiscreateFea('pMcPlvtr2', "1,0,1,2000,-1")
  genDiscreateFea('pMcPepstr', "1,0,1,2000,-1")
  genDiscreateFea('pMcPcestr', "1,0,1,2000,-1")

  genDiscreateFea('pEmpCtr', "1,0,1,10000,0")
  genDiscreateFea('pEmpLtr', "0.3,0,1,1000,0")
  genDiscreateFea('pEmpWtr', "0.3,0,1,1000,0")
  genDiscreateFea('pEmpFtr', "0.1,0,1,1000,0")
  genDiscreateFea('pEmpCmtr', "0.3,0,1,1000,0")
  genDiscreateFea('pEmpHtr', "0.001,0,1,1000,0")
  genDiscreateFea('pEmpPtr', "0.1,0,1,1000,0")

  genIdFea('pId', slot = 26, dim = 32,  expire = 7 * 24 * 3600)
  genIdFea('aId', slot = 128, dim = 32,  expire = 30 * 24 * 3600)
  genIdFea('pHotExptag')
  genIdFea('pHotLiving')
  genIdFea('pDurationMs', slot = 133, dim = 4)
  genIdFea('pUploadType')
  genIdFea('pCityId')
  genIdFea('pProvinceId')
  genIdFea('pTag')
  genIdFea('pContentLevel', dim = 2)
  genIdFea('pHetuTagLevel1Id', slot = 129, dim = 4, converter = "list")
  genIdFea('pHetuTagLevel2Id', slot = 130, dim = 4, converter = "list")
  genIdFea('pMusic')
  genIdFea('pAuthorGender')
  genIdFea('pAgeHour')
  genIdFea('pMmuImgClusterV1')
  genIdFea('pMmuImgClusterV3', slot = 142, dim = 4)
  genIdFea('pMmuContentId')
  genIdFea('pOcrCoverTextWordCount')
  genIdFea('pMusicComboId')

  genDiscreateFea('pHotShow', "200,0,100,1,0", 'exp_show_low')
  genDiscreateFea('pHotShow', "20000,0,100,1,0", 'exp_show_high')
  genDiscreateFea('pHotClick', "10,0,100,1,0", 'exp_click_low')
  genDiscreateFea('pHotClick', "5000,0,100,1,0", 'exp_click_high')
  genDiscreateFea('pHotLike', "1,0,100,1,0", 'exp_like_low')
  genDiscreateFea('pHotLike', "500,0,200,1,0", 'exp_like_high')
  genDiscreateFea('pHotFollow', "1,0,100,1,0", 'exp_follow_low')
  genDiscreateFea('pHotFollow', "250,0,200,1,0", 'exp_follow_high')
  genDiscreateFea('pHotHate', "1,0,200,1,0", 'exp_hate')
  genDiscreateFea('pHotReport', "1,0,200,1,0", 'exp_report')
  genDiscreateFea('pAuthorFansCount', "100,0,100,1,0", 'fans_count_low')
  genDiscreateFea('pAuthorFansCount', "10000,0,100,1,0", 'fans_count_high')
  genDiscreateFea('pAvgWatchtime', "1000,0,60,1,0", 'avg_watchtime')


def genSlidePhotoFea(loss):
  global loss_name
  loss_name = loss

  genNumericFea('pPctr', 'pctr')
  genNumericFea('pPltr', 'pltr')
  genNumericFea('pPptr', 'pptr')
  genNumericFea('pPwtr', 'pwtr')
  genNumericFea('pPcmtr', 'pcmtr')
  genNumericFea('pPftr', 'pftr')
  genNumericFea('pPhtr', 'phtr')
  genNumericFea('pPvtr', 'pvtr')
  genNumericFea('pPcmef', 'pcmef')
  genNumericFea('pPlvtr', 'plvtr')
  genNumericFea('pPsvtr', 'psvtr')
  genNumericFea('pPepstr', 'pepstr')
  genNumericFea('pMcPctr', 'mcPctr')
  genNumericFea('pMcPltr', 'mcPltr')
  genNumericFea('pMcPptr', 'mcPptr')
  genNumericFea('pMcPwtr', 'mcPwtr')
  genNumericFea('pMcPcmtr', 'mcPcmtr')
  genNumericFea('pMcPlvtr', 'mcPlvtr')
  genNumericFea('pMcPsvtr', 'mcPsvtr')
  genNumericFea('pMcPepstr', 'mcPepstr')
  genNumericFea('pMcPcestr', 'mcPcestr')
  genNumericFea('pPfrScore1', 'pfrScore1')
  genNumericFea('pPfrScore2', 'pfrScore2')
  genNumericFea('pEmpCtr', 'empCtr')
  genNumericFea('pEmpLtr', 'empLtr')
  genNumericFea('pEmpWtr', 'empWtr')
  genNumericFea('pEmpFtr', 'empFtr')
  genNumericFea('pEmpHtr', 'empHtr')
  genNumericFea('pEmpPtr', 'empPtr')
  genNumericFea('pEmpCmtr', 'empCmtr')
  genNumericFea('pDurationMs', 'pDuration')

def gen_label():
  global loss_name
  loss_name = ''
  genNumericFea('SlideEdgeWtdPosWeight', 'slide_mix_label')
  genNumericFea('SlideEdgeWtdPosLabel', 'slide_wtd_label')
  genNumericFea('SlideEvtrPosLabel', 'slide_ctr_label')
  genNumericFea('BottomEdgeWtdPosWeight', 'bottom_mix_label')
  genNumericFea('BottomEdgeWtdPosLabel', 'bottom_wtd_label')
  genNumericFea('BottomEvtrPosLabel', 'bottom_ctr_label')
  genNumericFea('BottomEdgeWtdNegWeight', 'bottom_neg_weight')
  genNumericFea('SlideEdgeWtdNegWeight', 'slide_neg_weight')
  genNumericFea('SlideEdgeWtdMaxWeight', 'slide_max_weight')
  genNumericFea('pSingleLabel', 'l2r_label')
  genNumericFea('pSingleWeightV2', 'l2r_weight')

def gen_user_fea(n) :
  name = 'realshow_'
  for i in range(n) :
    key = name + str(i)
    user_param.update({key: {"attrs": [{"key_type": 397, "converter": "id", "attr": [key], }, ], "dim": 32, "expire_second": 86400 * 7, "use_common_attr_only" : True}})
    key = name + "aid_" + str(i)
    user_param.update({key: {"attrs": [{"key_type": 398, "converter": "id", "attr": [key], }, ], "dim": 32, "expire_second": 86400 * 7, "use_common_attr_only" : True}})
    key = name + "tag_" + str(i)
    user_param.update({key: {"attrs": [{"key_type": 7, "converter": "id", "attr": [key], }, ], "dim": 8, "expire_second": 86400 * 7, "use_common_attr_only" : True}})
    key = name + "play_" + str(i)
    user_param.update({key: {"attrs": [{"key_type": 8, "converter": "id", "attr": [key], }, ], "dim": 8, "expire_second": 86400 * 7, "use_common_attr_only" : True}})

def gen_params() :
  ans = default_param
  gen_user_fea(30)
  ans.update(user_param)
  genPhotoFea('')
  gen_label()
  genSlidePhotoFea('')
  ans.update(photo_param)
  print('user_total_param_num = {}'.format(len(user_param)))
  print('photo_total_param_num = {}'.format(len(photo_param)))
  global global_slot
  print("global_slot = {}".format(global_slot))
  return ans
