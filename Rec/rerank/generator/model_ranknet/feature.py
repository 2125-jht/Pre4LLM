#!/bin/python
# -*- coding: UTF8 -*-
from __future__ import print_function
import os
import json
import sys

ID_DIM = 32
CATE_DIM = 16
LIST_JOIN_LIMIT = 32
global_slot = 500
default_dim = 8
listSize = 10
view_list_length = 50

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
  "uRequestHour": {"attrs": [{"attr": ["uRequestHour"], "key_type": 56, "converter": "id"}], "dim": 4},
  "uRequestWeekday": {"attrs": [{"attr": ["uRequestWeekday"], "key_type": 57, "converter": "id"}], "dim": 4},
  "uBasicAge": {"attrs": [{"attr": ["uBasicAge"], "key_type": 68, "converter": "id"}], "dim": 4, },
  "uGender": {"attrs": [{"attr": ["uBasicGender"], "key_type": 69, "converter": "id"}], "dim": 4, },

  "uExpClick": {"attrs": [{"attr": ["uExpClick"], "key_type": 58, "converter": "discrete", "converter_args": "10,1,10000000,1"}], "dim": 8},
  "uExpLike": {"attrs": [{"attr": ["uExpLike"], "key_type": 59, "converter": "discrete", "converter_args": "1,1,10000000,1"}], "dim": 8},
  "uExpFollow": {"attrs": [{"attr": ["uExpFollow"], "key_type": 60, "converter": "discrete", "converter_args": "1,1,10000,1"}], "dim": 8},
  "uExpLongView": {"attrs": [{"attr": ["uExpLongView"], "key_type": 61, "converter": "discrete", "converter_args": "1,1,10000,1"}], "dim": 8},
  "uExpWatchTime": {"attrs": [{"attr": ["uExpWatchTime"], "key_type": 62, "converter": "discrete", "converter_args": "1000,1,10000000,1"}], "dim": 8},


  # 实时reallist "uClickPids": {"attrs": [{"attr": ["uRealtimeClickList"], "key_type": 120, "converter": "list"}], "join_limit": 20, "type": 5, "dim": ID_DIM, "expire_second": 604800},
  "uClickPids": {"attrs": [{"attr": ["uRealtimeClickList"], "key_type": 120, "converter": "list","converter_args": {"enable_filter": False}}], "dim": ID_DIM, "expire_second": 604800},
  "uLikePids": {"attrs": [{"attr": ["uRealtimeLikeList"], "key_type": 121, "converter": "list", "converter_args": {"enable_filter": False}}], "dim": ID_DIM, "expire_second": 604800},
  "uFollowPids": {"attrs": [{"attr": ["uRealtimeFollowList"], "key_type": 122, "converter": "list", "converter_args": {"enable_filter": False}}], "dim": ID_DIM, "expire_second": 604800},
  "uHatePids": {"attrs": [{"attr": ["uRealtimeNegativeList"], "key_type": 123, "converter": "list", "converter_args": {"enable_filter": False}}], "dim": ID_DIM, "expire_second": 604800},
  
  "uFollowAids": {"attrs": [{"attr": ["uFollowPhotoAuthorList"], "key_type": 124, "converter": "list"}], "dim": ID_DIM, "expire_second": 604800},
  "uLikeAids": {"attrs": [{"attr": ["uLikePhotoAuthorList"], "key_type": 125, "converter": "list"}], "dim": ID_DIM, "expire_second": 604800},

  # 内流list l2r_ 特征值 = std::max(std::min(numerator / (denominator + smooth), max_val), min_value) * buckets
  "fountainClickPids": {"attrs": [{"attr": ["featureFountainProfileClikPidList"], "key_type": 63, "converter": "list","converter_args": {"enable_filter": False}}], "dim": ID_DIM, "expire_second": 86400*168, "type": 5, "join_limit": view_list_length},
  "fountainLikePids": {"attrs": [{"attr": ["featureFountainProfileLikePidList"], "key_type": 64, "converter": "list" ,"converter_args": {"enable_filter": False}}], "dim": ID_DIM, "expire_second": 86400*168, "type": 5, "join_limit": view_list_length},
  "fountainFollowAids": {"attrs": [{"attr": ["featureFountainProfileFollowAidList"], "key_type": 65, "converter": "list","converter_args": {"enable_filter": False}}], "dim": ID_DIM, "expire_second": 86400*168, "type": 5, "join_limit": view_list_length},
  "fountainLongviewPids": {"attrs": [{"attr": ["featureFountainProfileLongViewPidList"], "key_type": 66, "converter": "list","converter_args": {"enable_filter": False}}], "dim": ID_DIM, "expire_second": 86400*168, "type": 5, "join_limit": view_list_length},
  "fountainEffviewPids": {"attrs": [{"attr": ["featureFountainProfileEffViewPidList"], "key_type": 67, "converter": "list","converter_args": {"enable_filter": False}}], "dim": ID_DIM, "expire_second": 86400*168, "type": 5, "join_limit": view_list_length},

  # 全站曝光list，200条
  "user_view_list_pids": {"attrs": [{"key_type": 26, "attr": ["uViewPidListV1"], "converter": "list","converter_args": {"enable_filter": False}}], "dim": 32, "expire_second": 86400*168},
  "user_view_list_aids": {"attrs": [{"key_type": 128, "attr": ["uViewAidListV1"], "converter": "list","converter_args": {"enable_filter": False}}], "dim": 32, "expire_second": 86400*168},
  "user_view_list_ev": {"attrs": [{"key_type": 321, "attr": ["uEffectiveViewLabelListV1"], "converter": "list","converter_args": {"enable_filter": False}}], "dim": 32, "expire_second": 86400*168},
  "user_view_list_lv": {"attrs": [{"key_type": 322, "attr": ["uLongViewLabelListV1"], "converter": "list","converter_args": {"enable_filter": False}}], "dim": 32, "expire_second": 86400*168},
  "user_view_list_sv": {"attrs": [{"key_type": 323, "attr": ["uShortViewLabelListV1"], "converter": "list","converter_args": {"enable_filter": False}}], "dim": 32, "expire_second": 86400*168},
  "user_view_list_hetu1": {"attrs": [{"key_type": 324, "attr": ["uViewHetu1ListV1"], "converter": "list","converter_args": {"enable_filter": False}}],"dim": 32, "expire_second": 86400*168},
  "user_view_list_hetu2": {"attrs": [{"key_type": 325, "attr": ["uViewHetu2ListV1"], "converter": "list","converter_args": {"enable_filter": False}}], "dim": 32, "expire_second": 86400*168},
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
  genIdFea('pHetuTagLevel1Id', slot = 129, dim = 8, converter = "list")
  genIdFea('pHetuTagLevel2Id', slot = 130, dim = 8, converter = "list")
  genIdFea('pMusic')
  genIdFea('pAuthorGender')
  genIdFea('pMmuImgClusterV1')
  genIdFea('pMmuImgClusterV3', slot = 142, dim = 8)
  genIdFea('pMmuContentId')
  genIdFea('pOcrCoverTextWordCount')
  genIdFea('pMusicComboId')
  genIdFea("pCurrentPage")
  genNumericFea('position')
  genIdFea('pAgeHour')

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
  genNumericFea('pPctr', 'pPctr')
  genNumericFea('pPltr', 'pPltr')
  genNumericFea('pPptr', 'pPptr')
  genNumericFea('pPwtr', 'pPwtr')
  genNumericFea('pPcmtr', 'pPcmtr')
  genNumericFea('pPftr', 'pPftr')
  genNumericFea('pPhtr', 'pPhtr')
  genNumericFea('pPvtr', 'pPvtr')
  genNumericFea('pPwtd', 'pPwtd')
  genNumericFea('pPcmef', 'pPcmef')
  genNumericFea('pPlvtr', 'pPlvtr')
  genNumericFea('pPsvtr', 'pPsvtr')
  genNumericFea('pPepstr', 'pPepstr')
  genNumericFea('pMcPctr', 'pMcPctr')
  genNumericFea('pMcPltr', 'pMcPltr')
  genNumericFea('pMcPptr', 'mcPptr')
  genNumericFea('pMcPwtr', 'pMcPwtr')
  genNumericFea('pMcPcmtr', 'mcPcmtr')
  genNumericFea('pMcPlvtr', 'pMcPlvtr')
  genNumericFea('pMcPsvtr', 'pMcPsvtr')
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

  genIdFea('pPctr_buck')
  genIdFea('pPltr_buck')
  genIdFea('pPwtr_buck')
  genIdFea('pPftr_buck')
  genIdFea('pPptr_buck')
  genIdFea('pPcmtr_buck')
  genIdFea('pPepstr_buck')
  genIdFea('pPcmef_buck')
  genIdFea('pPlvtr_buck')
  genIdFea('pPsvtr_buck')
  genIdFea('pPvtr_buck')
  genIdFea('pMcPctr_buck')
  genIdFea('pMcPltr_buck')
  genIdFea('pMcPwtr_buck')
  genIdFea('pMcPlvtr_buck')
  genIdFea('pMcPsvtr_buck')


def gen_label():
  global loss_name
  loss_name = ''
  genNumericFea('slideWTDPosLabel', 'slide_wtd_label')
  genNumericFea('slideWTDCYPosWeight3', 'slide_wtd_weight')
  genNumericFea('playDurationWeight', 'slide_play_weight')

def gen_next_label():
  global loss_name
  loss_name = ''
  genNumericFea('SlideNextPosLabel', 'slide_next_label')
  genNumericFea('SlideNextPosWeight', 'slide_next_weight')

def gen_user_fea(n) :
  name = 'realshow_' # 按照时间正排，第0个不是最近看的
  for i in range(n) :
    key = name + str(i)
    user_param.update({key: {"attrs": [{"key_type": 397, "converter": "id", "attr": [key], }, ], "dim": 32, "expire_second": 86400 * 7, }})
    key = name + "aid_" + str(i)
    user_param.update({key: {"attrs": [{"key_type": 398, "converter": "id", "attr": [key], }, ], "dim": 32, "expire_second": 86400 * 7, }})
    key = name + "tag_" + str(i)
    user_param.update({key: {"attrs": [{"key_type": 7, "converter": "id", "attr": [key], }, ], "dim": 8, "expire_second": 86400 * 7, }})
    key = name + "play_" + str(i)
    user_param.update({key: {"attrs": [{"key_type": 8, "converter": "id", "attr": [key], }, ], "dim": 8, "expire_second": 86400 * 7, }})

def gen_params() :
  ans = default_param
  gen_user_fea(30)
  ans.update(user_param)
  genPhotoFea('')
  gen_label()
  gen_next_label()
  genSlidePhotoFea('')
  ans.update(photo_param)
  print('user_total_param_num = {}'.format(len(user_param)))
  print('photo_total_param_num = {}'.format(len(photo_param)))
  global global_slot
  print("global_slot = {}".format(global_slot))
  return ans
