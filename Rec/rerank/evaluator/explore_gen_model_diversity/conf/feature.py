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
  "ctr_uId": {"attrs": [{"attr": ["uId"], "key_type": 3, "converter": "id"}], "dim": 32, "expire_second": 86400*30,"use_common_attr_only": True},
  "ctr_dId": {"attrs": [{"attr": ["dId"], "key_type": 4, "converter": "id"}], "dim": 32, "expire_second": 86400*30,"use_common_attr_only": True},
  "l2r_uId": {"attrs": [{"attr": ["uId"], "key_type": 38, "converter": "id"}], "dim": 32, "expire_second": 86400*30,"use_common_attr_only": True},
  "l2r_dId": {"attrs": [{"attr": ["dId"], "key_type": 34, "converter": "id"}], "dim": 32, "expire_second": 86400*30,"use_common_attr_only": True},
  "l2r_uCityId": {"attrs": [{"attr": ["uCityId"], "key_type": 53, "converter": "id"}], "dim": 4,"use_common_attr_only": True},
  "l2r_uProvinceId": {"attrs": [{"attr": ["uRequstProvinceId"], "key_type": 54, "converter": "id"}], "dim": 4,"use_common_attr_only": True},
  "l2r_uIsDouYin": {"attrs": [{"attr": ["uIsDouYin"], "key_type": 55, "converter": "id"}], "dim": 4,"use_common_attr_only": True},
  "l2r_uRequestHour": {"attrs": [{"attr": ["uRequestHour"], "key_type": 56, "converter": "id"}], "dim": 4,"use_common_attr_only": True},
  "l2r_uRequestWeekday": {"attrs": [{"attr": ["uRequestWeekday"], "key_type": 57, "converter": "id"}], "dim": 4,"use_common_attr_only": True},
  #l2r_ 特征值 = std::max(std::min(numerator / (denominator + smooth), max_val), min_value) * buckets
  "l2r_uExpClick": {"attrs": [{"attr": ["uExpClick"], "key_type": 58, "converter": "discrete", "converter_args": "10,1,10000000,1"}], "dim": 8,"use_common_attr_only": True},
  "l2r_uExpLike": {"attrs": [{"attr": ["uExpLike"], "key_type": 59, "converter": "discrete", "converter_args": "1,1,10000000,1"}], "dim": 8,"use_common_attr_only": True},
  "l2r_uExpFollow": {"attrs": [{"attr": ["uExpFollow"], "key_type": 60, "converter": "discrete", "converter_args": "1,1,10000,1"}], "dim": 8,"use_common_attr_only": True},
  "l2r_uExpLongView": {"attrs": [{"attr": ["uExpLongView"], "key_type": 61, "converter": "discrete", "converter_args": "1,1,10000,1"}], "dim": 8,"use_common_attr_only": True},
  "l2r_uExpWatchTime": {"attrs": [{"attr": ["uExpWatchTime"], "key_type": 62, "converter": "discrete", "converter_args": "1000,1,10000000,1"}], "dim": 8,"use_common_attr_only": True},
  #l2r_ 作为author相关
  "l2r_uClickPids": {"attrs": [{"attr": ["uRealtimeClickList"], "key_type": 113, "converter": "list"}], "dim": ID_DIM, "expire_second": 86400*3, "join_limit": LIST_JOIN_LIMIT,"use_common_attr_only": True},
  "l2r_uLikePids": {"attrs": [{"attr": ["uRealtimeLikeList"], "key_type": 114, "converter": "list"}], "dim": ID_DIM, "expire_second": 86400*3, "join_limit": LIST_JOIN_LIMIT,"use_common_attr_only": True},
  "l2r_uFollowPids": {"attrs": [{"attr": ["uRealtimeFollowList"], "key_type": 115, "converter": "list"}], "dim": ID_DIM, "expire_second": 86400*3, "join_limit": LIST_JOIN_LIMIT,"use_common_attr_only": True},
  "l2r_uHatePids": {"attrs": [{"attr": ["uRealtimeNegativeList"], "key_type": 116, "converter": "list"}], "dim": ID_DIM, "expire_second": 86400*3, "join_limit": LIST_JOIN_LIMIT,"use_common_attr_only": True},
  "l2r_uFollowAids": {"attrs": [{"attr": ["uFollowPhotoAuthorList"], "key_type": 117, "converter": "list"}], "dim": ID_DIM, "expire_second": 86400*3, "join_limit": LIST_JOIN_LIMIT,"use_common_attr_only": True},
  "l2r_uLikeAids": {"attrs": [{"attr": ["uLikePhotoAuthorList"], "key_type": 118, "converter": "list"}], "dim": ID_DIM, "expire_second": 86400*3, "join_limit": LIST_JOIN_LIMIT,"use_common_attr_only": True},
  #l2r_ normal photo lists
  "l2r_fountainClickPids": {"attrs": [{"attr": ["featureFountainProfileClickPidList"], "key_type": 63, "converter": "list"}], "dim": 24, "expire_second": 86400*3,"use_common_attr_only": True},
  "l2r_fountainLikePids": {"attrs": [{"attr": ["featureFountainProfileLikePidList"], "key_type": 64, "converter": "list"}], "dim": 24, "expire_second": 86400*3,"use_common_attr_only": True},
  "l2r_fountainFollowAids": {"attrs": [{"attr": ["featureFountainProfileFollowAidList"], "key_type": 65, "converter": "list"}], "dim": 24, "expire_second": 86400*3,"use_common_attr_only": True},
  "l2r_fountainLongviewPids": {"attrs": [{"attr": ["featureFountainProfileLongViewPidList"], "key_type": 66, "converter": "list"}], "dim": 24, "expire_second": 86400*3,"use_common_attr_only": True},
  "l2r_fountainEffviewPids": {"attrs": [{"attr": ["featureFountainProfileEffViewPidList"], "key_type": 67, "converter": "list"}], "dim": 24, "expire_second": 86400*3,"use_common_attr_only": True},

  "l2r_uPlayActionLabel1m": { "dim" : 4, "attrs" : [ { "attr" : [ "uPlayActionLabel1m" ], "key_type" : 701, "converter" : "list" }, ], "default_expire_second": 86400*3, "use_common_attr_only": True},
  "l2r_uPlayActionLabel5m": { "dim" : 4, "attrs" : [ { "attr" : [ "uPlayActionLabel5m" ], "key_type" : 702, "converter" : "list" }, ], "default_expire_second": 86400*3, "use_common_attr_only": True},
  "l2r_uPlayActionLabel10m": { "dim" : 4, "attrs" : [ { "attr" : [ "uPlayActionLabel10m" ], "key_type" : 703, "converter" : "list" }, ], "default_expire_second": 86400*3, "use_common_attr_only": True},
  "l2r_uPlayActionLabel30m": { "dim" : 4, "attrs" : [ { "attr" : [ "uPlayActionLabel30m" ], "key_type" : 704, "converter" : "list" }, ], "default_expire_second": 86400*3, "use_common_attr_only": True},
  "l2r_uPlayActionLabel1h": { "dim" : 4, "attrs" : [ { "attr" : [ "uPlayActionLabel1h" ], "key_type" : 705, "converter" : "list" }, ], "default_expire_second": 86400*3, "use_common_attr_only": True},
  "l2r_uPlayActionLabel2h": { "dim" : 4, "attrs" : [ { "attr" : [ "uPlayActionLabel2h" ], "key_type" : 706, "converter" : "list" }, ], "default_expire_second": 86400*3, "use_common_attr_only": True},
  "l2r_uLongTermHetuLevel1topN": {"attrs": [{"key_type": 540, "attr": ["uLongTermHetuLevel1topN"], "converter": "list"}], "dim": 8, "use_common_attr_only": True},
  # "user_long_term_hetu_level_1_Legal_new": {"attrs": [{"key_type": 801, "attr": ["uLongTermHetuLevel1Legal"], "converter": "list"}], "use_common_attr_only": True},
  "l2r_uLongTermHetuLevel2topN": {"attrs": [{"key_type": 541, "attr": ["uLongTermHetuLevel2topN"], "converter": "list"}], "dim": 8, "use_common_attr_only": True},
  # "user_long_term_hetu_level_2_Legal_new": {"attrs": [{"key_type": 803, "attr": ["uLongTermHetuLevel2Legal"], "converter": "list"}], "use_common_attr_only": True},
  "l2r_uLongTermHetuLevel3topN": {"attrs": [{"key_type": 804, "attr": ["uLongTermHetuLevel3topN"], "converter": "list"}], "dim": 8, "use_common_attr_only": True},
  # "user_long_term_hetu_level_3_Legal_new": {"attrs": [{"key_type": 805, "attr": ["uLongTermHetuLevel3Legal"], "converter": "list"}], "use_common_attr_only": True},

  "l2r_uBasicAge": {"attrs": [{"attr": ["uAge"], "key_type": 68, "converter": "list"}], "dim": 4, "use_common_attr_only": True},
  "l2r_uGender": {"attrs": [{"attr": ["uGender", "uBasicGender"], "key_type": 69, "converter": "list"}], "dim": 4, "use_common_attr_only": True},
  ################ effective view list #####
  "l2r_user_view_list_pids": {"attrs": [{"key_type": 397, "attr": ["uViewPidListV1"], "converter": "list","converter_args": {"enable_filter": False}}], "dim": 32, 
                          "expire_second": 86400*3, "type": 5, "join_limit": view_list_length, "use_common_attr_only": True},
  "l2r_user_view_list_aids": {"attrs": [{"key_type": 398, "attr": ["uViewAidListV1"], "converter": "list","converter_args": {"enable_filter": False}}], "dim": 32, 
                          "expire_second": 86400*3, "type": 5, "join_limit": view_list_length, "use_common_attr_only": True},
  "l2r_user_view_list_ev": {"attrs": [{"key_type": 321, "attr": ["uEffectiveViewLabelListV1"], "converter": "list","converter_args": {"enable_filter": False}}], "dim": 4, 
                          "expire_second": 86400*3, "type": 5, "join_limit": view_list_length, "use_common_attr_only": True},
  "l2r_user_view_list_lv": {"attrs": [{"key_type": 322, "attr": ["uLongViewLabelListV1"], "converter": "list","converter_args": {"enable_filter": False}}], "dim": 4, 
                          "expire_second": 86400*3, "type": 5, "join_limit": view_list_length, "use_common_attr_only": True},
  "l2r_user_view_list_sv": {"attrs": [{"key_type": 323, "attr": ["uShortViewLabelListV1"], "converter": "list","converter_args": {"enable_filter": False}}], "dim": 4, 
                          "expire_second": 86400*3, "type": 5, "join_limit": view_list_length, "use_common_attr_only": True},
  "l2r_user_view_list_hetu1": {"attrs": [{"key_type": 544, "attr": ["uViewHetu1ListV1"], "converter": "list","converter_args": {"enable_filter": False}}],
                          "expire_second": 86400*3, "type": 5, "join_limit": view_list_length, "use_common_attr_only": True},
  "l2r_user_view_list_hetu2": {"attrs": [{"key_type": 545, "attr": ["uViewHetu2ListV1"], "converter": "list","converter_args": {"enable_filter": False}}],
                          "expire_second": 86400*3, "type": 5, "join_limit": view_list_length, "use_common_attr_only": True},
  "l2r_user_view_list_LEN": {"attrs": [{"key_type": 324, "attr": ["uViewPidListV1_LEN"], "converter": "numeric", 
                          "converter_args": {"min": 0, "max": 200, "scale": 1.0, "normalize": False}}], "dim": 1, "use_common_attr_only": True},
}

loss_name = 'ctr_'

def getListFea(name, attr, dim, slot, converter, expire) :
  return {loss_name + name: {"attrs": [{"attr": [attr], "key_type": slot, "converter": converter}], "dim": dim, "expire_second": expire},}

def getDiscreateFea(name, attr, converter_args, dim, slot, expire) :
  return {loss_name + name: {"attrs": [{"key_type": slot, "converter": "discrete", "converter_args": converter_args, "attr": [attr], }, ], "dim": dim, "expire_second": expire},}

def getIdFea(name, attr, dim, slot, expire, converter) :
  return {loss_name + name: {"attrs": [{"attr": [attr], "key_type": slot, "converter": converter}], "dim": dim, "expire_second": expire},}

def getNumericFea(name, attr, slot) :
   return {loss_name + name: {"attrs": [{"attr": [attr], "key_type": slot, "converter": "numeric"}], "dim" : 1, },}

photo_param = {}

def genListFea(attr, name = None, dim = None, slot = None, expire = 3*86400, converter = 'list') :
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
    photo_param.update(getListFea(name + '_idx' + str(i), attr + '_idx' + str(i), dim, slot, converter, expire))

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

def genContextAttr(attr, converter_type = None, converter_args = None, name = None, dim = None, slot = None, expire = None):
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
  if converter_type == "discrete" and converter_args is not None:
    photo_param.update({loss_name + name: {"attrs": [{"key_type": slot, "converter": "discrete", "converter_args": converter_args, "attr": [attr], }, ], "dim": dim, "expire_second": expire},})
  else:
    photo_param.update({loss_name + name: {"attrs": [{"attr": [attr], "key_type": slot, "converter": converter_type}], "dim": dim, "expire_second": expire},})


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

def genNumericFea(attr, name = None, slot = None) :
  global global_slot
  if slot is None:
    slot = global_slot
    global_slot += 1
  if name is None:
    name = attr
  for i in range(listSize) :
    photo_param.update(getNumericFea(name + '_idx' + str(i), attr + '_idx' + str(i), slot))

def genContextFea(loss):
  global loss_name
  loss_name = loss
  genContextAttr("maxPctr_context", converter_type = "discrete", converter_args = "1,0,1,10000,-1", slot = 401)
  genContextAttr("maxPltr_context", converter_type = "discrete", converter_args = "0.5,0,1,5000,-1", slot = 402)
  genContextAttr("maxPwtr_context", converter_type = "discrete", converter_args = "0.2,0,1,2000,-1", slot = 403)
  genContextAttr("maxPftr_context", converter_type = "discrete", converter_args = "0.2,0,1,2000,-1", slot = 404)
  genContextAttr("maxPcmtr_context", converter_type = "discrete", converter_args = "0.2,0,1,2000,-1", slot = 405)
  genContextAttr("maxPcmef_context", converter_type = "discrete", converter_args = "0.2,0,1,2000,-1", slot = 406)
  genContextAttr("maxPptr_context", converter_type = "discrete", converter_args = "0.2,0,1,2000,-1", slot = 407)
  genContextAttr("maxPepstr_context", converter_type = "discrete", converter_args = "0.2,0,1,2000,-1", slot = 408)
  genContextAttr("maxPvtr_context", converter_type = "discrete", converter_args = "1,0.5,1,5000,-1", slot = 409)
  genContextAttr("maxPcltr_context", converter_type = "discrete", converter_args = "0.2,0,1,2000,-1", slot = 410)
  genContextAttr("maxPfetr_context", converter_type = "discrete", converter_args = "1,0,1,10000,-1", slot = 411)
  genContextAttr("maxPfeff_context", converter_type = "discrete", converter_args = "1,0,1,10000,-1", slot = 412)
  genContextAttr("maxPfrscore1_context", converter_type = "discrete", converter_args = "1,0,1,10000,-1", slot = 413)
  genContextAttr("maxPfrscore2_context", converter_type = "discrete", converter_args = "1,0,120,10000,-1", slot = 414)
  genContextAttr("maxPlvtr_context", converter_type = "discrete", converter_args = "1,0,1,10000,-1", slot = 415)
  genContextAttr("maxPsvtr_context", converter_type = "discrete", converter_args = "1,0,1,10000,-1", slot = 416)

  genContextAttr("avgPctr_context", converter_type = "discrete", converter_args = "1,0,1,10000,-1", slot = 401)
  genContextAttr("avgPltr_context", converter_type = "discrete", converter_args = "0.5,0,1,5000,-1", slot = 402)
  genContextAttr("avgPwtr_context", converter_type = "discrete", converter_args = "0.2,0,1,2000,-1", slot = 403)
  genContextAttr("avgPftr_context", converter_type = "discrete", converter_args = "0.2,0,1,2000,-1", slot = 404)
  genContextAttr("avgPcmtr_context", converter_type = "discrete", converter_args = "0.2,0,1,2000,-1", slot = 405)
  genContextAttr("avgPcmef_context", converter_type = "discrete", converter_args = "0.2,0,1,2000,-1", slot = 406)
  genContextAttr("avgPptr_context", converter_type = "discrete", converter_args = "0.2,0,1,2000,-1", slot = 407)
  genContextAttr("avgPepstr_context", converter_type = "discrete", converter_args = "0.2,0,1,2000,-1", slot = 408)
  genContextAttr("avgPvtr_context", converter_type = "discrete", converter_args = "1,0.5,1,5000,-1", slot = 409)
  genContextAttr("avgPcltr_context", converter_type = "discrete", converter_args = "0.2,0,1,2000,-1", slot = 410)
  genContextAttr("avgPfetr_context", converter_type = "discrete", converter_args = "1,0,1,10000,-1", slot = 411)
  genContextAttr("avgPfeff_context", converter_type = "discrete", converter_args = "1,0,1,10000,-1", slot = 412)
  genContextAttr("avgPfrscore1_context", converter_type = "discrete", converter_args = "1,0,1,10000,-1", slot = 413)
  genContextAttr("avgPfrscore2_context", converter_type = "discrete", converter_args = "1,0,120,10000,-1", slot = 414)
  genContextAttr("avgPlvtr_context", converter_type = "discrete", converter_args = "1,0,1,10000,-1", slot = 415)
  genContextAttr("avgPsvtr_context", converter_type = "discrete", converter_args = "1,0,1,10000,-1", slot = 416)

  genContextAttr("avg_duration_context", converter_type = "id", slot = 666)
  genContextAttr("hetu_level_one_count", converter_type = "id")
  genContextAttr("hetu_level_two_count", converter_type = "id")
  genContextAttr("0_9s_duration_photo_count", converter_type = "id")
  genContextAttr("9_15s_duration_photo_count", converter_type = "id")
  genContextAttr("15_20s_duration_photo_count", converter_type = "id")
  genContextAttr("20_58s_duration_photo_count", converter_type = "id")
  genContextAttr("gt_58s_duration_photo_count", converter_type = "id")

def genPhotoFea(loss) :
  # "denominator,smooth,max_val,buckets,min_val,expr"
  # 特征值 = std::max(std::min(numerator / (denominator + smooth), max_val), min_value) * buckets
  global loss_name
  loss_name = loss
  genDiscreateFea('pPctr', "1,0,1,10000,-1", slot = 401)
  genDiscreateFea('pPltr', "0.5,0,1,5000,-1", slot = 402)
  genDiscreateFea('pPptr', "0.2,0,1,2000,-1", slot = 407)
  genDiscreateFea('pPwtr', "0.2,0,1,2000,-1", slot = 403)
  genDiscreateFea('pPcmtr', "0.2,0,1,2000,-1", slot = 405)
  genDiscreateFea('pPftr', "0.2,0,1,2000,-1", slot = 404)
  genDiscreateFea('pPvtr', "1,0.5,1,5000,-1", slot = 409)
  genDiscreateFea('pPcmef', "0.2,0,1,2000,-1", slot = 406)
  genDiscreateFea('pPcltr', "0.2,0,1,2000,-1", slot = 410)
  genDiscreateFea('pPepstr', "0.2,0,1,2000,-1", slot = 408)
  genDiscreateFea('pPlvtr', "1,0,1,10000,-1", slot = 415)
  genDiscreateFea('pPsvtr', "1,0,1,10000,-1", slot = 416)
  genDiscreateFea('pPfrScore1', "1,0,1,10000,-1", slot = 413)
  genDiscreateFea('pPfrScore2', "1,0,120,10000,-1", slot = 414)
  genDiscreateFea('pPfetr', "1,0,1,10000,-1", slot = 411)
  genDiscreateFea('pPfountainEff', "1,0,1,10000,-1", slot = 412)
  genDiscreateFea('pPhtr', "0.1,0,1,1000,-1")

  genDiscreateFea('pMcPctr', "1,0,1,10000,-1")
  genDiscreateFea('pMcPltr', "0.2,0,1,2000,-1")
  genDiscreateFea('pMcPptr', "0.2,0,1,2000,-1")
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

  genIdFea('pId', slot = 397, dim = 32,  expire = 3 * 24 * 3600)
  genIdFea('aId', slot = 398, dim = 32,  expire = 30 * 24 * 3600)
  genIdFea('pHotExptag')
  genIdFea('pHotLiving')
  genIdFea('pDurationMs', slot = 666)
  genIdFea('pUploadType')
  genIdFea('pCityId')
  genIdFea('pProvinceId')
  genIdFea('pTag')
  genIdFea('pContentLevel')
  genIdFea('pHetuTagLevel1Id', converter = "list", slot = 399, dim = 8)
  genIdFea('pHetuTagLevel2Id', converter = "list", slot = 400, dim = 8)
  genIdFea('pHetuTagLevel3Id', converter = "list")
  genIdFea('pHetuTagLevel5Id', converter = "list")
  genIdFea('pHetuTagId', converter = "list")
  genIdFea('pMusic')
  genIdFea('pAuthorGender')
  genIdFea('pAgeHour')
  genIdFea('pMmuImgClusterV1')
  genIdFea('pMmuImgClusterV3')
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

  

  # genListFea('pShortStatShowHetu1100n', dim=8)
  # genListFea('pShortStatShowHetu11000n', dim=8)
  # genListFea('pShortStatShowHetu2100n', dim=8)
  # genListFea('pShortStatShowHetu21000n', dim=8)
  # genListFea('pShortStatShowHetu3100n', dim=8)
  # genListFea('pShortStatShowHetu31000n', dim=8)
  # genListFea('pShortStatShowHetu4100n', dim=8)
  # genListFea('pShortStatShowHetu41000n', dim=8)
  # genListFea('pShortStatShowHetu5100n', dim=8)
  # genListFea('pShortStatShowHetu51000n', dim=8)
  # genListFea('pShortStatShowHetuTag100n', dim=8)
  # genListFea('pShortStatShowHetuTag1000n', dim=8)

  # genListFea('pShortStatClickHetu1100n', dim=8)
  # genListFea('pShortStatClickHetu11000n', dim=8)
  # genListFea('pShortStatClickHetu2100n', dim=8)
  # genListFea('pShortStatClickHetu21000n', dim=8)
  # genListFea('pShortStatClickHetu3100n', dim=8)
  # genListFea('pShortStatClickHetu31000n', dim=8)
  # genListFea('pShortStatClickHetu4100n', dim=8)
  # genListFea('pShortStatClickHetu41000n', dim=8)
  # genListFea('pShortStatClickHetu5100n', dim=8)
  # genListFea('pShortStatClickHetu51000n', dim=8)
  # genListFea('pShortStatClickHetuTag100n', dim=8)
  # genListFea('pShortStatClickHetuTag1000n', dim=8)

def gen_label():
  global loss_name
  loss_name = ''
  genNumericFea('ValidClickPosLabel', 'l2r_label')
  genNumericFea('pValidClickWeightV3', 'l2r_weight')

  genContextAttr('ValidListPosLabel', name = 'list_label', converter_type = 'numeric', dim = 1)
  genContextAttr('pValidListPosWeightV3', name = 'list_weight', converter_type = 'numeric', dim = 1)

def gen_user_fea(n) :
  name = 'realshow_'
  for i in range(n) :
    key = name + str(i)
    user_param.update({key: {"attrs": [{"key_type": 397, "converter": "id", "attr": [key], }, ], "dim": 32, "expire_second": 86400 * 7, "use_common_attr_only" : True}})
    key = name + "aid_" + str(i)
    user_param.update({key: {"attrs": [{"key_type": 398, "converter": "id", "attr": [key], }, ], "dim": 32, "expire_second": 86400 * 7, "use_common_attr_only" : True}})
    # key = name + "tag_" + str(i)
    # user_param.update({key: {"attrs": [{"key_type": 7, "converter": "id", "attr": [key], }, ], "dim": 8, "expire_second": 86400 * 7, "use_common_attr_only" : True}})
    # key = name + "play_" + str(i)
    # user_param.update({key: {"attrs": [{"key_type": 8, "converter": "id", "attr": [key], }, ], "dim": 8, "expire_second": 86400 * 7, "use_common_attr_only" : True}})

def gen_params() :
  ans = default_param
  # gen_user_fea(30)
  ans.update(user_param)
  for loss in ['l2r'] :
    genPhotoFea(loss + '_')
  for loss in ['l2r'] :
    genContextFea(loss + '_')
  gen_label()
  ans.update(photo_param)
  print('user_total_param_num = {}'.format(len(user_param)))
  print('photo_total_param_num = {}'.format(len(photo_param)))
  global global_slot
  print("global_slot = {}".format(global_slot))
  return ans                                                                                                                       