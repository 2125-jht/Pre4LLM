#!/bin/python
# -*- coding: UTF8 -*-
from __future__ import print_function
import os
import json
import sys

ID_DIM = 32
CATE_DIM = 16
LIST_JOIN_LIMIT = 32
global_slot = 500 # 共享特征，以用户侧为主0-100，文章侧特征100-200  500以上是自动生成keytype的，其中文章侧声明中的slot就是keytype
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
    # "default_expire_second": -1,
    "default_expire_second": 3600 * 24 * 30,
    "default_decay_rate": 0.9999,
    "default_initial_lr": 0.05,
    "default_initial_g2sum": 3,
}

user_param = {
  "uId": {"attrs": [{"attr": ["uId"], "key_type": 1, "converter": "id"}], "dim": 64},
  "dId": {"attrs": [{"attr": ["dId"], "key_type": 2, "converter": "id"}], "dim": 64},
  "uCityId": {"attrs": [{"attr": ["uCityId"], "key_type": 3, "converter": "id"}], "dim": 4},
  "uProvinceId": {"attrs": [{"attr": ["uRequstProvinceId"], "key_type": 4, "converter": "id"}], "dim": 4},
  "uIsDouYin": {"attrs": [{"attr": ["uIsDouYin"], "key_type": 5, "converter": "id"}], "dim": 4},
  "uRequestHour": {"attrs": [{"attr": ["uRequestHour"], "key_type": 6, "converter": "id"}], "dim": 4},
  "uRequestWeekday": {"attrs": [{"attr": ["uRequestWeekday"], "key_type": 7, "converter": "id"}], "dim": 4},
  "uBasicAge": {"attrs": [{"attr": ["uBasicAge"], "key_type": 8, "converter": "id"}], "dim": 4, },
  "uGender": {"attrs": [{"attr": ["uBasicGender"], "key_type": 9, "converter": "id"}], "dim": 4, },

   # 除了内流以外的实时特征，real系列 
  "uRealClickPids": {"attrs": [{"attr": ["uRealtimeClickList"], "key_type": 10, "converter": "list"}], "dim": ID_DIM, "expire_second": 604800},
  "uRealLikePids": {"attrs": [{"attr": ["uRealtimeLikeList"], "key_type": 11, "converter": "list"}], "dim": ID_DIM, "expire_second": 604800},
  "uRealFollowPids": {"attrs": [{"attr": ["uRealtimeFollowList"], "key_type": 12, "converter": "list"}], "dim": ID_DIM, "expire_second": 604800},
  "uRealForwardPids": {"attrs": [{"attr": ["uRealtimeForwardList"], "key_type": 13, "converter": "list"}], "dim": ID_DIM, "expire_second": 604800},
  # 虽然是取自getRealtimeNegativeListList，输入模型最多50个，累积了月约100个
  "uRealHatePids": {"attrs": [{"attr": ["uRealtimeNegativeList"], "key_type": 14, "converter": "list"}], "dim": ID_DIM, "expire_second": 604800},
  "uRealFollowAids": {"attrs": [{"attr": ["uFollowPhotoAuthorList"], "key_type": 15, "converter": "list"}], "dim": ID_DIM, "expire_second": 604800},
  "uRealLikeAids": {"attrs": [{"attr": ["uLikePhotoAuthorList"], "key_type": 16, "converter": "list"}], "dim": ID_DIM, "expire_second": 604800},

   #l2r_ 特征值 = std::max(std::min(numerator / (denominator + smooth), max_val), min_value) * buckets

  # 没有数据了
  "uExpClick": {"attrs": [{"attr": ["uExpClick"], "key_type": 20, "converter": "discrete", "converter_args": "10,1,10000000,1"}], "dim": 8},
  "uExpLike": {"attrs": [{"attr": ["uExpLike"], "key_type": 21, "converter": "discrete", "converter_args": "1,1,10000000,1"}], "dim": 8},
  "uExpFollow": {"attrs": [{"attr": ["uExpFollow"], "key_type": 22, "converter": "discrete", "converter_args": "1,1,10000,1"}], "dim": 8},
  "uExpLongView": {"attrs": [{"attr": ["uExpLongView"], "key_type": 23, "converter": "discrete", "converter_args": "1,1,10000,1"}], "dim": 8},
  "uExpWatchTime": {"attrs": [{"attr": ["uExpWatchTime"], "key_type": 24, "converter": "discrete", "converter_args": "1000,1,10000000,1"}], "dim": 8},
  
  # userInfo.getUserProfileV1()中的VideoPlayingStatList()是包含时长相关信息的list   pid list对应的是label list，模型处理的时候注意一下
  # user_view_list_ev，user_view_list_lv，user_view_list_sv这三个是list的值只有0或是1，处理的时候注意一下
  "uMidPlayViewPids": {"attrs": [{"key_type": 30, "attr": ["uViewPidListV1"], "converter": "list","converter_args": {"enable_filter": False}}], "dim": 32, 
                          "expire_second": 86400*3, "type": 5, "join_limit": view_list_length},
  "uMidPlayViewAids": {"attrs": [{"key_type": 31, "attr": ["uViewAidListV1"], "converter": "list","converter_args": {"enable_filter": False}}], "dim": 32, 
                          "expire_second": 86400*3, "type": 5, "join_limit": view_list_length},
  "uMidPlayViewEvLabels": {"attrs": [{"key_type": 32, "attr": ["uEffectiveViewLabelListV1"], "converter": "list","converter_args": {"enable_filter": False}}], "dim": 4, 
                          "expire_second": 86400*3, "type": 5, "join_limit": view_list_length},
  "uMidPlayViewLvLabels": {"attrs": [{"key_type": 33, "attr": ["uLongViewLabelListV1"], "converter": "list","converter_args": {"enable_filter": False}}], "dim": 4, 
                          "expire_second": 86400*3, "type": 5, "join_limit": view_list_length},
  "uMidPlayViewSvLabels": {"attrs": [{"key_type": 34, "attr": ["uShortViewLabelListV1"], "converter": "list","converter_args": {"enable_filter": False}}], "dim": 4, 
                          "expire_second": 86400*3, "type": 5, "join_limit": view_list_length},
  "uMidPlayViewHetu1": {"attrs": [{"key_type": 35, "attr": ["uViewHetu1ListV1"], "converter": "list","converter_args": {"enable_filter": False}}],
                          "expire_second": 86400*3, "type": 5, "join_limit": view_list_length},
  "uMidPlayViewHetu2": {"attrs": [{"key_type": 36, "attr": ["uViewHetu2ListV1"], "converter": "list","converter_args": {"enable_filter": False}}],
                          "expire_second": 86400*3, "type": 5, "join_limit": view_list_length},
  "uMidPlayEffectivePids": {"attrs": [{"attr": ["uViewEffectivePidListV1"], "key_type": 37, "converter": "list"}], "dim": 32, "expire_second": 604800},
  "uMidPlayLongPids": {"attrs": [{"attr": ["uViewLongPidListV1"], "key_type": 38, "converter": "list"}], "dim": 32, "expire_second": 604800},
  "uMidPlayShortPids": {"attrs": [{"attr": ["uViewShortPidListV1"], "key_type": 39, "converter": "list"}], "dim": 32, "expire_second": 604800},

    # userInfo.getUserProfileV1()中的互动list是200个，包含实时特征，除了内流以外短期兴趣
  "uMidClickPids": {"attrs": [{"attr": ["uClickPhotoList"], "key_type": 50, "converter": "list"}], "dim": ID_DIM, "expire_second": 604800},
  "uMidLikePids": {"attrs": [{"attr": ["uLikePhotoList"], "key_type": 51, "converter": "list"}], "dim": ID_DIM, "expire_second": 604800},
  "uMidFollowPids": {"attrs": [{"attr": ["uFollowPhotoList"], "key_type": 52, "converter": "list"}], "dim": ID_DIM, "expire_second": 604800},
  "uMidCommentPids": {"attrs": [{"attr": ["uCommentPhotoList"], "key_type": 53, "converter": "list"}], "dim": ID_DIM, "expire_second": 604800},
  "uMidClickAids": {"attrs": [{"attr": ["uClickPhotoAuthorList"], "key_type": 54, "converter": "list"}], "dim": ID_DIM, "expire_second": 604800},
  "uMidLikeAids": {"attrs": [{"attr": ["uLikePhotoAuthorList"], "key_type": 55, "converter": "list"}], "dim": ID_DIM, "expire_second": 604800},
  "uMidFollowAids": {"attrs": [{"attr": ["uFollowPhotoAuthorList"], "key_type": 56, "converter": "list"}], "dim": ID_DIM, "expire_second": 604800},
  "uMidCommentAids": {"attrs": [{"attr": ["uCommentPhotoAuthorList"], "key_type": 57, "converter": "list"}], "dim": ID_DIM, "expire_second": 604800},
  


  # 内流list特征 userInfo.getFountainRecoUserProfile() 但是userInfo.getUserProfileV1()和real中都没有内流的数据
  "fountainClickPids": {"attrs": [{"attr": ["featureFountainProfileClikPidList"], "key_type": 60, "converter": "list"}], "dim": 32, "expire_second": 604800},
  "fountainLikePids": {"attrs": [{"attr": ["featureFountainProfileLikePidList"], "key_type": 61, "converter": "list"}], "dim": 32, "expire_second": 604800},
  "fountainFollowPids": {"attrs": [{"attr": ["featureFountainProfileFollowPidList"], "key_type": 62, "converter": "list"}], "dim": 32, "expire_second": 604800},
  "fountainForwardPids": {"attrs": [{"attr": ["featureFountainProfileForwardPidList"], "key_type": 63, "converter": "list"}], "dim": 32, "expire_second": 604800},
  "fountainLongviewPids": {"attrs": [{"attr": ["featureFountainProfileLongViewPidList"], "key_type": 64, "converter": "list"}], "dim": 32, "expire_second": 604800},
  "fountainEffviewPids": {"attrs": [{"attr": ["featureFountainProfileEffViewPidList"], "key_type": 65, "converter": "list"}], "dim": 32, "expire_second": 604800},

  # 上下文特征
  "page": {"attrs": [{"attr": ["currentPage"], "key_type": 70, "converter": "numeric"}], "dim": 1, },
  "pListSlideNegLabel_1": {"attrs": [{"attr": ["pListSlideNegLabel_1"], "key_type": 71, "converter": "numeric"}], "dim": 1, },
  "pListSlideNegLabel_2": {"attrs": [{"attr": ["pListSlideNegLabel_2"], "key_type": 72, "converter": "numeric"}], "dim": 1, },
  "pListSlideNegLabel_3": {"attrs": [{"attr": ["pListSlideNegLabel_3"], "key_type": 73, "converter": "numeric"}], "dim": 1, },
  "pListSlideNegLabel_4": {"attrs": [{"attr": ["pListSlideNegLabel_4"], "key_type": 74, "converter": "numeric"}], "dim": 1, },
  "pListSlideNegLabel_5": {"attrs": [{"attr": ["pListSlideNegLabel_5"], "key_type": 75, "converter": "numeric"}], "dim": 1, },
  "pListSlideNegLabel_6": {"attrs": [{"attr": ["pListSlideNegLabel_6"], "key_type": 76, "converter": "numeric"}], "dim": 1, },

}


def getDiscreateFea(name, attr, converter_args, dim, slot, expire) :
  return {loss_name + name: {"attrs": [{"key_type": slot, "converter": "discrete", "converter_args": converter_args, "attr": [attr], }, ], "dim": dim, "expire_second": expire},}

def getIdFea(name, attr, dim, slot, expire, converter) :
  return {loss_name + name: {"attrs": [{"attr": [attr], "key_type": slot, "converter": converter}], "dim": dim, "expire_second": expire},}

def getNumericFea(name, attr, slot) :
   return {loss_name + name: {"attrs": [{"attr": [attr], "key_type": slot, "converter": "numeric"}], "dim" : 1, },}

def getNumericListFea(name, attr, slot) :
   return {loss_name + name: {"attrs": [{"attr": [attr], "key_type": slot, "converter": "numeric"}], "dim" : 19, },}

photo_param = {}

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

def genNumericListFea(attr, name = None, slot = None) :
  global global_slot
  if slot is None:
    slot = global_slot
    global_slot += 1
  if name is None:
    name = attr
  for i in range(listSize) :
    photo_param.update(getNumericListFea(name + '_idx' + str(i), attr + '_idx' + str(i), slot))
  photo_param.update(getNumericListFea(name, attr, slot))

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
  genDiscreateFea('pPfrScore2', "1,0,180,2000")
  genDiscreateFea('pPwtd', "1,0,180,2000")
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

  genIdFea('pId', slot = 30, dim = 32,  expire = 7 * 24 * 3600)
  genIdFea('aId', slot = 31, dim = 32,  expire = 30 * 24 * 3600)
  genIdFea('position', slot = 200, dim = 64)
  genIdFea('pHotExptag')
  genIdFea('pHotLiving')
  genIdFea('pDurationMs', slot = 201, dim = 4)
  genIdFea('pUploadType')
  genIdFea('pCityId')
  genIdFea('pProvinceId')
  genIdFea('pTag')
  genIdFea('pContentLevel', dim = 2)
  genIdFea('pHetuTagLevel1Id', slot = 35, dim = 8, converter = "list")
  genIdFea('pHetuTagLevel2Id', slot = 36, dim = 8, converter = "list")
  genIdFea('pMusic')
  genIdFea('pAuthorGender')
  genIdFea('pAgeHour')
  genIdFea('pMmuImgClusterV1')
  genIdFea('pMmuImgClusterV3', slot = 202, dim = 4)
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
  genNumericFea('slideWTDPosLabel', 'slide_wtd_label')
  genNumericFea('slideWTDPosWeight', 'slide_neg_weight')
  genNumericFea('playDurationWeight', 'slide_play_weight')
  genNumericFea('playByDurationRate', 'slide_play_rate')

def gen_next_label():
  global loss_name
  loss_name = ''
  genNumericFea('SlideNextPosLabel', 'slide_next_label')
  genNumericFea('SlideNextPosWeight', 'slide_next_weight')

def gen_user_fea(n) :
  name = 'realshow_'
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
  # gen_user_fea(30)
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
