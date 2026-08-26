import json
import sys


# 本次模型使用的所有特征，需要从feature pool抽取
# dim-特征维度，需要用户自定义
# share_id-需要share embedding的特征设置为相同的share id，具体值用户自定义
# expand-list特征是否需要展开，如果自定义list长度则需要
default_dim = 16
all_feats = {
  "user_id": {"dim": 64, "belong": "user"},
  "user_device_id": {"dim": 32, "belong": "user"},
  "user_age_segment": {"dim": 4, "belong": "user"},
  "user_gender": {"dim": 4, "belong": "user"},
  "user_visit_mod": {"belong": "user"},
  "user_city_id": {"belong": "user"},
  "user_client_id": {"belong": "user"},
  "user_level": {"belong": "user"},
  "user_emp_ctr": {"belong": "user"},
  "user_emp_ltr": {"belong": "user"},
  "user_emp_wtr": {"belong": "user"},
  "user_emp_lvtr": {"belong": "user"},
  "user_request_city_id": {"belong": "user"},
  "user_request_poi_type": {"belong": "user"},
  "user_region_type": {"belong": "user"},

  # rt seq feature
  "user_realtime_click_list": {"belong": "user_seq", "dim": 64, "expand":50},
  "user_realtime_like_list": {"belong": "user_seq", "dim": 64, "expand":50},
  # "user_realtime_follow_list": {"belong": "user_seq", "dim": 64, "expand":50},
  # explore seq feature
  "user_profile_v1_click_pid_list": {"belong": "user_seq", "dim": 64, "expand":200},
  "user_profile_v1_click_aid_list": {"belong": "user_seq", "dim": 64, "expand":200},
  "user_profile_v1_play18s_pid_list": {"belong": "user_seq", "dim": 64, "expand":200},
  "user_profile_v1_play18s_aid_list": {"belong": "user_seq", "dim": 64, "expand":200},
  # fountain seq feature
  # "user_fountain_profile_click_pid_list": {"belong": "user_seq", "dim": 64, "expand":200}, # 播放大于 3s
  # "user_fountain_profile_click_aid_list": {"belong": "user_seq", "dim": 64, "expand":200},
  "user_fountain_profile_like_pid_list": {"belong": "user_seq", "dim": 64, "expand":200},
  "user_fountain_profile_like_aid_list": {"belong": "user_seq", "dim": 64, "expand":200},
  # "user_fountain_profile_follow_pid_list": {"belong": "user_seq", "dim": 64, "expand":200},
  # "user_fountain_profile_follow_aid_list": {"belong": "user_seq", "dim": 64, "expand":200},
  # "user_fountain_profile_comment_pid_list": {"belong": "user_seq", "dim": 64, "expand":200},
  # "user_fountain_profile_comment_aid_list": {"belong": "user_seq", "dim": 64, "expand":200},
  # duration < 7 播放时长 >= 7; duration in [7, 18) 播放时长 >= duration; duration >= 18 播放时长 >= 18
  "user_fountain_profile_effective_view_pid_list": {"belong": "user_seq", "dim": 64, "expand":200},
  "user_fountain_profile_effective_view_aid_list": {"belong": "user_seq", "dim": 64, "expand":200},
  # duration < 18 播放时长 >= 18; duration in [18, 36) 播放时长 >= duration; duration >= 36 播放时长 >= 36
  "user_fountain_profile_long_view_pid_list": {"belong": "user_seq", "dim": 64, "expand":200},
  "user_fountain_profile_long_view_aid_list": {"belong": "user_seq", "dim": 64, "expand":200},
  # "user_fountain_profile_short_view_pid_list": {"expand":50},
  # "user_fountain_profile_short_view_aid_list": {"expand":50},

  # "user_profile_v1_play3s_pid_list": {"expand":50, "share_id": 2},
  # "user_profile_v1_play3s_aid_list": {"expand":50, "share_id": 3},
  # "user_profile_v1_play7s_pid_list": {"expand":50, "share_id": 2},
  # "user_profile_v1_play7s_aid_list": {"expand":50, "share_id": 3},
  # "user_profile_v1_play11s_pid_list": {"expand":50, "share_id": 2},
  # "user_profile_v1_play11s_aid_list": {"expand":50, "share_id": 3},
  # "user_profile_v1_play18s_pid_list": {"expand":50, "share_id": 2},
  # "user_profile_v1_play18s_aid_list": {"expand":50, "share_id": 3},

  "photo_id": {"dim": 64, "belong": "item"},
  "photo_author_id": {"dim": 64, "belong": "item"},
  "photo_author_gender": {"dim": 4, "belong": "item"},
  "photo_upload_type": {"dim": 4, "belong": "item"},
  "photo_hetu_tag_level1_list": {"belong": "item"},
  "photo_hetu_tag_level2_list": {"belong": "item"},
  "photo_hetu_tag_level3_list": {"belong": "item"},
  "photo_hetu_tag_level5_list": {"belong": "item"},
  "photo_tag": {"dim": 24, "belong": "item"},
  "photo_duration_ms": {"belong": "item"},
  "photo_city_id": {"belong": "item"},
  "photo_music": {"belong": "item"},

  "context_pctr": {"belong": "item"},
  "context_pltr": {"belong": "item"},
  "context_pwtr": {"belong": "item"}, # 关注
  "context_pftr": {"belong": "item"}, # 分享
  "context_plvtr": {"belong": "item"},
  "context_pvtr": {"belong": "item"},
  "context_pptr": {"belong": "item"},
  "context_pcmtr": {"belong": "item"},
  "context_pepstr": {"belong": "item"},
  "context_pcpr": {"belong": "item"},
  "context_pcltr": {"belong": "item"},
  "context_psvr": {"belong": "item"},
  "context_pwtd": {"belong": "item"},
  # "context_fullrank_ltr_score": {"belong": "item"},
  # "context_fullrank_act_wtd": {"belong": "item"},
  # "context_fullrank_ltr_v4_fountain_next": {"belong": "item"},
  "context_fountain_related_score_v2": {"belong": "item"},

  # "context_cascade_pctr": {"belong": "item"},
  # "context_cascade_pltr": {"belong": "item"},
  # "context_cascade_pwtr": {"belong": "item"},
  # "context_cascade_plvtr": {"belong": "item"},

  "context_source_pid": {"dim": 64, "belong": "source"},
  "context_source_aid": {"dim": 32, "belong": "source"},
  "context_source_hetu_tag_level1_top1": {"belong": "source"},
  "context_source_hetu_tag_level2_top1": {"belong": "source"},
  "context_source_tag": {"dim": 24, "belong": "source"},
  "context_source_duration_ms": {"belong": "source"},
}
copy_feats = {
  "photo_id":["photo_id_v2", 4000]
}

##加载特征池
feature_pool_config = json.load(open("./feature_pool.json", "r"))

user_fea_names, photo_fea_names, explore_profile_fea_names, fountain_seq_pid_names, fountain_seq_aid_names, source_fea_names = \
[],[],[],[],[],[]
class Attr:
    def __init__(self, attr_name, slot, is_common, dim, expand, belong):
        # model feature name
        self.attr_name = attr_name
        # feature dim
        self.dim = dim
        self.slots = slot
        self.expand = expand
        self.is_common = is_common
        self.belong = belong

def get_all_feature_attrs(all_feats):
    all_features = []
    all_share_id = {}
    share_input_slots = []
    share_output_slots = []
    copy_input_slots = []
    copy_output_slots = []
    for k in all_feats.keys():
        if k in feature_pool_config.keys():
            # get all attr
            slot = feature_pool_config[k].get("slot")
            is_common = feature_pool_config[k].get("use_common_attr_only", False)
            dim = all_feats[k].get("dim", default_dim)
            expand = all_feats[k].get("expand", None)
            belong = all_feats[k].get("belong", None)
            if belong is None:
              print (k, "attr belong is NONE, please check!!!")
              sys.exit()
            elif belong == "user":
              user_fea_names.append(k)
            elif belong == "item":
              photo_fea_names.append(k)
            # elif belong == "user_rt_pid_seq":
            #   explore_profile_fea_names.append(k)
            elif belong == "source":
              source_fea_names.append(k)
            #expand = None
            all_features.append(Attr(k, [slot], is_common, dim, expand, belong))
            # share id
            share_id = all_feats[k].get("share_id", None)
            if share_id is not None:
                if share_id in all_share_id:
                    all_share_id[share_id].append(slot)
                else:
                    all_share_id[share_id] = [slot]
        else:
            print (k, "feature is not in featue pool, please check!!!")

    ### share embedding, share到最小slot
    for k, v in all_share_id.items():
        v.sort()
        share_input_slots += v[1:]
        share_output_slots += [v[0]] * (len(v) - 1)

    return all_features, share_input_slots, share_output_slots, copy_input_slots, copy_output_slots

all_features, share_input_slots, share_output_slots, copy_input_slots, copy_output_slots = get_all_feature_attrs(all_feats)

print(f"user_fea_names: {user_fea_names}")
print(f"photo_fea_names: {photo_fea_names}")
print(f"explore_profile_fea_names: {explore_profile_fea_names}")
print(f"fountain_seq_pid_names: {fountain_seq_pid_names}")
print(f"fountain_seq_aid_names: {fountain_seq_aid_names}")
print(f"source_fea_names: {source_fea_names}")

dense_features_config = {
  "photo_info__explore_stat__real_show_count_list": {"norm_type": "x^0.7", "name": "emp_explore_show_count", "boundaries": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,21,23,25,28,31,34,38,43,48,54,62,71,81,94,108,124,142,162,185,210,237,267,299,333,372,417,471,535,613,704,807,921,1055,1224,1447,1747,2170,2805,3886,6319,21502]},
  "photo_info__explore_stat__click_count_list": {"norm_type": "x^0.7", "name": "emp_explore_click_count", "boundaries": [1,2,3,4,5,6,7,8,9,10,11,13,15,18,21,24,28,32,36,41,47,53,60,69,78,88,101,115,133,153,179,212,254,309,386,502,697,1137,3831]},
  "photo_info__fountain_stats__real_show_count_list": {"norm_type": "x^0.7", "name": "emp_fountain_show_count", "boundaries": [5,10,15,21,28,39,47,58,73,95,127,171,231,313,424,568,753,989,1287,1661,2133,2732,3498,4484,5779,7509,9849,13080,17669,24386,34643,51099,79871,138226,299160,1662878]},
  "photo_info__fountain_stats__like_count_list": {"name": "emp_fountain_like_count", "boundaries": [1,2,3,4,5,6,7,8,9,10,12,14,17,20,24,30,37,46,58,73,95,124,164,222,310,448,689,1178,2586,17782]},
  "photo_info__fountain_stats__follow_count_list": {"name": "emp_fountain_follow_count", "boundaries": [1,2,3,4,5,6,9,11,15,21,30,45,74,150,879]},
  "photo_info__fountain_stats__long_play_count_list": {"name": "emp_fountain_long_play_count", "boundaries": [1,2,3,4,5,6,7,9,13,18,25,35,50,71,100,140,192,262,353,474,633,846,1134,1526,2069,2837,3949,5612,8194,12436,19999,35498,78716,460612]},
  "photo_emp_explore_ctr": {"name": "photo_emp_explore_ctr", "boundaries": [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.15,0.18,0.26]},
  "photo_emp_explore_ltr": {"name": "photo_emp_explore_ltr", "boundaries": [0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.18,0.2,0.24]},
  "photo_emp_explore_avg_time": {"name": "photo_emp_explore_avg_time", "boundaries": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,27,29,31,34,36,40,44,50,57,66,79,98,133,267]},
  "photo_emp_fountain_svtr": {"name": "photo_emp_fountain_svtr", "boundaries": [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.28,0.29,0.31,0.33,0.36,0.4,0.52]},
  "photo_emp_fountain_lvtr": {"name": "photo_emp_fountain_lvtr", "boundaries": [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.15,0.16,0.17,0.19,0.2,0.21,0.23,0.24,0.26,0.28,0.3,0.32,0.35,0.39,0.44,0.6]},
  "photo_emp_fountain_ltr": {"name": "photo_emp_fountain_ltr", "boundaries": [0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.21,0.24,0.28]},
  "photo_emp_fountain_wtr": {"name": "photo_emp_fountain_wtr", "boundaries": [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.011,0.012,0.013,0.014,0.015,0.016,0.017,0.018,0.019,0.02,0.025,0.03,0.035,0.04,0.045,0.05]},
  "photo_emp_fountain_avg_time": {"name": "photo_emp_fountain_avg_time", "boundaries": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,21,24,27,31,36,46,85]},
  "photo_emp_fountain_avg_fintr": {"name": "photo_emp_fountain_avg_fintr", "boundaries": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]},
}

wtd_config = \
{
  "buckets":[0,9,13,20,38,71,118,195],
  "configs" : [
    [8, 12, 16, 19, 25, 29, 33, 39, 45, 52, 58, 65, 71, 77, 84, 92, 101, 111, 122, 135, 153, 173, 196, 223, 256, 297, 349, 420, 527, 730, 100000000],
    [9, 11, 13, 15, 17, 20, 23, 27, 32, 36, 39, 43, 47, 52, 56, 60, 64, 68, 72, 75, 78, 81, 84, 87, 90, 93, 97, 101, 104, 109, 113, 118, 123, 128, 132, 136, 140, 145, 149, 153, 158, 163, 169, 175, 181, 189, 198, 209, 221, 234, 249, 267, 291, 321, 340, 363, 392, 430, 483, 570, 757, 100000000],
    [9, 11, 13, 17, 21, 27, 31, 36, 40, 44, 49, 55, 61, 67, 73, 80, 87, 93, 98, 102, 106, 109, 113, 116, 120, 124, 128, 132, 135, 138, 142, 146, 151, 156, 161, 167, 173, 179, 186, 192, 199, 206, 213, 221, 230, 240, 251, 264, 283, 300, 315, 337, 362, 394, 443, 519, 686, 100000000],
    [9, 11, 12, 14, 16, 19, 22, 26, 30, 35, 39, 44, 49, 55, 62, 70, 76, 82, 88, 95, 101, 109, 116, 123, 130, 136, 141, 147, 152, 157, 162, 167, 172, 178, 184, 191, 197, 204, 212, 219, 225, 232, 239, 247, 256, 265, 275, 284, 295, 306, 317, 331, 346, 365, 379, 395, 415, 443, 481, 536, 581, 645, 759, 1006, 100000000],
    [9, 11, 13, 16, 21, 26, 31, 35, 39, 44, 50, 57, 65, 75, 86, 95, 104, 113, 124, 135, 147, 161, 174, 187, 199, 210, 218, 225, 232, 238, 245, 251, 258, 265, 272, 279, 285, 292, 299, 305, 312, 319, 326, 334, 342, 349, 357, 366, 373, 380, 386, 393, 401, 413, 429, 448, 472, 494, 521, 555, 597, 630, 675, 719, 777, 825, 897, 1002, 1167, 1556, 100000000],
    [9, 11, 14, 17, 22, 28, 33, 37, 42, 48, 54, 62, 71, 83, 96, 108, 120, 132, 146, 162, 178, 196, 215, 234, 256, 278, 301, 325, 350, 373, 393, 407, 419, 430, 442, 453, 465, 477, 489, 502, 515, 528, 542, 555, 569, 580, 590, 600, 610, 619, 629, 638, 648, 658, 670, 681, 694, 707, 717, 730, 755, 801, 873, 996, 1241, 100000000],
    [9, 13, 16, 21, 28, 34, 42, 47, 54, 62, 71, 82, 96, 112, 131, 147, 163, 181, 200, 220, 242, 266, 285, 302, 321, 339, 358, 378, 398, 418, 439, 461, 483, 506, 529, 553, 578, 603, 629, 655, 681, 705, 725, 744, 761, 778, 795, 812, 831, 849, 869, 888, 906, 926, 948, 971, 999, 1020, 1041, 1063, 1087, 1113, 1147, 1165, 1185, 1212, 1238, 1280, 1343, 1443, 1584, 1798, 2212, 100000000],
    [13, 23, 33, 34, 42, 53, 61, 70, 80, 93, 108, 125, 143, 159, 176, 195, 216, 239, 264, 290, 319, 350, 380, 404, 428, 452, 477, 503, 531, 560, 590, 621, 654, 688, 724, 762, 802, 844, 888, 935, 982, 1031, 1081, 1130, 1174, 1220, 1260, 1291, 1323, 1357, 1393, 1430, 1470, 1514, 1560, 1610, 1664, 1726, 1794, 1869, 1944, 2029, 2167, 2465, 3051, 100000000],
    [13, 23, 33, 34, 42, 53, 61, 70, 83, 96, 106, 115, 126, 139, 154, 170, 188, 207, 229, 253, 279, 308, 340, 374, 411, 451, 494, 528, 561, 596, 632, 672, 713, 758, 806, 855, 908, 966, 1020, 1077, 1129, 1184, 1243, 1304, 1369, 1438, 1512, 1589, 1670, 1757, 1845, 1930, 1988, 2031, 2073, 2130, 2190, 2255, 2327, 2409, 2496, 2599, 2711, 2830, 2951, 3050, 3203, 3387, 3575, 3736, 3965, 4227, 4636, 5353, 100000000],
  ]
}