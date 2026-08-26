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
  # "user_city_id": {"belong": "user"},
  # "user_client_id": {"belong": "user"},
  "user_level": {"belong": "user"},
  "user_emp_ctr": {"belong": "user"},
  "user_emp_ltr": {"belong": "user"},
  "user_emp_wtr": {"belong": "user"},
  "user_emp_lvtr": {"belong": "user"},
  "user_request_city_id": {"belong": "user"},
  # "user_request_poi_type": {"belong": "user"},
  # "user_region_type": {"belong": "user"},

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
  # "photo_city_id": {"belong": "item"},
  # "photo_music": {"belong": "item"},

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
  "photo_info__explore_stat__real_show_count_list": {"norm_type": "x^0.7", "name": "emp_explore_show_count", "boundaries": [1, 2, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2200, 2400, 2600, 2800, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 7000, 8000, 9000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 80000]},
  "photo_info__explore_stat__click_count_list": {"norm_type": "x^0.7", "name": "emp_explore_click_count", "boundaries": [1, 2, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2200, 2400, 2600, 2800, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 7000, 8000, 9000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 80000]},
  "photo_info__fountain_stats__real_show_count_list": {"norm_type": "x^0.7", "name": "emp_fountain_show_count", "boundaries": [1, 2, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2200, 2400, 2600, 2800, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 7000, 8000, 9000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 80000]},
  "photo_info__fountain_stats__like_count_list": {"name": "emp_fountain_like_count", "boundaries": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500, 2550, 2600, 2650, 2700, 2750, 2800, 2850, 2900, 2950, 3000, 3050, 3100, 3150, 3200, 3250, 3300, 3350, 3400, 3450, 3500, 3550, 3600, 3650, 3700, 3750, 3800, 3850, 3900, 3950, 4000, 4050, 4100, 4150, 4200, 4250, 4300, 4350, 4400, 4450, 4500, 4550, 4600, 4650, 4700, 4750, 4800, 4850, 4900, 4950]},
  "photo_info__fountain_stats__follow_count_list": {"name": "emp_fountain_follow_count", "boundaries": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500, 2550, 2600, 2650, 2700, 2750, 2800, 2850, 2900, 2950, 3000, 3050, 3100, 3150, 3200, 3250, 3300, 3350, 3400, 3450, 3500, 3550, 3600, 3650, 3700, 3750, 3800, 3850, 3900, 3950, 4000, 4050, 4100, 4150, 4200, 4250, 4300, 4350, 4400, 4450, 4500, 4550, 4600, 4650, 4700, 4750, 4800, 4850, 4900, 4950]},
  "photo_info__fountain_stats__long_play_count_list": {"name": "emp_fountain_long_play_count", "boundaries": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500, 2550, 2600, 2650, 2700, 2750, 2800, 2850, 2900, 2950, 3000, 3050, 3100, 3150, 3200, 3250, 3300, 3350, 3400, 3450, 3500, 3550, 3600, 3650, 3700, 3750, 3800, 3850, 3900, 3950, 4000, 4050, 4100, 4150, 4200, 4250, 4300, 4350, 4400, 4450, 4500, 4550, 4600, 4650, 4700, 4750, 4800, 4850, 4900, 4950]},
  "photo_emp_explore_ctr": {"name": "photo_emp_explore_ctr", "boundaries": [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1, 0.105, 0.11, 0.115, 0.12, 0.125, 0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, 0.18, 0.185, 0.19, 0.195, 0.2, 0.205, 0.21, 0.215, 0.22, 0.225, 0.23, 0.235, 0.24, 0.245, 0.25, 0.255, 0.26, 0.265, 0.27, 0.275, 0.28, 0.285, 0.29, 0.295, 0.3, 0.305, 0.31, 0.315, 0.32, 0.325, 0.33, 0.335, 0.34, 0.345, 0.35, 0.355, 0.36, 0.365, 0.37, 0.375, 0.38, 0.385, 0.39, 0.395, 0.4, 0.405, 0.41, 0.415, 0.42, 0.425, 0.43, 0.435, 0.44, 0.445, 0.45, 0.455, 0.46, 0.465, 0.47, 0.475, 0.48, 0.485, 0.49, 0.495, 0.5]},
  "photo_emp_explore_ltr": {"name": "photo_emp_explore_ltr", "boundaries": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.04, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049, 0.05, 0.051, 0.052, 0.053, 0.054, 0.055, 0.056, 0.057, 0.058, 0.059, 0.06, 0.061, 0.062, 0.063, 0.064, 0.065, 0.066, 0.067, 0.068, 0.069, 0.07, 0.071, 0.072, 0.073, 0.074, 0.075, 0.076, 0.077, 0.078, 0.079, 0.08, 0.081, 0.082, 0.083, 0.084, 0.085, 0.086, 0.087, 0.088, 0.089, 0.09, 0.091, 0.092, 0.093, 0.094, 0.095, 0.096, 0.097, 0.098, 0.099, 0.1]},
  "photo_emp_explore_avg_time": {"name": "photo_emp_explore_avg_time", "boundaries": [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0, 36.0, 39.0, 42.0, 45.0, 48.0, 51.0, 54.0, 57.0, 60.0, 63.0, 66.0, 69.0, 72.0, 75.0, 78.0, 81.0, 84.0, 87.0, 90.0, 93.0, 96.0, 99.0, 102.0, 105.0, 108.0, 111.0, 114.0, 117.0, 120.0, 123.0, 126.0, 129.0, 132.0, 135.0, 138.0, 141.0, 144.0, 147.0, 150.0, 153.0, 156.0, 159.0, 162.0, 165.0, 168.0, 171.0, 174.0, 177.0, 180.0, 183.0, 186.0, 189.0, 192.0, 195.0, 198.0, 201.0, 204.0, 207.0, 210.0, 213.0, 216.0, 219.0, 222.0, 225.0, 228.0, 231.0, 234.0, 237.0, 240.0, 243.0, 246.0, 249.0, 252.0, 255.0, 258.0, 261.0, 264.0, 267.0, 270.0, 273.0, 276.0, 279.0, 282.0, 285.0, 288.0, 291.0, 294.0, 297.0, 300.0]},
  "photo_emp_fountain_svtr": {"name": "photo_emp_fountain_svtr", "boundaries": [0.104, 0.112, 0.12, 0.128, 0.136, 0.144, 0.152, 0.16, 0.168, 0.176, 0.184, 0.192, 0.2, 0.208, 0.216, 0.224, 0.232, 0.24, 0.248, 0.256, 0.264, 0.272, 0.28, 0.288, 0.296, 0.304, 0.312, 0.32, 0.328, 0.336, 0.344, 0.352, 0.36, 0.368, 0.376, 0.384, 0.392, 0.4, 0.408, 0.416, 0.424, 0.432, 0.44, 0.448, 0.456, 0.464, 0.472, 0.48, 0.488, 0.496, 0.504, 0.512, 0.52, 0.528, 0.536, 0.544, 0.552, 0.56, 0.568, 0.576, 0.584, 0.592, 0.6, 0.608, 0.616, 0.624, 0.632, 0.64, 0.648, 0.656, 0.664, 0.672, 0.68, 0.688, 0.696, 0.704, 0.712, 0.72, 0.728, 0.736, 0.744, 0.752, 0.76, 0.768, 0.776, 0.784, 0.792, 0.8]},
  "photo_emp_fountain_lvtr": {"name": "photo_emp_fountain_lvtr", "boundaries": [0.104, 0.112, 0.12, 0.128, 0.136, 0.144, 0.152, 0.16, 0.168, 0.176, 0.184, 0.192, 0.2, 0.208, 0.216, 0.224, 0.232, 0.24, 0.248, 0.256, 0.264, 0.272, 0.28, 0.288, 0.296, 0.304, 0.312, 0.32, 0.328, 0.336, 0.344, 0.352, 0.36, 0.368, 0.376, 0.384, 0.392, 0.4, 0.408, 0.416, 0.424, 0.432, 0.44, 0.448, 0.456, 0.464, 0.472, 0.48, 0.488, 0.496, 0.504, 0.512, 0.52, 0.528, 0.536, 0.544, 0.552, 0.56, 0.568, 0.576, 0.584, 0.592, 0.6, 0.608, 0.616, 0.624, 0.632, 0.64, 0.648, 0.656, 0.664, 0.672, 0.68, 0.688, 0.696, 0.704, 0.712, 0.72, 0.728, 0.736, 0.744, 0.752, 0.76, 0.768, 0.776, 0.784, 0.792, 0.8]},
  "photo_emp_fountain_ltr": {"name": "photo_emp_fountain_ltr", "boundaries": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.04, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049, 0.05, 0.051, 0.052, 0.053, 0.054, 0.055, 0.056, 0.057, 0.058, 0.059, 0.06, 0.061, 0.062, 0.063, 0.064, 0.065, 0.066, 0.067, 0.068, 0.069, 0.07, 0.071, 0.072, 0.073, 0.074, 0.075, 0.076, 0.077, 0.078, 0.079, 0.08, 0.081, 0.082, 0.083, 0.084, 0.085, 0.086, 0.087, 0.088, 0.089, 0.09, 0.091, 0.092, 0.093, 0.094, 0.095, 0.096, 0.097, 0.098, 0.099, 0.1]},
  "photo_emp_fountain_wtr": {"name": "photo_emp_fountain_wtr", "boundaries": [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017, 0.0018, 0.0019, 0.002, 0.0021, 0.0022, 0.0023, 0.0024, 0.0025, 0.0026, 0.0027, 0.0028, 0.0029, 0.003, 0.0031, 0.0032, 0.0033, 0.0034, 0.0035, 0.0036, 0.0037, 0.0038, 0.0039, 0.004, 0.0041, 0.0042, 0.0043, 0.0044, 0.0045, 0.0046, 0.0047, 0.0048, 0.0049, 0.005, 0.0051, 0.0052, 0.0053, 0.0054, 0.0055, 0.0056, 0.0057, 0.0058, 0.0059, 0.006, 0.0061, 0.0062, 0.0063, 0.0064, 0.0065, 0.0066, 0.0067, 0.0068, 0.0069, 0.007, 0.0071, 0.0072, 0.0073, 0.0074, 0.0075, 0.0076, 0.0077, 0.0078, 0.0079, 0.008, 0.0081, 0.0082, 0.0083, 0.0084, 0.0085, 0.0086, 0.0087, 0.0088, 0.0089, 0.009, 0.0091, 0.0092, 0.0093, 0.0094, 0.0095, 0.0096, 0.0097, 0.0098, 0.0099, 0.01, 0.02, 0.05]},
  "photo_emp_fountain_avg_time": {"name": "photo_emp_fountain_avg_time", "boundaries": [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0, 36.0, 39.0, 42.0, 45.0, 48.0, 51.0, 54.0, 57.0, 60.0, 63.0, 66.0, 69.0, 72.0, 75.0, 78.0, 81.0, 84.0, 87.0, 90.0, 93.0, 96.0, 99.0, 102.0, 105.0, 108.0, 111.0, 114.0, 117.0, 120.0, 123.0, 126.0, 129.0, 132.0, 135.0, 138.0, 141.0, 144.0, 147.0, 150.0, 153.0, 156.0, 159.0, 162.0, 165.0, 168.0, 171.0, 174.0, 177.0, 180.0, 183.0, 186.0, 189.0, 192.0, 195.0, 198.0, 201.0, 204.0, 207.0, 210.0, 213.0, 216.0, 219.0, 222.0, 225.0, 228.0, 231.0, 234.0, 237.0, 240.0, 243.0, 246.0, 249.0, 252.0, 255.0, 258.0, 261.0, 264.0, 267.0, 270.0, 273.0, 276.0, 279.0, 282.0, 285.0, 288.0, 291.0, 294.0, 297.0, 300.0]},
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