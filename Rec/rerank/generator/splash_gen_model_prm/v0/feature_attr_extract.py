import json


user_fea_names = [
  "user_id",
  "user_device_id",
  "user_age_segment",
  "user_gender",
  "user_city_id",
  "user_client_id",
  "user_visit_mod",
  "user_level",
#   "user_active_days",
  "user_emp_ctr",
  "user_emp_ltr",
  "user_emp_wtr",
  "user_emp_lvtr",
  "user_request_province_id",
  "user_request_city_id",
  "user_request_poi_type",
  "user_region_type",
#   "context_similar_user_list",
]

explore_profile_fea_names = [
  "user_realtime_click_list",
  "user_realtime_like_list",
  "user_realtime_follow_list",
  "user_realtime_forward_list",
  # "user_profile_v1_click_pid_list",
  # "user_profile_v1_click_aid_list",
  # "user_profile_v1_like_pid_list",
  # "user_profile_v1_like_aid_list",
  # "user_profile_v1_follow_pid_list",
  # "user_profile_v1_follow_aid_list",
]

fountain_profile_fea_names = [
#   "user_fountain_profile_click_pid_list",
#   "user_fountain_profile_click_aid_list",
  "user_fountain_profile_like_pid_list",
  "user_fountain_profile_like_aid_list",
#   "user_fountain_profile_follow_pid_list",
#   "user_fountain_profile_follow_aid_list",
#   "user_fountain_profile_comment_pid_list",
#   "user_fountain_profile_comment_aid_list",
  "user_fountain_profile_effective_view_pid_list",
  "user_fountain_profile_effective_view_aid_list",
#   "user_fountain_profile_short_view_pid_list",
#   "user_fountain_profile_short_view_aid_list"
]

playtime_fea_names = [
  # "user_id",
  # "photo_id_v2",  ##特征池取不到需要额外remap，见feature_attr_extract.py
  "user_fountain_profile_effective_view_pid_list",
  "user_fountain_profile_effective_view_aid_list",
#   "user_fountain_profile_short_view_pid_list",
#   "user_fountain_profile_short_view_aid_list",
  "user_fountain_profile_long_view_pid_list",
  "user_fountain_profile_long_view_aid_list",
#   "user_profile_v1_play3s_pid_list",
#   "user_profile_v1_play3s_aid_list",
#   "user_profile_v1_play7s_pid_list",
#   "user_profile_v1_play7s_aid_list",
#   "user_profile_v1_play11s_pid_list",
#   "user_profile_v1_play11s_aid_list",
#   "user_profile_v1_play18s_pid_list",
#   "user_profile_v1_play18s_aid_list",
]

photo_fea_names = [
  "photo_id",
  "photo_author_id",
  "photo_author_gender",
#   "photo_author_age_segment",
#   "photo_province_id",
#   "photo_city_id",
#   "photo_mod",
  "photo_upload_type",
#   "photo_music",
  "photo_hetu_tag_level1_list",
  "photo_hetu_tag_level2_list",
  "photo_hetu_tag_level5_list",
  "photo_duration_ms",

#   "photo_emp_ctr",
#   "photo_emp_ltr",
#   "photo_emp_wtr",
#   "photo_emp_lvtr",
#   "photo_emp_svtr",
  "context_pctr",
  "context_pltr",
  "context_pwtr",
  "context_pftr",
  "context_plvtr",
  "context_pvtr",
  "context_pptr",
  "context_pcmtr",

  "context_pepstr",
  "context_pcpr",
  "context_pcltr",
  "context_psvr",
  "context_pwtd",
  "context_fullrank_ltr_score",
  "context_fullrank_act_wtd",
  "context_fullrank_ltr_v4_fountain_next",
  "context_fountain_related_score_v2",

  "context_cascade_pctr",
  "context_cascade_pltr",
  "context_cascade_pwtr",
  "context_cascade_plvtr",
  "context_cascade_pftr",
  "context_cascade_pptr",
  "context_cascade_pcmtr",
]

source_fea_names = [
  "context_source_pid",
  "context_source_aid",
  "context_source_hetu_tag_level1_top1",
  "context_source_hetu_tag_level2_top1",
  "context_source_tag",
  "context_source_duration_ms",
  "context_hour_of_day",
  "context_day_of_week",
  # "context_first_page",
  # "context_page",
]
# 本次模型使用的所有特征，需要从feature pool抽取
# dim-特征维度，需要用户自定义
# share_id-需要share embedding的特征设置为相同的share id，具体值用户自定义
# expand-list特征是否需要展开，如果自定义list长度则需要
default_dim = 16
all_feats = {
  "user_id": {"dim": 32, "share_id": 1},
  "user_device_id": {"dim": 32},
  "user_age_segment": {},
  "user_gender": {},
  "user_city_id": {},
  "user_client_id": {},
  "user_visit_mod": {},
  "user_level": {},
#   "user_active_days": {},
  "user_emp_ctr": {},
  "user_emp_ltr": {},
  "user_emp_wtr": {},
  "user_emp_lvtr": {},
  "user_request_province_id": {},
  "user_request_city_id": {},
  "user_request_poi_type": {},
  "user_region_type": {},
#   "context_similar_user_list": {"dim": 32, "share_id": 1},

  "user_realtime_click_list": {"expand":50, "share_id": 2},
  "user_realtime_like_list": {"expand":50, "share_id": 2},
  "user_realtime_follow_list": {"expand":50, "share_id": 2},
  "user_realtime_forward_list": {"expand":50, "share_id": 2},
#   "user_profile_v1_click_pid_list": {"expand":50, "share_id": 2},
#   "user_profile_v1_click_aid_list": {"expand":50, "share_id": 3}, 
#   "user_profile_v1_like_pid_list": {"expand":50, "share_id": 2},
#   "user_profile_v1_like_aid_list": {"expand":50, "share_id": 3},
#   "user_profile_v1_follow_pid_list": {"expand":50, "share_id": 2},
#   "user_profile_v1_follow_aid_list": {"expand":50, "share_id": 3},
#   "user_profile_v1_hate_pid_list": {"expand": 10, "share_id": 2},

#   "user_fountain_profile_click_pid_list": {"expand":50, "share_id": 2},
#   "user_fountain_profile_click_aid_list": {"expand":50, "share_id": 3},
  "user_fountain_profile_like_pid_list": {"expand":50, "share_id": 2},
  "user_fountain_profile_like_aid_list": {"expand":50, "share_id": 3},
#   "user_fountain_profile_follow_pid_list": {"expand":50, "share_id": 2},
#   "user_fountain_profile_follow_aid_list": {"expand":50, "share_id": 3},
#   "user_fountain_profile_comment_pid_list": {"expand":50, "share_id": 2},
#   "user_fountain_profile_comment_aid_list": {"expand":50, "share_id": 3},
  "user_fountain_profile_effective_view_pid_list": {"expand":50, "share_id": 2},
  "user_fountain_profile_effective_view_aid_list": {"expand":50, "share_id": 3},
#   "user_fountain_profile_short_view_pid_list": {"expand":50, "share_id": 2},
#   "user_fountain_profile_short_view_aid_list": {"expand":50, "share_id": 3},
  "user_fountain_profile_long_view_pid_list": {"expand":50, "share_id": 2},
  "user_fountain_profile_long_view_aid_list": {"expand":50, "share_id": 3},
#   "user_fountain_source_induced_valid_play_list": {"expand": 10, "share_id": 2},
#   "user_fountain_source_induced_shortview_play_list": {"expand": 10, "share_id": 2},


#   "user_profile_v1_play3s_pid_list": {"expand":50, "share_id": 2},
#   "user_profile_v1_play3s_aid_list": {"expand":50, "share_id": 3},
#   "user_profile_v1_play7s_pid_list": {"expand":50, "share_id": 2},
#   "user_profile_v1_play7s_aid_list": {"expand":50, "share_id": 3},
#   "user_profile_v1_play11s_pid_list": {"expand":50, "share_id": 2},
#   "user_profile_v1_play11s_aid_list": {"expand":50, "share_id": 3},
#   "user_profile_v1_play18s_pid_list": {"expand":50, "share_id": 2},
#   "user_profile_v1_play18s_aid_list": {"expand":50, "share_id": 3},


  "photo_id": {},
  "photo_author_id": {},
  "photo_author_gender": {},
  "photo_upload_type": {},
  "photo_hetu_tag_level1_list": {},
  "photo_hetu_tag_level2_list": {},
#   "photo_hetu_tag_level3_list": {},
  "photo_hetu_tag_level5_list": {},
#   "photo_tag": {"dim": 24, },
  "photo_duration_ms": {"share_id": 4},
#   "photo_duration_s": {},

  "context_pctr": {},
  "context_pltr": {},
  "context_pwtr": {}, # 关注
  "context_pftr": {}, # 分享
  "context_plvtr": {},
  "context_pvtr": {},
  "context_pptr": {},
  "context_pcmtr": {},
  "context_pepstr": {},
  "context_pcpr": {},
  "context_pcltr": {},
  "context_psvr": {},
  "context_pwtd": {},
  "context_fullrank_ltr_score": {},
  "context_fullrank_act_wtd": {},
  "context_fullrank_ltr_v4_fountain_next": {},
  "context_fountain_related_score_v2": {},

  "context_cascade_pctr": {},
  "context_cascade_pltr": {},
  "context_cascade_pwtr": {},
  "context_cascade_plvtr": {},
  "context_cascade_pftr": {},
  "context_cascade_pptr": {},
  "context_cascade_pcmtr": {},

  "context_source_pid": {"dim": 32},
  "context_source_aid": {"dim": 32},
  "context_source_hetu_tag_level1_top1": {},
  "context_source_hetu_tag_level2_top1": {},
  "context_source_tag": {},
  "context_source_duration_ms": {"share_id": 4},
#   "context_source_playtime_s": {},
#   "context_source_is_interacted": {},
  "context_hour_of_day": {},
  "context_day_of_week": {},
  # "context_first_page": {},
  # "context_page": {},
}
copy_feats = {
  "photo_id":["photo_id_v2", 4000]
}

##加载特征池
feature_pool_config = json.load(open("./feature_pool.json", "r"))

class Attr:
    def __init__(self, attr_name, slot, is_common, dim, expand):
        # model feature name
        self.attr_name = attr_name
        # feature dim
        self.dim = dim
        self.slots = slot
        self.expand = expand
        self.is_common = is_common


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
            #expand = None
            all_features.append(Attr(k, [slot], is_common, dim, expand))
            # share id
            share_id = all_feats[k].get("share_id", None)
            if share_id is not None:
                if share_id in all_share_id:
                    all_share_id[share_id].append(slot)
                else:
                    all_share_id[share_id] = [slot]
            # copy featues
            if k in copy_feats:
                new_name = copy_feats[k][0]
                new_slot = copy_feats[k][1]
                copy_input_slots.append(slot)
                copy_output_slots.append(new_slot)
                all_features.append(Attr(new_name, [new_slot], is_common, dim, expand))
        else:
            print (k, "feature is not in featue pool, please check!!!")

    ### share embedding, share到最小slot
    for k, v in all_share_id.items():
        v.sort()
        share_input_slots += v[1:]
        share_output_slots += [v[0]] * (len(v) - 1)

    return all_features, share_input_slots, share_output_slots, copy_input_slots, copy_output_slots

all_features, share_input_slots, share_output_slots, copy_input_slots, copy_output_slots = get_all_feature_attrs(all_feats)
