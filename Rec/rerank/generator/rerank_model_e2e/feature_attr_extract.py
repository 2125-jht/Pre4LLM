import json


# 本次模型使用的所有特征，需要从feature pool抽取
# dim-特征维度，需要用户自定义
# share_id-需要share embedding的特征设置为相同的share id，具体值用户自定义
# expand-list特征是否需要展开，如果自定义list长度则需要
default_dim = 8
id_dim = 32
all_feats = {
  "user_id": {"dim": 32},
  "user_device_id": {"dim": 32},
  "user_age_segment": {"dim": 4},
  "user_gender": {"dim": 4},
  "user_city_id": {"dim": 4},
  "user_client_id": {"dim": 4},
  "user_level": {"dim": 4},
  "user_active_days": {"dim": 4},
  "user_emp_ctr": {"dim": 4},
  "user_emp_ltr": {"dim": 4},
  "user_emp_wtr": {"dim": 4},
  "user_emp_lvtr": {"dim": 4},
  "user_request_province_id": {"dim": 4},
  "user_request_city_id": {"dim": 4},
  "user_request_poi_type": {"dim": 4},
  "user_region_type": {"dim": 4},

  "user_realtime_click_list": {"expand":50},
  "user_realtime_like_list": {"expand":50},
  "user_realtime_follow_list": {"expand":50},
  "user_realtime_forward_list": {"expand":50},

  "photo_id": {"dim": 64, "expand":60},
  "photo_author_id": {"dim": 64, "expand":60},
  "photo_author_gender": {"dim": 4, "expand":60},
  "photo_author_age_segment": {"expand":60},
  "photo_province_id": {"expand":60},
  "photo_city_id": {"expand":60},
  "photo_mod": {"expand":60},
  "photo_upload_type": {"expand":60},
  "photo_hetu_tag_level1_list": {"expand":60},
  "photo_hetu_tag_level2_list": {"expand":60},
  "photo_hetu_tag_level5_list": {"expand":60},
  "photo_duration_ms": {"expand":60},

  "photo_emp_ctr": {"expand":60},
  "photo_emp_ltr": {"expand":60},
  "photo_emp_wtr": {"expand":60},
  "photo_emp_lvtr": {"expand":60},
  "photo_emp_svtr": {"expand":60},
  "context_pctr": {"dim": 16, "expand":60},
  "context_pltr": {"expand":60},
  "context_pwtr": {"expand":60},
  "context_pftr": {"expand":60},
  "context_plvtr": {"expand":60},
  "context_psvtr": {"expand":60},
  "context_pvtr": {"expand":60},
  "context_pptr": {"expand":60},
  "context_pcmtr": {"expand":60},
  "context_pwtd": {"expand":60},
  "context_cascade_pctr": {"dim": 16, "expand":60},
  "context_cascade_pltr": {"expand":60},
  "context_cascade_pwtr": {"expand":60},
  "context_cascade_plvtr": {"expand":60},
  "context_cascade_psvtr": {"expand":60},
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
        else:
            print (k, "feature is not in featue pool, please check!!!")

    ### share embedding, share到最小slot
    for k, v in all_share_id.items():
        v.sort()
        share_input_slots += v[1:]
        share_output_slots += [v[0]] * (len(v) - 1)

    return all_features, share_input_slots, share_output_slots, copy_input_slots, copy_output_slots

all_features, share_input_slots, share_output_slots, copy_input_slots, copy_output_slots = get_all_feature_attrs(all_feats)
