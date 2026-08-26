import json
import sys


# 本次模型使用的所有特征，需要从feature pool抽取
# dim-特征维度，需要用户自定义
# share_id-需要share embedding的特征设置为相同的share id，具体值用户自定义
# expand-list特征是否需要展开，如果自定义list长度则需要
default_dim = 8
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

  "photo_id": {"dim": 64, "belong": "item"},
  "photo_author_id": {"dim": 32, "belong": "item"},
  "photo_author_gender": {"dim": 4, "belong": "item"},
  "photo_upload_type": {"dim": 4, "belong": "item"},
  "photo_hetu_tag_level1_list": {"belong": "item"},
  "photo_hetu_tag_level2_list": {"belong": "item"},
  "photo_hetu_tag_level3_list": {"belong": "item"},
  "photo_hetu_tag_level5_list": {"belong": "item"},
  # "photo_tag": {"dim": 24, "belong": "item"},
  "photo_duration_ms": {"belong": "item"},

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
  "context_fullrank_ltr_score": {"belong": "item"},
  "context_fullrank_act_wtd": {"belong": "item"},
  "context_fullrank_ltr_v4_fountain_next": {"belong": "item"},
  # "context_fountain_related_score_v2": {"belong": "item"},

  "context_cascade_pctr": {"belong": "item"},
  "context_cascade_pltr": {"belong": "item"},
  "context_cascade_pwtr": {"belong": "item"},
  "context_cascade_plvtr": {"belong": "item"},
  # "context_cascade_pftr": {"belong": "item"},
  # "context_cascade_pptr": {"belong": "item"},
  # "context_cascade_pcmtr": {"belong": "item"},

  # "context_source_pid": {"dim": 64, "belong": "source"},
  "context_source_aid": {"dim": 32, "belong": "source"},
  "context_source_hetu_tag_level1_top1": {"belong": "source"},
  "context_source_hetu_tag_level2_top1": {"belong": "source"},
  "context_source_tag": {"dim": 24, "belong": "source"},
  "context_source_duration_ms": {"belong": "source"},
}
user_fea_names, photo_fea_names, explore_profile_fea_names, fountain_seq_pid_names, fountain_seq_aid_names, source_fea_names = \
[],[],[],[],[],[]
##加载特征池
feature_pool_config = json.load(open("./feature_pool.json", "r"))

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
            elif belong == "user_rt_pid_seq":
              explore_profile_fea_names.append(k)
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