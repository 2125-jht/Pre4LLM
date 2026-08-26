import json


# 本次模型使用的所有特征，需要从feature pool抽取
# dim-特征维度，需要用户自定义
# share_id-需要share embedding的特征设置为相同的share id，具体值用户自定义
# expand-list特征是否需要展开，如果自定义list长度则需要
default_dim = 16
all_feats = {
    "user_id": {},

    # "photo_id": {},
    # "photo_author_id": {},

    # "photo_author_fans_count": {"dim": 16},
    # "photo_author_fans_count_2": {"dim": 16},
    # "photo_author_upload_count": {"dim": 16},
    # "photo_author_upload_count_2": {"dim": 16},
    # "photo_author_click_count": {"dim": 16},
    # "photo_author_click_count_2": {"dim": 16},
    # "photo_author_like_count": {"dim": 16},
    # "photo_author_like_count_2": {"dim": 16},
    # "photo_author_follow_count": {"dim": 16},
    # "photo_author_follow_count_2": {"dim": 16},
    # "photo_author_long_view_count": {"dim": 16},
    # "photo_author_long_view_count_2": {"dim": 16},
    # "photo_author_emp_ctr": {"dim": 16},
    # "photo_author_emp_ltr": {"dim": 16},
    # "photo_author_emp_wtr": {"dim": 16},
    # "photo_author_emp_lvtr": {"dim": 16},
    # "photo_author_emp_svtr": {"dim": 16},
    # "photo_author_emp_watch_time": {"dim": 16},
}

copy_feats = {
    # "user_id":[("user_emb", 4006, 16)],
    # "photo_id":[
    #     ("photo_emb", 4103, 16),
    # ],
    # "photo_author_id":[
    #     ("photo_author_id_v2", 4200),
    # ],
}

##加载特征池
feature_pool_config = json.load(open("./feature_pool.json", "r"))

infer_ignore_feat = ["user_emb", "photo_emb", "last_step","ave_step"]

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
                for config in copy_feats[k]:
                    new_name = config[0]
                    new_slot = config[1]
                    copy_input_slots.append(slot)
                    copy_output_slots.append(new_slot)
                    # 和被 copy 特征使用不同的 emb dim
                    if len(config) > 2:
                        dim = config[2]
                    all_features.append(Attr(new_name, [new_slot], is_common, dim, expand))
                    print("--->>> {} copy from {}".format(new_name, k))
        else:
            print (k, "feature is not in featue pool, please check!!!")

    ### share embedding, share到最小slot
    for k, v in all_share_id.items():
        v.sort()
        share_input_slots += v[1:]
        share_output_slots += [v[0]] * (len(v) - 1)

    return all_features, share_input_slots, share_output_slots, copy_input_slots, copy_output_slots

all_features, share_input_slots, share_output_slots, copy_input_slots, copy_output_slots = get_all_feature_attrs(all_feats)

print(share_input_slots)
print(share_output_slots)

print(copy_input_slots)
print(copy_output_slots)