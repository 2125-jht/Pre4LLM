import json

json_param_config = {
  "user_config": {
    "device_id": {"dim": 32, "attrs": [{"attr": ["device_id"], "key_type": 100, "converter": "id"}], "use_common_attr_only": True},
    "user_id": {"dim": 32, "attrs": [{"attr": ["user_id"], "key_type": 101, "converter": "id"}], "use_common_attr_only": True},
    "user_gender": {"dim": 4, "attrs": [{"attr": ["user_gender"], "key_type": 102, "converter": "id"}], "use_common_attr_only": True},
    "user_age_segment": {"dim": 4, "attrs": [{"attr": ["user_age_segment"], "key_type": 103, "converter": "id"}], "use_common_attr_only": True},
  },

  "item_config": {
    "photo_id_list": {"dim": 8, "attrs": [{"attr": ["photo_id_list"], "key_type":201, "converter": "list"}], "expand":10},
    "author_id_list": {"dim": 16, "attrs": [{"attr": ["author_id_list"], "key_type": 202, "converter": "list"}], "expand":10},
    "hetu_cluster_id_list": {"dim": 4, "attrs": [{"attr": ["hetu_cluster_id_list"], "key_type": 203, "converter": "list"}], "expand":10},
    "hetu_level_one_tag_list": {"dim": 4, "attrs": [{"attr": ["hetu_level_one_tag_list"], "key_type": 204, "converter": "list"}], "expand":10},
    "hetu_level_two_tag_list": {"dim": 4, "attrs": [{"attr": ["hetu_level_two_tag_list"], "key_type": 205, "converter": "list"}], "expand":10},
    "hetu_level_three_tag_list": {"dim": 4, "attrs": [{"attr": ["hetu_level_three_tag_list"], "key_type": 206, "converter": "list"}], "expand":10},
  },

  "extra_param": {
    "pctr_list": {"dim": 10},
    "pltr_list": {"dim": 10},
    "pwtr_list": {"dim": 10},
    "plvtr_list": {"dim": 10},
    "pcmtr_list": {"dim": 10},
    "pcmef_list": {"dim": 10},
    "pptr_list": {"dim": 10},
    # "pctr_index": {"dim": 1, "type":"bigint"},
    # "pltr_index": {"dim": 1, "type":"bigint"},
    # "pwtr_index": {"dim": 1, "type":"bigint"},
    # "pvtr_index": {"dim": 1, "type":"bigint"},
    # "plvtr_index": {"dim": 1, "type":"bigint"},

    "pctr_index_list": {"dim": 10},
    "pltr_index_list": {"dim": 10},
    "pwtr_index_list": {"dim": 10},
    "pvtr_index_list": {"dim": 10},
    "plvtr_index_list": {"dim": 10},

    "emp_ctr_list": {"dim": 10},
    "emp_ltr_list": {"dim": 10},
    "emp_wtr_list": {"dim": 10},
    "emp_lvtr_list": {"dim": 10},
  }
}

default_dim = 16


def get_all_config():
    user_config = {}
    item_config = {}
    extra_param = {}
    for k, v in json_param_config.items():
        if k == 'user_config':
          user_config.update(v)
        elif k == 'item_config':
          item_config.update(v)
        else:
          extra_param.update(v)
    return user_config, item_config, extra_param



class Attr:
    def __init__(self, attr_name, attr_config):
        # model feature name
        self.attr_name = attr_name
        # feature dim
        self.dim = attr_config.get("dim", default_dim)
        # 多个slot转为一个embedding
        self.slots = list(set(attr.get("key_type") for attr in attr_config.get("attrs")))
        self.expand = attr_config.get("expand", None)
        # is common
        self.is_common = attr_config.get("use_common_attr_only", False)


user_config, item_config, extra_param_config = get_all_config()
user_features = [Attr(attr_name, attr_config) for attr_name, attr_config in user_config.items()]
item_features = [Attr(attr_name, attr_config) for attr_name, attr_config in item_config.items()]
