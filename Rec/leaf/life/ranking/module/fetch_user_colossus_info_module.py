from ranking import CommonModule

colossus_action_names = [
  "Play", 
  "Play0To7", 
  "Play7To20", 
  "Play20To58", 
  "Play58More", 
  "Ev", 
  "Lv", 
  "Interact"
]

colossus_feature_prefix = "pColossus"
colossus_score_name = "pColossusScore"

def get_photo_colossus_features():
  features = [colossus_score_name] # 老 leaf 中的 user_colos_score
  colos_feat_day_period_strs = ["D7", "D15", "D30", "All"]
  for action_name in colossus_action_names:
    for period in colos_feat_day_period_strs:
      for suffix in ["Hetu1", "Hetu2", "Hetu3"]:
        features.append(colossus_feature_prefix + action_name + period + suffix)
  return features

photo_colossus_features = get_photo_colossus_features()

class FetchUserColossusInfoModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def process(self) -> None:
        self.flow \
        .if_("enable_user_colossus_info == 1") \
          .explore_user_colossus_x_item_enricher(
            feature_prefix = colossus_feature_prefix,
            author_id_attr = "author__id",
            hetu_level_one_tag_list_attr = "hetu_tag_level_info__hetu_level_one",
            hetu_level_two_tag_list_attr = "hetu_tag_level_info__hetu_level_two",
            hetu_level_three_tag_list_attr = "hetu_tag_level_info__hetu_level_three",
            hetu_level_four_tag_list_attr = "hetu_tag_level_info__hetu_level_four",
            colossus_score_name_attr = colossus_score_name,
            colossus_resp_attr = "colossus_resp_old",
            colossus_resp_attr_v2 = "colossus_resp_v2",
            colossus_feature_periods_attr = "colossus_feature_periods",
            colossus_action_names_attrs = colossus_action_names,
            user_colossus_info_min_sec_ago="{{user_colossus_info_min_sec_ago}}",
            user_colossus_info_max_sec_ago="{{user_colossus_info_max_sec_ago}}",
            enable_colossus_v2="{{enable_colossus_v2}}",
            user_colossus_interact_weight_like="{{user_colossus_interact_weight_like}}",
            user_colossus_interact_weight_follow="{{user_colossus_interact_weight_follow}}",
            user_colossus_interact_weight_forward="{{user_colossus_interact_weight_forward}}",
            user_colossus_interact_weight_comment="{{user_colossus_interact_weight_comment}}",
            user_colossus_interact_weight_profile="{{user_colossus_interact_weight_profile}}",
            user_colossus_item_max_parse_num="{{user_colossus_item_max_parse_num}}",
            user_colossus_info_min_playtime_sec="{{user_colossus_info_min_playtime_sec}}",
            user_colossus_info_weight_hetu2="{{user_colossus_info_weight_hetu2}}",
            user_colossus_info_weight_hetu3="{{user_colossus_info_weight_hetu3}}",
            user_colossus_info_weight_hetu4="{{user_colossus_info_weight_hetu4}}",
            user_colossus_info_weight_aid="{{user_colossus_info_weight_aid}}",
            user_colossus_interact_action_max_bucket="{{user_colossus_interact_action_max_bucket}}",
            consume_time_ltr_disable_hetu_relation_update="{{consume_time_ltr_disable_hetu_relation_update}}",
            output_item_attrs = photo_colossus_features,
          ) \
        .end_()

    def post_process(self) -> None:
      self.flow \
      .log_debug_info(
        common_attrs = [
          "colossus_feature_periods"
        ],
        item_attrs = photo_colossus_features,
        for_debug_request_only = True
      )