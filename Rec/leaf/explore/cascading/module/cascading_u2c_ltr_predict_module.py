from cascading import CommonModule

class CascadingU2cLtrPredictModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def uni_feature_trim_user_info(self):
    features = [
      "id",
      "device_id",
      "request_location.province_id",
      "request_location.city_id",
      "gender",
      "infer_gender",
      "true_gender",
      "infer_year",
      "basic_info.age_segment",
      "location.city_id",
      "location.city_level",
      "client_id",
      "visit_mod",
      "visit_net",
      "user_profile.user_level",
      "active_days",
      "follow_count",
      "feature_collection.explore_low_active_level",
      "user_profile_v1.click_list.hetu_tag_level_info.hetu_level_one",
      "user_profile_v1.click_list.hetu_tag_level_info.hetu_level_two",
      "user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_one",
      "user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_two",
      "user_profile_v1.like_list.hetu_tag_level_info.hetu_level_one",
      "user_profile_v1.like_list.hetu_tag_level_info.hetu_level_two",
      "user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_one",
      "user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_two",
      "user_profile.exp_stat.exp_click",
      "user_profile.exp_stat.exp_like",
      "user_profile.exp_stat.exp_follow",
      "user_profile.exp_stat.exp_realshow",
      "user_profile.exp_stat.exp_long_view",
    ]
    return features

          
  def process(self) -> None:
    self.flow \
      .if_("enable_explore_u2c_ltr_predict == 1") \
        .explore_custom_trim_user_info(
          user_info_attr = "userInfo",
          save_trimed_user_info_to_attr = "user_info_str",
          trim_user_info = self.uni_feature_trim_user_info(),
        ) \
        .pack_item_attr(
          item_source = {
            "reco_results": True,
          },
          mappings = [{
            "from_item_attr": "hetu_sim_cluster_id",
            "to_common_attr": "candidate_cids",
            "dedup_to_common_attr": True
          }]
        ) \
        .delegate_enrich(
          item_list_from_attr = "candidate_cids",
          kess_service = "{{explore_u2c_ltr_service}}",
          recv_item_attrs = [
            "explore_u2c",
          ],
          timeout_ms = 100,
          partition_size = "{{explore_u2c_ltr_kai_partition_size}}",
          send_common_attrs = [
            "user_info_str",
            "uOldMmuClusterId300ListList",
          ],
        ) \
      .end_()
