from ranking import CommonModule
from ranking.fountain_ranking_features import user_features_v3, photo_features

class FountainFullRankDeepLtrModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("skip_fountain_deep_ltr_predict == 0") \
        .if_("enable_fountain_fullrank_deep_ltr_kai2 == 1") \
          .explore_custom_trim_user_info(
            user_info_attr = "userInfo",
            save_trimed_user_info_to_attr = "fr_deep_ltr_trimmed_user_info",
            trim_user_info = [
              "active_days",
              "basic_info.age_segment",
              "location.city_id",
              "location.region_type",
              "client_id",
              "device_id",
              "gender",
              "infer_gender",
              "true_gender",
              "request_location.poi_type",
              "request_location.province_id",
              "request_location.city_id",
              "visit_mod",
              "user_profile.exp_stat.exp_click",
              "user_profile.exp_stat.exp_like",
              "user_profile.exp_stat.exp_follow",
              "user_profile.exp_stat.exp_realshow",
              "user_profile.exp_stat.exp_long_view",
              "user_profile.user_level",
              "fountain_reco_user_profile.click_list.author_id",
              "fountain_reco_user_profile.click_list.photo_id",
              "fountain_reco_user_profile.comment_list.author_id",
              "fountain_reco_user_profile.comment_list.photo_id",
              "fountain_reco_user_profile.follow_list.author_id",
              "fountain_reco_user_profile.follow_list.photo_id",
              "fountain_reco_user_profile.like_list.author_id",
              "fountain_reco_user_profile.like_list.photo_id",
              "fountain_reco_user_profile.video_play_stat.photo_id",
              "fountain_reco_user_profile.video_play_stat.author_id",
              "fountain_reco_user_profile.video_play_stat.video_duration",
              "fountain_reco_user_profile.video_play_stat.playing_time",
              "user_profile_v1.click_list.author_id",
              "user_profile_v1.click_list.photo_id",
              "user_profile_v1.follow_list.author_id",
              "user_profile_v1.follow_list.photo_id",
              "user_profile_v1.like_list.author_id",
              "user_profile_v1.like_list.photo_id",
              "user_profile_v1.video_playing_stat.playing_time",
              "user_profile_v1.video_playing_stat.author_id",
              "user_profile_v1.video_playing_stat.photo_id",
              "realtime_click_list",
              "realtime_follow_list",
              "realtime_forward_list",
              "realtime_like_list",
            ],
          ) \
          .delegate_enrich(
            kess_service = "{{fountain_fullrank_deep_ltr_kess_service}}",
            recv_item_attrs = [
              {"name": "l2r", "as": "fullrank_ltr_score"},
              {"name": "ctr", "as": "fullrank_act_ctr"},
              {"name": "wtd", "as": "fullrank_act_wtd"},
              {"name": "finish_rate", "as": "fullrank_ltr_v4_fountain_finish_rate"},
              {"name": "next", "as": "fullrank_ltr_v4_fountain_next"},
            ],
            timeout_ms = 100,
            send_item_attrs = [
              "cascade_pctr",
              "cascade_pltr",
              "cascade_pwtr",
              "cascade_plvtr",
              "cascade_psvtr",
              "fullrank_detail_pctr",
              "fullrank_detail_pltr",
              "fullrank_detail_pwtr",
              "fullrank_detail_pftr",
              "fullrank_detail_plvtr",
              "fullrank_detail_pvtr",
              "fullrank_detail_psvr",
              "fullrank_detail_pcmtr",
              "fullrank_detail_pptr",
              "fullrank_detail_pwtd",
            ],
            send_common_attrs = [
              { "name": "fr_deep_ltr_trimmed_user_info", "as": "user_info_str" },
              { "name": "featureSourcePId", "as": "source_pid" },
              { "name": "sourcePidDuration", "as": "source_duration_ms" },
              { "name": "sourcePidTagId", "as": "source_tag" },
              { "name": "sourcePidAuthorId", "as": "source_aid" },
              { "name": "sourcePidHetuLevelOneList", "as": "source_hetu_tag_level1_list" },
              { "name": "sourcePidHetuLevelTwoList", "as": "source_hetu_tag_level2_list" },
              { "name": "featureSimilarUserList", "as": "similar_user_list" },
            ],
            request_type = "{{fountain_deep_ltr_request_type}}",
            partition_size = "{{fountain_deep_ltr_partition_size}}",
          ) \
        .else_() \
          .delegate_enrich(
            kess_service = "{{fountain_fullrank_deep_ltr_kess_service}}",
            recv_item_attrs = [
              {"name": "l2r", "as": "fullrank_ltr_score"},
              {"name": "ctr", "as": "fullrank_act_ctr"},
              {"name": "wtd", "as": "fullrank_act_wtd"},
              {"name": "finish_rate", "as": "fullrank_ltr_v4_fountain_finish_rate"},
              {"name": "next", "as": "fullrank_ltr_v4_fountain_next"},
              {"name": "ltr", "as": "fullrank_ltr_v4_fountain_reward"},
            ],
            timeout_ms = 100,
            send_item_attrs = [feature["name"] for feature in photo_features if feature["name"]],
            send_common_attrs = user_features_v3,
            request_type = "{{fountain_deep_ltr_request_type}}",
            partition_size = "{{fountain_deep_ltr_partition_size}}",
          ) \
        .end_() \
      .end_if_() \
  
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_attrs = [
          "fullrank_ltr_score",
          "fullrank_act_ctr",
          "fullrank_act_wtd",
          "fullrank_ltr_v4_fountain_finish_rate",
          "fullrank_ltr_v4_fountain_next",
          "fullrank_ltr_v4_fountain_reward"
        ],
        for_debug_request_only = True
      )
