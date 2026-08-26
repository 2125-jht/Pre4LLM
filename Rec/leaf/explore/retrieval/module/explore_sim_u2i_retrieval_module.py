from retrieval.retrieval_module import RetrievalModule


USER_MAX_HETU_ITER = 4

class ExploreSimU2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .explore_common_user_feature_enricher(
        user_info_attr = "user_info_ptr",
        user_uid_attr = "uId",
        user_did_attr = "dId",
        user_province_attr =  "uProvinceId",
        user_city_attr = "uCityId",
        user_click_pids_attr = "uClickPids",
        user_like_pids_attr = "uLikePids",
        user_follow_pids_attr = "uFollowAids",
        user_gender_attr = "uGender",
        user_infer_gender_attr = "uInferGender",
        user_ture_gender_attr = "uTrueGender",
        user_basic_age_attr = "uAgeSeg",
        user_app_package_attr = "uAppList",
        user_pic_play_list_attr = "uPlayPics",  
        user_pic_follow_pids_attr = "uFollowListpid",
        user_pic_play_aid_list_attr = "featPicRecentPlayAidList",
        user_pic_play_tag_list_attr = "featPicRecentPlayTagList",
        user_pic_play_ts_list_attr = "featPicRecentPlayTsList",
        user_pic_play_time_list_attr = "featPicRecentPlayTimeList",
        user_video_play_list_attr = "featRecentPlayPidList",
        user_video_play_aid_list_attr = "featRecentPlayAidList",
        user_video_play_tag_list_attr = "featRecentPlayTagList",
        user_video_play_ts_list_attr = "featRecentPlayTsList",
        user_video_play_time_list_attr = "featRecentPlayTimeList",
      ) \
      .explore_pic_colossus_stat(
        colossus_attr_name = "colossus_resp_v2",
        user_info_ptr_attr = "user_info_ptr",
        active_days_gt_5min_rate = "active_days_gt_5min_rate",
        # output attrs
        save_pic_play_list = "pic_play_list",
        save_pic_like_list = "pic_like_list",
        save_pic_follow_list = "pic_follow_list",
        save_pic_comment_list = "pic_comment_list",
        save_pic_comment_aid_list = "pic_comment_aid_list",
        save_pic_hetu_l1_cnt = "pic_hetu_l1_cnt",
        save_photo_hetu_l1_cnt = "photo_hetu_l1_cnt",
        save_photo_like_list = "photo_like_list",
        save_photo_follow_list = "photo_follow_list",
        save_photo_comment_list = "photo_comment_list",
        # sim retr input attrs
        explore_only_hot_tab = "{{explore_only_hot_tab}}",
        enable_user_photo_sim_hetu_attr = "{{enable_explore_sim_u2i_retrieval}}",
        photo_sim_action_weight_str = "{{explore_sim_u2i_action_weight_str}}",
        photo_sim_hetu_count = "{{explore_sim_u2i_hetu_count}}",
        photo_sim_exploit_count = "{{explore_sim_u2i_exploit_hetu_count}}",
        hetu_l1_white_list_str = "{{explore_sim_u2i_hetu_l1_white_list}}",
        photo_sim_reward_window_size = "{{explore_sim_u2i_reward_window_size}}",
        enable_photo_sim_hetu_blacklist = "{{enable_explore_sim_u2i_hetu_blacklist}}",
        photo_sim_hetu_short_view_rate_thd = "{{explore_sim_u2i_short_view_rate_thd}}",
        photo_sim_hetu_blacklist_min_play_cnt = "{{explore_sim_u2i_blacklist_min_play_cnt}}",
        enable_photo_sim_reward_channel_filter = "{{enable_explore_sim_u2i_reward_channel_filter}}",
        photo_sim_reward_channel_whitelist = "{{explore_sim_u2i_reward_channel_whitelist}}",
        enable_sim_u2i_watch_time_user_filter = "{{enable_explore_sim_u2i_watch_time_user_filter}}",
        sim_u2i_watch_time_user_threshold = "{{explore_sim_u2i_watch_time_user_threshold}}",
      ) \
      .set_attr_value(
        common_attrs=[
          {
            "name": "reason",
            "type": "int",
            "value": self.reason
          }
        ],
        skip = "{{skip_personal_quota}}"
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_sim_u2i_retr_total_num", "as": "retr_size"},
          "reason_ratio_map_attr",
          "reason"
        ],
        export_common_attr = [
          {"name": "retr_size", "as": "explore_sim_u2i_retr_total_num"}
        ],
        function_name = "DynamicRetrQuota",
        class_name = "ExploreLightFunctionSetV2",
        skip = "{{skip_personal_quota}}"
      ) \
      .delegate_retrieve(
        kess_service = "{{explore_sim_u2i_retr_kess_name}}",
        timeout_ms = 100,
        shard_num = 1,
        reason = self.reason,
        request_num = "{{explore_sim_u2i_retr_total_num}}",
        send_common_attrs=[
          "uId", "dId", "uClickPids", "uLikePids", "uFollowAids", "uFollowListpid", "uGender", "uInferGender",
          "uTrueGender", "uAgeSeg", "uProvinceId", "uCityId", "uAppList", "uPlayPics",
          {"name": "uPlayPics", "as": "featPicRecentPlayPidList"}, 
          "featPicRecentPlayAidList", "featPicRecentPlayTagList",
          "featPicRecentPlayTsList", "featPicRecentPlayTimeList",
          "featRecentPlayPidList", 
          "featRecentPlayAidList", "featRecentPlayTagList",
          "featRecentPlayTsList", "featRecentPlayTimeList",
          {"name": "photo_hetu_l1_cnt", "as": "featPicColHetuListV2"}, 
          {"name": "photo_like_list", "as": "featPicColLkPids"}, 
          {"name": "photo_follow_list", "as": "featPicColFlPids"}, 
          {"name": "photo_comment_list", "as": "featPicColCmtPids"},
          "selected_photo_sim_hetu_list",
          "uIsExploreTnuCrowdUser",
        ] + self.get_hetu_features(),
      )
    
  def get_hetu_features(self):
    features = ["selected_photo_sim_hetu_list"]
    for prefix in ["pHetuL1ActList", "pHetuL1ActPicList"]:
        for fea_type in ["Pids", "Aids", "Ts", "Ptime", "Action"]:
            for suffix in range(USER_MAX_HETU_ITER):
                features.append(prefix + fea_type + "__photo_Top" + str(suffix+1))
    return features
