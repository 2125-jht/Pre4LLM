from retrieval.retrieval_module import RetrievalModule

class SimU2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .explore_pic_colossus_stat(
        colossus_attr_name = "colossus_resp_v2",
        user_info_ptr_attr = "userInfoPb",
        # output attrs
        save_photo_hetu_l1_cnt = "photo_hetu_l1_cnt",
        save_photo_like_list = "photo_like_list",
        save_photo_follow_list = "photo_follow_list",
        save_photo_comment_list = "photo_comment_list",
        # sim retr input attrs
        enable_user_photo_sim_hetu_attr = True,
        photo_sim_action_weight_str = "{{fountain_sim_u2i_action_weight_str}}",
        photo_sim_hetu_count = "{{fountain_sim_u2i_hetu_count}}",
        photo_sim_exploit_count = "{{fountain_sim_u2i_exploit_hetu_count}}",
        photo_sim_reward_window_size = "{{fountain_sim_u2i_reward_window_size}}",
        enable_photo_sim_hetu_blacklist = "{{enable_fountain_sim_u2i_hetu_blacklist}}",
        photo_sim_hetu_short_view_rate_thd = "{{fountain_sim_u2i_short_view_rate_thd}}",
        photo_sim_hetu_blacklist_min_play_cnt = "{{fountain_sim_u2i_blacklist_min_play_cnt}}",
        enable_photo_sim_reward_channel_filter = "{{enable_fountain_sim_u2i_reward_channel_filter}}",
        photo_sim_reward_channel_whitelist = "{{fountain_sim_u2i_reward_channel_whitelist}}",
      ) \
      .delegate_retrieve(
        kess_service = "{{fountain_sim_u2i_retr_kess_name}}",
        timeout_ms = 100,
        shard_num = 1,
        reason = self.reason,
        request_type = "default",
        request_num = "{{fountain_sim_u2i_retr_total_num}}",
        send_common_attrs=[
          {"name": "featureUId", "as": "uId"},
          {"name": "featureDeviceId", "as": "dId"},
          "uClickPids",
          "uLikePids",
          "uFollowAids",
          "uProvinceId", 
          "uCityId",
          "uAppList",
          {"name": "uRecent20FollowPidList",  "as": "uFollowListpid"},
          {"name": "gender",                  "as": "uGender"}, 
          {"name": "infer_gender",            "as": "uInferGender"},
          {"name": "true_gender",             "as": "uTrueGender"}, 
          {"name": "basic_info_age_segment",  "as": "uAgeSeg"},
          {"name": "uRecent50PlayPidList",    "as": "featRecentPlayPidList"},
          {"name": "uRecent50PlayAidList",    "as": "featRecentPlayAidList"},
          {"name": "uRecent50PlayTagList",    "as": "featRecentPlayTagList"},
          {"name": "uRecent50PlayTsList",     "as": "featRecentPlayTsList"},
          {"name": "uRecent50PlayTimeList",   "as": "featRecentPlayTimeList"},
          {"name": "photo_hetu_l1_cnt",       "as": "featPicColHetuListV2"}, 
          {"name": "photo_like_list",         "as": "featPicColLkPids"}, 
          {"name": "photo_follow_list",       "as": "featPicColFlPids"}, 
          {"name": "photo_comment_list",      "as": "featPicColCmtPids"},
          "selected_photo_sim_hetu_list",
          # source photo attr
          {"name": "featureSourcePId",        "as": "uSourcePId"},
          {"name": "SourcePhotoAuthorId",     "as": "uSourcePAId"},
          {"name": "source_hetu_level_one",   "as": "uSourcePHetuL1"},
          {"name": "source_hetu_level_two",   "as": "uSourcePHetuL2"},
          {"name": "source_hetu_level_three", "as": "uSourcePHetuL3"},
          {"name": "source_hetu_face_ids",    "as": "uSourcePHetuIP"},
          {"name": "source_hetu_level_five",  "as": "uSourcePHetuL5"},
        ] + [prefix + fea_type + "__photo_Top" + str(suffix+1) 
              for suffix in range(4) 
              for fea_type in ["Pids", "Aids", "Ts", "Ptime", "Action"] 
              for prefix in ["pHetuL1ActList"]],
      )
