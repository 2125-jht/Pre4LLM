from retrieval.retrieval_module import RetrievalModule

class ExploreTriggerPromptRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def input_feature(self):
      features = [
        "uId",
        "dId",
        # 实时行为序列
        "uLikePids",
        "uFollowAids",
        "uClickPids",
        # 行为计数
        "uFollowCount",
        "uFansCount",
        "uUploadCount",
        "uUploadRate",
        # 地理位置
        "uCityId",
        "uProvinceId",
        # 性别年龄
        "uGender",
        "uTrueGender",
        "uInferYear",
        "uTrueYear",
        "uBasicAge",
        # 其他
        "uNetwork",
        "uCityLevelNew"
        "uIsDouyin",
        "uTown",
        "uCommunityType",
        "uAgeGenderCity",
        "uAppList",
        "uHourOfDay",
        "uDayOfWeek",
        # 长期河图统计
        "uLongTermHetuLevel1topN",
        "uLongTermHetuLevel2topN",
        "uLongTermHetuLevel3topN",
        # 行为序列长版, 参与 target attention
        {"name": "click_list", "as": "uClickPidsHot"},
        "uLongViewPidsHot",
        "uPlayViewPidsGlobal",
        "uLikePidsHot",
        "uFollowPidsHot",
        "like_list",
        "follow_list",
        "forward_list",
        "comment_list",
        "collect_list",
        # 播放序列，用于召回 leaf 抽取 trigger
        {"name": "videoPlayingPid", "as": "playstat_pids"},
        {"name": "profile_v1_click_trigger_aids", "as": "playstat_aids"},
        "playstat_durations",
        "playstat_playtimes",
        "playstat_hetu1s",
        "playstat_hetu2s",
      ]
      return features

  def process(self) -> None:
    self.flow \
      .explore_common_user_feature_enricher(
        user_info_attr = "user_info_ptr",
        user_uid_attr = "uId",
        user_did_attr = "dId",
        # 实时行为序列
        user_click_pids_attr = "uClickPids",
        user_like_pids_attr = "uLikePids",
        user_follow_pids_attr = "uFollowAids",
        # 行为计数
        user_exp_follow_count_attr = "uFollowCount",
        user_upload_cnt_attr = "uUploadCount",
        user_fans_cnt_attr = "uFansCount",
        user_upload_rate_attr = "uUploadRate",
        # 地理
        user_city_attr = "uCityId",
        user_province_attr = "uProvinceId",
        # 性别年龄
        user_gender_attr = "uGender",
        user_infer_gender_int_attr = "uInferGender",
        user_infer_year_int_attr = "uInferYear",
        user_true_year_int_attr = "uTrueYear",
        user_basic_age_attr = "uBasicAge",
        # 其他
        user_visit_net_attr = "uNetwork",
        user_city_level_attr = "uCityLevelNew",
        user_is_douyin_attr = "uIsDouyin",
        # hetu
        user_longterm_hetu_level1_attr = "uLongTermHetuLevel1topN",
        user_longterm_hetu_level2_attr = "uLongTermHetuLevel2topN",
        user_longterm_hetu_level3_attr = "uLongTermHetuLevel3topN",
        #
        user_request_town_attr = "uTown",
        user_request_community_type_attr = "uCommunityType",
        user_age_gender_city_attr = "uAgeGenderCity",
        user_app_package_attr = "uAppList",
        context_hour_of_day_attr = "uHourOfDay",
        context_day_of_week_attr = "uDayOfWeek",
        # 行为序列特征
        hot_lv_pids_attr = "uLongViewPidsHot",
        global_pv_pids_attr = "uPlayViewPidsGlobal",
        hot_like_aids_attr = "uLikePidsHot",
        hot_follow_pids_attr = "uFollowPidsHot"
      ) \
      .delegate_retrieve(
        kess_service = "{{retr_kess_name}}",
        timeout_ms = 100,
        shard_num = 1,
        reason = self.reason,
        request_type = "default",
        request_num = "{{retr_total_num}}",
        send_common_attrs = self.input_feature(),
        send_browse_set = False,
        send_common_attrs_in_request = False
      )
