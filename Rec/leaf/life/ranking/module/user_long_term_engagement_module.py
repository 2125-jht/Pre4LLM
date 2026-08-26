from ranking import CommonModule

class UserLongTermEngagementModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def user_feture(self):
      features = [
        "uId",
        "dId",
        "uFollowCount",
        "uFansCount",
        "uUploadCount",
        "uCityId",
        "uProvinceId",
        "uGender",
        "uInferYear",
        "uTrueYear",
        "uBasicAge",
        "uNetwork",
        "uIsDouyin",
        "uCityLevelNew",
        "uHourOfDay",
        "uDayOfWeek",
        "uIsLowActiveUser",
        "uIsHighPotentialUser",
        "uIsLa",
        "uAgeGenderCity",

        "uClickPidsGlobal",
        "uClickAidsGlobal",
        "uClickHetu1Global",
        "uClickHetu2Global",
        "uClickChannelsGlobal",
        "uClickPidsGlobal_LEN",
        "uLikePidsGlobal",
        "uLikeAidsGlobal",
        "uFollowPidsGlobal",
        "uFollowAidsGlobal",
        "uCommentPidsGlobal",
        "uCommentAidsGlobal",
        "uHatePidsGlobal",
        "uHateAidsGlobal",
        "uClickPidsHot",
        "uClickAidsHot",
        "uClickHetu1Hot",
        "uClickHetu2Hot",
        "uClickPidsHot_LEN",
        "uLikePidsHot",
        "uLikeAidsHot",
        "uFollowPidsHot",
        "uFollowAidsHot",
        "uCommentPidsHot",
        "uCommentAidsHot",
        "uHatePidsHot",
        "uHateAidsHot",
        "uClickPidsFountain",
        "uClickAidsFountain",
        "uLikePidsFountain",
        "uLikeAidsFountain",
        "uFollowPidsFountain",
        "uFollowAidsFountain",
        "uCommentPidsFountain",
        "uCommentAidsFountain",
        "uHatePidsFountain",
        "uHateAidsFountain",
        "uShortViewPidsHot",
        "uShortViewAidsHot",
        "uLongViewPidsHot",
        "uLongViewAidsHot",
        "uShortViewPidsFountain",
        "uShortViewAidsFountain",
        "uLongViewPidsFountain",
        "uLongViewAidsFountain",

        "uLongTermHetuLevel1topN",
        "uLongTermHetuLevel2topN",
        "uLongTermHetuLevel3topN"
      ]

      for key in ["uHotShow", "uHotClick", "uHotLike", "uHotFollow", "uHotHate"]:
          for suffix in ["5m", "1h", "1d"]:
              features.append(key + suffix + "Hot")
      
      return features

    def photo_features(self):
        features = [
            {"name": "photo_id", "as": "pId"},
            {"name": "author__id", "as": "aId"},
            {"name": "author__fans_count", "as": "aFansCount"},
            {"name": "duration_ms", "as": "pDurationMs"},
            {"name": "upload_type", "as": "pUploadType"},
            {"name": "reason", "as": "pReason"},
            {"name": "location__city_id", "as": "pCityId"},
            {"name": "location__province_id", "as": "pProvinceId"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "pHetuTagLevel1"},
            {"name": "hetu_tag_level_info__hetu_level_two", "as": "pHetuTagLevel2"},
            {"name": "hetu_tag_level_info__hetu_level_three", "as": "pHetuTagLevel3"},
            {"name": "hetu_tag_level_info__hetu_level_five", "as": "pHetuTagLevel5"},
            {"name": "hetu_tag_level_info__hetu_tag", "as": "pHetuTagLevelTag"},
            {"name": "hetu_tag_level_info__hetu_face_id", "as": "pHetuTagFaceId"},
            {"name": "hetu_tag_level_info__hetu_cluster_id", "as": "pHetuClusterId"},
            {"name": "user_hash_tag_id", "as": "pHashTagIdList"},
            {"name": "author__upload_count", "as": "aUploadCount"},
            {"name": "music", "as": "pMusic"},
            {"name": "audit_hot_high_tag_level", "as": "pAuditHotHighTagLevel"},
            {"name": "content_safety_level_with_namespace__level_hot_online", "as": "pLevelHotOnline"},
            {"name": "cascade_pctr", "as": "pMcPctr"},
            {"name": "cascade_pltr", "as": "pMcPltr"},
            {"name": "cascade_pwtr", "as": "pMcPwtr"},
            {"name": "cascade_plvtr", "as": "pMcPlvtr"},
            {"name": "cascade_psvtr", "as": "pMcPsvtr"},          
            {"name": "explore_stat__show_count", "as": "pHotShow"},
            {"name": "explore_stat__click_count", "as": "pHotClick"},
            {"name": "explore_stat__like_count", "as": "pHotLike"},
            {"name": "explore_stat__follow_count", "as": "pHotFollow"},
            {"name": "explore_stat__negative_count", "as": "pHotHate"},
            {"name": "is_picture", "as": "pIsPicture"},
            {"name": "is_follow_author", "as": "aIsFollowAuthor"},
            "pAgeHour",
            "pChannel",
            "pEmpCtrHot",
            "pEmpLtrHot",
            "pEmpWtrHot",
            "pEmpCmtrHot",
            "pEmpRealShowCountHot",
            "pEmpClickCountHot",
            "pEmpWatchTimeHot",
        ]

        for key in ["Hetu1", "Hetu2", "Hetu5", "HetuTag"]:
            for suffix in ["100n", "1000n"]:
                features.append("pShortStatShow" + key + suffix)
                features.append("pShortStatClick" + key + suffix)
                features.append("pShortStatClickRate" + key + suffix)

        return features

    def process(self) -> None:
        self.flow \
          .if_("explore_lte_rank_model == 1") \
            .explore_common_user_feature_enricher(
              global_click_max_len_attr = "explore_lte_rank_model_global_click_max_len",
              hot_click_max_len_attr = "explore_lte_rank_model_hot_click_max_len",
              user_info_attr="user_info_ptr",
              
              user_uid_attr = "uId",
              user_did_attr = "dId",
              user_follow_cnt_attr = "uFollowCount",
              user_fans_cnt_attr = "uFansCount",
              user_upload_cnt_attr = "uUploadCount",
              user_city_attr = "uCityId",
              user_province_attr = "uProvinceId",
              user_gender_attr = "uGender",
              user_infer_year_attr = "uInferYear",
              user_true_year_attr = "uTrueYear",
              user_basic_age_attr = "uBasicAge",
              user_visit_net_attr = "uNetwork",
              user_is_douyin_attr = "uIsDouyin",
              user_city_level_attr = "uCityLevelNew",
              user_age_gender_city_attr = "uAgeGenderCity",
              user_low_active_attr = "uIsLowActiveUser",
              user_high_potential_attr = "uIsHighPotentialUser",
              user_hot_low_active_attr = "uIsLa",
              context_hour_of_day_attr = "uHourOfDay",
              context_day_of_week_attr = "uDayOfWeek",
              user_longterm_hetu_level1_attr = "uLongTermHetuLevel1topN",
              user_longterm_hetu_level2_attr = "uLongTermHetuLevel2topN",
              user_longterm_hetu_level3_attr = "uLongTermHetuLevel3topN",

              global_click_pids_attr = "uClickPidsGlobal",
              global_click_aids_attr = "uClickAidsGlobal",
              global_click_hetu1_attr = "uClickHetu1Global",
              global_click_hetu2_attr = "uClickHetu2Global",
              global_click_channels_attr = "uClickChannelsGlobal",
              global_click_len_attr = "uClickPidsGlobal_LEN",
              global_like_pids_attr = "uLikePidsGlobal",
              global_like_aids_attr = "uLikeAidsGlobal",
              global_follow_pids_attr = "uFollowPidsGlobal",
              global_follow_aids_attr = "uFollowAidsGlobal",
              global_comment_pids_attr = "uCommentPidsGlobal",
              global_comment_aids_attr = "uCommentAidsGlobal",
              global_hate_pids_attr = "uHatePidsGlobal",
              global_hate_aids_attr = "uHateAidsGlobal",

              hot_click_pids_attr = "uClickPidsHot",
              hot_click_aids_attr = "uClickAidsHot",
              hot_click_hetu1_attr = "uClickHetu1Hot",
              hot_click_hetu2_attr = "uClickHetu2Hot",
              hot_click_len_attr = "uClickPidsHot_LEN",
              hot_like_pids_attr = "uLikePidsHot",
              hot_like_aids_attr = "uLikeAidsHot",
              hot_follow_pids_attr = "uFollowPidsHot",
              hot_follow_aids_attr = "uFollowAidsHot",
              hot_comment_pids_attr = "uCommentPidsHot",
              hot_comment_aids_attr = "uCommentAidsHot",
              hot_hate_pids_attr = "uHatePidsHot",
              hot_hate_aids_attr = "uHateAidsHot",

              ft_click_pids_attr = "uClickPidsFountain",
              ft_click_aids_attr = "uClickAidsFountain",
              ft_like_pids_attr = "uLikePidsFountain",
              ft_like_aids_attr = "uLikeAidsFountain",
              ft_follow_pids_attr = "uFollowPidsFountain",
              ft_follow_aids_attr = "uFollowAidsFountain",
              ft_comment_pids_attr = "uCommentPidsFountain",
              ft_comment_aids_attr = "uCommentAidsFountain",
              ft_hate_pids_attr = "uHatePidsFountain",
              ft_hate_aids_attr = "uHateAidsFountain",

              hot_sv_pids_attr = "uShortViewPidsHot",
              hot_sv_aids_attr = "uShortViewAidsHot",
              hot_lv_pids_attr = "uLongViewPidsHot",
              hot_lv_aids_attr = "uLongViewAidsHot",
              ft_sv_pids_attr = "uShortViewPidsFountain",
              ft_sv_aids_attr = "uShortViewAidsFountain",
              ft_lv_pids_attr = "uLongViewPidsFountain",
              ft_lv_aids_attr = "uLongViewAidsFountain",

              user_count_action_attr = "cnt_",
            ) \
            .explore_common_item_feature_enricher(
              user_info_attr="user_info_ptr",
              upload_time_attr = "upload_time",
              explore_stat_real_show_count_attr = "explore_stat__real_show_count",
              explore_stat_click_count_attr = "explore_stat__click_count",
              explore_stat_like_count_attr = "explore_stat__like_count",
              explore_stat_comment_count_attr = "explore_stat__comment_count",
              explore_stat_follow_count_attr = "explore_stat__follow_count",
              explore_stat_view_length_sum_attr = "explore_stat__view_length_sum",
              hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
              hetu_level_two_attr = "hetu_tag_level_info__hetu_level_two",
              hetu_level_three_attr = "hetu_tag_level_info__hetu_level_three",
              hetu_level_four_attr = "hetu_tag_level_info__hetu_level_four",
              hetu_level_five_attr = "hetu_tag_level_info__hetu_level_five",
              hetu_level_tag_attr = "hetu_tag_level_info__hetu_tag",

              item_age_hour_attr = "pAgeHour",
              item_channel_attr = "pChannel",
              item_emp_ctr_hot_attr = "pEmpCtrHot",
              item_emp_ltr_hot_attr = "pEmpLtrHot",
              item_emp_wtr_hot_attr = "pEmpWtrHot",
              item_emp_cmtr_hot_attr = "pEmpCmtrHot",
              item_emp_real_show_count_hot_attr = "pEmpRealShowCountHot",
              item_emp_click_count_hot_attr = "pEmpClickCountHot",
              item_emp_watch_time_hot_attr = "pEmpWatchTimeHot",
              short_stat_list_attr = "short_stat_list_"
            ) \
            .delegate_enrich(
              kess_service = "{{explore_lte_rank_kess_service}}",
              recv_item_attrs = [
                {"name": "ctr", "as": "lte_ctr"},
                {"name": "ltr", "as": "lte_ltr"},
              ],
              timeout_ms = 100,
              send_item_attrs = self.photo_features(),
              send_common_attrs = self.user_feture(),
              request_type = "default",
              partition_size = "{{lte_rank_predict_partition_size}}",
            ) \
          .end_if_()

    def post_process(self) -> None:
      self.flow \
        .log_debug_info(
          common_attrs = self.user_feture(),
            for_debug_request_only = True
        )