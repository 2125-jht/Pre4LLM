from retrieval import CommonModule

class UserInfoParsingModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .parse_protobuf_from_string(
        input_attr="userInfo",
        output_attr="userInfoPb",
        class_name="ks.reco.UserInfo",) \
      .enrich_with_protobuf(
        from_extra_var = "userInfoPb",
        attrs = [
          dict(name = "uIsBigG", path = "feature_collection.features", sample_attr_name = "uIsBigG", skip_unset_field = True),
        ],
      ) \
      .gen_common_attr_by_lua(
        attr_map = {
          "is_live_big_g_user": "uIsBigG or 0",
          "is_live_paying_user": "uUserKuaishouLivePayTag ~= nil and uUserKuaishouLivePayTag >= 0 and uUserKuaishouLivePayTag <= 10 and 1 or 0",
          "is_live_high_paying_user": "uUserKuaishouLivePayTag ~= nil and uUserKuaishouLivePayTag >= 3 and uUserKuaishouLivePayTag <= 6 and 1 or 0",
        },
      ) \
      .gen_common_attr_by_lua(
        attr_map = {
          "_ABTEST_USER_TAG_NAMES_": "{\"uIsBigG\", \"uIsLivePayingUser\", \"uIsLiveHighPayingUser\"}",
          "_ABTEST_USER_TAG_VALUES_": "{tostring(is_live_big_g_user), tostring(is_live_paying_user), tostring(is_live_high_paying_user)}",
        },
      ) \
      .enrich_with_protobuf(
        from_extra_var = "userInfoPb",
        attrs = [
          dict(path="browsed_photo_ids", name="user_browsed_photo_ids"),
          dict(path="user_profile_v1.video_playing_stat.photo_id", name="video_playing_stat_pid_list"),
          dict(path="user_profile_v1.video_playing_stat.author_id", name="video_playing_stat_aid_list"),
          dict(path="user_profile_v1.video_playing_stat.video_duration", name="video_playing_stat_duration_list"),
          dict(path="user_profile_v1.video_playing_stat.playing_time", name="video_playing_stat_play_time_list"),
          dict(name="video_playing_stat_timestamp_list", path="user_profile_v1.video_playing_stat.client_timestamp"),
          dict(name="video_playing_stat_page_list", path="user_profile_v1.video_playing_stat.page"),
          dict(name="video_playing_stat_hetu_list", path="user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_one", repeat_limit={"user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_one": 1}, repeat_align=True),
          dict(name="video_playing_stat_hetu2_list", path="user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_two", repeat_limit={"user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_two": 1}, repeat_align=True),
          dict(name="video_playing_stat_hetu3_list", path="user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_three", repeat_limit={"user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_three": 1}, repeat_align=True),
          dict(name="video_playing_stat_hetu4_list", path="user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_four", repeat_limit={"user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_four": 1}, repeat_align=True), 
          dict(name="profile_v1_collect_pid_list", path="user_profile_v1.collect_list.photo_id"),
          dict(name="user_risk_level", path="feature_collection.risk_level"),
          dict(name="user_fountain_play_id_list", path="fountain_reco_user_profile.video_play_stat.photo_id"),
          dict(name="user_fountain_play_aid_list", path="fountain_reco_user_profile.video_play_stat.author_id"),
          dict(name="user_fountain_play_time_list", path="fountain_reco_user_profile.video_play_stat.playing_time"),
          dict(name="user_fountain_play_duration_list", path="fountain_reco_user_profile.video_play_stat.video_duration"),
          dict(name="user_fountain_play_timestamp_list", path="fountain_reco_user_profile.video_play_stat.client_timestamp"),
          dict(name="user_fountain_play_page_list", path="fountain_reco_user_profile.video_play_stat.page"),
          dict(name="user_fountain_play_hetu_list", path="fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_one", repeat_limit={"fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_one": 1}, repeat_align=True),
          dict(name="user_fountain_play_hetu2_list", path="fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_two", repeat_limit={"fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_two": 1}, repeat_align=True),
          dict(name="user_fountain_play_hetu3_list", path="fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_three", repeat_limit={"fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_three": 1}, repeat_align=True),
          dict(name="user_fountain_play_hetu4_list", path="fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_four", repeat_limit={"fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_four": 1}, repeat_align=True),
          dict(name="fountain_forward_pid_list", path="fountain_reco_user_profile.forward_list.photo_id"),
          dict(name="user_fountain_forward_aid_list", path="fountain_reco_user_profile.forward_list.author_id"),
          dict(name="user_fountain_follow_aid_list", path="fountain_reco_user_profile.follow_list.author_id"),
          dict(name="user_fountain_follow_pid_list", path="fountain_reco_user_profile.follow_list.photo_id"),
          dict(name="user_fountain_like_aid_list", path="fountain_reco_user_profile.like_list.author_id"),
          dict(name="user_fountain_comment_aid_list", path="fountain_reco_user_profile.comment_list.author_id"),
          dict(name="follow_timestamps", path="fountain_reco_user_profile.follow_list.time_ms"),
          dict(name="follow_aids", path="user_profile_v1.follow_list.author_id"),
          dict(name="follow_pids", path="user_profile_v1.follow_list.photo_id"),
          dict(name="long_term_interest_authors", path="feature_collection.features_map.value", sample_attr_name="uLongViewAuthorList"),
          dict(name="user_follow_type", path="feature_collection.features_map.value", sample_attr_name="uFollowPeopleType"),
          dict(name="upload_count", path="upload_count"),
          dict(name="fans_count", path="fans_count"),
          dict(name="location_city_level_v2", path="location.city_level"),
          dict(name="upload_rate", path="upload_rate"),
          dict(name="gender", path="gender", skip_unset_field=True),
          dict(name="true_gender", path="true_gender", skip_unset_field=True),
          dict(name="infer_gender", path="infer_gender", skip_unset_field=True),
          dict(name="true_year", path="true_year", skip_unset_field=True),
          dict(name="infer_year", path="infer_year", skip_unset_field=True),
          dict(name="is_douyin", path="is_douyin"),
          dict(name="featrueUserLongTermHetu1Id", path="user_interest_profile.hetu_level_one_long_term_id"),
          dict(name="featrueUserLongTermHetu1Score", path="user_interest_profile.hetu_level_one_long_term_score"),
          dict(name="featrueUserLongTermHetu2Id", path="user_interest_profile.hetu_level_two_long_term_id"),
          dict(name="featrueUserLongTermHetu2Score", path="user_interest_profile.hetu_level_two_long_term_score"),
          dict(name="featrueUserLongTermHetu3Id", path="user_interest_profile.hetu_level_three_long_term_id"),
          dict(name="featrueUserLongTermHetu3Score", path="user_interest_profile.hetu_level_three_long_term_score"),
          dict(name="uClickPids", path="realtime_click_list"),
          dict(name="uLikePids", path="realtime_like_list"),
          dict(name="uFollowAids", path="realtime_follow_list"),
          dict(name="uLikePidsFountain", path="fountain_reco_user_profile.like_list.photo_id"),
          dict(name="uFollowAidsFountain", path="fountain_reco_user_profile.follow_list.author_id"),
          dict(name="uLikePidsGlobal", path="user_profile_v1.like_list.photo_id"),
          dict(name="uLikeAidsGlobal", path="user_profile_v1.like_list.author_id"),
          dict(name="friendAids", path="friend_info_v2.bid_follow_list.friend_id"),
          dict(name="hate_list", path="user_profile_v1.hate_list.photo_id"),
          dict(name="hate_list_timestamps", path="user_profile_v1.hate_list.time_ms"),
          dict(name="fountain_hate_list_timestamps", path="fountain_reco_user_profile.hate_list.time_ms"),
          dict(name="followAids", path="follow_list.user.id"),
          dict(name="basic_info_age_segment_v2", path="basic_info.age_segment_v2"),
          dict(name="basic_info_age_segment", path="basic_info.age_segment"),
          dict(name="uProvinceId", path="request_location.province_id"),
          dict(name="uCityId", path="request_location.city_id"),
          dict(name="gamora_hetu_adjust_history_list", path="feature_collection.gamora_hetu_adjust_history"),
          dict(name="opt_card_like_list", path="feature_collection.features", sample_attr_name="uRecoOptCardActionLikeList"),
          dict(name="opt_card_dis_like_list", path="feature_collection.features", sample_attr_name="uRecoOptCardActionDislikeList"),
          dict(name="find_user_active_degree", path="find_user_active_degree")
        ]
      ) \
      .explore_common_user_feature_enricher(
        user_info_attr = "userInfoPb",
        user_app_package_attr = "uAppList",
        user_video_play_list_attr = "uRecent50PlayPidList",
        user_video_play_aid_list_attr = "uRecent50PlayAidList",
        user_video_play_tag_list_attr = "uRecent50PlayTagList",
        user_video_play_ts_list_attr = "uRecent50PlayTsList",
        user_video_play_time_list_attr = "uRecent50PlayTimeList",
        user_pic_follow_pids_attr = "uRecent20FollowPidList",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "userInfoPb", "as": "user_info_ptr"},
        ],
        export_common_attr = [
          "uIsExploreTnuCrowdUser"
        ],
        function_name = "IsTnuCrowdUser",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .gen_common_attr_by_lua(
        attr_map = {
          "is_teenager": "basic_info_age_segment_v2 ~= nil and (basic_info_age_segment_v2 == 1 or basic_info_age_segment_v2 == 2)"
        },
      ) \
      .if_("enable_fountain_get_workday_hour_gender == 1") \
        .enrich_attr_by_light_function(
          export_common_attr = [
            "is_work_day"
          ],
          function_name = "IsWorkDay",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .enrich_attr_by_light_function(
          export_common_attr = [
            "request_hour"
          ],
          function_name = "GetHour",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .if_("uGender ~= nil and #uGender > 0") \
          .gen_common_attr_by_lua(
            attr_map={
              "user_gender": "uGender[1]",
            }
          ) \
        .end_() \
      .end_() \
      .if_("enable_fountain_koc_htr == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_hate_list_timestamps", "as": "hate_time_ms_list"},
            {"name": "fountain_koc_htr_time_gap_minute", "as": "time_gap_minute"},
          ],
          export_common_attr = [
            {"name": "recent_hate_count", "as": "fountain_recent_hate_count"},
          ],
          function_name = "CalcFountainRecentHateCount",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_fountain_get_user_message_cnt_today_from_redis == 1") \
        .gen_common_attr_by_lua(
          attr_map = {
            "cur_date_yyMMdd": "os.date('%y%m%d')"
          }
        ) \
        .str_format(
          format_string = 'ssm_%d_%s',
          input_attrs = ['_USER_ID_', 'cur_date_yyMMdd'],
          output_attr = 'user_msg_cnt_ssm_today_redis_key'
        ) \
        .str_format(
          format_string = 'gsm_%d_%s',
          input_attrs = ['_USER_ID_', 'cur_date_yyMMdd'],
          output_attr = 'user_msg_cnt_gsm_today_redis_key'
        ) \
        .get_common_attr_from_redis(
          cluster_name = "followNotifyUA",
          timeout_ms = 10,
          redis_params = [
            {
              "redis_key": "{{user_msg_cnt_ssm_today_redis_key}}",
              "output_attr_name": "user_msg_cnt_ssm_today",
              "output_attr_type": "int"
            },
            {
              "redis_key": "{{user_msg_cnt_gsm_today_redis_key}}",
              "output_attr_name": "user_msg_cnt_gsm_today",
              "output_attr_type": "int"
            }
          ],
          is_async = True,
        ) \
      .end_() \
      .gen_common_attr_by_lua(
        attr_map = {
          "bid_follow_num": "#(friendAids or {})",
          "u_share_num_30d": "uToUserShareSendNum30dKV or 0",
          "u_message_active_degree": "uMessageActiveDegreeCode or 0",
          "user_msg_cnt_ssm_today": "user_msg_cnt_ssm_today or 0",
          "user_msg_cnt_gsm_today": "user_msg_cnt_gsm_today or 0",
          "u_inside_share_active_degree_detail_code": "uInsideShareActiveDegreeDetailCode or 0"
        }
      ) \
      .if_("enable_fountain_user_long_view_pid_list_compute == 1") \
        .explore_common_user_feature_enricher(
          user_info_attr = "userInfoPb",
          user_action_list_long_version_attr = "enable",
          user_long_view_pids_attr = "user_long_view_pid_list"
        ) \
      .end_() \
      .if_("enable_fountain_user_is_low_interact_compute == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "colossus_label_list",
            "colossus_timestamp_list",
            {"name": "fountain_user_is_low_interact_positive_interact_thres", "as": "positive_interact_thres"},
            {"name": "fountain_user_is_low_interact_comment_thres", "as": "comment_thres"},
            {"name": "fountain_user_is_low_interact_follow_thres", "as": "follow_thres"},
            {"name": "fountain_user_is_low_interact_forward_thres", "as": "forward_thres"},
            {"name": "fountain_user_is_low_interact_colossus_length_thres", "as": "colossus_length_thres"},
          ],
          export_common_attr = [
            "user_is_low_interact",
          ],
          function_name = "CalUserIsLowInteract",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()
