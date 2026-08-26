from retrieval import CommonModule

class SplashUserInfoParsingModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .parse_protobuf_from_string(
        input_attr="userInfo",
        output_attr="userInfoPb",
        class_name="ks.reco.UserInfo",)  \
      .enrich_with_protobuf(
        from_extra_var = "userInfoPb",
        attrs = [
          dict(name = "uIsBigG", path = "feature_collection.features", sample_attr_name = "uIsBigG", skip_unset_field = True),
          dict(name = "location_city_level_v2", path = "location.city_level"),
          dict(name="friendAids", path="friend_info_v2.bid_follow_list.friend_id"),
          dict(name="user_risk_level", path="feature_collection.risk_level"),
          dict(name="user_fountain_play_id_list", path="fountain_reco_user_profile.video_play_stat.photo_id"),
          dict(name="user_fountain_play_time_list", path="fountain_reco_user_profile.video_play_stat.playing_time"),
          dict(name="user_fountain_play_hetu_list", path="fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_one", repeat_limit={"fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_one": 1}, repeat_align=True),
          dict(name="fountain_hate_list_timestamps", path="fountain_reco_user_profile.hate_list.time_ms"),
        ]
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
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "user_fountain_play_id_list", "as": "photo_id_list_attr"},
          {"name": "user_fountain_play_hetu_list", "as": "hetu_list_attr"},
          {"name": "source_hetu_level_one", "as": "hetu_tag_attr"},
          {"name": "fountain_splash_trigger_play_ms_threshold", "as": "play_time_ms_threshold_attr"},
          {"name": "user_fountain_play_time_list", "as": "play_time_ms_list_attr"},
          {"name": "fountain_splash_enable_append_other_tags", "as": "enable_append_other_tags"}
        ],
        export_common_attr = [
          {"name": "hetu1_filtered_photo_id_list", "as": "hetu1_filtered_play_photo_id_list"},
        ],
        function_name = "FilterActionList",
        class_name = "ExploreLightFunctionSetV2",
        skip = "{{return skip_fountain_splash_filter_action_list == 1 or user_fountain_play_id_list == nil}}"
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
      )