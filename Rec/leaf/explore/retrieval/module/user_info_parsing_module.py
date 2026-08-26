from retrieval import CommonModule

class UserInfoParsingModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .if_("explore_enable_sim_high_active_user_fill_action_list == 1", to_be_delete = "date=2025-02-15;committer=huzongyao") \
        .delegate_retrieve(
          kess_service = "grpc_LLMU2URetrAndFixedMiddleLowUserActionList",
          request_type = "default",
          timeout_ms = 15,
          recv_common_attrs = ["llm_sim_user_list"],
          send_common_attrs = [{"name": "explore_fill_action_list_llm_u2u_similar_user_num", "as": "llm_u2u_sim_top_k"}],
        ) \
        .if_("llm_sim_user_list ~= nil") \
          .gen_common_attr_by_lua(
            attr_map = {
              "llm_sim_user_list_1": "#llm_sim_user_list >= 1 and llm_sim_user_list[1] or nil",
              "llm_sim_user_list_2": "#llm_sim_user_list >= 2 and llm_sim_user_list[2] or nil"
            }
          ) \
          .if_("llm_sim_user_list_1 ~= nil") \
            .fetch_user_info(
              kess_service = "grpc_recoUserProfileNew_recoUserProfileRpcService",
              biz_name= "SimUserExplore",
              save_to_common_attr = "llm_sim_user_list_1_user_info_str",
              user_id = "{{llm_sim_user_list_1}}",
              timeout_ms = 50
            ) \
          .end_() \
          .if_("llm_sim_user_list_2 ~= nil") \
            .fetch_user_info(
              kess_service = "grpc_recoUserProfileNew_recoUserProfileRpcService",
              biz_name= "SimUserExplore",
              save_to_common_attr = "llm_sim_user_list_2_user_info_str",
              user_id = "{{llm_sim_user_list_2}}",
              timeout_ms = 50
            ) \
            .parse_protobuf_from_string(
              input_attr = "llm_sim_user_list_2_user_info_str",
              output_attr = "llm_sim_user_list_2_user_info",
              class_name = "ks.reco.UserInfo",
            ) \
          .end_() \
          .if_("llm_sim_user_list_1_user_info_str ~= nil") \
            .parse_protobuf_from_string(
              input_attr = "llm_sim_user_list_1_user_info_str",
              output_attr = "llm_sim_user_list_1_user_info",
              class_name = "ks.reco.UserInfo",
            ) \
          .end_() \
          .enrich_attr_by_light_function(
            import_common_attr = [
              "user_info_ptr",
              "llm_sim_user_list_1_user_info",
              "llm_sim_user_list_2_user_info",
              "explore_find_action_seq_num_thres",
              "explore_user_action_seq_num_thres",
              "explore_user_click_list_length_thres",
              "explore_user_playing_pid_list_length_thres"
            ],
            function_name = "MergeUserInfoPtr",
            class_name = "ExploreLightFunctionSetV2"
          ) \
        .end_() \
      .end_() \
      .enrich_with_protobuf(
        from_extra_var = "user_info_ptr",
        attrs = [
          dict(name="user_gender", path="gender"),
          dict(name="user_age_segment", path="basic_info.age_segment"),
          dict(name="click_list", path="user_profile_v1.click_list.photo_id"),
          dict(name="like_list", path="user_profile_v1.like_list.photo_id"),
          dict(name="follow_list", path="user_profile_v1.follow_list.photo_id"),
          dict(name="forward_list", path="user_profile_v1.forward_list.photo_id"),
          dict(name="comment_list", path="user_profile_v1.comment_list.photo_id"),
          dict(name="collect_list", path="user_profile_v1.collect_list.photo_id"),
          dict(name="download_list", path="user_profile_v1.download_video_list.photo_id"),
          dict(name="profile_enter_list", path="user_profile_v1.profile_enter_list.photo_id"),
          dict(name="search_click_list", path="user_profile_v1.search_click_photo_list.photo_id"),
          dict(name="search_click_list_timestamps", path="user_profile_v1.search_click_photo_list.time_ms"),
          dict(name="search_play_pid_list", path="user_profile_v1.search_photo_play_list.photo_id"),
          dict(name="search_play_timestamp_list", path="user_profile_v1.search_photo_play_list.timestamp"),
          dict(name="search_play_video_duration_list", path="user_profile_v1.search_photo_play_list.video_duration"),
          dict(name="search_play_time_list", path="user_profile_v1.search_photo_play_list.play_duration"),
          dict(name="search_query_list_keyword", path="user_profile_v1.search_query_list.keyword"),
          dict(name="search_query_list_timestamp", path="user_profile_v1.search_query_list.timestamp"),
          dict(name="hate_list", path="user_profile_v1.hate_list.photo_id"),
          dict(name="hate_list_timestamps", path="user_profile_v1.hate_list.time_ms"),
          dict(name="hate_list_page_types", path="user_profile_v1.hate_list.page_type"),
          dict(name="hate_list_reasons", path="user_profile_v1.hate_list.hate_reason"),
          dict(name="user_risk_level", path="feature_collection.risk_level"),
          dict(name="basic_info_gender_v2", path="basic_info.gender_v2"),
          dict(name="basic_info_age_segment_v2", path="basic_info.age_segment_v2"),
          dict(name="location_city_level_v2", path="location.city_level"),
          dict(name="profile_v1_click_trigger_aids", path="user_profile_v1.video_playing_stat.author_id"),
          dict(name="click_aids", path="user_profile_v1.click_list.author_id"),
          dict(name="like_aids", path="user_profile_v1.like_list.author_id"),
          dict(name="follow_aids", path="user_profile_v1.follow_list.author_id"),
          dict(name="forward_aids", path="user_profile_v1.forward_list.author_id"),
          dict(name="comment_aids", path="user_profile_v1.comment_list.author_id"),
          dict(name="profile_enter_aids", path="user_profile_v1.profile_enter_list.author_id"),
          dict(name="download_aids", path="user_profile_v1.download_video_list.author_id"),
          dict(name="collect_aids", path="user_profile_v1.collect_list.author_id"),
          dict(name="hate_aids", path="user_profile_v1.hate_list.author_id"),
          dict(name="follow_timestamps", path="user_profile_v1.follow_list.time_ms"),
          dict(name="explore_low_active_level", path="feature_collection.explore_low_active_level"),
          dict(name="user_persona_tag_id_list", path="feature_collection.features", sample_attr_name="uInterestHetuContentTagIdList"),
          dict(name="user_persona_tag_score_list", path="feature_collection.features", sample_attr_name="uInterestHetuContentTagScoreList"),
          dict(name="user_persona_ip_id_list", path="feature_collection.features", sample_attr_name="uInterestHetuIpIdList"),
          dict(name="user_persona_ip_score_list", path="feature_collection.features", sample_attr_name="uInterestHetuIpScoreList"),
          dict(name="explore_la_long_view_author_list", path="feature_collection.features_map.value", sample_attr_name="uLongViewAuthorList"),
          dict(name="page_index", path="req_type.page"),
          dict(name="gamora_hetu_adjust_history_list", path="feature_collection.gamora_hetu_adjust_history"),
          dict(name="opt_card_like_list", path="feature_collection.features", sample_attr_name="uRecoOptCardActionLikeList"),
          dict(name="opt_card_dis_like_list", path="feature_collection.features", sample_attr_name="uRecoOptCardActionDislikeList"),
          dict(name="videoPlayingPid", path="user_profile_v1.video_playing_stat.photo_id"),
          dict(name="realtimeClickList", path="realtime_click_list"),
          dict(name="searchList", path="user_profile_v1.search_click_photo_list.photo_id"),
          dict(name="be_black_list", path="feature_collection.features_map.value", sample_attr_name="uBeBlackedAuthorList"),
          dict(name="user_follow_type", path="feature_collection.features_map.value", sample_attr_name="uFollowPeopleType"),
          dict(name="playstat_playtimes", path="user_profile_v1.video_playing_stat.playing_time"),
          dict(name="playstat_durations", path="user_profile_v1.video_playing_stat.video_duration"),
          dict(name="playstat_hetu1s", path="user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_one", repeat_limit={"user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_one": 1}, repeat_align=True),
          dict(name="playstat_hetu2s", path="user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_two", repeat_limit={"user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_two": 1}, repeat_align=True),
          dict(name="playstat_hetu3s", path="user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_three", repeat_limit={"user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_three": 1}, repeat_align=True),
          dict(name="playstat_hetu4s", path="user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_four", repeat_limit={"user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_four": 1}, repeat_align=True),
          dict(name="playstat_hetutags", path="user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_tag", repeat_limit={"user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_tag": 20}, repeat_align=True),
          dict(name="playstat_reasons", path="user_profile_v1.video_playing_stat.reason"),
          dict(name="userRecentViewTimeListRaw", path="user_profile_v1.video_playing_stat.client_timestamp"),
          dict(name="userRecentViewPageListRaw", path="user_profile_v1.video_playing_stat.page"),
          dict(name="featureUId", path="id"),
          dict(name="gender", path="gender", skip_unset_field=True),
          dict(name="true_gender", path="true_gender", skip_unset_field=True),
          dict(name="infer_gender", path="infer_gender", skip_unset_field=True),
          dict(name="true_year", path="true_year", skip_unset_field=True),
          dict(name="infer_year", path="infer_year", skip_unset_field=True),
          dict(name="featureAgeSegment", path="basic_info.age_segment", skip_unset_field=True),
          dict(name="featureProvinceId", path="location.province_id", skip_unset_field=True),
          dict(name="featureCityId", path="location.city_id", skip_unset_field=True),
          dict(name="featureClientId", path="client_id", skip_unset_field=True),
          dict(name="featureVisitMod", path="visit_mod", skip_unset_field=True),
          dict(name="featureVisitNet", path="visit_net", skip_unset_field=True),
          dict(name="app_list", path="apps.app.package"),
          dict(name="app_list_ios", path="apps.app.app_name"),
          dict(name="featureUserLevel", path="user_profile.user_level"),
          dict(name="featureActiveDays", path="active_days"),
          dict(name="featureTopDislikeTopic", path="top_dislike_topic.ids"),
          dict(name="featureRiskLevel", path="risk_level"),
          dict(name="featureLongTermInterestPhotoDnnClusterId", path="long_term_interest_photo_dnn_cluster_id"),
          dict(name="featureUserRequestProvinceId", path="request_location.province_id"),
          dict(name="featureUserRequestCityId", path="request_location.city_id"),
          dict(name="featureUserRequestPoiType", path="request_location.poi_type"),
          dict(name="exp_show", path="user_profile.exp_stat.exp_realshow"),
          dict(name="exp_click", path="user_profile.exp_stat.exp_click"),
          dict(name="exp_like", path="user_profile.exp_stat.exp_like"),
          dict(name="exp_follow", path="user_profile.exp_stat.exp_follow"),
          dict(name="exp_forward", path="user_profile.exp_stat.exp_forward"),
          dict(name="exp_long_view", path="user_profile.exp_stat.exp_long_view"),
          dict(name="exp_short_view", path="user_profile.exp_stat.exp_short_view"),
          dict(name="exp_watch_time", path="user_profile.exp_stat.exp_watch_time"),
          dict(name="did", path="device_id"),
          dict(name="upload_count", path="upload_count"),
          dict(name="upload_rate", path="upload_rate"),
          dict(name="follow_count", path="follow_count"),
          dict(name="fans_count", path="fans_count"),
          dict(name="is_douyin", path="is_douyin"),
          dict(name="featrueUserLongTermHetu1Id", path="user_interest_profile.hetu_level_one_long_term_id"),
          dict(name="featrueUserLongTermHetu1Score", path="user_interest_profile.hetu_level_one_long_term_score"),
          dict(name="featrueUserLongTermHetu2Id", path="user_interest_profile.hetu_level_two_long_term_id"),
          dict(name="featrueUserLongTermHetu2Score", path="user_interest_profile.hetu_level_two_long_term_score"),
          dict(name="featrueUserLongTermHetu3Id", path="user_interest_profile.hetu_level_three_long_term_id"),
          dict(name="featrueUserLongTermHetu3Score", path="user_interest_profile.hetu_level_three_long_term_score"),
          dict(name="user_fountain_play_id_list", path="fountain_reco_user_profile.video_play_stat.photo_id"),
          dict(name="user_fountain_play_aid_list", path="fountain_reco_user_profile.video_play_stat.author_id"),
          dict(name="user_fountain_play_time_list", path="fountain_reco_user_profile.video_play_stat.playing_time"),
          dict(name="user_fountain_play_duration_list", path="fountain_reco_user_profile.video_play_stat.video_duration"),
          dict(name="user_fountain_play_timestamp_list", path="fountain_reco_user_profile.video_play_stat.client_timestamp"),
          dict(name="user_fountain_play_page_list", path="fountain_reco_user_profile.video_play_stat.page"),
          dict(name="user_fountain_play_hetu_l1_top1_list", path="fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_one", repeat_limit={"fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_one": 1}, repeat_align=True),
          dict(name="user_fountain_play_hetu_l2_top1_list", path="fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_two", repeat_limit={"fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_two": 1}, repeat_align=True),
          dict(name="user_fountain_play_hetu_l3_top1_list", path="fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_three", repeat_limit={"fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_three": 1}, repeat_align=True),
          dict(name="user_fountain_play_hetu_l4_top1_list", path="fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_four", repeat_limit={"fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_four": 1}, repeat_align=True),
          dict(name="realtime_like_list", path="realtime_like_list"),
          dict(name="realtime_follow_list", path="realtime_follow_list"),
          dict(name="fountain_like_pids", path="fountain_reco_user_profile.like_list.photo_id"),
          dict(name="fountain_like_aids", path="fountain_reco_user_profile.like_list.author_id"),
          dict(name="fountain_follow_pids", path="fountain_reco_user_profile.follow_list.photo_id"),
          dict(name="fountain_follow_aids", path="fountain_reco_user_profile.follow_list.author_id"),
          dict(name="downloadAids", path="user_profile_v1.download_video_list.author_id", repeat_limit={"user_profile_v1.download_video_list": 10}),
          dict(name="searchClickAids", path="user_profile_v1.search_click_author_list.author_id", repeat_limit={"user_profile_v1.search_click_author_list": 100}),
          dict(name="dupClickAids", path="user_profile_v1.dup_click_list.author_id", repeat_limit={"user_profile_v1.dup_click_list": 3}),
          dict(name="profileEnterAids", path="user_profile_v1.profile_enter_list.author_id", repeat_limit={"user_profile_v1.profile_enter_list": 200}),
          dict(name="likeAids", path="user_profile_v1.like_list.author_id", repeat_limit={"user_profile_v1.like_list": 150}),
          dict(name="forwardAids", path="user_profile_v1.forward_list.author_id", repeat_limit={"user_profile_v1.forward_list": 100}),
          dict(name="commentAids", path="user_profile_v1.comment_list.author_id", repeat_limit={"user_profile_v1.comment_list": 200}),
          dict(name="friendAids", path="friend_info_v2.bid_follow_list.friend_id"),
          dict(name="followAids", path="follow_list.user.id"),
          dict(name="current_location_ad_code", path="request_location_new.ad_code", skip_unset_field=True),
          dict(name="hometown_ad_code", path="request_location_new.hometown_ad_code", skip_unset_field=True),
          dict(name="freq_ad_code", path="request_location_new.freq_ad_code", skip_unset_field=True),
          dict(name="is_growth_reflux", path="is_growth_reflux", skip_unset_field=True),
          dict(name="is_new_device", path="user_class.new_device_status", skip_unset_field=True),
          dict(name="migrate_ad_code", path="request_location_new.migrate_location.ad_code", repeat_limit={"request_location_new.migrate_location.ad_code":5}),
        ],
      ) \
      .pack_common_attr(
        input_common_attrs = ["like_list", "follow_list", "forward_list", "comment_list", "collect_list", "download_list", "search_click_list"],
        output_common_attr = "profile_v1_interaction_trigger_list",
        deduplicate = True
      ) \
      .filter_by_common_attr(
        item_list_from_attr = "profile_v1_interaction_trigger_list",
        common_attr=["hate_list"]
      ) \
      .shuffle_list_attr(
        common_attr = "profile_v1_interaction_trigger_list"
      ) \
      .pack_common_attr(
        input_common_attrs = [
          "like_aids", "follow_aids", "forward_aids", "comment_aids", "profile_enter_aids", "download_aids", "collect_aids"
        ],
        output_common_attr = "profile_v1_interaction_trigger_aids",
        deduplicate = True
      ) \
       .filter_by_common_attr(
        item_list_from_attr = "profile_v1_interaction_trigger_aids",
        common_attr=["hate_aids"]
      ) \
      .if_("enable_high_level_index_retr == 1") \
      .enrich_with_protobuf(
        from_extra_var = "user_info_ptr",
        attrs = [
          dict(name="click_list_hetu2", path="user_profile_v1.click_list.hetu_tag_level_info.hetu_level_two"),
          dict(name="like_list_hetu2", path="user_profile_v1.like_list.hetu_tag_level_info.hetu_level_two"),
          dict(name="follow_list_hetu2", path="user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_two"),
          dict(name="forward_list_hetu2", path="user_profile_v1.forward_list.hetu_tag_level_info.hetu_level_two"),
          dict(name="comment_list_hetu2", path="user_profile_v1.comment_list.hetu_tag_level_info.hetu_level_two"),
          dict(name="collect_list_hetu2", path="user_profile_v1.collect_list.hetu_tag_level_info.hetu_level_two"),
          dict(name="download_list_hetu2", path="user_profile_v1.download_video_list.hetu_tag_level_info.hetu_level_two"),
          dict(name="search_click_list_hetu2", path="user_profile_v1.search_click_photo_list.hetu_tag_level_info.hetu_level_two"),
          dict(name="find_user_active_degree", path="find_user_active_degree")
        ]
      ) \
      .pack_common_attr(
        input_common_attrs = ["like_list_hetu2", "follow_list_hetu2", "forward_list_hetu2", "comment_list_hetu2", "collect_list_hetu2", "download_list_hetu2", "search_click_list_hetu2"],
        output_common_attr = "profile_v1_interaction_hetu2_list",
        deduplicate = False
      ) \
      .shuffle_list_attr(
        common_attr = "hetu2_all_set"
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "profile_v1_interaction_hetu2_list",
          "click_list_hetu2",
          "hetu2_select_num",
          "hetu2_all_set"
        ],
        export_common_attr = [
          "user_interest_hetu2"
        ],
        function_name = "GetUserInterestHetu2",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "explore_low_active_level",
        ],
        export_common_attr = [
          "explore_vv_3d",
          "find_visit_days_30d",
          "explore_zero_play_days_15d",
          "infer_uv_ctr",
          "conti_zero_click_num",
          "conti_zero_realshow_num",
          "zero_visit_gap",
          "active_days_avg_vv",
          "uExploreActiveDays",
          "uIsExploreLaUser",
          {"name": "uIsExploreNewLaUser", "as": "is_explore_new_la_user"},
        ],
        function_name = "SplitLauUserInfoField",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .if_("explore_enable_expand_tnu == 1") \
        .split_string(
          input_common_attr = "explore_user_level_for_expand_tnu_str",
          output_common_attr = "explore_user_level_for_expand_tnu",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True,
        ) \
      .end_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "user_info_ptr",
          {"name": "explore_user_level_for_expand_tnu", "as": "user_level_for_expand_tnu"},
        ],
        export_common_attr = [
          "uIsExploreTnuCrowdUser"
        ],
        function_name = "IsTnuCrowdUser",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "user_info_ptr",
        ],
        export_common_attr = [
          "uIsRefluxCrowdUser"
        ],
        function_name = "IsRefluxCrowdUser",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .if_("enable_open_zero_play_user == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "user_info_ptr",
            "explore_vv_3d", 
            "explore_vv_3d_threshold_val",
            "infer_uv_ctr",
            "explore_infer_uv_ctr_threshold_val",
            "enable_open_current_day"
          ],
          export_common_attr = [
            "is_zero_play_user",
            "explore_today_vv"
          ],
          function_name = "SelectZeroPlayUser",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .split_string(
        input_common_attr = "explore_la_app_name_str",
        output_common_attr = "explore_la_app_names",
        delimiters = "|",
        trim_spaces = True,
        skip_empty_tokens = True,
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "app_list", "as": "app_list_android"},
          "app_list_ios",
          {"name": "explore_la_app_names", "as": "satisfy_app_list"},
        ],
        export_common_attr = [
          "is_la_correct_user"
        ],
        function_name = "SatisfyDelineateRequirement",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .parse_protobuf_from_string(
        input_attr = "kuibaUserAttrStr",
        output_attr = "kuiba_user_attr",
        class_name = "kuiba::PredictItem",
      ) \
      .extract_kuiba_sample_attr(
        output_attrs = [
          "uBuyerEffectiveType",
          "uSexyInterestScore",
          "uStandardRealShowPicAllIdList",
          "uGamoraUploadDayNum30d",
          "uNebulaUploadDayNum30d",
          "uStandardClickPicAllIdList",
          "uStandardLongviewPicAllIdList",
          "uStandardLikePicAllIdList",
          "uStandardFollowPicAllIdList",
          "uStandardCollectPicAllIdList",
          "uOldMmuClusterId300ListList",
          "uLongTermInterestAndScoreList",
          "uUnbiasTriggerPidsList",
          "uInterestAndScoreList",
          "uValideInterestAndScoreList",
          "uCocoonCodeKV",
          "uFollowHighValueAuthorList",
          "uHighTimeDauRateKV",
          "uLongViewAuthorListV2",
          "uShareidCntKV",
          "uOpenShareidCntKV",
          "uOpenDeviceCntKV",
          "uPullNumKV",
          "uShareBringNewDeviceNumKV",
          "uAttributionPerShareKV",
          "uMultiDimensionGroupKV",
          "uMultiDimensionGroupDetailKV",
          "uFansList",
          "uPicValidInterestClusterIdList",
          "uPicLongInterestClusterIdList",
          "uPicSearchInterestClusterIdList",
          "uPicSearchInterestClusterScoreList",
          "uDevelopInterestV2AndScoreList",
          "uValidInterestV2AndScoreList",
          "uMarriageLabelKV",
          "uBirthLabelKV",
          "uEduLabelKV",
          "uMinorLabelKV",
          "uStudentLabelKV",
          "uHetuCategoryInterestlv1IdList",
          "uHetuCategoryInterestlv1ScoreList",
          "uMessageActiveDegreeCode",
          "uToUserShareSendNum30dKV",
          "uDoubleOutsideValidPicClusterCnt7dKV",
          "uPicGrowthCidList",
          "uToleranceScoreKV",
          "uLLMHetuKV",
          "uInsideShareActiveDegreeDetailCode",
          "uDoubleOutsideValidPicCluster7dList",
          "uSingleValidPicCluster7dList",
          "uStandardExploreRealshowPhotoIdList",
          "uStandardExploreRealshowAuthorIdList",
          "uStandardExploreRealshowTimestampList",
          "uStandardExploreRealshowLabelList",
          "uStandardExploreRealshowDurationList", # 实际上是play time, 单位 ms
          "uStandardExploreRealshowHetuTag1List",
          "uStandardExploreRealshowHetuTag2List",
          "uStandardExploreRealshowHetuTag5List",
          "uValidTagIdAndScoreList",
          "uPicU2CTopkCidList",
          "uPicU2CTopkProbList",
          "uInduceInteractionToleranceScoreListList",
          "uHackHighpToleranceScoreListList",
          "uYoungMutualLikeToleranceScoreListList",
          "uCoverNegTolerance1KV",
          "uCoverNegTolerance2KV",
          "uCoverNegTolerance3KV",
          "uGuanganNegTolerance1KV",
          "uGuanganNegTolerance2KV",
          "uGuanganNegTolerance3KV",
          "uExploreShortValidInterestAndScoreList",
          "uExploreShortDevelopInterestAndScoreList",
          "uExploreFountainPreferenceTypeKV",
          "uJobIdLv2KV",
          "uMarriageLabelV1KV",
          "uBirthLabelV1KV",
          "uStudentLabelV1KV",
          "uExplorePicUpliftValuesKV",
          "uExploreVidRealshowCntKV",
          "uExplorePicRealshowCntKV",
          "uExploreVidCtrKV",
          "uExplorePicCtrKV"
        ],
        predict_item = "kuiba_user_attr",
        is_common_attr = True
      ) \
      .if_("enable_explore_user_long_term_interest_parse == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "uLongTermInterestAndScoreList", "as": "tag_id_score_list"},
            {"name": "interest_and_score_threshold", "as": "score_thresh"}
          ],
          export_common_attr = [
            {"name": "tag_id_list", "as": "user_long_term_interest_cid_list"}
          ],
          function_name = "ParseTagIdAndScoreList",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_use_long_view_author_list_v2 == 1") \
        .copy_attr(
          attrs = [
            {"from_common": "uLongViewAuthorListV2", "to_common": "explore_la_long_view_author_list"}
          ]
        ) \
      .end_() \
      .if_("enable_use_standard_explore_realshow_list == 1") \
        .copy_attr(
          attrs=[{
            "from_common": "uStandardExploreRealshowPhotoIdList",
            "to_common": "standard_explore_realshow_pid_list"
          }]
        ) \
      .end_() \
      .if_("enable_use_explore_recent_click_list == 1") \
        .copy_attr(
          attrs=[{
            "from_common": "click_list",
            "to_common": "explore_user_recent_click_list"
          }]
        ) \
      .end_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "uBuyerEffectiveType"
        ],
        export_common_attr = [
          "merchant_buyer_type"
        ],
        function_name = "MerchantCalcBuyerType",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "uHighTimeDauRateKV"
        ],
        export_common_attr = [
          "active_days_high_time_rate",
          "active_days_gt_5min_rate",
          "user_active_decline_score"
        ],
        function_name = "ParseUserHighTimeInfo",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .if_("explore_enable_calc_recent_valid_click_count == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "user_info_ptr",
            {"name": "explore_recent_valid_click_time_gap_minute", "as": "time_gap_minute"},
            {"name": "explore_recent_valid_click_max", "as": "keep_size"},
            {"name": "explore_recent_valid_click_short_view_threshold", "as": "short_view_play_time_ms_threshold"},          
          ],
          export_common_attr = [
            "explore_recent_valid_click_count",
          ],
          function_name = "CalcRecentValidClickCount",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_calc_pic_low_show_boost_user == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "explore_pic_recent_show_interval_hour",
            "uStandardRealShowPicAllIdList",
            "uStandardExploreRealshowPhotoIdList",
            "uStandardExploreRealshowTimestampList",
          ],
          export_common_attr = [
            "user_pic_recent_show_cnt",
          ],
          function_name = "CalcUserPicRecentShow",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .log_debug_info(
        common_attrs = [
          "click_list_hetu2",
          "profile_v1_interaction_hetu2_list",
          "user_interest_hetu2",
          "find_user_active_degree",
          "follow_timestamps",
          "conti_zero_click_num",
          "conti_zero_realshow_num",
          "zero_visit_gap",
          "active_days_avg_vv",
          "uExploreActiveDays",
          "uIsExploreLaUser",
          "user_risk_level"
        ],
        for_debug_request_only = True
      ) \
      .end_() \
      .enrich_attr_by_light_function(
        export_common_attr = [
          "request_hour"
        ],
        function_name = "GetHour",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .gen_common_attr_by_lua(
        attr_map = {
          "is_teenager": "basic_info_age_segment_v2 ~= nil and (basic_info_age_segment_v2 == 1 or basic_info_age_segment_v2 == 2)"
        },
      ) \
      .if_("enable_explore_get_user_message_cnt_today_from_redis == 1") \
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
          "u_inside_share_active_degree_detail_code": "uInsideShareActiveDegreeDetailCode or 0",
          "is_first_refresh": "page_index == 1 and refreshTimes == 0",
        }
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "hate_list_page_types", "as": "hate_page_type_list"},
          {"name": "hate_list_timestamps", "as": "hate_time_ms_list"},
          {"name": "explore_koc_htr_time_gap_minute", "as": "time_gap_minute"},
        ],
        export_common_attr = [
          "recent_hate_count",
        ],
        function_name = "CalcRecentHateCount",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .if_("enable_explore_get_recent_hate_pid_list == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_hate_time_gap_minute", "as": "xtr_weight"},
            {"name": "uExploreActiveDays", "as": "user_vv"},
            {"name": "explore_hate_time_gap_minute_active_days_weight_adjust_exp_upper", "as": "exp_upper"},
            {"name": "explore_hate_time_gap_minute_active_days_weight_adjust_alpha", "as": "alpha"},
            {"name": "explore_hate_time_gap_minute_active_days_weight_adjust_beta", "as": "beta"},
            {"name": "explore_hate_time_gap_minute_active_days_weight_adjust_omega", "as": "omega"},
            {"name": "explore_hate_time_gap_minute_active_days_weight_adjust_max", "as": "coeff_max"},
            {"name": "explore_hate_time_gap_minute_active_days_weight_adjust_min", "as": "coeff_min"},
          ],
          export_common_attr = [
            {"name": "xtr_weight", "as": "explore_hate_time_gap_minute"},
          ],
          function_name = "AdjustWeightByUserVv",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "hate_list", "as": "hate_pid_list"},
            {"name": "hate_list_page_types", "as": "hate_page_type_list"},
            {"name": "hate_list_timestamps", "as": "hate_time_ms_list"},
            {"name": "explore_hate_time_gap_minute", "as": "time_gap_minute"},
          ],
          export_common_attr = [
            "recent_hate_pid_list",
          ],
          function_name = "GetRecentHatePidList",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_calc_recent_search_user == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "search_query_list_timestamp", "as": "query_timestamp_list"},
            {"name": "explore_recent_search_time_gap_min", "as": "time_gap_min"},
          ],
          export_common_attr = [
            "is_recent_search_user",
          ],
          function_name = "CalcRecentSearchUser",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_get_search_valid_play_list == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "search_play_pid_list",
            "search_play_timestamp_list",
            "search_play_video_duration_list",
            "search_play_time_list",
            {"name": "explore_search_valid_play_time_window", "as": "time_window"},
            {"name": "explore_recent_search_valid_play_time_window", "as": "recent_time_window"},
            {"name": "explore_search_valid_play_time_thresh", "as": "play_time_thresh"}
          ],
          export_common_attr = [
            "user_search_valid_play_rate",
            "user_search_valid_play_pid_list",
            "user_recent_search_valid_play_pid_list",
            "user_recent_search_valid_play_timestamp_list"
          ],
          function_name = "GetSearchValidPlayList",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_cal_user_poor_quality_hate_reason_count == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "hate_list_page_types", "as": "hate_page_type_list"},
            {"name": "hate_list_timestamps", "as": "hate_time_ms_list"},
            {"name": "hate_list_reasons", "as": "hate_reason_list"},
            {"name": "explore_cal_user_poor_quality_hate_reason_count_gap_minute", "as": "time_gap_minute"},
          ],
          export_common_attr = [
            {"name": "hate_reason_count", "as": "user_poor_quality_hate_reason_count"}
          ],
          function_name = "CalcPoorQualityHateReasonCount",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_parse_user_valid_tag_id_score_list == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "uValidTagIdAndScoreList", "as": "tag_id_score_list"},
          ],
          export_common_attr = [
            {"name": "tag_id_list", "as": "user_valid_tag_id_list"},
            {"name": "tag_score_list", "as": "user_valid_tag_score_list"},
          ],
          function_name = "ParseTagIdAndScoreList",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .switch_("explore_user_bad_cover_tolerance_switch") \
        .case_(1) \
          .copy_attr(
            attrs = [{
              "from_common": "uCoverNegTolerance1KV",
              "to_common": "user_bad_cover_tolerance",
            }]
          ) \
        .case_(2) \
          .copy_attr(
            attrs = [{
              "from_common": "uCoverNegTolerance2KV",
              "to_common": "user_bad_cover_tolerance",
            }]
          ) \
        .case_(3) \
          .copy_attr(
            attrs = [{
              "from_common": "uCoverNegTolerance3KV",
              "to_common": "user_bad_cover_tolerance",
            }]
          ) \
      .end_() \
      .switch_("explore_user_bad_sense_tolerance_switch") \
        .case_(1) \
          .copy_attr(
            attrs = [{
              "from_common": "uGuanganNegTolerance1KV",
              "to_common": "user_bad_sense_tolerance",
            }]
          ) \
        .case_(2) \
          .copy_attr(
            attrs = [{
              "from_common": "uGuanganNegTolerance2KV",
              "to_common": "user_bad_sense_tolerance",
            }]
          ) \
        .case_(3) \
          .copy_attr(
            attrs = [{
              "from_common": "uGuanganNegTolerance3KV",
              "to_common": "user_bad_sense_tolerance",
            }]
          ) \
      .end_() \
      .if_("enable_explore_user_long_view_pid_list_compute == 1") \
        .explore_common_user_feature_enricher(
          user_info_attr = "user_info_ptr",
          user_action_list_long_version_attr = "enable",
          user_long_view_pids_attr = "user_long_view_pid_list"
        ) \
      .end_() \
      .if_("explore_enable_stat_last_action_time_gap == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "user_info_ptr"
          ],
          export_common_attr = [
            {"name": "like_gap_hour", "as": "user_explore_last_like_gap_hour"},
            {"name": "follow_gap_hour", "as": "user_explore_last_follow_gap_hour"},
            {"name": "comment_gap_hour", "as": "user_explore_last_comment_gap_hour"},
            {"name": "collect_gap_hour", "as": "user_explore_last_collect_gap_hour"},
          ],
          function_name = "ExploreStatLastActionTimeGap",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_enable_gen_pic_crowd_show == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "user_info_ptr",
            {"name": "explore_gen_pic_crowd_show_target_exptag", "as": "target_exptag"},
            {"name": "explore_gen_pic_crowd_show_gap_hour", "as": "gap_hour"}
          ],
          export_common_attr = [
            {"name": "has_exptag_show", "as": "user_has_pic_crowd_show"},
          ],
          function_name = "HasExptagShow",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_cal_user_pic_ctr_preference_coeff == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "uExploreVidRealshowCntKV",
            "uExplorePicRealshowCntKV",
            "uExploreVidCtrKV",
            "uExplorePicCtrKV",
            {"name": "explore_emp_pic_video_ctr_ratio", "as": "emp_pic_video_ctr_ratio"},
            {"name": "explore_user_pic_ctr_preference_coeff_video_real_show_threshold", "as":"video_real_show_threshold"},
            {"name": "explore_user_pic_ctr_preference_coeff_pic_real_show_threshold", "as":"pic_real_show_threshold"},
            {"name": "explore_user_pic_ctr_preference_coeff_upper_bound", "as": "pic_ctr_preference_upper_bound"},
            {"name": "explore_user_pic_ctr_preference_coeff_lower_bound", "as": "pic_ctr_preference_lower_bound"},
          ],
          export_common_attr = [
            "pic_ctr_preference_coeff",
          ],
          function_name = "CalcUserPicCtrPreferenceCoeff",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()
    self.explore_get_user_interest_tagnex_tgi_list("group", ["uMultiDimensionGroupDetailKV"])
    self.explore_get_user_interest_tagnex_tgi_list("career", ["uJobIdLv2KV"])
    self.explore_get_user_interest_tagnex_tgi_list("stage", ["uMarriageLabelV1KV", "uBirthLabelV1KV", "uStudentLabelV1KV"])
    self.explore_get_user_interest_tagnex_tgi_list("age", ["basic_info_age_segment_v2"])
    self.explore_get_user_interest_tagnex_tgi_list("pic_career", ["uJobIdLv2KV"])
    self.explore_get_user_interest_tagnex_tgi_list("pic_age", ["basic_info_age_segment_v2"])
    self.explore_get_user_valid_and_develop_interest()

  def explore_get_user_valid_and_develop_interest(self):
    self.flow \
    .switch_("explore_user_valid_interest_switch") \
      .case_(1) \
        .copy_attr(
          attrs = [{
            "from_common": "uValideInterestAndScoreList",
            "to_common": "user_valid_interest_cid_and_score_list",
          }]
        ) \
      .case_(2) \
        .copy_attr(
          attrs = [{
            "from_common": "uValidInterestV2AndScoreList",
            "to_common": "user_valid_interest_cid_and_score_list",
          }]
        ) \
      .case_(3) \
        .copy_attr(
          attrs = [{
            "from_common": "uExploreShortValidInterestAndScoreList",
            "to_common": "user_valid_interest_cid_and_score_list",
          }]
        ) \
    .end_() \
    .switch_("explore_user_develop_interest_switch") \
      .case_(1) \
        .copy_attr(
          attrs = [{
            "from_common": "uInterestAndScoreList",
            "to_common": "user_develop_interest_cid_and_score_list",
          }]
        ) \
      .case_(2) \
        .copy_attr(
          attrs = [{
            "from_common": "uDevelopInterestV2AndScoreList",
            "to_common": "user_develop_interest_cid_and_score_list",
          }]
        ) \
      .case_(3) \
        .copy_attr(
          attrs = [{
            "from_common": "uExploreShortDevelopInterestAndScoreList",
            "to_common": "user_develop_interest_cid_and_score_list",
          }]
        ) \
    .end_() \
    .if_("enable_explore_pack_develop_and_valid_interest == 1") \
      .pack_common_attr(
        input_common_attrs = ["user_valid_interest_cid_and_score_list", "user_develop_interest_cid_and_score_list"],
        output_common_attr = "user_postive_interest_score_list",
      ) \
    .end_() \
    .if_("enable_explore_parse_user_valid_interest_cid_and_score_list == 1 or is_traceback_request == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "user_valid_interest_cid_and_score_list", "as": "tag_id_score_list"},
          {"name": "interest_and_score_threshold", "as": "score_thresh"}
        ],
        export_common_attr = [
          {"name": "tag_id_list", "as": "user_valid_interest_cid_list"}
        ],
        function_name = "ParseTagIdAndScoreList",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("explore_enable_get_user_continuous_unclick_count == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "uStandardExploreRealshowLabelList", "as": "real_show_lables"},
          {"name": "uStandardExploreRealshowDurationList", "as": "real_show_play_times"},
          {"name": "uStandardExploreRealshowTimestampList", "as": "real_show_timestamps"},
          {"name": "explore_user_continuous_unclick_count_minute_threshold", "as": "minute_threshold"},
          {"name": "explore_user_continuous_unclick_count_is_short_view_involved", "as": "is_short_view_involved"},
        ],
        export_common_attr = [
          "user_continuous_unclick_count"
        ],
        function_name = "GenUserContinuousUnclickCount",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .split_string(
        input_common_attr = "explore_user_continuous_unclick_count_threshold_adjust_str",
        output_common_attr = "explore_user_continuous_unclick_count_threshold_adjust_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_double = True,
      ) \
      .select_list_values(
        index_attr = "find_user_active_degree",
        list_values = [ 
          {"from": "explore_user_continuous_unclick_count_threshold_adjust_list", "to": "explore_user_continuous_unclick_count_threshold_adjust_score"},
        ],  
        is_common_attr=True
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_user_continuous_unclick_count_threshold", "as": "value"},
          {"name": "explore_user_continuous_unclick_count_threshold_adjust_score", "as": "weight"},
        ],
        export_common_attr = [
          {"name": "new_value", "as": "explore_user_continuous_unclick_count_threshold_adjusted"}
        ],
        function_name = "CalExploreIntMultiDouble",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .if_("user_continuous_unclick_count > explore_user_continuous_unclick_count_threshold_adjusted") \
        .set_attr_value(
          common_attrs=[
            {   
              "name": "user_need_saving_flag",
              "type": "int",
              "value": 1
            }   
          ]   
        ) \
      .end_() \
    .end_() \
    .if_("enable_explore_parse_user_short_develop_interest_cid_list == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "uExploreShortDevelopInterestAndScoreList", "as": "tag_id_score_list"},
          {"name": "interest_and_score_threshold", "as": "score_thresh"}
        ],
        export_common_attr = [
          {"name": "tag_id_list", "as": "user_short_develop_interest_cid_list"}
        ],
        function_name = "ParseTagIdAndScoreList",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_()

  def explore_get_user_interest_tagnex_tgi_list(self, interest_type, dimension_attrs):
    enable_switch = f"enable_explore_get_user_{interest_type}_interest_tagnex_tgi_list"
    format_version = f"explore_user_{interest_type}_interest_format_version"
    tgi_prefix_attr = f"explore_user_{interest_type}_interest_tagnex_tgi_prefix"
    tgi_key_attr = f"explore_user_{interest_type}_interest_tagnex_tgi_key"
    tgi_list_attr = f"explore_user_{interest_type}_interest_tagnex_tgi_list"
    self.flow \
    .if_(enable_switch) \
      .switch_(format_version) \
        .case_(1) \
          .str_format(
            format_string = "%s_%s_%d_%d",
            input_attrs = [tgi_prefix_attr] + dimension_attrs,
            output_attr = tgi_key_attr,
          ) \
        .default_() \
          .str_format(
            format_string = "%s_%d",
            input_attrs = [tgi_prefix_attr] + dimension_attrs,
            output_attr = tgi_key_attr,
          ) \
      .end_() \
      .get_kconf_params(
        kconf_configs = [{
          "kconf_key": "reco.offline.userGroupandCareerInterestTagnexTgiStat",
          "json_path": "{{" + tgi_key_attr + "}}",
          "value_type": "list_int64",
          "default_value": [],
          "export_common_attr": tgi_list_attr
        }]
      ) \
    .end_()
    
    return self
