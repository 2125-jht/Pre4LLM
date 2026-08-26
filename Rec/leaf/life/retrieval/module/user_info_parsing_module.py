from retrieval import CommonModule

class UserInfoParsingModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .enrich_with_protobuf(
        from_extra_var = "user_info_ptr",
        attrs = [
          dict(name="user_browsed_photo_ids", path="browsed_photo_ids"),
          dict(name="user_gender", path="gender"),
          dict(name="user_age_segment", path="basic_info.age_segment"),
          dict(name="device_active_degree", path="user_class.device_active_degree"),
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
          dict(name="search_play_list", path="user_profile_v1.search_photo_play_list.photo_id"),
          dict(name="search_play_list_timestamps", path="user_profile_v1.search_photo_play_list.timestamp"),
          dict(name="search_play_list_play_duration", path="user_profile_v1.search_photo_play_list.play_duration"),
          dict(name="search_play_list_video_duration", path="user_profile_v1.search_photo_play_list.video_duration"),
          dict(name="hate_list", path="user_profile_v1.hate_list.photo_id"),
          dict(name="user_risk_level", path="feature_collection.risk_level"),
          dict(name="basic_info_gender_v2", path="basic_info.gender_v2"),
          dict(name="basic_info_age_segment_v2", path="basic_info.age_segment_v2"),
          dict(name="location_city_level_v2", path="location.city_level"),
          dict(name="profile_v1_click_trigger_aids", path="user_profile_v1.video_playing_stat.author_id"),
          dict(name="like_aids", path="user_profile_v1.like_list.author_id"),
          dict(name="follow_aids", path="user_profile_v1.follow_list.author_id"),
          dict(name="forward_aids", path="user_profile_v1.forward_list.author_id"),
          dict(name="comment_aids", path="user_profile_v1.comment_list.author_id"),
          dict(name="profile_enter_aids", path="user_profile_v1.profile_enter_list.author_id"),
          dict(name="download_aids", path="user_profile_v1.download_video_list.author_id"),
          dict(name="collect_aids", path="user_profile_v1.collect_list.author_id"),
          dict(name="hate_aids", path="user_profile_v1.hate_list.author_id"),
          dict(name="like_timestamps", path="user_profile_v1.like_list.time_ms"),
          dict(name="forward_timestamps", path="user_profile_v1.forward_list.time_ms"),
          dict(name="comment_timestamps", path="user_profile_v1.comment_list.time_ms"),
          dict(name="collect_timestamps", path="user_profile_v1.collect_list.time_ms"),
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
          # dict(name="be_black_list", path="feature_collection.features_map.value", sample_attr_name="uBeBlackedAuthorList"),
          dict(name="user_follow_type", path="feature_collection.features_map.value", sample_attr_name="uFollowPeopleType"),
          dict(name="playstat_playtimes", path="user_profile_v1.video_playing_stat.playing_time"),
          dict(name="playstat_durations", path="user_profile_v1.video_playing_stat.video_duration"),
          dict(name="playstat_hetu1s", path="user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_one", repeat_limit={"user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_one": 1}, repeat_align=True),
          dict(name="playstat_hetu2s", path="user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_two", repeat_limit={"user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_two": 1}, repeat_align=True),
          dict(name="playstat_hetu3s", path="user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_three", repeat_limit={"user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_three": 1}, repeat_align=True),
          dict(name="playstat_hetu4s", path="user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_four", repeat_limit={"user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_four": 1}, repeat_align=True),
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
          dict(name="uLikePidsFountain", path="fountain_reco_user_profile.like_list.photo_id"),
          dict(name="uLikeAidsFountain", path="fountain_reco_user_profile.like_list.author_id"),
          dict(name="uFollowAidsFountain", path="fountain_reco_user_profile.follow_list.author_id"),
          dict(name="uFollowPidsFountain", path="fountain_reco_user_profile.follow_list.photo_id"),
          dict(name="uCommentPidsFountain", path="fountain_reco_user_profile.comment_list.photo_id"),
          dict(name="uForwardAidsFountain", path="fountain_reco_user_profile.forward_list.author_id"),
          dict(name="uForwardPidsFountain", path="fountain_reco_user_profile.forward_list.photo_id"),
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
          dict(name="click_list_hetu2_id", path="user_profile_v1.click_list.hetu_tag_level_info.hetu_level_two", repeat_limit={"user_profile_v1.click_list.hetu_tag_level_info.hetu_level_two": 1}, repeat_align=True),
          dict(name="click_list_cluster_id", path="user_profile_v1.click_list.hetu_tag_level_info.hetu_cluster_id"),
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
      .parse_protobuf_from_string(
        input_attr = "kuibaUserAttrStr",
        output_attr = "kuiba_user_attr",
        class_name = "kuiba::PredictItem",
      ) \
      .extract_kuiba_sample_attr(
        output_attrs = [
          "uIsLifeHighActive",
          "uIsNotLifePassBy",
          "uLifeLongTermAuthorList",
          "uLifeLongTermAuthorListV2",
          "uLifePreferAuthor",
          "uHetuCategoryInterestlv1IdList",  # size 25
          "uHetuCategoryInterestlv1ScoreList",
          "uHetuCategoryInterestlv2IdList",  # size 50
          "uHetuCategoryInterestlv2ScoreList",
          "uNebulaXlifeVisitDays30dKV",
          "uNebulaDoubleFindVisitDays30dKV"
        ],
        predict_item = "kuiba_user_attr",
        is_common_attr = True
      ) \
      .if_("life_enable_cal_tnu_crowd_user == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "user_info_ptr",
          ],
          export_common_attr = [
            {"name": "uIsExploreTnuCrowdUser", "as": "uIsTnuCrowdUser"}
          ],
          function_name = "IsTnuCrowdUser",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
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
          "uExploreActiveDays",
          "uIsExploreLaUser"
        ],
        function_name = "SplitLauUserInfoField",
        class_name = "ExploreLightFunctionSetV2",
      ) \
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
          "uSexyInterestScore",
          "uStandardRealShowPicAllIdList",
          "uGamoraUploadDayNum30d",
          "uNebulaUploadDayNum30d",
          "uOldMmuClusterId300ListList"
        ],
        predict_item = "kuiba_user_attr",
        is_common_attr = True
      ) \
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
          "uExploreActiveDays",
          "uIsExploreLaUser",
          "download_aids",
          "profile_enter_aids",
          "hate_aids",
          "basic_info_age_segment_v2",
          "basic_info_gender_v2",
          "uNebulaXlifeVisitDays30dKV",
          "uNebulaDoubleFindVisitDays30dKV"
        ],
        for_debug_request_only = True
      ) \
      .log_debug_info(
        common_attrs = [
          'collect_aids', 'comment_aids', 'did', 'exp_click', 'exp_follow', 'exp_forward', 'exp_like', 'exp_long_view', 'exp_short_view', 'exp_show', 'exp_watch_time', 
          'explore_la_long_view_author_list', 'follow_aids', 'forward_aids', 'like_aids', 'profile_enter_list', 'realtimeClickList', 'searchList', 'user_age_segment', 
          'user_persona_ip_id_list', 'user_persona_ip_score_list', 'user_persona_tag_id_list', 'user_persona_tag_score_list', "uFollowAidsFountain", "uLikePidsFountain",
          "uForwardPidsFountain", "uCommentPidsFountain", "uFollowPidsFountain", "realtime_follow_list", "realtime_like_list", "uLikeAidsFountain", "uOldMmuClusterId300ListList",
          'collect_timestamps', 'comment_timestamps', 'forward_timestamps', 'like_timestamps', "user_browsed_photo_ids", "uForwardAidsFountain", "click_list_cluster_id", "click_list_hetu2_id",
        ]
      ) \
      .end_()
