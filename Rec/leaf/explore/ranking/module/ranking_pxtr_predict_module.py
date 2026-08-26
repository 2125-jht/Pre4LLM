from ranking import CommonModule

class RankingPxtrPredictModule(CommonModule):
  FR_MODEL_RECV_ITEM_ATTRS = [
    { "name": "ctr", "as": "pctr" },
    { "name": "ltr", "as": "pltr" },
    { "name": "wtr", "as": "pwtr" },
    { "name": "ftr", "as": "pftr" },
    { "name": "osftr", "as": "posftr" },
    { "name": "svr", "as": "psvr" },
    { "name": "cmtr", "as": "pcmtr" },
    { "name": "ptr", "as": "pptr" },
    { "name": "cmef", "as": "pcmef" },
    { "name": "htr", "as": "phtr" },
    { "name": "evtr", "as": "pevtr" },
    { "name": "lvtr", "as": "plvtr"},
    { "name": "fr_score1", "as": "fr_score1" },
    { "name": "fr_score2", "as": "fr_score2" },
    { "name": "epstr", "as": "pepstr" },
    { "name": "dtr", "as": "pdtr" },
    { "name": "cltr", "as": "pcltr" },
    { "name": "fetr", "as": "fetr" },
    { "name": "fountain_eff", "as": "fountain_eff" },
    { "name": "living_click", "as": "pliving_ctr"},
    { "name": "live_wtr", "as": "pliving_wtr"},
    { "name": "fvtr", "as": "pfvtr"},
    { "name": "awesome_wtd", "as": "awesome_wtd"},
    { "name": "dctr", "as": "pdctr"},
    { "name": "live", "as": "pvtr"},
    { "name": "adaptive_wtd_v2", "as": "adaptive_wtd_v2"},
    { "name": "cpr", "as": "cpr"},
    { "name": "wtd_evtr", "as": "wtd_evtr"},
    { "name": "wtd_lvtr", "as": "wtd_lvtr"},
    { "name": "future_xtr", "as": "future_xtr"},
    { "name": "lstr", "as": "plstr"},
    { "name": "lsst", "as": "plsst"},
    { "name": "etcm", "as": "petcm"},
    { "name": "swpst", "as": "pswpst"},
    # picture
    { "name": "pic_wtdPlaytime", "as": "pic_wtd"},
    { "name": "pic_lvtr", "as": "pic_lvtr"},
    { "name": "pic_cpr", "as": "pic_cpr"},
  ]

  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
    .enrich_attr_by_lua(
      import_item_attr = ["reason"],
      export_item_attr = ["reason_str"],
      function_for_item = "trans_reason_to_str",
      lua_script_file = "explore/ranking/lua/module/ranking_score__trans_reason_to_str.lua"
    ) \
    .build_protobuf(
      class_name = "ks.reco.RecoPhotoInfo",
      inputs = [
        {
          "item_attr": "cascade_pctr",
          "path": "context_info.cascade_pctr"
        },
        {
          "item_attr": "cascade_pltr",
          "path": "context_info.cascade_pltr"
        },
        {
          "item_attr": "cascade_pwtr",
          "path": "context_info.cascade_pwtr"
        },
        {
          "item_attr": "cascade_plvtr",
          "path": "context_info.cascade_plvtr"
        },
        {
          "item_attr": "cascade_psvtr",
          "path": "context_info.cascade_psvr"
        },
        {
          "item_attr": "cascade_pftr",
          "path": "context_info.cascade_pftr"
        },
        {
          "item_attr": "cascade_ptr",
          "path": "context_info.cascade_pptr"
        },
        {
          "item_attr": "cascade_pepstr",
          "path": "context_info.cascade_pepstr"
        },
        {
          "item_attr": "photo_id",
          "path": "ar_result.pid"
        },
        {
          "item_attr": "content_safety_level_with_namespace__level_hot_online",
          "path": "ar_result.content_safety_level"
        },
        {
          "item_attr": "reason_str",
          "path": "reason"
        },
        {
          "item_attr": "live_photo_info__is_living",
          "path": "living"
        },
        {
          "item_attr": "cascade_pdtr",
          "path": "context_info.cascade_pdtr"
        },
        {
          "item_attr": "cascade_pcmtr",
          "path": "context_info.cascade_pcmtr"
        },
        {
          "item_attr": "cascade_phtr",
          "path": "context_info.cascade_phtr"
        },
        {
          "item_attr": "cascade_pctr_index",
          "path": "cascade_pctr_index"
        },
        {
          "item_attr": "cascade_plvtr_index",
          "path": "cascade_plvtr_index"
        },
        {
          "item_attr": "cascade_pvtr_index",
          "path": "cascade_pvtr_index"
        },
        {
          "item_attr": "cascade_pltr_index",
          "path": "cascade_pltr_index"
        },
        {
          "item_attr": "cascade_pftr_index",
          "path": "cascade_pftr_index"
        },
        {
          "item_attr": "cascade_pwtr_index",
          "path": "cascade_pwtr_index"
        },
        {
          "item_attr": "cascade_pesptr_index",
          "path": "cascade_pesptr_index"
        },
        {
          "item_attr": "cascade_psvr_index",
          "path": "cascade_psvr_index"
        }
      ],
      as_string = True,
      output_item_attr = "reco_photo_info_str",
    ) \
    .if_("skip_explore_fullrank_sim_predict == 0") \
      .delegate_enrich(
        name = "explore_fullrank_sim_predict",
        kess_service = "{{gamora_common_predict_server_kess_name}}",
        partition_size = "{{ranking_server_partition_size}}",
        recv_item_attrs = RankingPxtrPredictModule.FR_MODEL_RECV_ITEM_ATTRS,
        request_type = "{{gamora_common_predict_server_request_type}}",
        send_common_attrs = [
          { "name": "userInfo", "as": "user_info_str" },
          { "name": "virtualTabId", "as": "virtual_tab_id" },
        ],
        send_item_attrs = [
          "live_photo_info__is_living",
          "reco_photo_info_str"
        ],
      ) \
    .end_() \
    .explore_embedding_candidates_attr_enricher(
      trans_type = "embedding_candidates",
      user_info_ptr_attr = "user_info_ptr",
      export_common_attr = "unexpected_source_pids",
      check_point = "fr",
      enable_report = "{{explore_fullrank_sim_enable_report}}"
    ) \
    .if_("enable_explore_action_list_similar_score_by_colossus == 1") \
      .explore_colossus_v2_trigger_enrich(
        colossus_resp_attr = "colossus_resp_v2",
        output_colossus_trigger_attr = "colossus_user_info__positive_action_photo_id_list",
        knowledge_hetu_set_attr = "knowledge_hetu_set",
        enable_default_select_triggers = "{{enable_explore_default_select_triggers}}",
        enable_different_signals_triggers = "{{enable_explore_different_signals_triggers}}",
        different_signals_triggers_select_num = "{{explore_different_signals_triggers_select_num}}",
        different_signals_triggers_min_play_time = "{{explore_different_signals_triggers_min_play_time}}",
        different_signals_triggers_play_time_ratio = "{{explore_different_signals_triggers_play_time_ratio}}",
        different_signals_triggers_min_days_ago = "{{explore_different_signals_triggers_min_days_ago}}",
        different_signals_triggers_max_days_ago = "{{explore_different_signals_triggers_max_days_ago}}",
        enable_different_signals_triggers_action_explore_list = "{{enable_explore_different_signals_triggers_action_explore_list}}",
        enable_different_signals_triggers_action_completion_list = "{{enable_explore_different_signals_triggers_action_completion_list}}",
        enable_different_signals_triggers_action_hetu_tag_list = "{{enable_explore_different_signals_triggers_action_hetu_tag_list}}",
        enable_different_signals_triggers_action_interact_list = "{{enable_explore_different_signals_triggers_action_interact_list}}",
        enable_different_signals_triggers_action_timestamp_order = "{{enable_explore_different_signals_triggers_action_timestamp_order}}",
        enable_not_select_bottom_selection_page = "{{enable_explore_not_select_bottom_selection_page}}",
        enable_only_select_explore_colossus_list = "{{enable_explore_only_select_explore_colossus_list}}",
        enable_only_select_high_interest_tab = "{{enable_explore_only_select_high_interest_tab}}",
        enable_select_high_interest_and_profile_tab = "{{enable_explore_select_high_interest_and_profile_tab}}",
        enable_select_high_interest_and_nearby_tab = "{{enable_explore_select_high_interest_and_nearby_tab}}"
      ) \
    .end_() \
    .if_("enable_explore_negtive_list_similar_score_by_colossus == 1") \
      .explore_colossus_v2_trigger_enrich(
        colossus_resp_attr = "colossus_resp_v2",
        output_colossus_trigger_attr = "colossus_user_info__negtive_action_photo_id_list",
        knowledge_hetu_set_attr = "knowledge_hetu_set",
        enable_default_select_triggers = "{{enable_explore_default_select_triggers}}",
        enable_negtive_signals_triggers = "{{enable_explore_negtive_signals_triggers}}",
        enable_select_bottom_page_negtive_signals = "{{enable_explore_select_bottom_page_negtive_signals}}",
        enable_select_explore_page_negtive_signals = "{{enable_explore_select_explore_page_negtive_signals}}",
        enable_negtive_signals_triggers_short_play = "{{enable_explore_negtive_signals_triggers_short_play}}",
        enable_negtive_signals_triggers_hate = "{{enable_explore_negtive_signals_triggers_hate}}",
        negtive_signals_triggers_min_days_ago= "{{explore_negtive_signals_triggers_min_days_ago}}",
        negtive_signals_triggers_max_days_ago = "{{explore_negtive_signals_triggers_max_days_ago}}",
        negtive_signals_triggers_select_num = "{{explore_negtive_signals_triggers_select_num}}",
        negtive_signals_triggers_play_time_thres = "{{explore_negtive_signals_triggers_play_time_thres}}"
      ) \
    .end_() \
    .if_("enable_explore_positive_list_by_colossus_for_good_quality == 1") \
      .explore_colossus_v2_trigger_enrich(
        colossus_resp_attr = "colossus_resp_v2",
        output_colossus_trigger_attr = "colossus_user_info_explore_positive_photo_id_list_for_good_quality",
        enable_default_select_triggers = "{{enable_explore_default_select_triggers_for_good_quality}}",
        enable_different_signals_triggers = "{{enable_explore_positive_triggers_for_good_quality}}",
        different_signals_triggers_select_num = "{{explore_positive_triggers_select_num_for_good_quality}}",
        different_signals_triggers_min_play_time = "{{explore_positive_triggers_min_play_time_for_good_quality}}",
        different_signals_triggers_play_time_ratio = "{{explore_positive_triggers_for_good_quality_play_time_ratio}}",
        different_signals_triggers_min_days_ago = "{{explore_positive_triggers_for_good_quality_min_days_ago}}",
        different_signals_triggers_max_days_ago = "{{explore_positive_triggers_for_good_quality_max_days_ago}}",
        enable_different_signals_triggers_action_explore_list = "{{enable_explore_positive_triggers_for_good_quality_action_explore_list}}",
        enable_different_signals_triggers_action_completion_list = "{{enable_explore_positive_triggers_for_good_quality_action_completion_list}}",
        enable_different_signals_triggers_action_interact_list = "{{enable_explore_positive_triggers_for_good_quality_action_interact_list}}",
        enable_different_signals_triggers_action_timestamp_order = "{{enable_explore_positive_triggers_for_good_quality_timestamp_order}}",
        enable_not_select_bottom_selection_page = "{{enable_explore_positive_triggers_for_good_quality_not_select_bottom_selection_page}}",
        enable_only_select_explore_colossus_list = "{{enable_explore_positive_triggers_for_good_quality_only_select_explore_colossus_list}}",
        enable_only_select_high_interest_tab = "{{enable_explore_positive_triggers_for_good_quality_only_select_high_interest_tab}}",
        enable_select_high_interest_and_profile_tab = "{{enable_explore_positive_triggers_for_good_quality_select_high_interest_and_profile_tab}}",
        enable_only_select_fountain_colossus_list =  "{{enable_explore_positive_triggers_for_good_quality_select_only_select_fountain_colossus_list}}",
        enable_only_unselect_explore_colossus_list =  "{{enable_explore_positive_triggers_for_good_quality_only_unselect_explore_colossus_list}}",
        enable_only_unselect_fountain_colossus_list =  "{{enable_explore_positive_triggers_for_good_quality_only_unselect_fountain_colossus_list}}",
        enable_get_longview_trigger = "{{enable_explore_positive_triggers_for_good_quality_get_longview_trigger}}",
      ) \
      .gen_common_attr_by_lua(
        attr_map={
          "colossus_user_info_explore_positive_photo_id_list_for_good_quality_size": "#(colossus_user_info_explore_positive_photo_id_list_for_good_quality or {})",
        }
      ) \
    .end_() \
    .if_("enable_explore_outer_positive_list_by_colossus_for_good_quality == 1") \
      .explore_colossus_v2_trigger_enrich(
        colossus_resp_attr = "colossus_resp_v2",
        output_colossus_trigger_attr = "colossus_user_info_explore_outer_positive_photo_id_list_for_good_quality",
        enable_default_select_triggers = "{{enable_explore_outer_default_select_triggers_for_good_quality}}",
        enable_different_signals_triggers = "{{enable_explore_outer_positive_triggers_for_good_quality}}",
        different_signals_triggers_select_num = "{{explore_outer_positive_triggers_select_num_for_good_quality}}",
        different_signals_triggers_min_play_time = "{{explore_outer_positive_triggers_min_play_time_for_good_quality}}",
        different_signals_triggers_play_time_ratio = "{{explore_outer_positive_triggers_for_good_quality_play_time_ratio}}",
        different_signals_triggers_min_days_ago = "{{explore_outer_positive_triggers_for_good_quality_min_days_ago}}",
        different_signals_triggers_max_days_ago = "{{explore_outer_positive_triggers_for_good_quality_max_days_ago}}",
        enable_different_signals_triggers_action_explore_list = "{{enable_explore_outer_positive_triggers_for_good_quality_action_explore_list}}",
        enable_different_signals_triggers_action_completion_list = "{{enable_explore_outer_positive_triggers_for_good_quality_action_completion_list}}",
        enable_different_signals_triggers_action_interact_list = "{{enable_explore_outer_positive_triggers_for_good_quality_action_interact_list}}",
        enable_different_signals_triggers_action_timestamp_order = "{{enable_explore_outer_positive_triggers_for_good_quality_timestamp_order}}",
        enable_not_select_bottom_selection_page = "{{enable_explore_outer_positive_triggers_for_good_quality_not_select_bottom_selection_page}}",
        enable_only_select_explore_colossus_list = "{{enable_explore_outer_positive_triggers_for_good_quality_only_select_explore_colossus_list}}",
        enable_only_select_high_interest_tab = "{{enable_explore_outer_positive_triggers_for_good_quality_only_select_high_interest_tab}}",
        enable_select_high_interest_and_profile_tab = "{{enable_explore_outer_positive_triggers_for_good_quality_select_high_interest_and_profile_tab}}",
        enable_only_select_fountain_colossus_list =  "{{enable_explore_outer_positive_triggers_for_good_quality_select_only_select_fountain_colossus_list}}",
        enable_only_unselect_explore_colossus_list =  "{{enable_explore_outer_positive_triggers_for_good_quality_only_unselect_explore_colossus_list}}",
        enable_only_unselect_fountain_colossus_list =  "{{enable_explore_outer_positive_triggers_for_good_quality_only_unselect_fountain_colossus_list}}",
        enable_get_longview_trigger = "{{enable_explore_outer_positive_triggers_for_good_quality_get_longview_trigger}}",
      ) \
      .gen_common_attr_by_lua(
        attr_map={
          "colossus_user_info_explore_outer_positive_photo_id_list_for_good_quality_size": "#(colossus_user_info_explore_outer_positive_photo_id_list_for_good_quality or {})",
        }
      ) \
    .end_() \
    .if_("enable_explore_negative_list_by_colossus_for_good_quality == 1") \
      .explore_colossus_v2_trigger_enrich(
        colossus_resp_attr = "colossus_resp_v2",
        output_colossus_trigger_attr = "colossus_user_info_explore_negative_photo_id_for_good_quality",
        enable_default_select_triggers = "{{enable_explore_default_select_negative_triggers_for_good_quality}}",
        enable_negtive_signals_triggers = "{{enable_explore_negative_triggers_for_good_quality}}",
        negtive_signals_triggers_min_days_ago = "{{explore_negtive_signals_triggers_min_days_ago_for_good_quality}}",
        negtive_signals_triggers_max_days_ago = "{{explore_negtive_signals_triggers_max_days_ago_for_good_quality}}",
        negtive_signals_triggers_select_num = "{{explore_negtive_signals_triggers_select_num_for_good_quality}}",
        negtive_signals_triggers_play_time_thres = "{{explore_negtitive_signals_triggers_play_time_thres_for_good_quality}}",
        enable_negtive_signals_triggers_short_play = "{{explore_enable_negtive_signals_triggers_short_play_for_good_quality}}",
        enable_negtive_signals_triggers_hate = "{{enable_explore_enable_negtive_signals_triggers_hate_for_good_quality}}",
        enable_select_explore_page_negtive_signals = "{{enable_explore_enable_select_explore_page_negtive_signals_for_good_quality}}",
        enable_get_longview_trigger = "{{enable_explore_negative_triggers_for_good_quality_get_longview_trigger}}",
      ) \
      .gen_common_attr_by_lua(
        attr_map={
          "colossus_user_info_explore_negative_photo_id_for_good_quality_size": "#(colossus_user_info_explore_negative_photo_id_for_good_quality or {})",
        }
      ) \
    .end_() \
    .pack_common_attr(
      input_common_attrs = [
        "unexpected_source_pids",
        "retrieval_only_bad_cover_input_item_key_list",
        "retrieval_only_bad_sense_input_item_key_list",
        "colossus_user_info__positive_action_photo_id_list",
        "retrieval_only_bad_hot_audit_input_item_key_list",
        "retrieval_only_bad_topk_audit_input_item_key_list",
        "fountain_splash_video_play_list",
        "colossus_user_info__negtive_action_photo_id_list",
        "sim_user_hate_photo_id_list",
        "colossus_user_info_explore_positive_photo_id_list_for_good_quality",
        "colossus_user_info_explore_outer_positive_photo_id_list_for_good_quality",
        "colossus_user_info_explore_negative_photo_id_for_good_quality",
        "retrieval_questionnaire_good_input_item_key_list"
      ],
      output_common_attr = "unexpected_source_pids",
      deduplicate = True
    ) \
    .if_("skip_explore_mc_click_emb_server == 0") \
      .get_remote_embedding_lite(
        kess_service = "grpc_hotMcEmbed",
        shard_num = 8,
        id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
        input_attr_name = "unexpected_source_pids",
        output_attr_name = "mc_embeddings_fr",
        query_source_type = "common_attr",
        size = 128,
        client_side_shard=True
      ) \
      .if_("enable_skip_hot_unexpected_score == 0") \
        .if_("enable_hot_unexpected_score_action_bucket_dot == 1") \
          .if_("enable_fr_cal_unexpected_score_hour_adjust == 1") \
            .enrich_attr_by_light_function(
              import_common_attr = [
                {"name": "uExploreActiveDays", "as": "user_vv"},
                {"name": "explore_fr_cal_unexpected_score_hour_adjust_exp_upper", "as": "exp_upper"},
                {"name": "explore_fr_cal_unexpected_score_hour_adjust_alpha", "as": "alpha"},
                {"name": "explore_fr_cal_unexpected_score_hour_adjust_beta", "as": "beta"},
                {"name": "explore_fr_cal_unexpected_score_hour_adjust_omega", "as": "omega"},
                {"name": "explore_fr_cal_unexpected_score_hour_adjust_max", "as": "coeff_max"},
                {"name": "explore_fr_cal_unexpected_score_hour_adjust_min", "as": "coeff_min"},
              ],
              export_common_attr = [
                {"name": "boost_weight", "as": "explore_fr_cal_unexpected_score_hour_boost_weight"},
              ],
              function_name = "AdjustWeightByUserVv",
              class_name = "ExploreLightFunctionSetV2",
            ) \
            .gen_common_attr_by_lua(
              attr_map = {
                "explore_hot_unexpected_score_hate_limit_min" : "explore_fr_cal_unexpected_score_hour_boost_weight * explore_hot_unexpected_score_hate_limit_min",
                "explore_hot_unexpected_score_report_limit_min" : "explore_fr_cal_unexpected_score_hour_boost_weight * explore_hot_unexpected_score_report_limit_min",
                "explore_hot_unexpected_score_not_click_limit_hour" : "explore_fr_cal_unexpected_score_hour_boost_weight * explore_hot_unexpected_score_not_click_limit_hour",
                "explore_hot_unexpected_score_not_click_weight" : "explore_fr_cal_unexpected_score_hour_boost_weight * explore_hot_unexpected_score_not_click_weight",
                "explore_hot_unexpected_score_short_view_weight" : "explore_fr_cal_unexpected_score_hour_boost_weight * explore_hot_unexpected_score_short_view_weight",
                "explore_hot_unexpected_score_extra_not_click_weight" : "explore_fr_cal_unexpected_score_hour_boost_weight * explore_hot_unexpected_score_extra_not_click_weight",
                "explore_hot_unexpected_score_hate_weight" : "explore_fr_cal_unexpected_score_hour_boost_weight * explore_hot_unexpected_score_hate_weight",
                "explore_hot_unexpected_score_report_weight" : "explore_fr_cal_unexpected_score_hour_boost_weight * explore_hot_unexpected_score_report_weight",
              }
            ) \
          .end_() \
          .explore_custom_embedding_score_enricher(
            user_info_ptr_attr = "user_info_ptr",
            embedding_list_attr = "mc_embeddings_fr",
            source_pids_list_attr = "unexpected_source_pids", # 在 user_info_module 里产出
            calc_type = "action_bucket_dot",
            export_item_attr = "fr_mc_embedding_score",
            dim_size = 128,
            check_point_ = "fr",
            enable_avg_pooling = "{{explore_hot_unexpected_score_enable_avg_pooling}}",
            not_click_weight = "{{explore_hot_unexpected_score_not_click_weight}}",
            short_view_weight = "{{explore_hot_unexpected_score_short_view_weight}}",
            extra_not_click_weight = "{{explore_hot_unexpected_score_extra_not_click_weight}}",
            hate_weight = "{{explore_hot_unexpected_score_hate_weight}}",
            report_weight = "{{explore_hot_unexpected_score_report_weight}}",
            like_weight = "{{explore_hot_unexpected_score_like_weight}}",
            follow_weight = "{{explore_hot_unexpected_score_follow_weight}}",
            not_click_limit_hour = "{{explore_hot_unexpected_score_not_click_limit_hour}}",
            extra_not_click_limit_hour = "{{explore_hot_unexpected_score_extra_not_click_limit_hour}}",
            play_stat_limit_hour = "{{explore_hot_unexpected_score_play_stat_limit_hour}}",
            pos_action_limit_hour = "{{explore_hot_unexpected_score_pos_action_limit_hour}}",
            hate_limit_min = "{{explore_hot_unexpected_score_hate_limit_min}}",
            report_limit_min = "{{explore_hot_unexpected_score_report_limit_min}}"
          ) \
        .else_() \
          .explore_custom_embedding_score_enricher(
            user_info_ptr_attr = "user_info_ptr",
            embedding_list_attr = "mc_embeddings_fr",
            source_pids_list_attr = "unexpected_source_pids", # 在 user_info_module 里产出
            calc_type = "combo_dot",
            export_item_attr = "fr_mc_embedding_score",
            dim_size = 128,
            check_point_ = "fr"
          ) \
        .end_() \
      .end_() \
      .if_("enable_explore_ranking_report_similar_score == 0") \
        .explore_custom_embedding_score_enricher(
          user_info_ptr_attr = "user_info_ptr",
          embedding_list_attr = "mc_embeddings_fr",
          source_pids_list_attr = "unexpected_source_pids", # 在 user_info_module 里产出
          calc_type = "action_bucket_dot",
          export_item_attr = "fr_report_mc_embedding_score",
          dim_size = 128,
          check_point_ = "fr",
          enable_avg_pooling = "{{explore_ranking_report_similar_score_enable_avg_pooling}}",
          not_click_weight = 0.0,
          short_view_weight = 0.0,
          report_weight = 1,
          not_click_limit_hour = 0.0,
          extra_not_click_limit_hour = 0.0,
          play_stat_limit_hour = 0,
          report_limit_min = "{{explore_ranking_report_similar_score_report_limit_min}}"
        ) \
      .end_() \
      .if_("enable_item_similarity_score == 1") \
        .explore_custom_embedding_score_enricher(
          user_info_ptr_attr = "user_info_ptr",
          embedding_list_attr = "mc_embeddings_fr",
          source_pids_list_attr = "unexpected_source_pids", # 在 user_info_module 里产出
          calc_type = "similarity",
          export_common_similarity_pid_list_attr = "common_similarity_pid_list",
          export_item_similarity_score_attr = "item_similarity_score",
          dim_size = 128,
          check_point_ = "fr"
        ) \
      .end_() \
      .if_("enable_explore_bad_item_list_similarity_score == 1") \
        .explore_custom_embedding_score_enricher(
          user_info_ptr_attr = "user_info_ptr",
          embedding_list_attr = "mc_embeddings_fr",
          source_pids_list_attr = "unexpected_source_pids",
          target_pids_list_attr = "retrieval_only_bad_cover_input_item_key_list",
          calc_type = "list_similarity",
          dim_size = 128,
          export_item_attr = "bad_cover_similary_score",
          select_item = {
            "attr_name": "audit_hot_cover_level",
            "compare_to": 0,
            "select_if": "<=",
            "select_if_attr_missing": True
          }   
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_fr_cal_quality_score_sub_coeff", "as": "sub_coeff"},
          ],
          import_item_attr = [
            {"name": "bad_cover_similary_score", "as": "score"},
          ],
          export_item_attr = [
            {"name": "score", "as": "good_cover_similary_score"}
          ],
          function_name = "CalExploreDoubleMinusDouble",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_() \
      .if_("enable_explore_sense_bad_item_list_similarity_score == 1") \
        .explore_custom_embedding_score_enricher(
          user_info_ptr_attr = "user_info_ptr",
          embedding_list_attr = "mc_embeddings_fr",
          source_pids_list_attr = "unexpected_source_pids",
          target_pids_list_attr = "retrieval_only_bad_sense_input_item_key_list",
          calc_type = "list_similarity",
          dim_size = 128,
          export_item_attr = "bad_sense_similary_score",
          select_item = {
            "attr_name": "audit_b_second_tag",
            "compare_to": 0,
            "select_if": "<=",
            "select_if_attr_missing": True
          }   
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_fr_cal_quality_score_sub_coeff", "as": "sub_coeff"},
          ],
          import_item_attr = [
            {"name": "bad_sense_similary_score", "as": "score"},
          ],
          export_item_attr = [
            {"name": "score", "as": "good_sense_similary_score"}
          ],
          function_name = "CalExploreDoubleMinusDouble",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_() \
      .if_("enable_explore_hot_audit_bad_item_list_similarity_score == 1") \
        .explore_custom_embedding_score_enricher(
          user_info_ptr_attr = "user_info_ptr",
          embedding_list_attr = "mc_embeddings_fr",
          source_pids_list_attr = "unexpected_source_pids",
          target_pids_list_attr = "retrieval_only_bad_hot_audit_input_item_key_list",
          calc_type = "list_similarity",
          dim_size = 128,
          export_item_attr = "bad_hot_audit_similary_score",
          select_item = {
            "attr_name": "audit_hot_high_tag_level",
            "compare_to": 0,
            "select_if": "<=",
            "select_if_attr_missing": True
          }   
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_fr_cal_quality_score_sub_coeff", "as": "sub_coeff"},
          ],
          import_item_attr = [
            {"name": "bad_hot_audit_similary_score", "as": "score"},
          ],
          export_item_attr = [
            {"name": "score", "as": "good_hot_audit_similary_score"}
          ],
          function_name = "CalExploreDoubleMinusDouble",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_() \
      .if_("enable_explore_topk_audit_bad_item_list_similarity_score == 1") \
        .explore_custom_embedding_score_enricher(
          user_info_ptr_attr = "user_info_ptr",
          embedding_list_attr = "mc_embeddings_fr",
          source_pids_list_attr = "unexpected_source_pids",
          target_pids_list_attr = "retrieval_only_bad_topk_audit_input_item_key_list",
          calc_type = "list_similarity",
          dim_size = 128,
          export_item_attr = "bad_topk_audit_similary_score",
          select_item = {
            "attr_name": "topk_audit_level",
            "compare_to": 0,
            "select_if": "<=",
            "select_if_attr_missing": True
          }   
        ) \
      .end_() \
      .if_("enable_explore_action_list_similar_score_by_colossus == 1") \
        .explore_custom_embedding_score_enricher(
          user_info_ptr_attr = "user_info_ptr",
          embedding_list_attr = "mc_embeddings_fr",
          source_pids_list_attr = "unexpected_source_pids",
          target_pids_list_attr = "colossus_user_info__positive_action_photo_id_list",
          calc_type = "list_similarity",
          dim_size = 128,
          export_item_attr = "user_positive_action_photo_similary_score"  
        ) \
      .end_() \
      .if_("enable_explore_negtive_list_similar_score_by_colossus == 1") \
        .explore_custom_embedding_score_enricher(
          user_info_ptr_attr = "user_info_ptr",
          embedding_list_attr = "mc_embeddings_fr",
          source_pids_list_attr = "unexpected_source_pids",
          target_pids_list_attr = "colossus_user_info__negtive_action_photo_id_list",
          calc_type = "list_similarity",
          dim_size = 128,
          export_item_attr = "user_negtive_action_photo_similary_score"  
        ) \
      .end_() \
      .if_("enable_explore_positive_similarity_score_for_good_quality == 1") \
        .explore_custom_embedding_score_enricher(
          user_info_ptr_attr = "user_info_ptr",
          embedding_list_attr = "mc_embeddings_fr",
          source_pids_list_attr = "unexpected_source_pids",
          target_pids_list_attr = "colossus_user_info_explore_positive_photo_id_list_for_good_quality",
          calc_type = "list_similarity",
          dim_size = 128,
          export_item_attr = "explore_positive_similarity_score_for_good_quality",
          target_item = {"is_good_author_pool_photo": 1}
        ) \
      .end_() \
      .if_("enable_explore_outer_positive_similarity_score_for_good_quality == 1") \
        .explore_custom_embedding_score_enricher(
          user_info_ptr_attr = "user_info_ptr",
          embedding_list_attr = "mc_embeddings_fr",
          source_pids_list_attr = "unexpected_source_pids",
          target_pids_list_attr = "colossus_user_info_explore_outer_positive_photo_id_list_for_good_quality",
          calc_type = "list_similarity",
          dim_size = 128,
          export_item_attr = "explore_outer_positive_similarity_score_for_good_quality",
          target_item = {"is_good_author_pool_photo": 1}
        ) \
      .end_() \
      .if_("enable_explore_negative_similarity_score_for_good_quality == 1") \
        .explore_custom_embedding_score_enricher(
          user_info_ptr_attr = "user_info_ptr",
          embedding_list_attr = "mc_embeddings_fr",
          source_pids_list_attr = "unexpected_source_pids",
          target_pids_list_attr = "colossus_user_info_explore_negative_photo_id_for_good_quality",
          calc_type = "list_similarity",
          dim_size = 128,
          export_item_attr = "explore_negative_similarity_score_for_good_quality",
          target_item = {"is_good_author_pool_photo": 1}
        ) \
      .end_() \
      .if_("enable_explore_sim_user_hate_item_list_similarity_score == 1 and recent_hate_count <= explore_koc_htr_count_threshold") \
        .explore_custom_embedding_score_enricher(
          user_info_ptr_attr = "user_info_ptr",
          embedding_list_attr = "mc_embeddings_fr",
          source_pids_list_attr = "unexpected_source_pids",
          target_pids_list_attr = "sim_user_hate_photo_id_list",
          calc_type = "list_similarity",
          dim_size = 128,
          export_item_attr = "hate_photo_id_similary_score"  
        ) \
      .end_() \
      .if_("enable_explore_questionnaire_good_item_list == 1 and user_age_segment > 0 and user_age_segment <= explore_questionnaire_age_threshold") \
        .explore_custom_embedding_score_enricher(
          user_info_ptr_attr = "user_info_ptr",
          embedding_list_attr = "mc_embeddings_fr",
          source_pids_list_attr = "unexpected_source_pids",
          target_pids_list_attr = "retrieval_questionnaire_good_input_item_key_list",
          calc_type = "list_similarity",
          dim_size = 128,
          export_item_attr = "user_questionnaire_good_photo_similary_score"  
        ) \
      .end_() \
    .end_()


  def post_process(self) -> None:
    pass
