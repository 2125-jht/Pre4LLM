from ranking import CommonModule

class RankingScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
    .if_("skip_photo_life_filter == 0") \
    .enrich_attr_by_lua(
      import_common_attr = [
        "enable_follow_author_retr_skip_26h_filter",
        "normal_photo_life_time_hours"
      ],
      import_item_attr = [
        "upload_time",
        "long_term_photo",
        "is_follow_author"
      ],
      export_item_attr = [
        "is_filter"
      ],
      function_for_item = "upload_time_filer",
      lua_script_file = "life/ranking/lua/module/ranking_score__upload_time_filter.lua",
    ) \
    .filter_by_attr(
      attr_name = "is_filter",
      remove_if = "==",
      compare_to = 1
    ) \
    .end_() \
    .explore_embedding_candidates_attr_enricher(
      trans_type = "embedding_candidates",
      user_info_ptr_attr = "user_info_ptr",
      export_common_attr = "unexpected_source_pids",
      check_point = "fr"
    ) \
    .enrich_attr_by_lua(
      import_item_attr = ["reason"],
      export_item_attr = ["reason_str"],
      function_for_item = "trans_reason_to_str",
      lua_script_file = "life/ranking/lua/module/ranking_score__trans_reason_to_str.lua"
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
    .delegate_enrich(
      skip = "{{skip_explore_fullrank_sim_predict}}",
      kess_service = "{{xlife_gamora_common_predict_server_kess_name}}",
      partition_size = "{{ranking_server_partition_size}}",
      recv_item_attrs = [
        { "name": "ctr", "as": "pctr" },
        { "name": "ltr", "as": "pltr" },
        { "name": "wtr", "as": "pwtr" },
        { "name": "ftr", "as": "pftr" },
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
        # picture
        { "name": "pic_wtdPlaytime", "as": "pic_wtd"},
        { "name": "pic_lvtr", "as": "pic_lvtr"},
        { "name": "pic_cpr", "as": "pic_cpr"},
      ],
      request_type = "{{gamora_common_predict_server_request_type}}",
      send_common_attrs = [
        { "name": "userInfo", "as": "user_info_str" },
      ],
      send_item_attrs = [
        "live_photo_info__is_living",
        "reco_photo_info_str"
      ],
    ) \
    .if_("set_default_value_pxtr == 1") \
    .set_default_value(
      no_overwrite=True,
      item_attrs=[
        {
          "name": "pctr",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "pltr",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "pwtr",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "pftr",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "psvr",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "pcmtr",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "pptr",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "pcmef",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "phtr",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "pevtr",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "plvtr",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "fr_score1",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "fr_score2",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "pepstr",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "pdtr",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "pcltr",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "fetr",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "fountain_eff",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "pliving_ctr",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "pliving_wtr",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "pfvtr",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "awesome_wtd",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "pdctr",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "pvtr",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "adaptive_wtd_v2",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "cpr",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "wtd_evtr",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "wtd_lvtr",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "future_xtr",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "pic_wtd",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "pic_lvtr",
          "type": "double",
          "value": -1.0
        },
        {
          "name": "pic_cpr",
          "type": "double",
          "value": -1.0
        },
      ]
    ) \
    .end_() \
    .if_("skip_explore_mc_click_emb_server == 0") \
      .if_("life_neg_sim_use_hot_mc_emb_server == 1") \
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
      .else_() \
        .get_remote_embedding_lite(
          kess_service = "grpc_hotClickEmbeddingServer",
          shard_num = 1,
          id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
          input_attr_name = "unexpected_source_pids",
          output_attr_name = "mc_embeddings_fr",
          query_source_type = "common_attr",
          size = 136
        ) \
      .end_() \
    .end_() \
    .if_("enable_skip_hot_unexpected_score == 0") \
      .if_("life_neg_sim_use_hot_mc_emb_server == 1") \
        .explore_life_custom_embedding_score_enricher(
          user_info_ptr_attr = "user_info_ptr",
          embedding_list_attr = "mc_embeddings_fr",
          source_pids_list_attr = "unexpected_source_pids", # 在 user_info_module 里产出
          calc_type = "combo_dot",
          export_item_attr = "fr_mc_embedding_score",
          dim_size = 128,
          play_stat_limit_hour = "{{life_neg_sim_play_stat_limit_hour}}",
          short_view_threshold = "{{life_neg_sim_short_view_threshold}}",
          play_stat_limit_explore = "{{life_neg_sim_play_stat_limit_explore}}",
          enable_not_click = "{{life_neg_sim_enable_not_click}}",
          check_point_ = "fr"
        ) \
      .else_if_("life_neg_sim_use_mmu_emb_server == 1") \
        .explore_life_custom_embedding_score_enricher(
          user_info_ptr_attr = "user_info_ptr",
          embedding_list_attr = "mmu_embeddings",
          source_pids_list_attr = "embedding_source_pids", # 在 user_info_module 里产出
          calc_type = "combo_dot",
          export_item_attr = "fr_mc_embedding_score",
          dim_size = 64,
          play_stat_limit_hour = "{{life_neg_sim_play_stat_limit_hour}}",
          short_view_threshold = "{{life_neg_sim_short_view_threshold}}",
          play_stat_limit_explore = "{{life_neg_sim_play_stat_limit_explore}}",
          enable_not_click = "{{life_neg_sim_enable_not_click}}",
          check_point_ = "fr"
        ) \
      .else_() \
        .explore_custom_embedding_score_enricher(
          user_info_ptr_attr = "user_info_ptr",
          embedding_list_attr = "mc_embeddings_fr",
          source_pids_list_attr = "unexpected_source_pids", # 在 user_info_module 里产出
          calc_type = "combo_dot",
          export_item_attr = "fr_mc_embedding_score",
          dim_size = 136,
          check_point_ = "fr"
        ) \
      .end_() \
    .end_() \
    .enrich_attr_by_lua(
      import_item_attr = [
        "fr_score2",
      ],
      export_item_attr = [
        "fr_score2",
      ],
      function_for_item = "fr_score2_change",
      lua_script_file = "life/ranking/lua/module/ranking_score__fr_score2.lua",
      skip = "{{explore_skip_fr_score2_change}}"
    ) \
    .set_attr_default_value(
      skip = "{{skip_pcltr_set_default_value}}",
      item_attrs = [
        {
          "name": "pcltr",
          "type": "double",
          "value": -1.0
        }
      ]
    ) \
    .sort(
      score_from_attr = "pcltr",
      partial_sort = True,
      partial_num = "{{fr_pctr_filter_top_pcltr_save_num}}"
    ) \
    .enrich_attr_by_lua(
      import_common_attr = [
        "exploreRank_cls_pctr_filter_flag",
        "enable_produce_v4_fr_refactor",
        "exploreRank_cls_pctr_filter_threshold",
        "exploreRank_cls_pctr_filter_threshold_old",
        "exploreRank_cls_pctr_filter_remitted_exptags",
        "enable_audit_hot_skip_rank_pctr_filter",
        "enable_user_high_level_skip_pctr_filter",
        "fr_pctr_filter_top_pcltr_save_num",
        "user_risk_level",
        "user_risk_min",
        "enable_follow_author_pwtr_corr",
        "ranking_follow_author_pwtr_corr_coef",
        "explore_fullrank_calibration_ctr_param",
        "enable_picture_skip_pctr_filter",
      ],
      import_item_attr = [
        "pctr",
        "fetr",
        "fountain_eff",
        "reason",
        "audit_hot_high_tag_level",
        "pcmtr",
        "pwtr",
        "is_follow_author",
        "is_picture",
      ],
      export_item_attr = [
        "corr_pctr",
        "corr_fetr",
        "corr_fountain_eff",
        "is_satisfy_ctr_filter",
        "pctr_x_pcmtr",
        "save_pcltr_top_n",
        "corr_pwtr"
      ],
      function_for_item = "pxtr_change",
      lua_script_file = "life/ranking/lua/module/ranking_score__pxtr_change.lua",
      skip = "{{explore_skip_pxtr_change}}"
    ) \
    .if_("enable_life_cal_corr_pctr_psvr == 1") \
      .cal_corr_pctr_psvr() \
    .end_() \
    .enrich_attr_by_light_function(
      import_common_attr = [
        "explore_wtd_evtr_pctr_weight",
        "explore_wtd_lvtr_pctr_weight",
        "explore_future_xtr_pctr_weight",
        "explore_pic_wtd_pctr_weight",
        "explore_pic_lvtr_pctr_weight",
        "explore_pic_cpr_pctr_weight",
        "hot_fr_pic_cpr_max_pic_cnt",
      ],
      import_item_attr = [
        "corr_pctr",
        "wtd_evtr",
        "wtd_lvtr",
        "future_xtr",
        "pic_wtd",
        "pic_lvtr",
        "pic_cpr",
        "photo_picture_count",
      ],
      export_item_attr = [
        "corr_wtd_evtr",
        "corr_wtd_lvtr",
        "corr_future_xtr",
        "corr_pic_wtd",
        "corr_pic_lvtr",
        "corr_pic_cpr",
      ],
      function_name = "FrPxtrChange",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_lua(
      import_item_attr = [
        "corr_pctr",
        "pwtr",
        "fr_score2",
        "pltr",
        "pcmtr",
        "pepstr",
        "psvr",
        "plvtr"
      ],
      export_item_attr = [
        "svr_act_score"
      ],
      function_for_item = "svr_act_queue",
      lua_script_file = "life/ranking/lua/module/ranking_score__pxtr_change.lua"
    ) \
    .explore_life_custom_embedding_score_enricher(
      skip = "{{enable_skip_hot_expected_score}}",
      user_info_ptr_attr = "user_info_ptr",
      embedding_list_attr = "mmu_embeddings",
      source_pids_list_attr = "embedding_source_pids", # 在 user_info_module 里产出
      calc_type = "action_bucket_dot",
      short_view_threshold = "{{life_neg_sim_mmu_short_view_threshold}}",
      short_view_weight = "{{life_neg_sim_mmu_short_view_weight}}",
      hate_weight = "{{life_neg_sim_mmu_hate_weight}}",
      not_click_weight = "{{life_neg_sim_mmu_not_click_weight}}",
      not_click_limit_hour = "{{life_neg_sim_mmu_not_click_limit_hour}}",
      play_stat_limit_hour = "{{life_neg_sim_mmu_play_stat_limit_hour}}",
      export_item_attr = "fr_mmu_embedding_score",
      dim_size = 64,
      check_point_ = "fr"
    )

    self.flow \
    .if_("enable_report_discount_cal == 1") \
      .calc_report_discount() \
    .end_() \
    .calc_hate_discount() \
    .if_("enable_pcmef_gender_debias == 1") \
      .calc_pcmef_gender_debias_score() \
    .end_() \
    .gen_score_stage1()

  def post_process(self) -> None:
    self.flow \
      .if_("_IS_PERF_SAMPLING_REQUEST_ == 1 and skip_pcltr_set_default_value == 0") \
      .count_reco_result(
        save_count_to = "save_pcltr_top_n_sum",
        target_item = {"save_pcltr_top_n": 1}
      ) \
      .perflog_attr_value(
        check_point = "ranking_score",
        common_attrs=[
          "save_pcltr_top_n_sum"
        ]
      ) \
      .end_()