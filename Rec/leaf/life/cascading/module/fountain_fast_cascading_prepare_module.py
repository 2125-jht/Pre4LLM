from cascading import CommonModule

class FountainFastCascadingPrepareModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
    .copy_user_meta_info(
      save_request_type_to_attr = "common_request_type"
    ) \
    .enrich_attr_by_lua(
      import_common_attr=[
        "common_request_type"
      ],
      export_common_attr=[
        "fountain_casade_is_fast",
      ],
      function_for_common="cascade_control_model",
      lua_script_file="./life/cascading/lua/module/fountain_splash_cascading_predict__control.lua",
    )
    self._prepare()
    self._enrich_more_user_features()
    self._hack_attr()
    return self

  def _hack_attr(self): # hack一些后面需要的attr
    self.flow \
    .set_attr_value(
      no_overwrite=True,
      item_attrs = [
        {
          "name": "cascade_pcltr_fractile_score",
          "type": "double",
          "value": 0.0,
        },
        {
          "name": "cascade_pcmtr_fractile_score",
          "type": "double",
          "value": 0.0,
        },
        {
          "name": "cascade_pctr_fractile_score",
          "type": "double",
          "value": 0.0,
        },
        {
          "name": "cascade_pepstr_fractile_score",
          "type": "double",
          "value": 0.0,
        },
        {
          "name": "cascade_pltr_fractile_score",
          "type": "double",
          "value": 0.0,
        },
        {
          "name": "cascade_pwatch_time_fractile_score",
          "type": "double",
          "value": 0.0,
        },
        {
          "name": "cascade_pwtd_fractile_score",
          "type": "double",
          "value": 0.0,
        },
        {
          "name": "cascade_pwtr_fractile_score",
          "type": "double",
          "value": 0.0,
        },
        {
          "name": "cascade_slide_kai_fractile_score",
          "type": "double",
          "value": 0.0,
        },
        {
          "name": "cascade_wtd_kai_mix",
          "type": "double",
          "value": 0.0,
        },
        {
          "name": "is_follow_author",
          "type": "int",
          "value": 0
        }
      ],
    )
    return self

  def _prepare(self):
    self.flow \
    .transform_item_attr(
      mappings = [{
        "check_attr_name": "author__id",
        "check_attr_type": "int",
        "output_attr_name": "is_photo_author_followed",
        "output_attr_type": "int",
        "rules": [{
          "check_values": [ "{{followAuthors}}" ],
          "output_value": 1
        }],
      }]) \
    .transform_item_attr(
      mappings = [{
        "check_attr_name": "upload_type",
        "check_attr_type": "int",
        "output_attr_name": "picture_variant_attr",
        "output_attr_type": "int",
        "rules": [{
          "check_values": [ 10, 11],
          "output_value": 1
        }]
      }])
    self._get_colossus_v2()
    self.flow \
    .explore_memory_data_enrich(
      data_key = "{{fountain_global_hetu_distribution_map}}",
      data_type = "string_int32_map",
      save_data_ptr_to_attr = "fountain_latest_global_hetu_distribution_map",
    ) \
    .explore_photo_distribution_colossus_stat_enricher(
      enable_only_fountain_stat = "{{fountain_hetu_distribution_stat_only_fountain}}",
      enable_only_positive_stat = "{{fountain_hetu_distribution_stat_only_positive}}",
      colossus_resp_attr = "colossus_resp_v2",
      save_total_count = "colossus_hetu_distribution_total_count",
      save_user_hetu_stat_attr = "colossus_hetu_distribution_hetu_stat",
      interest_stat_use_reward = "{{fountain_interest_stat_use_reward}}",
      interest_stat_vv_weight = "{{fountain_interest_stat_vv_weight}}",
      interest_stat_reward_weight = "{{fountain_interest_stat_reward_weight}}",
      interest_stat_avg_reward_smooth = "{{fountain_interest_stat_avg_reward_smooth}}",
      enable_interest_stat_avg_reward = "{{fountain_enable_interest_stat_avg_reward}}",
      minus_hate_stat_coeff = "{{fountain_interest_stat_minus_hate_coeff}}",
      minus_sv_stat_coeff = "{{fountain_interest_stat_minus_sv_coeff}}",
    ) \
    .if_("fountain_enable_save_user_mixed_hetu_stat == 1") \
      .explore_mix_user_interest_stat_enricher(
        user_hetu_stat_attr = "colossus_hetu_distribution_hetu_stat",
        global_hetu_stat_attr = "fountain_latest_global_hetu_distribution_map",
        global_hetu_stat_redis_key_prefix = "{{fountain_global_hetu_stat_redis_key_prefix}}",
        enable_debias_with_global_stat = "{{fountain_enable_debias_with_global_stat}}",
        enable_debias_multipy_original_stat = "{{fountain_enable_debias_multipy_original_stat}}",
        global_fuse_corr = "{{fountain_user_interest_global_fuse_corr}}",
        save_user_mixed_hetu_stat_attr = "user_mixed_interest_stat"
      ) \
    .end_if_()
    self._get_emp_xtr()
    self.flow \
    .split_string(
      input_common_attr="fountain_mc_colossus_short_interest_reward_weights",
      output_common_attr="colossus_short_interest_reward_weights_list",
      delimiters=",",
      parse_to_double=True,
    ) \
    .split_string(
      input_common_attr="fountain_mc_colossus_long_interest_reward_weights",
      output_common_attr="colossus_long_interest_reward_weights_list",
      delimiters=",",
      parse_to_double=True,
    ) \
    .split_string(
      input_common_attr="fountain_mc_colossus_explore_interest_reward_weights",
      output_common_attr="colossus_explore_interest_reward_weights_list",
      delimiters=",",
      parse_to_double=True,
    ) \
    .explore_life_interest_hetu_enricher(
      colossus_resp_attr = "colossus_resp_v2",
      # 统计 top 短播河图
      save_user_top_sv_hetu_attr = "colossus_hetu_emp_svtr_stat",
      enable_top_sv_hetu2 = "{{xlife_fountain_enable_top_sv_hetu2}}",
      enable_stat_top_sv_only_fountain = "{{xlife_fountain_enable_stat_top_sv_only_fountain}}",
      top_sv_stat_max_show = "{{xlife_fountain_top_sv_stat_max_show}}",
      enable_top_sv_stat_use_rate = "{{xlife_enable_fountain_top_sv_stat_use_rate}}",
      top_sv_stat_default_svtr = "{{xlife_fountain_top_sv_stat_default_svtr}}",
      top_sv_stat_base_show = "{{xlife_fountain_top_sv_stat_base_show}}",
      # 统计长短期兴趣
      enable_stat_short_interest = "{{xlife_fountain_mc_enable_stat_short_interest}}",
      enable_stat_long_interest = "{{xlife_fountain_mc_enable_stat_long_interest}}",
      enable_stat_explore_interest = "{{xlife_fountain_mc_enable_stat_explore_interest}}",
      get_short_interest_attr = "short_interest",
      save_short_interest_attr = "short_interest",
      save_long_interest_attr = "long_interest",
      save_explore_interest_attr = "random_explore_interest",
      short_interest_reward_weights_attr = "colossus_short_interest_reward_weights_list",
      long_interest_reward_weights_attr = "colossus_long_interest_reward_weights_list",
      explore_interest_reward_weights_attr = "colossus_explore_interest_reward_weights_list",
      enable_stat_short_interest_only_explore_fountain = "{{xlife_fountain_mc_enable_stat_short_interest_only_explore_fountain}}",
      short_interest_max_hours = "{{xlife_fountain_cascade_short_interest_limit_hour}}",
      enable_interest_use_hetu1 = "{{xlife_fountain_cascade_interest_use_level_one}}",
      play_time_slope = "{{xlife_fountain_mc_interest_reward_play_time_slope}}",
      play_time_max = "{{xlife_fountain_mc_interest_reward_play_time_max}}",
      enable_stat_long_interest_only_explore_fountain = "{{xlife_fountain_mc_enable_stat_long_interest_only_explore_fountain}}",
      long_interest_max_days = "{{xlife_fountain_cascade_longterm_interest_max_history_days}}",
      long_interest_min_days = "{{xlife_fountain_cascade_longterm_interest_min_history_days}}",
      long_interest_min_play_time = "{{xlife_fountain_cascade_interest_min_play_second}}",
      short_interest_reward_lower_bound = "{{xlife_fountain_mc_short_interest_reward_lower_bound}}",
      short_interest_num = "{{xlife_fountain_colossus_short_interest_max_num}}",
      long_interest_reward_lower_bound = "{{xlife_fountain_mc_long_interest_reward_lower_bound}}",
      long_interest_num = "{{xlife_fountain_colossus_longterm_interest_max_num}}",
      explore_interest_reward_lower_bound = "{{xlife_fountain_mc_explore_interest_reward_lower_bound}}",
      explore_interest_num = "{{xlife_fountain_colossus_explore_interest_max_num}}",
      enable_interest_reward_use_rate = "{{xlife_fountain_cascade_interest_calc_use_percent}}",
    )
    self._interactive_emp_xtr_change()
    self.flow \
    .enrich_attr_by_lua(
      import_common_attr = [
        "enableFountainFullrankExp",
        "fullrank_fast_before_variant_mc_limit_size",
        "fullrank_fast_before_variant_mc_limit_size_exp",
        "increase_quota_status",
        "fountain_mc_increase_quota_factor_list",
        "increase_quota_window_len",
        "increase_quota_current_index"
      ],
      import_item_attr = [
        "hetu_tag_level_info__hetu_level_one",
        "hetu_tag_level_info__hetu_level_two",
        "hetu_tag_level_info_v2__hetu_level_one",
        "duration_ms",
        "explore_stat__show_count",
        "explore_stat__negative_count",
      ],
      export_item_attr = [
        "hetu_level_one_index",
        "hetu_level_one_v2_index_cascade",
        "duration_s",
        "hetu_level_one",
        "hetu_level_two",
        "emp_htr",
        "duration_perf_id"
      ],
      export_common_attr = [
        "fullrank_fast_before_variant_mc_limit_size",
      ],
      function_for_common = "cascade_control_fast",
      function_for_item = "cascade_feature_trans",
      lua_script_file = "./life/cascading/lua/module/fountain_splash_cascading_predict__control.lua",
    ) \
    .if_("enable_fountain_calc_xhs_target_qualified_photo == 1") \
      .get_kconf_params(
        kconf_configs = [{
          "kconf_key": "reco.hot.exploreKnowledgeHetuSetExpNew",
          "value_type": "set_int64",
          "default_value": [4,5,25],
          "export_common_attr": "xhs_target_hetu_set"
        },{
          "kconf_key": "reco.hot.exploreXhsTargetContentAuditBResultSet",
          "value_type": "set_int64",
          "default_value": [2000866,2019671,2019672,2022202,2022203],
          "export_common_attr": "xhs_target_audit_b_set"
        }]
      ) \
    .end_() \
    .if_("disable_merchant_explore_all_photo_optimize == 0 and enable_fountain_merchant_photo_calc_type == 1") \
      .explore_memory_data_enrich(
        data_key = "merchant_live_authors_set",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "merchant_live_authors_set__memory_data",
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          {"name": "author__id", "as": "author__id"},
          {"name": "merchant_item_info__item_id_list", "as": "cart_itemlist"},
          {"name": "merchant_photo_cart_relation", "as": "cart_relation"},
        ],
        import_common_attr = [
          "merchant_live_authors_set__memory_data",
        ],
        export_item_attr = [
          "is_merchant_cart",
          "is_merchant_living",
          "merchant_author_in_living"
        ],
        function_name = "MerchantGetAuthorInLiving",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("enable_hot_content_thompson_sampling_corr_calculate == 1") \
      .set_attr_value(
        no_overwrite = True,
        common_attrs = [
          {
            "name": "hot_content_thompson_sampling_exp_tag_list",
            "type": "int_list",
            "value": [341, 416],
          }
        ]
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "user_info_ptr", "as": "user_info_ptr"},
          {"name": "hot_content_thompson_sampling_positive_corr", "as": "positive_corr"},
          {"name": "hot_content_thompson_sampling_negative_corr", "as": "negative_corr"},
          {"name": "hot_content_thompson_sampling_exp_tag_list", "as": "exp_tag_list"},
          {"name": "hot_content_thompson_sampling_type", "as": "reward_type"},
        ],
        export_common_attr = [
          "hot_content_positive_count",
          "hot_content_negative_count",
        ],
        function_name = "GetFountainHotContentCount",
        class_name = "ExploreLightFunctionSetV2"
      ) \
    .end_() \
    .if_("life_fountain_enable_adjust_marketing_compensation_photo == 1") \
      .fountain_gen_is_marketing_compensation_photo() \
    .end_() \
    .if_("enable_life_fountain_gen_is_low_cost_photo == 1") \
      .gen_is_low_cost_photo() \
    .end_() \
    .if_("enable_life_fountain_fast_gen_minority_photo == 1") \
      .gen_is_minority_photo() \
    .end_() \

    return self
  
  def _get_colossus_v2(self):
    self.flow \
    .colossus(
      service_name = "grpc_colossusSimV2",
      client_type = "common_item_client",
      output_attr = "colossus_resp_v2",
      parse_to_pb = False
    )

    return self
  
  def _get_emp_xtr(self):
    self.flow \
    .explore_user_emp_xtr_enricher(
      colossus_resp_attr = "colossus_resp_v2",
      save_user_click_count = "user_colossus_click_count",
      save_user_emp_ltr = "user_emp_ltr",
      save_user_emp_wtr = "user_emp_wtr",
      save_user_emp_ftr = "user_emp_ftr",
      save_user_emp_htr = "user_emp_htr",
      save_user_emp_cmtr = "user_emp_cmtr",
      save_user_emp_eptr = "user_emp_eptr",
      save_user_emp_svtr = "user_emp_svtr",
      save_user_emp_evtr = "user_emp_evtr",
      save_user_emp_lvtr = "user_emp_lvtr",
      save_user_emp_fintr = "user_emp_fintr",
      save_user_emp_watch_time = "user_emp_watch_time",
      save_user_emp_finish_rate = "user_emp_finish_rate",
      save_user_emp_watch_time_long_video = "user_emp_watch_time_long_video",
      save_user_emp_finish_rate_long_video = "user_emp_finish_rate_long_video",
      use_fountain_count_threshold = "{{use_fountain_count_threshold}}"
    ) \
    .perflog_attr_value(
      check_point="fountain.fast.emp_xtr",
      common_attrs=["user_colossus_click_count",
        "user_emp_ltr","user_emp_wtr",
        "user_emp_ftr","user_emp_htr",
        "user_emp_cmtr","user_emp_eptr",
        "user_emp_watch_time", "user_emp_finish_rate",
        "user_emp_watch_time_long_video",
        "user_emp_finish_rate_long_video",
        ]
    ) \
    .log_debug_info(
      common_attrs = [
        "user_emp_watch_time",
        "user_emp_finish_rate",
        "user_emp_watch_time_long_video",
        "user_emp_finish_rate_long_video",
      ],
      for_debug_request_only = True,
    )

    return self
  
  def _interactive_emp_xtr_change(self):
    self.flow \
    .enrich_attr_by_lua(
      import_common_attr = [
        "user_emp_ltr",
        "user_emp_wtr",
        "user_emp_cmtr",
        "user_emp_ftr",
        "user_emp_eptr"
      ],
      export_common_attr = [
        "userExpLtr",
        "userExpWtr",
        "userExpCmtr",
        "userExpFtr",
        "userExpEptr"
      ],
      function_for_common = "emp_xtr_change",
      lua_script_file = "./life/cascading/lua/module/fountain_splash_cascading_predict__control.lua"
    )

    return self
  
  def _enrich_more_user_features(self):
    """
    填充更多用户特征, 粗精排共用
    """
    self.flow \
    .explore_common_user_feature_enricher(
      skip = "{{fountain_skip_fullrank_deep_ltr_use_more_user_features}}",
      user_info_attr = "user_info_ptr",
      context_hour_of_day_attr = "userRequestHour",
      context_day_of_week_attr = "userRequestDayOfWeek",
      user_find_active_degree_attr = "uFindUserActiveDegree",
      user_low_active_attr = "uIsLowActiveUser",
      user_ft_realtime_like_count = "uFountainRealtimeLikeCountAttr",
      user_ft_realtime_follow_count = "uFountainRealtimeFollowCountAttr",
      user_ft_realtime_forward_count = "uFountainRealtimeForwardCountAttr",
      user_ft_realtime_comment_count = "uFountainRealtimeCommentCountAttr",
      user_ft_realtime_short_view_count = "uFountainRealtimeShortViewCountAttr",
      user_ft_realtime_long_view_count = "uFountainRealtimeLongViewCountAttr",
      user_ft_realtime_effective_view_count = "uFountainRealtimeEffectiveViewCountAttr",
      user_ft_realtime_finish_view_count = "uFountainRealtimeFinishViewCountAttr",
      user_ft_realtime_count_time_threshold = "userFtRealtimeCountTimeThreshold",
    )
    return self