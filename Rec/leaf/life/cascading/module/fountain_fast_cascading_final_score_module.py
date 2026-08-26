from cascading import CommonModule
from cascading.module.fountain_fast_cascading_queues import *

class FountainFastCascadingFinalScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
    .if_("enable_xlife_fountain_topk_mgs_score == 1") \
      .explore_life_embedding_candidates_attr_enricher(
        trans_type = "fountain_candidates",
        enable_fix_low_hit_rate = True,
        enable_not_click = False,
        enable_play_stat = True,
        enable_hate = False,
        enable_explore_not_click = False,
        enable_source_photo = True,
        source_pid_attr = "featureSourcePId",
        session_history_max_size = "{{fountain_mc_mgs_diversity_max_size}}",
        user_info_ptr_attr = "user_info_ptr",
        export_common_attr = "topk_mgs_embedding_source_pids",
        check_point = "cascade",
      ) \
      .get_remote_embedding_lite(
        kess_service = "grpc_MMUHetuSimContentEmbedding",
        shard_num = 4,
        id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
        input_attr_name = "topk_mgs_embedding_source_pids",
        output_attr_name = "topk_mgs_embeddings",
        query_source_type = "common_attr",
        size = 64,
        client_side_shard = True
      ) \
      .explore_get_embedding_map_enricher(
        embedding_list_attr = "topk_mgs_embeddings",
        source_pids_list_attr = "topk_mgs_embedding_source_pids",
        dim_size = 64,
        export_common_attr = "topk_mgs_pid_embedding_map",
      ) \
      .explore_diversity_update_enricher(
        user_info_ptr_attr = "user_info_ptr",
        pid_embedding_common_attr = "topk_mgs_pid_embedding_map",
        export_item_attr = "topk_mgs_expected_score",
        history_feed_back_version = 3,
        dim_size = 64,
        expected_score_cand_size = "{{fountain_mc_topk_mgs_expected_score_cand_num}}",
        max_interval_second = "{{fountain_mc_topk_mgs_expected_score_max_interval_second}}",
        min_duration_threshold = "{{fountain_mc_topk_mgs_expected_score_min_duration_threshold}}",
        dpp_diversity_mgs_topk = "{{fountain_mc_topk_mgs_expected_score_topk_num}}",
        max_playtime_threshold = "{{fountain_mc_topk_mgs_expected_score_max_playtime_threshold}}",
        min_playtime_threshold = "{{fountain_mc_topk_mgs_expected_score_min_playtime_threshold}}",
        enable_use_weight = "{{fountain_mc_topk_mgs_expected_score_enable_use_weight}}",
        weight_version = "{{fountain_mc_topk_mgs_expected_score_weight_version}}",
        ratio_scale = "{{fountain_mc_topk_mgs_expected_score_ratio_scale}}",
        ratio_pow_weight = "{{fountain_mc_topk_mgs_expected_score_ratio_pow_weight}}",
      ) \
    .end_() \
    .calc_long_term_interest_ee_score(
      user_info_pb_name = "user_info_ptr",
      hetu_attrs = "hetu_tag_level_info__hetu_level_one;hetu_tag_level_info__hetu_level_two;hetu_tag_level_info__hetu_level_three;hetu_tag_level_info__hetu_level_four;hetu_tag_level_info__hetu_face_id;hetu_tag_level_info__hetu_tag",
      enable_click_history = "{{fountain_mc_enable_click_history}}",
      enable_like_history = "{{fountain_mc_enable_like_history}}",
      enable_follow_history = "{{fountain_mc_enable_follow_history}}",
      enable_long_view_history = "{{fountain_mc_enable_long_view_history}}",
      long_view_threshold = "{{fountain_mc_long_view_threshold}}",
      export_item_attr = "cascade_long_term_interest_ee_score",
      enable_division_way = "{{fountain_mc_enable_division_way}}",
      photo_hetu_tag_level_info_type = "{{foutnain_mc_photo_hetu_tag_level_info_type}}",
      boost_threshold = "{{fountain_mc_long_term_interest_ee_boost_threshold}}",
    ) \
    .get_kconf_params(
      kconf_configs = [{
        "kconf_key": "reco.fountain.highValueHetuList",
        "value_type": "list_int64",
        "defult_value": [134, 120, 114, 189, 220, 316, 179, 199, 325, 161, 208, 203],
        "export_common_attr": "high_value_hetu_list"
      }]
    ) \
    .pack_item_attr(
      item_source = {
        "reco_results": True,
      },
      mappings = [{
        "from_item_attr": "hetu_level_one_v2_index_cascade",
        "to_common_attr": "hetu_level_one_v2_index_cascade_list_no_dedup",
        "dedup_to_common_attr": False,
      }],
    ) \
    .enrich_attr_by_lua(
      import_common_attr = [
        "fountain_mc_high_value_hetu_debias_coef",
        "fountain_mc_enable_only_longterm_debias",
        "high_value_hetu_list",
        "fountain_mc_enable_lt_weight_adjust",
        "hetu_level_one_v2_index_cascade_list_no_dedup",
        "fountain_fast_ensemble_weight_cascade_long_term_interest_ee_score",
        "fountain_mc_lt_weight_adjust_threshold",
        "fountain_mc_lt_weight_adjust_coef",
      ],
      import_item_attr = [
        "cascade_long_term_interest_ee_score",
        "hetu_tag_level_info_v2__hetu_level_one",
        ],
      export_item_attr = [
        "cascade_long_term_interest_ee_score",
      ],
      export_common_attr = [
        "fountain_fast_ensemble_weight_cascade_long_term_interest_ee_score"
      ],
      function_for_item = "calc_mc_high_value_hetu_debias",
      function_for_common = "calc_mc_max_hetu_one_rate",
      lua_script_file = "./life/cascading/lua/module/fountain_fast_cascading_score__high_value_hetu_debias.lua",
    ) \
    .if_('fountain_fast_cascade_ensemble_enable_personally_weight == 1') \
      .get_common_attr_from_redis(
      cluster_name = "recoNewUserPhotos",
      timeout_ms = 10,
      cache_bits = 2,
      cache_expire_second = 600,
      redis_params = [
        {
          "redis_key": "{{fountain_cascade_personnally_maxtrix_redis_key}}",
          "output_attr_name": "fountain_cascade_ensemble_cem_maxtrix"
        },
        {
          "redis_key": "{{fountain_cascade_personnally_feature_redis_key}}",
          "output_attr_name": "fountain_cascade_ensemble_cem_feature"
        },
      ]
    ) \
    .pack_item_attr(
      item_source = {"reco_results": True},
      mappings = [
        {
          "from_item_attr": "cascade_pwtd",
          "to_common_attr": "cascade_pwtd_avg",
          "aggregator":"avg"
        },
        {
          "from_item_attr": "cascade_pwtd",
          "to_common_attr": "cascade_pwtd_dev",
          "aggregator":"dev"
        },
        {
          "from_item_attr": "cascade_pctr",
          "to_common_attr": "cascade_pctr_avg",
          "aggregator":"avg"
        },
        {
          "from_item_attr": "cascade_pctr",
          "to_common_attr": "cascade_pctr_dev",
          "aggregator":"dev"
        },
         {
          "from_item_attr": "cascade_pwatch_time",
          "to_common_attr": "cascade_pwatch_time_avg",
          "aggregator":"avg"
        },
        {
          "from_item_attr": "cascade_pwatch_time",
          "to_common_attr": "cascade_pwatch_time_dev",
          "aggregator":"dev"
        },
        {
          "from_item_attr": "cascade_action_once_interact_score",
          "to_common_attr": "cascade_action_once_interact_score_avg",
          "aggregator":"avg"
        },
        {
          "from_item_attr": "cascade_action_once_interact_score",
          "to_common_attr": "cascade_action_once_interact_score_dev",
          "aggregator":"dev"
        },
      ]
    ) \
    .explore_personally_ensemble_weight(
      matrix_weight_attr = "fountain_cascade_ensemble_cem_maxtrix",
      feature_vector_attr = "fountain_cascade_ensemble_cem_feature",
      weight_config =[
        {
          "weight_name":"fountain_fast_ensemble_weight_cascade_pwtd",
          "weight_config_key":"wtd"
        },
        {
          "weight_name":"xlife_fountain_fast_ensemble_power_weight_cascade_click_score",
          "weight_config_key":"click"
        },
        {
          "weight_name":"fountain_fast_ensemble_power_weight_cascade_pwatch_time",
          "weight_config_key":"watchtime"
        },
        {
          "weight_name":"fountain_fast_ensemble_weight_action_once_interact_score",
          "weight_config_key":"actiononce"
        },
      ],
      feature_config=[
        {
          "fature_name":"page",
          "treat_type":"maxmin",
          "value_type":"int",
          "min_value_attr":"fountain_cascade_page_feature_min",
          "max_value_attr":"fountain_cascade_page_feature_max",
        },
        {
          "fature_name":"userRequestHour",
          "treat_type":"maxmin",
          "value_type":"int",
          "min_value_attr":"fountain_cascade_hour_feature_min",
          "max_value_attr":"fountain_cascade_hour_feature_max",
        },
        {
          "fature_name":"user_emp_ltr",
          "treat_type":"maxmin",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_emp_ltr_feature_min",
          "max_value_attr":"fountain_cascade_emp_ltr_feature_max",
        },
        {
          "fature_name":"user_emp_wtr",
          "treat_type":"maxmin",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_emp_wtr_feature_min",
          "max_value_attr":"fountain_cascade_emp_wtr_feature_max",
        },
        {
          "fature_name":"user_emp_cmtr",
          "treat_type":"maxmin",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_emp_cmtr_feature_min",
          "max_value_attr":"fountain_cascade_emp_cmtr_feature_max",
        },
        {
          "fature_name":"user_emp_ftr",
          "treat_type":"maxmin",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_emp_ftr_feature_min",
          "max_value_attr":"fountain_cascade_emp_ftr_feature_max",
        },
        {
          "fature_name":"user_emp_watch_time",
          "treat_type":"maxmin",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_emp_watchtime_feature_min",
          "max_value_attr":"fountain_cascade_emp_watchtime_feature_max",
        },
        {
          "fature_name":"user_emp_evtr",
          "treat_type":"maxmin",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_emp_evtr_feature_min",
          "max_value_attr":"fountain_cascade_emp_evtr_feature_max",
        },
        {
          "fature_name":"featureUserClickCount",
          "treat_type":"maxmin",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_user_click_cnt_feature_min",
          "max_value_attr":"fountain_cascade_user_click_cnt_feature_max",
        },
        {
          "fature_name":"cascade_pwtd_avg",
          "treat_type":"maxmin",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_pwtd_avg_feature_min",
          "max_value_attr":"fountain_cascade_pwtd_avg_feature_max",
        },
        {
          "fature_name":"cascade_pwtd_dev",
          "treat_type":"original",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_pwtd_dev_feature_min",
          "max_value_attr":"fountain_cascade_pwtd_dev_feature_max",
        },
        {
          "fature_name":"cascade_pctr_avg",
          "treat_type":"maxmin",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_pctr_avg_feature_min",
          "max_value_attr":"fountain_cascade_pctr_avg_feature_max",
        },
        {
          "fature_name":"cascade_pctr_dev",
          "treat_type":"original",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_pctr_dev_feature_min",
          "max_value_attr":"fountain_cascade_pctr_dev_feature_max",
        },
        {
          "fature_name":"cascade_pwatch_time_avg",
          "treat_type":"maxmin",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_pwatch_time_avg_feature_min",
          "max_value_attr":"fountain_cascade_pwatch_time_avg_feature_max",
        },
        {
          "fature_name":"cascade_pwatch_time_dev",
          "treat_type":"original",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_pwatch_time_dev_feature_min",
          "max_value_attr":"fountain_cascade_pwatch_time_dev_feature_max",
        },
        {
          "fature_name":"cascade_action_once_interact_score_avg",
          "treat_type":"maxmin",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_action_once_interact_score_avg_feature_min",
          "max_value_attr":"fountain_cascade_action_once_interact_score_avg_feature_max",
        },
        {
          "fature_name":"cascade_action_once_interact_score_dev",
          "treat_type":"original",
          "value_type":"double",
          "min_value_attr":"fountain_cascade_action_once_interact_score_dev_feature_min",
          "max_value_attr":"fountain_cascade_action_once_interact_score_dev_feature_max",
        },
      ],
    ) \
    .end_if_() \
    .fountain_calc_ensemble_score(
      use_dist_calc = "{{fountain_cascade_ensemble_use_dist_calc}}",
      dis_factor = "{{fountain_cascade_ensemble_dis_factor}}",
      range_end = "{{fountain_fast_cascade_ensemble_range_end}}",
      user_new_proportion = "{{fountain_fast_cascade_ensemble_sort_enable_proportion}}",
      user_power_calc = "{{fountain_fast_cascade_ensemble_sort_enable_power_calc}}",
      user_power_calc_v2 = "{{fountain_fast_cascade_ensemble_sort_enable_power_calc_v2}}",
      enable_time_cost_opt = "{{fountain_cascade_enable_time_cost_opt}}",
      user_info_ptr_attr = "user_info_ptr",
      action_day = "{{mc_variant_weight_action_day_num}}",
      queues = cascade_ensemble_sort_queues,
      save_score_to_attr = "cascade_ensemble_score",
      rank_smooth = "{{fountain_fast_cascade_rank_smooth}}",
      use_queue_smooth_as_rank_smooth = "{{fountain_fast_cascade_ensemble_use_queue_smooth_as_rank_smooth}}",
      rank_use_hyperbolic = "{{fountain_fast_cascade_ensemble_sort_enable_rank_use_hyperbolic}}",
      hyperbolic_scale = "{{fountain_fast_cascade_ensemble_sort_hyperbolic_scale}}",
      hyperbolic_alpha = "{{fountain_fast_cascade_ensemble_sort_hyperbolic_alpha}}",
      hyperbolic_beta = "{{fountain_fast_cascade_ensemble_sort_hyperbolic_beta}}",
      min_rank_weight = "{{fountain_mc_fullrank_min_rank_weight}}"
    ) \
    .if_("enable_xlife_fountain_mc_top_sv_hetu_discount == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "xlife_fountain_mc_top_sv_hetu_discount_coeff", "as": "discount_coeff"},
          {"name": "xlife_fountain_enable_top_sv_hetu2", "as": "enable_top_sv_hetu2"},
          {"name": "xlife_fountain_top_sv_hetu_count", "as": "top_sv_hetu_count"},
          {"name": "xlife_fountain_hetu_psvtr_mix_coeff", "as": "hetu_psvtr_mix_coeff"},
          {"name": "xlife_fountain_enable_dynamic_coeff", "as": "enable_dynamic_coeff"},
          {"name": "xlife_fountain_top_sv_stat_hetu_score_lower_bound", "as": "top_sv_stat_hetu_score_lower_bound"},
          "colossus_hetu_emp_svtr_stat"
        ],
        import_item_attr = [
          {"name": "cascade_ensemble_score", "as": "es_score"},
          {"name": "cascade_psvtr", "as": "psvtr"},
          {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_level_one_list"},
          {"name": "hetu_tag_level_info__hetu_level_two", "as": "hetu_level_two_list"},
        ],
        export_item_attr = [
          {"name": "es_score", "as": "cascade_ensemble_score"},
        ],
        function_name = "DiscountTopSvHetus",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .sort(
      range_end = "{{fountain_cascade_ensemble_range_end}}",
      score_from_attr = "cascade_ensemble_score",
    ) \
    .if_("fountain_mc_enable_dedup_on_same_author == 1") \
      .deduplicate(
        on_item_attr = "author__id",
      ) \
    .end_() \
    .if_('fountain_fast_cascade_final_control_hetu_count == 1') \
      .explore_control_hetu_count_arranger(
        user_hetu_stat_attr = "user_mixed_interest_stat",
        hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
        hetu_level_two_attr = "hetu_tag_level_info__hetu_level_two",
        hetu_level_five_attr = "hetu_tag_level_info__hetu_level_five",
        duration_ms_attr = "duration_ms",
        author_attr = "author__id",
        enable_hetu_control_interest = "{{cascade_enable_hetu_control_interest}}",
        enable_hetu_control_diversity = "{{cascade_enable_hetu_control_diversity}}",
        enable_duration_control_diversity = "{{cascade_enable_duration_control_diversity}}",
        enable_author_control_diversity = "{{cascade_enable_author_control_diversity}}",
        hetu_control_interest_start = "{{cascade_hetu_control_interest_start}}",
        hetu_control_diversity_start = "{{cascade_hetu_control_diversity_start}}",
        duration_control_diversity_start = "{{cascade_duration_control_diversity_start}}",
        author_control_diversity_start = "{{cascade_author_control_diversity_start}}",
        keep_size = "{{fullrank_fast_before_variant_mc_limit_size}}",
        hetu1_max_size = "{{cascade_control_hetu1_max_size}}",
        hetu2_max_size = "{{cascade_control_hetu2_max_size}}",
        hetu5_max_size = "{{cascade_control_hetu5_max_size}}",
        duration_0_7s_max_size = "{{cascade_control_duration_0_7s_max_size}}",
        duration_7_9s_max_size = "{{cascade_control_duration_7_9s_max_size}}",
        duration_9_12s_max_size = "{{cascade_control_duration_9_12s_max_size}}",
        duration_12_17s_max_size = "{{cascade_control_duration_12_17s_max_size}}",
        duration_17_20s_max_size = "{{cascade_control_duration_17_20s_max_size}}",
        hetu_adjust_max_value = "{{fountain_mc_hetu_control_hetu_adjust_max_value}}",
        hetu_adjust_min_value = "{{fountain_mc_hetu_control_hetu_adjust_min_value}}",
        hetu_adjust_coef = "{{fountain_mc_hetu_control_hetu_adjust_coef}}",
        enable_actual_hetu_control = "{{enable_fountain_mc_hetu_actual_hetu_control}}",
        same_author_max_size = "{{cascade_control_same_author_max_size}}"
      ) \
    .else_() \
      .if_('fountian_skip_fullrank_model_limit_v9_fast == 0') \
        .truncate(
          size_limit = "{{fullrank_fast_before_variant_mc_limit_size}}",
        ) \
      .end_if_() \
    .end_if_() \
    .explore_custom_embedding_score_enricher(
      check_point_ = "cascade",
      enable_fountain_version = True,
      enable_fix_low_hit_rate = "{{fountain_mc_enable_fix_mmu_embedding_low_hit_rate}}",
      user_info_ptr_attr = "user_info_ptr",
      embedding_list_attr = "mmu_embeddings",
      source_pids_list_attr = "embedding_source_pids",
      calc_type = "action_bucket_dot",
      not_click_limit_hour = "{{fountain_mc_neg_feedback_sim_score_not_click_hour_limit}}",
      play_stat_limit_hour = "{{fountain_mc_neg_feedback_sim_score_play_stat_hour_limit}}",
      extra_not_click_limit_hour = "{{fountain_mc_neg_feedback_sim_score_extra_not_click_hour_limit}}",
      short_view_threshold = "{{fountain_mc_neg_feedback_sim_score_short_view_threshold}}",
      not_click_weight = "{{fountain_mc_neg_feedback_sim_score_not_click_weight}}",
      short_view_weight = "{{fountain_mc_neg_feedback_sim_score_short_view_weight}}",
      extra_not_click_weight = "{{fountain_mc_neg_feedback_sim_score_extra_not_click_weight}}",
      export_item_attr = "hate_similary_score",
      dim_size = 64
    ) \
    .log_debug_info(
      item_attrs = [
        'duration_perf_id', 'duration_s', 'emp_htr', 'hetu_level_one_index', 'is_merchant_cart', 'is_merchant_living', 'cascade_fc_pvtr2', 'cascade_debias_wtd', 'cascade_click_comment_button', 'cascade_ftr_kai',
        'cascade_distill_rerank', 'cascade_ftr_kai', 'cascade_wtd_fintr', 'cascade_variant_sort_score'
      ],
      common_attrs = [
        'enable_fountain_calc_duration_bucket', 'enable_fountain_cascade_produce_need_divuser', 'enable_fountain_cascade_produce_need_divuser_v2',
        'enable_fountain_cascade_produce_photo_predict', 'fountain_cascade_produce_his_magic_face_threholds', 'fountain_cascade_produce_his_zhongcao_threholds',
        'fountain_cascade_produce_real_show_photo_recent_hours', 'fountain_xhs_target_spec_hetu_type_map_str', 'hot_content_thompson_sampling_range', 'xhs_target_photo_remove_author_bucket',
        'fountain_new_arch_longterm_kess_service', 'fountain_new_arch_longterm_request_type', 'fountain_new_arch_longterm_skip_predict', 'skip_fountain_cascade_new_interface_predict',
        'fountain_mc_time_cluster_sort_queue_num', 'fountain_mc_time_cluster_sort_queue_ratio_string', 'fountain_mc_time_cluster_sort_range_end',
        'fountain_fast_cascade_enable_distill_sort', 'colossus_hetu_emp_svtr_stat', 'userExpEptr', 'xhs_target_audit_b_set', 'xhs_target_hetu_set', 
        'hot_content_negative_count', 'hot_content_positive_count', 'mc_s1_cluster_cnt', 'mc_s1_cluster_id', 'fountain_full_link_reco_log_message',
        'fountain_fast_cascade_ensemble_sort_enable_rank_use_hyperbolic'
      ]
    )
    return self