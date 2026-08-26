#!/usr/bin/env python3
# coding=utf-8



from dragonfly.ext.explore.explore_api_mixin import ExploreApiMixin
from ranking.ranking_util import value_and_rank_score_queues

class ExploreRankingStrategyMixin(ExploreApiMixin):
  """
  双列发现页外流精排策略函数 Mixin 实现
  """
  def trucate_by_user_features(self):
    """
    Module: RankingStageOneTruncateModule
    功能：根据用户特征判断精排一阶段截断比例
    Owner: liubaoan
    Date: 
    """
    self.split_string(
        input_common_attr = "explore_rank_stage1_age_truncate_info",
        output_common_attr = "explore_rank_stage1_age_truncate_params",
        delimiters="-",
        parse_to_double = True
      ) \
      .split_string(
        input_common_attr = "explore_rank_stage1_city_level_truncate_info",
        output_common_attr = "explore_rank_stage1_city_level_truncate_params",
        delimiters="-",
        parse_to_double = True
      ) \
      .split_string(
        input_common_attr = "explore_rank_stage1_active_degree_truncate_info",
        output_common_attr = "explore_rank_stage1_active_degree_truncate_params",
        delimiters="-",
        parse_to_double = True
      ) \
      .enrich_attr_by_light_function(    
        import_common_attr = [
          "basic_info_age_segment_v2",
          "location_city_level_v2",
          "explore_low_active_level",
          "explore_rank_stage1_age_truncate_params",
          "explore_rank_stage1_city_level_truncate_params",
          "explore_rank_stage1_active_degree_truncate_params"
        ],
        export_common_attr = [
          "explore_rank_stage1_truncate_param",
        ],
        function_name = "GenFullRankStageOneTruncateRatio",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .gen_common_attr_by_lua(
        attr_map={
          "ctr_filter_distribue_coff": "ctr_filter_distribue_coff * explore_rank_stage1_truncate_param",
        }
      )
    return self

  def resort_new_interest_items(self):
    self.item_attr_operation(
        item_attr_a = "explore_fr_ensemble_score",
        item_attr_b = "user_group_interest_tgi_score",
        operator = "*",
        output_attr = "explore_fr_ensemble_score_resort",
        target_item = {
          "is_new_interest_explore": 1
        }
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          {"name": "explore_fr_ensemble_score", "as": "origin_score"},
          {"name": "explore_fr_ensemble_score_resort", "as": "resort_score"},
        ],
        export_item_attr = [
          {"name": "final_score", "as": "explore_fr_ensemble_score"},
        ],
        function_name = "ReplaceScoreByResortOrder",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          "is_new_interest_explore": 1
        }
      )
    return self

  def calc_value_and_rank_score(self):
    """
    Module: RankingEnsembleSortModule
    功能：根据 ValueAndRank 公式计算 EnsembleSort 分,
          打分公式见:https://docs.corp.kuaishou.com/d/home/fcABiXW20UMlDmE0C45t9VAWZ 
    Owner: libingchen
    Date: 2023-04-12
    """
    self.split_string(
        input_common_attr = "explore_vrs_rank_score_queue_name",
        output_common_attr = "vrs_need_rank_score_queue_names",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
      ) \
      .split_string(
        input_common_attr = "explore_vrs_value_score_queue_name",
        output_common_attr = "vrs_need_value_score_queue_names",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
      ) \
      .split_string(
        input_common_attr = "explore_vrs_enable_queue_name",
        output_common_attr = "vrs_enable_queue_names",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
      ) \
      .split_string(
        input_common_attr = "explore_vrs_need_multiply_ctr_queue_name",
        output_common_attr = "vrs_need_multiply_ctr_queue_names",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
      ) \
      .split_string(
        input_common_attr = "explore_vrs_fr_update_alpha_queue_names_list",
        output_common_attr = "vrs_fr_update_alpha_queue_names_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
      ) \
      .if_("explore_fr_skip_infer_uv_ctr_boost_handle_remake_formula == 1", to_be_delete = "date=2024-05-29;committer=fengjingping") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "infer_uv_ctr", 
            {"name": "refreshTimes", "as": "refresh_times"},
            {"name": "explore_fr_uv_ctr_refresh_times_threshold_remake_formula", "as": "refresh_times_threshold"},
            {"name": "explore_fr_uv_ctr_infer_uv_ctr_threshold_remake_formula", "as": "infer_uv_ctr_threshold"},
            {"name": "explore_fr_uv_ctr_weight_max_remake_formula", "as": "weight_max"},
            {"name": "explore_fr_uv_ctr_weight_min_remake_formula", "as": "weight_min"}, 
            {"name": "explore_fr_uv_ctr_alpha_remake_formula", "as": "alpha"},
            {"name": "explore_fr_uv_ctr_beta_remake_formula", "as": "beta"},
            {"name": "explore_fr_uv_ctr_omega_remake_formula", "as": "omega"},
            {"name": "explore_fr_uv_ctr_boost_type_remake_formula", "as": "boost_type"},
            {"name": "explore_vrs_score_pctr_value_alpha", "as": "xtr_weight"},
          ],
          export_common_attr = [
            {"name": "xtr_weight", "as": "explore_vrs_score_pctr_value_alpha"}
          ],
          function_name = "CalcXtrWeightByInferUvCtr",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .explore_calc_value_and_rank_score(
        enable_queue_names = "vrs_enable_queue_names",
        update_alpha_queue_names_list = "vrs_fr_update_alpha_queue_names_list",
        enable_open_update_alpha_handle = "{{enable_fr_open_update_alpha_handle}}",
        es_rank_power = "explore_vrs_rank_power",
        es_value_power = "explore_vrs_value_power",
        need_rank_queue_names = "vrs_need_rank_score_queue_names",
        need_value_queue_names = "vrs_need_value_score_queue_names",
        multiply_ctr_queue_names = "vrs_need_multiply_ctr_queue_names",
        pctr_queue_name = "explore_vrs_pctr_queue_name",
        pfsh_queue_name = "explore_vrs_pfsh_queue_name",
        pct_queue_name = "explore_vrs_pct_queue_name",
        pdlk_queue_name = "explore_vrs_pdlk_queue_name",
        save_score_to_attr = "explore_fr_ensemble_score",
        queues = value_and_rank_score_queues(),
      )
    return self

  def calc_report_discount(self):
    """
    Module: RankingScoreModule
    功能：根据举报计算调权系数
    Owner: liuhao07
    Date: 2023-04-20
    """
    self.enrich_attr_by_lua(
        import_common_attr = [
          "rerank_photo_rstr_norm_show",
          "rerank_photo_rstr_norm_report",
          "rerank_rstr_discount_param_n",
          "rerank_rstr_discount_param_o",
          "rerank_rstr_discount_param_n_high_good"
        ],
        import_item_attr = [
          "explore_stat__report_detail__total_report_count",
          "explore_stat__show_count",
          "audit_hot_high_tag_level"
        ],
        export_item_attr = [
          "report_discount",
        ],
        function_for_item = "report_discount_calculate",
        lua_script_file = "explore/ranking/lua/module/ranking_ensemble_sort__score_coeff.lua"
      )
    return self

  def calc_hate_discount(self):
    """
    Module: RankingScoreModule
    功能：根据讨厌计算调权系数
    Owner: liuhao07
    Date: 2023-04-20
    """
    self.enrich_with_protobuf(
        from_extra_var = "user_info_ptr",
        attrs = [
          dict(name="recentUploadBitMap", path="recent_upload_bitmap"),
          dict(name="hateTimeMs", path="user_profile_v1.hate_list.time_ms"),
        ]
      ) \
      .enrich_attr_by_lua(
        import_common_attr = [
          "hateTimeMs",
          "recentUploadBitMap"
        ],
        import_item_attr = [
          "phtr"
        ],
        export_item_attr = [
          "hate_discount"
        ],
        function_for_item = "hate_discount",
        function_for_common = "collect_garbage",
        lua_script_file = "explore/ranking/lua/module/ranking_ensemble_sort__score_coeff.lua"
      )
    return self
  
  def calc_pcmef_gender_debias_score(self):
    """
    Module: RankingScoreModule
    功能：计算评论区有效停留纠偏分
    Owner: liuhao07
    Date: 2023-04-20
    """
    self.enrich_with_protobuf(
        from_extra_var = "user_info_ptr",
        attrs = [
          dict(name="gender", path="gender")
        ]
      ) \
      .enrich_attr_by_lua(
        import_common_attr = [
          "gender"
        ],
        import_item_attr = [
          "hetu_tag_level_info__hetu_level_one"
        ],
        export_item_attr = [
          "pcmef_debias_bucket_name"
        ],
        function_for_item = "get_cmef_debias_bucket_name",
        lua_script_file = "explore/ranking/lua/module/ranking_ensemble_sort__print.lua"
      ) \
      .get_kconf_params(
        kconf_configs = [{
          "kconf_key": "reco.hot.cmef_gender_hetu_debias_json",
          "json_path": "{{pcmef_debias_bucket_name}}",
          "default_value": "1.83565643",
          "export_item_attr": "pcmef_debias_bucket_score"
        }]
      ) \
      .enrich_attr_by_lua(
        import_item_attr = [
          "pcmef",
          "corr_pctr",
          "pcmef_debias_bucket_score"
        ],
        export_item_attr = [
          "pcmef_debias_score"
        ],
        function_for_item = "get_cmef_debias_score",
        lua_script_file = "explore/ranking/lua/module/ranking_ensemble_sort__print.lua"
      )
    return self

  def gen_score_stage1(self):
    """
    Module: RankingScoreModule
    功能：精排stage1排序分预处理
    Owner: liuhao07
    Date: 2023-04-20
    """
    self.enrich_attr_by_lua(
      import_common_attr = [
        "fr_enable_neg_queue_report_discount",
        "fetr_feff_ctr_power",
        "explore_fr_ctr_power",
        "explore_fr_fetr_feff_power",
        "explore_fr_svr_power"
      ],
      import_item_attr = [
        "corr_pctr",
        "pltr",
        "corr_pwtr",
        "pftr",
        "pcmtr",
        "petcm",
        "pptr",
        "psvr",
        "report_discount",
        "hate_discount",
        "pdtr",
        "pepstr",
        "pcltr",
        "pcmef",
        "phtr",
        "fr_score2",
        "fetr",
        "fountain_eff",
      ],
      export_item_attr = [
        "score_pctr",
        "score_pltr",
        "score_pwtr",
        "score_pftr",
        "score_pcmtr",
        "score_petcm",
        "score_pptr",
        "score_psvr",
        "score_pdtr",
        "score_pepstr",
        "score_pcltr",
        "score_pcmef",
        "score_phtr",
        "score_pctr_x_psvr",
        "corr_fetr",
        "corr_fountain_eff",
      ],
      function_for_item = "gen_score_stage1",
      lua_script_file = "explore/ranking/lua/module/ranking_score__gen_score_stage1.lua"
    ) \
    .if_("enable_explore_fr_enter_fountain_score_debias_by_picture_type == 1") \
      .fr_enter_fountain_score_debias_by_picture_type() \
    .end_()
    return self

  def gen_score_stage2(self):
    """
    Module: RankingScoreModule
    功能：精排stage2排序分预处理
    Owner: liuhao07
    Date: 2023-04-20
    """
    self.enrich_attr_by_lua(
      import_common_attr = [
        "explore_fr_es_pctr_x_pxtr_power_beta_time",
        "explore_fr_es_pctr_x_pxtr_power_beta_action",
        "awesome_wtd_pctr_weight"
      ],
      import_item_attr = [
        "corr_pctr",
        "report_discount",
        "hate_discount",
        "consume_time_ltr",
        "pcltr",
        "pepstr",
        "pptr",
        "pdtr",
        "pftr",
        "pevtr",
        "fr_score2",
        "awesome_wtd"
      ],
      export_item_attr = [
        "awesome_wtd_score",
        "score_consume_time_ltr",
        "score_pctr_x_pcltr",
        "score_pctr_x_pepstr",
        "score_pctr_x_pptr",
        "score_pctr_x_pdtr",
        "score_pctr_x_pftr",
        "score_pctr_x_pevtr",
        "score_pctr_x_fr_score2",
        "score_pctr_x_awesome_wtd",
      ],
      function_for_item = "score_coeff_calculate_stage2",
      function_for_common = "collect_garbage",
      lua_script_file = "explore/ranking/lua/module/ranking_ensemble_sort__score_coeff.lua"
    )
    return self

  def boost_young_photo(self):
    """
    boost_young_photo
    Owner: caozhong
    Date: 2023-04-19
    :return:
    """
    self.enrich_attr_by_light_function(
      import_item_attr=[
        "explore_fr_ensemble_score",
        "is_young_photo"
      ],
      import_common_attr=[
        "young_photo_boost_fr_coeff"
      ],
      export_item_attr=[
        "explore_fr_ensemble_score"
      ],
      function_name="EnsembleScoreChangeForLifeYoung",
      class_name="ExploreLightFunctionSetV2",
    )
    return self

  def discount_life_photo_hetu(self):
    """
    Module: RankingEnsembleSortModule
    功能: 用于对生活tab部分河图降权
    Owner: hanbiyun
    Date: 2023-05-06
    :return:
    """
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "hetu_tag_level_info_v2__hetu_level_one", "as": "extract_hetu_tag_list"},  # 替换为已有的河图一级
      ],
      export_item_attr = [
        {"name": "first_hetu_tag_id", "as": "hetu_v3_level2"}
      ],
      function_name = "ExtractFirstHetuV2Tag",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "ranking_hetu_v3_level2_discount_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "hetu_v3_level2": "{{fullrank_discount_hetu_v3_level2}}"
      },
    )
    return self

  def gen_l2r_score_fusion(self):
    """
    Module: RankingEnsembleSortModule
    功能: 生成分数，用于排序
    Owner: xuwei09
    Date: 2023-05-08
    :return:
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "fullrank_gen_l2r_fusion_l2r_weight",
        "fullrank_gen_l2r_fusion_wtd_weight"
      ],
      import_item_attr = [
        "awesome_wtd",
        "gen_l2r_score"
      ],
      export_item_attr = [
        "gen_l2r_fusion_score"
      ],
      function_name = "CalFusionGenLtrScore",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def gen_emp_action_score_fusion(self):
    """
    Module: RankingEnsembleSortModule
    功能: 生成分数，用于排序
    Owner: caoying03
    Date: 2025-11-13
    :return:
    """
    self.calc_weighted_sum(
      channels = [
        { "name": "pctr", "weight": "{{explore_rank_emp_action_pctr_weight}}" },
        { "name": "pltr", "weight": "{{explore_rank_emp_action_pltr_weight}}" },
        { "name": "pwtr", "weight": "{{explore_rank_emp_action_pwtr_weight}}" },
        { "name": "pftr", "weight": "{{explore_rank_emp_action_pftr_weight}}" },
        { "name": "pcmtr", "weight": "{{explore_rank_emp_action_pcmtr_weight}}" },
        { "name": "plvtr", "weight": "{{explore_rank_emp_action_plvtr_weight}}" },
        { "name": "fr_score1", "weight": "{{explore_rank_emp_action_pf1_weight}}" },
        { "name": "fr_score2", "weight": "{{explore_rank_emp_action_pf2_weight}}" },
        { "name": "empirical_ctr", "weight": "{{explore_rank_emp_action_ctr_weight}}" },
        { "name": "empirical_ltr", "weight": "{{explore_rank_emp_action_ltr_weight}}" },
        { "name": "empirical_wtr", "weight": "{{explore_rank_emp_action_wtr_weight}}" },
        { "name": "empirical_ftr", "weight": "{{explore_rank_emp_action_ftr_weight}}" },
        { "name": "empirical_cmtr", "weight": "{{explore_rank_emp_action_cmtr_weight}}" },
        { "name": "empirical_watch_time", "weight": "{{explore_rank_emp_action_watch_time_weight}}" },
      ],
      output_item_attr = "emp_action_score",
    ) \
    .pack_item_attr(
      item_source={
        "reco_results": True,
      },
      mappings=[
        {
          "aggregator": "avg",
          "from_item_attr": "emp_action_score",
          "to_common_attr": "ctr_empirical_action_emp_action_score_avg"
        },
        {
          "aggregator": "max",
          "from_item_attr": "emp_action_score",
          "to_common_attr": "ctr_empirical_action_emp_action_score_max"
        },
      ],
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "corr_pctr", "as": "ctr"},
        "emp_action_score",
        "explore_stat__real_show_count",
        "thanos_stats__real_show_count",
        "explore_stat__click_count"
      ],
      import_common_attr = [
        "ctr_empirical_action_emp_action_score_max",
        "ctr_empirical_action_emp_action_score_avg",
        {"name": "explore_rank_ctr_empirical_action_ctr_weight", "as": "ctr_weight"},
        {"name": "explore_rank_ctr_empirical_action_bias_weight", "as": "bias_weight"},
        {"name": "explore_rank_ctr_empirical_action_emp_action_score_weight", "as": "emp_action_score_weight"},
        {"name": "explore_rank_ctr_empirical_action_show_limit_threshold", "as": "show_limit_threshold"},
        {"name": "explore_rank_ctr_empirical_action_ctr_limit_threshold", "as": "ctr_limit_threshold"},
        {"name": "explore_rank_ctr_empirical_action_percent_threshold", "as": "percent_threshold"},
      ],
      export_item_attr = [
        {"name": "ctr_emp_action", "as": "rank_ctr_emp_action"},
      ],
      function_name = "CalCtrEmpiricalActionScore",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def impression_audit_adjust(self):
    """
    Module: RankingEnsembleSortModule
    功能: 根据审核调整score
    Owner: liuhao07
    Date: 2023-05-22
    :return:
    """
    self.transform_item_attr( # 粗排阶段有，推全后修改，观感审二级字段大于0才是已审核
      mappings = [{
        "check_attr_name": "audit_b_second_tag",
        "check_attr_type": "int",
        "output_attr_name": "is_impression_audit",
        "output_attr_type": "int",
        "output_default_value": 0,
        "rules": [{
          "check_range": {
            "lower_bound": 1
          },
          "output_value": 1
        }]
      }]
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "impression_audit_adjust_coeff_map_str", "as": "adjust_coeff_map_str_attr"},
        {"name": "impression_audit_emp_xtr_adjust_flag", "as": "emp_xtr_adjust_flag"},
        {"name": "impression_audit_emp_ctr_avg", "as": "emp_ctr_avg"},
        {"name": "impression_audit_emp_watchtime_avg", "as": "emp_watchtime_avg"},
        {"name": "impression_audit_emp_xtr_coeff_a", "as": "emp_xtr_coeff_a"},
        {"name": "impression_audit_emp_xtr_coeff_b", "as": "emp_xtr_coeff_b"}
      ],
      import_item_attr = [
        {"name": "content_safety_level_with_namespace__level_hot_online", "as": "audit_level_attr"},
        {"name": "explore_fr_ensemble_score", "as": "ensemble_score_attr"},
        "upload_time",
        {"name": "explore_stat__real_show_count", "as": "realshow_count"},
        {"name": "explore_stat__click_count", "as": "click_count"},
        {"name": "explore_stat__view_length_sum", "as": "watchtime_sum"}
      ],
      export_item_attr = [
        {"name": "ensemble_score_attr", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "AuditAdjustScore",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "is_impression_audit": 1,
      },
    )
    return self

  def high_hot_audit_adjust(self):
    """
    Module: RankingEnsembleSortModule
    功能: 根据审核调整score
    Owner: liuhao07
    Date: 2023-05-22
    :return:
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "high_hot_audit_adjust_coeff_map_str", "as": "adjust_coeff_map_str_attr"}
      ],
      import_item_attr = [
        {"name": "audit_hot_high_tag_level", "as": "audit_level_attr"},
        {"name": "explore_fr_ensemble_score", "as": "ensemble_score_attr"},
        "upload_time"
      ],
      export_item_attr = [
        {"name": "ensemble_score_attr", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "AuditAdjustScore",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def topk_audit_adjust(self):
    """
    Module: RankingEnsembleSortModule
    功能: 根据审核调整score
    Owner: liuhao07
    Date: 2023-05-22
    :return:
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "topk_audit_adjust_coeff_map_str", "as": "adjust_coeff_map_str_attr"}
      ],
      import_item_attr = [
        {"name": "topk_audit_level", "as": "audit_level_attr"},
        {"name": "explore_fr_ensemble_score", "as": "ensemble_score_attr"},
        "upload_time"
      ],
      export_item_attr = [
        {"name": "ensemble_score_attr", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "AuditAdjustScore",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def click_rate_boost(self):
    """
    Module: RankingEnsembleSortModule
    功能: 根据 xhs 用户在当前item的点击调整分数
    Owner: libingchen
    Date: 2023-06-02
    :return:
    """
    self.enrich_attr_by_light_function( # (libingchen) xhs 原始 click rate boost v2 
      import_common_attr = [
        {"name": "explore_fr_whole_boost_click_count_alpha", "as": "whole_boost_click_count_alpha"},
        {"name": "explore_fr_whole_boost_click_count_beta", "as": "whole_boost_click_count_beta"},
        {"name": "explore_fr_whole_boost_click_count_omega", "as": "whole_boost_click_count_omega"},
        {"name": "explore_fr_outflow_boost_click_count_alpha", "as": "outflow_boost_click_count_alpha"},
        {"name": "explore_fr_outflow_boost_click_count_beta", "as": "outflow_boost_click_count_beta"},
        {"name": "explore_fr_outflow_boost_click_count_omega", "as": "outflow_boost_click_count_omega"},
      ],
      import_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": "input_score"},
        {"name": "xhs_install_find_click_value", "as": "whole_click"},
        {"name": "xhs_install_find_outflow_click_value", "as": "outflow_click"}
      ],
      export_item_attr = [
        {"name": "output_score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "ExploreBoostByClickRateV2",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
  
  def merchant_photo_boost_by_buyer_type(self):
    """
    Module: RankingEnsembleSortModule
    功能: 【挂车短视频】根据买家分层调整对电商视频调权，新买家降权，老买家提权，整体控电商load
    Owner: zhanglinjiang
    Date: 2023-07-05
    :return:
    """
    self.enrich_attr_by_light_function( # 计算挂车精排权重系数
      import_common_attr = [
        {"name": "merchant_buyer_type", "as": "buyer_type"},
        {"name": "explore_fr_merchant_photo_boost_coef", "as": "buyer_boost_coef"},
      ],
      export_common_attr = [
        {"name": "merchant_boost_coef", "as": "fr_merchant_photo_boost_coef"}
      ],
      function_name = "MerchantCalcBoostCoef",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_merchant_photo_boost_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "is_merchant_cart" : 1
      }
    )
    return self
  
  def merchant_live_boost_by_buyer_type(self):
    """
    Module: RankingEnsembleSortModule
    功能: 【live头像】根据买家分层调整对电商视频调权，新买家降权，老买家提权，整体控电商load
    Owner: zhanglinjiang
    Date: 2023-07-05
    :return:
    """
    self.enrich_attr_by_light_function( # 计算挂车精排权重系数
      import_common_attr = [
        {"name": "merchant_buyer_type", "as": "buyer_type"},
        {"name": "explore_fr_merchant_live_boost_coef", "as": "buyer_boost_coef"},
      ],
      export_common_attr = [
        {"name": "merchant_boost_coef", "as": "fr_merchant_live_boost_coef"}
      ],
      function_name = "MerchantCalcBoostCoef",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_merchant_live_boost_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "is_merchant_living" : 1
      }
    )
    return self

  def merchant_price_inferior_reduce_weight(self):
    """
    Module: RankingEnsembleSortModule
    功能: 【产品需求】【挂车短视频】产品侧要求对挂价格力劣质商品的短视频打压
    Owner: zhanglinjiang
    Date: 2023-11-03
    :return:
    """
    self.enrich_attr_by_light_function( # price_info=102表示价格劣质
      import_common_attr = [
        {"name": "explore_fr_merchant_price_inferior_reduce_weight", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "is_merchant_cart" : 1,
        "price_info": 102
      }
    )
    return self

  def rank_stage2_emp_fetr_boost_coef(self):
    """
    Module: RankingEnsembleSortModule.py
    功能: 个性化下滑系数计算
    Owner: xuwei09
    Date: 2024-01-22
    :return:
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_weight_adjust_emp_fountain_time_base_stat", "as": "user_base_stat"},
        {"name": "user_emp_fountain_time_ratio", "as": "user_dynamic_stat"},
        {"name": "explore_weight_adjust_emp_fetr_boost_coef_lower", "as": "boost_coef_lower"},
        {"name": "explore_weight_adjust_emp_fetr_boost_coef_upper", "as": "boost_coef_upper"},
        {"name": "explore_weight_adjust_emp_fetr_is_boost", "as": "is_boost"},
        {"name": "explore_weight_adjust_avg_fetr_power_weight", "as": "action_power_weight"},
      ],
      export_common_attr = [
        {"name": "user_dynamic_action", "as": "emp_rank_boost_fetr"}
      ],
      function_name = "CalcUserDynamicAction",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
  
  def rank_stage2_fetr_adjust(self):
    """
    Module: RankingEnsembleSortModule.py
    功能: 个性化下滑
    Owner: xuwei09
    Date: 2024-01-22
    :return:
    """
    self.gen_common_attr_by_lua(
      attr_map = {
        "hot_fountain_fetr_weight_push" : "emp_rank_boost_fetr * hot_fountain_fetr_weight_push",
        "hot_fountain_fountain_eff_weight_push" : "emp_rank_boost_fetr * hot_fountain_fountain_eff_weight_push",
      }
    )
    return self

  def explore_cal_rank_ensemble_pftr_dur(self):
    self \
    .if_("explore_rank_pltr_dur_social_condition == 1 and (bid_follow_num == 0 or (u_inside_share_active_degree_detail_code > 3))") \
      .gen_common_attr_by_lua(
        attr_map={
          "explore_ensemble_power_weight_fullrank_pftr_dur_score_social": "0.0"
        }
      ) \
    .end_() \
    .if_("explore_rank_pltr_dur_social_condition == 2 and (bid_follow_num == 0 or (u_share_num_30d == 0 and u_message_active_degree ~= 5 and u_message_active_degree ~= 6))") \
      .gen_common_attr_by_lua(
        attr_map={
          "explore_ensemble_power_weight_fullrank_pftr_dur_score_social": "0.0"
        }
      ) \
    .end_() \
    .if_("explore_rank_pltr_dur_social_condition == 3 and (bid_follow_num == 0 or (user_msg_cnt_ssm_today + user_msg_cnt_gsm_today > 0))") \
      .gen_common_attr_by_lua(
        attr_map={
          "explore_ensemble_power_weight_fullrank_pftr_dur_score_social": "0.0"
        }
      ) \
    .end_() \
    .split_string(
      input_common_attr = "explore_rank_pftr_dur_percentile_str",
      output_common_attr = "explore_rank_pftr_dur_percentile_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_rank_pftr_dur_percentile_list", "as": "percentile_list"},
        {"name": "explore_rank_pftr_dur_gama", "as": "gama"},
        {"name": "explore_rank_pftr_dur_threshold", "as": "threshold"}
      ],
      import_item_attr = [
        {"name": "duration_ms", "as": "duration"},
        {"name": "score_pftr", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "score_pftr_dur_social"},
      ],
      function_name = "CalculateCascadePftrDurScore",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
  
  def rank_stage2_request_pxtr_boost_coef(self):
    """
    Module: RankingEnsembleSortModule.py
    功能: request 维度个性化权重
    Owner: xuwei09
    Date: 2024-01-15
    :return:
    """
    self.pack_item_attr(
      item_source = {
        "reco_results": True,
      },
      mappings = [
        {
          "aggregator": "avg",
          "from_item_attr": "corr_pctr",
          "to_common_attr": "pctr_avg"
        },
        {
          "aggregator": "avg",
          "from_item_attr": "pltr",
          "to_common_attr": "pltr_avg"
        },
        {
          "aggregator": "avg",
          "from_item_attr": "pwtr",
          "to_common_attr": "pwtr_avg"
        },
        {
          "aggregator": "avg",
          "from_item_attr": "pftr",
          "to_common_attr": "pftr_avg"
        },
        {
          "aggregator": "avg",
          "from_item_attr": "pcmtr",
          "to_common_attr": "pcmtr_avg"
        },
        {
          "aggregator": "avg",
          "from_item_attr": "pptr",
          "to_common_attr": "pptr_avg"
        },
        {
          "aggregator": "avg",
          "from_item_attr": "fr_score1",
          "to_common_attr": "fr_score1_avg"
        },
        {
          "aggregator": "avg",
          "from_item_attr": "fr_score2",
          "to_common_attr": "fr_score2_avg"
        },
        {
          "aggregator": "avg",
          "from_item_attr": "awesome_wtd",
          "to_common_attr": "awesome_wtd_avg"
        },
        {
          "aggregator": "avg",
          "from_item_attr": "fetr",
          "to_common_attr": "fetr_avg"
        },
        {
          "aggregator": "avg",
          "from_item_attr": "fountain_eff",
          "to_common_attr": "fountain_eff_avg"
        },
        {
          "aggregator": "avg",
          "from_item_attr": "pcltr",
          "to_common_attr": "pcltr_avg"
        }
      ],
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_weight_adjust_avg_pctr_base_stat", "as": "user_base_stat"},
        {"name": "pctr_avg", "as": "user_dynamic_stat"},
        {"name": "explore_weight_adjust_avg_rerank_boost_coef_lower", "as": "boost_coef_lower"},
        {"name": "explore_weight_adjust_avg_rerank_boost_coef_upper", "as": "boost_coef_upper"},
        {"name": "explore_weight_adjust_avg_rerank_is_boost", "as": "is_boost"},
        {"name": "explore_weight_adjust_avg_rerank_power_weight", "as": "action_power_weight"},
      ],
      export_common_attr = [
        {"name": "user_dynamic_action", "as": "rerank_boost_pctr"}
      ],
      function_name = "CalcUserDynamicAction",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_weight_adjust_avg_pltr_base_stat", "as": "user_base_stat"},
        {"name": "pltr_avg", "as": "user_dynamic_stat"},
        {"name": "explore_weight_adjust_avg_rerank_boost_coef_lower", "as": "boost_coef_lower"},
        {"name": "explore_weight_adjust_avg_rerank_boost_coef_upper", "as": "boost_coef_upper"},
        {"name": "explore_weight_adjust_avg_rerank_is_boost", "as": "is_boost"},
        {"name": "explore_weight_adjust_avg_rerank_power_weight", "as": "action_power_weight"},
      ],
      export_common_attr = [
        {"name": "user_dynamic_action", "as": "rerank_boost_pltr"}
      ],
      function_name = "CalcUserDynamicAction",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_weight_adjust_avg_pwtr_base_stat", "as": "user_base_stat"},
        {"name": "pwtr_avg", "as": "user_dynamic_stat"},
        {"name": "explore_weight_adjust_avg_rerank_boost_coef_lower", "as": "boost_coef_lower"},
        {"name": "explore_weight_adjust_avg_rerank_boost_coef_upper", "as": "boost_coef_upper"},
        {"name": "explore_weight_adjust_avg_rerank_is_boost", "as": "is_boost"},
        {"name": "explore_weight_adjust_avg_rerank_power_weight", "as": "action_power_weight"},
      ],
      export_common_attr = [
        {"name": "user_dynamic_action", "as": "rerank_boost_pwtr"}
      ],
      function_name = "CalcUserDynamicAction",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_weight_adjust_avg_pftr_base_stat", "as": "user_base_stat"},
        {"name": "pftr_avg", "as": "user_dynamic_stat"},
        {"name": "explore_weight_adjust_avg_rerank_boost_coef_lower", "as": "boost_coef_lower"},
        {"name": "explore_weight_adjust_avg_rerank_boost_coef_upper", "as": "boost_coef_upper"},
        {"name": "explore_weight_adjust_avg_rerank_is_boost", "as": "is_boost"},
        {"name": "explore_weight_adjust_avg_rerank_power_weight", "as": "action_power_weight"},
      ],
      export_common_attr = [
        {"name": "user_dynamic_action", "as": "rerank_boost_pftr"}
      ],
      function_name = "CalcUserDynamicAction",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_weight_adjust_avg_pcmtr_base_stat", "as": "user_base_stat"},
        {"name": "pcmtr_avg", "as": "user_dynamic_stat"},
        {"name": "explore_weight_adjust_avg_rerank_boost_coef_lower", "as": "boost_coef_lower"},
        {"name": "explore_weight_adjust_avg_rerank_boost_coef_upper", "as": "boost_coef_upper"},
        {"name": "explore_weight_adjust_avg_rerank_is_boost", "as": "is_boost"},
        {"name": "explore_weight_adjust_avg_rerank_power_weight", "as": "action_power_weight"},
      ],
      export_common_attr = [
        {"name": "user_dynamic_action", "as": "rerank_boost_pcmtr"}
      ],
      function_name = "CalcUserDynamicAction",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_weight_adjust_avg_pptr_base_stat", "as": "user_base_stat"},
        {"name": "pptr_avg", "as": "user_dynamic_stat"},
        {"name": "explore_weight_adjust_avg_rerank_boost_coef_lower", "as": "boost_coef_lower"},
        {"name": "explore_weight_adjust_avg_rerank_boost_coef_upper", "as": "boost_coef_upper"},
        {"name": "explore_weight_adjust_avg_rerank_is_boost", "as": "is_boost"},
        {"name": "explore_weight_adjust_avg_rerank_power_weight", "as": "action_power_weight"},
      ],
      export_common_attr = [
        {"name": "user_dynamic_action", "as": "rerank_boost_pptr"}
      ],
      function_name = "CalcUserDynamicAction",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_weight_adjust_avg_pcltr_base_stat", "as": "user_base_stat"},
        {"name": "pcltr_avg", "as": "user_dynamic_stat"},
        {"name": "explore_weight_adjust_avg_rerank_boost_coef_lower", "as": "boost_coef_lower"},
        {"name": "explore_weight_adjust_avg_rerank_boost_coef_upper", "as": "boost_coef_upper"},
        {"name": "explore_weight_adjust_avg_rerank_is_boost", "as": "is_boost"},
        {"name": "explore_weight_adjust_avg_rerank_power_weight", "as": "action_power_weight"},
      ],
      export_common_attr = [
        {"name": "user_dynamic_action", "as": "rerank_boost_pcltr"}
      ],
      function_name = "CalcUserDynamicAction",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_weight_adjust_avg_fr_score1_base_stat", "as": "user_base_stat"},
        {"name": "fr_score1_avg", "as": "user_dynamic_stat"},
        {"name": "explore_weight_adjust_avg_rerank_boost_coef_lower", "as": "boost_coef_lower"},
        {"name": "explore_weight_adjust_avg_rerank_boost_coef_upper", "as": "boost_coef_upper"},
        {"name": "explore_weight_adjust_avg_rerank_is_boost", "as": "is_boost"},
        {"name": "explore_weight_adjust_avg_rerank_power_weight", "as": "action_power_weight"},
      ],
      export_common_attr = [
        {"name": "user_dynamic_action", "as": "rerank_boost_fr_score1"}
      ],
      function_name = "CalcUserDynamicAction",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_weight_adjust_avg_fr_score2_base_stat", "as": "user_base_stat"},
        {"name": "fr_score2_avg", "as": "user_dynamic_stat"},
        {"name": "explore_weight_adjust_avg_rerank_boost_coef_lower", "as": "boost_coef_lower"},
        {"name": "explore_weight_adjust_avg_rerank_boost_coef_upper", "as": "boost_coef_upper"},
        {"name": "explore_weight_adjust_avg_rerank_is_boost", "as": "is_boost"},
        {"name": "explore_weight_adjust_avg_rerank_power_weight", "as": "action_power_weight"},
      ],
      export_common_attr = [
        {"name": "user_dynamic_action", "as": "rerank_boost_fr_score2"}
      ],
      function_name = "CalcUserDynamicAction",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_weight_adjust_avg_awesome_wtd_base_stat", "as": "user_base_stat"},
        {"name": "awesome_wtd_avg", "as": "user_dynamic_stat"},
        {"name": "explore_weight_adjust_avg_rerank_boost_coef_lower", "as": "boost_coef_lower"},
        {"name": "explore_weight_adjust_avg_rerank_boost_coef_upper", "as": "boost_coef_upper"},
        {"name": "explore_weight_adjust_avg_rerank_is_boost", "as": "is_boost"},
        {"name": "explore_weight_adjust_avg_rerank_power_weight", "as": "action_power_weight"},
      ],
      export_common_attr = [
        {"name": "user_dynamic_action", "as": "rerank_boost_awesome_wtd"}
      ],
      function_name = "CalcUserDynamicAction",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_weight_adjust_avg_fetr_base_stat", "as": "user_base_stat"},
        {"name": "fetr_avg", "as": "user_dynamic_stat"},
        {"name": "explore_weight_adjust_avg_rerank_boost_coef_lower", "as": "boost_coef_lower"},
        {"name": "explore_weight_adjust_avg_rerank_boost_coef_upper", "as": "boost_coef_upper"},
        {"name": "explore_weight_adjust_avg_rerank_is_boost", "as": "is_boost"},
        {"name": "explore_weight_adjust_avg_rerank_power_weight", "as": "action_power_weight"},
      ],
      export_common_attr = [
        {"name": "user_dynamic_action", "as": "rerank_boost_fetr"}
      ],
      function_name = "CalcUserDynamicAction",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_weight_adjust_avg_fountain_eff_base_stat", "as": "user_base_stat"},
        {"name": "fountain_eff_avg", "as": "user_dynamic_stat"},
        {"name": "explore_weight_adjust_avg_rerank_boost_coef_lower", "as": "boost_coef_lower"},
        {"name": "explore_weight_adjust_avg_rerank_boost_coef_upper", "as": "boost_coef_upper"},
        {"name": "explore_weight_adjust_avg_rerank_is_boost", "as": "is_boost"},
        {"name": "explore_weight_adjust_avg_rerank_power_weight", "as": "action_power_weight"},
      ],
      export_common_attr = [
        {"name": "user_dynamic_action", "as": "rerank_boost_fountain_eff"}
      ],
      function_name = "CalcUserDynamicAction",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
  
  def rank_stage2_request_personal_boost(self):
    """
    Module: RankingEnsembleSortModule.py
    功能: request 个性化权重
    Owner: xuwei09
    Date: 2024-01-15
    :return:
    """
    self.gen_common_attr_by_lua(
      attr_map = {
        "explore_ensemble_power_weight_fullrank_pctr_score" : "rerank_boost_pctr * explore_ensemble_power_weight_fullrank_pctr_score",
        "explore_ensemble_power_weight_fullrank_pltr_score" : "rerank_boost_pltr * explore_ensemble_power_weight_fullrank_pltr_score",
        "explore_ensemble_power_weight_fullrank_pwtr_score" : "rerank_boost_pwtr * explore_ensemble_power_weight_fullrank_pwtr_score",
        "explore_ensemble_power_weight_fullrank_pftr_score" : "rerank_boost_pftr * explore_ensemble_power_weight_fullrank_pftr_score",
        "explore_ensemble_power_weight_fullrank_pcltr_score" : "rerank_boost_pcltr * explore_ensemble_power_weight_fullrank_pcltr_score",
        "explore_ensemble_power_weight_fullrank_pptr_score" : "rerank_boost_pptr * explore_ensemble_power_weight_fullrank_pptr_score",
        "fr_pmctr_rank_weight" : "rerank_boost_pcmtr * fr_pmctr_rank_weight",
        "explore_ensemble_power_weight_fullrank_fr_score1_score" : "rerank_boost_fr_score1 * explore_ensemble_power_weight_fullrank_fr_score1_score",
        "explore_ensemble_power_weight_fullrank_fr_score2_score" : "rerank_boost_fr_score2 * explore_ensemble_power_weight_fullrank_fr_score2_score",
        "awesome_wtd_weight_push" : "rerank_boost_awesome_wtd * awesome_wtd_weight_push",
        "hot_fountain_fetr_weight_push" : "rerank_boost_fetr * hot_fountain_fetr_weight_push",
        "hot_fountain_fountain_eff_weight_push" : "rerank_boost_fountain_eff * hot_fountain_fountain_eff_weight_push",
      }
    )
    return self

  def calculate_user_timely_diversity_score(self):
    """
    Module: RankingEnsembleSortModule.py
    功能: calculate diversity entropy ratio
    Owner: guohao
    Date: 2025-01-08
    :return:
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "user_info_ptr",
        {"name": "explore_user_timely_diversity_entropy_num_threshold", "as": "num_threshold"},
        {"name": "explore_user_timely_diversity_entropy_time_ms_threshold", "as": "time_ms_threshold"},
        {"name": "explore_user_timely_diversity_entropy_adjust_ratio", "as": "adjust_ratio"},
      ],
      export_common_attr = [
        {"name": "output_weight", "as": "explore_user_timely_diversity_entropy_score"},
        {"name": "click_output_weight", "as": "explore_user_timely_diversity_click_entropy_score"},
        {"name": "show_output_weight", "as": "explore_user_timely_diversity_show_entropy_score"},
      ],
      function_name = "GetTimelyHetuEntropyRate",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def user_timely_diversity_pxtr_weight_adjust(self):
    """
    Module: RankingEnsembleSortModule.py
    功能: pxtr weight adjust
    Owner: guohao
    Date: 2025-01-08
    :return:
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_user_timely_diversity_entropy_score", "as": "weight"},
        {"name": "explore_ensemble_power_weight_fullrank_pctr_score", "as": "value"},
      ],
      export_common_attr = [
        {"name": "new_value", "as": "explore_ensemble_power_weight_fullrank_pctr_score"},
      ],
      function_name = "CalExploreDoubleMultiDouble",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def rank_stage2_personal_cem(self):
    """
    Module: RankingEnsembleSortModule.py
    功能: 个性化 cem
    Owner: liuhao07
    Date: 2023-07-13
    :return:
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "_USER_ID_", "as": "user_id"},
        {"name": "explore_rank_stage2_personal_cem_exp_group_num", "as": "exp_group_num"},
        {"name": "explore_rank_stage2_personal_cem_model_key", "as": "model_key"},
      ],
      export_common_attr = [
        {"name": "personal_cem_redis_key", "as": "explore_rank_stage2_personal_cem_redis_key"}
      ],
      function_name = "GenPersonalCemKey",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .get_common_attr_from_redis(
      cluster_name = "recoPersonalCem",
      timeout_ms = 10,
      cache_bits = 8,
      cache_expire_second = 600,
      redis_params = [
        {
          "redis_key": "{{explore_rank_stage2_personal_cem_redis_key}}",
          "output_attr_name": "explore_rank_stage2_personal_cem_model"
        }
      ]
    ) \
    .pack_item_attr(
      item_source = {"reco_results": True},
      mappings = [
        {
          "from_item_attr": "score_pwtr",
          "to_common_attr": "score_pwtr_avg",
          "aggregator":"avg"
        },
        {
          "from_item_attr": "score_pwtr",
          "to_common_attr": "score_pwtr_dev",
          "aggregator":"dev"
        },
        {
          "from_item_attr": "score_pltr",
          "to_common_attr": "score_pltr_avg",
          "aggregator":"avg"
        },
        {
          "from_item_attr": "score_pltr",
          "to_common_attr": "score_pltr_dev",
          "aggregator":"dev"
        },
        {
          "from_item_attr": "score_pcmtr",
          "to_common_attr": "score_pcmtr_avg",
          "aggregator":"avg"
        },
        {
          "from_item_attr": "score_pcmtr",
          "to_common_attr": "score_pcmtr_dev",
          "aggregator":"dev"
        },
        {
          "from_item_attr": "score_pftr",
          "to_common_attr": "score_pftr_avg",
          "aggregator":"avg"
        },
        {
          "from_item_attr": "score_pftr",
          "to_common_attr": "score_pftr_dev",
          "aggregator":"dev"
        },
      ]
    ) \
    .gen_common_attr_by_lua(
      attr_map={
        "user_emp_ltr_group_base": "user_emp_ltr / user_group_emp_ltr",
        "user_emp_wtr_group_base": "user_emp_wtr / user_group_emp_wtr",
        "user_emp_ftr_group_base": "user_emp_ftr / user_group_emp_ftr",
        "user_emp_cmtr_group_base": "user_emp_cmtr / user_group_emp_cmtr",
      }
    ) \
    .explore_personal_cem_weight(
      model_str_attr = "explore_rank_stage2_personal_cem_model",
      save_exp_info_attr = "personal_cem_exp_info",
      weight_configs = [
        {
          "weight_name": "explore_rank_stage2_es_wtr_coeff",
          "output_attr": "explore_rank_stage2_es_wtr_coeff",
          "enable_activation": "explore_rank_stage2_personal_cem_es_wtr_coeff_enable_activation",
          "min_value_attr": "explore_rank_stage2_personal_cem_es_wtr_coeff_min_value",
          "max_value_attr": "explore_rank_stage2_personal_cem_es_wtr_coeff_max_value"
        },
        {
          "weight_name": "explore_rank_stage2_es_ltr_coeff",
          "output_attr": "explore_rank_stage2_es_ltr_coeff",
          "enable_activation": "explore_rank_stage2_personal_cem_es_ltr_coeff_enable_activation",
          "min_value_attr": "explore_rank_stage2_personal_cem_es_ltr_coeff_min_value",
          "max_value_attr": "explore_rank_stage2_personal_cem_es_ltr_coeff_max_value"
        },
        {
          "weight_name": "explore_rank_stage2_es_cmtr_coeff",
          "output_attr": "explore_rank_stage2_es_cmtr_coeff",
          "enable_activation": "explore_rank_stage2_personal_cem_es_cmtr_coeff_enable_activation",
          "min_value_attr": "explore_rank_stage2_personal_cem_es_cmtr_coeff_min_value",
          "max_value_attr": "explore_rank_stage2_personal_cem_es_cmtr_coeff_max_value"
        },
        {
          "weight_name": "explore_rank_stage2_es_ftr_coeff",
          "output_attr": "explore_rank_stage2_es_ftr_coeff",
          "enable_activation": "explore_rank_stage2_personal_cem_es_ftr_coeff_enable_activation",
          "min_value_attr": "explore_rank_stage2_personal_cem_es_ftr_coeff_min_value",
          "max_value_attr": "explore_rank_stage2_personal_cem_es_ftr_coeff_max_value"
        }
      ],
      feature_configs = [
        {
          "feature_name": "score_pwtr_avg",
          "feature_attr": "score_pwtr_avg",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage2_personal_cem_feature_score_pwtr_avg_min_value",
          "max_value_attr": "explore_rank_stage2_personal_cem_feature_score_pwtr_avg_max_value",
        },
        {
          "feature_name": "score_pwtr_dev",
          "feature_attr": "score_pwtr_dev",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage2_personal_cem_feature_score_pwtr_dev_min_value",
          "max_value_attr": "explore_rank_stage2_personal_cem_feature_score_pwtr_dev_max_value",
        },
        {
          "feature_name": "score_pltr_avg",
          "feature_attr": "score_pltr_avg",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage2_personal_cem_feature_score_pltr_avg_min_value",
          "max_value_attr": "explore_rank_stage2_personal_cem_feature_score_pltr_avg_max_value",
        },
        {
          "feature_name": "score_pltr_dev",
          "feature_attr": "score_pltr_dev",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage2_personal_cem_feature_score_pltr_dev_min_value",
          "max_value_attr": "explore_rank_stage2_personal_cem_feature_score_pltr_dev_max_value",
        },
        {
          "feature_name": "score_pcmtr_avg",
          "feature_attr": "score_pcmtr_avg",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage2_personal_cem_feature_score_pcmtr_avg_min_value",
          "max_value_attr": "explore_rank_stage2_personal_cem_feature_score_pcmtr_avg_max_value",
        },
        {
          "feature_name": "score_pcmtr_dev",
          "feature_attr": "score_pcmtr_dev",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage2_personal_cem_feature_score_pcmtr_dev_min_value",
          "max_value_attr": "explore_rank_stage2_personal_cem_feature_score_pcmtr_dev_max_value",
        },
        {
          "feature_name": "score_pftr_avg",
          "feature_attr": "score_pftr_avg",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage2_personal_cem_feature_score_pftr_avg_min_value",
          "max_value_attr": "explore_rank_stage2_personal_cem_feature_score_pftr_avg_max_value",
        },
        {
          "feature_name": "score_pftr_dev",
          "feature_attr": "score_pftr_dev",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage2_personal_cem_feature_score_pftr_dev_min_value",
          "max_value_attr": "explore_rank_stage2_personal_cem_feature_score_pftr_dev_max_value",
        },
        {
          "feature_name": "user_emp_ltr",
          "feature_attr": "user_emp_ltr",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage2_personal_cem_feature_user_emp_ltr_min_value",
          "max_value_attr": "explore_rank_stage2_personal_cem_feature_user_emp_ltr_max_value",
        },
        {
          "feature_name": "user_emp_wtr",
          "feature_attr": "user_emp_wtr",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage2_personal_cem_feature_user_emp_wtr_min_value",
          "max_value_attr": "explore_rank_stage2_personal_cem_feature_user_emp_wtr_max_value",
        },
        {
          "feature_name": "user_emp_cmtr",
          "feature_attr": "user_emp_cmtr",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage2_personal_cem_feature_user_emp_cmtr_min_value",
          "max_value_attr": "explore_rank_stage2_personal_cem_feature_user_emp_cmtr_max_value",
        },
        {
          "feature_name": "user_emp_ftr",
          "feature_attr": "user_emp_ftr",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage2_personal_cem_feature_user_emp_ftr_min_value",
          "max_value_attr": "explore_rank_stage2_personal_cem_feature_user_emp_ftr_max_value",
        },
        {
          "feature_name": "user_group_emp_ltr",
          "feature_attr": "user_group_emp_ltr",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage2_personal_cem_feature_user_group_emp_ltr_min_value",
          "max_value_attr": "explore_rank_stage2_personal_cem_feature_user_group_emp_ltr_max_value",
        },
        {
          "feature_name": "user_group_emp_wtr",
          "feature_attr": "user_group_emp_wtr",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage2_personal_cem_feature_user_group_emp_wtr_min_value",
          "max_value_attr": "explore_rank_stage2_personal_cem_feature_user_group_emp_wtr_max_value",
        },
        {
          "feature_name": "user_group_emp_ftr",
          "feature_attr": "user_group_emp_ftr",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage2_personal_cem_feature_user_group_emp_ftr_min_value",
          "max_value_attr": "explore_rank_stage2_personal_cem_feature_user_group_emp_ftr_max_value",
        },
        {
          "feature_name": "user_group_emp_cmtr",
          "feature_attr": "user_group_emp_cmtr",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage2_personal_cem_feature_user_group_emp_cmtr_min_value",
          "max_value_attr": "explore_rank_stage2_personal_cem_feature_user_group_emp_cmtr_max_value",
        },
        {
          "feature_name": "user_emp_ltr_group_base",
          "feature_attr": "user_emp_ltr_group_base",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage2_personal_cem_feature_user_emp_ltr_group_base_min_value",
          "max_value_attr": "explore_rank_stage2_personal_cem_feature_user_emp_ltr_group_base_max_value",
        },
        {
          "feature_name": "user_emp_wtr_group_base",
          "feature_attr": "user_emp_wtr_group_base",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage2_personal_cem_feature_user_emp_wtr_group_base_min_value",
          "max_value_attr": "explore_rank_stage2_personal_cem_feature_user_emp_wtr_group_base_max_value",
        },
        {
          "feature_name": "user_emp_ftr_group_base",
          "feature_attr": "user_emp_ftr_group_base",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage2_personal_cem_feature_user_emp_ftr_group_base_min_value",
          "max_value_attr": "explore_rank_stage2_personal_cem_feature_user_emp_ftr_group_base_max_value",
        },
        {
          "feature_name": "user_emp_cmtr_group_base",
          "feature_attr": "user_emp_cmtr_group_base",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage2_personal_cem_feature_user_emp_cmtr_group_base_min_value",
          "max_value_attr": "explore_rank_stage2_personal_cem_feature_user_emp_cmtr_group_base_max_value",
        },
      ]
    ) \
    .export_attr_to_kafka(
      kafka_topic = "explore_personal_cem",
      common_attrs = ["request_id", "_USER_ID_", "_DEVICE_ID_", "personal_cem_exp_info"],
    )
    return self

  def rank_stage2_personal_cem_es_weight_adjust(self):
    """
    Module: RankingEnsembleSortModule
    功能: 个性化 cem 调权
    Owner: liuhao07
    Date: 2023-07-13
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_ensemble_power_weight_fullrank_pwtr_score", "as": "ori_weight"},
        {"name": "explore_rank_stage2_es_wtr_coeff", "as": "coeff"},
        {"name": "explore_rank_stage2_es_wtr_weight_min_value", "as": "weight_min_value"},
        {"name": "explore_rank_stage2_es_wtr_weight_max_value", "as": "weight_max_value"},
      ],
      export_common_attr = [
        {"name": "new_weight", "as": "explore_ensemble_power_weight_fullrank_pwtr_score"}
      ],
      function_name = "AdjustWeight",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_ensemble_power_weight_fullrank_pltr_score", "as": "ori_weight"},
        {"name": "explore_rank_stage2_es_ltr_coeff", "as": "coeff"},
        {"name": "explore_rank_stage2_es_ltr_weight_min_value", "as": "weight_min_value"},
        {"name": "explore_rank_stage2_es_ltr_weight_max_value", "as": "weight_max_value"},
      ],
      export_common_attr = [
        {"name": "new_weight", "as": "explore_ensemble_power_weight_fullrank_pltr_score"}
      ],
      function_name = "AdjustWeight",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_pmctr_rank_weight", "as": "ori_weight"},
        {"name": "explore_rank_stage2_es_cmtr_coeff", "as": "coeff"},
        {"name": "explore_rank_stage2_es_cmtr_weight_min_value", "as": "weight_min_value"},
        {"name": "explore_rank_stage2_es_cmtr_weight_max_value", "as": "weight_max_value"},
      ],
      export_common_attr = [
        {"name": "new_weight", "as": "fr_pmctr_rank_weight"}
      ],
      function_name = "AdjustWeight",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_ensemble_power_weight_fullrank_pftr_score", "as": "ori_weight"},
        {"name": "explore_rank_stage2_es_ftr_coeff", "as": "coeff"},
        {"name": "explore_rank_stage2_es_ftr_weight_min_value", "as": "weight_min_value"},
        {"name": "explore_rank_stage2_es_ftr_weight_max_value", "as": "weight_max_value"},
      ],
      export_common_attr = [
        {"name": "new_weight", "as": "explore_ensemble_power_weight_fullrank_pftr_score"}
      ],
      function_name = "AdjustWeight",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def refinement_boost_personified_author(self):
    """
    Module: RankingEnsembleSortModule
    功能: 细分用户和视频维度，精细化对人格化账号提权
    Owner: xubaoquan
    Date: 2023-07-12
    :return:
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "basic_info_age_segment_v2", "as": "basic_info_age_segment_v2"},
        {"name": "basic_info_gender_v2", "as": "basic_info_gender_v2"},
        {"name": "explore_personifed_author_boost_ptr", "as": "boost_map_ptr"},
        {"name": "refinement_boost_personified_author_redis_prefix", "as": "redis_prefix"},
        {"name": "rank_refinement_boost_personified_author_power_weight", "as": "power_weight"},
      ],
      import_item_attr = [
        {"name": "author__gender", "as": "author__gender"},
        {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_level_one_list"},
        {"name": "explore_fr_ensemble_score", "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "explore_fr_ensemble_score"},
      ],
      target_item = { 
        "eyeshot_source" : 1
      },
      function_name = "UniverseRefinementBoost",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
  
  def write_rank_neg_result_to_redis(self):
    """
    Module: RankingEnsembleSortModule
    功能: 将精排尾部结果写入redis进行过滤
    Owner: liuhao07
    Date: 2023-07-27
    :return:
    """
    self.if_("explore_enable_rank_select_rank_neg_result == 1", to_be_delete = "date=2024-05-29;committer=liuhao07") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_rank_neg_photo_target_ratio", "as": "target_ratio"},
        ],
        import_item_attr = [
          {"name": "cascade_final_index", "as": "before_index"},
          {"name": "rank_final_index", "as": "after_index"},
          "photo_id"
        ],
        export_common_attr = [
          {"name": "target_pids", "as": "rank_neg_photo_id_list"},
        ],
        function_name = "SelectRecoNegPids",
        class_name = "ExploreLightFunctionSetV2",
        range_start = "{{explore_rank_neg_photo_index}}"
      ) \
    .else_() \
      .pack_item_attr(
        item_source = {
          "reco_results": True
        },
        mappings = [{
          "from_item_attr": "item_key",
          "to_common_attr": "rank_neg_photo_id_list",
          "aggregator": "concat"
        }],
        range_start = "{{explore_rank_neg_photo_index}}"
      ) \
    .end_() \
    .if_("explore_enable_rank_write_rank_stage1_result_to_redis == 1") \
      .pack_common_attr(
        input_common_attrs = [
          "photo_id_trunc_stage1",
          "rank_neg_photo_id_list"
        ],
        output_common_attr = "rank_neg_photo_id_list",
        deduplicate = True,
      ) \
    .end_() \
    .pack_common_attr(
      input_common_attrs = [
        "rank_neg_photo_id_list",
        "rank_neg_photo_id_filter_list"
      ],
      output_common_attr = "rank_neg_photo_id_list",
      deduplicate = True,
      limit_num = "{{explore_rank_neg_photo_size}}",
    ) \
    .write_to_redis(
      kcc_cluster = "recoExploreNegPhoto",
      timeout = 10,
      expire_second = "{{explore_rank_neg_photo_redis_expire_seconds}}",
      key_prefix = "{{explore_rank_neg_photo_key_prefix}}",
      key = "{{_DEVICE_ID_}}",
      value = "{{rank_neg_photo_id_list}}"
    )
    return self
  
  def write_rank_pos_result_to_redis(self):
    """
    Module: RankingEnsembleSortModule
    功能: 将精排头部结果写入redis进行召回
    Owner: liuhao07
    Date: 2023-08-31
    :return:
    """
    self.pack_item_attr(
      item_source = {
        "reco_results": True
      },
      mappings = [{
        "from_item_attr": "item_key",
        "to_common_attr": "rank_pos_photo_id_list",
        "aggregator": "concat"
      }],
      range_end = "{{explore_rank_pos_photo_end_index}}"
    ) \
    .pack_common_attr(
      input_common_attrs = [
        "rank_pos_photo_id_list",
        "explore_rank_pos_photo_id_retrieval_list"
      ],
      output_common_attr = "rank_pos_photo_id_list",
      deduplicate = True,
      limit_num = "{{explore_rank_pos_photo_size}}",
    ) \
    .write_to_redis(
      kcc_cluster = "recoExploreNegPhoto",
      timeout = 10,
      expire_second = "{{explore_rank_pos_photo_redis_expire_seconds}}",
      key_prefix = "{{explore_rank_pos_photo_key_prefix}}",
      key = "{{_DEVICE_ID_}}",
      value = "{{rank_pos_photo_id_list}}"
    )
    return self

  def boost_bot_content_retr(self):
    """
    Module: RankingEnsembleSortModule
    功能: 对bot_content进行boost
    Owner: lijinyu
    Date: 2023-08-08
    :return:
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_fr_hot_content_retr_boost_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "reason" : [10030, 10031, 10032]
      }
    )
    return self

  def gen_hot_ranking_retr_score(self):
    self.set_attr_value(
      item_attrs = [{
        "name": "is_hot_ranking_retr_score",
        "type": "double",
        "value": 1.0,
      }],
      target_item = {
        "reason": [10042]
      }
    )
    return self

  def gen_prefer_author_ranking_retr_score(self):
    self.set_attr_value(
      item_attrs = [{
        "name": "is_prefer_author_ranking_retr_score",
        "type": "double",
        "value": 1.0,
      }],
      target_item = {
        "reason": [946, 674, 820, 1899, 1836]
      }
    )
    return self
  
  def calc_coordinated_queues_score(self):
    self \
    .pack_item_attr(
      item_source = {
        "reco_results": True,
      },
      mappings = [
        {
          "aggregator": "min",
          "from_item_attr": "pevtr",
          "to_common_attr": "pevtr_min"
        },
        {
          "aggregator": "max",
          "from_item_attr": "pevtr",
          "to_common_attr": "pevtr_max"
        },
        {
          "aggregator": "min",
          "from_item_attr": "awesome_wtd_score",
          "to_common_attr": "awesome_wtd_score_min"
        },
        {
          "aggregator": "max",
          "from_item_attr": "awesome_wtd_score",
          "to_common_attr": "awesome_wtd_score_max"
        },
        {
          "aggregator": "min",
          "from_item_attr": "fr_score1",
          "to_common_attr": "fr_score1_min"
        },
        {
          "aggregator": "max",
          "from_item_attr": "fr_score1",
          "to_common_attr": "fr_score1_max"
        },
        {
          "aggregator": "min",
          "from_item_attr": "corr_cpr",
          "to_common_attr": "corr_cpr_min"
        },
        {
          "aggregator": "max",
          "from_item_attr": "corr_cpr",
          "to_common_attr": "corr_cpr_max"
        },
        {
          "aggregator": "min",
          "from_item_attr": "score_psvr",
          "to_common_attr": "corr_pctr_min"
        },
        {
          "aggregator": "max",
          "from_item_attr": "score_psvr",
          "to_common_attr": "corr_pctr_max"
        },
        {
          "aggregator": "min",
          "from_item_attr": "fr_score2",
          "to_common_attr": "fr_score2_min"
        },
        {
          "aggregator": "max",
          "from_item_attr": "fr_score2",
          "to_common_attr": "fr_score2_max"
        },
      ]
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "coordinated_pevtr_weight", "as": "fullrank_triplem_evtr_weight"},
        {"name": "coordinated_awesome_wtd_score_weight", "as": "fullrank_triplem_vtr_weight"},
        {"name": "coordinated_fr_score1_weight", "as": "fullrank_triplem_lvtr_weight"},
        {"name": "coordinated_corr_cpr_weight", "as": "fullrank_triplem_cpr_weight"},
        {"name": "coordinated_corr_pctr_weight", "as": "fullrank_triplem_fintr_weight"},
        {"name": "coordinated_fr_score2_weight", "as": "fullrank_triplem_enable_evtr_v2_weight"},
        "pevtr_min",
        "pevtr_max",
        {"name": "awesome_wtd_score_min", "as": "pvtr_min"},
        {"name": "awesome_wtd_score_max", "as": "pvtr_max"},
        {"name": "fr_score1_min", "as":  "plvtr_min"},
        {"name": "fr_score1_max", "as": "plvtr_max"},
        {"name": "corr_cpr_min", "as":  "pcpr_min"},
        {"name": "corr_cpr_max", "as": "pcpr_max"},
        {"name": "corr_pctr_min", "as": "pfintr_min"},
        {"name": "corr_pctr_max", "as": "pfintr_max"},
        {"name": "fr_score2_min", "as": "pevtr_v2_min"},
        {"name": "fr_score2_max", "as": "pevtr_v2_max"}
      ],
      import_item_attr = [
        {"name": "pevtr", "as": "evtr"},
        {"name": "awesome_wtd_score", "as": "fintr"},
        {"name": "fr_score1", "as": "lvtr"},
        {"name": "corr_cpr", "as": "cpr"},
        {"name": "score_psvr", "as": "evtr_v2"},
        {"name": "fr_score2", "as": "vtr"}
      ],
      export_item_attr = [
        {"name": "fullrank_triplem_score", "as": "coordinated_watchtime_score"},
      ],
      function_name = "CalTriplemScore",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def user_age_based_weight_adjust_all(self):
    self \
    .user_attr_based_weight_adjust(
      "user_age_segment",
      "explore_rank_s2_age_based_pctr_weight_adjust_list",
      "explore_ensemble_power_weight_fullrank_pctr_score"
    ) \
    .user_attr_based_weight_adjust(
      "user_age_segment",
      "explore_rank_s2_age_based_pltr_weight_adjust_list",
      "explore_ensemble_power_weight_fullrank_pltr_score"
    ) \
    .user_attr_based_weight_adjust(
      "user_age_segment",
      "explore_rank_s2_age_based_pwtr_weight_adjust_list",
      "explore_ensemble_power_weight_fullrank_pwtr_score"
    ) \
    .user_attr_based_weight_adjust(
      "user_age_segment",
      "explore_rank_s2_age_based_pftr_weight_adjust_list",
      "explore_ensemble_power_weight_fullrank_pftr_score"
    ) \
    .user_attr_based_weight_adjust(
      "user_age_segment",
      "explore_rank_s2_age_based_pcmtr_weight_adjust_list",
      "fr_pmctr_rank_weight"
    ) \
    .user_attr_based_weight_adjust(
      "user_age_segment",
      "explore_rank_s2_age_based_awesome_wtd_weight_adjust_list",
      "awesome_wtd_weight_push"
    )
    return self

  def user_attr_based_weight_adjust(self, user_attr, weight_list_str, weight_attr):
    weight_list = weight_list_str + "_to_list"
    weight_adjust_coef = weight_attr + "_adjust_coef_by_" + user_attr

    self.split_string(
      input_common_attr = weight_list_str,
      output_common_attr = weight_list,
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_double = True,
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": user_attr, "as": "index"},
        {"name": weight_list, "as": "weight_list"},
      ],
      export_common_attr = [
        {"name": "weight", "as": weight_adjust_coef},
      ],
      function_name = "GetDoubleValueInList",
      class_name = "ExploreLightFunctionSetV2",\
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": weight_attr, "as": "value"},
        {"name": weight_adjust_coef, "as": "weight"},
      ],
      export_common_attr = [
        {"name": "new_value", "as": weight_attr},
      ],
      function_name = "CalExploreDoubleMultiDouble",
      class_name = "ExploreLightFunctionSetV2",\
    )
    return self

  def not_cover_audit_photo_discount(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_fr_not_cover_audit_enable_follow_author_exemption", "as": "enable_follow_author_exemption"},
        "follow_aids",
        {"name": "page_index", "as": "page"},
      ],
      import_item_attr = [
        "author__id",
        "audit_hot_cover_level",
      ],
      export_item_attr = [
        {"name": "is_not_cover_audit_for_first_page", "as": "is_not_cover_audit_for_first_page_fr"}
      ],
      function_name = "IsNotCoverAuditForFirstPage",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_fr_not_cover_audit_photo_discount_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "is_not_cover_audit_for_first_page_fr" : 1
      }
    )
    return self

  def boost_recent_consume_photo(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_fr_s2_recent_consume_photo_boost_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "is_user_recent_consume_photo" : 1,
      }
    )
    return self
  
  def boost_audit_good_photo(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_fr_s2_audit_good_photo_boost_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "is_audit_good_photo" : 1,
      }
    )
    return self

  def cropped_photo_discount(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_cropped_photo_discount_threshold", "as": "threshold"},
      ],
      import_item_attr = [
        "cover_origin_width",
        "cover_origin_height",
      ],
      export_item_attr = [
        "is_cropped_photo"
      ],
      function_name = "IsCroppedPhoto",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_fr_cropped_photo_discount_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "is_cropped_photo" : 1
      }
    )
    return self

  def high_photo_count_author_adjust(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "high_photo_count_author_map_ptr",
        {"name": "explore_rank_high_photo_count_author_photo_coeff", "as": "boost_discount_coeff"},
        {"name": "explore_rank_high_photo_count_author_post_num_base", "as": "post_num_base"},
      ],
      import_item_attr = [
        "author__id",
        {"name": "explore_fr_ensemble_score", "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "HighPhotoCountAuthorPhotoAdjust",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def high_photo_count_author_adjust_v2(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "high_upload_photo_author_map_ptr",
        {"name": "explore_rank_high_photo_count_author_photo_coeff", "as": "boost_discount_coeff"},
        {"name": "explore_rank_high_photo_count_author_pos_neg_ratio_coeff", "as": "pos_neg_ratio_coeff"},
      ],
      import_item_attr = [
        "author__id",
        {"name": "explore_fr_ensemble_score", "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "HighPhotoCountAuthorPhotoAdjustV2",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def new_interest_explore_boost(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_fr_new_interest_explore_boost_coeff", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item={
        "is_new_interest_explore": 1,
      },
    )
    return self

  def high_global_emphtr_discount(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_fr_high_global_emphtr_discount_coeff", "as": "boost_discount_coeff"},
        {"name": "explore_fr_high_global_emphtr_discount_thres", "as": "boost_discount_thres"},
      ],
      import_item_attr = [
        {"name": "global_emphtr_score", "as": "need_item_attr"},
        {"name": "explore_fr_ensemble_score", "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "BoostOrDiscountWithThres",
      class_name = "ExploreLightFunctionSetV2",
    ) \

    return self

  def fr_search_score_boost(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_fr_search_score_boost_coeff", "as": "boost_discount_coeff"},
        {"name": "explore_fr_search_score_boost_thres", "as": "boost_discount_thres"},
      ],
      import_item_attr = [
        {"name": "search_score", "as": "need_item_attr"},
        {"name": "explore_fr_ensemble_score", "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "BoostOrDiscountWithThres",
      class_name = "ExploreLightFunctionSetV2",
    ) \

    return self
  
  def fr_boost_ua_long_view(self):
    self \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_boost_ua_long_view_alpha", "as": "alpha_boost_weight"},
        {"name": "fr_boost_ua_long_view_beta", "as": "beta_boost_weight"},
      ],
      import_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "explore_fr_ensemble_score"},
      ],
      target_item = {"is_long_view_author": 1},
      function_name = "EnsembleScorePowBoost",
      class_name = "ExploreLightFunctionSetV2",
    ) \

    return self

  def fr_boost_click_count(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": "ensemble_score"},
        "explore_stat__click_count",
      ],
      import_common_attr = [
        {"name": "fr_boost_click_thred", "as": "click_thred"},
        {"name": "fr_boost_click_count_alpha", "as": "boost_click_count_alpha"},
        {"name": "fr_boost_click_count_beta", "as": "boost_click_count_beta"},
        {"name": "fr_boost_click_count_omega", "as": "boost_click_count_omega"},
        {"name": "fr_boost_click_val_max", "as": "boost_click_val_max"},
        {"name": "fr_boost_click_val_min", "as": "boost_click_val_min"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "explore_fr_ensemble_score"}
      ],
      function_name = "BoostClickCount",
      class_name = "ExploreLightFunctionSetV2"
    )
    return self

  def fr_boost_user_author_reason(self):
    self.enrich_attr_by_light_function(
      target_reason = [10045],
      import_common_attr = [
        {"name": "fr_boost_ua_reason_weight", "as": "boost_weight"},
        {"name": "fr_weaken_ua_reason_weight", "as": "weaken_weight"},
      ],
      import_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "EnsembleScoreBoost",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def explore_pxtr_calibration(self):
    """
    Module: RankingScoreModule
    功能: 精排pxtr校准模块
    Owner: wangyalong03
    Date: 2024-04-16
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_pctr_calibration_upload_time", "as": "pctr_calibration_upload_time"},
        {"name": "explore_pltr_calibration_upload_time", "as": "pltr_calibration_upload_time"},
        {"name": "explore_pwtr_calibration_upload_time", "as": "pwtr_calibration_upload_time"},
        {"name": "explore_awesome_wtd_calibration_duration", "as": "awesome_wtd_calibration_duration"},
      ],
      import_item_attr = [
        "upload_time",
        "duration_ms",
        {"name": "corr_pctr", "as": "pctr"},
        {"name": "corr_pwtr", "as": "pwtr"},
        "pltr",
        "awesome_wtd",
      ],
      export_item_attr = [
        {"name": "pctr", "as": "corr_pctr"},
        {"name": "pwtr", "as": "corr_pwtr"},
        "pltr",
        "awesome_wtd",
      ],
      function_name = "PxtrCalibration",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def fr_timeliness_photo_boost(self):
    """
    Owner: liuhao07
    Date: 2024-05-06
    """
    self.split_string(
      input_common_attr = "explore_timeliness_hetu_list_str",
      output_common_attr = "explore_timeliness_hetu_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True,
    ) \
    .enrich_attr_by_light_function( # 时效性内容根据上传时间提权
      import_common_attr = [
        {"name": "explore_timeliness_photo_boost_map_str", "as": "boost_map_str"},
        {"name": "explore_timeliness_hetu_list", "as": "timeliness_hetu_list"},
      ],
      import_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": "score"},
        "upload_time",
        "hetu_tag_level_info_v2__hetu_level_one"
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "TimelinessPhotoBoost",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self


  def explore_eff_ctr_corr(self):
    """
    Module: RankingScoreModule
    功能: 真实ctr替换ctr
    Owner: wangyalong03
    Date: 2024-04-19
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "eff_ctr_corr_svr_thres",
        "eff_ctr_corr_pic_coeff",
        "eff_ctr_corr_alpha",
        "eff_ctr_corr_power",
        "eff_ctr_corr_category_power"
      ],
      import_item_attr = [
        {"name": "corr_pctr", "as": "pctr"},
        "hetu_tag_level_info_v2__hetu_level_one",
        "psvr",
        "is_picture"
      ],
      export_item_attr = [
        {"name": "pctr", "as": "corr_pctr"},
      ],
      function_name = "RealCtrReplaceCtr",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
  
  def explore_replace_ctr_corr(self):
    """
    Module: RankingScoreModule
    功能: 真实ctr替换ctr队列
    Owner: xuwei09
    Date: 2024-05-09
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_replace_ctr_corr_pic_coeff", "as": "eff_ctr_corr_pic_coeff"},
        {"name": "explore_replace_ctr_corr_alpha", "as": "eff_ctr_corr_alpha"},
        {"name": "explore_replace_ctr_corr_power", "as": "eff_ctr_corr_power"},
      ],
      import_item_attr = [
        {"name": "corr_pctr", "as": "pctr"},
        "hetu_tag_level_info_v2__hetu_level_one",
        "psvr",
        "is_picture"
      ],
      export_item_attr = [
        {"name": "pctr", "as": "corr_pctr_psvr"},
      ],
      function_name = "RealCtrReplaceCtr",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def fr_boost_topk_high_ctr_photo(self):
    """
    Owner: fengjingping
    Date: 2024-05-13
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "user_info_ptr",
      ],
      export_common_attr = [
        "user_hot_click_cnt",
      ],
      function_name = "HotUserClickPhotoCount",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        "fr_exptag_reason_topk",
        "weaken_exptag_reason_weight",
        "enable_open_judge_fr_click_count_threshold",
        "fr_click_count_threshold",
        "user_hot_click_cnt",
      ],
      import_item_attr = [
        "explore_fr_ensemble_score",
        "score_pctr",
      ],
      export_item_attr = [
        "explore_fr_ensemble_score",
      ],
      function_name = "BoostTopkHighCtrPhoto",
      class_name = "ExploreLightFunctionSetV2",
      target_reason = 800
    )
    return self

  def is_lower_avg_ctr_users(self):
    """
    Owner: fengjingping
    Date: 2024-05-29
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "boost_avg_ctr_threshold",
      ],
      import_item_attr = [
        "score_pctr"
      ],
      export_common_attr = [
        "is_lower_avg_ctr_users",
      ],
      function_name = "IsLowAvgCtrUser",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def fr_boost_lower_avg_ctr_photo(self):
    """
    Owner: fengjingping
    Date: 2024-05-29
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "boost_lower_avg_ctr_photo_weight", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "reason" : [800]
      },
    )
    return self

  def fr_cal_quantile_relative_score(self):
    """
    Owner: xuwei09
    Date: 2024-06-03
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_pctr_quantile_threshold", "as": "quantile_threshold"},
        {"name": "fr_pctr_quantile_k", "as": "quantile_k"},
        {"name": "fr_pctr_quantile_alpha", "as": "quantile_alpha"},
      ],
      import_item_attr = [
        {"name": "corr_pctr", "as": "pxtr"},
      ],
      export_item_attr = [
        {"name": "quantile_score", "as": "pctr_quantile_score"},
      ],
      function_name = "QuantileScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_pltr_quantile_threshold", "as": "quantile_threshold"},
        {"name": "fr_pltr_quantile_k", "as": "quantile_k"},
        {"name": "fr_pltr_quantile_alpha", "as": "quantile_alpha"},
      ],
      import_item_attr = [
        {"name": "pltr", "as": "pxtr"},
      ],
      export_item_attr = [
        {"name": "quantile_score", "as": "pltr_quantile_score"},
      ],
      function_name = "QuantileScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_pwtr_quantile_threshold", "as": "quantile_threshold"},
        {"name": "fr_pwtr_quantile_k", "as": "quantile_k"},
        {"name": "fr_pwtr_quantile_alpha", "as": "quantile_alpha"},
      ],
      import_item_attr = [
        {"name": "pwtr", "as": "pxtr"},
      ],
      export_item_attr = [
        {"name": "quantile_score", "as": "pwtr_quantile_score"},
      ],
      function_name = "QuantileScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_pftr_quantile_threshold", "as": "quantile_threshold"},
        {"name": "fr_pftr_quantile_k", "as": "quantile_k"},
        {"name": "fr_pftr_quantile_alpha", "as": "quantile_alpha"},
      ],
      import_item_attr = [
        {"name": "pftr", "as": "pxtr"},
      ],
      export_item_attr = [
        {"name": "quantile_score", "as": "pftr_quantile_score"},
      ],
      function_name = "QuantileScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_pcmtr_quantile_threshold", "as": "quantile_threshold"},
        {"name": "fr_pcmtr_quantile_k", "as": "quantile_k"},
        {"name": "fr_pcmtr_quantile_alpha", "as": "quantile_alpha"},
      ],
      import_item_attr = [
        {"name": "pcmtr", "as": "pxtr"},
      ],
      export_item_attr = [
        {"name": "quantile_score", "as": "pcmtr_quantile_score"},
      ],
      function_name = "QuantileScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_awesome_wtd_quantile_threshold", "as": "quantile_threshold"},
        {"name": "fr_awesome_wtd_quantile_k", "as": "quantile_k"},
        {"name": "fr_awesome_wtd_quantile_alpha", "as": "quantile_alpha"},
      ],
      import_item_attr = [
        {"name": "awesome_wtd", "as": "pxtr"},
      ],
      export_item_attr = [
        {"name": "quantile_score", "as": "awesome_wtd_quantile_score"},
      ],
      function_name = "QuantileScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_fetr_quantile_threshold", "as": "quantile_threshold"},
        {"name": "fr_fetr_quantile_k", "as": "quantile_k"},
        {"name": "fr_fetr_quantile_alpha", "as": "quantile_alpha"},
      ],
      import_item_attr = [
        {"name": "fetr", "as": "pxtr"},
      ],
      export_item_attr = [
        {"name": "quantile_score", "as": "fetr_quantile_score"},
      ],
      function_name = "QuantileScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_fr_score1_quantile_threshold", "as": "quantile_threshold"},
        {"name": "fr_fr_score1_quantile_k", "as": "quantile_k"},
        {"name": "fr_fr_score1_quantile_alpha", "as": "quantile_alpha"},
      ],
      import_item_attr = [
        {"name": "fr_score1", "as": "pxtr"},
      ],
      export_item_attr = [
        {"name": "quantile_score", "as": "fr_score1_quantile_score"},
      ],
      function_name = "QuantileScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .calc_weighted_sum(
      fomula_version = 1,
      channels = [
        { "name": "pctr_quantile_score", "weight": "{{final_pctr_quantile_score_weight}}" },
        { "name": "pltr_quantile_score", "weight": "{{final_pltr_quantile_score_weight}}" },
        { "name": "pwtr_quantile_score", "weight": "{{final_pwtr_quantile_score_weight}}" },
        { "name": "pftr_quantile_score", "weight": "{{final_pftr_quantile_score_weight}}" },
        { "name": "pcmtr_quantile_score", "weight": "{{final_pcmtr_quantile_score_weight}}" },
        { "name": "awesome_wtd_quantile_score", "weight": "{{final_awesome_wtd_quantile_score_weight}}" },
        { "name": "fetr_quantile_score", "weight": "{{final_fetr_quantile_score_weight}}" },
        { "name": "fr_score1_quantile_score", "weight": "{{final_fr_score1_quantile_score_weight}}" },
      ],
      output_item_attr = "quantile_relative_score",
    )
    return self

  def fr_boost_loyal_fans_reason(self):
    self.enrich_attr_by_light_function(
      target_reason = [866],
      import_common_attr = [
        {"name": "fr_boost_loyal_fans_reason_weight", "as": "boost_weight"},
        {"name": "fr_weaken_loyal_fans_reason_weight", "as": "weaken_weight"},
      ],
      import_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "EnsembleScoreBoost",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def fr_marketing_compensation_discount(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_marketing_compensation_discount_ctr_weight", "as": "ctr_weight"},
        {"name": "fr_marketing_compensation_discount_watchtime_weight", "as": "watchtime_weight"},
        {"name": "fr_marketing_compensation_discount_score_base", "as": "score_base"},
        {"name": "fr_marketing_compensation_discount_score_base_ratio", "as": "score_base_ratio"},
        {"name": "fr_marketing_compensation_discount_coef", "as": "old_coeff"},
      ],
      import_item_attr = [
        {"name": "corr_pctr", "as": "ctr"},
        {"name": "awesome_wtd", "as": "watchtime"},
      ],
      export_common_attr = [
       {"name": "coeff", "as": "fr_marketing_compensation_discount_reward_coeff"}
      ],
      function_name = "CalcRewardCoeff",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_marketing_compensation_discount_scale_factor", "as": "scale_factor"},
        {"name": "fr_marketing_compensation_discount_reward_coeff", "as": "reward_coeff"},
        {"name": "fr_marketing_compensation_discount_coef", "as": "old_coeff"},
      ],
      export_common_attr = [
       {"name": "new_coeff", "as": "fr_marketing_compensation_discount_coef"}
      ],
      function_name = "MarketingCompensationPhotoAdjust",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_marketing_compensation_discount_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"}
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
          "is_marketing_compensation_photo" : 1
        }
    )
    return self

  def explore_ranking_cold_photo_boost(self):
    self.calc_by_formula1(
      kconf_key = "formula.scenarioKey85.explore_cold_combined_score",
      import_item_attr = [
        {"name": "explore_stat__real_show_count", "as": "current_impr", "default_val": 0},
        {"name": "explore_stat__click_count", "as": "current_click", "default_val": 0},
        {"name": "cold_item_quality_score", "as": "item_quality_score", "default_val": 0.0},
        {"name": "item_upload_second", "as": "created_second", "default_val": 0},
        {"name": "good_cover_similary_score", "as": "good_cover_similary_score", "default_val": 1.0},
        {"name": "is_lowvv", "as": "is_lowvv", "default_val": 0},
        {"name": "is_cold_recall", "as": "is_cold_recall", "default_val": 0},
      ],
      export_formula_value = [
        {"name": "cold_combined_score", "as": "fr_explore_cold_photo_score"}
      ],
      abtest_biz_name = "KUAISHOU_APPS",
      perf_tag = "explore_cold_combined_score",
      target_item = {
        "is_picture": 0,
        "is_same_author_tail": 1
      }
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "fr_explore_cold_photo_score", "as": "boost_discount_coeff"},
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "BoostOrDiscountWithItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "is_picture": 0,
        "is_same_author_tail": 1
      }
    )
    return self
  
  def fr_marketing_compensation_personal_discount(self):
    self \
    .calc_by_formula1(
      kconf_key = "formula.scenarioKey92.FrExploreMarketingPhotoDeboost",
      import_item_attr = [
        "explore_marketing_compensation_positive_trigger_similarity_score",
      ],
      import_common_attr = [
        "explore_marketing_compensation_positive_trigger_size",
      ],
      export_formula_value = [
        {"name": "final_score", "as": "fr_marketing_compensation_personal_discount_coef"}
      ],
      abtest_biz_name = "KUAISHOU_APPS",
      target_item = {
        "is_marketing_compensation_photo": 1
      }
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "fr_marketing_compensation_personal_discount_coef", "as": "boost_discount_coeff"},
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "BoostOrDiscountWithItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "is_marketing_compensation_photo": 1
      }
    )
    return self

  def author_circle_cluster_id_boost(self, score_attr, pxtr_names=["corr_pctr"]):
    self.enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "u2a_author_id_circle_id_detail_kuaishou_ptr", "as": "map_ptr"}
        ],
        import_item_attr = [
          {"name": "author__id", "as": "key_attr"}
        ],
        export_item_attr = [
          {"name": "target_item_attr", "as": "author_circle_cluster_id"},
        ],
        function_name = "GetItemAttrByIntToIntMapPtr",
        class_name = "ExploreLightFunctionSetV2"
      )
    for pxtr_name in pxtr_names:
      self.cluster_id_pxtr_boost(score_attr, "author_circle_cluster_id", pxtr_name, "author_circle")

    return self

  def interest_cluster_id_boost(self, score_attr, pxtr_names=["corr_pctr"]):
    for pxtr_name in pxtr_names:
      self.cluster_id_pxtr_boost(score_attr, "hetu_sim_cluster_id", pxtr_name, "interest")
    return self

  def cluster_id_pxtr_boost(self, score_attr, cluster_id_name, pxtr_name, strategy_prefix):
    switch_attr_name = "enable_" + strategy_prefix + "_cluster_id_pxtr_boost_" + pxtr_name
    alpha_attr_name = strategy_prefix + "_cluster_id_pxtr_boost_" + pxtr_name + "_alpha"
    beta_attr_name = strategy_prefix + "_cluster_id_pxtr_boost_" + pxtr_name + "_beta"
    debias_coef_attr_name = strategy_prefix + "_cluster_id_pxtr_boost_" + pxtr_name + "_debias_coef"
    default_coef_attr_name = strategy_prefix + "_cluster_id_pxtr_boost_" + pxtr_name + "_default_coef"
    boost_score_name = strategy_prefix + "_" + pxtr_name + "_boost_score"
    self \
      .if_(switch_attr_name + " == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": alpha_attr_name, "as": "alpha"},
            {"name": beta_attr_name, "as": "beta"},
            {"name": debias_coef_attr_name, "as": "debias_coef"},
            {"name": default_coef_attr_name, "as": "default_coef"}
          ],
          import_item_attr = [
            {"name": cluster_id_name, "as": "cluster_id"},
            {"name": pxtr_name, "as": "origin_score"}
          ],
          export_item_attr = [
            {"name": "debias_score", "as": boost_score_name},
          ],
          function_name = "ClusterDebiasByItemAttr",
          class_name = "ExploreLightFunctionSetV2"
        ) \
        .enrich_attr_by_light_function(
          import_item_attr = [
            {"name": boost_score_name, "as": "boost_discount_coeff"},
            {"name": score_attr, "as": "score"},
          ],
          export_item_attr = [
            {"name": "score", "as": score_attr}
          ],
          function_name = "BoostOrDiscountByItemCoeff",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()
    return self

  def boost_user_short_develop_interest(self, score_attr, stage_name="fr_s2", strategy_name="user_short_develop_interest"):
    all_item_count = "explore_" + stage_name + "_" + strategy_name + "_all_item_count"
    target_item_count = "explore_" + stage_name + "_" + strategy_name + "_target_item_count"
    boost_coef = "explore_" + stage_name + "_" + strategy_name + "_boost_coef"
    alpha = "explore_" + stage_name + "_" + strategy_name + "_alpha"
    max_ratio = "explore_" + stage_name + "_" + strategy_name + "_max_ratio"
    empirical_ctr_threshold = "explore_" + strategy_name + "_emp_ctr_threshold"
    self.count_reco_result(
      save_count_to = all_item_count,
    ) \
    .count_reco_result(
      save_count_to = target_item_count,
      target_item = {
        "is_user_short_develop_interest" : 1,
      }, 
    ) \
    .gen_common_attr_by_lua(
      attr_map={
        boost_coef: "{coef} * (1 + {alpha} * ({max_ratio} - {target_cnt} / {total_cnt} > 0.0 and {max_ratio} - {target_cnt} / {total_cnt} or 0.0))".format(coef=boost_coef, alpha=alpha, max_ratio=max_ratio, target_cnt=target_item_count, total_cnt=all_item_count)
      }   
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": boost_coef, "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      select_item = {
        "join": "and",
        "filters": [{
            "attr_name": "is_user_short_develop_interest",
            "select_if": "==",
            "compare_to": 1,
        }, {
            "attr_name": "empirical_ctr",
            "select_if": ">",
            "compare_to": "{{" + empirical_ctr_threshold + "}}",
        }]
      }
    )
    return self

  def unbias_interest_photo_boost(self, score_attr, stage_name):
    coeff_param_name = "explore_" + stage_name + "_unbias_interest_photo_boost_coeff"
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": coeff_param_name, "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "is_in_selected_cids", "as": "need_item_attr"},
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      function_name = "BoostOrDiscount",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    
    return self

  def hot_list_photo_boost(self, score_attr, stage_name):
    coeff_param_name = "explore_" + stage_name + "_hot_list_photo_boost_coeff"
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": coeff_param_name, "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "is_hot_list_flag" : 1
      }
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "hot_list_adjust_coeff", "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "is_hot_list_flag" : 1
      }
    )
    
    return self

  def short_uninterest_photo_discount(self, score_attr, stage_name):
    coeff_param_name = "explore_" + stage_name + "_short_uninterest_photo_discount_coeff"
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": coeff_param_name, "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "is_short_uninterested_photo", "as": "need_item_attr"},
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      function_name = "BoostOrDiscount",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    
    return self

  def short_uninterest_decay_discount(self, score_attr, stage_name):
    discount_coeff = "explore_" + stage_name + "_short_uninterest_decay_discount_coeff"
    discount_alpha = "explore_" + stage_name + "_short_uninterest_decay_discount_alpha"
    discount_beta = "explore_" + stage_name + "_short_uninterest_decay_discount_beta"
    discount_thres = "explore_" + stage_name + "_short_uninterest_decay_discount_thres"
    discount_hetu5 = "explore_" + stage_name + "_short_uninterest_decay_discount_hetu5"
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": discount_coeff, "as": "boost_discount_coeff"},
        {"name": discount_alpha, "as": "interest_decay_alpha"},
        {"name": discount_beta, "as": "interest_decay_beta"},
        {"name": discount_thres, "as": "interest_decay_thres"},
        {"name": discount_hetu5, "as": "interest_decay_hetu5"},
      ],
      import_item_attr = [
        {"name": "short_uninterest_hetu5_num", "as": "decay_num"},
        "hetu_tag_level_info__hetu_level_five",
      ],
      export_item_attr = [
        {"name": "interest_decay_coeff", "as": discount_coeff}
      ],
      function_name = "CalInterestDecayCoeff",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": discount_coeff, "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def interest_migration_photo_boost(self, score_attr):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "interest_migration_photo_coef", "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
  

  def fr_protogenetic_advertise_discount(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_protogenetic_advertis_discount_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"}
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
          "is_protogenetic_advertise_photo" : 1
        }
    )
    return self

  def fr_unbias_interest_cluster_boost(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_fr_s2_unbias_interest_cids_coeff", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "is_in_cluster_unbias_cids", "as": "need_item_attr"},
        {"name": "explore_fr_ensemble_score", "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "explore_fr_ensemble_score"}
      ],
      function_name = "BoostOrDiscount",
      class_name = "ExploreLightFunctionSetV2",
    ) \

    return self

  def fr_interest_generalization_boost(self, score_attr, stage_name):
    boost_coef_name = "explore_" + stage_name + "_interest_generalization_boost_coef"
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": boost_coef_name, "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "is_generalization_photo", "as": "need_item_attr"},
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      function_name = "BoostOrDiscount",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def fr_reach_content_keep_item(self, score_attr, stage_name):
    boost_coef_name = "explore_" + stage_name + "_reach_content_boost_coef"
    boost_item_limit  = "explore_" + stage_name + "_reach_content_boost_item_limit"
    self.sort(
      score_from_attr = "ctr_filter_ensemble_score",
      target_item = {"is_picture" : 0}
    ) \
    .item_attr_operation(
      item_attr_a = score_attr,
      common_attr_b = "{{" + boost_coef_name + "}}",
      operator = "*",
      output_attr = score_attr,
      select_item = {
        "join": "and",
        "filters": [{
          "attr_name": "is_picture",
          "select_if": "==",
          "compare_to": 0,
        }, {
          "attr_name": "reach_content",
          "select_if": "==",
          "compare_to": 1,
        }],
        "limit": "{{" + boost_item_limit + "}}"
      }
    )
    return self
  
  def fr_cal_interest_cid_coeff(self, interest_and_score_list_name="user_develop_interest_cid_and_score_list"):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "uOldMmuClusterId300ListList",
        {"name": interest_and_score_list_name, "as": "interest_and_score_list"},
        {"name": "fr_s2_interest_cluster_id_num_threshold", "as": "interest_cluster_id_num_threshold"},
        {"name": "fr_s2_identified_interest_cluster_id_num_threshold", "as": "identified_interest_cluster_id_num_threshold"},
        {"name": "fr_s2_unidentified_interest_cluster_id_num_threshold", "as": "unidentified_interest_cluster_id_num_threshold"},
        {"name": "fr_s2_interest_score_cids_ori_boost_coeff", "as": "interest_score_cids_ori_boost_coeff"},
        {"name": "fr_s2_identified_interest_boost_alpha_coeff", "as": "identified_interest_boost_alpha_coeff"},
        {"name": "fr_s2_identified_interest_boost_beta_coeff", "as": "identified_interest_boost_beta_coeff"},
        {"name": "fr_s2_identified_interest_boost_omega_coeff", "as": "identified_interest_boost_omega_coeff"},
        {"name": "fr_s2_unidentified_interest_boost_alpha_coeff", "as": "unidentified_interest_boost_alpha_coeff"},
        {"name": "fr_s2_unidentified_interest_boost_beta_coeff", "as": "unidentified_interest_boost_beta_coeff"},
        {"name": "fr_s2_unidentified_interest_boost_omega_coeff", "as": "unidentified_interest_boost_omega_coeff"},
        {"name": "fr_s2_develop_interest_score_lower_bound", "as": "develop_interest_score_lower_bound"},
        {"name": "fr_s2_develop_interest_score_upper_bound", "as": "develop_interest_score_upper_bound"},
        {"name": "fr_s2_identified_interest_score_lower_bound", "as": "identified_interest_score_lower_bound"},
        {"name": "fr_s2_unidentified_interest_score_lower_bound", "as": "unidentified_interest_score_lower_bound"},
        {"name": "enable_fr_s2_interest_score_boost", "as": "enable_interest_score_cids_boost"},
        {"name": "enable_fr_s2_identified_interest_score_boost", "as": "enable_identified_interest_score_cids_boost"},
        {"name": "enable_fr_s2_unidentified_interest_score_boost", "as": "enable_unidentified_interest_score_cids_boost"},
      ],
      import_item_attr = [
        {"name": "interest_cluster_id", "as": "hetu_sim_cluster_id862"},
      ],
      export_item_attr = [
        {"name": "interest_cids_coeff", "as": "ranking_interest_cids_coeff"},
      ],
      function_name = "CalInterestCidsCoeff",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    
    return self

  def fr_interest_score_cids_boost(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "ranking_interest_cids_coeff", "as": "boost_discount_coeff"},
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    ) \

    return self

  def fr_cal_valid_interest_cid_coeff(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "uOldMmuClusterId300ListList",
        "user_valid_interest_cid_and_score_list",
        {"name": "fr_s2_valid_interest_cluster_id_num_threshold", "as": "valid_interest_cluster_id_num_threshold"},
        {"name": "fr_s2_valid_interest_user_boost_alpha_coeff", "as": "valid_interest_user_boost_alpha_coeff"},
        {"name": "fr_s2_valid_interest_user_boost_beta_coeff", "as": "valid_interest_user_boost_beta_coeff"},
        {"name": "fr_s2_valid_interest_user_boost_omega_coeff", "as": "valid_interest_user_boost_omega_coeff"},
        {"name": "fr_s2_develop_valid_interest_score_lower_bound", "as": "develop_valid_interest_score_lower_bound"},
        {"name": "enable_fr_s2_valid_interest_score_boost", "as": "enable_valid_interest_score_cids_boost"},
      ],
      import_item_attr = [
        {"name": "interest_cluster_id", "as": "hetu_sim_cluster_id862"},
      ],
      export_item_attr = [
        {"name": "valid_interest_cids_coeff", "as": "ranking_valid_interest_cids_coeff"},
      ],
      function_name = "CalValidInterestCidsCoeff",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    
    return self

  def fr_valid_interest_score_cids_boost(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "ranking_valid_interest_cids_coeff", "as": "boost_discount_coeff"},
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    ) \

    return self

  def fr_cal_short_valid_interest_first_refresh_coeff(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "uOldMmuClusterId300ListList",
        {"name": "uExploreShortValidInterestAndScoreList", "as": "user_valid_interest_cid_and_score_list"},
        {"name": "fr_s2_short_valid_interest_first_refresh_cluster_id_num_threshold", "as": "valid_interest_cluster_id_num_threshold"},
        {"name": "fr_s2_short_valid_interest_first_refresh_user_boost_alpha_coeff", "as": "valid_interest_user_boost_alpha_coeff"},
        {"name": "fr_s2_short_valid_interest_first_refresh_user_boost_beta_coeff", "as": "valid_interest_user_boost_beta_coeff"},
        {"name": "fr_s2_short_valid_interest_first_refresh_user_boost_omega_coeff", "as": "valid_interest_user_boost_omega_coeff"},
        {"name": "fr_s2_short_valid_interest_first_refresh_develop_score_lower_bound", "as": "develop_valid_interest_score_lower_bound"},
        {"name": "enable_fr_s2_short_valid_interest_first_refresh_score_boost", "as": "enable_valid_interest_score_cids_boost"},
      ],
      import_item_attr = [
        {"name": "interest_cluster_id", "as": "hetu_sim_cluster_id862"},
      ],
      export_item_attr = [
        {"name": "valid_interest_cids_coeff", "as": "ranking_short_valid_interest_first_refresh_coeff"},
      ],
      function_name = "CalValidInterestCidsCoeff",
      class_name = "ExploreLightFunctionSetV2",
    ) \

    return self

  def fr_short_valid_interest_first_refresh_boost(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "ranking_short_valid_interest_first_refresh_coeff", "as": "boost_discount_coeff"},
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    ) \

    return self

  def fr_cal_no_ctr_score(self):
    self.item_attr_operation(
      item_attr_a = "corr_pctr",
      common_attr_b = 1.0,
      operator = "-",
      output_attr = "raw_score_no_ctr"
    ) \
    .item_attr_operation(
      item_attr_a = "psvr",
      item_attr_b = "corr_pctr",
      operator = "*",
      output_attr = "raw_score_svr"
    ) \
    .calc_weighted_sum(
      channels = [
        {"name": "raw_score_no_ctr", "weight": "{{explore_fr_cal_no_ctr_or_svr_score_no_ctr_weight}}"},
        {"name": "raw_score_svr", "weight": "{{explore_fr_cal_no_ctr_or_svtr_score_svr_weight}}"},
        {"name": "bad_cover_similary_score", "weight": "{{explore_fr_cal_no_ctr_or_svtr_bad_cover_weight}}"},
      ],
      output_item_attr = "score_no_ctr",
    )
    return self

  def fr_cal_ctr_adjust_by_pcoc_score(self):
    self.str_format(
      format_string = "%s_%d_%d",
      input_attrs = ["explore_fr_pctr_adjust_by_pcoc_user_group_interest_pcoc_prefix", "basic_info_age_segment_v2", "basic_info_gender_v2"],
      output_attr = "user_group_interest_pcoc_key",
    ) \
    .get_kconf_params(
      kconf_configs = [{ 
        "kconf_key": "reco.offline.userGroupInterestPcocStat",
        "json_path": "{{user_group_interest_pcoc_key}}",
        "value_type": "list_double",
        "default_value": [],
        "export_common_attr": "explore_user_group_interest_pcoc_list"
      }]   
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_user_group_interest_pcoc_list", "as": "value_list"},
      ],   
      import_item_attr = [
        {"name" : "hetu_sim_cluster_id", "as" : "item_key_attr"}
      ],   
      export_item_attr = [
        {"name": "target_item_attr", "as": "user_group_interest_pcoc"}
      ],   
      function_name = "AddItemAttrByCommonList",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .item_attr_operation(
      item_attr_a = "corr_pctr",
      item_attr_b = "user_group_interest_pcoc",
      operator = "/",
      output_attr = "corr_pctr_adjust_by_pcoc"
    )
    return self

  def fr_cal_svtr_rid_ctr_score(self):
    self.calc_weighted_sum(
      channels = [
        {"name": "psvr", "weight": "{{explore_fr_cal_svtr_rid_ctr_score_svtr_weight}}"},
        {"name": "bad_cover_similary_score", "weight": "{{explore_fr_cal_svtr_rid_ctr_score_bad_cover_weight}}"},
      ],
      output_item_attr = "linear_score_of_svr_bad_cover",
    ) \
    .item_attr_operation(
      item_attr_a = "linear_score_of_svr_bad_cover",
      common_attr_b = "{{explore_fr_svtr_shift_coef}}",
      operator = "+",
      output_attr = "fr_shift_svtr"
    ) \
    .item_attr_operation(
      item_attr_a = "corr_pctr",
      common_attr_b = "{{explore_fr_ctr_shift_coef}}",
      operator = "+",
      output_attr = "fr_shift_ctr"
    ) \
    .item_attr_operation(
      item_attr_a = "fr_shift_svtr",
      item_attr_b = "fr_shift_ctr",
      operator = "/",
      output_attr = "svtr_rid_ctr_score"
    )
    return self

  def request_pxtr_weight_adjust(self):
    self.if_("explore_enable_request_avg_top == 1") \
      .sort(
        score_from_attr = "corr_pctr",
      ) \
      .pack_item_attr(
        item_source={
          "reco_results": True,
        },
        mappings=[
          {
            "aggregator": "avg",
            "from_item_attr": "corr_pctr",
            "to_common_attr": "pctr_request_avg",
            "item_attr_limit": "{{explore_request_avg_top_size}}"
          },
        ],
      ) \
    .else_() \
      .pack_item_attr(
        item_source = {
          "reco_results": True,
        },
        mappings = [
          {
            "aggregator": "avg",
            "from_item_attr": "corr_pctr",
            "to_common_attr": "pctr_request_avg"
          },
        ],
      ) \
    .end_() \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_ensemble_power_weight_fullrank_pctr_score", "as": "weight"},
        {"name": "pctr_request_avg", "as": "pxtr_avg"},
        {"name": "user_emp_ctr", "as": "user_emp_xtr"},
        {"name": "explore_request_pctr_weight_adjust_lower", "as": "lower"},
        {"name": "explore_request_pctr_weight_adjust_upper", "as": "upper"},
        {"name": "explore_request_pctr_weight_adjust_power_weight", "as": "power_weight"},
        {"name": "explore_request_pctr_weight_adjust_bias", "as": "bias"},
      ],
      export_common_attr = [
        {"name": "weight", "as": "explore_ensemble_power_weight_fullrank_pctr_score"}
      ],
      function_name = "RequestPxtrWeightAdjust",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def user_vv_type_weight_adjust(self, weight_name):
    adjust_coef = weight_name + "_vv_type_adjust_coef"
    self \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": adjust_coef, "as": "weight"},
          {"name": weight_name, "as": "value"}
        ],
        export_common_attr = [
          {"name": "new_value", "as": weight_name}
        ],
        function_name = "CalExploreDoubleMultiDouble",
        class_name = "ExploreLightFunctionSetV2",
      )
    return self

  def user_cocoon_weight_adjust(self, weight_name):
    adjust_coef = weight_name + "_cocoon_adjust_coef"
    self \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": adjust_coef, "as": "weight"},
          {"name": weight_name, "as": "value"}
        ],
        export_common_attr = [
          {"name": "new_value", "as": weight_name}
        ],
        function_name = "CalExploreDoubleMultiDouble",
        class_name = "ExploreLightFunctionSetV2",
      )
    return self

  def int_value_adjust(self, int_value_name, strategy_name):
    adjust_coef = int_value_name + "_" + strategy_name + "_adjust_coef"
    self \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": adjust_coef, "as": "weight"},
          {"name": int_value_name, "as": "value"}
        ],
        export_common_attr = [
          {"name": "new_value", "as": int_value_name}
        ],
        function_name = "CalExploreIntMultiDouble",
        class_name = "ExploreLightFunctionSetV2",
      )
    return self

  def multi_int_value_adjust(self, int_value_name_list, strategy_name):
    for int_value_name in int_value_name_list:
      self.int_value_adjust(int_value_name, strategy_name)
    return self

  def user_vv_weight_adjust(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "emp_htr_score_in_order_weight", "as": "xtr_weight"},
        {"name": "active_days_avg_vv", "as": "user_vv"},
        {"name": "explore_fr_hate_like_weight_adjust_exp_upper", "as": "exp_upper"},
        {"name": "explore_fr_hate_like_weight_adjust_alpha", "as": "alpha"},
        {"name": "explore_fr_hate_like_weight_adjust_beta", "as": "beta"},
        {"name": "explore_fr_hate_like_weight_adjust_omega", "as": "omega"},
        {"name": "explore_fr_hate_like_weight_adjust_max", "as": "coeff_max"},
        {"name": "explore_fr_hate_like_weight_adjust_min", "as": "coeff_min"},
      ],
      export_common_attr = [
        {"name": "xtr_weight", "as": "emp_htr_score_in_order_weight"},
      ],
      function_name = "AdjustWeightByUserVv",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def user_vv_ensemble_power_weight_adjust(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_ensemble_power_weight_fullrank_pctr_score", "as": "xtr_weight"},
        {"name": "explore_recent_valid_click_count", "as": "user_vv"},
        {"name": "explore_fr_ensemble_power_weight_adjust_exp_upper", "as": "exp_upper"},
        {"name": "explore_fr_ensemble_power_weight_adjust_alpha", "as": "alpha"},
        {"name": "explore_fr_ensemble_power_weight_adjust_beta", "as": "beta"},
        {"name": "explore_fr_ensemble_power_weight_adjust_omega", "as": "omega"},
        {"name": "explore_fr_ensemble_power_weight_adjust_max", "as": "coeff_max"},
        {"name": "explore_fr_ensemble_power_weight_adjust_min", "as": "coeff_min"},
      ],
      export_common_attr = [
        {"name": "xtr_weight", "as": "explore_ensemble_power_weight_fullrank_pctr_score"},
      ],
      function_name = "AdjustWeightByUserVv",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def fr_cal_debias_xtr_by_pcoc_score(self, prefix, raw_xtr, debias_xtr_by_pcoc):
    self.str_format(
      format_string = f"{prefix}_%d",
      input_attrs = ["basic_info_age_segment_v2"],
      output_attr = "user_debias_xtr_pcoc_key",
    ) \
    .get_kconf_params(
      kconf_configs = [{
        "kconf_key": "reco.offline.userHetuInteractPcocStat",
        "json_path": "{{user_debias_xtr_pcoc_key}}",
        "value_type": "list_double",
        "default_value": [],
        "export_common_attr": "explore_debias_xtr_by_pcoc_list"
      }]
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_debias_xtr_by_pcoc_list", "as": "value_list"},
      ],   
      import_item_attr = [
        {"name": "hetu_level_one_top1", "as" : "item_key_attr"}
      ],
      export_item_attr = [
        {"name": "target_item_attr", "as": "user_debias_xtr_by_pcoc"}
      ],
      function_name = "AddItemAttrByCommonList",
      class_name = "ExploreLightFunctionSetV2"
    ) \
    .set_attr_default_value(
      item_attrs=[{
        "name": "user_debias_xtr_by_pcoc",
        "type": "double", 
        "value": 1.0
      }]
    ) \
    .item_attr_operation(
      item_attr_a = raw_xtr,
      item_attr_b = "user_debias_xtr_by_pcoc",
      operator = "/",
      output_attr = debias_xtr_by_pcoc
    )
    return self

  def fr_category_boost(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_category_boost_coef_map", "as": "category_boost_coeff_map"},
      ],
      import_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": "score"},
        "hetu_tag_level_info__hetu_level_one"
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"}
      ],
      function_name = "BoostOrDiscountWithCategory",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def user_group_consume_weight_adjust(self, weight_param_dict, stage_name):
    input_common_attrs = [
      {"name": "explore_all_user_consume_str", "as": "all_user_consume_stat_str"},
      {"name": "explore_user_group_consume_str", "as": "user_consume_stat_str"}
    ]
    output_common_attrs = []
    ratio_prefix = "explore_user_group_consume_weight_adjust_ratio_" + stage_name + "_"
    for xtr in weight_param_dict.keys():
      input_common_attrs.append({"name": weight_param_dict[xtr], "as": xtr + "_weight"})
      input_common_attrs.append({"name": ratio_prefix + xtr, "as": xtr + "_adjust_ratio"})
      output_common_attrs.append({"name": xtr + "_weight", "as": weight_param_dict[xtr]})
    
    self.enrich_attr_by_light_function(
      import_common_attr = input_common_attrs,
      export_common_attr = output_common_attrs,
      function_name = "UserGroupWeightAdjustCoef",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def fr_boost_topk_hot_list_photo(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_hot_list_exptag_reason_topk", "as": "fr_quality_signal_exptag_reason_topk"},
        {"name": "hot_list_exptag_reason_weight", "as": "quality_signal_exptag_reason_weight"}
      ],
      import_item_attr = [
        "explore_fr_ensemble_score"
      ],
      export_item_attr = [
        "explore_fr_ensemble_score",
        {"name": "is_quality_singal_topk", "as": "is_quality_singal_hot_list_topk"}
      ],
      function_name = "BoostTopkQualitySignalPhoto",
      class_name = "ExploreLightFunctionSetV2",
      target_reason = 10042,
    )
    return self

  def fr_boost_topk_prior_author_photo(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_prior_author_exptag_reason_topk", "as": "fr_quality_signal_exptag_reason_topk"},
        {"name": "prior_author_exptag_reason_weight", "as": "quality_signal_exptag_reason_weight"}
      ],
      import_item_attr = [
        "explore_fr_ensemble_score"
      ],
      export_item_attr = [
        "explore_fr_ensemble_score",
        {"name": "is_quality_singal_topk", "as": "is_quality_singal_prior_author_topk"}
      ],
      function_name = "BoostTopkQualitySignalPhoto",
      class_name = "ExploreLightFunctionSetV2",
      target_reason = 10030,
    )
    return self

  def fr_boost_topk_life_prior_photo(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_life_prior_exptag_reason_topk", "as": "fr_quality_signal_exptag_reason_topk"},
        {"name": "life_prior_exptag_reason_weight", "as": "quality_signal_exptag_reason_weight"}
      ],
      import_item_attr = [
        "explore_fr_ensemble_score"
      ],
      export_item_attr = [
        "explore_fr_ensemble_score",
        {"name": "is_quality_singal_topk", "as": "is_quality_singal_life_prior_topk"}
      ],
      function_name = "BoostTopkQualitySignalPhoto",
      class_name = "ExploreLightFunctionSetV2",
      target_reason = 10049,
    )
    return self

  def fr_boost_topk_original_author_photo(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_original_author_exptag_reason_topk", "as": "fr_quality_signal_exptag_reason_topk"},
        {"name": "original_author_exptag_reason_weight", "as": "quality_signal_exptag_reason_weight"}
      ],
      import_item_attr = [
        "explore_fr_ensemble_score"
      ],
      export_item_attr = [
        "explore_fr_ensemble_score",
        {"name": "is_quality_singal_topk", "as": "is_quality_original_author_list_topk"}
      ],
      function_name = "BoostTopkQualitySignalPhoto",
      class_name = "ExploreLightFunctionSetV2",
      target_reason = 10039,
    )
    return self

  def fr_deboost_over_distribute_photo(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_ensemble_score_over_distribute_alpha", "as": "ensemble_score_over_distribute_alpha"},
        {"name": "fr_ensemble_score_over_distribute_beta", "as": "ensemble_score_over_distribute_beta"},
        {"name": "fr_explore_show_limit_threshold", "as": "explore_show_limit_threshold"},
        {"name": "fr_explore_ctr_limit_threshold", "as": "explore_ctr_limit_threshold"},
        {"name": "fr_percent_threshold", "as": "percent_threshold"}
      ],
      import_item_attr = [
        "explore_stat__real_show_count",
        "thanos_stats__real_show_count",
        "explore_stat__click_count",
        "explore_fr_ensemble_score",
      ],
      export_item_attr = [
        "explore_fr_ensemble_score",
      ],
      function_name = "DeboostOverDistributePhoto",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def fr_boost_long_worth_author_photo(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_boost_long_worth_author_reason_topk", "as": "fr_quality_signal_exptag_reason_topk"},
        {"name": "fr_boost_long_worth_author_reason_weight", "as": "quality_signal_exptag_reason_weight"}
      ],
      import_item_attr = [
        "explore_fr_ensemble_score",
      ],
      export_item_attr = [
        "explore_fr_ensemble_score",
        {"name": "is_quality_singal_topk", "as": "is_pid_for_long_worth_author"}
      ],
      function_name = "BoostTopkQualitySignalPhoto",
      class_name = "ExploreLightFunctionSetV2",
      target_reason = 820,
    )
    return self

  def fr_boost_useful_author(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_useful_author_boost_coeff", "as": "boost_discount_coeff"},
        {"name": "explore_useful_author_boost_thres", "as": "boost_discount_thres"},
      ],
      import_item_attr = [
        {"name": "userfulness_author_score", "as": "need_item_attr"},
        {"name": "explore_fr_ensemble_score", "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "BoostOrDiscountWithThres",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def fr_boost_top_and_deboost_reciprocal_like_action(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_boost_topk_like_photos_coeff", "as": "boost_topk_coeff"},
        {"name": "explore_boost_topk_like_photos_thres", "as": "boost_topk_thres"},
        {"name": "explore_deboost_topk_like_photos_coeff", "as": "deboost_topk_coeff"},
        {"name": "explore_deboost_topk_like_photos_thres", "as": "deboost_topk_thres"},
      ],
      import_item_attr = [
        {"name": "score_pltr", "as": "score_xtr"},
        {"name": "explore_fr_ensemble_score", "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "BoostOrDiscountTopkActionPhoto",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def fr_boost_top_and_deboost_reciprocal_follow_action(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_boost_topk_follow_photos_coeff", "as": "boost_topk_coeff"},
        {"name": "explore_boost_topk_follow_photos_thres", "as": "boost_topk_thres"},
        {"name": "explore_deboost_topk_follow_photos_coeff", "as": "deboost_topk_coeff"},
        {"name": "explore_deboost_topk_follow_photos_thres", "as": "deboost_topk_thres"},
      ],
      import_item_attr = [
        {"name": "score_pwtr", "as": "score_xtr"},
        {"name": "explore_fr_ensemble_score", "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "BoostOrDiscountTopkActionPhoto",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
  
  def explore_ranking_low_time_active_weight_adjust(self):
    self.calc_by_formula1(
      kconf_key = "formula.scenarioKey94.explore_ranking_low_time_active_weight_adjust_f1",
      import_common_attr = [
        "explore_ensemble_power_weight_fullrank_pctr_score",
        "explore_ensemble_power_weight_fullrank_pltr_score",
        "explore_ensemble_power_weight_fullrank_pwtr_score",
        "explore_ensemble_power_weight_fullrank_pftr_score",
        "fr_pmctr_rank_weight",
        "awesome_wtd_weight_push",
        "active_days_high_time_rate"
      ],
      export_formula_value = [
        {"name": "explore_ensemble_power_weight_fullrank_pctr_score", "to_common": True},
        {"name": "explore_ensemble_power_weight_fullrank_pltr_score", "to_common": True},
        {"name": "explore_ensemble_power_weight_fullrank_pwtr_score", "to_common": True},
        {"name": "explore_ensemble_power_weight_fullrank_pftr_score", "to_common": True},
        {"name": "fr_pmctr_rank_weight", "to_common": True},
        {"name": "awesome_wtd_weight_push", "to_common": True}
      ],
      abtest_biz_name = "KUAISHOU_APPS"
    )
    return self
    
  def rank_update_bar_boost(self):
    self.split_string(
      input_common_attr = "explore_rank_update_bar_proportion_str",
      output_common_attr = "explore_rank_update_bar_proportion_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_rank_update_bar_score_weight_str",
      output_common_attr = "explore_rank_update_bar_score_weight_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_rank_update_bar_proportion_list", "as": "update_bar_proportion_list"},
        {"name": "explore_rank_update_bar_score_weight_list", "as": "update_bar_score_weight_list"},
        {"name": "explore_rank_update_bar_audit_limit", "as": "update_bar_audit_limit"},
      ],
      import_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": "score"},
        {"name": "content_safety_level_with_namespace__level_hot_online", "as": "level_hot_online"},
        "upload_time",
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"}
      ],
      function_name = "BoostUpdateTimeBar",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def fr_short_window_ctr_cali(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "corr_pctr", "as": "score"},
        {"name": "short_window_ctr_cali_coeff", "as": "boost_discount_coeff"},
      ],
      export_item_attr = [
        {"name": "score", "as": "corr_pctr"}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def llm_negative_photo_adjust(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_rank_llm_negative_photo_adjust_tag_coeff_map_str", "as": "tag_coeff_map_str"},
      ],
      import_item_attr = [
        "hetu_tag_level_info_v2__hetu_tag",
        "explore_stat__click_count",
        "explore_stat__report_count",
        "fountain_stats__real_show_count",
        "fountain_stats__report_count",
        {"name": "explore_fr_ensemble_score", "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "explore_fr_ensemble_score"}
      ],
      function_name = "LlmNegativePhotoAdjust",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def fr_llm_negative_photo_personal_adjust(self):
    self.calc_by_formula1(
      kconf_key = "formula.scenarioKey05.FrExploreLlmNeagtivePhotoDeboost",
      import_item_attr = [
        "hetu_info_for_llm_negative",
        "explore_stat__click_count",
        "explore_stat__report_count",
        "fountain_stats__real_show_count",
        "fountain_stats__report_count",
      ],
      import_common_attr = [
        "uToleranceScoreKV"
      ],
      export_formula_value = [
        {"name": "final_score", "as": "final_explore_fr_llm_personal_score"}
      ],
      abtest_biz_name = "KUAISHOU_APPS"
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": "score"},
        {"name": "final_explore_fr_llm_personal_score", "as": "boost_discount_coeff"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2"
    )
    return self

  def explore_cal_update_xtr_score_rank(self):
    self.split_string(
      input_common_attr = "explore_update_fix_xtr_weight_rank_str",
      output_common_attr = "explore_update_fix_xtr_weight_rank_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_update_fix_xtr_power_rank_str",
      output_common_attr = "explore_update_fix_xtr_power_rank_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_update_fix_xtr_buttom_rank_str",
      output_common_attr = "explore_update_fix_xtr_buttom_rank_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_update_fix_xtr_upper_rank_str",
      output_common_attr = "explore_update_fix_xtr_upper_rank_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .set_attr_value( 
      no_overwrite=True,
      common_attrs=[
        {
          "name": "explore_rank_update_xtr_name_list",
          "type": "string_list",
          "value": self.update_fix_xtr_name()
        }
      ]
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_update_fix_xtr_weight_rank_list", "as": "update_fix_xtr_weight_list"},
        {"name": "explore_update_fix_xtr_power_rank_list", "as": "update_fix_xtr_power_list"},
        {"name": "explore_update_fix_xtr_buttom_rank_list", "as": "update_fix_xtr_buttom_list"},
        {"name": "explore_update_fix_xtr_upper_rank_list", "as": "update_fix_xtr_upper_list"},
        {"name": "explore_update_window_width_rank", "as": "window_width"},
        {"name": "explore_rank_window_duration_ratio", "as": "window_duration_ratio"},
        {"name": "explore_rank_update_xtr_name_list", "as": "fix_xtr_list"},
      ],
      import_item_attr = [
        "upload_time",
        "duration_ms",
        "corr_pctr",
        "pltr",
        "pwtr",
        "pcmtr",
        "pcltr",
        "fr_score2",
        "awesome_wtd",
        "fetr",
        "fr_score1",
        "pftr",
        "pctr"
      ],
      export_item_attr = [
        {"name": "update_bar_score", "as": "fr_update_xtr_fix_score"}
      ],
      function_name = "FixWindowXtr",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def update_fix_xtr_name(self):
    update_fix_xtrs = [
      "corr_pctr",
      "pltr",
      "pwtr",
      "pcmtr",
      "pcltr",
      "fr_score2",
      "awesome_wtd",
      "fetr",
      "fr_score1",
      "pftr",
      "pctr"
    ]
    return update_fix_xtrs
    
  def fr_cal_diversity_distribution(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        "hetu_level_one_top1",
      ],
      export_item_attr = [
        {"name": "hetu_level_one_ratio", "as": "explore_rank_hetu_level_one_ratio"},
      ],
      function_name = "CalHetuOneRatio",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def diversity_distribution_adjust(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_rank_diversity_distribution_boost_coeff", "as": "boost_coeff"},
        {"name": "explore_rank_diversity_distribution_discount_coeff", "as": "discount_coeff"},
      ],
      import_item_attr = [
        {"name": "explore_prerank_hetu_level_one_ratio", "as": "score_stage1"},
        {"name": "explore_rank_hetu_level_one_ratio", "as": "score_stage2"},
        {"name": "explore_fr_ensemble_score", "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "explore_fr_ensemble_score"}
      ],
      function_name = "BoostOrDiscountWithScoreStage",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def user_active_days_ensemble_power_weight_adjust(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_positive_action_sim_score_in_order_weight", "as": "xtr_weight"},
        {"name": "uExploreActiveDays", "as": "user_vv"},
        {"name": "explore_fr_ensemble_power_active_days_weight_adjust_exp_upper", "as": "exp_upper"},
        {"name": "explore_fr_ensemble_power_active_days_weight_adjust_alpha", "as": "alpha"},
        {"name": "explore_fr_ensemble_power_active_days_weight_adjust_beta", "as": "beta"},
        {"name": "explore_fr_ensemble_power_active_days_weight_adjust_omega", "as": "omega"},
        {"name": "explore_fr_ensemble_power_active_days_weight_adjust_max", "as": "coeff_max"},
        {"name": "explore_fr_ensemble_power_active_days_weight_adjust_min", "as": "coeff_min"},
      ],
      export_common_attr = [
        {"name": "xtr_weight", "as": "explore_positive_action_sim_score_in_order_weight"},
      ],
      function_name = "AdjustWeightByUserVv",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self  

  def sort_ratio_and_cal_slide_pxtr(self, strategies):
    for strategy_prefix, sort_base_name, numerator_name, denominator_name, type_name, pxtr_name in strategies:
      mul_switch_attr_name = "explore_enable_" + strategy_prefix + "_mul_universal_score"
      type_attr_name = "explore_fr_sorted_" + type_name + "_type"
      win_size_attr_name = "explore_" + strategy_prefix + "_window_size"
      numerator_base_name = "explore_fr_" + sort_base_name + "_numerator_base"
      denominator_base_name = "explore_fr_" + sort_base_name + "_denominator_base"
      slide_score_name = strategy_prefix + "_score"
      self \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": mul_switch_attr_name, "as": "enable_mul_universal_score"},
            {"name": type_attr_name, "as": "input_attr_type"},
            {"name": win_size_attr_name, "as": "window_size"},
            {"name": numerator_base_name, "as": "pc_base"},
            {"name": denominator_base_name, "as": "rc_base"},
          ],
          import_item_attr = [
            {"name": denominator_name, "as": "rc"},
            {"name": numerator_name, "as": "pc"},
            {"name": pxtr_name, "as": "origin_pxtr"},
          ],
          export_item_attr = [
            {"name": "slide_pxtr_score", "as": slide_score_name}
          ],
          function_name = "CalcSlidePxtrScore",
          class_name = "ExploreLightFunctionSetV2",
        )
    return self

  def sort_individual_score_and_cal_slide_pxtr(self, strategies):
    for strategy_prefix, sort_base_name, score_name, type_name, pxtr_name in strategies:
      mul_switch_attr_name = "explore_enable_" + strategy_prefix + "_mul_universal_score"
      type_attr_name = "explore_fr_sorted_" + type_name + "_type"
      win_size_attr_name = "explore_" + strategy_prefix + "_window_size"
      score_base_name = "explore_fr_" + sort_base_name + "_base"
      slide_score_name = strategy_prefix + "_score"
      self \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": mul_switch_attr_name, "as": "enable_mul_universal_score"},
            {"name": type_attr_name, "as": "input_attr_type"},
            {"name": win_size_attr_name, "as": "window_size"},
            {"name": score_base_name, "as": "pc_base"},
          ],
          import_item_attr = [
            {"name": score_name, "as": "pc"},
            {"name": pxtr_name, "as": "origin_pxtr"},
          ],
          export_item_attr = [
            {"name": "slide_pxtr_score", "as": slide_score_name}
          ],
          function_name = "CalcSlidePxtrScore",
          class_name = "ExploreLightFunctionSetV2",
        )
    return self

  def fr_boost_authority_author(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_fr_authority_author_boost_coeff", "as": "boost_discount_coeff"},
        {"name": "explore_fr_authority_author_boost_thres", "as": "boost_discount_thres"},
      ],
      import_item_attr = [
        {"name": "authority_author_score", "as": "need_item_attr"},
        {"name": "explore_fr_ensemble_score", "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "BoostOrDiscountWithThres",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
  
  def fr_boost_expertise_author(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_fr_expertise_author_boost_coeff", "as": "boost_discount_coeff"},
        {"name": "explore_fr_expertise_author_boost_thres", "as": "boost_discount_thres"},
      ],
      import_item_attr = [
        {"name": "expertise_author_score", "as": "need_item_attr"},
        {"name": "explore_fr_ensemble_score", "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "BoostOrDiscountWithThres",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
  
  def fr_boost_original_submission_author(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_fr_original_submission_author_boost_coeff", "as": "boost_discount_coeff"}
      ],
      import_item_attr = [
        {"name": "original_submission_author_tag", "as": "need_item_attr"},
        {"name": "explore_fr_ensemble_score", "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "BoostOrDiscount",
      class_name = "ExploreLightFunctionSetV2",
    ) 
    return self
  
  def fr_boost_personalization_author(self):
    self.if_("enable_explore_fr_personalization_author_individual_boost == 0") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_fr_personalization_author_boost_coeff", "as": "boost_discount_coeff"}
        ],
        import_item_attr = [
          {"name": "personalization_author_tag", "as": "need_item_attr"},
          {"name": "explore_fr_ensemble_score", "as": "ensemble_score"},
        ],
        export_item_attr = [
          {"name": "ensemble_score", "as": "explore_fr_ensemble_score"},
        ],
        function_name = "BoostOrDiscount",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .else_() \
      .split_string(
        input_common_attr = "explore_fr_personalization_author_individual_boost_pcmef_attr_str",
        output_common_attr = "explore_fr_personalization_author_individual_boost_pcmef_attr_list",
        delimiters = ",",
        parse_to_double = True
      ) \
      .split_string(
        input_common_attr = "explore_fr_personalization_author_individual_boost_pwtr_attr_str",
        output_common_attr = "explore_fr_personalization_author_individual_boost_pwtr_attr_list",
        delimiters = ",",
        parse_to_double = True
      ) \
      .split_string(
        input_common_attr = "explore_fr_personalization_author_individual_boost_plsst_attr_str",
        output_common_attr = "explore_fr_personalization_author_individual_boost_plsst_attr_list",
        delimiters = ",",
        parse_to_double = True
      ) \
      .split_string(
        input_common_attr = "explore_fr_personalization_author_individual_boost_pvtr_attr_str",
        output_common_attr = "explore_fr_personalization_author_individual_boost_pvtr_attr_list",
        delimiters = ",",
        parse_to_double = True
      ) \
      .split_string(
        input_common_attr = "explore_fr_personalization_author_individual_boost_pswpst_attr_str",
        output_common_attr = "explore_fr_personalization_author_individual_boost_pswpst_attr_list",
        delimiters = ",",
        parse_to_double = True
      ) \
      .split_string(
        input_common_attr = "explore_fr_personalization_author_individual_boost_cid_valid_interest_score_attr_str",
        output_common_attr = "explore_fr_personalization_author_individual_boost_cid_valid_interest_score_attr_list",
        delimiters = ",",
        parse_to_double = True
      ) \
      .split_string(
        input_common_attr = "explore_fr_personalization_author_individual_boost_pltr_attr_str",
        output_common_attr = "explore_fr_personalization_author_individual_boost_pltr_attr_list",
        delimiters = ",",
        parse_to_double = True
      ) \
      .split_string(
        input_common_attr = "explore_fr_personalization_author_individual_boost_pcltr_attr_str",
        output_common_attr = "explore_fr_personalization_author_individual_boost_pcltr_attr_list",
        delimiters = ",",
        parse_to_double = True
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          {"name": "hetu_sim_cluster_id", "as": "attr"},
        ],
        import_common_attr = [
          {"name": "uOldMmuClusterId300ListList", "as": "attr_list"},
        ],
        export_item_attr = [
           {"name": "is_in_set", "as": "is_cluster_id_in_user_interest_set"},
        ],
        function_name = "AttrIsInSet",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          {"name": "hetu_sim_cluster_id", "as": "hetu_sim_cluster_id862"},
          "is_cluster_id_in_user_interest_set"
        ],
        import_common_attr = [
          {"name": "uInterestAndScoreList", "as": "interest_score_list"},
        ],
        export_item_attr = [
          "cid_valid_interest_score",
        ],
        function_name = "GetClusterInterestScore",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_fr_personalization_author_boost_skip_full_and_high_users", "as": "skip_full_and_high_users"},
          {"name": "explore_fr_personalization_author_boost_max_score", "as": "max_score"},
          {"name": "explore_fr_personalization_author_individual_boost_pcmef_attr_list", "as": "pcmef_attr_list"},
          {"name": "explore_fr_personalization_author_individual_boost_pwtr_attr_list", "as": "pwtr_attr_list"},
          {"name": "explore_fr_personalization_author_individual_boost_plsst_attr_list", "as": "plsst_attr_list"},
          {"name": "explore_fr_personalization_author_individual_boost_pvtr_attr_list", "as": "pvtr_attr_list"},
          {"name": "explore_fr_personalization_author_individual_boost_pswpst_attr_list", "as": "pswpst_attr_list"},
          {"name": "explore_fr_personalization_author_individual_boost_cid_valid_interest_score_attr_list", "as": "pcvis_attr_list"},
          {"name": "explore_fr_personalization_author_individual_boost_pltr_attr_list", "as": "pltr_attr_list"},
          {"name": "explore_fr_personalization_author_individual_boost_pcltr_attr_list", "as": "pcltr_attr_list"},            
          {"name": "uExploreActiveDays", "as": "active_days"},
        ],
        import_item_attr = [
          {"name": "personalization_author_tag", "as": "need_item_attr"},
          {"name": "explore_fr_ensemble_score", "as": "ensemble_score"},
          "pcmef",
          "pwtr",
          "plsst",
          "pvtr",
          "pswpst",
          "pltr",
          "pcltr",
          {"name": "cid_valid_interest_score", "as": "pcvis"},
        ],
        export_item_attr = [
          {"name": "ensemble_score", "as": "explore_fr_ensemble_score"},
        ],
        function_name = "HighQualityAuthorIndividualBoost",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_()
    return self
  
  def explore_cal_upload_xtr_score_rank(self):
    self.split_string(
      input_common_attr = "explore_upload_fix_xtr_weight_rank_str",
      output_common_attr = "explore_upload_fix_xtr_weight_rank_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_upload_fix_xtr_power_rank_str",
      output_common_attr = "explore_upload_fix_xtr_power_rank_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_upload_fix_xtr_buttom_rank_str",
      output_common_attr = "explore_upload_fix_xtr_buttom_rank_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_upload_fix_xtr_upper_rank_str",
      output_common_attr = "explore_upload_fix_xtr_upper_rank_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .set_attr_value( 
      no_overwrite=True,
      common_attrs=[
        {
          "name": "explore_rank_upload_xtr_name_list",
          "type": "string_list",
          "value": self.update_fix_xtr_name()
        }
      ]
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_upload_fix_xtr_weight_rank_list", "as": "update_fix_xtr_weight_list"},
        {"name": "explore_upload_fix_xtr_power_rank_list", "as": "update_fix_xtr_power_list"},
        {"name": "explore_upload_fix_xtr_buttom_rank_list", "as": "update_fix_xtr_buttom_list"},
        {"name": "explore_upload_fix_xtr_upper_rank_list", "as": "update_fix_xtr_upper_list"},
        {"name": "explore_upload_window_width_rank", "as": "window_width"},
        {"name": "explore_upload_rank_window_duration_ratio", "as": "window_duration_ratio"},
        {"name": "explore_rank_upload_xtr_name_list", "as": "fix_xtr_list"},
      ],
      import_item_attr = [
        "upload_time",
        "duration_ms",
        "corr_pctr",
        "pltr",
        "pwtr",
        "pcmtr",
        "pcltr",
        "fr_score2",
        "awesome_wtd",
        "fetr",
        "fr_score1",
        "pftr",
        "pctr"
      ],
      export_item_attr = [
        {"name": "update_bar_score", "as": "fr_upload_xtr_fix_score"}
      ],
      function_name = "FixWindowXtr",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def explore_cal_hetu_one_debias_score_fr(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "hetu_tag_level_info__hetu_level_one", "as": "extract_hetu_tag_list"},
      ],
      export_item_attr = [
        {"name": "first_hetu_tag", "as": "hetu_level_one_top1"},
      ],
      function_name = "ExtractFirstHetuTag",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .split_string(
      input_common_attr = "explore_hetu_one_debias_xtr_weight_fr_str",
      output_common_attr = "explore_hetu_one_debias_xtr_weight_fr_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_hetu_one_debias_xtr_power_fr_str",
      output_common_attr = "explore_hetu_one_debias_xtr_power_fr_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_hetu_one_debias_xtr_buttom_fr_str",
      output_common_attr = "explore_hetu_one_debias_xtr_buttom_fr_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "explore_hetu_one_debias_xtr_upper_fr_str",
      output_common_attr = "explore_hetu_one_debias_xtr_upper_fr_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .set_attr_value( 
      no_overwrite=True,
      common_attrs=[
        {
          "name": "explore_fr_hetu_one_debias_xtr_name_list",
          "type": "string_list",
          "value": self.update_fix_xtr_name()
        }
      ]
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_hetu_one_debias_xtr_weight_fr_list", "as": "id_debias_xtr_weight_list"},
        {"name": "explore_hetu_one_debias_xtr_power_fr_list", "as": "id_debias_xtr_power_list"},
        {"name": "explore_hetu_one_debias_xtr_buttom_fr_list", "as": "id_debias_xtr_buttom_list"},
        {"name": "explore_hetu_one_debias_xtr_upper_fr_list", "as": "id_debias_xtr_upper_list"},
        {"name": "explore_fr_hetu_one_debias_xtr_name_list", "as": "fix_xtr_list"},
      ],
      import_item_attr = [
        {"name": "hetu_level_one_top1", "as": "debias_id_feature"},
        "corr_pctr",
        "pltr",
        "pwtr",
        "pcmtr",
        "pcltr",
        "fr_score2",
        "awesome_wtd",
        "fetr",
        "fr_score1",
        "pftr",
        "pctr"
      ],
      export_item_attr = [
        {"name": "debias_score", "as": "hetu_one_xtr_debias_fr_score"}
      ],
      function_name = "GenXtrScoreByIdFeature",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
  
  def explore_fr_good_author_pool_photo_personal_adjust(self):
    self.calc_by_formula1(
      kconf_key = "formula.scenarioKey83.FrExploreGoodAuthorPhotoBoost",
      import_item_attr = [
        "explore_positive_similarity_score_for_good_quality",
        "explore_outer_positive_similarity_score_for_good_quality",
        "explore_negative_similarity_score_for_good_quality",
        "original_submission_author_tag",
        "personalization_author_tag",
        "userfulness_author_tag",
        "author_grade_key",
        "corr_pctr"
      ],
      import_common_attr = [
        "colossus_user_info_explore_positive_photo_id_list_for_good_quality_size",
        "colossus_user_info_explore_outer_positive_photo_id_list_for_good_quality_size",
        "colossus_user_info_explore_negative_photo_id_for_good_quality_size",
        "active_days_gt_5min_rate",
        "page_index",
        "explore_today_vv",
        "refreshTimes"
      ],
      export_formula_value = [
        {"name": "final_score", "as": "final_fr_good_author_pool_boost_score"}
      ],
      abtest_biz_name = "KUAISHOU_APPS"
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "final_fr_good_author_pool_boost_score", "as": "boost_discount_coeff"},
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "BoostOrDiscountWithItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {"is_good_author_pool_photo": 1}
    )
    return self
  
  def explore_fr_hetu_tag_time_preference_boost(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": "score"},
        {"name": "hetu_tag_level_info__hetu_level_two", "as": "hetu_tag"},
      ],
      import_common_attr = [
        {"name": "colossus_tag_list", "as": "colossus_tag_list"},
        {"name": "colossus_play_time_list", "as": "colossus_play_time_list"},
        {"name": "colossus_label_list", "as": "colossus_label_list"},
        {"name": "colossus_channel_list", "as": "colossus_channel_list"},
        {"name": "colossus_duration_list", "as": "colossus_duration_list"},
        {"name": "colossus_timestamp_list", "as": "colossus_timestamp_list"},
        {"name": "highest_hetu_tag_map_ptr", "as": "highest_hetu_tag_map_ptr"},
        {"name": "explore_fr_cid_time_prefer_boost_boost_mode", "as": "boost_mode"},
        {"name": "explore_fr_cid_time_prefer_boost_channel_id", "as": "channel_id"},
        {"name": "explore_fr_cid_time_prefer_boost_cur_hour_day_limit", "as": "cur_hour_day_limit"},
        {"name": "explore_fr_cid_time_prefer_boost_his_hour_day_limit", "as": "his_hour_day_limit"},
        {"name": "explore_fr_cid_time_prefer_boost_short_term_day_limit", "as": "short_term_day_limit"},
        {"name": "explore_fr_cid_time_prefer_boost_long_term_day_limit", "as": "long_term_day_limit"},
        {"name": "explore_fr_cid_time_prefer_boost_short_term_count_limit", "as": "short_term_count_limit"},
        {"name": "explore_fr_cid_time_prefer_boost_cur_hour_count_limit", "as": "cur_hour_count_limit"},
        {"name": "explore_fr_cid_time_prefer_boost_cur_hour_expose_count_limit", "as": "cur_hour_expose_count_limit"},
        {"name": "explore_fr_cid_time_prefer_boost_alpha1", "as": "alpha1"},
        {"name": "explore_fr_cid_time_prefer_boost_bias1", "as": "bias1"},
        {"name": "explore_fr_cid_time_prefer_boost_beta1", "as": "beta1"},
        {"name": "explore_fr_cid_time_prefer_boost_alpha2", "as": "alpha2"},
        {"name": "explore_fr_cid_time_prefer_boost_bias2", "as": "bias2"},
        {"name": "explore_fr_cid_time_prefer_boost_beta2", "as": "beta2"},
        {"name": "explore_fr_cid_time_prefer_boost_upper_bound", "as": "upper_bound"},
        {"name": "explore_fr_cid_time_prefer_boost_lower_bound", "as": "lower_bound"}
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "HetuTagTimePreferenceBoost",
      class_name = "ExploreLightFunctionSetV2"
    )
    return self

  def partial_time_based_tagnex_boost(self, score_attr, stage):
    adjust_param = "explore_partial_time_based_tagnex_boost_adjust_coef_" + stage
    boost_coef = "explore_partial_time_based_tagnex_boost_coef_" + stage
    current_keys_name = "partial_time_based_tagnex_keys_" + stage
    current_values_name = "partial_time_based_tagnex_ratios_" + stage
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_tagnex_id_min", "as": "attr_min"},
        {"name": "explore_tagnex_id_max", "as": "attr_max"},
      ],
      import_item_attr = [
        {"name": "hetu_tag_level_info__hetu_tag", "as": "item_list_attr_name"},
      ],
      export_common_attr = [
        {"name": "key_list", "as": current_keys_name},
        {"name": "value_list", "as": current_values_name},
      ],
      function_name = "CalItemListAttrFrequency",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": adjust_param, "as": "adjust_param"},
        {"name": "partial_time_based_tagnex_keys", "as": "history_key_list"},
        {"name": "partial_time_based_tagnex_ratios", "as": "history_value_list"},
        {"name": current_keys_name, "as": "candidate_key_list"},
        {"name": current_values_name, "as": "candidate_value_list"},
        {"name": "explore_tagnex_id_min", "as": "attr_min"},
        {"name": "explore_tagnex_id_max", "as": "attr_max"},
      ],
      import_item_attr = [
        {"name": "hetu_tag_level_info__hetu_tag", "as": "item_list_attr_name"},
      ],
      export_item_attr = [
        {"name": "final_coef", "as": boost_coef},
      ],
      function_name = "CalTagListRatioDiff",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": boost_coef, "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr},
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def partial_time_based_interest_boost(self, score_attr, stage="fr_s2"):
    adjust_param = "explore_partial_time_based_interest_boost_adjust_coef_" + stage
    boost_coef = "explore_partial_time_based_interest_boost_coef_" + stage
    current_keys_name = "partial_time_based_interest_keys_" + stage
    current_values_name = "partial_time_based_interest_ratios_" + stage
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "cluster_id_632", "as": "item_attr_name"},
      ],
      export_common_attr = [
        {"name": "key_list", "as": current_keys_name},
        {"name": "value_list", "as": current_values_name},
      ],
      function_name = "CalItemAttrFrequency",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": adjust_param, "as": "adjust_param"},
        {"name": "partial_time_based_interest_keys", "as": "history_key_list"},
        {"name": "partial_time_based_interest_ratios", "as": "history_value_list"},
        {"name": current_keys_name, "as": "candidate_key_list"},
        {"name": current_values_name, "as": "candidate_value_list"},
      ],
      import_item_attr = [
        {"name": "cluster_id_632", "as": "item_attr_name"},
      ],
      export_item_attr = [
        {"name": "final_coef", "as": boost_coef},
      ],
      function_name = "CalTagRatioDiff",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": boost_coef, "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr},
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def fr_ctr_ensemble_power_weight_adjust(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_fr_ctr_power", "as": "xtr_weight"},
        {"name": "explore_recent_valid_click_count", "as": "user_vv"},
        {"name": "explore_fr_ctr_ensemble_power_weight_adjust_exp_upper", "as": "exp_upper"},
        {"name": "explore_fr_ctr_ensemble_power_weight_adjust_alpha", "as": "alpha"},
        {"name": "explore_fr_ctr_ensemble_power_weight_adjust_beta", "as": "beta"},
        {"name": "explore_fr_ctr_ensemble_power_weight_adjust_omega", "as": "omega"},
        {"name": "explore_fr_ctr_ensemble_power_weight_adjust_max", "as": "coeff_max"},
        {"name": "explore_fr_ctr_ensemble_power_weight_adjust_min", "as": "coeff_min"},
      ],
      export_common_attr = [
        {"name": "xtr_weight", "as": "explore_fr_ctr_power"},
      ],
      function_name = "AdjustWeightByUserVv",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def fr_poor_quality_author_personal_deboost(self):
    self.calc_by_formula1(
      kconf_key = "formula.scenarioKey34.ExploreFrSexyInduceDeboost",
      import_item_attr = [
        "author_grade_key",
        "is_sexy_induce_photo"
      ],
      import_common_attr = [
        "uSexyInterestScore",
        "active_days_gt_5min_rate"
      ],
      export_formula_value = [
        {"name": "final_deboost_score", "as": "final_fr_bad_author_deboost_score"}
      ],
      abtest_biz_name = "KUAISHOU_APPS",
      select_item = {
        "join": "or",
        "filters": [{
          "attr_name": "is_sexy_induce_photo",
          "select_if": ">",
          "compare_to": 0,
        }, 
        {
          "attr_name": "author_grade_key",
          "select_if": "<",
          "compare_to": 9,
        }],
      }
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "final_fr_bad_author_deboost_score", "as": "boost_discount_coeff"},
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "BoostOrDiscountWithItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      select_item = {
        "join": "or",
        "filters": [{
          "attr_name": "is_sexy_induce_photo",
          "select_if": ">",
          "compare_to": 0,
        }, 
        {
          "attr_name": "author_grade_key",
          "select_if": "<",
          "compare_to": 9,
        }],
      }
    )
    return self

  def user_active_days_ensemble_koc_cover_htr_power_weight_adjust(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "koc_cover_htr_score_in_order_weight", "as": "xtr_weight"},
        {"name": "recent_hate_count", "as": "user_vv"},
        {"name": "explore_fr_ensemble_power_koc_cover_htr_weight_adjust_exp_upper", "as": "exp_upper"},
        {"name": "explore_fr_ensemble_power_koc_cover_htr_weight_adjust_alpha", "as": "alpha"},
        {"name": "explore_fr_ensemble_power_koc_cover_htr_weight_adjust_beta", "as": "beta"},
        {"name": "explore_fr_ensemble_power_koc_cover_htr_weight_adjust_omega", "as": "omega"},
        {"name": "explore_fr_ensemble_power_koc_cover_htr_weight_adjust_max", "as": "coeff_max"},
        {"name": "explore_fr_ensemble_power_koc_cover_htr_weight_adjust_min", "as": "coeff_min"},
      ],
      export_common_attr = [
        {"name": "xtr_weight", "as": "koc_cover_htr_score_in_order_weight"},
      ],
      function_name = "AdjustWeightByUserVv",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def user_active_days_ensemble_koc_detail_htr_power_weight_adjust(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "koc_detail_htr_score_in_order_weight", "as": "xtr_weight"},
        {"name": "recent_hate_count", "as": "user_vv"},
        {"name": "explore_fr_ensemble_power_koc_detail_htr_weight_adjust_exp_upper", "as": "exp_upper"},
        {"name": "explore_fr_ensemble_power_koc_detail_htr_weight_adjust_alpha", "as": "alpha"},
        {"name": "explore_fr_ensemble_power_koc_detail_htr_weight_adjust_beta", "as": "beta"},
        {"name": "explore_fr_ensemble_power_koc_detail_htr_weight_adjust_omega", "as": "omega"},
        {"name": "explore_fr_ensemble_power_koc_detail_htr_weight_adjust_max", "as": "coeff_max"},
        {"name": "explore_fr_ensemble_power_koc_detail_htr_weight_adjust_min", "as": "coeff_min"},
      ],
      export_common_attr = [
        {"name": "xtr_weight", "as": "koc_detail_htr_score_in_order_weight"},
      ],
      function_name = "AdjustWeightByUserVv",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def user_hate_reason_bad_score_power_weight_adjust(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_bad_sim_score_in_order_weight", "as": "xtr_weight"},
        {"name": "user_poor_quality_hate_reason_count", "as": "user_vv"},
        {"name": "explore_fr_user_poor_quality_hate_reason_weight_adjust_exp_upper", "as": "exp_upper"},
        {"name": "explore_fr_user_poor_quality_hate_reason_weight_adjust_alpha", "as": "alpha"},
        {"name": "explore_fr_user_poor_quality_hate_reason_weight_adjust_beta", "as": "beta"},
        {"name": "explore_fr_user_poor_quality_hate_reason_weight_adjust_omega", "as": "omega"},
        {"name": "explore_fr_user_poor_quality_hate_reason_weight_adjust_max", "as": "coeff_max"},
        {"name": "explore_fr_user_poor_quality_hate_reason_weight_adjust_min", "as": "coeff_min"},
      ],
      export_common_attr = [
        {"name": "xtr_weight", "as": "explore_bad_sim_score_in_order_weight"},
      ],
      function_name = "AdjustWeightByUserVv",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def fr_pic_real_show_not_click_decay(self, hetu_attr_name):
    self.enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "_USER_ID_", "as": "user_id"},
          {"name": "colossus_pic_consec_show_decay", "as": "hetu_decay_map"},
          {"name": "explore_pic_rerank_div_realshow_decay_hetu_topk", "as": "decay_hetu_topk"},
          {"name": "explore_pic_rerank_div_realshow_decay_power", "as": "decay_power"},
          {"name": "explore_pic_rerank_div_realshow_decay_min_page_index", "as": "min_page_index"},
          "page_index",
        ],
        import_item_attr = [
          {"name": hetu_attr_name, "as": "hetu_tag_level_info"},
          {"name": "fr_pic_rerank_score_for_div", "as": "origin_score"},
        ],
        export_item_attr = [
          {"name": "output_score", "as": "fr_pic_rerank_score_for_div"},
        ],
        function_name = "DecayByRealshowNotClick",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          "is_picture": 1
        }
      )
    return self

  def short_term_photo_tagnex_score_adjust(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "short_term_item_tagnex_score", "as": "boost_discount_coeff"},
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def short_term_photo_cluster_id_score_adjust(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "short_term_item_cluster_id_score", "as": "boost_discount_coeff"},
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def short_term_photo_hetu_level2_score_adjust(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "short_term_item_hetu_level2_score", "as": "boost_discount_coeff"},
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def short_term_photo_hashtag_score_adjust(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "short_term_item_hashtag_score", "as": "boost_discount_coeff"},
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def short_term_photo_hetu_tag_score_adjust(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "short_term_item_hetu_tag_score", "as": "boost_discount_coeff"},
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def short_term_photo_interest_community_tag_score_adjust(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "short_term_item_interest_community_tag_score", "as": "boost_discount_coeff"},
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def short_term_photo_sid_score_adjust(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "short_term_item_sid_score", "as": "boost_discount_coeff"},
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def explore_cover_video_not_correlation_ranking_deboost(self):
    self.enrich_attr_by_light_function(
          import_item_attr = [
            "hetu_tag_level_info__hetu_tag"
          ],
          export_item_attr = [
            {"name": "mmu_not_correlation_tag", "as": "mmu_not_correlation_tag_ranking"}
          ],
          function_name = "ExploreCoverVideoNotCorrelationTag",
          class_name = "ExploreLightFunctionSetV2"
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_not_correlation_deboost_weight_ranking_s2", "as": "boost_discount_coeff"}
          ],
          import_item_attr = [
             {"name": "explore_fr_ensemble_score", "as": "score"}
          ],
          export_item_attr = [
             {"name": "score", "as": "explore_fr_ensemble_score"}
          ],
          function_name = "BoostOrDiscountV2",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "mmu_not_correlation_tag_ranking": 1
          }
        )
    return self

  def gen_author_click_value_score(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "author_click_value_score_high_ptr", "as": "map_ptr"}
      ],
      import_item_attr = [
        {"name": "author__id", "as": "key_attr"}
      ],
      export_item_attr = [
        {"name": "target_item_attr", "as": "author_click_value_score_high"},
      ],
      function_name = "GetItemAttrByIntToDoubleMapPtr",
      class_name = "ExploreLightFunctionSetV2"
    ) \
    .item_attr_operation(
      item_attr_a = "author_click_value_score_high",
      common_attr_b = "{{explore_author_click_value_score_max}}",
      operator = "+",
      output_attr = "author_click_value_score_high"
    ) \
    .if_("explore_enable_author_click_value_score_high_plus_pctr == 1") \
      .calc_by_simple_formula(
        formulas = [
          dict(
            expr = "[[author_click_value_score_high]] * ([[corr_pctr]] ^ {{explore_author_click_value_score_high_pctr_alpha}})",
            output_attr = "author_click_value_score_high"
          )
        ]
      ) \
    .end_() \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "author_click_value_score_low_ptr", "as": "map_ptr"}
      ],
      import_item_attr = [
        {"name": "author__id", "as": "key_attr"}
      ],
      export_item_attr = [
        {"name": "target_item_attr", "as": "author_click_value_score_low"},
      ],
      function_name = "GetItemAttrByIntToDoubleMapPtr",
      class_name = "ExploreLightFunctionSetV2"
    ) \
    .item_attr_operation(
      item_attr_a = "author_click_value_score_low",
      common_attr_b = -1,
      operator = "*",
      output_attr = "author_click_value_score_low"
    ) \
    .item_attr_operation(
      item_attr_a = "author_click_value_score_low",
      common_attr_b = "{{explore_author_click_value_score_max}}",
      operator = "+",
      output_attr = "author_click_value_score_low"
    ) \
    .if_("explore_enable_author_click_value_score_low_plus_pctr == 1") \
      .calc_by_simple_formula(
        formulas = [
          dict(
            expr = "[[author_click_value_score_low]] * ([[corr_pctr]] ^ {{explore_author_click_value_score_low_pctr_alpha}})",
            output_attr = "author_click_value_score_low"
          )
        ]
      ) \
    .end_()
    return self

  def explore_good_author_show_case_boost(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "ecommerce_good_author_show_case_ptr", "as": "aid_set_ptr"}
      ],
      import_item_attr = [
        "author__id"
      ],
      export_item_attr = [
        {"name": "is_target_photo", "as": "is_good_author_show_case"}
      ],
      function_name = "AidInSet",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .calc_by_formula1(
      kconf_key = "formula.scenarioKey35.ExploreRankLiveShowBoostScore",
      import_item_attr = [
        "score_pctr_es",
        "score_pwtr_es",
        "pwtr",
        "fr_score2",
        "fr_score1"
      ],
      export_formula_value = [
        {"name": "final_score", "as": "live_show_boost_score"}
      ],
      abtest_biz_name = "KUAISHOU_APPS",
      target_item = {
        "is_good_author_show_case": 1
      }
    ) \
    .item_attr_operation(
      item_attr_a = "explore_fr_ensemble_score",
      item_attr_b = "live_show_boost_score",
      operator = "*",
      output_attr = "explore_fr_ensemble_score",
      target_item = {
        "is_good_author_show_case": 1
      }
    )
    return self

  def explore_good_author_e_commerce_boost(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "ecommerce_good_author_e_commerce_ptr", "as": "aid_set_ptr"}
      ],
      import_item_attr = [
        "author__id"
      ],
      export_item_attr = [
        {"name": "is_target_photo", "as": "is_good_author_e_commerce"}
      ],
      function_name = "AidInSet",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .calc_by_formula1(
      kconf_key = "formula.scenarioKey14.ExploreRankECommerceBoostScore",
      import_item_attr = [
        "score_pctr_es",
        "score_pwtr_es",
        "pwtr",
        "fr_score2",
        "fr_score1"
      ],
      export_formula_value = [
        {"name": "final_score", "as": "e_commerce_boost_score"}
      ],
      abtest_biz_name = "KUAISHOU_APPS",
      target_item = {
        "is_good_author_e_commerce": 1
      }
    ) \
    .item_attr_operation(
      item_attr_a = "explore_fr_ensemble_score",
      item_attr_b = "e_commerce_boost_score",
      operator = "*",
      output_attr = "explore_fr_ensemble_score",
      target_item = {
        "is_good_author_e_commerce": 1
      }
    )
    return self

  def explore_uninterest_ctr_score_adjust(self):
    self.copy_attr(
      attrs=[{
        "from_item": "explore_uninterest_ctr_score",
        "to_item": "explore_uninterest_ctr_adjust_score"
      }]
    ) \
    .switch_("explore_fr_s1_uninterest_ctr_adjust_score_limit_method") \
      .case_(1) \
        .pack_item_attr(
          item_source = {
            "reco_results": True,
          },
          mappings = [
            {
              "aggregator": "avg",
              "from_item_attr": "explore_uninterest_ctr_adjust_score",
              "to_common_attr": "explore_uninterest_ctr_adjust_score_limit",
            },
          ],
          target_item = {"zero_exposure_flag": 1}
        ) \
      .case_(2) \
        .pack_item_attr(
          item_source = {
            "reco_results": True,
          },
          mappings = [
            {
              "aggregator": "min",
              "from_item_attr": "explore_uninterest_ctr_adjust_score",
              "to_common_attr": "explore_uninterest_ctr_adjust_score_limit",
            },
          ],
          target_item = {"zero_exposure_flag": 1}
        ) \
      .default_() \
        .copy_attr(
          attrs = [{
            "from_common": "explore_fr_s1_uninterest_ctr_adjust_score_limit_default",
            "to_common": "explore_uninterest_ctr_adjust_score_limit",
          }]
        ) \
    .end_() \
    .set_attr_value(
      item_attrs = [
        {
          "name": "explore_uninterest_ctr_adjust_score",
          "type": "double",
          "value": 1.0
        }
      ],
      select_item = {
        "attr_name": "explore_uninterest_ctr_adjust_score",
        "compare_to": "{{explore_uninterest_ctr_adjust_score_limit}}",
        "select_if": ">",
      },
    )
    return self

  def user_active_days_ensemble_power_consume_time_slide_weight_adjust(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_fr_consume_time_slide_weight", "as": "xtr_weight"},
        {"name": "uExploreActiveDays", "as": "user_vv"},
        {"name": "explore_fr_ensemble_power_consume_time_slide_weight_adjust_exp_upper", "as": "exp_upper"},
        {"name": "explore_fr_ensemble_power_consume_time_slide_weight_adjust_alpha", "as": "alpha"},
        {"name": "explore_fr_ensemble_power_consume_time_slide_weight_adjust_beta", "as": "beta"},
        {"name": "explore_fr_ensemble_power_consume_time_slide_weight_adjust_omega", "as": "omega"},
        {"name": "explore_fr_ensemble_power_consume_time_slide_weight_adjust_max", "as": "coeff_max"},
        {"name": "explore_fr_ensemble_power_consume_time_slide_weight_adjust_min", "as": "coeff_min"},
      ],
      export_common_attr = [
        {"name": "xtr_weight", "as": "explore_fr_consume_time_slide_weight"},
      ],
      function_name = "AdjustWeightByUserVv",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def fr_enter_fountain_score_debias_by_picture_type(self):
    self.set_attr_value(
      item_attrs=[{
        "name": "corr_fetr",
        "type": "double",
        "value": 0.0
      }, {
        "name": "corr_fountain_eff",
        "type": "double",
        "value": 0.0
      }],
      target_item = {
        "picture_type": [2, 3]
      }
    )
    return self

  def explore_fr_interest_card_photo_score_adjust(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "interest_card_adjust_score", "as": "boost_discount_coeff"},
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
