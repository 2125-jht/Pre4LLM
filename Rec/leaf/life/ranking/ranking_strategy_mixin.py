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
      .explore_calc_value_and_rank_score(
        enable_queue_names = "vrs_enable_queue_names",
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
        lua_script_file = "life/ranking/lua/module/ranking_ensemble_sort__score_coeff.lua"
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
        lua_script_file = "life/ranking/lua/module/ranking_ensemble_sort__score_coeff.lua"
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
        lua_script_file = "life/ranking/lua/module/ranking_ensemble_sort__print.lua"
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
        lua_script_file = "life/ranking/lua/module/ranking_ensemble_sort__print.lua"
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
      import_item_attr = [
        "corr_pctr",
        "pltr",
        "corr_pwtr",
        "pftr",
        "pctr_x_pcmtr",
        "pptr",
        "psvr",
        "report_discount",
        "hate_discount",
        "pdtr",
        "pepstr",
        "pcltr",
        "pcmef",
        "phtr",
        "fr_score2"
      ],
      export_item_attr = [
        "score_pctr",
        "score_pltr",
        "score_pwtr",
        "score_pftr",
        "score_pcmtr",
        "score_pptr",
        "score_psvr",
        "score_pdtr",
        "score_pepstr",
        "score_pcltr",
        "score_pcmef",
        "score_phtr",
      ],
      function_for_item = "gen_score_stage1",
      lua_script_file = "life/ranking/lua/module/ranking_score__gen_score_stage1.lua"
    )
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
        "explore_fr_es_pctr_x_pxtr_power_beta_action"
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
        "fr_score2",
        "awesome_wtd"
      ],
      export_item_attr = [
        "score_consume_time_ltr",
        "score_pctr_x_pcltr",
        "score_pctr_x_pepstr",
        "score_pctr_x_pptr",
        "score_pctr_x_pdtr",
        "score_pctr_x_pftr",
        "score_pctr_x_fr_score2",
        "score_pctr_x_awesome_wtd",
      ],
      function_for_item = "score_coeff_calculate_stage2",
      function_for_common = "collect_garbage",
      lua_script_file = "life/ranking/lua/module/ranking_ensemble_sort__score_coeff.lua"
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

  def rank_stage1_personal_cem(self):
    """
    Module: RankingStageOneTruncateModule
    功能: 个性化 cem
    Owner: liuhao07
    Date: 2023-06-29
    :return:
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "_USER_ID_", "as": "user_id"},
        {"name": "explore_rank_stage1_personal_cem_exp_group_num", "as": "exp_group_num"},
        {"name": "explore_rank_stage1_personal_cem_model_key", "as": "model_key"},
      ],
      export_common_attr = [
        {"name": "personal_cem_redis_key", "as": "explore_rank_stage1_personal_cem_redis_key"}
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
          "redis_key": "{{explore_rank_stage1_personal_cem_redis_key}}",
          "output_attr_name": "explore_rank_stage1_personal_cem_model"
        }
      ]
    ) \
    .pack_item_attr(
      item_source = {"reco_results": True},
      mappings = [
        {
          "from_item_attr": "pctr",
          "to_common_attr": "pctr_avg",
          "aggregator":"avg"
        },
        {
          "from_item_attr": "pctr",
          "to_common_attr": "pctr_dev",
          "aggregator":"dev"
        },
        {
          "from_item_attr": "fr_score2",
          "to_common_attr": "fr_score2_avg",
          "aggregator":"avg"
        },
        {
          "from_item_attr": "fr_score2",
          "to_common_attr": "fr_score2_dev",
          "aggregator":"dev"
        },
        {
          "from_item_attr": "fetr",
          "to_common_attr": "fetr_avg",
          "aggregator":"avg"
        },
        {
          "from_item_attr": "fetr",
          "to_common_attr": "fetr_dev",
          "aggregator":"dev"
        },
      ],
      target_item = {"is_picture" : 0}
    ) \
    .explore_personal_cem_weight(
      model_str_attr = "explore_rank_stage1_personal_cem_model",
      save_exp_info_attr = "personal_cem_exp_info",
      weight_configs = [
        {
          "weight_name": "explore_rank_stage1_cut_off_ratio",
          "output_attr": "explore_rank_stage1_cut_off_ratio",
          "enable_activation": "explore_rank_stage1_personal_cem_cut_off_ratio_enable_activation",
          "min_value_attr": "explore_rank_stage1_personal_cem_cut_off_ratio_min_value",
          "max_value_attr": "explore_rank_stage1_personal_cem_cut_off_ratio_max_value"
        },
        {
          "weight_name": "explore_rank_stage1_es_ctr_weight",
          "output_attr": "explore_rank_stage1_es_ctr_weight",
          "enable_activation": "explore_rank_stage1_personal_cem_es_ctr_weight_enable_activation",
          "min_value_attr": "explore_rank_stage1_personal_cem_es_ctr_weight_min_value",
          "max_value_attr": "explore_rank_stage1_personal_cem_es_ctr_weight_max_value"
        },
        {
          "weight_name": "explore_rank_stage1_es_fr_score2_weight",
          "output_attr": "explore_rank_stage1_es_fr_score2_weight",
          "enable_activation": "explore_rank_stage1_personal_cem_es_fr_score2_weight_enable_activation",
          "min_value_attr": "explore_rank_stage1_personal_cem_es_fr_score2_weight_min_value",
          "max_value_attr": "explore_rank_stage1_personal_cem_es_fr_score2_weight_max_value"
        },
        {
          "weight_name": "explore_rank_stage1_es_fetr_weight",
          "output_attr": "explore_rank_stage1_es_fetr_weight",
          "enable_activation": "explore_rank_stage1_personal_cem_es_fetr_weight_enable_activation",
          "min_value_attr": "explore_rank_stage1_personal_cem_es_fetr_weight_min_value",
          "max_value_attr": "explore_rank_stage1_personal_cem_es_fetr_weight_max_value"
        },
      ],
      feature_configs = [
        {
          "feature_name": "pctr_avg",
          "feature_attr": "pctr_avg",
          "treat_type": "original",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage1_personal_cem_feature_pctr_avg_min_value",
          "max_value_attr": "explore_rank_stage1_personal_cem_feature_pctr_avg_max_value",
        },
        {
          "feature_name": "pctr_avg_maxmin",
          "feature_attr": "pctr_avg",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage1_personal_cem_feature_pctr_avg_min_value",
          "max_value_attr": "explore_rank_stage1_personal_cem_feature_pctr_avg_max_value",
        },
        {
          "feature_name": "pctr_dev",
          "feature_attr": "pctr_dev",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage1_personal_cem_feature_pctr_dev_min_value",
          "max_value_attr": "explore_rank_stage1_personal_cem_feature_pctr_dev_max_value",
        },
        {
          "feature_name": "fr_score2_avg",
          "feature_attr": "fr_score2_avg",
          "treat_type": "original",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage1_personal_cem_feature_fr_score2_avg_min_value",
          "max_value_attr": "explore_rank_stage1_personal_cem_feature_fr_score2_avg_max_value",
        },
        {
          "feature_name": "fr_score2_avg_maxmin",
          "feature_attr": "fr_score2_avg",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage1_personal_cem_feature_fr_score2_avg_min_value",
          "max_value_attr": "explore_rank_stage1_personal_cem_feature_fr_score2_avg_max_value",
        },
        {
          "feature_name": "fr_score2_dev",
          "feature_attr": "fr_score2_dev",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage1_personal_cem_feature_fr_score2_dev_min_value",
          "max_value_attr": "explore_rank_stage1_personal_cem_feature_fr_score2_dev_max_value",
        },
        {
          "feature_name": "fetr_avg",
          "feature_attr": "fetr_avg",
          "treat_type": "original",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage1_personal_cem_feature_fetr_avg_min_value",
          "max_value_attr": "explore_rank_stage1_personal_cem_feature_fetr_avg_max_value",
        },
        {
          "feature_name": "fetr_avg_maxmin",
          "feature_attr": "fetr_avg",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage1_personal_cem_feature_fetr_avg_min_value",
          "max_value_attr": "explore_rank_stage1_personal_cem_feature_fetr_avg_max_value",
        },
        {
          "feature_name": "fetr_dev",
          "feature_attr": "fetr_dev",
          "treat_type": "maxmin",
          "value_type":"double",
          "min_value_attr": "explore_rank_stage1_personal_cem_feature_fetr_dev_min_value",
          "max_value_attr": "explore_rank_stage1_personal_cem_feature_fetr_dev_max_value",
        },
      ],
    ) \
    .export_attr_to_kafka(
      kafka_topic = "explore_personal_cem",
      common_attrs = ["request_id", "_USER_ID_", "_DEVICE_ID_", "personal_cem_exp_info"],
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

  def rank_stage1_personal_cem_es_weight_adjust(self):
    """
    Module: RankingStageOneTruncateModule
    功能: 个性化 cem 调权
    Owner: liuhao07
    Date: 2023-06-29
    :return:
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_fullrank_filter_ctr_weight", "as": "ori_weight"},
        {"name": "explore_rank_stage1_es_ctr_weight", "as": "coeff"},
        {"name": "explore_rank_stage1_es_ctr_weight_min_value", "as": "weight_min_value"},
        {"name": "explore_rank_stage1_es_ctr_weight_max_value", "as": "weight_max_value"},
      ],
      export_common_attr = [
        {"name": "new_weight", "as": "explore_fullrank_filter_ctr_weight"}
      ],
      function_name = "AdjustWeight",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_fullrank_filter_fr_score2_weight", "as": "ori_weight"},
        {"name": "explore_rank_stage1_es_fr_score2_weight", "as": "coeff"},
        {"name": "explore_rank_stage1_es_fr_score2_weight_min_value", "as": "weight_min_value"},
        {"name": "explore_rank_stage1_es_fr_score2_weight_max_value", "as": "weight_max_value"},
      ],
      export_common_attr = [
        {"name": "new_weight", "as": "explore_fullrank_filter_fr_score2_weight"}
      ],
      function_name = "AdjustWeight",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_fullrank_filter_fetr_weight", "as": "ori_weight"},
        {"name": "explore_rank_stage1_es_fetr_weight", "as": "coeff"},
        {"name": "explore_rank_stage1_es_fetr_weight_min_value", "as": "weight_min_value"},
        {"name": "explore_rank_stage1_es_fetr_weight_max_value", "as": "weight_max_value"},
      ],
      export_common_attr = [
        {"name": "new_weight", "as": "explore_fullrank_filter_fetr_weight"}
      ],
      function_name = "AdjustWeight",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def rank_stage1_personal_cem_cut_off_ratio_adjust(self):
    """
    Module: RankingStageOneTruncateModule
    功能: 个性化 cem 调权
    Owner: liuhao07
    Date: 2023-06-29
    :return:
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "ensemble_filter_coeff", "as": "ori_weight"},
        {"name": "explore_rank_stage1_cut_off_ratio", "as": "coeff"},
        {"name": "explore_rank_stage1_cut_off_ratio_min_value", "as": "weight_min_value"},
        {"name": "explore_rank_stage1_cut_off_ratio_max_value", "as": "weight_max_value"},
      ],
      export_common_attr = [
        {"name": "new_weight", "as": "ensemble_filter_coeff"}
      ],
      function_name = "AdjustWeight",
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
      ]
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
        }
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
    self.if_("explore_enable_rank_select_rank_neg_result == 1") \
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

  def gen_min_act_rank_reci(self):
    """
    精排生成最小互动rank
    """
    self\
    .sort(
      score_from_attr = "fullrank_sim_like_score",
    ) \
    .copy_item_meta_info(
      save_item_seq_to_attr = "fullrank_like_rank",
    ) \
    .sort(
      score_from_attr = "fullrank_detail_pcmtr",
    ) \
    .copy_item_meta_info(
      save_item_seq_to_attr = "fullrank_cmtr_rank",
    ) \
    .sort(
      score_from_attr = "fullrank_detail_pcmef",
    ) \
    .copy_item_meta_info(
      save_item_seq_to_attr = "fullrank_cmef_rank",
    ) \
    .sort(
      score_from_attr = "fullrank_final_lstr",
    ) \
    .copy_item_meta_info(
      save_item_seq_to_attr = "fullrank_lstr_rank",
    ) \
    .sort(
      score_from_attr = "fullrank_detail_pepstr",
    ) \
    .copy_item_meta_info(
      save_item_seq_to_attr = "fullrank_epstr_rank",
    ) \
    .sort(
      score_from_attr = "fullrank_sim_follow_score",
    ) \
    .copy_item_meta_info(
      save_item_seq_to_attr = "fullrank_follow_rank",
    ) \
    .sort(
      score_from_attr = "fullrank_detail_pftr",
    ) \
    .copy_item_meta_info(
      save_item_seq_to_attr = "fullrank_ftr_rank",
    ) \
    .sort(
      score_from_attr = "fullrank_sim_pcltr",
    ) \
    .copy_item_meta_info(
      save_item_seq_to_attr = "fullrank_cltr_rank",
    ) \
    .split_string(
      input_common_attr = "fountain_fullrank_min_act_rank_weights_str",
      output_common_attr = "fountain_fullrank_min_act_rank_weights",
      delimiters = ":",
      parse_to_int = True,
    ) \
    .enrich_attr_by_lua(
      import_common_attr = [
        "fountain_fullrank_min_act_rank_weights",
      ],
      import_item_attr = [
        "fullrank_like_rank",
        "fullrank_cmtr_rank",
        "fullrank_cmef_rank",
        "fullrank_lstr_rank",
        "fullrank_epstr_rank",
        "fullrank_follow_rank",
        "fullrank_ftr_rank",
        "fullrank_cltr_rank",
      ],
      export_item_attr = [
        "fullrank_min_act_rank_reci"
      ],
      function_for_item = "calc_min_act_rank_reci",
      lua_script_file = "life/ranking/lua/module/fountain_ranking_score__calc_min_act_rank_reci.lua",
    )
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

  def cal_corr_pctr_psvr(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "corr_pctr", "as": "pctr"},
        "psvr"
      ],
      export_item_attr = [
        {"name": "pctr_psvr", "as": "corr_pctr_psvr"},
      ],
      function_name = "CalcPctrPsvr",
      class_name = "ExploreLifeLightFunctionSet",
    )
    return self

  def fr_marketing_compensation_discount(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "life_fr_marketing_compensation_discount_coef", "as": "boost_discount_coeff"},
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

  def rank_marketing_compensation_adjust(self): 
    self \
    .if_("life_fountain_enable_rank_calc_marketing_compensation_coeff == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "life_fountain_rank_marketing_compensation_adjust_ctr_weight", "as": "ctr_weight"},
          {"name": "life_fountain_rank_marketing_compensation_adjust_watchtime_weight", "as": "watchtime_weight"},
          {"name": "life_fountain_rank_marketing_compensation_adjust_score_base", "as": "score_base"},
          {"name": "life_fountain_rank_marketing_compensation_adjust_adjust_version", "as": "adjust_version"},
          {"name": "life_fountain_rank_marketing_compensation_adjust_score_base_ratio", "as": "score_base_ratio"},
        ],
        import_item_attr = [
          {"name": "fullrank_sim_click_score", "as": "ctr"},
          {"name": "fullrank_sim_pfintr", "as": "watchtime"},
        ],
        export_item_attr = [
          {"name": "coeff", "as": "rank_marketing_compensation_coeff"},
        ],
        function_name = "CalcRewardCoeff",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {"is_marketing_compensation_photo": 1}
      ) \
    .end_() \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "life_fountain_rank_marketing_compensation_adjust_scale_factor", "as": "scale_factor"},
        {"name": "life_fountain_rank_marketing_compensation_adjust_base_coeff", "as": "base_coeff"},
      ],
      import_item_attr = [
        {"name": "rank_marketing_compensation_coeff", "as": "reward_coeff"},
      ],
      export_item_attr = [
        {"name": "new_coeff", "as": "fullrank_ensemble_score_adjust_coeff"},
      ],
      function_name = "MarketingCompensationPhotoAdjust",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {"is_marketing_compensation_photo": 1}
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "fullrank_ensemble_score_adjust_coeff", "as": "boost_discount_coeff"},
        {"name": "fullrank_ensemble_score_after_adjust", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "fullrank_ensemble_score_after_adjust"},
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {"is_marketing_compensation_photo": 1}
    )
    return self

  def fr_search_topk_boost(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "life_fr_search_topk_boost_coef", "as": "boost_coeff"},
        {"name": "life_fr_search_boost_topk", "as": "topk"},
      ],
      import_item_attr = [
        {"name": "explore_fr_ensemble_score", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "explore_fr_ensemble_score"},
      ],
      function_name = "BoostTopk",
      class_name = "ExploreLifeLightFunctionSet",
      target_item = {
        "reason" : [2704]
      }
    )
    return self

  def request_pxtr_weight_adjust(self):
    self.pack_item_attr(
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

  def life_fr_s2_diversity_weight_adjust(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "recent_unclick_rate", "as": "alpha"},
        {"name": "life_fr_s2_diversity_weight_adjust_beta", "as": "beta"},
        {"name": "life_fr_s2_diversity_weight_adjust_gamma", "as": "gamma"},
        {"name": "life_fr_s2_diversity_weight_adjust_coeff_max", "as": "adjust_coeff_max"},
        {"name": "xlife_diversity_fr_weight", "as": "input_weight"},
      ],
      export_common_attr = [
        {"name": "output_weight", "as": "xlife_diversity_fr_weight"},
      ],
      function_name = "AdjustWeightByMultiply",
      class_name = "ExploreLifeLightFunctionSet",
    )
    return self
  
  def life_fr_s2_neg_sim_weight_adjust(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "recent_unclick_rate", "as": "alpha"},
        {"name": "life_fr_s2_neg_sim_weight_adjust_beta_by_unclk", "as": "beta"},
        {"name": "life_fr_s2_neg_sim_weight_adjust_gamma_by_unclk", "as": "gamma"},
        {"name": "life_fr_s2_neg_sim_weight_adjust_coeff_max_by_unclk", "as": "adjust_coeff_max"},
        {"name": "hot_user_unexpected_score_weight_new", "as": "input_weight"},
      ],
      export_common_attr = [
        {"name": "output_weight", "as": "hot_user_unexpected_score_weight_new"},
      ],
      function_name = "AdjustWeightByMultiply",
      class_name = "ExploreLifeLightFunctionSet",
    ).enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "recent_short_play_rate", "as": "alpha"},
        {"name": "life_fr_s2_neg_sim_weight_adjust_beta_by_svtr", "as": "beta"},
        {"name": "life_fr_s2_neg_sim_weight_adjust_gamma_by_svtr", "as": "gamma"},
        {"name": "life_fr_s2_neg_sim_weight_adjust_coeff_max_by_svtr", "as": "adjust_coeff_max"},
        {"name": "hot_user_unexpected_score_weight_new", "as": "input_weight"},
      ],
      export_common_attr = [
        {"name": "output_weight", "as": "hot_user_unexpected_score_weight_new"},
      ],
      function_name = "AdjustWeightByMultiply",
      class_name = "ExploreLifeLightFunctionSet",
    )
    return self
  
  def calc_active_hetu_debias_score(self):
    self.get_kconf_params(
        kconf_configs = [
        {
          "kconf_key": "reco.hot.life_ranking_active_hetu_pctr_debias_pcoc", 
          "value_type": "json",
          "json_path": "shallow_consumer_user",
          "export_common_attr": "shallow_consumer_user_pcoc" 
        },
        {
          "kconf_key": "reco.hot.life_ranking_active_hetu_pctr_debias_pcoc", 
          "value_type": "json",
          "json_path": "secondary_shallow_consumer_user",
          "export_common_attr": "secondary_shallow_consumer_user_pcoc" 
        },
        {
          "kconf_key": "reco.hot.life_ranking_active_hetu_pctr_debias_pcoc", 
          "value_type": "json",
          "json_path": "moderate_user",
          "export_common_attr": "moderate_user_pcoc" 
        },
        {
          "kconf_key": "reco.hot.life_ranking_active_hetu_pctr_debias_pcoc", 
          "value_type": "json",
          "json_path": "sub_deep_user",
          "export_common_attr": "sub_deep_user_pcoc" 
        },
        {
          "kconf_key": "reco.hot.life_ranking_active_hetu_pctr_debias_pcoc", 
          "value_type": "json",
          "json_path": "deep_user",
          "export_common_attr": "deep_user_pcoc" 
        },
        ]
    ) \
    .enrich_attr_by_lua(
        import_item_attr = [
          "hetu_tag_level_info__hetu_level_one",
        ],
        import_common_attr = [
          "uNebulaXlifeVisitDays30dKV",
          "uNebulaDoubleFindVisitDays30dKV",
          "shallow_consumer_user_pcoc",
          "secondary_shallow_consumer_user_pcoc",
          "moderate_user_pcoc",
          "sub_deep_user_pcoc",
          "deep_user_pcoc",
        ],
        export_item_attr = [
          "active_hetu_pctr_pcoc_score",
        ],
        function_for_item = "get_life_pctr_pcoc_score",
        lua_script_file = "life/ranking/lua/module/ranking_ensemble_sort__score_coeff.lua" 
    )
    return self
  
  def life_ranking_s2_active_hetu_pctr_adjust(self):
    self.calc_by_formula1(
      kconf_key = "formula.scenarioKey84.life_ranking_active_hetu_pctr_adjust",
      import_item_attr = [
        "report_discount",
        "hate_discount",
        "corr_pctr",
        "pltr",
        "pctr",
        "corr_pwtr",
        "pctr_x_pcmtr",
        "plvtr",
        "psvr",
        "pevtr",
        "pptr",
        "pwtr",
        "pcmtr",
        "pepstr",
        "pftr",
        "pdtr",
        "pcltr",
        "phtr",
        "pcmef",
        "active_hetu_pctr_pcoc_score",
      ],
      export_formula_value = [
        "score_pctr"
      ],
      abtest_biz_name = "KUAISHOU_APPS"
    )
    return self
