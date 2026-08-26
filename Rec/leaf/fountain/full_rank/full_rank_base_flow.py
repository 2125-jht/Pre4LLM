#!/usr/bin/env python3
# coding=utf-8

from dragonfly.common_leaf_dsl import LeafFlow
from dragonfly.ext.subdivision.subdivision_api_mixin import subdivisionApiMixin
from dragonfly.ext.retrieval.retrieval_api_mixin import RetrievalApiMixin
from dragonfly.ext.explore.explore_api_mixin import ExploreApiMixin
from dragonfly.ext.merchant.merchant_api_mixin import MerchantApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from dragonfly.ext.embed_calc.embed_calc_api_mixin import EmbedCalcApiMixin
from full_rank.fullrank_base_features import *
from full_rank.fullrank_base_queues import *
from dump_attr_to_kafka import dump_attr_to_kafka

class FullRankBaseFlow(LeafFlow, subdivisionApiMixin, ExploreApiMixin, RetrievalApiMixin, MerchantApiMixin, GsuApiMixin, EmbedCalcApiMixin):
  def prepare_reco_photo_info(self):
    self \
    .enrich_attr_by_lua(
      import_item_attr = ["reason"],
      export_item_attr = ["reason_str"],
      function_for_item = "trans_reason_to_str",
      lua_script_file = "fountain/full_rank/lua/trans_reason_to_str.lua"
    ) \
    .if_("fountain_fullrank_build_reco_photo_with_reason == 0") \
      .build_protobuf(
        class_name = "ks.reco.RecoPhotoInfo",
        inputs = [
          {
            "attr_name": "cascade_pctr",
            "path": "context_info.cascade_pctr"
          },
          {
            "attr_name": "cascade_pltr",
            "path": "context_info.cascade_pltr"
          },
          {
            "attr_name": "cascade_pwtr",
            "path": "context_info.cascade_pwtr"
          },
          {
            "attr_name": "cascade_plvtr",
            "path": "context_info.cascade_plvtr"
          },
          {
            "attr_name": "cascade_psvtr",
            "path": "context_info.cascade_psvr"
          },
          {
            "attr_name": "item_id",
            "path": "ar_result.pid"
          },
          {
            "attr_name": "content_safety_level_with_namespace__level_hot_online",
            "path": "ar_result.content_safety_level"
          },
          {
            "attr_name": "reason_str",
            "path": "reason"
          },
          {
            "attr_name": "cascade_pftr",
            "path": "context_info.cascade_pftr"
          },
          {
            "attr_name": "cascade_pepstr",
            "path": "context_info.cascade_pepstr"
          },
          {
            "attr_name": "cascade_pcmtr",
            "path": "context_info.cascade_pcmtr"
          },
          {
            "attr_name": "cascade_phtr",
            "path": "context_info.cascade_phtr"
          },
          {
            "attr_name": "cascade_pctr_index",
            "path": "cascade_pctr_index"
          },
          {
            "attr_name": "cascade_plvtr_index",
            "path": "cascade_plvtr_index"
          },
          {
            "attr_name": "cascade_pvtr_index",
            "path": "cascade_pvtr_index"
          },
          {
            "attr_name": "cascade_pltr_index",
            "path": "cascade_pltr_index"
          },
          {
            "attr_name": "cascade_pftr_index",
            "path": "cascade_pftr_index"
          },
          {
            "attr_name": "cascade_pwtr_index",
            "path": "cascade_pwtr_index"
          },
          {
            "attr_name": "cascade_pesptr_index",
            "path": "cascade_pesptr_index"
          },
          {
            "attr_name": "cascade_psvr_index",
            "path": "cascade_psvr_index"
          }
        ],
        is_common_attr = False,
        output_attr = "reco_photo_info",
      ) \
    .else_() \
      .build_protobuf(
        class_name = "ks.reco.RecoPhotoInfo",
        inputs = [
          {
            "attr_name": "cascade_pctr",
            "path": "context_info.cascade_pctr"
          },
          {
            "attr_name": "cascade_pltr",
            "path": "context_info.cascade_pltr"
          },
          {
            "attr_name": "cascade_pwtr",
            "path": "context_info.cascade_pwtr"
          },
          {
            "attr_name": "cascade_pftr",
            "path": "context_info.cascade_pftr"
          },
          {
            "attr_name": "cascade_plvtr",
            "path": "context_info.cascade_plvtr"
          },
          {
            "attr_name": "cascade_psvtr",
            "path": "context_info.cascade_psvr"
          },
          {
            "attr_name": "item_id",
            "path": "ar_result.pid"
          },
          {
            "attr_name": "content_safety_level_with_namespace__level_hot_online",
            "path": "ar_result.content_safety_level"
          },
          {
            "attr_name": "reason_str",
            "path": "reason"
          }
        ],
        is_common_attr = False,
        output_attr = "reco_photo_info",
      ) \
    .end_if_()
    return self

  def gen_min_wt_rank_reci(self):
    """
    精排生成最小wt rank
    """
    self \
    .sort(
      score_from_attr = "fullrank_trans_pvtr_score",
    ) \
    .copy_item_meta_info(
      save_item_seq_to_attr = "fullrank_trans_pvtr_rank",
    ) \
    .sort(
      score_from_attr = "fullrank_sim_pvtr",
    ) \
    .copy_item_meta_info(
      save_item_seq_to_attr = "fullrank_sim_pvtr_rank",
    ) \
    .sort(
      score_from_attr = "fullrank_sim_pevtr",
    ) \
    .copy_item_meta_info(
      save_item_seq_to_attr = "fullrank_sim_pevtr_rank",
    ) \
    .sort(
      score_from_attr = "fullrank_detail_new_pevtr_v2",
    ) \
    .copy_item_meta_info(
      save_item_seq_to_attr = "fullrank_detail_new_pevtr_v2_rank",
    ) \
    .sort(
      score_from_attr = "fullrank_sim_plvtr",
    ) \
    .copy_item_meta_info(
      save_item_seq_to_attr = "fullrank_sim_plvtr_rank",
    ) \
    .sort(
      score_from_attr = "fullrank_sim_pfintr",
    ) \
    .copy_item_meta_info(
      save_item_seq_to_attr = "fullrank_sim_pfintr_rank",
    ) \
    .sort(
      score_from_attr = "fullrank_ltr_v4_fountain_finish_rate",
    ) \
    .copy_item_meta_info(
      save_item_seq_to_attr = "fullrank_ltr_v4_fountain_finish_rate_rank",
    ) \
    .sort(
      score_from_attr = "fullrank_ltr_v4_fountain_next",
    ) \
    .copy_item_meta_info(
      save_item_seq_to_attr = "fullrank_ltr_v4_fountain_next_rank",
    ) \
    .split_string(
      input_common_attr = "fountain_fullrank_min_wt_rank_weights_str",
      output_common_attr = "fountain_fullrank_min_wt_rank_weights",
      delimiters = ":",
      parse_to_double = True,
    ) \
    .enrich_attr_by_lua(
      import_common_attr = [
        "fountain_fullrank_min_wt_rank_weights",
      ],
      import_item_attr = [
        "fullrank_trans_pvtr_rank", # no use in single rank
        "fullrank_sim_pvtr_rank", # no use in single rank
        "fullrank_sim_pevtr_rank", # ctr in actiononce,no use in single rank
        "fullrank_detail_new_pevtr_v2_rank",
        "fullrank_sim_plvtr_rank",
        "fullrank_sim_pfintr_rank",
        "fullrank_ltr_v4_fountain_finish_rate_rank",
        "fullrank_ltr_v4_fountain_next_rank",
        "fullrank_like_rank",
        "fullrank_follow_rank",
        "fullrank_cmtr_rank",
      ],
      export_item_attr = [
        "fullrank_min_wt_rank_reci"
      ],
      function_for_item = "calc_min_wt_rank_reci",
      lua_script_file = "fountain/full_rank/lua/calc_min_act_rank_reci.lua",
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
      lua_script_file = "fountain/full_rank/lua/calc_min_act_rank_reci.lua",
    )
    return self

  def gen_high_multiply_score(self):
    """
    精排生成互动时长高级队列
    """
    self \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "high_multiply_ctr_beta", "as": "ctr_beta"},
        {"name": "high_multiply_wtr_beta", "as": "wtr_beta"},
        {"name": "high_multiply_ltr_beta", "as": "ltr_beta"},
        {"name": "high_multiply_ftr_beta", "as": "ftr_beta"},
        {"name": "high_multiply_cmtr_beta", "as": "cmtr_beta"},
        {"name": "high_multiply_cltr_beta", "as": "cltr_beta"},
        {"name": "high_multiply_lvtr_beta", "as": "lvtr_beta"},
        {"name": "high_multiply_fintr_beta", "as": "fintr_beta"},
        {"name": "high_multiply_pcpr_beta", "as": "pcpr_beta"}
      ],
      import_item_attr = [
        {"name": "fullrank_sim_click_score", "as": "ctr"},
        {"name": "fullrank_sim_pwtr", "as": "wtr"},
        {"name": "fullrank_sim_like_score", "as": "ltr"},
        {"name": "fullrank_sim_pftr", "as": "ftr"},
        {"name": "fullrank_sim_pcmtr", "as": "cmtr"},
        {"name": "fullrank_sim_pcltr", "as": "cltr"},
        {"name": "fullrank_sim_plvtr", "as": "lvtr"},
        {"name": "fullrank_sim_pfintr", "as": "fintr"},
        {"name": "fullrank_sim_pcpr", "as": "pcpr"}
        ],
      export_item_attr = [
        "high_multiply_score",
      ],
      function_name = "CalHighMultiplyScore",
      class_name = "ExploreLightFunctionSetV2",
    )

    return self

  def gen_pcmef_gender_debias_score(self):
    """
    精排 pcmef_gender_debias 队列
    """
    self \
    .enrich_with_protobuf(
      from_extra_var = "userInfoPb",
      attrs = [
        dict(name="user_gender", path="gender"),
      ]
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "user_gender", "as": "gender"},
      ],
      import_item_attr = [
        "hetu_tag_level_info__hetu_level_one"
      ],
      export_item_attr = [
        "pcmef_debias_bucket_name",
      ],
      function_name = "CalCmefDebiasBucketName",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .get_kconf_params(
      kconf_configs = [{
        "kconf_key": "reco.hot.cmef_gender_hetu_debias_json_ft",
        "value_type": "double",
        "json_path": "{{pcmef_debias_bucket_name}}",
        "default_value": 1.83565643,
        "export_item_attr": "pcmef_debias_bucket_score"
      }]
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "fullrank_sim_pcmef", "as": "cmef"},
        {"name": "fullrank_sim_click_score", "as": "ctr"},
        {"name": "pcmef_debias_bucket_score", "as": "pcmef_debias_bucket_score"},
      ],
      export_item_attr = [
        "pcmef_debias_score",
      ],
      function_name = "CalCmefDebiasScore",
      class_name = "ExploreLightFunctionSetV2",
    )

    return self

  def fullrank_get_evtr_multiply_time_score(self):
    self \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "fullrank_sim_pfintr", "as": "score"},
        {"name": "fullrank_detail_new_pevtr_v2", "as": "boost_discount_coeff"},
      ],
      export_item_attr = [
        {"name": "score", "as": "fullrank_evtr_multiply_playtime"},
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "duration_ms", "as": "duration"},
        {"name": "fullrank_detail_new_pevtr_v2", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "fullrank_evtr_multiply_duration"},
      ],
      function_name = "CalEvtrMulDuration",
      class_name = "ExploreLightFunctionSetV2",
    )

    return self

  def fullrank_get_xtr_fractile_score(self):
    self \
    .explore_absolute_xtr_score_que_enricher(
      explore_absolute_xtr_boost_threshold = "{{fountain_fr_pxtr_fractile_boost_threshold}}",
      explore_absolute_xtr_boost_weight = "{{fountain_fr_pxtr_fractile_boost_weight}}",
      enable_explore_absolute_xtr_cliff = "{{enable_fountain_fr_pxtr_fractile_cliff}}",
      pxtr_fractile_kconf_path = "reco.hot.FountainFullrankPxtrFractile",
      absolute_xtr_score_que_attr = "fr_pxtr_fractile_score",
      queues = fr_pxtr_fractile_score_queues
    )

    return self

  def fullrank_calc_duration_xtr_debias_score(self):
    self \
    .explore_duration_xtr_debias_enricher(
      duration_xtr_debias_user_add = "{{fountain_fr_duration_xtr_debias_user_add}}",
      duration_xtr_debias_window_size = "{{fountain_fr_duration_xtr_debias_window_size}}",
      enable_use_bidirectional_window = "{{enable_fountain_fullrank_duration_xtr_debias_use_bidirectional_window}}",
      que_score_type = "{{fountain_fr_duration_xtr_que_score_type}}",
      duration_xtr_debias_score_que_attr = "fr_duration_xtr_bias_score",
      queues = fr_duration_xtr_bias_score_queues
    )

    return self


  def fullrank_get_xtr_fractile_score_by_memdata(self):
    self \
    .explore_memory_data_enrich(
      data_key = "xtr_fractile_score_map",
      data_type = "string_double_vector_map",
      save_data_ptr_to_attr = "xtr_fractile_score_attr_from_redis_ptr",
    ) \
    .explore_calc_fractile_score_by_multiple_bucket_enricher(
      user_info_ptr_attr = "userInfoPb",
      pxtr_fractile_redis_attr = "xtr_fractile_score_attr_from_redis_ptr",
      use_hetu_level_one = "{{fountain_fullrank_calc_fractile_score_use_hetu_level_one}}",
      hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
      use_gender_multi_age_frac = "{{fountain_fullrank_use_gender_multi_age_frac}}",
      queues = fr_pxtr_fractile_score_queues
    )

    return self

  def fullrank_calc_pairwise_rank_score(self):
    self \
    .explore_calc_pairwise_rank_score_enricher(
      pairwise_score_helper_conf_path = "reco.exploreRank.recoFtFrWtdPairScoreMap",
      smooth = "{{fountain_fullrank_cal_pairwise_rank_score_smooth}}",
      wtd_attr = "fullrank_sim_pfintr",
      pairwise_rank_score_attr = "fullrank_pairwise_rank_score",
      pairwise_rank_raw_score_attr = "fullrank_pairwise_rank_raw_score" 
    ) \

    return self
  
  def cal_user_group_dynamic_weight(self):
    self \
    .gen_common_attr_by_lua(
      attr_map={
        "fountain_ensemble_power_weight_fullrank_like_score": "fountain_ensemble_power_weight_fullrank_like_score * user_group_emp_ltr",
        "fountain_ensemble_power_weight_fullrank_follow_score": "fountain_ensemble_power_weight_fullrank_follow_score * user_group_emp_wtr",
        "fountain_ensemble_power_weight_fullrank_pcmtr_score": "fountain_ensemble_power_weight_fullrank_pcmtr_score * user_group_emp_cmtr",
        "fountain_ensemble_weight_forward_score": "fountain_ensemble_weight_forward_score * user_group_emp_ftr",
      }
    )
    return self

  def cal_user_group_emp_xtr_all(self):
    self \
    .gen_common_attr_by_lua(
      attr_map={
        "fountain_ensemble_power_weight_fullrank_like_emp": "fountain_ensemble_power_weight_fullrank_like_emp * user_group_emp_ltr",
        "fountain_ensemble_power_weight_fullrank_follow_emp": "fountain_ensemble_power_weight_fullrank_follow_emp * user_group_emp_wtr",
        "fountain_ensemble_power_weight_fullrank_forward_emp": "fountain_ensemble_power_weight_fullrank_forward_emp * user_group_emp_ftr",
        "fountain_ensemble_power_weight_fullrank_pcmtr_emp": "fountain_ensemble_power_weight_fullrank_pcmtr_emp * user_group_emp_cmtr",
        "fountain_ensemble_power_weight_fullrank_pptr_emp": "fountain_ensemble_power_weight_fullrank_pptr_emp * user_group_emp_ptr",
      }
    )
    return self

  def fullrank_calc_ltr_pairwise_rank_score(self):
    self \
    .get_kconf_params(
      kconf_configs = [
        {
          "kconf_key": "reco.exploreRank.recoFtFrXtrFractScoreMap",
          "value_type": "json",
          "json_path": "ltrSeg",
          "export_common_attr": "explore_fr_cal_ctr_pairwise_rank_score_conf"
        }
      ]
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "fullrank_sim_like_score", "as": "score"},
      ],
      import_common_attr = [
        {"name": "explore_fr_cal_ctr_pairwise_rank_score_conf", "as": "frac_conf"}
      ],
      export_item_attr = [
        {"name": "score", "as": "ltr_frac_score"},
      ],
      function_name = "GetFracScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "ltr_frac_score", "as": "score"},
      ],
      import_common_attr = [
        {"name": "explore_fr_cal_ctr_pairwise_rank_score_alpha", "as": "boost_discount_coeff"}
      ],
      export_item_attr = [
        {"name": "score", "as": "ltr_for_pairwise_rank_score"},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .explore_calc_pairwise_rank_score_enricher(
      pairwise_score_helper_conf_path = "reco.exploreRank.recoFtFrLtrPairScoreMap",
      smooth = "{{fountain_fullrank_cal_ltr_pairwise_rank_score_smooth}}",
      max_key = "{{fountain_fullrank_cal_ltr_pairwise_rank_score_max_key}}",
      wtd_attr = "ltr_for_pairwise_rank_score",
      pairwise_rank_score_attr = "fullrank_ltr_pairwise_rank_score",
      pairwise_rank_raw_score_attr = "fullrank_ltr_pairwise_rank_raw_score" 
    ) \

    return self

  def fullrank_calc_wtd_sharpe_ratio_score(self):
    self \
    .calc_weighted_sum(
      channels = [
        {"name": "fullrank_ltr_lph", "weight": "{{fountain_fr_cal_sharpe_ratio_score_ltr_lph_weight}}"},
        {"name": "fullrank_sim_pfintr", "weight": "{{fountain_fr_cal_sharpe_ratio_score_wtd_weight}}"},
      ],
      output_item_attr = "linear_score_of_lph_fintr",
    ) \
    .item_attr_operation(
      item_attr_a = "linear_score_of_lph_fintr",
      item_attr_b = "fullrank_sim_pltr",
      operator = "*",
      output_attr = "linear_score_of_lph_fintr_multiply_ltr",
    ) \
    .pack_item_attr(
      item_source = {
        "reco_results": True,
      },
      mappings = [
        {
          "aggregator": "avg",
          "from_item_attr": "linear_score_of_lph_fintr_multiply_ltr",
          "to_common_attr": "linear_score_of_lph_fintr_multiply_ltr_avg"
        },
      ]
    ) \
    .explore_calc_sharpe_ratio_score_enricher(
      mean_conf_path = "reco.exploreRank.recoFountainFrWtdMeanMap",
      std_conf_path = "reco.exploreRank.recoFountainFrWtdStdMap",
      ctr_attr = "fullrank_sim_pltr",
      xtr_attr = "linear_score_of_lph_fintr",
      risk_free_attr = "{{fountain_fr_cal_sharpe_ratio_score_risk_free}}",
      request_risk_free_attr = "{{linear_score_of_lph_fintr_multiply_ltr_avg}}",
      std_beta_attr = "{{fountain_fr_cal_sharpe_ratio_score_std_beta}}",
      use_raw_attr = "{{fountain_fr_cal_sharpe_ratio_score_use_raw}}",
      global_rf_weight_attr = "{{fountain_fr_cal_sharpe_ratio_score_global_risk_free_weight}}",
      request_rf_weight_attr = "{{fountain_fr_cal_sharpe_ratio_score_request_risk_free_weight}}",
      sharpe_ratio_score_attr = "fountain_fr_wtd_sharpe_ratio_score"
    ) \

    return self

  def fullrank_cal_svtr_adapt_wtd_score(self):
    self \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_svtr_adapt_wtd_threshold",  "as": "threshold"},
        {"name": "fountain_svtr_adapt_wtd_beta",  "as": "beta"},
      ],
      import_item_attr = [
        {"name": "fullrank_sim_psvr",  "as": "svtr"},
        {"name": "fullrank_sim_pfintr",  "as": "wtd"},
      ],
      export_item_attr = [
        {"name": "output_score",  "as": "svtr_adapt_wtd_score"},
      ],
      function_name = "CalcSvtrAdaptWtdScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \

    return self

  def fullrank_calc_fractile_fusion_score(self):
    self \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "fullrank_sim_pwatchtime_no_bias_fractile_score",  "as": "pvtr_fractile_score"},
        {"name": "fullrank_sim_plvtr_fractile_score",  "as": "plvtr_fractile_score"},
        {"name": "fullrank_sim_pfintr_fractile_score",  "as": "pwtd_fractile_score"},
        {"name": "fullrank_sim_click_score_fractile_score",  "as": "pctr_fractile_score"},
        {"name": "fullrank_sim_pevtr_fractile_score",  "as": "pevtr_fractile_score"},
        {"name": "fullrank_sim_pltr_fractile_score",  "as": "pltr_fractile_score"},
        {"name": "fullrank_sim_pwtr_fractile_score",  "as": "pwtr_fractile_score"},
        {"name": "fullrank_sim_pftr_fractile_score",  "as": "pftr_fractile_score"},
        {"name": "fullrank_sim_pcmtr_fractile_score",  "as": "pcmtr_fractile_score"},
        {"name": "fullrank_sim_pepstr_fractile_score",  "as": "pepstr_fractile_score"},
      ],
      import_common_attr = [
        {"name": "fountain_fr_pxtr_fractile_ensemble_wtd_wgt",  "as": "pvtr_fractile_score_weight"},
        {"name": "fountain_fr_pxtr_fractile_ensemble_lvtr_wgt",  "as": "plvtr_fractile_score_weight"},
        {"name": "fountain_fr_pxtr_fractile_ensemble_fintr_wgt",  "as": "pwtd_fractile_score_weight"},
        {"name": "fountain_fr_pxtr_fractile_ensemble_ctr_wgt",  "as": "pctr_fractile_score_weight"},
        {"name": "fountain_fr_pxtr_fractile_ensemble_pevtr_wgt",  "as": "pevtr_fractile_score_weight"},
        {"name": "fountain_fr_pxtr_fractile_ensemble_ltr_wgt",  "as": "pltr_fractile_score_weight"},
        {"name": "fountain_fr_pxtr_fractile_ensemble_wtr_wgt",  "as": "pwtr_fractile_score_weight"},
        {"name": "fountain_fr_pxtr_fractile_ensemble_ftr_wgt",  "as": "pftr_fractile_score_weight"},
        {"name": "fountain_fr_pxtr_fractile_ensemble_cmtr_wgt",  "as": "pcmtr_fractile_score_weight"},
        {"name": "fountain_fr_pxtr_fractile_ensemble_pepstr_wgt",  "as": "pepstr_fractile_score_weight"},
        "enable_fullrank_fractile_score_sum_fusion",
      ],
      export_item_attr = [
        "fullrank_fractile_fusion_score"
      ],
      function_name = "CalFractileFusionScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    
    return self

  def fullrank_calc_confidence_pxtr_fusion_score(self):
    self \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "fullrank_detail_new_pevtr_v2", "as": "pevtr"},
        {"name": "fullrank_sim_pltr", "as": "pltr"},
        {"name": "fullrank_sim_pwtr", "as": "pwtr"},
        {"name": "fullrank_sim_pcmtr", "as": "pcmtr"},
        {"name": "fullrank_sim_pftr", "as": "pftr"},
        {"name": "fullrank_sim_pfintr", "as": "pwtd"},
      ],
      import_common_attr = [
        {"name": "fountain_fr_confidence_pxtr_fusion_pevtr_upper_bound",  "as": "pevtr_upper_bound"},
        {"name": "fountain_fr_confidence_pxtr_fusion_pevtr_lower_bound",  "as": "pevtr_lower_bound"},
        {"name": "fountain_fr_confidence_pxtr_fusion_pltr_upper_bound",  "as": "pltr_upper_bound"},
        {"name": "fountain_fr_confidence_pxtr_fusion_pltr_lower_bound",  "as": "pltr_lower_bound"},
        {"name": "fountain_fr_confidence_pxtr_fusion_pwtr_upper_bound",  "as": "pwtr_upper_bound"},
        {"name": "fountain_fr_confidence_pxtr_fusion_pwtr_lower_bound",  "as": "pwtr_lower_bound"},
        {"name": "fountain_fr_confidence_pxtr_fusion_pftr_upper_bound",  "as": "pftr_upper_bound"},
        {"name": "fountain_fr_confidence_pxtr_fusion_pftr_lower_bound",  "as": "pftr_lower_bound"},
        {"name": "fountain_fr_confidence_pxtr_fusion_pcmtr_upper_bound",  "as": "pcmtr_upper_bound"},
        {"name": "fountain_fr_confidence_pxtr_fusion_pcmtr_lower_bound",  "as": "pcmtr_lower_bound"},
        {"name": "fountain_fr_confidence_pxtr_fusion_pwtd_upper_bound",  "as": "pwtd_upper_bound"},
        {"name": "fountain_fr_confidence_pxtr_fusion_pwtd_lower_bound",  "as": "pwtd_lower_bound"},
      ],
      export_item_attr = [
        "fullrank_confidence_pxtr_fusion_score"
      ],
      function_name = "CalConfidencePxtrFusionScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \

    return self


  def enrich_fullrank_features_by_lua(self):
    """
    精排模型特征填充
    """
    self\
    .explore_common_user_feature_enricher(
      skip = "{{fountain_skip_fullrank_deep_ltr_use_fountain_session_realshow_features}}",
      user_info_attr = "userInfoPb",
      fountain_session_realshow_attr = "userFountainSessionRealshowList",
      source_pid_attr = "featureSourcePId",
    ) \
    .enrich_attr_by_lua(
      skip = "{{fountain_skip_fullrank_lua_feature_trans}}",
      import_common_attr = [
        "currentTimeMs"
      ],
      import_item_attr = [
        "upload_time",
        "explore_stat__click_count",
        "explore_stat__like_count",
        "explore_stat__follow_count",
        "explore_stat__forward_count",
        "explore_stat__long_play_count",
        "explore_stat__real_show_count",
        "explore_stat__short_play_count",
        "explore_stat__view_length_sum",
        "author__exp_stat__exp_click",
        "author__exp_stat__exp_like",
        "author__exp_stat__exp_follow",
        "author__exp_stat__exp_long_view",
        "author__exp_stat__exp_realshow",
        "author__exp_stat__exp_forward",
        "author__exp_stat__exp_short_view",
        "author__exp_stat__exp_watch_time",
        "hetu_tag_level_info__hetu_level_one",
        "hetu_tag_level_info_v2__hetu_level_one",
        "hetu_tag_level_info_v2__hetu_level_two",
        "hetu_tag_level_info_v2__hetu_level_three",
        "hetu_level_one_v2",
        "duration_ms"
      ],
      export_item_attr = [
        "featurePUploadTimeDiff",
        "featurePHotClickCount",
        "featurePHotLikeCount",
        "featurePHotFollowCount",
        "featurePHotLongViewCount",
        "featurePHotCtr",
        "featurePHotLtr",
        "featurePHotWtr",
        "featurePHotFtr",
        "featurePHotLvtr",
        "featurePHotSvtr",
        "featurePHotAvgWatchTime",
        "featurePAClickCount",
        "featurePALikeCount",
        "featurePAFollowCount",
        "featurePALongViewCount",
        "featurePACtr",
        "featurePALtr",
        "featurePAWtr",
        "featurePAFtr",
        "featurePALvtr",
        "featurePASvtr",
        "featurePAAvgWatchTime",
        "featurePHetu0",
        "hetu_level_one_tag_index",
        "hetu_level_two_tag_index",
        "hetu_level_three_tag_index",
        "hetu_level_one_v2_index",
        "fountainDurationPercent",
        "fullrank_empirical_ctr",
        "fullrank_empirical_ltr",
        "fullrank_empirical_wtr",
        "fullrank_empirical_ftr",
        "fullrank_empirical_ptr",
        "fullrank_empirical_cmtr",
        "fullrank_empirical_htr",
        "fullrank_empirical_watchtime"
      ],
      function_for_item = "fullrank_feature_trans",
      lua_script_file = "fountain/full_rank/lua/fullrank_feature_trans.lua",
    ) \
    .if_("fountain_fast_get_simple_ltr_feature == 1 and (fountain_sl_only_fast_v1 == 0 or (fountain_sl_only_fast_v1 == 1 and page ~= nil and page > 1))") \
      .get_simple_ltr_feature() \
    .end_()

    return self
  
  def fullrank_calc_fractile_sum_score(self):
    self \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "fullrank_sim_pwatchtime_no_bias_fractile_score",  "as": "pvtr_fractile_score"},
        {"name": "fullrank_sim_plvtr_fractile_score",  "as": "plvtr_fractile_score"},
        {"name": "fullrank_sim_pfintr_fractile_score",  "as": "pwtd_fractile_score"},
        {"name": "fullrank_sim_click_score_fractile_score",  "as": "pctr_fractile_score"},
        {"name": "fullrank_sim_pevtr_fractile_score",  "as": "pevtr_fractile_score"},
        {"name": "fullrank_sim_pltr_fractile_score",  "as": "pltr_fractile_score"},
        {"name": "fullrank_sim_pwtr_fractile_score",  "as": "pwtr_fractile_score"},
        {"name": "fullrank_sim_pftr_fractile_score",  "as": "pftr_fractile_score"},
        {"name": "fullrank_sim_pcmtr_fractile_score",  "as": "pcmtr_fractile_score"},
        {"name": "fullrank_sim_pepstr_fractile_score",  "as": "pepstr_fractile_score"},
      ],
      import_common_attr = [
        {"name": "fountain_fr_fractile_sum_score_pvtr_wgt",  "as": "pvtr_fractile_score_weight"},
        {"name": "fountain_fr_fractile_sum_score_lvtr_wgt",  "as": "plvtr_fractile_score_weight"},
        {"name": "fountain_fr_fractile_sum_score_fintr_wgt",  "as": "pwtd_fractile_score_weight"},
        {"name": "fountain_fr_fractile_sum_score_ctr_wgt",  "as": "pctr_fractile_score_weight"},
        {"name": "fountain_fr_fractile_sum_score_pevtr_wgt",  "as": "pevtr_fractile_score_weight"},
        {"name": "fountain_fr_fractile_sum_score_ltr_wgt",  "as": "pltr_fractile_score_weight"},
        {"name": "fountain_fr_fractile_sum_score_wtr_wgt",  "as": "pwtr_fractile_score_weight"},
        {"name": "fountain_fr_fractile_sum_score_ftr_wgt",  "as": "pftr_fractile_score_weight"},
        {"name": "fountain_fr_fractile_sum_score_cmtr_wgt",  "as": "pcmtr_fractile_score_weight"},
        {"name": "fountain_fr_fractile_sum_score_pepstr_wgt",  "as": "pepstr_fractile_score_weight"},
        {"name": "fountain_fr_fractile_score_sum_fusion", "as": "enable_fullrank_fractile_score_sum_fusion"}
      ],
      export_item_attr = [
        {"name": "fullrank_fractile_fusion_score", "as": "fullrank_fractile_sum_score"}
      ],
      function_name = "CalFractileFusionScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    
    return self

  def enrich_fullrank_score_attr_splash(self):
    self \
    .count_reco_result(
      save_count_to="rank_splash_model_input_count"
    ) \
    .if_("skip_fountain_rank_splash_server == 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .delegate_enrich(  # 首屏请求
        kess_service = "{{fountain_rank_splash_kess_service}}",
        partition_size = "{{fountain_rank_splash_partition_size}}",
        recv_item_attrs = [
          { "name": "ctr", "as": "fullrank_sim_out_pctr" },
          { "name": "ltr", "as": "fullrank_sim_pltr" },
          { "name": "wtr", "as": "fullrank_sim_pwtr" },
          { "name": "ftr", "as": "fullrank_sim_pftr" },
          { "name": "svr", "as": "fullrank_sim_psvr" },
          { "name": "lvtr", "as": "fullrank_sim_plvtr" },
          { "name": "cmtr", "as": "fullrank_sim_pcmtr" },
          { "name": "ptr", "as": "fullrank_sim_pptr" },
          { "name": "cmef", "as": "fullrank_sim_pcmef" },
          { "name": "htr", "as": "fullrank_sim_phtr" },
          { "name": "evtr", "as": "fullrank_sim_pevtr" },
          { "name": "vtr", "as": "fullrank_sim_pvtr" },
          { "name": "wtd_playtime", "as": "fullrank_sim_pwtd_playtime" },
          { "name": "epstr", "as": "fullrank_sim_pepstr" },
          { "name": "fintr", "as": "fullrank_sim_pfintr" },
          { "name": "cltr", "as": "fullrank_sim_pcltr" },
          { "name": "cpr", "as": "fullrank_sim_pcpr" },
          { "name": "wtd_v2_playtime", "as": "fullrank_sim_pwtd_v2_playtime" },
          { "name": "playtime_finish", "as": "fullrank_sim_playtime_finish" },
          { "name": "ordinal_playtime", "as": "fullrank_sim_ordinal_playtime" },
          { "name": "lstr", "as": "fullrank_sim_lstr" },
          { "name": "swptr", "as": "fullrank_ori_pswptr" },
          { "name": "fountain_evtr_v2", "as": "fullrank_detail_new_pevtr_v2" },
          { "name":"slide", "as":"fountain_splash_slide" },
          { "name":"evtr_playtime", "as":"fullrank_sim_evtr_playtime" },
          { "name":"evtr_duration", "as":"fullrank_sim_evtr_duration" },
          { "name":"wtd_duration_score", "as":"fullrank_sim_wtd_duration" },
          { "name":"click_live", "as":"fullrank_sim_living_pctr" },
        ],
        timeout_ms = "{{fountain_rank_splash_timeout_ms}}",
        request_type = "fountain_rank_splash",
        send_common_attrs = user_features_v2,
        send_item_attrs = [feature["name"] for feature in photo_features if feature["name"] not in photo_pxtr_features],
      ) \
    .else_() \
      .if_("skip_fountain_fullrank_sim_predict == 0") \
        .prepare_reco_photo_info() \
        .delegate_enrich(
          name = "fountain_fullrank_sim_predict_splash",
          kess_service = "{{fountain_fullrank_sim_predict_kess_service}}",
          partition_size = "{{fountain_fullrank_sim_predict_partition_size}}",
          recv_item_attrs = [
            { "name": "ctr", "as": "fullrank_sim_out_pctr" },
            { "name": "ltr", "as": "fullrank_sim_pltr" },
            { "name": "wtr", "as": "fullrank_sim_pwtr" },
            { "name": "ftr", "as": "fullrank_sim_pftr" },
            { "name": "svr", "as": "fullrank_sim_psvr" },
            { "name": "lvtr", "as": "fullrank_sim_plvtr" },
            { "name": "cmtr", "as": "fullrank_sim_pcmtr" },
            { "name": "ptr", "as": "fullrank_sim_pptr" },
            { "name": "cmef", "as": "fullrank_sim_pcmef" },
            { "name": "htr", "as": "fullrank_sim_phtr" },
            { "name": "evtr", "as": "fullrank_sim_pevtr" },
            { "name": "vtr", "as": "fullrank_sim_pvtr" },
            { "name": "wtd_playtime", "as": "fullrank_sim_pwtd_playtime" },
            { "name": "epstr", "as": "fullrank_sim_pepstr" },
            { "name": "fintr", "as": "fullrank_sim_pfintr" },
            { "name": "cltr", "as": "fullrank_sim_pcltr" },
            { "name": "cpr", "as": "fullrank_sim_pcpr" },
            { "name": "wtd_v2_playtime", "as": "fullrank_sim_pwtd_v2_playtime" },
            { "name": "playtime_finish", "as": "fullrank_sim_playtime_finish" },
            { "name": "ordinal_playtime", "as": "fullrank_sim_ordinal_playtime" },
            # 左滑进入个人页
            { "name": "lstr", "as": "fullrank_sim_lstr" },
            { "name": "lsst", "as": "fullrank_sim_lsst" },
            { "name": "swptr", "as": "fullrank_ori_pswptr" },
            { "name": "fountain_evtr_v2", "as": "fullrank_detail_new_pevtr_v2" },
            { "name":"evtr_playtime", "as":"fullrank_sim_evtr_playtime" },
            { "name":"evtr_duration", "as":"fullrank_sim_evtr_duration" },
            { "name":"wtd_duration_score", "as":"fullrank_sim_wtd_duration" },
            { "name":"click_live", "as":"fullrank_sim_living_pctr" },
            { "name":"etcm", "as":"fullrank_sim_etcm" },
            { "name":"dfvr", "as":"fullrank_sim_pdfvr" },
          ],
          request_type = "{{fountain_fullrank_sim_predict_request_type}}",
          send_common_attrs = [
            { "name": "userInfo", "as": "user_info_str" },
            { "name": "featureSourcePId", "as": "source_photo_id"  },
            { "name": "page", "as": "page_common"  },
          ],
          send_item_attrs = [
            { "name": "reco_photo_info", "as": "reco_photo_info_str" },
          ],
        ) \
      .end_if_() \
    .end_if_()
    return self

  def enrich_fullrank_score_attr_fast(self):
    self \
    .count_reco_result(
      save_count_to="rank_model_input_count"
    ) \
    .prepare_reco_photo_info() \
    .delegate_enrich(
      name = "fountain_fullrank_sim_predict_fast",
      skip = "{{skip_fountain_fullrank_sim_predict}}",
      kess_service = "{{fountain_fullrank_sim_predict_kess_service}}",
      partition_size = "{{fountain_fullrank_sim_predict_partition_size}}",
      recv_item_attrs = [
        { "name": "ctr", "as": "fullrank_sim_out_pctr" },
        { "name": "ltr", "as": "fullrank_sim_pltr" },
        { "name": "wtr", "as": "fullrank_sim_pwtr" },
        { "name": "ftr", "as": "fullrank_sim_pftr" },
        { "name": "svr", "as": "fullrank_sim_psvr" },
        { "name": "lvtr", "as": "fullrank_sim_plvtr" },
        { "name": "cmtr", "as": "fullrank_sim_pcmtr" },
        { "name": "ptr", "as": "fullrank_sim_pptr" },
        { "name": "cmef", "as": "fullrank_sim_pcmef" },
        { "name": "htr", "as": "fullrank_sim_phtr" },
        { "name": "evtr", "as": "fullrank_sim_pevtr" },
        { "name": "vtr", "as": "fullrank_sim_pvtr" },
        { "name": "wtd_playtime", "as": "fullrank_sim_pwtd_playtime" },
        { "name": "epstr", "as": "fullrank_sim_pepstr" },
        { "name": "fintr", "as": "fullrank_sim_pfintr" },
        { "name": "cltr", "as": "fullrank_sim_pcltr" },
        { "name": "cpr", "as": "fullrank_sim_pcpr" },
        { "name": "wtd_v2_playtime", "as": "fullrank_sim_pwtd_v2_playtime" },
        { "name": "playtime_finish", "as": "fullrank_sim_playtime_finish" },
        { "name": "ordinal_playtime", "as": "fullrank_sim_ordinal_playtime" },
        # 左滑进入个人页
        { "name": "lstr", "as": "fullrank_sim_lstr" },
        { "name": "lsst", "as": "fullrank_sim_lsst" },
        { "name": "swptr", "as": "fullrank_ori_pswptr" },
        { "name": "fountain_evtr_v2", "as": "fullrank_detail_new_pevtr_v2" },
        { "name":"evtr_playtime", "as":"fullrank_sim_evtr_playtime" },
        { "name":"evtr_duration", "as":"fullrank_sim_evtr_duration" },
        { "name":"wtd_duration_score", "as":"fullrank_sim_wtd_duration" },
        { "name":"click_live", "as":"fullrank_sim_living_pctr" },
        { "name":"etcm", "as":"fullrank_sim_etcm" },
        { "name":"swpst", "as":"fullrank_sim_swpst" },
        { "name":"dfvr", "as":"fullrank_sim_pdfvr" },
      ],
      request_type = "{{fountain_fullrank_sim_predict_request_type}}",
      send_common_attrs = [
        { "name": "userInfo", "as": "user_info_str" },
        { "name": "featureSourcePId", "as": "source_photo_id"  },
        { "name": "page", "as": "page_common"  },
      ],
      send_item_attrs = [
        { "name": "reco_photo_info", "as": "reco_photo_info_str" },
      ],
    )
    return self

  def trans_sim_pxtr_names(self):
    """
    精排模型组打分名字统一变换
    """
    self \
      .enrich_attr_by_lua(
      import_item_attr = [
        "fullrank_sim_pevtr",
        "fullrank_sim_pltr",
        "fullrank_sim_pwtr",
        "fullrank_sim_pftr",
        "fullrank_sim_plvtr",
        "fullrank_sim_pvtr",
        "fullrank_sim_out_pctr",
        "fullrank_sim_pcmtr",
        "fullrank_sim_pcmef",
        "fullrank_sim_pptr",
        "fullrank_sim_pepstr",
        "fullrank_sim_phtr",
        "fullrank_sim_lstr",
        "fullrank_sim_pcltr",
        "fullrank_sim_pfintr"
      ],
      export_item_attr = [
        "fullrank_detail_pctr",
        "fullrank_detail_pltr",
        "fullrank_detail_pwtr",
        "fullrank_detail_pftr",
        "fullrank_detail_plvtr",
        "fullrank_detail_pvtr",
        "fullrank_detail_psvr",
        "fullrank_detail_pcmtr",
        "fullrank_detail_pcmef",
        "fullrank_detail_pptr",
        "fullrank_detail_pepstr",
        "fullrank_detail_phtr",
        "fullrank_final_lstr",
        "fullrank_sim_click_score",
        "fullrank_sim_like_score",
        "fullrank_sim_follow_score",
        "fullrank_sim_pcltr",
        "fullrank_detail_pwtd"
      ],
      function_for_item = "fullrank_trans_pxtr",
      lua_script_file = "fountain/full_rank/lua/fullrank_trans_pxtr.lua",
    )
    return self

  def replace_sim_pxtr_by_reco_model(self):
    """
    用 reco 基础模型打分替换 sim 模型打分
    """
    self \
      .if_("enable_fountain_reco_base_model_replace_pwtd == 1") \
        .copy_attr(attrs=[{"from_item": "fullrank_reco_base_model_gen_time_wt", "to_item": "fullrank_detail_pwtd"}]) \
      .end_() \
      .if_("enable_fountain_reco_base_model_replace_pctr == 1") \
        .copy_attr(attrs=[{"from_item": "fullrank_reco_base_model_evtr", "to_item": "fullrank_detail_pctr"}]) \
      .end_() \
      .if_("enable_fountain_reco_base_model_replace_plvtr == 1") \
        .copy_attr(attrs=[{"from_item": "fullrank_reco_base_model_lvtr", "to_item": "fullrank_detail_plvtr"}]) \
      .end_() \
      .if_("enable_fountain_reco_base_model_replace_pltr == 1") \
        .copy_attr(attrs=[{"from_item": "fullrank_reco_base_model_ltr", "to_item": "fullrank_detail_pltr"}]) \
      .end_() \
      .if_("enable_fountain_reco_base_model_replace_pwtr == 1") \
        .copy_attr(attrs=[{"from_item": "fullrank_reco_base_model_wtr", "to_item": "fullrank_detail_pwtr"}]) \
      .end_() \
      .if_("enable_fountain_reco_base_model_replace_pftr == 1") \
        .copy_attr(attrs=[{"from_item": "fullrank_reco_base_model_ftr", "to_item": "fullrank_detail_pftr"}]) \
      .end_() \
      .if_("enable_fountain_reco_base_model_replace_pcmtr == 1") \
        .copy_attr(attrs=[{"from_item": "fullrank_reco_base_model_cmtr", "to_item": "fullrank_detail_pcmtr"}]) \
      .end_()
    return self

  def fountain_fullrank_pxtr_calibration(self):
    """
    精排模型打分校准
    """
    self \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_awesome_wtd_calibration_duration", "as": "awesome_wtd_calibration_duration"},
      ],
      import_item_attr = [
        "duration_ms",
        {"name": "fullrank_detail_pwtd", "as": "awesome_wtd"},
      ],
      export_item_attr = [
        {"name": "awesome_wtd", "as": "fullrank_detail_pwtd"},
      ],
      function_name = "PxtrCalibration",
      class_name = "ExploreLightFunctionSetV2",
    )

    return self

  def fullrank_ltr_predict_fast(self):
    """
    仅在非首屏生效的ltr模型，后续仅首屏生效ltr模型可迁移此处
    """
    self \
      .if_("skip_fountain_deep_ltr_predict_fast_v1 == 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .delegate_enrich(
          kess_service = "{{fountain_fullrank_deep_ltr_kess_service}}",
          recv_item_attrs = [
            {"name": "l2r", "as": "fullrank_ltr_score"},
            {"name": "ctr", "as": "fullrank_act_ctr"},
            {"name": "wtd", "as": "fullrank_act_wtd"},
            {"name": "finish_rate", "as": "fullrank_ltr_v4_fountain_finish_rate"},
            {"name": "next", "as": "fullrank_ltr_v4_fountain_next"},
          ],
          timeout_ms = 100,
          send_item_attrs = [feature["name"] for feature in photo_features if feature["name"]],
          send_common_attrs = user_features_v3,
          request_type = "{{fountain_deep_ltr_request_type}}",
          partition_size = "{{fountain_deep_ltr_partition_size}}",
        ) \
      .end_if_()
    return self
  
  def fullrank_ltr_predict_splash(self):
    """
    仅在首屏生效的ltr模型,后续仅首屏生效ltr模型可迁移此处
    """
    self \
      .if_("skip_fountain_deep_ltr_predict_splash == 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .delegate_enrich(
          kess_service = "{{fountain_fullrank_deep_ltr_kess_service}}",
          recv_item_attrs = [
            {"name": "l2r", "as": "fullrank_ltr_score"},
            {"name": "ctr", "as": "fullrank_act_ctr"},
            {"name": "wtd", "as": "fullrank_act_wtd"},
            {"name": "finish_rate", "as": "fullrank_ltr_v4_fountain_finish_rate"},
            {"name": "next", "as": "fullrank_ltr_v4_fountain_next"},
          ],
          timeout_ms = 100,
          send_item_attrs = [feature["name"] for feature in photo_features if feature["name"]],
          send_common_attrs = user_features_v3,
          request_type = "{{fountain_deep_ltr_request_type}}",
          partition_size = "{{fountain_deep_ltr_partition_size}}",
        ) \
      .end_if_() \
      .if_("enable_fountain_splash_fullrank_ltr == 1") \
        .explore_custom_trim_user_info(
          user_info_attr = "userInfo",
          save_trimed_user_info_to_attr = "fr_deep_ltr_trimmed_user_info",
          trim_user_info = [
            "active_days",
            "basic_info.age_segment",
            "location.city_id",
            "location.region_type",
            "client_id",
            "device_id",
            "gender",
            "infer_gender",
            "true_gender",
            "request_location.poi_type",
            "request_location.province_id",
            "request_location.city_id",
            "visit_mod",
            "upload_count",
            "infer_year",
            "follow_count",
            "fans_count",
            "visit_net",
            "location.city_level",
            "is_douyin",
            "feature_collection.explore_low_active_level",
            "user_profile.exp_stat.exp_click",
            "user_profile.exp_stat.exp_like",
            "user_profile.exp_stat.exp_follow",
            "user_profile.exp_stat.exp_realshow",
            "user_profile.exp_stat.exp_long_view",
            "user_profile.user_level",
            "fountain_reco_user_profile.click_list.author_id",
            "fountain_reco_user_profile.click_list.photo_id",
            "fountain_reco_user_profile.comment_list.author_id",
            "fountain_reco_user_profile.comment_list.photo_id",
            "fountain_reco_user_profile.follow_list.author_id",
            "fountain_reco_user_profile.follow_list.photo_id",
            "fountain_reco_user_profile.like_list.author_id",
            "fountain_reco_user_profile.like_list.photo_id",
            "fountain_reco_user_profile.video_play_stat.photo_id",
            "fountain_reco_user_profile.video_play_stat.author_id",
            "fountain_reco_user_profile.video_play_stat.video_duration",
            "fountain_reco_user_profile.video_play_stat.playing_time",
            "fountain_reco_user_profile.video_play_stat.client_timestamp",
            "user_profile_v1.click_list.author_id",
            "user_profile_v1.click_list.photo_id",
            "user_profile_v1.follow_list.author_id",
            "user_profile_v1.follow_list.photo_id",
            "user_profile_v1.like_list.author_id",
            "user_profile_v1.like_list.photo_id",
            "user_profile_v1.hate_list.photo_id",
            "user_profile_v1.video_playing_stat.playing_time",
            "user_profile_v1.video_playing_stat.author_id",
            "user_profile_v1.video_playing_stat.photo_id",
            "user_profile_v1.video_playing_stat.client_timestamp",
            "realtime_click_list",
            "realtime_follow_list",
            "realtime_forward_list",
            "realtime_like_list",
            "user_profile_v1.real_show_list.photo_id",
            "user_profile_v1.real_show_list.author_id",
            "user_profile_v1.real_show_list.time_ms",
            "user_profile_v1.real_show_list.page_type",
            "user_profile_v1.real_show_list.label.click",
            "user_profile_v1.real_show_list.label.like",
            "user_profile_v1.real_show_list.label.follow",
            "user_profile_v1.real_show_list.label.hate",
            "user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_one",
            "user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_two",
            "user_profile_v1.like_list.hetu_tag_level_info.hetu_level_one",
            "user_profile_v1.like_list.hetu_tag_level_info.hetu_level_two",
          ],
        ) \
        .delegate_enrich(
          name = "fountain_splash_fullrank_deep_ltr",
          kess_service = "{{fountain_splash_fullrank_deep_ltr_kess_service}}",
          recv_item_attrs = [
            {"name": "multi_rank_score", "as": "splash_fullrank_ltr_fusion_score"},
            {"name": "pact", "as": "splash_fullrank_ltr_act_score"},
            {"name": "pact_v2", "as": "splash_fullrank_ltr_act_v2_score"},
            {"name": "pwtd", "as": "splash_fullrank_ltr_wtd_score"},
            {"name": "plvtr", "as": "splash_fullrank_ltr_lvtr_score"},
            {"name": "psvtr", "as": "splash_fullrank_ltr_svtr_score"},
            {"name": "pltr", "as": "splash_fullrank_ltr_like_score"},
            {"name": "pwtr", "as": "splash_fullrank_ltr_follow_score"},
            {"name": "pcmtr", "as": "splash_fullrank_ltr_comment_score"},
            {"name": "pnext", "as": "splash_fullrank_ltr_next_score"},
            {"name": "relate_evtr", "as": "splash_fullrank_ltr_relate_evtr_score"},
          ],
          recv_common_attrs = [
            {"name": "user_relavance_intention", "as": "splash_fullrank_ltr_user_relavance_intention_score"},
          ],
          timeout_ms = 100,
          send_item_attrs = [
            "cascade_pctr",
            "cascade_pltr",
            "cascade_pwtr",
            "cascade_plvtr",
            "cascade_psvtr",
            "cascade_pftr",
            "cascade_ptr",
            "cascade_pcmtr",
            "fullrank_detail_pctr",
            "fullrank_detail_pltr",
            "fullrank_detail_pwtr",
            "fullrank_detail_pftr",
            "fullrank_detail_plvtr",
            "fullrank_detail_pvtr",
            "fullrank_detail_psvr",
            "fullrank_detail_pcmtr",
            "fullrank_detail_pptr",
            "fullrank_detail_pwtd",
            "fullrank_detail_pcmef",
            "fullrank_detail_pwtd",
            "fullrank_detail_pepstr",
            "fullrank_detail_phtr",
            "fullrank_detail_new_pevtr_v2",
            "fullrank_sim_pcpr",
            "fullrank_sim_pcltr",
            "fullrank_sim_pepstr",
            "fullrank_sim_psvr",
            "fountain_related_score_v2",
          ],
          send_common_attrs = [
            { "name": "fr_deep_ltr_trimmed_user_info", "as": "user_info_str" },
            { "name": "featureSourcePId", "as": "source_pid" },
            { "name": "sourcePidDuration", "as": "source_duration_ms" },
            { "name": "sourcePidTagId", "as": "source_tag" },
            { "name": "sourcePidAuthorId", "as": "source_aid" },
            { "name": "sourcePidHetuLevelOneList", "as": "source_hetu_tag_level1_list" },
            { "name": "sourcePidHetuLevelTwoList", "as": "source_hetu_tag_level2_list" },
            { "name": "featureSimilarUserList", "as": "similar_user_list" },
            "page",
            "colossus_photo_id_list",
            "colossus_tag_list",
            "colossus_play_time_list",
            "colossus_label_list",
            "colossus_channel_list",
            "colossus_duration_list",
            "colossus_timestamp_list",
          ],
          request_type = "{{fountain_splash_deep_ltr_request_type}}",
          partition_size = "{{fountain_splash_deep_ltr_partition_size}}",
        ) \
      .end_()
    return self

  def enrich_fullrank_score_attr(self):
    """
    精排模型预估值及特征填充，以及精排各种打分的变换
    此处增加模型时， ab 开关请使用后缀区分首屏相关推荐、非首屏，避免不必要的耗时
    """
    self \
    .fullrank_all_ltr_predict() \
    .fullrank_all_ltv_predict() \
    .fullrank_esnn_predict() \
    .fullrank_session_predict() \
    .fullrank_effective_follow_predict() \
    .fullrank_effective_follow_ua_predict() \
    .fullrank_effective_follow_ua_pfd_predict() \
    .fullrank_ua_tracking_predict() \
    .fullrank_wtd_by_frac_predict() \
    .fullrank_batch_similar_predict() \
    .fullrank_duration_debias() \
    .fullrank_upload_time_debias() \
    .fullrank_pxtr_debais() \
    .enrich_attr_by_lua(
      import_common_attr = [
        "fountain_fullrank_duration_discount_weight",
        "fountain_fullrank_lvtr_sigmoid_bias",
        "fountain_vtr_max_value",
        "fountain_vtr_sigmoid_decay_rate",
        "fountain_vtr_smooth_rate",
        "fountain_vtr_sigmoid_bias",
        "enable_fountain_pwatch_time_sigmoid_bias_new",
        "fountian_vtr_big_duration_discount_bias",
        "fountian_vtr_big_duration_discount_slope",
        "fountain_vtr_score_discount_fix",
        "enable_fountain_longview_score_remove_click_coef",
        "fountain_fullrank_sim_pevtr_coef_weight",
        "fountain_fullrank_sim_pvtr_coef_weight",
        "fullrank_pvtr_trans_score_threshold",
        "skip_fountain_act_vtr_norm",
        "fountain_fullrank_act_l2r_max",
        "fountain_fullrank_act_l2r_merge_weight",
        "fountain_fullrank_act_vtr_merge_weight",
        "skip_fountain_act_vtr_merge",
        "skip_fountain_act_l2r_replace",
        "fountain_fullrank_distill_score_evtr_v2_weight",
        "enable_fr_origin_pvtr"
      ],
      import_item_attr = [
        "fullrank_detail_new_pevtr_v2",
        "fullrank_sim_pwtd_v2_playtime",
        "duration_ms",
        "fullrank_sim_pvtr",
        "explore_stat__view_length_sum",
        "explore_stat__click_count",
        "fullrank_sim_click_score",
        "fullrank_sim_plvtr",
        "picture_variant_attr",
        "fullrank_sim_pwtr",
        "fullrank_sim_pptr",
        "fountain_act_vtr_max",
        "fullrank_act_wtd",
        "fullrank_ltr_score",
        "fullrank_distill_rerank_score",
      ],
      export_item_attr = [
        "fullrank_sim_pwatchtime_no_bias",
        "fullrank_sim_longview_score_no_bias",
        "fullrank_sim_pvtr_multi_pwtr",
        "fullrank_sim_pvtr_multi_pptr",
        "fullrank_sim_evtr_v2_multi_pfintr",
        "fullrank_trans_pvtr_score",
        "fullrank_act_wtd",
        "fullrank_ltr_score",
        "fullrank_evtr_distill_score",
      ],
      skip = "{{skip_fullrank_calc_fullrank_score_lua_v1}}",
      function_for_item = "calc_fullrank_score",
      lua_script_file = "fountain/full_rank/lua/calc_fullrank_score.lua",
    ) \
    .enrich_attr_by_lua(
      import_common_attr = [
        "fountain_fullrank_diff_ctr_weight",
        "fountain_fullrank_diff_ctr_weight_2",
        "fountain_fullrank_diff_lvtr_weight",
        "fountain_fullrank_diff_wtr_weight",
        "fountain_fullrank_diff_ltr_weight",
        "fountain_fullrank_diff_ptr_weight",
        "fountain_fullrank_diff_cmtr_weight",
        "fountain_fullrank_diff_fintr_weight",
        "fountain_fullrank_diff_evtr_weight",
        "fountain_fullrank_diff_act_weight",
      ],
      import_item_attr = [
        "fullrank_sim_click_score",
        "fullrank_sim_plvtr",
        "fullrank_sim_pwtr",
        "fullrank_sim_pptr",
        "fullrank_sim_like_score",
        "fullrank_sim_pcmtr",
        "fullrank_sim_pfintr",
        "fullrank_detail_new_pevtr_v2",
        "fullrank_act_ctr",
      ],
      export_item_attr = [
        "fullrank_diff_score",
      ],
      skip = "{{skip_fullrank_calc_fullrank_diff_score}}",
      function_for_item = "calc_diff_score",
      lua_script_file = "fountain/full_rank/lua/calc_fullrank_score.lua",
    ) \
    .if_("skip_fullrank_gen_min_act_rank_reci ==0") \
      .gen_min_act_rank_reci() \
    .end_if_() \
    .if_("fountain_enable_fullrank_gen_min_wt_rank_reci == 1") \
      .gen_min_wt_rank_reci() \
    .end_if_() \
    .if_("enable_fullrank_gen_high_multiply_score == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .gen_high_multiply_score() \
    .end_if_() \
    .if_("ft_enable_pcmef_gender_debias == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .gen_pcmef_gender_debias_score() \
    .end_if_() \
    .if_("enable_fullrank_evtr_multiply_time_score == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .fullrank_get_evtr_multiply_time_score() \
    .end_if_() \
    .if_("enable_fountain_fr_get_xtr_fractile_score == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .fullrank_get_xtr_fractile_score() \
    .end_if_() \
    .if_("enable_fountain_fr_get_fractile_by_memdata == 1") \
      .fullrank_get_xtr_fractile_score_by_memdata() \
    .end_if_() \
    .if_("enable_fountain_fullrank_calc_pairwise_rank_score == 1", to_be_delete = "date=2024-05-29;committer=liucong03") \
      .fullrank_calc_pairwise_rank_score() \
    .end_if_() \
    .if_("enable_fountain_fullrank_calc_ltr_pairwise_rank_score == 1 and (basic_info_age_segment_v2 ~= fountain_fullrank_calc_ltr_pairwise_rank_score_skip_age or basic_info_gender_v2 ~= fountain_fullrank_calc_ltr_pairwise_rank_score_skip_gender)") \
      .fullrank_calc_ltr_pairwise_rank_score() \
    .end_if_() \
    .if_("enable_fountain_fullrank_calc_wtd_sharpe_ratio_score == 1") \
      .fullrank_calc_wtd_sharpe_ratio_score() \
    .end_if_() \
    .if_("enable_fountain_fr_calc_svtr_adapt_wtd_score == 1", to_be_delete = "date=2024-05-29;committer=liucong03") \
      .fullrank_cal_svtr_adapt_wtd_score() \
    .end_if_() \
    .if_("enable_fountain_fullrank_calc_fractile_fusion_score == 1") \
      .fullrank_calc_fractile_fusion_score() \
    .end_if_() \
    .if_("enable_fountain_fullrank_calc_confidence_pxtr_fusion_score == 1") \
      .fullrank_calc_confidence_pxtr_fusion_score() \
    .end_if_() \
    .if_("enable_fountain_fullrank_calc_fractile_sum_score == 1") \
      .fullrank_calc_fractile_sum_score() \
    .end_if_() \
    .if_("enable_fountain_get_dynamic_play_ratio_dynamic_weight == 1") \
      .get_dynamic_play_ratio_dynamic_weight() \
    .end_if_() \
    .if_("enable_fountain_fullrank_user_dynamic_play_ratio_weights == 1") \
      .cal_user_dynamic_play_ratio_weights() \
    .end_if_() \
    .if_("enable_fountain_get_dynamic_actiononce_dynamic_weight == 1") \
      .get_dynamic_actiononce_dynamic_weight() \
    .end_if_() \
    .if_("enable_fountain_fullrank_user_dynamic_actiononce_weights == 1") \
      .cal_user_dynamic_actiononce_weights() \
    .end_if_() \
    .if_("enable_fountain_cal_quantile_relative_score == 1") \
      .fountain_cal_quantile_relative_score() \
    .end_if_() \
    .if_("enable_fountain_fullrank_duration_xtr_debias_score == 1") \
      .sort(
        score_from_attr = "duration_ms",
        desc = False,
      ) \
      .fullrank_calc_duration_xtr_debias_score() \
    .end_if_() \
    .if_("enable_fountain_fullrank_cal_hetu_second_tag_debias_score == 1") \
      .fountain_cal_hetu_second_debias_score_fr_s2() \
    .end_if_() \
    .if_("enable_fountain_fullrank_dynamic_i2i_score == 1") \
      .fountain_fullrank_dynamic_i2i_score() \
    .end_() \
    .if_("enable_fountain_fullrank_action_calibration_fusion_score == 1") \
      .fullrank_calc_action_calibration_fusion_score() \
    .end_() \
    .if_("enable_fountain_fullrank_pvtr_derive_watchtime_score == 1") \
      .fullrank_calc_pvtr_derive_watchtime_score() \
    .end_() \
    .if_("enable_fountain_fullrank_same_hetu_long_video_compensate_score == 1") \
      .fountain_fullrank_calc_same_hetu_compensate_score() \
    .end_() \
    .log_debug_info(
      item_attrs = [
        "fullrank_trans_pvtr_score",
        "fullrank_sim_evtr_v2_multi_pfintr",
      ],
      for_debug_request_only = True,
      item_num_limit = 10,
    )
    return self

  def fullrank_cl_ltr_predict_pre(self):
    self \
    .delegate_enrich(
      kess_service = "{{fountain_cl_rank_predict_kess_service}}",
      recv_item_attrs = [
        {"name": "fountain_time", "as": "fullrank_cl_score"},
        {"name": "fountain_play_time", "as": "fullrank_cl_play_time"}
      ],
      timeout_ms = 150,
      send_item_attrs = [feature["name"] for feature in photo_features if feature["name"] not in photo_pxtr_features],
      send_common_attrs = user_features_v2,
      request_type = "kai_predict",
      skip = "{{skip_fountain_cl_rank_predict_kess_service}}",
      partition_size = "{{fountain_cl_rank_predict_partition_size}}",
    )
    return self
  
  def fullrank_cl_ltr_predict_post(self):
    self \
    .get_kconf_params(
      skip = "{{skip_fullrank_cl_score_with_duration}}",
      kconf_configs = [{
        "kconf_key": "{{fullrank_cl_score_finish_threshold_kconf}}",
        "value_type": "list_double",
        "defult_value": [],
        "export_common_attr": "duration_finish_threshold"
      }]
    ) \
    .enrich_attr_by_lua(
      skip = "{{skip_fullrank_cl_score_with_duration}}",
      import_common_attr = [
        "fountain_fullrank_cl_time_score_weight",
        "fountain_fullrank_cl_duration_weight",
        "fountain_fullrank_cl_click_weight",
        "fountain_fullrank_cl_duration_seg",
        "fountain_fullrank_cl_duration_max",
        "fountain_fullrank_cl_enable_threshold_bias",
        "fountain_fullrank_cl_enable_threshold_bias_v2",
        "fountain_fullrank_cl_enable_duration",
        "fountain_fullrank_cl_threshold_weight",
        "duration_finish_threshold"
      ],
      import_item_attr = [
        "fullrank_cl_score",
        "duration_ms",
        "fullrank_sim_pevtr"
      ],
      export_item_attr = [
        "fullrank_cl_tran_score"
      ],
      function_for_item = "fullrank_cal_cl_score",
      lua_script_file = "fountain/full_rank/lua/fullrank_cal_cl_score.lua",
    )
    return self

  def fullrank_ensemble_pre_filter(self):
    """
    xtr ensemble filter
    """
    self \
    .fountain_ensemble_pre_filter(
      guarantee_num = "{{fullrank_splash_pre_filter_guarantee_num}}",
      keep_photo_size = "{{fullrank_splash_pre_filter_keep_photo_size}}",
      buckets_configs = "{{fullrank_splash_pre_filter_buckets_configs}}",
      save_score_to_attr="fullrank_pre_filter_score",
      queues = fullrank_pre_filter_queues,
    ) \
    .perflog_reason_count(
      check_point = "post_fullrank_ensemble_filter",
    ) \

    return self
  
  def get_dynamic_actiononce_dynamic_weight(self):
    """
    xtr ensemble dynamic weight
    """
    self \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_weight_adjust_emp_actiononce_base_stat", "as": "user_base_stat"},
        {"name": "user_emp_actiononce", "as": "user_dynamic_stat"},
        {"name": "fountain_weight_adjust_emp_actiononce_boost_coef_lower", "as": "boost_coef_lower"},
        {"name": "fountain_weight_adjust_emp_actiononce_boost_coef_upper", "as": "boost_coef_upper"},
        {"name": "fountain_weight_adjust_emp_actiononce_is_boost", "as": "is_boost"},
        {"name": "fountain_weight_adjust_emp_actiononce_action_power_weight", "as": "action_power_weight"}
      ],
      export_common_attr = [
        {"name": "user_dynamic_action", "as": "emp_dynamic_actiononce"}
      ],
      function_name = "CalcUserDynamicAction",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    
    return self
  
  def cal_user_dynamic_actiononce_weights(self):
    self \
    .gen_common_attr_by_lua(
      attr_map={
        "fountain_ensemble_weight_fullrank_action_interact_once_score": "emp_dynamic_actiononce * fountain_ensemble_weight_fullrank_action_interact_once_score",
      }
    )
    return self
  
  def get_dynamic_play_ratio_dynamic_weight(self):
    """
    xtr ensemble dynamic weight
    """
    self \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_weight_adjust_emp_psvr_base_stat", "as": "user_base_stat"},
        {"name": "user_emp_svtr", "as": "user_dynamic_stat"},
        {"name": "fountain_weight_adjust_emp_psvr_boost_coef_lower", "as": "boost_coef_lower"},
        {"name": "fountain_weight_adjust_emp_psvr_boost_coef_upper", "as": "boost_coef_upper"},
        {"name": "fountain_weight_adjust_emp_psvr_is_boost", "as": "is_boost"},
        {"name": "fountain_weight_adjust_emp_psvr_action_power_weight", "as": "action_power_weight"}
      ],
      export_common_attr = [
        {"name": "user_dynamic_action", "as": "emp_dynamic_psvr"}
      ],
      function_name = "CalcUserDynamicAction",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_weight_adjust_emp_pevtr_base_stat", "as": "user_base_stat"},
        {"name": "user_emp_evtr", "as": "user_dynamic_stat"},
        {"name": "fountain_weight_adjust_emp_pevtr_boost_coef_lower", "as": "boost_coef_lower"},
        {"name": "fountain_weight_adjust_emp_pevtr_boost_coef_upper", "as": "boost_coef_upper"},
        {"name": "fountain_weight_adjust_emp_pevtr_is_boost", "as": "is_boost"},
        {"name": "fountain_weight_adjust_emp_pevtr_action_power_weight", "as": "action_power_weight"}
      ],
      export_common_attr = [
        {"name": "user_dynamic_action", "as": "emp_dynamic_pevtr"}
      ],
      function_name = "CalcUserDynamicAction",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_weight_adjust_emp_plvtr_base_stat", "as": "user_base_stat"},
        {"name": "user_emp_lvtr", "as": "user_dynamic_stat"},
        {"name": "fountain_weight_adjust_emp_plvtr_boost_coef_lower", "as": "boost_coef_lower"},
        {"name": "fountain_weight_adjust_emp_plvtr_boost_coef_upper", "as": "boost_coef_upper"},
        {"name": "fountain_weight_adjust_emp_plvtr_is_boost", "as": "is_boost"},
        {"name": "fountain_weight_adjust_emp_plvtr_action_power_weight", "as": "action_power_weight"}
      ],
      export_common_attr = [
        {"name": "user_dynamic_action", "as": "emp_dynamic_plvtr"}
      ],
      function_name = "CalcUserDynamicAction",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_weight_adjust_emp_watchtime_base_stat", "as": "user_base_stat"},
        {"name": "user_emp_watch_time", "as": "user_dynamic_stat"},
        {"name": "fountain_weight_adjust_emp_watchtime_boost_coef_lower", "as": "boost_coef_lower"},
        {"name": "fountain_weight_adjust_emp_watchtime_boost_coef_upper", "as": "boost_coef_upper"},
        {"name": "fountain_weight_adjust_emp_watchtime_is_boost", "as": "is_boost"},
        {"name": "fountain_weight_adjust_emp_watchtime_action_power_weight", "as": "action_power_weight"}
      ],
      export_common_attr = [
        {"name": "user_dynamic_action", "as": "emp_dynamic_watchtime"}
      ],
      function_name = "CalcUserDynamicAction",
      class_name = "ExploreLightFunctionSetV2",
    )

    return self
  
  def cal_user_dynamic_play_ratio_weights(self):
    self \
    .gen_common_attr_by_lua(
      attr_map={
        "fountain_ensemble_power_weight_fullrank_click_score": "emp_dynamic_pevtr * fountain_ensemble_power_weight_fullrank_click_score",
        "fountain_ensemble_weight_fullrank_sim_plvtr": "emp_dynamic_plvtr * fountain_ensemble_weight_fullrank_sim_plvtr",
        "fountain_ensemble_power_weight_fullrank_pfintr_score": "emp_dynamic_watchtime * fountain_ensemble_power_weight_fullrank_pfintr_score",
        "fountain_ensemble_power_weight_fullrank_svr_in_order_score": "emp_dynamic_psvr * fountain_ensemble_power_weight_fullrank_svr_in_order_score" 
      }
    )
    return self

  def calculate_comment_ltr(self):
    """
    计算comment ltr
    """
    self \
    .get_abtest_params(
      biz_name = "MOBILE",
      prioritized_suffix = "{{_ABTEST_SUFFIX_LIST_}}",
      ab_params = [
      {
        "param_name": "fountain_skip_cal_enrich_comment_ltr_item_attr_from_redis_mobile",
        "param_type": "int",
        "default_value": 1,
        "attr_name": "fountain_skip_cal_enrich_comment_ltr_item_attr_from_redis_mobile",
      },
      {
        "param_name": "fountain_skip_cal_enrich_comment_ltr_common_attr_from_redis_mobile",
        "param_type": "int",
        "default_value": 1,
        "attr_name": "fountain_skip_cal_enrich_comment_ltr_common_attr_from_redis_mobile",
      },
      {
        "param_name": "fountain_skip_cal_comment_ltr_mobile",
        "param_type": "int",
        "default_value": 1,
        "attr_name": "fountain_skip_cal_comment_ltr_mobile",
      },
      {
        "param_name": "fountain_comment_ltr_model_kconf_key_mobile",
        "param_type": "string",
        "default_value":"reco.comment.fountianCommentLtrModel",
        "attr_name": "fountain_comment_ltr_model_kconf_key_mobile",
      },
    ]) \
    .enrich_attr_by_lua(
      import_common_attr = [
        "fountain_skip_cal_enrich_comment_ltr_item_attr_from_redis",
        "fountain_skip_cal_enrich_comment_ltr_common_attr_from_redis",
        "fountain_skip_cal_comment_ltr",
        "fountain_comment_ltr_model_kconf_key",
        "fountain_skip_cal_enrich_comment_ltr_item_attr_from_redis_mobile",
        "fountain_skip_cal_enrich_comment_ltr_common_attr_from_redis_mobile",
        "fountain_skip_cal_comment_ltr_mobile",
        "fountain_comment_ltr_model_kconf_key_mobile",
      ],
      export_common_attr = [
        "fountain_skip_cal_enrich_comment_ltr_item_attr_from_redis",
        "fountain_skip_cal_enrich_comment_ltr_common_attr_from_redis",
        "fountain_skip_cal_comment_ltr",
        "fountain_comment_ltr_model_kconf_key",
      ],
      function_for_common = "is_mobile_enable_comment_ltr",
      lua_script_file = "fountain/full_rank/lua/fullrank_comment_ltr_mobile_ab.lua",
    ) \
    .item_attr_enrich_from_redis_json(
      skip = "{{fountain_skip_cal_enrich_comment_ltr_item_attr_from_redis}}",
      cluster_name="mmuNlpComment",
      redis_key_prefix="nlp_photo_analysis",
      timeout_ms=30,
      json_queues=[
        {
          "name":"avg_like_with_show",
          "type":"double",
          "output_attr":"photo_avg_like_with_show"
        },
        {
          "name":"max_like_with_show",
          "type":"double",
          "output_attr":"photo_max_like_with_show"
        },
        {
          "name":"like_cnt",
          "type":"double",
          "output_attr":"photo_like_cnt"
        },
        {
          "name":"show_cnt",
          "type":"double",
          "output_attr":"photo_show_cnt"
        },
        {
          "name":"comment_cnt",
          "type":"double",
          "output_attr":"photo_comment_cnt"
        },
        {
          "name":"author_like_cnt",
          "type":"double",
          "output_attr":"photo_author_like_cnt"
        },
        {
          "name":"update_7d_cnt",
          "type":"double",
          "output_attr":"photo_update_7d_cnt"
        },
        {
          "name":"first_comment_cnt",
          "type":"double",
          "output_attr":"photo_first_comment_cnt"
        },
        {
          "name":"second_comment_cnt",
          "type":"double",
          "output_attr":"photo_second_comment_cnt"
        },
        {
          "name":"author_reply_cnt",
          "type":"double",
          "output_attr":"photo_author_reply_cnt"
        },
        {
          "name":"author_show_cnt",
          "type":"double",
          "output_attr":"photo_author_show_cnt"
        },
        {
          "name":"god_cnt",
          "type":"double",
          "output_attr":"photo_god_cnt"
        },
        {
          "name":"hot_cnt",
          "type":"double",
          "output_attr":"photo_hot_cnt"
        },
        {
          "name":"pre_god_cnt",
          "type":"double",
          "output_attr":"photo_pre_god_cnt"
        },
        {
          "name":"pre_hot_cnt",
          "type":"double",
          "output_attr":"photo_pre_hot_cnt"
        },
        {
          "name":"only_at_cnt",
          "type":"double",
          "output_attr":"photo_only_at_cnt"
        },
        {"name":"emoji_cnt",
          "type":"double",
          "output_attr":"photo_emoji_cnt"
        },
        {
          "name":"kmoji_cnt",
          "type":"double",
          "output_attr":"photo_kmoji_cnt"
        },
        {
          "name":"only_punctuation_cnt",
          "type":"double",
          "output_attr":"photo_only_punctuation_cnt"
        },
        {
          "name":"only_num_cnt",
          "type":"double",
          "output_attr":"photo_only_num_cnt"
        },
        {"name":"yuqi_cnt",
          "type":"double",
          "output_attr":"photo_yuqi_cnt"
        },
        {
          "name":"qiugoumai_cnt",
          "type":"double",
          "output_attr":"photo_qiugoumai_cnt"
        },
        {
          "name":"qiuziyuan_cnt",
          "type":"double",
          "output_attr":"photo_qiuziyuan_cnt"
        },
        {
          "name":"qiuhudong_cnt",
          "type":"double",
          "output_attr":"photo_qiuhudong_cnt"
        },
        {
          "name":"zhuixing_cnt",
          "type":"double",
          "output_attr":"photo_zhuixing_cnt"
        },
        {
          "name":"aicheng_cnt",
          "type":"double",
          "output_attr":"photo_aicheng_cnt"
        },
        {
          "name":"zanshang_cnt",
          "type":"double",
          "output_attr":"photo_zanshang_cnt"
        },
        {
          "name":"feiwenben_cnt",
          "type":"double",
          "output_attr":"photo_feiwenben_cnt"
        }
        ]
    ) \
    .common_attr_enrich_from_redis_json(
      skip = "{{fountain_skip_cal_enrich_comment_ltr_common_attr_from_redis}}",
      cluster_name="mmuNlpComment",
      redis_key_prefix="user_analysis",
      timeout_ms=30,
      json_queues=[
        {
          "name":"comment_count",
          "type":"double",
          "output_attr":"user_comment_count"
        },
        {
          "name":"like_count",
          "type":"double",
          "output_attr":"user_like_count"
        },
        {
          "name":"show_count",
          "type":"double",
          "output_attr":"user_show_count"
        },
        {
          "name":"comment_count_3d",
          "type":"double",
          "output_attr":"user_comment_count_3d"
        },
        {
          "name":"like_count_3d",
          "type":"double",
          "output_attr":"user_like_count_3d"
        },
        {
          "name":"show_count_3d",
          "type":"double",
          "output_attr":"user_show_count_3d"
        },
        {
          "name":"comment_count_1d",
          "type":"double",
          "output_attr":"user_comment_count_1d"
        },
        {
          "name":"like_count_1d",
          "type":"double",
          "output_attr":"user_like_count_1d"
        },
        {
          "name":"show_count_1d",
          "type":"double",
          "output_attr":"user_show_count_1d"
        },
        ]
    ) \
    .item_predict_by_xgb(
      skip = "{{fountain_skip_cal_comment_ltr}}",
      model_kconf_key="{{fountain_comment_ltr_model_kconf_key}}",
      output_attr="comment_ltr",
      common_feature_attrs=["user_comment_count","user_like_count","user_show_count","user_comment_count_3d","user_like_count_3d","user_show_count_3d","user_comment_count_1d","user_like_count_1d","user_show_count_1d"],
      item_feature_attrs=["photo_avg_like_with_show","photo_max_like_with_show","photo_like_cnt","photo_show_cnt","photo_update_7d_cnt","photo_comment_cnt","photo_first_comment_cnt","photo_second_comment_cnt","photo_author_reply_cnt","photo_author_like_cnt","photo_author_show_cnt","photo_god_cnt","photo_hot_cnt","photo_pre_god_cnt","photo_pre_hot_cnt","photo_only_at_cnt","photo_emoji_cnt","photo_kmoji_cnt","photo_only_punctuation_cnt","photo_only_num_cnt","photo_yuqi_cnt","photo_qiugoumai_cnt","photo_qiuziyuan_cnt","photo_qiuhudong_cnt","photo_zhuixing_cnt","photo_aicheng_cnt","photo_zanshang_cnt","photo_feiwenben_cnt"]
    ) \
    .item_xgb_kml_predict_enrich(
      skip = "{{fountain_skip_cal_kml_comment_ltr}}",
      model_name = "{{fountain_kml_comment_ltr_model_name}}",
      kml_service_name = "{{fountain_kml_comment_ltr_kml_service_name}}",
      common_attrs_oredr_kconf = "{{fountain_comment_ltr_common_attr_order}}",
      item_attrs_oredr_kconf = "{{fountain_comment_ltr_item_attr_order}}",
      output_attr = "comment_ltr",
      timeout_ms = 100,
      common_feature_attrs=["user_comment_count","user_like_count","user_show_count","user_comment_count_3d","user_like_count_3d","user_show_count_3d","user_comment_count_1d","user_like_count_1d","user_show_count_1d"],
      item_feature_attrs=["photo_avg_like_with_show","photo_max_like_with_show","photo_like_cnt","photo_show_cnt","photo_update_7d_cnt","photo_comment_cnt","photo_first_comment_cnt","photo_second_comment_cnt","photo_author_reply_cnt","photo_author_like_cnt","photo_author_show_cnt","photo_god_cnt","photo_hot_cnt","photo_pre_god_cnt","photo_pre_hot_cnt","photo_only_at_cnt","photo_emoji_cnt","photo_kmoji_cnt","photo_only_punctuation_cnt","photo_only_num_cnt","photo_yuqi_cnt","photo_qiugoumai_cnt","photo_qiuziyuan_cnt","photo_qiuhudong_cnt","photo_zhuixing_cnt","photo_aicheng_cnt","photo_zanshang_cnt","photo_feiwenben_cnt"]
    )
    return self

  def cal_user_rl_xtr_score(self):
    return self \
    .if_("fullrank_enable_cal_user_rl_xtr_score > 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .explore_sphinx_param_enrich(
        user_info_pb_name = "userInfoPb",
        session_id_attr = "sessionId",
        user_stat_attr = "user_stat_xtr_attr_from_redis",
        user_app_attr = "user_app_attr_from_redis",
        use_emp_play="{{fountain_fullrank_param_jarvis_use_emp_play}}",
        request_based_jarvis_enabled=True,
        jarvis_kess_service="{{fullrank_param_jarvis_service}}",
        jarvis_model_name="{{fullrank_param_jarvis_model_name}}",
        app_name="{{fullrank_param_jarvis_app_name}}",
        action_type="{{fullrank_param_jarvis_action_type}}",
        jarvis_time_out=150,
        item_attrs={
          "author_id": "author__id",
          "pevtr": "fullrank_sim_pevtr",
          "plvtr": "fullrank_sim_plvtr",
          "psvr": "fullrank_sim_psvr",
          "pvtr": "fullrank_sim_pvtr",
          "pltr": "fullrank_sim_pltr",
          "phtr": "fullrank_sim_phtr",
          "pwtr": "fullrank_sim_pwtr",
          "pftr": "fullrank_sim_pftr",
          "pptr": "fullrank_sim_pptr",
          "pcmtr": "fullrank_sim_pcmtr",
          "pcmef": "fullrank_sim_pcmef",
          "pepstr": "fullrank_sim_pepstr",
          "plstr": "fullrank_sim_lstr",
          "pcltr": "fullrank_sim_pcltr",
          "pfr_score1": "fullrank_sim_pfintr",
          "duration_ms": "duration_ms",
          "hetu_level_one": "hetu_tag_level_info__hetu_level_one",
          "hetu_level_two": "hetu_tag_level_info__hetu_level_two",
            "hetu_level_three": "hetu_tag_level_info__hetu_level_three"
          },
          queues=[
            {
              "name": "adjust:rl_rerank_0",
              "param_attr": "fullrank_rl_ctr_weight"
            },
            {
              "name": "adjust:rl_rerank_1",
              "param_attr": "fullrank_rl_lvtr_weight"
            },
            {
              "name": "adjust:rl_rerank_2",
              "param_attr": "fullrank_rl_htr_weight"
            },
            {
              "name": "adjust:rl_rerank_3",
              "param_attr": "fullrank_rl_ltr_weight"
            },
            {
              "name": "adjust:rl_rerank_4",
              "param_attr": "fullrank_rl_wtr_weight"
            },
            {
              "name": "adjust:rl_rerank_5",
              "param_attr": "fullrank_rl_ftr_weight"
            },
            {
              "name": "adjust:rl_rerank_6",
              "param_attr": "fullrank_rl_ptr_weight"
            },
            {
              "name": "adjust:rl_rerank_7",
              "param_attr": "fullrank_rl_cmtr_weight"
            },
            {
              "name": "adjust:rl_rerank_8",
              "param_attr": "fullrank_rl_cmef_weight"
            },
            {
              "name": "adjust:rl_rerank_9",
              "param_attr": "fullrank_rl_epstr_weight"
            },
            {
              "name": "adjust:rl_rerank_10",
              "param_attr": "fullrank_rl_fintr_weight"
            },
            {
              "name": "adjust:rl_rerank_11",
              "param_attr": "fullrank_rl_vtr_weight"
            },
            {
              "name": "adjust:rl_rerank_12",
              "param_attr": "fullrank_rl_bias"
            }
          ]
       ) \
      .enrich_attr_by_lua(
        import_common_attr = [
          "fullrank_rl_ctr_weight",
          "fullrank_rl_lvtr_weight",
          "fullrank_rl_htr_weight",
          "fullrank_rl_ltr_weight",
          "fullrank_rl_wtr_weight",
          "fullrank_rl_ftr_weight",
          "fullrank_rl_ptr_weight",
          "fullrank_rl_cmtr_weight",
          "fullrank_rl_cmef_weight",
          "fullrank_rl_epstr_weight",
          "fullrank_rl_fintr_weight",
          "fullrank_rl_vtr_weight",
          "fullrank_rl_bias"
        ],
        import_item_attr = [
          "fullrank_sim_pevtr",
          "fullrank_sim_plvtr",
          "fullrank_sim_phtr",
          "fullrank_sim_pltr",
          "fullrank_sim_pwtr",
          "fullrank_sim_pftr",
          "fullrank_sim_pptr",
          "fullrank_sim_pcmtr",
          "fullrank_sim_pcmef",
          "fullrank_sim_pepstr",
          "fullrank_sim_pfintr",
          "fullrank_sim_pvtr"
        ],
        export_item_attr = [
          "fullrank_rl_xtr_score",
        ],
        function_for_item = "cal_rl_xtr_score",
        lua_script_file = "fountain/full_rank/lua/cal_opportunity_cost_score.lua",
      ) \
      .log_debug_info(
        item_attrs = [
          "fullrank_rl_xtr_score",
        ],
        item_num_limit = 10,
        for_debug_request_only = True,
      ) \
    .end_if_() \

  def cal_fit_ptime_score(self):
    self \
    .enrich_attr_by_lua(
      skip = "{{skip_cal_fit_ptime_score}}",
      import_item_attr = [
        "fullrank_sim_pevtr",
        "fullrank_sim_plvtr",
        "duration_ms",
      ],
      export_item_attr = [
        "fullrank_fit_ptime_score",
      ],
      function_for_item = "cal_fit_ptime_score",
      lua_script_file = "fountain/full_rank/lua/cal_opportunity_cost_score.lua",
    ) \

    return self
  
  def cal_triplem_time_score(self):
    self \
    .pack_item_attr(
      item_source = {
        "reco_results": True,
      },
      mappings = [
        {
          "aggregator": "min",
          "from_item_attr": "fullrank_sim_pevtr",
          "to_common_attr": "pevtr_min"
        },
        {
          "aggregator": "max",
          "from_item_attr": "fullrank_sim_pevtr",
          "to_common_attr": "pevtr_max"
        },
        {
          "aggregator": "min",
          "from_item_attr": "fullrank_sim_pvtr",
          "to_common_attr": "pvtr_min"
        },
        {
          "aggregator": "max",
          "from_item_attr": "fullrank_sim_pvtr",
          "to_common_attr": "pvtr_max"
        },
        {
          "aggregator": "min",
          "from_item_attr": "fullrank_sim_plvtr",
          "to_common_attr": "plvtr_min"
        },
        {
          "aggregator": "max",
          "from_item_attr": "fullrank_sim_plvtr",
          "to_common_attr": "plvtr_max"
        },
        {
          "aggregator": "min",
          "from_item_attr": "fullrank_sim_pcpr",
          "to_common_attr": "pcpr_min"
        },
        {
          "aggregator": "max",
          "from_item_attr": "fullrank_sim_pcpr",
          "to_common_attr": "pcpr_max"
        },
        {
          "aggregator": "min",
          "from_item_attr": "fullrank_sim_pfintr",
          "to_common_attr": "pfintr_min"
        },
        {
          "aggregator": "max",
          "from_item_attr": "fullrank_sim_pfintr",
          "to_common_attr": "pfintr_max"
        },
        {
          "aggregator": "min",
          "from_item_attr": "fullrank_detail_new_pevtr_v2",
          "to_common_attr": "pevtr_v2_min"
        },
        {
          "aggregator": "max",
          "from_item_attr": "fullrank_detail_new_pevtr_v2",
          "to_common_attr": "pevtr_v2_max"
        },
      ]
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fullrank_triplem_evtr_weight", "as": "triplem_evtr_weight"},
        {"name": "fullrank_triplem_vtr_weight", "as": "triplem_vtr_weight"},
        {"name": "fullrank_triplem_lvtr_weight", "as": "triplem_lvtr_weight"},
        {"name": "fullrank_triplem_cpr_weight", "as": "triplem_cpr_weight"},
        {"name": "fullrank_triplem_fintr_weight", "as": "triplem_fintr_weight"},
        {"name": "fullrank_triplem_enable_evtr_v2_weight", "as": "triplem_enable_evtr_v2_weight"},
        "pevtr_min",
        "pevtr_max",
        "pvtr_min",
        "pvtr_max",
        "plvtr_min",
        "plvtr_max",
        "pcpr_min",
        "pcpr_max",
        "pfintr_min",
        "pfintr_max",
        "pevtr_v2_min",
        "pevtr_v2_max"
      ],
      import_item_attr = [
        {"name": "fullrank_sim_pevtr", "as": "evtr"},
        {"name": "fullrank_sim_pvtr", "as": "vtr"},
        {"name": "fullrank_sim_plvtr", "as": "lvtr"},
        {"name": "fullrank_sim_pcpr", "as": "cpr"},
        {"name": "fullrank_sim_pfintr", "as": "fintr"},
        {"name": "fullrank_detail_new_pevtr_v2", "as": "evtr_v2"}
      ],
      export_item_attr = [
        {"name": "triplem_time_score", "as": "fullrank_triplem_score"}
      ],
      function_name = "CalTriplemScore",
      class_name = "ExploreLightFunctionSetV2",
    ) 
    
    return self
  
  def cal_triplem_interaction_score(self):
    self \
    .pack_item_attr(
      item_source = {
        "reco_results": True,
      },
      mappings = [
        {
          "aggregator": "min",
          "from_item_attr": "fullrank_sim_pltr",
          "to_common_attr": "pltr_min"
        },
        {
          "aggregator": "max",
          "from_item_attr": "fullrank_sim_pltr",
          "to_common_attr": "pltr_max"
        },
        {
          "aggregator": "min",
          "from_item_attr": "fullrank_sim_pwtr",
          "to_common_attr": "pwtr_min"
        },
        {
          "aggregator": "max",
          "from_item_attr": "fullrank_sim_pwtr",
          "to_common_attr": "pwtr_max"
        },
        {
          "aggregator": "min",
          "from_item_attr": "fullrank_sim_pftr",
          "to_common_attr": "pftr_min"
        },
        {
          "aggregator": "max",
          "from_item_attr": "fullrank_sim_pftr",
          "to_common_attr": "pftr_max"
        },
        {
          "aggregator": "min",
          "from_item_attr": "fullrank_sim_pcmtr",
          "to_common_attr": "pcmtr_min"
        },
        {
          "aggregator": "max",
          "from_item_attr": "fullrank_sim_pcmtr",
          "to_common_attr": "pcmtr_max"
        },
        {
          "aggregator": "min",
          "from_item_attr": "fullrank_sim_pcltr",
          "to_common_attr": "pcltr_min"
        },
        {
          "aggregator": "max",
          "from_item_attr": "fullrank_sim_pcltr",
          "to_common_attr": "pcltr_max"
        },
        {
          "aggregator": "min",
          "from_item_attr": "fullrank_sim_pcmef",
          "to_common_attr": "pcmef_min"
        },
        {
          "aggregator": "max",
          "from_item_attr": "fullrank_sim_pcmef",
          "to_common_attr": "pcmef_max"
        },
        {
          "aggregator": "min",
          "from_item_attr": "fullrank_sim_pepstr",
          "to_common_attr": "pepstr_min"
        },
        {
          "aggregator": "max",
          "from_item_attr": "fullrank_sim_pepstr",
          "to_common_attr": "pepstr_max"
        },
      ]
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        "fullrank_triplem_pltr_weight",
        "fullrank_triplem_pwtr_weight",
        "fullrank_triplem_pftr_weight",
        "fullrank_triplem_pcmtr_weight",
        "fullrank_triplem_pcltr_weight",
        "fullrank_triplem_pcmef_weight",
        "fullrank_triplem_pepstr_weight",
        "pltr_min",
        "pltr_max",
        "pwtr_min",
        "pwtr_max",
        "pftr_min",
        "pftr_max",
        "pcmtr_min",
        "pcmtr_max",
        "pcltr_min",
        "pcltr_max",
        "pcmef_min",
        "pcmef_max",
        "pepstr_min",
        "pepstr_max"
      ],
      import_item_attr = [
        {"name": "fullrank_sim_pltr", "as": "pltr"},
        {"name": "fullrank_sim_pwtr", "as": "pwtr"},
        {"name": "fullrank_sim_pftr", "as": "pftr"},
        {"name": "fullrank_sim_pcmtr", "as": "pcmtr"},
        {"name": "fullrank_sim_pcltr", "as": "pcltr"},
        {"name": "fullrank_sim_pcmef", "as": "pcmef"},
        {"name": "fullrank_sim_pepstr", "as": "pepstr"}
      ],
      export_item_attr = [
        "fullrank_triplem_interaction_score",
      ],
      function_name = "CalTriplemInteractionScore",
      class_name = "ExploreLightFunctionSetV2",
    )
    
    return self


  def cal_user_ada_xtr_score(self):
    self \
    .explore_common_user_feature_enricher(
      user_info_attr = "userInfoPb",
      user_uid_attr = "uId",
      user_did_attr = "dId",
      user_province_attr =  "uProvinceId",
      user_city_attr = "uCityId",
      user_visit_mod_attr =  "uVisitMod",
      user_visit_net_attr = "uNetwork",
      user_follow_cnt_attr = "uFollowCount",
      user_upload_cnt_attr = "uUploadCount",
      user_risk_level_attr = "uRiskLevel",
      user_fans_cnt_attr =   "uFansCount",
      user_click_pids_attr = "uClickPids",
      user_like_pids_attr = "uLikePids",
      user_follow_pids_attr = "uFollowAids",
      user_upload_rate_attr = "uUploadRate",
      user_true_new_attr = "uTrueNewUser",
      user_login_attr = "uLogin",
      user_gender_attr = "uGender",
      user_infer_gender_attr = "uInferGender",
      user_ture_gender_attr = "uTrueGender",
      user_basic_gender_attr = "uBasicGender",
      user_infer_year_attr = "uInferYear",
      user_true_year_attr = "uTrueYear",
      user_basic_age_attr = "uBasicAge",
      user_city_level_attr = "uCityLevelNew",
      user_is_douyin_attr = "uIsDouyin",
      user_longview_action_attr = "longview_",
      user_shortview_action_attr = "shortview_",
      user_count_action_attr = "cnt_",
    ) \
    .get_kuiba_user_embedding(
      tensor_request_layer='joint_fountain_ltr',
      kess_service="{{fountain_ada_xtr_weight_service_name}}",
      timeout_ms=20,
      input_common_attr=user_ada_weight_feature,
      output_tensor_attr='user_ada_weight_tensor',
    ) \
    .enrich_attr_by_lua(
      import_common_attr = [
        "user_ada_weight_tensor",
        "fountain_ada_xtr_use_linear_score",
        "fountain_ada_xtr_use_reverse_score"
      ],
      import_item_attr = [
        "fullrank_sim_pevtr",
        "fullrank_sim_pltr",
        "fullrank_sim_pwtr",
        "fullrank_sim_pftr",
        "fullrank_sim_pptr",
        "fullrank_sim_pepstr",
        "fullrank_sim_pcltr",
        "fullrank_sim_pcmtr",
        "fullrank_sim_pcmef",
        "fullrank_sim_pvtr",
        "fullrank_sim_plvtr"
      ],
      export_item_attr = [
        "fullrank_ada_xtr_score",
      ],
      export_common_attr = [
        "ctr_weight",
        "ltr_weight",
        "wtr_weight",
        "ftr_weight",
        "ptr_weight",
        "epstr_weight",
        "cltr_weight",
        "cmtr_weight",
        "cmef_weight",
        "lvtr_weight",
        "vtr_weight",
      ],
      function_for_item = "cal_opportunity_cost_score_v2",
      function_for_common = "get_ada_weight",
      lua_script_file = "fountain/full_rank/lua/cal_opportunity_cost_score.lua",
    ) \
    .log_debug_info(
      common_attrs = user_ada_weight_feature +[
        "user_ada_weight_tensor",
        "user_click_pids_attr",
        "user_upload_rate_attr", "user_true_new_attr", "user_login_attr", "user_gender_attr",
        "user_infer_gender_attr", "user_ture_gender_attr", "user_basic_gender_attr",
        "user_infer_year_attr", "user_true_year_attr", "user_basic_age_attr",
        "user_city_level_attr", "user_is_douyin_attr", "user_longview_action_attr",
        "user_shortview_action_attr", "user_count_action_attr", "user_like_pids_attr", "user_follow_pids_attr",
        "user_uid_attr", "user_did_attr", "user_province_attr", "user_city_attr", "user_visit_mod_attr",
        "user_visit_net_attr", "user_fans_cnt_attr", "user_follow_cnt_attr", "user_upload_cnt_attr", "user_risk_level_attr"
      ],
      item_attrs = [
        "fullrank_ada_xtr_score",
      ],
      item_num_limit = 10,
      for_debug_request_only = True,
    ) \
    .perflog_attr_value(
      check_point = "fountain_ada_weight",
      common_attrs = [
        "ctr_weight",
        "ltr_weight",
        "wtr_weight",
        "ftr_weight",
        "ptr_weight",
        "epstr_weight",
        "cltr_weight",
        "cmtr_weight",
        "cmef_weight",
        "lvtr_weight",
        "vtr_weight",
      ]
    )

    return self

  def calculate_debias_pxtr(self):
    self \
    .get_common_attr_from_redis(
      cluster_name = "recoNewUserPhotos",
      skip = "{{fountain_skip_fullrank_bias_enricher}}",
      timeout_ms = 50,
      cache_bits = 16,
      redis_params = [
        {
          "redis_key": "fountain_fullrank_debias_hetu_level_one",
          "output_attr_name": "fountain_fullrank_debias_hetu_level_one"
        }
      ]
    ) \
    .explore_cascade_debias_enricher(
      skip = "{{fountain_skip_fullrank_bias_enricher}}",
      use_hetu_level_one = "{{fountain_fullrank_bias_use_hetu_level_one}}",
      hetu_emp_xtr_level_one = "fountain_fullrank_debias_hetu_level_one",
      hetu_emp_xtr_level_two = "",
      hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
      hetu_level_two_attr = "hetu_tag_level_info__hetu_level_two",
      queues=[
        {
          "original_name":"fullrank_sim_click_score",
          "debias_name":"fullrank_sim_click_score_debias",
          "max_score":10
        },
        {"original_name":"fullrank_sim_like_score",
          "debias_name":"fullrank_sim_like_score_debias",
          "max_score":10
        },
        {"original_name":"fullrank_sim_longview_score_no_bias",
          "debias_name":"fullrank_sim_longview_score_no_bias_debias",
          "max_score":200
        },
        {"original_name":"fullrank_sim_pwatchtime_no_bias",
          "debias_name":"fullrank_sim_pwatchtime_no_bias_debias",
          "max_score":10
        }
      ]
    )
    return self

  def calculate_xgb_ltr(self):
    """
    计算comment ltr
    """
    self \
    .item_xgb_kml_predict_enrich(
      skip = "{{fountain_skip_cal_kml_xgb_ltr}}",
      model_name = "{{fountain_kml_xgb_ltr_model_name}}",
      kml_service_name = "{{fountain_kml_xgb_ltr_kml_service_name}}",
      common_attrs_oredr_kconf = "{{fountain_xgb_ltr_common_attr_order}}",
      item_attrs_oredr_kconf = "{{fountain_xgb_ltr_item_attr_order}}",
      output_attr = "xgb_ltr",
      timeout_ms = 100,
      common_feature_attrs=[
        "featureCityId",
        "featureUserAvgWatchTime",
        "featureUserCtr",
        "featureUserLtr",
        "featureUserWtr",
        "featureDeviceModel"
      ],
      item_feature_attrs=[
        "featurePAFansCount",
        "fullrank_detail_pctr",
        "fullrank_detail_pltr",
        "fullrank_detail_pwtr",
        "fullrank_detail_pftr",
        "fullrank_detail_plvtr",
        "fullrank_detail_psvr",
        "fullrank_detail_pvtr",
        "fullrank_detail_pptr",
        "fullrank_detail_pcmtr",
        "fullrank_detail_phtr",
        "cascade_pctr",
        "cascade_pltr",
        "cascade_pwtr",
        "cascade_plvtr",
        "cascade_psvtr",
        "featurePDurationMs",
        "featurePUploadType",
        "featurePHotAvgWatchTime",
        "featurePHotCtr",
        "featurePHotLtr",
        "featurePHotWtr",
        "featurePHotFtr",
        "featurePHotLvtr",
        "featurePHotSvtr",
        "hetu_level_one_tag_index",
        "hetu_level_two_tag_index",
        "hetu_level_three_tag_index"
        ]
    )
    return self

  def calc_hate_list_similary_score(self):
    self \
      .if_("skip_fullrank_hate_similary_score_in_ensemble_sort == 0") \
        .explore_embedding_candidates_attr_enricher(
          trans_type = "fountain_candidates",
          enable_fix_low_hit_rate = "{{fountain_fullrank_enable_fix_mmu_embedding_low_hit_rate}}",
          enable_report = "{{fountain_fullrank_hate_similary_score_enable_report}}",
          user_info_ptr_attr = "userInfoPb",
          export_common_attr = "embedding_source_pids",
          check_point = "fullrank",
        ) \
        .pack_common_attr(
          input_common_attrs = [
            "embedding_source_pids",
            "fountain_retr_bad_input_item_key_list",
          ],
          output_common_attr = "embedding_source_pids",
          deduplicate = True
        ) \
        .get_remote_embedding_lite(
          kess_service = "{{fullrank_emb_kess_name_for_hate_similary_score}}",
          shard_num = 4,
          id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
          input_attr_name = "embedding_source_pids",
          output_attr_name = "mmu_embeddings",
          query_source_type = "common_attr",
          size = 64,
          client_side_shard = True
        ) \
        .explore_custom_embedding_score_enricher(
          check_point_ = "fullrank",
          enable_fountain_version = True,
          enable_fix_low_hit_rate = "{{fountain_fullrank_enable_fix_mmu_embedding_low_hit_rate}}",
          user_info_ptr_attr = "userInfoPb",
          embedding_list_attr = "mmu_embeddings",
          source_pids_list_attr = "embedding_source_pids",
          calc_type = "action_bucket_dot",
          not_click_limit_hour = "{{fullrank_hate_similary_score_not_click_hour_limit}}",
          play_stat_limit_hour = "{{fullrank_hate_similary_score_play_stat_hour_limit}}",
          extra_not_click_limit_hour = "{{fullrank_hate_similary_score_extra_not_click_hour_limit}}",
          hate_limit_hour = "{{fullrank_hate_similary_score_hate_hour_limit}}",
          report_limit_min = "{{fullrank_hate_similary_score_report_min_limit}}",
          not_click_weight = "{{fullrank_hate_similary_score_not_click_weight}}",
          short_view_weight = "{{fullrank_hate_similary_score_short_view_weight}}",
          extra_not_click_weight = "{{fullrank_hate_similary_score_extra_not_click_weight}}",
          hate_weight = "{{fullrank_hate_similary_score_hate_weight}}",
          report_weight = "{{fullrank_hate_similary_score_report_weight}}",
          export_item_attr = "fullrank_hate_similary_score",
          dim_size = 64
        ) \
        .if_("enable_fountain_bad_item_list_similarity_score == 1") \
          .explore_custom_embedding_score_enricher(
            user_info_ptr_attr = "userInfoPb",
            embedding_list_attr = "mmu_embeddings",
            source_pids_list_attr = "embedding_source_pids",
            target_pids_list_attr = "fountain_retr_bad_input_item_key_list",
            calc_type = "list_similarity",
            dim_size = 64,
            export_item_attr = "fountain_fullrank_bad_item_similary_score",
            select_item = {
              "attr_name": "audit_b_second_tag",
              "compare_to": 0,
              "select_if": "<=",
              "select_if_attr_missing": True
            }
          ) \
        .end_() \
      .end_if_()

    return self

  def cal_opportunity_cost_score(self):
    self \
    .enrich_attr_by_lua(
      skip = "{{skip_fullrank_cal_opportunity_cost_score}}",
      import_common_attr = [
        "fullrank_opportunity_score_ctr_weight",
        "fullrank_opportunity_score_ltr_weight",
        "fullrank_opportunity_score_wtr_weight",
        "fullrank_opportunity_score_ftr_weight",
        "fullrank_opportunity_score_cmtr_weight",
        "fullrank_opportunity_score_cmef_weight",
        "fullrank_opportunity_score_ptr_weight",
        "fullrank_opportunity_score_epstr_weight",
        "fullrank_opportunity_score_lstr_weight",
        "fullrank_opportunity_score_evtrv2_weight",
        "fullrank_opportunity_score_vtr_power_weight",
        "fullrank_opportunity_score_alpha",
        "fullrank_opportunity_score_beta",
        "fountain_enable_opportunity_score_use_trans_pvtr_score",
        "fountain_fullrank_trans_pvtr_score_max",
        "fountain_enable_opportunity_score_use_linear_score",
        "fullrank_opportunity_score_smooth",
        "fullrank_opportunity_score_pcltr_weight",
        "fullrank_opportunity_score_plvtr_weight",
        "fullrank_opportunity_score_pfintr_weight",
        "fullrank_opportunity_score_psvtr_weight",
        "fullrank_opportunity_score_phtr_weight",
        "fullrank_action_once_score_pctr_weight",
        "fullrank_action_once_score_pltr_weight",
        "fullrank_action_once_score_pwtr_weight",
        "fullrank_action_once_score_pftr_weight",
        "fullrank_action_once_score_pcmtr_weight",
        "fullrank_action_once_score_pcmef_weight",
        "fullrank_action_once_score_pptr_weight",
        "fullrank_action_once_score_pepstr_weight",
        "fullrank_action_once_score_lstr_weight",
        "fullrank_action_once_score_evtr_v2_weight",
        "fullrank_action_once_score_pcltr_weight",
        "fullrank_action_once_score_plvtr_weight",
        "fullrank_action_once_score_pfintr_weight",
        "fullrank_action_once_score_phtr_weight",
        "fountain_enable_opportunity_score_use_action_once_score"
      ],
      import_item_attr = [
        "fullrank_sim_pevtr",
        "fullrank_sim_pltr",
        "fullrank_sim_pwtr",
        "fullrank_sim_pftr",
        "fullrank_sim_pptr",
        "fullrank_sim_pcmtr",
        "fullrank_sim_pcmef",
        "fullrank_sim_pvtr",
        "fullrank_sim_pepstr",
        "fullrank_sim_lstr",
        "fullrank_detail_new_pevtr_v2",
        "fullrank_trans_pvtr_score",
        "fullrank_sim_pcltr",
        "fullrank_sim_plvtr",
        "fullrank_sim_pfintr",
        "fullrank_sim_psvr",
        "fullrank_sim_phtr"
      ],
      export_item_attr = [
        "fullrank_opportunity_cost_score",
        "fullrank_action_once_score",
      ],
      function_for_item = "cal_opportunity_cost_score",
      lua_script_file = "fountain/full_rank/lua/cal_opportunity_cost_score.lua",
    ) \
    .log_debug_info(
      item_attrs = [
        "fullrank_opportunity_cost_score",
        "fullrank_action_once_score",
        "fullrank_sim_pvtr",
        "duration_ms",
        "fullrank_trans_pvtr_score",
      ],
      common_attrs = [
        "fullrank_opportunity_score_alpha"
      ],
      for_debug_request_only = True,
      item_num_limit = 10,
    )
    return self

  def cal_satisfy_score(self):
    self \
      .enrich_attr_by_lua(
        skip = "{{skip_fullrank_cal_cal_satisfy_score}}",
        import_common_attr = [
          "fullrank_satisfy_score_duration_max",
          "fullrank_satisfy_score_pfintr_weight",
        ],
        import_item_attr = [
          "fullrank_sim_pfintr",
          "duration_ms",
        ],
        export_item_attr = [
          "fullrank_satisfy_score"
        ],
        function_for_item = "cal_satisfy_score",
        lua_script_file = "fountain/full_rank/lua/cal_opportunity_cost_score.lua",
      ) \
      .log_debug_info(
        item_attrs = [
          "fullrank_satisfy_score"
        ],
        for_debug_request_only = True,
        item_num_limit = 10,
      )
    return self

  def refactor_ori_es_module(self):
    self \
      .fountain_calc_ensemble_score(
        save_score_to_attr = "fullrank_ori_ensemble_watchtime_score",
        use_dist_calc = "{{fountain_fullrank_ensemble_use_dist_calc}}",
        dis_factor = "{{fountain_fullrank_ensemble_dis_factor}}",
        user_new_proportion = "{{fullrank_ensemble_score_watchtime_user_new_proportion}}",
        user_power_calc = "{{fountain_fullrank_watchtime_variant_enable_power_calc}}",
        rank_smooth = "{{fountain_fullrank_rank_watchtime_smooth}}",
        use_reciprocal = "{{fountain_fullrank_watchtime_use_reciprocal}}",
        duration_min = "{{fountain_fullrank_duration_min}}",
        duration_max = "{{fountain_fullrank_duration_max}}",
        user_info_ptr_attr = "userInfoPb",
        action_day = "{{fountain_fullrank_variant_weight_action_day_num}}",
        queues = fullrank_ensemble_watchtime_queues,
      ) \
      .fountain_calc_ensemble_score(
        save_score_to_attr = "fullrank_ori_ensemble_vv_score",
        use_dist_calc = "{{fountain_fullrank_ensemble_use_dist_calc}}",
        dis_factor = "{{fountain_fullrank_ensemble_dis_factor}}",
        user_new_proportion = "{{fullrank_ensemble_score_vv_user_new_proportion}}",
        user_power_calc = "{{fountain_fullrank_vv_variant_enable_power_calc}}",
        rank_smooth = "{{fountain_fullrank_rank_vv_smooth}}",
        use_reciprocal = "{{fountain_fullrank_vv_use_reciprocal}}",
        duration_min = "{{fountain_fullrank_duration_min}}",
        duration_max = "{{fountain_fullrank_duration_max}}",
        user_info_ptr_attr = "userInfoPb",
        action_day = "{{fountain_fullrank_variant_weight_action_day_num}}",
        queues = fullrank_ensemble_vv_queues,
      ) \
      .fountain_calc_ensemble_score(
        save_score_to_attr = "fullrank_ori_ensemble_interact_score",
        use_dist_calc = "{{fountain_fullrank_ensemble_use_dist_calc}}",
        dis_factor = "{{fountain_fullrank_ensemble_dis_factor}}",
        user_new_proportion = "{{fullrank_ensemble_score_interact_user_new_proportion}}",
        user_power_calc = "{{fountain_fullrank_interact_variant_enable_power_calc}}",
        rank_smooth = "{{fountain_fullrank_rank_interact_smooth}}",
        use_reciprocal = "{{fountain_fullrank_interact_use_reciprocal}}",
        duration_min = "{{fountain_fullrank_duration_min}}",
        duration_max = "{{fountain_fullrank_duration_max}}",
        user_info_ptr_attr = "userInfoPb",
        action_day = "{{fountain_fullrank_variant_weight_action_day_num}}",
        queues = fullrank_ensemble_interact_queues,
      ) \
      .fountain_calc_ensemble_score(
        save_score_to_attr = "fullrank_ori_ensemble_neg_score",
        use_dist_calc = "{{fountain_fullrank_ensemble_use_dist_calc}}",
        dis_factor = "{{fountain_fullrank_ensemble_dis_factor}}",
        user_new_proportion = "{{fullrank_ensemble_score_neg_user_new_proportion}}",
        user_power_calc = "{{fountain_fullrank_neg_variant_enable_power_calc}}",
        rank_smooth = "{{fountain_fullrank_rank_neg_smooth}}",
        use_reciprocal = "{{fountain_fullrank_neg_use_reciprocal}}",
        duration_min = "{{fountain_fullrank_duration_min}}",
        duration_max = "{{fountain_fullrank_duration_max}}",
        user_info_ptr_attr = "userInfoPb",
        action_day = "{{fountain_fullrank_variant_weight_action_day_num}}",
        queues = fullrank_ensemble_neg_queues,
      ) \
      .if_("enable_use_rank_combo_for_ori_ensemble == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .fountain_calc_ensemble_score(
          save_score_to_attr = "fullrank_ensemble_score",
          use_dist_calc = "{{fountain_fullrank_ensemble_use_dist_calc}}",
          dis_factor = "{{fountain_fullrank_ensemble_dis_factor}}",
          user_new_proportion = "{{fullrank_ensemble_score_combo_user_new_proportion}}",
          user_power_calc = "{{fountain_fullrank_combo_variant_enable_power_calc}}",
          rank_smooth = "{{fountain_fullrank_rank_combo_smooth}}",
          use_reciprocal = "{{fountain_fullrank_combo_use_reciprocal}}",
          duration_min = "{{fountain_fullrank_duration_min}}",
          duration_max = "{{fountain_fullrank_duration_max}}",
          user_info_ptr_attr = "userInfoPb",
          action_day = "{{fountain_fullrank_variant_weight_action_day_num}}",
          queues = fullrank_ensemble_combo_queues,
        ) \
      .else_() \
        .enrich_attr_by_lua(
          import_common_attr = [
            "fullrank_ori_ensemble_watchtime_score_power_weight",
            "fullrank_ori_ensemble_interact_score_power_weight",
            "fullrank_ori_ensemble_neg_score_power_weight",
            "fullrank_ori_ensemble_vv_score_power_weight",
          ],
          import_item_attr = [
            "fullrank_ori_ensemble_watchtime_score",
            "fullrank_ori_ensemble_interact_score",
            "fullrank_ori_ensemble_neg_score",
            "fullrank_ori_ensemble_vv_score",
          ],
          export_item_attr = [
            "fullrank_ensemble_score"
          ],
          function_for_item = "cal_ori_ensemble_score",
          lua_script_file = "fountain/full_rank/lua/cal_ori_ensemble_score.lua"
        ) \
      .end_if_() \
      .log_debug_info(
        item_attrs = [
          "fullrank_ori_ensemble_watchtime_score",
          "fullrank_ori_ensemble_interact_score",
          "fullrank_ori_ensemble_neg_score",
          "fullrank_ensemble_score"
        ],
        item_num_limit = 10,
        for_debug_request_only = True,
      )
    return self

  def cal_cascade_linear_score(self):
    self \
    .enrich_attr_by_lua(
      skip = "{{skip_fullrank_cal_cascade_linear_score}}",
      import_common_attr = [
        "fountain_fr_cascade_linear_score_ctr_weight",
        "fountain_fr_cascade_linear_score_ltr_weight",
        "fountain_fr_cascade_linear_score_wtr_weight",
        "fountain_fr_cascade_linear_score_ftr_weight",
        "fountain_fr_cascade_linear_score_cmtr_weight",
        "fountain_fr_cascade_linear_score_cestr_weight",
        "fountain_fr_cascade_linear_score_ptr_weight",
        "fountain_fr_cascade_linear_score_pepstr_weight",
        "fountain_fr_cascade_linear_score_svtr_weight",
        "fountain_fr_cascade_linear_score_lvtr_weight",
        "fountain_fr_cascade_linear_score_wtd_weight",
        "fountain_fr_cascade_linear_score_cltr_weight",
        "fountain_fr_cascade_linear_score_htr_weight",
        "fountain_fr_cascade_linear_score_watchtime_weight"
      ],
      import_item_attr = [
        "cascade_pctr",
        "cascade_pltr",
        "cascade_pwtr",
        "cascade_pftr",
        "cascade_pcmtr",
        "cascade_pcestr",
        "cascade_ptr",
        "cascade_pepstr",
        "cascade_psvtr",
        "cascade_plvtr",
        "cascade_pwtd",
        "cascade_pcltr",
        "cascade_phtr",
        "cascade_pwatch_time"
      ],
      export_item_attr = [
        "fullrank_cascade_linear_score"
      ],
      function_for_item = "cal_cascade_linear_score",
      lua_script_file = "fountain/full_rank/lua/cal_cascade_linear_score.lua"
    ) \
    .log_debug_info(
      common_attrs = [
        "fountain_fr_cascade_linear_score_ctr_weight",
        "fountain_fr_cascade_linear_score_ltr_weight",
        "fountain_fr_cascade_linear_score_wtr_weight",
        "fountain_fr_cascade_linear_score_ftr_weight",
        "fountain_fr_cascade_linear_score_cmtr_weight",
        "fountain_fr_cascade_linear_score_cestr_weight",
        "fountain_fr_cascade_linear_score_ptr_weight",
        "fountain_fr_cascade_linear_score_pepstr_weight",
        "fountain_fr_cascade_linear_score_svtr_weight",
        "fountain_fr_cascade_linear_score_lvtr_weight",
        "fountain_fr_cascade_linear_score_wtd_weight",
        "fountain_fr_cascade_linear_score_cltr_weight",
        "fountain_fr_cascade_linear_score_htr_weight",
        "fountain_fr_cascade_linear_score_watchtime_weight",
        "skip_fullrank_cal_cascade_linear_score"
      ],
      item_attrs = [
        "cascade_pctr",
        "cascade_pltr",
        "cascade_pwtr",
        "cascade_pftr",
        "cascade_pcmtr",
        "cascade_pcestr",
        "cascade_ptr",
        "cascade_pepstr",
        "cascade_psvtr",
        "cascade_plvtr",
        "cascade_pwtd",
        "cascade_pcltr",
        "cascade_phtr",
        "cascade_pwatch_time",
        "fullrank_cascade_linear_score"
      ],
      for_debug_request_only=True,
    )

    return self

  def cal_distill_fusion_score(self):
    self \
      .if_("fountain_enable_fullrank_distill_fusion_score == 1 and (fountain_rerank_distill_only_fast_v1 == 0 or page > 1)") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "fountain_distill_l2r_fusion_l2r_weight",
            "fountain_distill_l2r_fusion_wtd_weight",
            "fountain_distill_l2r_fusion_l2r_beta",
            "fountain_distill_l2r_fusion_wtd_beta",
            "fountain_distill_l2r_fusion_evtr_weight",
            "fountain_distill_l2r_fusion_evtr_beta",
            "fountain_distill_l2r_fusion_interact_weight",
            "fountain_distill_l2r_fusion_interact_beta"
          ],
          import_item_attr = [
            "fullrank_distill_rerank_score",
            "fullrank_sim_pfintr",
            "fullrank_sim_pevtr",
            "fullrank_action_once_interact_score"
          ],
          export_item_attr = [
            "fullrank_distill_fusion_score"
          ],
          function_name = "CalFusionDistillLtrScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_if_()
    return self

  def cal_value_multiply_score(self):
    self \
      .enrich_attr_by_lua(
        skip = "{{skip_fullrank_cal_value_multiply_score}}",
        import_common_attr = [
          "fullrank_multiply_score_pctr_alpha",
          "fullrank_multiply_score_evtr_v2_alpha",
          "fullrank_multiply_score_plvtr_alpha",
          "fullrank_multiply_score_pfintr_alpha",
          "fullrank_multiply_score_pvtr_alpha",
          "fullrank_multiply_score_trans_pvtr_alpha",
          "fullrank_multiply_score_pltr_alpha",
          "fullrank_multiply_score_pwtr_alpha",
          "fullrank_multiply_score_pftr_alpha",
          "fullrank_multiply_score_pcmtr_alpha",
          "fullrank_multiply_score_pcmef_alpha",
          "fullrank_multiply_score_pptr_alpha",
          "fullrank_multiply_score_pepstr_alpha",
          "fullrank_multiply_score_lstr_alpha",
          "fullrank_multiply_score_pcltr_alpha",
          "fullrank_multiply_score_pctr_beta",
          "fullrank_multiply_score_evtr_v2_beta",
          "fullrank_multiply_score_plvtr_beta",
          "fullrank_multiply_score_pfintr_beta",
          "fullrank_multiply_score_pvtr_beta",
          "fullrank_multiply_score_trans_pvtr_beta",
          "fullrank_multiply_score_pltr_beta",
          "fullrank_multiply_score_pwtr_beta",
          "fullrank_multiply_score_pftr_beta",
          "fullrank_multiply_score_pcmtr_beta",
          "fullrank_multiply_score_pcmef_beta",
          "fullrank_multiply_score_pptr_beta",
          "fullrank_multiply_score_pepstr_beta",
          "fullrank_multiply_score_lstr_beta",
          "fullrank_multiply_score_pcltr_beta",
        ],
        import_item_attr = [
          "fullrank_sim_pevtr",
          "fullrank_sim_pltr",
          "fullrank_sim_pwtr",
          "fullrank_sim_pftr",
          "fullrank_sim_pptr",
          "fullrank_sim_pcmtr",
          "fullrank_sim_pcmef",
          "fullrank_sim_pvtr",
          "fullrank_sim_pepstr",
          "fullrank_sim_lstr",
          "fullrank_detail_new_pevtr_v2",
          "fullrank_trans_pvtr_score",
          "fullrank_sim_pcltr",
          "fullrank_sim_plvtr",
          "fullrank_sim_pfintr",
        ],
        export_item_attr = [
          "fullrank_value_multiply_score",
        ],
        function_for_item = "cal_value_multiply_score",
        lua_script_file = "fountain/full_rank/lua/cal_opportunity_cost_score.lua",
      ) \
      .perflog_attr_value(
        check_point = "fullrank_value_multiply_score",
        item_attrs = [
          "fullrank_value_multiply_score"
        ]
      )
    return self

  def cal_action_once_score(self):
    self \
      .enrich_attr_by_lua(
        skip = "{{skip_fullrank_cal_action_once_score}}",
        import_common_attr = [
          "fullrank_action_once_watchtime_score_pctr_weight",
          "fullrank_action_once_watchtime_score_evtr_v2_weight",
          "fullrank_action_once_watchtime_score_plvtr_weight",
          "fullrank_action_once_watchtime_score_pfintr_weight",
          "fullrank_action_once_watchtime_score_pvtr_weight",
          "fullrank_action_once_watchtime_score_trans_pvtr_weight",
          "fullrank_action_once_interact_score_pltr_weight",
          "fullrank_action_once_interact_score_pwtr_weight",
          "fullrank_action_once_interact_score_pftr_weight",
          "fullrank_action_once_interact_score_pcmtr_weight",
          "fullrank_action_once_interact_score_pcmef_weight",
          "fullrank_action_once_interact_score_pptr_weight",
          "fullrank_action_once_interact_score_pepstr_weight",
          "fullrank_action_once_interact_score_lstr_weight",
          "fullrank_action_once_interact_score_pcltr_weight",
          "fullrank_action_once_interact_score_phtr_weight",
        ],
        import_item_attr = [
          "fullrank_sim_pevtr",
          "fullrank_sim_pltr",
          "fullrank_sim_pwtr",
          "fullrank_sim_pftr",
          "fullrank_sim_pptr",
          "fullrank_sim_pcmtr",
          "fullrank_sim_pcmef",
          "fullrank_sim_pvtr",
          "fullrank_sim_pepstr",
          "fullrank_sim_lstr",
          "fullrank_detail_new_pevtr_v2",
          "fullrank_trans_pvtr_score",
          "fullrank_sim_pcltr",
          "fullrank_sim_plvtr",
          "fullrank_sim_pfintr",
          "fullrank_sim_phtr"
        ],
        export_item_attr = [
          "fullrank_action_once_interact_score",
          "fullrank_action_once_watchtime_score",
        ],
        function_for_item = "cal_action_once_score",
        lua_script_file = "fountain/full_rank/lua/cal_opportunity_cost_score.lua",
      )
    return self
  
  def fullrank_calc_action_calibration_fusion_score(self):
    self \
    .calc_by_formula1(
      kconf_key = "formula.scenarioKey27.FrFountainActionRecalibrationFuse",
      import_item_attr = [
        "fullrank_sim_like_score",
        "fullrank_sim_follow_score",
        "fullrank_sim_pcmtr",
        "fullrank_sim_pcltr"
      ],
      export_formula_value = [
        {"name": "final_score", "as": "fullrank_action_recalibration_fusion_score"}
      ],
      abtest_biz_name = "KUAISHOU_APPS"
    )
    return self

  def fullrank_calc_pvtr_derive_watchtime_score(self):
    self \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_fullrank_pvtr_derive_watchtime_score_upper_bound", "as": "derived_watchtime_upper_bound"},
        {"name": "fountain_fullrank_pvtr_derive_watchtime_score_lower_bound", "as": "derived_watchtime_lower_bound"}
      ],
      import_item_attr = [
        {"name": "fullrank_sim_pvtr", "as": "pvtr"},
      ],
      export_item_attr = [
        {"name": "derived_watchtime_score", "as": "fullrank_sim_pvtr_derive_watchtime_score"}
      ],
      function_name = "CalcDerivedWatchtimeScore",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def fountain_fullrank_calc_same_hetu_compensate_score(self):
    self \
    .calc_weighted_sum(
      channels = [
        {
          "name": "duration_ms",
          "weight": "{{fountain_fullrank_same_hetu_long_video_compensate_duration_weight}}"
        },
        {
          "name": "fullrank_sim_pfintr",
          "weight": "{{fountain_fullrank_same_hetu_long_video_compensate_wtd_weight}}"
        },
      ],
      output_item_attr = "duration_combine_wtd_score",
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        "duration_combine_wtd_score",
        "hetu_tag_level_info__hetu_level_two"
      ],
      export_item_attr = [
        "fullrank_same_hetu_long_video_compensate_score"
      ],
      function_name = "CalcSameHetuLongVideoScore",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def cal_request_pxtr_ada_weight(self):
    self \
      .split_string(
        input_common_attr = "fountain_fullrank_ensemble_req_adjust_ratio_min_str",
        output_common_attr = "fountain_fullrank_ensemble_req_adjust_ratio_min_list",
        delimiters = ",",
        parse_to_double = True,
      ) \
      .split_string(
        input_common_attr = "fountain_fullrank_ensemble_req_adjust_ratio_max_str",
        output_common_attr = "fountain_fullrank_ensemble_req_adjust_ratio_max_list",
        delimiters = ",",
        parse_to_double = True,
      ) \
      .log_debug_info(
        common_attrs = [
          "fountain_ensemble_power_weight_fullrank_click_score",
          "fountain_ensemble_weight_fullrank_detail_new_pevtr_v2",
          "fountain_ensemble_power_weight_fullrank_longview_score",
          "fountain_ensemble_power_weight_fullrank_pfintr_score",
          "fountain_ensemble_power_weight_fullrank_pvtr_score",
        ],
        for_debug_request_only = True,
      ) \
      .enrich_attr_by_lua(
          import_common_attr = [
            "fountain_ensemble_power_weight_fullrank_click_score",
            "fountain_ensemble_weight_fullrank_detail_new_pevtr_v2",
            "fountain_ensemble_power_weight_fullrank_longview_score",
            "fountain_ensemble_power_weight_fullrank_pfintr_score",
            "fountain_ensemble_power_weight_fullrank_pvtr_score",
            "fountain_fullrank_ensemble_req_adjust_ratio_min_list",
            "fountain_fullrank_ensemble_req_adjust_ratio_max_list",
            "fountain_fullrank_ensemble_req_adjust_ratio_pow_w",
            "fountain_fullrank_ensemble_req_adjust_ratio_bias",
            "fountain_fullrank_ensemble_req_pxtr_dev_ada_rank_smooth",
            "fountain_fullrank_req_pxtr_ada_dev_switch",
            "fountain_fullrank_req_pxtr_dev_ada_weights",
            "user_emp_evtr",
            "user_emp_lvtr",
            "user_emp_watch_time",
            "pevtr_avg",
            "pevtr_v2_avg",
            "plvtr_avg",
            "pfintr_avg",
            "pwatchtime_avg",
          ],
          export_common_attr = [
            "fountain_ensemble_power_weight_fullrank_click_score",
            "fountain_ensemble_weight_fullrank_detail_new_pevtr_v2",
            "fountain_ensemble_power_weight_fullrank_longview_score",
            "fountain_ensemble_power_weight_fullrank_pfintr_score",
            "fountain_ensemble_power_weight_fullrank_pvtr_score",
          ],
          function_for_common = "cal_request_pxtr_ada_weight",
          lua_script_file = "fountain/full_rank/lua/cal_adaptive_weight.lua",
        ) \
      .perflog_attr_value(
        check_point = "fullrank_ada_weight",
        common_attrs = [
          "pevtr_avg",
          "pevtr_v2_avg",
          "plvtr_avg",
          "pfintr_avg",
          "pwatchtime_avg",
          "user_emp_evtr",
          "user_emp_lvtr",
          "user_emp_watch_time",
        ],
      ) \
      .log_debug_info(
        common_attrs = [
          "userFtRealtimeCountTimeThreshold",
          "fountain_ensemble_power_weight_fullrank_click_score",
          "fountain_ensemble_weight_fullrank_detail_new_pevtr_v2",
          "fountain_ensemble_power_weight_fullrank_longview_score",
          "fountain_ensemble_power_weight_fullrank_pfintr_score",
          "fountain_ensemble_power_weight_fullrank_pvtr_score",
          "fountain_fullrank_ensemble_hate_rank_cliff_attr",
          "fountain_fullrank_ensemble_hate_rank_cliff_ratio_attr",
          "fountain_fullrank_ensemble_hate_rank_height_attr",
          "fountain_fullrank_ensemble_hate_cliff_score_bias_attr",
          "user_emp_evtr",
          "user_emp_lvtr",
          "user_emp_watch_time",
          "pevtr_avg",
          "pevtr_v2_avg",
          "plvtr_avg",
          "pfintr_avg",
          "pwatchtime_avg",
        ],
        for_debug_request_only = True,
      ) \

    return self
  
  def request_xtr_adap_weight(self):
    self \
    .pack_item_attr(
      item_source={
        "reco_results": True,
      },
      mappings=[
        {
          "aggregator": "avg",
          "from_item_attr": "fullrank_sim_pevtr",
          "to_common_attr": "pevtr_avg_rank"
        },
        {
          "aggregator": "avg",
          "from_item_attr": "fullrank_sim_pltr",
          "to_common_attr": "pltr_avg_rank"
        },
        {
          "aggregator": "avg",
          "from_item_attr": "fullrank_sim_pwtr",
          "to_common_attr": "pwtr_avg_rank"
        },
        {
          "aggregator": "avg",
          "from_item_attr": "fullrank_sim_pftr",
          "to_common_attr": "pftr_avg_rank"
        },
        {
          "aggregator": "avg",
          "from_item_attr": "fullrank_sim_pcmtr",
          "to_common_attr": "pcmtr_avg_rank"
        },
        {
          "aggregator": "avg",
          "from_item_attr": "fullrank_sim_pptr",
          "to_common_attr": "pptr_avg_rank"
        },
        {
          "aggregator": "avg",
          "from_item_attr": "fullrank_sim_pepstr",
          "to_common_attr": "pepstr_avg_rank"
        },
        {
          "aggregator": "avg",
          "from_item_attr": "fullrank_sim_plvtr",
          "to_common_attr": "plvtr_avg_rank"
        },
        {
          "aggregator": "avg",
          "from_item_attr": "fullrank_sim_pfintr",
          "to_common_attr": "pfintr_avg_rank"
        },
      ]
    ) \
    .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_weight_adjust_avg_pevtr_base_stat", "as": "user_base_stat"},
          {"name": "pevtr_avg_rank", "as": "user_dynamic_stat"},
          {"name": "fountain_weight_adjust_avg_boost_coef_lower", "as": "boost_coef_lower"},
          {"name": "fountain_weight_adjust_avg_boost_coef_upper", "as": "boost_coef_upper"},
          {"name": "fountain_weight_adjust_avg_is_boost", "as": "is_boost"},
          {"name": "fountain_weight_adjust_avg_action_power_weight", "as": "action_power_weight"}
        ],
        export_common_attr = [
          {"name": "user_dynamic_action", "as": "rank_boost_pevtr"}
        ],
        function_name = "CalcUserDynamicAction",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_weight_adjust_avg_pltr_base_stat", "as": "user_base_stat"},
          {"name": "pltr_avg_rank", "as": "user_dynamic_stat"},
          {"name": "fountain_weight_adjust_avg_boost_coef_lower", "as": "boost_coef_lower"},
          {"name": "fountain_weight_adjust_avg_boost_coef_upper", "as": "boost_coef_upper"},
          {"name": "fountain_weight_adjust_avg_is_boost", "as": "is_boost"},
          {"name": "fountain_weight_adjust_avg_action_power_weight", "as": "action_power_weight"}
        ],
        export_common_attr = [
          {"name": "user_dynamic_action", "as": "rank_boost_pltr"}
        ],
        function_name = "CalcUserDynamicAction",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_weight_adjust_avg_pwtr_base_stat", "as": "user_base_stat"},
          {"name": "pwtr_avg_rank", "as": "user_dynamic_stat"},
          {"name": "fountain_weight_adjust_avg_boost_coef_lower", "as": "boost_coef_lower"},
          {"name": "fountain_weight_adjust_avg_boost_coef_upper", "as": "boost_coef_upper"},
          {"name": "fountain_weight_adjust_avg_is_boost", "as": "is_boost"},
          {"name": "fountain_weight_adjust_avg_action_power_weight", "as": "action_power_weight"}
        ],
        export_common_attr = [
          {"name": "user_dynamic_action", "as": "rank_boost_pwtr"}
        ],
        function_name = "CalcUserDynamicAction",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_weight_adjust_avg_pftr_base_stat", "as": "user_base_stat"},
          {"name": "pftr_avg_rank", "as": "user_dynamic_stat"},
          {"name": "fountain_weight_adjust_avg_boost_coef_lower", "as": "boost_coef_lower"},
          {"name": "fountain_weight_adjust_avg_boost_coef_upper", "as": "boost_coef_upper"},
          {"name": "fountain_weight_adjust_avg_is_boost", "as": "is_boost"},
          {"name": "fountain_weight_adjust_avg_action_power_weight", "as": "action_power_weight"}
        ],
        export_common_attr = [
          {"name": "user_dynamic_action", "as": "rank_boost_pftr"}
        ],
        function_name = "CalcUserDynamicAction",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_weight_adjust_avg_pcmtr_base_stat", "as": "user_base_stat"},
          {"name": "pcmtr_avg_rank", "as": "user_dynamic_stat"},
          {"name": "fountain_weight_adjust_avg_boost_coef_lower", "as": "boost_coef_lower"},
          {"name": "fountain_weight_adjust_avg_boost_coef_upper", "as": "boost_coef_upper"},
          {"name": "fountain_weight_adjust_avg_is_boost", "as": "is_boost"},
          {"name": "fountain_weight_adjust_avg_action_power_weight", "as": "action_power_weight"}
        ],
        export_common_attr = [
          {"name": "user_dynamic_action", "as": "rank_boost_pcmtr"}
        ],
        function_name = "CalcUserDynamicAction",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_weight_adjust_avg_pptr_base_stat", "as": "user_base_stat"},
          {"name": "pptr_avg_rank", "as": "user_dynamic_stat"},
          {"name": "fountain_weight_adjust_avg_boost_coef_lower", "as": "boost_coef_lower"},
          {"name": "fountain_weight_adjust_avg_boost_coef_upper", "as": "boost_coef_upper"},
          {"name": "fountain_weight_adjust_avg_is_boost", "as": "is_boost"},
          {"name": "fountain_weight_adjust_avg_action_power_weight", "as": "action_power_weight"}
        ],
        export_common_attr = [
          {"name": "user_dynamic_action", "as": "rank_boost_pptr"}
        ],
        function_name = "CalcUserDynamicAction",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_weight_adjust_avg_pepstr_base_stat", "as": "user_base_stat"},
          {"name": "pepstr_avg_rank", "as": "user_dynamic_stat"},
          {"name": "fountain_weight_adjust_avg_boost_coef_lower", "as": "boost_coef_lower"},
          {"name": "fountain_weight_adjust_avg_boost_coef_upper", "as": "boost_coef_upper"},
          {"name": "fountain_weight_adjust_avg_is_boost", "as": "is_boost"},
          {"name": "fountain_weight_adjust_avg_action_power_weight", "as": "action_power_weight"}
        ],
        export_common_attr = [
          {"name": "user_dynamic_action", "as": "rank_boost_pepstr"}
        ],
        function_name = "CalcUserDynamicAction",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_weight_adjust_avg_pfintr_base_stat", "as": "user_base_stat"},
          {"name": "pfintr_avg_rank", "as": "user_dynamic_stat"},
          {"name": "fountain_weight_adjust_avg_boost_coef_lower", "as": "boost_coef_lower"},
          {"name": "fountain_weight_adjust_avg_boost_coef_upper", "as": "boost_coef_upper"},
          {"name": "fountain_weight_adjust_avg_is_boost", "as": "is_boost"},
          {"name": "fountain_weight_adjust_avg_action_power_weight", "as": "action_power_weight"}
        ],
        export_common_attr = [
          {"name": "user_dynamic_action", "as": "rank_boost_pfintr"}
        ],
        function_name = "CalcUserDynamicAction",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_weight_adjust_avg_plvtr_base_stat", "as": "user_base_stat"},
          {"name": "plvtr_avg_rank", "as": "user_dynamic_stat"},
          {"name": "fountain_weight_adjust_avg_boost_coef_lower", "as": "boost_coef_lower"},
          {"name": "fountain_weight_adjust_avg_boost_coef_upper", "as": "boost_coef_upper"},
          {"name": "fountain_weight_adjust_avg_is_boost", "as": "is_boost"},
          {"name": "fountain_weight_adjust_avg_action_power_weight", "as": "action_power_weight"}
        ],
        export_common_attr = [
          {"name": "user_dynamic_action", "as": "rank_boost_plvtr"}
        ],
        function_name = "CalcUserDynamicAction",
        class_name = "ExploreLightFunctionSetV2",
      )
    return self
  
  def cal_request_rank_weight_adjust(self):
    self.gen_common_attr_by_lua(
        attr_map={
          "fountain_ensemble_power_weight_fullrank_like_score": "rank_boost_pltr * fountain_ensemble_power_weight_fullrank_like_score",
          "fountain_ensemble_power_weight_fullrank_follow_score": "rank_boost_pwtr * fountain_ensemble_power_weight_fullrank_follow_score",
          "fountain_ensemble_power_weight_fullrank_pcmtr_score": "rank_boost_pcmtr * fountain_ensemble_power_weight_fullrank_pcmtr_score",
          "fountain_ensemble_power_weight_fullrank_pptr_score": "rank_boost_pptr * fountain_ensemble_power_weight_fullrank_pptr_score",
          "fountain_ensemble_weight_forward_score": "rank_boost_pftr * fountain_ensemble_weight_forward_score",
          "fountain_ensemble_power_weight_fullrank_pepstr_score": "rank_boost_pepstr * fountain_ensemble_power_weight_fullrank_pepstr_score",
          "fountain_ensemble_power_weight_fullrank_click_score": "rank_boost_pevtr * fountain_ensemble_power_weight_fullrank_click_score",
          "fountain_ensemble_weight_fullrank_sim_plvtr": "rank_boost_plvtr * fountain_ensemble_weight_fullrank_sim_plvtr",
          "fountain_ensemble_power_weight_fullrank_pfintr_score": "rank_boost_pfintr * fountain_ensemble_power_weight_fullrank_pfintr_score",
        }
      )
    return self
  
  def cal_request_adaptive_score(self):
    self \
      .pack_item_attr(
        item_source = {
          "reco_results": True,
        },
        mappings = [
          {
            "aggregator": "dev",
            "from_item_attr": "fullrank_sim_click_score",
            "to_common_attr": "pctr_dev"
          },
          {
            "aggregator": "dev",
            "from_item_attr": "fullrank_sim_pltr",
            "to_common_attr": "pltr_dev"
          },
          {
            "aggregator": "dev",
            "from_item_attr": "fullrank_sim_pwtr",
            "to_common_attr": "pwtr_dev"
          },
          {
            "aggregator": "dev",
            "from_item_attr": "fullrank_sim_pftr",
            "to_common_attr": "pftr_dev"
          },
          {
            "aggregator": "dev",
            "from_item_attr": "fullrank_sim_pcmtr",
            "to_common_attr": "pcmtr_dev"
          },
          {
            "aggregator": "dev",
            "from_item_attr": "fullrank_detail_new_pevtr_v2",
            "to_common_attr": "pevtr_v2_dev"
          },
        ]
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "pctr_avg",
          "pltr_avg",
          "pwtr_avg",
          "pftr_avg",
          "pcmtr_avg",
          "pevtr_v2_avg",
          "pctr_dev",
          "pltr_dev",
          "pwtr_dev",
          "pftr_dev",
          "pcmtr_dev",
          "pevtr_v2_dev",
          "fountain_ensemble_power_weight_fullrank_click_score",
          "fountain_ensemble_power_weight_fullrank_like_score",
          "fountain_ensemble_power_weight_fullrank_follow_score",
          "fountain_ensemble_weight_forward_score",
          "fountain_ensemble_power_weight_fullrank_pcmtr_score",
          "fountain_ensemble_weight_fullrank_detail_new_pevtr_v2",
          "fountain_fullrank_request_adative_ratio_min",
          "fountain_fullrank_request_adative_ratio_max",
          "fountain_fullrank_emp_pctr_adaptive_coef",
          "fountain_fullrank_emp_pltr_adaptive_coef",
          "fountain_fullrank_emp_pwtr_adaptive_coef",
          "fountain_fullrank_emp_pftr_adaptive_coef",
          "fountain_fullrank_emp_pcmtr_adaptive_coef",
          "fountain_fullrank_emp_pevtr_adaptive_coef",
        ],
        import_item_attr = [
          "fullrank_sim_click_score",
          "fullrank_sim_pltr",
          "fullrank_sim_pwtr",
          "fullrank_sim_pftr",
          "fullrank_sim_pcmtr",
          "fullrank_detail_new_pevtr_v2",
        ],
        export_item_attr = [
          "fullrank_request_adaptive_score",
        ],
        function_name = "CalRequestAdaptiveScore",
        class_name = "ExploreLightFunctionSetV2",
      ) \

    return self

  def use_hierarchical_es(self):
    self \
      .fountain_calc_ensemble_score(
        save_score_to_attr = "fullrank_act_raw_score",
        use_dist_calc = "{{fountain_fullrank_ensemble_use_dist_calc}}",
        dis_factor = "{{fountain_fullrank_ensemble_dis_factor}}",
        user_new_proportion = 0,
        user_power_calc = 0,
        user_power_calc_v2 = 0,
        use_xtr_raw_score = 1,
        queues = [
          {
            "name": "fullrank_sim_pcltr",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_raw_power_weight_pcltr",
          },
          {
            "name": "fullrank_sim_like_score",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_raw_power_weight_like",
          },
          {
            "name": "fullrank_sim_pcmtr",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_raw_power_weight_pcmtr",
          },
          {
            "name": "fullrank_sim_pcmef",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_raw_power_weight_pcmef",
          },
          {
            "name": "fullrank_sim_lstr",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_raw_power_weight_lstr",
          },
          {
            "name": "fullrank_sim_pepstr",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_raw_power_weight_pepstr",
          },
          {
            "name": "fullrank_sim_follow_score",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_raw_power_weight_follow",
          },
          {
            "name": "fullrank_detail_new_pevtr_v2",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_raw_power_weight_normalized_pevtr_v2",
          },
          {
            "name": "fullrank_sim_pvtr",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_raw_power_weight_normalized_pvtr",
          },
          {
            "name": "fullrank_sim_plvtr",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_raw_power_weight_normalized_plvtr",
          },
          {
            "name": "fullrank_sim_psvr",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_raw_power_weight_normalized_psvr_reversed",
          },
          {
            "name": "fullrank_sim_pfintr",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_raw_power_weight_normalized_pfintr",
          },
        ],
      ) \
      .fountain_calc_ensemble_score(
        save_score_to_attr = "fullrank_wt_rank_score",
        use_dist_calc = "{{fountain_fullrank_ensemble_use_dist_calc}}",
        dis_factor = "{{fountain_fullrank_ensemble_dis_factor}}",
        user_new_proportion = "{{fountain_fullrank_hierarchical_es_user_new_proportion}}",
        user_power_calc = "{{fountain_fullrank_hierarchical_es_user_power_calc}}",
        use_reciprocal = "{{fountain_fullrank_hierarchical_es_use_reciprocal}}",
        rank_smooth = "{{fountain_fullrank_hier_es_wt_rank_score_rank_smooth}}",
        user_power_calc_v2 = "{{fountain_fullrank_hierarchical_es_user_power_calc_v2}}",
        use_xtr_raw_score = "{{fountain_fullrank_hierarchical_es_use_xtr_raw_score}}",
        queues = [
          {
            "name": "fullrank_sim_longview_score_no_bias",
            "weight": 0.0,
            "temperature_attr": "fountain_fullrank_hier_es_temperature_attr",
            "weight_attr": "fountain_ensemble_power_weight_fullrank_longview_score",
            "power_weight_attr": "fountain_ensemble_power_weight_fullrank_longview_score",
          },
          {
            "name": "fullrank_sim_pfintr",
            "weight": 0.0,
            "temperature_attr": "fountain_fullrank_hier_es_temperature_attr",
            "weight_attr": "fountain_ensemble_power_weight_fullrank_pfintr_score",
            "power_weight_attr": "fountain_ensemble_power_weight_fullrank_pfintr_score",
          },
          {
            "name": "fullrank_sim_plvtr",
            "weight": 0.0,
            "temperature_attr": "fountain_fullrank_hier_es_temperature_attr",
            "weight_attr": "fountain_ensemble_weight_fullrank_sim_plvtr",
            "power_weight_attr": "fountain_ensemble_weight_fullrank_sim_plvtr",
          },
          {
            "name": "fullrank_sim_pwatchtime_no_bias",
            "weight": 0.0,
            "temperature_attr": "fountain_fullrank_hier_es_temperature_attr",
            "weight_attr": "fountain_ensemble_power_weight_fullrank_pvtr_score",
            "power_weight_attr": "fountain_ensemble_power_weight_fullrank_pvtr_score",
          },
        ],
      ) \
      .fountain_calc_ensemble_score(
        save_score_to_attr = "fullrank_act_rank_score",
        use_dist_calc = "{{fountain_fullrank_ensemble_use_dist_calc}}",
        dis_factor = "{{fountain_fullrank_ensemble_dis_factor}}",
        user_new_proportion = "{{fountain_fullrank_hierarchical_es_user_new_proportion}}",
        user_power_calc = "{{fountain_fullrank_hierarchical_es_user_power_calc}}",
        use_reciprocal = "{{fountain_fullrank_hierarchical_es_use_reciprocal}}",
        user_power_calc_v2 = "{{fountain_fullrank_hierarchical_es_user_power_calc_v2}}",
        use_xtr_raw_score = "{{fountain_fullrank_hierarchical_es_use_xtr_raw_score}}",
        rank_smooth = "{{fountain_fullrank_hier_es_act_rank_score_rank_smooth}}",
        queues = [
          {
            "name": "fullrank_sim_like_score",
            "weight": 0.0,
            "temperature_attr": "fountain_fullrank_hier_es_temperature_attr",
            "weight_attr": "fountain_ensemble_power_weight_fullrank_like_score",
            "power_weight_attr": "fountain_ensemble_power_weight_fullrank_like_score",
          },
          {
            "name": "fullrank_sim_pcmtr",
            "weight": 0.0,
            "temperature_attr": "fountain_fullrank_hier_es_temperature_attr",
            "weight_attr": "fountain_ensemble_power_weight_fullrank_pcmtr_score",
            "power_weight_attr": "fountain_ensemble_power_weight_fullrank_pcmtr_score",
          },
          {
            "name": "fullrank_sim_pcmef",
            "weight": 0.0,
            "temperature_attr": "fountain_fullrank_hier_es_temperature_attr",
            "weight_attr": "fountain_ensemble_power_weight_fullrank_cmef_score",
            "power_weight_attr": "fountain_ensemble_power_weight_fullrank_cmef_score",
          },
          {
            "name": "fullrank_sim_lstr",
            "weight": 0.0,
            "temperature_attr": "fountain_fullrank_hier_es_temperature_attr",
            "weight_attr": "fountain_ensemble_power_weight_fullrank_lstr_score",
            "power_weight_attr": "fountain_ensemble_power_weight_fullrank_lstr_score",
          },
          {
            "name": "fullrank_sim_pepstr",
            "weight": 0.0,
            "temperature_attr": "fountain_fullrank_hier_es_temperature_attr",
            "weight_attr": "fountain_ensemble_power_weight_fullrank_pepstr_score",
            "power_weight_attr": "fountain_ensemble_power_weight_fullrank_pepstr_score",
          },
          {
            "name": "fullrank_sim_follow_score",
            "weight": 0.0,
            "temperature_attr": "fountain_fullrank_hier_es_temperature_attr",
            "weight_attr": "fountain_ensemble_power_weight_fullrank_follow_score",
            "power_weight_attr": "fountain_ensemble_power_weight_fullrank_follow_score",
          },
          {
            "name": "fullrank_sim_pftr",
            "weight": 0.0,
            "temperature_attr": "fountain_fullrank_hier_es_temperature_attr",
            "weight_attr": "fountain_ensemble_weight_forward_score",
            "power_weight_attr": "fountain_ensemble_weight_forward_score",
          },
          {
            "name": "fullrank_sim_pcltr",
            "weight": 0.0,
            "temperature_attr": "fountain_fullrank_hier_es_temperature_attr",
            "weight_attr": "fountain_ensemble_power_weight_fullrank_pcltr_score",
            "power_weight_attr": "fountain_ensemble_power_weight_fullrank_pcltr_score",
          },
        ],
      ) \
      .fountain_calc_ensemble_score(
        save_score_to_attr = "fullrank_vv_rank_score",
        use_dist_calc = "{{fountain_fullrank_ensemble_use_dist_calc}}",
        dis_factor = "{{fountain_fullrank_ensemble_dis_factor}}",
        user_new_proportion = "{{fountain_fullrank_hierarchical_es_user_new_proportion}}",
        user_power_calc = "{{fountain_fullrank_hierarchical_es_user_power_calc}}",
        use_reciprocal = "{{fountain_fullrank_hierarchical_es_use_reciprocal}}",
        user_power_calc_v2 = "{{fountain_fullrank_hierarchical_es_user_power_calc_v2}}",
        use_xtr_raw_score = "{{fountain_fullrank_hierarchical_es_use_xtr_raw_score}}",
        rank_smooth = "{{fountain_fullrank_hier_es_vv_rank_score_rank_smooth}}",
        queues = [
          {
            "name": "fullrank_act_ctr",
            "weight": 0.0,
            "temperature_attr": "fountain_fullrank_hier_es_temperature_attr",
            "weight_attr": "fountain_ensemble_weight_fullrank_act_ctr",
            "power_weight_attr": "fountain_ensemble_weight_fullrank_act_ctr",
          },
          {
            "name": "fullrank_cl_tran_score",
            "weight": 0.0,
            "temperature_attr": "fountain_fullrank_hier_es_temperature_attr",
            "weight_attr": "fountain_ensemble_power_weight_fullrank_cl_score",
            "power_weight_attr": "fountain_ensemble_power_weight_fullrank_cl_score",
          },
          {
            "name": "fullrank_hate_similary_score",
            "weight": 0.0,
            "temperature_attr": "fountain_fullrank_hier_es_temperature_attr",
            "weight_attr": "fountain_ensemble_power_weight_fullrank_hate_similary_score",
            "power_weight_attr": "fountain_ensemble_power_weight_fullrank_hate_similary_score",
          },
          {
            "name": "fullrank_detail_new_pevtr_v2",
            "weight": 0.0,
            "temperature_attr": "fountain_fullrank_hier_es_temperature_attr",
            "weight_attr": "fountain_ensemble_weight_fullrank_detail_new_pevtr_v2",
            "power_weight_attr": "fountain_ensemble_weight_fullrank_detail_new_pevtr_v2",
          },
          {
            "name": "fullrank_sim_click_score",
            "weight": 0.0,
            "temperature_attr": "fountain_fullrank_hier_es_temperature_attr",
            "weight_attr": "fountain_ensemble_power_weight_fullrank_click_score",
            "power_weight_attr" : "fountain_ensemble_power_weight_fullrank_click_score",
          },
          {
            "name": "fullrank_ltr_v4_fountain_next",
            "weight": 0.0,
            "temperature_attr": "fountain_fullrank_hier_es_temperature_attr",
            "weight_attr": "fountain_ensemble_power_weight_fullrank_ltr_v4_next",
            "power_weight_attr": "fountain_ensemble_power_weight_fullrank_ltr_v4_next",
          },
          {
            "name": "fullrank_sim_psvr",
            "weight": -0.35,
            "temperature_attr": "fountain_fullrank_hier_es_temperature_attr",
            "weight_attr": "fountain_ensemble_power_weight_fullrank_svr_in_order_score",
            "power_weight_attr": "fountain_ensemble_power_weight_fullrank_svr_in_order_score",
          },
        ],
      ) \
      .sort(
        score_from_attr = "fullrank_ensemble_score",
      ) \
      .copy_item_meta_info(
        save_item_seq_to_attr = "fullrank_ensemble_rank",
      ) \
      .enrich_attr_by_lua(
        import_common_attr = [
          "fountain_fullrank_hierarchical_es_es_seq_score_pow_weight",
          "fountain_fullrank_xtr_ensemble_fusion_way",
          "fountian_fullrank_xtr_raw_score_pow_weight",
          "fountain_fullrank_hierarchical_es_wt_rank_score_pow_weight",
          "fountain_fullrank_hierarchical_es_act_rank_score_pow_weight",
          "fountain_fullrank_hierarchical_es_vv_rank_score_pow_weight",
          "fullrank_ensemble_seq_rank_smooth",
        ],
        import_item_attr = [
          "fullrank_ensemble_score",
          "fullrank_ensemble_rank",
          "fullrank_act_raw_score",
          "fullrank_wt_rank_score",
          "fullrank_act_rank_score",
          "fullrank_vv_rank_score",
        ],
        export_item_attr = [
          "fullrank_ensemble_score",
        ],
        function_for_item = "calc_hierarchical_es_score",
        lua_script_file = "fountain/full_rank/lua/calc_hierarchical_es_score.lua",) \
      .log_debug_info(
        item_attrs = [
          "item_id",
          "fullrank_ensemble_score",
          "fullrank_ensemble_rank",
          "fullrank_act_raw_score",
          "fullrank_sim_pwatchtime_no_bias",
          "fullrank_wt_rank_score",
          "fullrank_act_rank_score",
          "fullrank_vv_rank_score",
          "fullrank_min_act_rank_reci",
          "fullrank_act_ctr",
          "fullrank_cl_tran_score",
          "fullrank_hate_similary_score",
        ],
        item_num_limit = 10,
        for_debug_request_only = True,
      )
    return self

  def raw_weighted_score_es(self):
    self \
      .fountain_calc_ensemble_score(
        save_score_to_attr="fullrank_act_weight_raw_score",
        use_xtr_raw_score=0,
        use_xtr_weight_raw_score=1,
        queues=[
          {
            "name": "fullrank_sim_pcltr",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_weight_raw_power_weight_pcltr",
            "weight_attr": "fountain_ensemble_xtr_weight_raw_weight_pcltr",
          },
          {
            "name": "fullrank_sim_like_score",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_weight_raw_power_weight_like",
            "weight_attr": "fountain_ensemble_xtr_weight_raw_weight_like",
          },
          {
            "name": "fullrank_sim_pcmtr",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_weight_raw_power_weight_pcmtr",
            "weight_attr": "fountain_ensemble_xtr_weight_raw_weight_pcmtr",
          },
          {
            "name": "fullrank_sim_pftr",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_weight_raw_power_weight_pftr",
            "weight_attr": "fountain_ensemble_xtr_weight_raw_weight_forward",
          },
          {
            "name": "fullrank_sim_pcmef",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_weight_raw_power_weight_pcmef",
            "weight_attr": "fountain_ensemble_xtr_weight_raw_weight_pcmef",
          },
          {
            "name": "fullrank_sim_lstr",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_weight_raw_power_weight_lstr",
            "weight_attr": "fountain_ensemble_xtr_weight_raw_weight_lstr",
          },
          {
            "name": "fullrank_sim_pepstr",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_weight_raw_power_weight_pepstr",
            "weight_attr": "fountain_ensemble_xtr_weight_raw_weight_pepstr",
          },
          {
            "name": "fullrank_sim_follow_score",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_weight_raw_power_weight_follow",
            "weight_attr": "fountain_ensemble_xtr_weight_raw_weight_follow",
          },
          {
            "name": "fullrank_detail_new_pevtr_v2",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_weight_raw_power_weight_normalized_pevtr_v2",
            "weight_attr": "fountain_ensemble_xtr_weight_raw_weight_normalized_pevtr_v2",
          },
          {
            "name": "fullrank_sim_pvtr",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_weight_raw_power_weight_normalized_pvtr",
            "weight_attr": "fountain_ensemble_xtr_weight_raw_weight_normalized_pvtr",
          },
          {
            "name": "fullrank_sim_plvtr",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_weight_raw_power_weight_normalized_plvtr",
            "weight_attr": "fountain_ensemble_xtr_weight_raw_weight_normalized_plvtr",
          },
          {
            "name": "fullrank_sim_psvr",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_weight_raw_power_weight_normalized_psvr_reversed",
            "weight_attr": "fountain_ensemble_xtr_weight_raw_weight_normalized_psvr_reversed",
          },
          {
            "name": "fullrank_sim_pfintr",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_weight_raw_power_weight_normalized_pfintr",
            "weight_attr": "fountain_ensemble_xtr_weight_raw_weight_normalized_pfintr",
          },
          {
            "name": "fullrank_ltr_v4_fountain_next",
            "weight": 0.0,
            "power_weight_attr": "fountain_ensemble_xtr_weight_raw_power_weight_ltr_v4_fountain_next",
            "weight_attr": "fountain_ensemble_xtr_weight_raw_weight_ltr_v4_fountain_next",
          },
        ],
      ) \
      .sort(
        score_from_attr="fullrank_ensemble_score",
      ) \
      .copy_attr(
        attrs=[{
          "from_item": "fullrank_act_weight_raw_score",
          "to_item": "fullrank_ensemble_score"
        }]
      )
    return self

  def fullrank_multi_stage(self):
    self \
      .enrich_attr_by_lua(
        import_common_attr = [
          "fountain_fullrank_fast_watchtime_limit_size",
          "fountain_fullrank_splash_watchtime_limit_size",
          "fountain_fullrank_fast_interact_limit_size",
          "fountain_fullrank_splash_interact_limit_size",
          "fountain_fullrank_fast_vv_limit_size",
          "fountain_fullrank_splash_vv_limit_size",
          "page"
        ],
        export_common_attr = [
          "fountain_fullrank_watchtime_limit_size",
          "fountain_fullrank_interact_limit_size",
          "fountain_fullrank_vv_limit_size"
        ],
        function_for_common = "cal_multi_stage_size_limit",
        lua_script_file = "fountain/full_rank/lua/cal_adaptive_weight.lua",
      ) \
      .fountain_calc_ensemble_score(
        save_score_to_attr = "fullrank_stage3_vv_score",
        user_new_proportion = "{{fountain_fullrank_vv_stage_new_proportion}}",
        user_power_calc = "{{fountain_fullrank_vv_stage_power_calc}}",
        use_reciprocal = "{{fountain_fullrank_vv_stage_use_reciprocal}}",
        min_rank_weight = "{{fountain_fr_fullrank_min_rank_weight}}",
        queues = fullrank_ensemble_interact_queues
      ) \
      .sort(
        score_from_attr = "fullrank_stage3_vv_score",
      ) \
      .truncate(
        size_limit = "{{fountain_fullrank_vv_limit_size}}",
      ) \
      .fountain_calc_ensemble_score(
        save_score_to_attr = "fullrank_stage2_interact_score",
        user_new_proportion = "{{fountain_fullrank_interact_stage_new_proportion}}",
        user_power_calc = "{{fountain_fullrank_interact_stage_power_calc}}",
        use_reciprocal = "{{fountain_fullrank_interact_stage_use_reciprocal}}",
        min_rank_weight = "{{fountain_fr_fullrank_min_rank_weight}}",
        queues = fullrank_ensemble_interact_queues
      ) \
      .sort(
        score_from_attr = "fullrank_stage2_interact_score",
      ) \
      .truncate(
        size_limit = "{{fountain_fullrank_interact_limit_size}}",
      ) \
        .fountain_calc_ensemble_score(
        save_score_to_attr = "fullrank_stage1_watchtime_score",
        user_new_proportion = "{{fountain_fullrank_watchtime_stage_new_proportion}}",
        user_power_calc = "{{fountain_fullrank_watchtime_stage_power_calc}}",
        use_reciprocal = "{{fountain_fullrank_watchtime_stage_use_reciprocal}}",
        queues = fullrank_ensemble_watchtime_queues
      ) \
      .sort(
        score_from_attr = "fullrank_stage1_watchtime_score",
      ) \
      .truncate(
        size_limit = "{{fountain_fullrank_watchtime_limit_size}}",
      )
    return self

  def calc_fullrank_ensemble_score(self):
    """
    计算 ensemble score
    """
    self \
    .enrich_attr_by_lua(
      skip = "{{skip_fullrank_user_adaptive_weight_cal}}",
      import_common_attr = [
        "fountain_fullrank_user_ada_pxtr_avg_weight",
        "enable_filter_fountain_ada_weight_lower_one",
        "enable_filter_fountain_ada_weight_over_one",
        "fullrank_splash_pre_filter_keep_photo_size",
        "fountain_ensemble_power_weight_fullrank_like_score",
        "fountain_ensemble_power_weight_fullrank_follow_score",
        "fountain_ensemble_power_weight_fullrank_pcmtr_score",
        "fountain_ensemble_power_weight_fullrank_pptr_score",
        "fountain_ensemble_power_weight_fullrank_pepstr_score",
        "fountain_ensemble_weight_forward_score",
        "fountain_ensemble_power_weight_fullrank_pvtr_multi_pwtr",
        "fountain_ensemble_power_weight_fullrank_pvtr_multi_pptr",
        "fountain_ensemble_weight_fullrank_pthanos_svr",
        "fountain_ensemble_power_weight_fullrank_svr_in_order_score",
        "fountain_ensemble_weight_fullrank_sim_longview_score_no_bias_debias",
        "fountain_ensemble_power_weight_adjust_ratio_min",
        "fountain_ensemble_power_weight_adjust_ratio_max",
        "fountain_ensemble_power_weight_fullrank_like_emp",
        "fountain_ensemble_power_weight_fullrank_follow_emp",
        "fountain_ensemble_power_weight_fullrank_pcmtr_emp",
        "fountain_ensemble_power_weight_fullrank_pptr_emp",
        "fountain_ensemble_power_weight_fullrank_psvtr_emp",
        "fountain_ensemble_power_weight_fullrank_plvtr_emp",
        "fountain_ensemble_power_weight_fullrank_forward_emp",
        "fountain_fullrank_ensemble_adaptive_interact_power_weight_with_emp_xtr",
        "fountain_fullrank_ensemble_use_absolute_score_queue_power_weight",
        "fountain_fullrank_ensemble_like_raw_pow_weight_attr",
        "fountain_fullrank_ensemble_follow_raw_pow_weight_attr",
        "fountain_fullrank_ensemble_comment_raw_pow_weight_attr",
        "fountain_fullrank_ensemble_pepstr_raw_pow_weight_attr",
        "fountain_fullrank_ensemble_pftr_raw_pow_weight_attr",
        "fountain_fullrank_ensemble_like_hyperbolic_raw_pow_weight_attr",
        "fountain_fullrank_ensemble_follow_hyperbolic_raw_pow_weight_attr",
        "fountain_fullrank_ensemble_pftr_hyperbolic_raw_pow_weight_attr",
        "userExpLtr",
        "userExpWtr",
        "userExpCmtr",
        "userExpPtr",
        "userExpSvtr",
        "userExpLvtr",
        "userExpFtr",
        "psvr_avg",
        "pltr_avg",
        "pwtr_avg",
        "pftr_avg",
        "pcmtr_avg",
        "pptr_avg",
        "plvtr_avg",
      ],
      export_common_attr = [
        "fountain_ensemble_power_weight_fullrank_like_score",
        "fountain_ensemble_power_weight_fullrank_follow_score",
        "fountain_ensemble_power_weight_fullrank_pcmtr_score",
        "fountain_ensemble_power_weight_fullrank_pptr_score",
        "fountain_ensemble_power_weight_fullrank_pepstr_score",
        "fountain_ensemble_power_weight_fullrank_pvtr_multi_pwtr",
        "fountain_ensemble_power_weight_fullrank_pvtr_multi_pptr",
        "fountain_ensemble_weight_fullrank_pthanos_svr",
        "fountain_ensemble_power_weight_fullrank_svr_in_order_score",
        "fountain_ensemble_weight_fullrank_sim_longview_score_no_bias_debias",
        "fountain_ensemble_weight_forward_score",
        "fountain_fullrank_ensemble_like_raw_pow_weight_attr",
        "fountain_fullrank_ensemble_follow_raw_pow_weight_attr",
        "fountain_fullrank_ensemble_comment_raw_pow_weight_attr",
        "fountain_fullrank_ensemble_pepstr_raw_pow_weight_attr",
        "fountain_fullrank_ensemble_pftr_raw_pow_weight_attr",
        "fountain_fullrank_ensemble_like_hyperbolic_raw_pow_weight_attr",
        "fountain_fullrank_ensemble_follow_hyperbolic_raw_pow_weight_attr",
        "fountain_fullrank_ensemble_pftr_hyperbolic_raw_pow_weight_attr",
      ],
      function_for_common = "cal_fullrank_adaptive_weights_v2",
      lua_script_file = "fountain/full_rank/lua/cal_adaptive_weight.lua",
    ) \
    .if_("enable_fountain_rank_s2_user_adaptive_weight_with_active_score == 1") \
      .user_adaptive_weight_with_active_score() \
    .end_() \
    .if_("enable_fountain_rank_s2_age_based_adjust == 1") \
      .user_age_based_weight_adjust_all() \
    .end_() \
    .fountain_sl_only_fast() \
    .fountain_pure_value_only_fast() \
    .boost_comment_weights() \
    .boost_low_follow_user_follow_weight() \
    .if_("fountain_enable_useful_author_revisit == 1") \
      .item_attr_operation(
        item_attr_a = "userfulness_author_score",
        common_attr_b = "{{fountain_useful_author_revisit_score_coef}}",
        operator = "*",
        output_attr = "userfulness_author_score_tur"
      ) \
      .item_attr_operation(
        item_attr_a = "user_author_ltv_model",
        item_attr_b = "userfulness_author_score_tur",
        operator = "+",
        output_attr = "user_author_ltv"
      ) \
      .item_attr_operation(
        item_attr_a = "user_author_ltv_time_model",
        item_attr_b = "userfulness_author_score_tur",
        operator = "+",
        output_attr = "user_author_ltv_time"
      ) \
    .else_() \
      .copy_attr(
        attrs=[
          {"from_item": "user_author_ltv_model", "to_item": "user_author_ltv"},
          {"from_item": "user_author_ltv_time_model", "to_item": "user_author_ltv_time"},
        ]
      ) \
    .end_if_() \
    .if_("fountain_enable_use_pure_value == 1") \
      .explore_ensemble_score_calc_pure_value_enricher(
        save_score_to_attr = "fullrank_pure_value_score",
        user_power_calc = "{{fountain_pure_value_use_pow_calc}}",
        log_add_alpha = "{{fountain_pure_value_log_add_alpha}}",
        queues = fullrank_pure_value_queue,
      ) \
    .end_() \
    .if_("fountain_enable_use_pure_value_fullrank == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .explore_ensemble_score_calc_pure_value_enricher(
        save_score_to_attr = "fullrank_pure_value_score_fullrank",
        user_power_calc = "{{fountain_pure_value_use_pow_calc_fullrank}}",
        log_add_alpha = "{{fountain_pure_value_log_add_alpha_fullrank}}",
        queues = fullrank_pure_value_queue_fullrank,
      ) \
    .end_() \
    .if_("skip_fountain_fullrank_multi_stage == 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .fullrank_multi_stage() \
    .end_if_() \
    .if_("skip_fullrank_ensemble_score_v8 == 0") \
      .if_("fountain_fullrank_enable_commercial_queues == 0") \
        .gen_common_attr_by_lua(  # 队列 enable 开关仅允许在前边修改
          attr_map = { q["enable"]: "0" for q in fullrank_ensemble_commercial_queues }
        ) \
      .end_() \
      .fullrank_s2_interact_playtime_adjust() \
      .fullrank_follow_touch_high_adjust() \
      .fountain_calc_ensemble_score(
        save_score_to_attr = "fullrank_ensemble_score",
        save_ori_ensemble_score_to_attr = "fullrank_ensemble_ori_score",
        save_absolute_score_to_attr = "fullrank_ensemble_absolute_score",
        save_fractile_score_to_attr = "fullrank_ensemble_fractile_score",
        use_dist_calc = "{{fountain_fullrank_ensemble_use_dist_calc}}",
        dis_factor = "{{fountain_fullrank_ensemble_dis_factor}}",
        user_new_proportion = "{{fullrank_ensemble_score_user_new_proportion}}",
        user_power_calc = "{{fountain_fullrank_variant_enable_power_calc}}",
        user_power_calc_v2 = "{{fountain_fullrank_variant_enable_power_calc_v2}}",
        rank_smooth = "{{fountain_fullrank_rank_smooth}}",
        fractile_smooth = "{{fountain_fullrank_fractile_smooth}}",
        use_queue_smooth_as_rank_smooth = "{{fountain_fullrank_ensemble_use_queue_smooth_as_rank_smooth}}",
        use_reciprocal = "{{fountain_fullrank_use_reciprocal}}",
        duration_min = "{{fountain_fullrank_duration_min}}",
        duration_max = "{{fountain_fullrank_duration_max}}",
        user_info_ptr_attr = "userInfoPb",
        action_day = "{{fountain_fullrank_variant_weight_action_day_num}}",
        min_rank_weight = "{{fountain_fr_fullrank_min_rank_weight}}",
        queue_head_boost_index = "{{fountain_fullrank_ensemble_queue_head_boost_index}}",
        queue_tail_discount_index = "{{fountain_fullrank_ensemble_queue_tail_discount_index}}",
        queues = fullrank_ensemble_queues + fullrank_ensemble_commercial_queues + splash_fullrank_ensemble_queues,  # 双列内流精排效率队列 + 业务侧队列 + 首屏队列
        use_absolute_score_queue_power_weight = "{{fountain_fullrank_ensemble_use_absolute_score_queue_power_weight}}",
        queue_head_boost_threshold = "{{fountain_fullrank_ensemble_queue_head_boost_threshold}}",
        queue_tail_discount_threshold = "{{fountain_fullrank_ensemble_queue_tail_discount_threshold}}",
        ensemble_score_head_coef = "{{fountain_fullrank_ensemble_ensemble_score_head_coef}}",
        ensemble_score_tail_coef = "{{fountain_fullrank_ensemble_ensemble_score_tail_coef}}",
        use_rank_with_absolute_score = "{{fountain_fullrank_ensemble_use_rank_with_absolute_score}}",
        enable_time_cost_opt = "{{fountain_fullrank_enable_time_cost_opt}}",
      ) \
    .else_() \
      .refactor_ori_es_module() \
    .end_if_() \
    .if_("enable_fountain_emp_xtr_debias_f1 == 1") \
      .fullrank_emp_xtr_debias() \
    .end_if_() \
    .if_("skip_fullrank_use_xtr_raw_score == 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .use_hierarchical_es() \
    .end_if_() \
    .if_("enable_fullrank_use_xtr_raw_weighted_score == 1", to_be_delete = "date=2024-05-29;committer=denghong") \
      .raw_weighted_score_es() \
    .end_if_() \
    .if_("skip_fullrank_pure_value_es_rank_score == 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .fountain_calc_ensemble_score(
        save_score_to_attr = "fullrank_ensemble_score",
        user_power_calc = "{{fountain_fullrank_pure_value_es_rank_use_pow}}",
        use_reciprocal = "{{fountain_fullrank_pure_value_es_rank_use_reciprocal}}",
        rank_smooth = "{{fountain_fullrank_pure_value_es_rank_rank_smooth}}",
        use_superscript_rank = True,
        queues = fullrank_pure_value_es_queue
      ) \
    .end_() \
    .if_("skip_fullrank_pure_value_es_value_score == 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .perflog_attr_value(
        check_point = "dryrun_test",
        item_attrs = [
          "fullrank_ensemble_score",
          "fullrank_pure_value_score_fullrank"
        ],

      ) \
      .enrich_attr_by_lua(
        import_common_attr = [
          "fullrank_pure_value_es_value_use_multiply",
          "fullrank_pure_value_es_value_alpha",
          "fullrank_pure_value_es_value_beta",
          "fullrank_pure_value_es_value_bias"
        ],
        import_item_attr = [
          "fullrank_ensemble_score",
          "fullrank_pure_value_score_fullrank"
        ],
        export_item_attr = [
          "fullrank_ensemble_score"
        ],
        function_for_item = "pure_value_es_value_score",
        lua_script_file = "fountain/full_rank/lua/calc_pure_value_es_value_score.lua",
      ) \
    .end_() \
    .if_("skip_fullrank_ensemble_ltr_score_fix == 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .fountain_calc_ensemble_score(
          save_score_to_attr = "fullrank_ensemble_ltr_score",
          user_power_calc = "{{fountain_fullrank_ensemble_ltr_use_pow}}",
          use_reciprocal = "{{fountain_fullrank_ensemble_ltr_use_reciprocal}}",
          rank_smooth = "{{fountain_fullrank_ensemble_ltr_rank_smooth}}",
          use_superscript_rank = True,
          queues = fullrank_ensemble_ltr_queue
      ) \
      .fountain_calc_ensemble_score(
        save_score_to_attr = "fullrank_ensemble_score",
        user_power_calc = "{{fountain_fullrank_final_ensemble_use_pow}}",
        use_reciprocal = "{{fountain_fullrank_final_ensemble_use_reciprocal}}",
        rank_smooth = "{{fountain_fullrank_final_ensemble_rank_smooth}}",
        use_superscript_rank = True,
        queues = [
          {
            "name": "fullrank_ensemble_score",
            "weight": 0.0,
            "weight_attr": "fountain_final_es_rank_weight_fullrank_ensemble_score",
            "power_weight_attr": "fountain_final_es_rank_weight_fullrank_ensemble_score",
          },
          {
            "name": "fullrank_ensemble_ltr_score",
            "weight": 0.0,
            "weight_attr": "fountain_final_ltr_rank_weight_fullrank_es_ltr_score",
            "power_weight_attr": "fountain_final_ltr_rank_weight_fullrank_es_ltr_score",
          }]
      ) \
    .end_() \
    .if_("enable_fountain_fullrank_esnn_ltr_adjust_kai2 == 1") \
      .fountain_calc_ensemble_score(
        save_ori_ensemble_score_to_attr = "fullrank_ensemble_score",
        use_queue_smooth_as_rank_smooth = "{{fountain_fullrank_ensemble_use_queue_smooth_as_rank_smooth}}",
        user_power_calc_v2 = 1,
        queues = [
          {
            "name": "fullrank_esnn_ltr_score",
            "weight": 0.0,
            "weight_attr": 0.0,
            "power_weight_attr": "fountain_ensemble_power_weight_fullrank_esnn_ltr",
            "temperature_attr": "fountain_fullrank_ensemble_adjust_temperature_attr",
            "smooth_attr": "fountain_fullrank_ensemble_adjust_smooth_attr",
          },
          {
            "name": "fullrank_ensemble_score",
            "weight": 0.0,
            "weight_attr": 0.0,
            "power_weight_attr": "fountain_ensemble_power_weight_fullrank_origin",
            "temperature_attr": "fountain_fullrank_ensemble_adjust_temperature_attr",
            "smooth_attr": "fountain_fullrank_ensemble_adjust_smooth_attr",
          }]
      ) \
    .end_if_() \
    .if_("_SEND_RANK_S2_STAGE_SAMPLE_ == 1") \
      .send_stage_sample("i_rank_s2") \
    .end_() \
    ._dump_attr_to_kafka( # ES 排序之后, 将全部item的重要 item attr 落盘
      stage_name = "fr_s2_score",
      dump_item_attr_list = [
        # 推全排序队列
        "fullrank_sim_click_score",
        "fullrank_sim_like_score",
        "fullrank_sim_pvtr_multi_pwtr",
        "fullrank_sim_pcmtr",
        "fullrank_sim_pvtr_multi_pptr",
        "fullrank_sim_pcmef",
        "fullrank_sim_pcltr",
        "fullrank_sim_plvtr",
        "fullrank_sim_pwatchtime_no_bias",
        "fullrank_sim_pcpr",
        "fullrank_sim_pepstr",
        "fullrank_sim_phtr",
        "fullrank_sim_pfintr",
        "fullrank_hate_similary_score",
        "fullrank_action_once_watchtime_score",
        "fullrank_ltr_score",
        "fullrank_ltr_v4_fountain_finish_rate",
        "fullrank_opportunity_cost_score",
        "fullrank_sim_pftr",
        "fullrank_min_act_rank_reci",
        "fullrank_sim_longview_score_no_bias",
        "fullrank_sim_out_pctr",
        "fullrank_sim_lstr",
        "fullrank_sim_psvr",
        "fullrank_ltr_v4_fountain_next",
        "fullrank_detail_new_pevtr_v2",
        "fullrank_ori_pswptr",
        "comment_ltr",
        "fullrank_sim_pwatchtime_no_bias_debias",
        "xgb_ltr",
        "fullrank_pre_filter_score",
        "fullrank_ada_xtr_score",
        "fullrank_trans_pvtr_score",
        # 排序分
        "fullrank_ensemble_score",
        # 相关 ltr pxtr , 如后续未推全则删除 @dengyingjie03
        "splash_fullrank_ltr_act_v2_score",
        "splash_fullrank_ltr_wtd_score",
        "splash_fullrank_ltr_like_score",
      ]
    )
    return self

  def user_adaptive_weight_with_active_score(self):
    self \
    .calc_by_formula1(
      kconf_key = "formula.scenarioKey98.FrFountainActiveScoreAdjust",
      import_common_attr = [
        "fountain_ensemble_power_weight_fullrank_like_score",
        "fountain_ensemble_power_weight_fullrank_follow_score",
        "fountain_ensemble_weight_forward_score",
        "fountain_ensemble_power_weight_fullrank_pcmtr_score",
        "fountain_ensemble_power_weight_fullrank_pcltr_score",
        "fountain_ensemble_power_weight_adjust_ratio_min",
        "fountain_ensemble_power_weight_adjust_ratio_max",
        "fountain_ensemble_power_weight_fullrank_like_emp",
        "fountain_ensemble_power_weight_fullrank_follow_emp",
        "fountain_ensemble_power_weight_fullrank_forward_emp",
        "fountain_ensemble_power_weight_fullrank_pcmtr_emp",
        "fountain_ensemble_weight_fullrank_action_interact_once_score"
        "userExpLtr",
        "userExpWtr",
        "userExpFtr",
        "userExpCmtr",
        "uLikeActiveScore",
        "uFollowActiveScore",
        "uShareActiveScore",
        "uCommentActiveScore",
        "uCollectActiveScore",
      ],
      export_formula_value = [
        {"name": "final_like_score", "as": "fountain_ensemble_power_weight_fullrank_like_score", "to_common": True},
        {"name": "final_follow_score", "as": "fountain_ensemble_power_weight_fullrank_follow_score", "to_common": True},
        {"name": "final_forward_score", "as": "fountain_ensemble_weight_forward_score", "to_common": True},
        {"name": "final_pcmtr_score", "as": "fountain_ensemble_power_weight_fullrank_pcmtr_score", "to_common": True},
        {"name": "final_collect_score", "as": "fountain_ensemble_power_weight_fullrank_pcltr_score", "to_common": True},
        {"name": "final_action_once_score", "as": "fountain_ensemble_weight_fullrank_action_interact_once_score", "to_common": True}
      ],
      abtest_biz_name = "KUAISHOU_APPS"
    )
    return self

  def user_age_based_weight_adjust_all(self):
    self \
    .user_attr_based_weight_adjust(
      "basic_info_age_segment_v2",
      "fountain_rank_s2_age_based_pctr_weight_adjust_list",
      "fountain_ensemble_power_weight_fullrank_click_score"
    ) \
    .user_attr_based_weight_adjust(
      "basic_info_age_segment_v2",
      "fountain_rank_s2_age_based_pltr_weight_adjust_list",
      "fountain_ensemble_power_weight_fullrank_like_score"
    ) \
    .user_attr_based_weight_adjust(
      "basic_info_age_segment_v2",
      "fountain_rank_s2_age_based_pwtr_weight_adjust_list",
      "fountain_ensemble_power_weight_fullrank_follow_score"
    ) \
    .user_attr_based_weight_adjust(
      "basic_info_age_segment_v2",
      "fountain_rank_s2_age_based_pftr_weight_adjust_list",
      "fountain_ensemble_weight_forward_score"
    ) \
    .user_attr_based_weight_adjust(
      "basic_info_age_segment_v2",
      "fountain_rank_s2_age_based_pcmtr_weight_adjust_list",
      "fountain_ensemble_power_weight_fullrank_pcmtr_score"
    ) \
    .user_attr_based_weight_adjust(
      "basic_info_age_segment_v2",
      "fountain_rank_s2_age_based_wtd_weight_adjust_list",
      "fountain_ensemble_power_weight_fullrank_pfintr_score"
    ) \
    .user_attr_based_weight_adjust(
      "basic_info_age_segment_v2",
      "fountain_rank_s2_age_based_ltr_next_weight_adjust_list",
      "fountain_ensemble_power_weight_fullrank_ltr_v4_next"
    ) \
    .user_attr_based_weight_adjust(
      "basic_info_age_segment_v2",
      "fountain_rank_s2_age_based_action_interact_once_weight_adjust_list",
      "fountain_ensemble_weight_fullrank_action_interact_once_score"
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

  def rrr_discount(self):
    self \
      .enrich_attr_by_lua(
        import_common_attr = [
          "fountain_fullrank_rrr_discount_report_smooth",
          "fountain_fullrank_rrr_discount_show_smooth",
          "fountain_fullrank_rrr_discount_param_n",
          "fountain_fullrank_rrr_discount_param_o",
          "fountain_fullrank_rrr_discount_enable_exp_discount"
        ],
        import_item_attr = [
          "fullrank_ensemble_score",
          "explore_stat__report_detail__total_report_count",
          "explore_stat__real_show_count",
        ],
        export_item_attr = [
          "fullrank_ensemble_score",
          "fullrank_rrr_discount_factor",
        ],
        function_for_item = "calc_rrr_discount",
        lua_script_file = "fountain/full_rank/lua/calc_rrr_discount_score.lua",
      ) \
      .log_debug_info(
        common_attrs = [
          "fountain_fullrank_rrr_discount_report_smooth",
          "fountain_fullrank_rrr_discount_show_smooth",
          "fountain_fullrank_rrr_discount_param_n",
          "fountain_fullrank_rrr_discount_param_o",
        ],
        item_attrs = [
          "fullrank_ensemble_score",
          "fullrank_rrr_discount_factor",
          "explore_stat__report_detail__total_report_count",
          "explore_stat__real_show_count",
        ],
        item_num_limit = 10,
        for_debug_request_only = True,
      )
    return self

  def fullrank_all_ltr_predict(self):
    self \
      .if_("fountain_enable_rerank_distill == 1 and (fountain_rerank_distill_only_fast_v1 == 0 or page > 1)") \
        .switch_("fountain_rerank_distill_model_predict_mode") \
          .case_(1, to_be_delete = "date=2024-05-29;committer=lijinyu") \
            .delegate_enrich(
              kess_service = "{{fountain_rerank_distill_kess_service}}",
              recv_item_attrs = [
                {"name": "distill_rerank", "as": "fullrank_distill_rerank_score"},
                {"name": "ltr", "as": "fullrank_distill_rerank_ltr"},
              ],
              timeout_ms = 100,
              send_item_attrs = rerank_distill_fullchain_item_feature,
              send_common_attrs = rerank_distill_fullchain_user_feature,
              request_type = "{{fountain_rerank_distill_request_type}}",
              partition_size = "{{fountain_rerank_distill_partition_size}}",
            ) \
          .case_(2) \
            .explore_custom_trim_user_info(
              user_info_attr = "userInfo",
              save_trimed_user_info_to_attr = "fr_distill_ltr_trimmed_user_info",
              trim_user_info = [
                "active_days",
                "basic_info.age_segment",
                "location.city_id",
                "location.region_type",
                "client_id",
                "device_id",
                "gender",
                "infer_gender",
                "true_gender",
                "follow_count",
                "fans_count",
                "upload_count",
                "request_location.poi_type",
                "request_location.province_id",
                "request_location.city_id",
                "visit_mod",
                "user_profile.exp_stat.exp_click",
                "user_profile.exp_stat.exp_like",
                "user_profile.exp_stat.exp_follow",
                "user_profile.exp_stat.exp_realshow",
                "user_profile.exp_stat.exp_long_view",
                "user_profile.user_level",
                "fountain_reco_user_profile.click_list.author_id",
                "fountain_reco_user_profile.click_list.photo_id",
                "fountain_reco_user_profile.comment_list.author_id",
                "fountain_reco_user_profile.comment_list.photo_id",
                "fountain_reco_user_profile.follow_list.author_id",
                "fountain_reco_user_profile.follow_list.photo_id",
                "fountain_reco_user_profile.like_list.author_id",
                "fountain_reco_user_profile.like_list.photo_id",
                "fountain_reco_user_profile.video_play_stat.photo_id",
                "fountain_reco_user_profile.video_play_stat.author_id",
                "fountain_reco_user_profile.video_play_stat.video_duration",
                "fountain_reco_user_profile.video_play_stat.playing_time",
                "user_profile_v1.click_list.author_id",
                "user_profile_v1.click_list.photo_id",
                "user_profile_v1.follow_list.author_id",
                "user_profile_v1.follow_list.photo_id",
                "user_profile_v1.like_list.author_id",
                "user_profile_v1.like_list.photo_id",
                "user_profile_v1.video_playing_stat.playing_time",
                "user_profile_v1.video_playing_stat.author_id",
                "user_profile_v1.video_playing_stat.photo_id",
                "realtime_click_list",
                "realtime_follow_list",
                "realtime_forward_list",
                "realtime_like_list",
              ],
            ) \
            .delegate_enrich(
              kess_service = "{{fountain_rerank_distill_kess_service}}",
              recv_item_attrs = [
                {"name": "distill_rerank", "as": "fullrank_distill_rerank_score"},
                {"name": "ltr", "as": "fullrank_distill_rerank_ltr"},
              ],
              timeout_ms = 100,
              send_item_attrs = [
                "cascade_pctr",
                "cascade_pltr",
                "cascade_plvtr",
                "cascade_pwtr",
                "cascade_pftr",
                "cascade_ptr",
                "cascade_pcmtr",
                "fullrank_detail_pcmtr",
                "fullrank_detail_pctr",
                "fullrank_detail_pftr",
                "fullrank_detail_pltr",
                "fullrank_detail_plvtr",
                "fullrank_detail_pptr",
                "fullrank_detail_pwtr",
              ],
              send_common_attrs = [
                { "name": "fr_distill_ltr_trimmed_user_info", "as": "user_info_str" }
              ],
              request_type = "{{fountain_rerank_distill_request_type}}",
              partition_size = "{{fountain_rerank_distill_partition_size}}",
            ) \
          .default_() \
            .delegate_enrich(
              kess_service = "{{fountain_rerank_distill_kess_service}}",
              recv_item_attrs = [
                {"name": "distill_rerank", "as": "fullrank_distill_rerank_score"}
              ],
              timeout_ms = 100,
              send_item_attrs = item_features + item_features_rerank,
              send_common_attrs = user_features,
              request_type = "{{fountain_rerank_distill_request_type}}",
              partition_size = "{{fountain_rerank_distill_partition_size}}",
            ) \
        .end_() \
      .end_() \
      .if_("enable_fountain_deep_ltr_predict == 1") \
        .if_("enable_fountain_fullrank_deep_ltr_kai2 == 1") \
          .explore_custom_trim_user_info(
            user_info_attr = "userInfo",
            save_trimed_user_info_to_attr = "fr_deep_ltr_trimmed_user_info",
            trim_user_info = [
              "active_days",
              "basic_info.age_segment",
              "location.city_id",
              "location.region_type",
              "client_id",
              "device_id",
              "gender",
              "infer_gender",
              "true_gender",
              "request_location.poi_type",
              "request_location.province_id",
              "request_location.city_id",
              "visit_mod",
              "upload_count",
              "infer_year",
              "follow_count",
              "fans_count",
              "visit_net",
              "location.city_level",
              "is_douyin",
              "feature_collection.explore_low_active_level",
              "user_profile.exp_stat.exp_click",
              "user_profile.exp_stat.exp_like",
              "user_profile.exp_stat.exp_follow",
              "user_profile.exp_stat.exp_realshow",
              "user_profile.exp_stat.exp_long_view",
              "user_profile.user_level",
              "fountain_reco_user_profile.click_list.author_id",
              "fountain_reco_user_profile.click_list.photo_id",
              "fountain_reco_user_profile.comment_list.author_id",
              "fountain_reco_user_profile.comment_list.photo_id",
              "fountain_reco_user_profile.follow_list.author_id",
              "fountain_reco_user_profile.follow_list.photo_id",
              "fountain_reco_user_profile.like_list.author_id",
              "fountain_reco_user_profile.like_list.photo_id",
              "fountain_reco_user_profile.video_play_stat.photo_id",
              "fountain_reco_user_profile.video_play_stat.author_id",
              "fountain_reco_user_profile.video_play_stat.video_duration",
              "fountain_reco_user_profile.video_play_stat.playing_time",
              "fountain_reco_user_profile.video_play_stat.client_timestamp",
              "user_profile_v1.click_list.author_id",
              "user_profile_v1.click_list.photo_id",
              "user_profile_v1.follow_list.author_id",
              "user_profile_v1.follow_list.photo_id",
              "user_profile_v1.like_list.author_id",
              "user_profile_v1.like_list.photo_id",
              "user_profile_v1.hate_list.photo_id",
              "user_profile_v1.video_playing_stat.playing_time",
              "user_profile_v1.video_playing_stat.author_id",
              "user_profile_v1.video_playing_stat.photo_id",
              "user_profile_v1.video_playing_stat.client_timestamp",
              "realtime_click_list",
              "realtime_follow_list",
              "realtime_forward_list",
              "realtime_like_list",
              "user_profile_v1.real_show_list.photo_id",
              "user_profile_v1.real_show_list.author_id",
              "user_profile_v1.real_show_list.time_ms",
              "user_profile_v1.real_show_list.page_type",
              "user_profile_v1.real_show_list.label.click",
              "user_profile_v1.real_show_list.label.like",
              "user_profile_v1.real_show_list.label.follow",
              "user_profile_v1.real_show_list.label.hate",
              "user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_one",
              "user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_two",
              "user_profile_v1.like_list.hetu_tag_level_info.hetu_level_one",
              "user_profile_v1.like_list.hetu_tag_level_info.hetu_level_two",
            ],
          ) \
          .delegate_enrich(
            name = "fountain_fullrank_deep_ltr_fast",
            kess_service = "{{fountain_fullrank_deep_ltr_kess_service}}",
            recv_item_attrs = [
              {"name": "l2r", "as": "fullrank_ltr_score"},
              {"name": "ctr", "as": "fullrank_act_ctr"},
              {"name": "wtd", "as": "fullrank_act_wtd"},
              {"name": "finish_rate", "as": "fullrank_ltr_v4_fountain_finish_rate"},
              {"name": "next", "as": "fullrank_ltr_v4_fountain_next"},
              {"name": "ordinal_wtd", "as": "fullrank_ltr_v4_fountain_ordinal_wtd"},
              {"name": "cst", "as": "fullrank_ltr_comment_staytime"},
              {"name": "d2co", "as": "fullrank_ltr_d2co_playtime"},
              {"name": "intn", "as": "fullrank_ltr_intn_rate"},
              {"name": "ord2q", "as": "fullrank_ltr_ord2q"},
              {"name": "lph", "as": "fullrank_ltr_lph"},
              {"name": "dfvr", "as": "fullrank_ltr_dfvr"},
            ],
            timeout_ms = 100,
            send_item_attrs = [
              "cascade_pctr",
              "cascade_pltr",
              "cascade_pwtr",
              "cascade_plvtr",
              "cascade_psvtr",
              "cascade_pftr",
              "cascade_ptr",
              "cascade_pcmtr",
              "fullrank_detail_pctr",
              "fullrank_detail_pltr",
              "fullrank_detail_pwtr",
              "fullrank_detail_pftr",
              "fullrank_detail_plvtr",
              "fullrank_detail_pvtr",
              "fullrank_detail_psvr",
              "fullrank_detail_pcmtr",
              "fullrank_detail_pptr",
              "fullrank_detail_pwtd",
              "fullrank_detail_pcmef",
              "fullrank_detail_pwtd",
              "fullrank_detail_pepstr",
              "fullrank_detail_phtr",
              "fullrank_detail_new_pevtr_v2",
              "pctr_index",
              "plvtr_index",
              "pvtr_index",
              "pltr_index",
              "pftr_index",
              "pwtr_index",
              "pesptr_index",
              "psvr_index",
              "pfintr_index",
              "pcmtr_index",
              "pcltr_index",
            ],
            send_common_attrs = [
              { "name": "fr_deep_ltr_trimmed_user_info", "as": "user_info_str" },
              { "name": "featureSourcePId", "as": "source_pid" },
              { "name": "sourcePidDuration", "as": "source_duration_ms" },
              { "name": "sourcePidTagId", "as": "source_tag" },
              { "name": "sourcePidAuthorId", "as": "source_aid" },
              { "name": "sourcePidHetuLevelOneList", "as": "source_hetu_tag_level1_list" },
              { "name": "sourcePidHetuLevelTwoList", "as": "source_hetu_tag_level2_list" },
              { "name": "featureSimilarUserList", "as": "similar_user_list" },
              "page",
              "source_playtime_s",
              "source_is_interacted",
            ],
            request_type = "{{fountain_deep_ltr_request_type}}",
            partition_size = "{{fountain_deep_ltr_partition_size}}",
          ) \
        .else_() \
          .delegate_enrich(
            kess_service = "{{fountain_fullrank_deep_ltr_kess_service}}",
            recv_item_attrs = [
              {"name": "l2r", "as": "fullrank_ltr_score"},
              {"name": "ctr", "as": "fullrank_act_ctr"},
              {"name": "wtd", "as": "fullrank_act_wtd"},
              {"name": "finish_rate", "as": "fullrank_ltr_v4_fountain_finish_rate"},
              {"name": "next", "as": "fullrank_ltr_v4_fountain_next"},
              {"name": "ltr", "as": "fullrank_ltr_v4_fountain_reward"},
            ],
            timeout_ms = 100,
            send_item_attrs = [feature["name"] for feature in photo_features if feature["name"]] + item_sim_gsu_feature,
            send_common_attrs = user_features_v3,
            request_type = "{{fountain_deep_ltr_request_type}}",
            partition_size = "{{fountain_deep_ltr_partition_size}}",
          ) \
        .end_() \
      .end_if_() \
      .if_("fountain_skip_ltr_predict_v4 == 0 and (fountain_skip_fr_pred_only_fast_v1 == 1 or (fountain_skip_fr_pred_only_fast_v1 == 0 and page ~= nil and page > 1))") \
        .explore_gen_user_top_wt_pids(
          colossus_resp_attr = "colossus_resp_v2",
          output_common_attr = "longTermInterestList",
          top_wt_item_size = "{{fountain_fullrank_top_wt_item_size}}",
          min_dura = "{{fountain_fullrank_gen_top_wt_items_min_dura}}",
          min_wt = "{{fountain_fullrank_gen_top_wt_items_min_wt}}",
          max_seconds_ago = "{{fountain_fullrank_gen_top_wt_items_max_seconds_ago}}",
          min_seconds_ago = "{{fountain_fullrank_gen_top_wt_items_min_seconds_ago}}",
        ) \
        .delegate_enrich(
          kess_service = "{{fountain_fullrank_ltr_predict_kess_service_v4}}",
          recv_item_attrs = [
            {"name": "fountain_finish_rate", "as": "fullrank_ltr_v4_fountain_finish_rate"},
            {"name": "fountain_next", "as": "fullrank_ltr_v4_fountain_next"},
          ],
          timeout_ms = "{{fountain_fullrank_pfr_predict_timeout}}",
          send_item_attrs = [feature["name"] for feature in photo_features if feature["name"]],
          send_common_attrs = user_features_v3 + ["longTermInterestList"],
          request_type = "ltr_v4_predict",
          partition_size = "{{fountain_ltr_v4_predict_partition_size}}",
        ) \
      .end_if_() \
      .if_("skip_fountain_act_l2r_predict == 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .delegate_enrich(
          kess_service = "{{fountain_fullrank_act_l2r_kess_service}}",
          recv_item_attrs = [
            {"name": "l2r", "as": "fullrank_ltr_score"},
            {"name": "ctr", "as": "fullrank_act_ctr"},
            {"name": "wtd", "as": "fullrank_act_wtd"},
            {"name": "ltr", "as": "fullrank_act_ltr"},
            {"name": "cmtr", "as": "fullrank_act_cmtr"},
            {"name": "wtr", "as": "fullrank_act_wtr"},
            {"name": "ptr", "as": "fullrank_act_ptr"},
            {"name": "cmef", "as": "fullrank_act_cmef"},
            {"name": "estr", "as": "fullrank_act_estr"},
          ],
          timeout_ms = "{{fountain_fullrank_act_l2r_timeout}}",
          send_item_attrs = [feature["name"] for feature in photo_features if feature["name"]],
          send_common_attrs = user_features_v3,
          request_type = "kai_predict",
          partition_size = "{{fountain_act_l2r_partition_size}}",
        ) \
      .end_if_() \
      .if_("skip_fountain_fullrank_ptime_l2r == 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .delegate_enrich(
          kess_service = "{{fountain_fullrank_ptime_l2r_kess_service}}",
          recv_item_attrs = [
            {"name": "l2r", "as": "fullrank_ltr_score"},
          ],
          timeout_ms = 150,
          send_item_attrs = [feature["name"] for feature in photo_features if feature["name"]],
          send_common_attrs = user_features_v2,
          request_type = "fountain_ptime_l2r",
          partition_size = "{{fountain_ptime_l2r_partition_size}}",
        ) \
      .end_if_() \
      .if_("skip_fountain_sl_predict_kess_service == 0 and (fountain_sl_only_fast_v1 == 0 or (fountain_sl_only_fast_v1 == 1 and page ~= nil and page > 1))") \
      .delegate_enrich(
        kess_service = "{{fountain_simple_ltr_kess_service}}",
        recv_item_attrs = [
          {"name": "fountain_interactive_score", "as": "fountain_sl_interactive_score"}
        ],
        timeout_ms = 150,
        send_item_attrs = simple_ltr_photo_feature,
        send_common_attrs = simple_ltr_user_feature,
        request_type = "kai_predict",
        partition_size = "{{fountain_sl_predict_partition_size}}",
      ) \
      .end_if_()

    return self

  def fullrank_reco_pxtr_predict(self):
    self \
      .if_("enable_fountain_reco_pxtr_predict == 1") \
        .explore_custom_trim_user_info(
          user_info_attr = "userInfo",
          save_trimed_user_info_to_attr = "fr_reco_pxtr_trimmed_user_info",
          trim_user_info = [
            "id",
            "active_days",
            "basic_info.age_segment",
            "location.city_id",
            "location.region_type",
            "client_id",
            "device_id",
            "gender",
            "infer_gender",
            "true_gender",
            "request_location.poi_type",
            "request_location.province_id",
            "request_location.city_id",
            "visit_mod",
            "user_profile.exp_stat.exp_click",
            "user_profile.exp_stat.exp_like",
            "user_profile.exp_stat.exp_follow",
            "user_profile.exp_stat.exp_realshow",
            "user_profile.exp_stat.exp_long_view",
            "user_profile.user_level",
            "upload_count",
            "fans_count",
            "visit_net",
            "is_douyin",
            "apps.app.package",
            "feature_collection.explore_low_active_level",
            "user_interest_profile.hetu_level_one_long_term_id",
            "user_interest_profile.hetu_level_one_long_term_score",
            "user_interest_profile.hetu_level_two_long_term_id",
            "user_interest_profile.hetu_level_two_long_term_score",
            "user_interest_profile.hetu_level_three_long_term_id",
            "user_interest_profile.hetu_level_three_long_term_score",
            "fountain_reco_user_profile.click_list.author_id",
            "fountain_reco_user_profile.click_list.photo_id",
            "fountain_reco_user_profile.comment_list.author_id",
            "fountain_reco_user_profile.comment_list.photo_id",
            "fountain_reco_user_profile.follow_list.author_id",
            "fountain_reco_user_profile.follow_list.photo_id",
            "fountain_reco_user_profile.like_list.author_id",
            "fountain_reco_user_profile.like_list.photo_id",
            "fountain_reco_user_profile.video_play_stat.photo_id",
            "fountain_reco_user_profile.video_play_stat.author_id",
            "fountain_reco_user_profile.video_play_stat.video_duration",
            "fountain_reco_user_profile.video_play_stat.playing_time",
            "fountain_reco_user_profile.video_play_stat.client_timestamp",
            "user_profile_v1.click_list.author_id",
            "user_profile_v1.click_list.photo_id",
            "user_profile_v1.follow_list.author_id",
            "user_profile_v1.follow_list.photo_id",
            "user_profile_v1.like_list.author_id",
            "user_profile_v1.like_list.photo_id",
            "user_profile_v1.hate_list.photo_id",
            "user_profile_v1.video_playing_stat.playing_time",
            "user_profile_v1.video_playing_stat.client_timestamp",
            "user_profile_v1.video_playing_stat.author_id",
            "user_profile_v1.video_playing_stat.photo_id",
            "user_profile_v1.click_list.hetu_tag_level_info.hetu_level_one",
            "user_profile_v1.click_list.hetu_tag_level_info.hetu_level_two",
            "user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_one",
            "user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_two",
            "user_profile_v1.like_list.hetu_tag_level_info.hetu_level_one",
            "user_profile_v1.like_list.hetu_tag_level_info.hetu_level_two",
            "realtime_click_list",
            "realtime_follow_list",
            "realtime_forward_list",
            "realtime_like_list"
          ],
        ) \
        .delegate_enrich(
          kess_service = "{{fountain_fullrank_reco_pxtr_kess_service}}",
          recv_item_attrs = [
            {"name": "fountain_gen_time_wt", "as": "fullrank_reco_base_model_gen_time_wt"},
            {"name": "fountain_evtr", "as": "fullrank_reco_base_model_evtr"},
            {"name": "fountain_lvtr", "as": "fullrank_reco_base_model_lvtr"},
            {"name": "fountain_ltr", "as": "fullrank_reco_base_model_ltr"},
            {"name": "fountain_wtr", "as": "fullrank_reco_base_model_wtr"},
            {"name": "fountain_ftr", "as": "fullrank_reco_base_model_ftr"},
            {"name": "fountain_cmtr", "as": "fullrank_reco_base_model_cmtr"}
          ],
          timeout_ms = 150,
          send_item_attrs = [
            "cascade_psvtr",
            "cascade_pctr",
            "cascade_plvtr",
            "cascade_pwtr",
            "cascade_pltr",
            "cascade_pftr",
            "cascade_ptr",
            "cascade_pcmtr",
            "cascade_pwatch_time",
            "cascade_pctr_index",
            "cascade_plvtr_index",
            "cascade_pvtr_index",
            "cascade_pltr_index",
            "cascade_pftr_index",
            "cascade_pwtr_index",
            "cascade_pesptr_index",
            "cascade_psvr_index",
            "fountain_related_score_v2",
          ],
          send_common_attrs = [
            { "name": "fr_reco_pxtr_trimmed_user_info", "as": "user_info_str" },
            { "name": "featureSourcePId", "as": "source_pid" },
            { "name": "sourcePidDuration", "as": "source_duration_ms" },
            { "name": "sourcePidTagId", "as": "source_tag" },
            { "name": "sourcePidAuthorId", "as": "source_aid" },
            { "name": "sourcePidHetuLevelOneList", "as": "source_hetu_tag_level1_list" },
            { "name": "sourcePidHetuLevelTwoList", "as": "source_hetu_tag_level2_list" },
            { "name": "featureSimilarUserList", "as": "similar_user_list" },
            "page",
          ],
          request_type = "{{fountain_reco_pxtr_predict_request_type}}",
          partition_size = "{{fountain_reco_pxtr_predict_partition_size}}",
        ) \
      .end_if_()
    return self

  def fullrank_all_ltv_predict(self):
    self \
    .if_("enable_fountain_fullrank_user_author_ltv_kai2 == 1") \
      .delegate_enrich(
        kess_service = "{{fountain_fullrank_user_author_ltv_kess_service}}",
        recv_item_attrs = [
          {"name": "ftn_revisit", "as": "user_author_ltv_model"},
          {"name": "ftn_revisit2", "as": "user_author_ltv_time_model"},
        ],
        timeout_ms = 100,
        send_item_attrs = [ 
          "photo_id",
          {"name": "author__id", "as": "author_id"},
          {"name": "author__fans_count", "as": "author_fans_count"},  
          {"name": "hetu_tag_level_info__hetu_cluster_id", "as": "hetu_cluster_id"}, 
          {"name": "hetu_tag_level_info__hetu_level_two", "as": "hetu_level_two_tag"},  
          {"name": "fullrank_detail_pctr", "as": "pctr"},
          {"name": "fullrank_detail_pltr", "as": "pltr"},
          {"name": "fullrank_detail_pwtr", "as": "pwtr"},
          {"name": "fullrank_detail_plvtr", "as": "plvtr"},
          {"name": "fullrank_detail_pcmtr", "as": "pcmtr"},
          {"name": "fullrank_detail_pcmef", "as": "pcmef"},
          {"name": "fullrank_detail_pptr", "as": "pptr"},
          {"name": "fullrank_detail_psvr", "as": "psvtr"},
          "pctr_index",
          "pltr_index",
          "pwtr_index",
          "pvtr_index",
          "plvtr_index",
          {"name": "fullrank_empirical_ctr", "as": "emp_ctr"},
          {"name": "fullrank_empirical_ltr", "as": "emp_ltr"},
          {"name": "fullrank_empirical_wtr", "as": "emp_wtr"}
        ],
        send_common_attrs = [ 
          {"name": "_USER_ID_", "as": "user_id"},
          {"name": "gender", "as": "user_gender"}, 
          {"name": "age_segment", "as": "user_age_segment"},
          "tab"
        ],
        request_type = "{{fountain_fullrank_user_author_ltv_request_type}}",
        partition_size = "{{fountain_fullrank_user_author_ltv_partition_size}}",
      ) \
    .end_if_()
    return self
  
  def fullrank_esnn_predict(self):
    self \
    .if_("enable_fountain_fullrank_esnn_kai2 == 1") \
      .explore_custom_trim_user_info(
        user_info_attr = "userInfo",
        save_trimed_user_info_to_attr = "deep_esnn_trimmed_user_info",
        trim_user_info = [
          "active_days",
          "basic_info.age_segment",
          "location.city_id",
          "gender",
          "infer_gender",
          "true_gender",
          "upload_count",
          "infer_year",
          "follow_count",
          "fans_count",
          "location.city_level",
          "request_location.province_id",
          "fountain_reco_user_profile.video_play_stat.photo_id",
          "fountain_reco_user_profile.video_play_stat.author_id",
          "fountain_reco_user_profile.video_play_stat.video_duration",
          "fountain_reco_user_profile.video_play_stat.playing_time",
          "fountain_reco_user_profile.video_play_stat.client_timestamp",
          "fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_one",
          "fountain_reco_user_profile.video_play_stat.hetu_tag_level_info.hetu_level_two",
          "user_profile.user_level",
          "fountain_reco_user_profile.click_list.author_id",
          "fountain_reco_user_profile.click_list.photo_id",
          "fountain_reco_user_profile.comment_list.author_id",
          "fountain_reco_user_profile.comment_list.photo_id",
          "fountain_reco_user_profile.follow_list.author_id",
          "fountain_reco_user_profile.follow_list.photo_id",
          "fountain_reco_user_profile.like_list.author_id",
          "fountain_reco_user_profile.like_list.photo_id",
          "user_profile.exp_stat.exp_click",
          "user_profile.exp_stat.exp_like",
          "user_profile.exp_stat.exp_follow",
          "user_profile.exp_stat.exp_realshow",
          "user_profile.exp_stat.exp_long_view",
        ]
      ) \
      .delegate_enrich(
        kess_service = "{{fountain_fullrank_esnn_kess_service}}",
        recv_item_attrs=[
          {"name": "es_score", "as": "fullrank_esnn_ltr_score"},
        ],
        timeout_ms = 100,
        send_item_attrs = [
          {"name": "fullrank_detail_pltr"},
          {"name": "fullrank_detail_pwtr"},
          {"name": "fullrank_detail_pftr"},
          {"name": "fullrank_sim_psvr"},
          {"name": "fullrank_detail_plvtr"},
          {"name": "fullrank_detail_pcmtr"},
          {"name": "fullrank_detail_pptr"},
          {"name": "fullrank_detail_pcmef"},
          {"name": "fullrank_detail_phtr"},
          {"name": "fullrank_detail_pctr"},
          {"name": "fullrank_detail_pvtr"},
          {"name": "fullrank_detail_pepstr"},
          {"name": "fullrank_sim_pfintr"},
          {"name": "fullrank_sim_pcltr"},
          {"name": "fullrank_sim_pcpr"},
          {"name": "fullrank_detail_new_pevtr_v2"},
        ],
        send_common_attrs = [
          { "name": "deep_esnn_trimmed_user_info", "as": "user_info_str"},
          { "name": "featureSourcePId", "as": "source_pid"},
          { "name": "featureSimilarUserList", "as": "similar_user_list"},
          "page"
        ],
        request_type = "{{fountain_fullrank_esnn_request_type}}",
        partition_size = "{{fountain_fullrank_esnn_partition_size}}",
      ) \
    .end_if_()
    
    return self

  def fullrank_session_predict(self):
    self \
    .if_("enable_fountain_fullrank_session_kai2 == 1") \
      .explore_custom_trim_user_info(
        user_info_attr = "userInfo",
        save_trimed_user_info_to_attr = "deep_session_trimmed_user_info",
        trim_user_info = [
          "active_days",
          "basic_info.age_segment",
          "location.city_id",
          "gender",
          "infer_gender",
          "true_gender",
          "upload_count",
          "infer_year",
          "follow_count",
          "fans_count",
          "location.city_level",
          "request_location.province_id",
          "request_location.city_id",
        ]
      ) \
      .delegate_enrich(
        kess_service = "{{fountain_fullrank_session_kess_service}}",
        recv_item_attrs=[
          {"name": "es_rpc", "as": "fullrank_esnn_score"},
          {"name": "es_session", "as": "fullrank_session_score"},
        ],
        timeout_ms = 100,
        send_item_attrs = [
          {"name": "fullrank_detail_pctr", "as": "pctr"},
          {"name": "fullrank_detail_pltr", "as": "pltr"},
          {"name": "fullrank_detail_pwtr", "as": "pwtr"},
          {"name": "fullrank_detail_pcmtr", "as": "pcmtr"},
          {"name": "fullrank_sim_pcltr", "as": "pcltr"},
          {"name": "fullrank_detail_pcmef", "as": "pcmef"},
          {"name": "fullrank_detail_pptr", "as": "pptr"},
          {"name": "fullrank_detail_plvtr", "as": "plvtr"},
          {"name": "fullrank_detail_psvr", "as": "psvtr"},
          {"name": "fullrank_detail_pwtd", "as": "pwtd"},

          {"name": "cascade_pctr", "as": "mc_pctr"},
          {"name": "cascade_pltr", "as": "mc_pltr"},
          {"name": "cascade_pwtr", "as": "mc_pwtr"},
          {"name": "cascade_pcmtr", "as": "mc_pcmtr"},
          {"name": "cascade_pcltr", "as": "mc_pcltr"},
          {"name": "cascade_pepstr", "as": "mc_pepstr"},
          {"name": "cascade_plvtr", "as": "mc_plvtr"},
          {"name": "cascade_psvtr", "as": "mc_psvtr"},
          {"name": "cascade_phtr", "as": "mc_phtr"},
          "pctr_index",
          "plvtr_index",
          "pvtr_index",
          "pltr_index",
          "pftr_index",
          "pwtr_index",
          "pesptr_index",
          "psvr_index",
          "pfintr_index",
          "pcmtr_index",
          "pcltr_index",
        ],
        send_common_attrs = [
          { "name": "deep_session_trimmed_user_info", "as": "user_info_str"},
          { "name": "featureSourcePId", "as": "source_pid"},
          "page"
        ],
        request_type = "{{fountain_fullrank_session_request_type}}",
        partition_size = "{{fountain_fullrank_session_partition_size}}",
      ) \
    .end_if_()

    return self

  def fullrank_effective_follow_predict(self):
    self \
      .if_("enable_fountain_effective_follow_model_predict == 1") \
        .explore_custom_trim_user_info(
          user_info_attr = "userInfo",
          save_trimed_user_info_to_attr = "fullrank_effective_follow_trimmed_user_info",
          trim_user_info = [
            "device_stat.human_action",
            "device_stat.device_status_flags",
            "id",
            "gender",
            "device_id",
            "user_active_level",
            "location.province_id",
            "location.city_id",
            "user_profile_v1.click_list.photo_id",
            "user_profile_v1.like_list.photo_id",
            "user_profile_v1.follow_list.photo_id",
          ],
        ) \
        .delegate_enrich(
          kess_service = "{{fountain_effective_follow_rank_model_service}}",
          send_common_attrs = [
            { "name": "fullrank_effective_follow_trimmed_user_info", "as": "user_info_str" }
          ],
          send_item_attrs = [
            { "name": "fullrank_sim_pevtr", "as": "pctr"},
            { "name": "fullrank_sim_pltr", "as": "pltr"},
            { "name": "fullrank_sim_pwtr", "as": "pwtr"},
            { "name": "fullrank_sim_pftr", "as": "pftr"},
            { "name": "fullrank_sim_plvtr", "as": "plvtr"},
            { "name": "fullrank_sim_pvtr", "as": "pvtr"},
            { "name": "fullrank_sim_pptr", "as": "pptr"},
            { "name": "fullrank_sim_pcmtr", "as": "pcmtr"},
            "cascade_pctr",
            "cascade_pltr",
            "cascade_pwtr",
            "cascade_pftr",
            "cascade_plvtr",
            {"name": "cascade_psvtr", "as": "cascade_psvr"},
          ],
          recv_item_attrs = [
            {"name": "is_effective_follow", "as": "effective_follow_rate_score"},
            {"name": "effective_follow_value", "as": "effective_follow_value_score"}
          ],
          timeout_ms = 100,
          request_type = "{{fountain_effective_follow_rank_model_request_type}}",
          partition_size = "{{fountain_effective_follow_rank_model_partition_size}}",
        ) \
      .else_if_("enable_fountain_effective_interact_model_predict == 1") \
        .explore_custom_trim_user_info(
          user_info_attr = "userInfo",
          save_trimed_user_info_to_attr = "fullrank_effective_interact_trimmed_user_info",
          trim_user_info = [
            "device_stat.human_action",
            "device_stat.device_status_flags",
            "id",
            "gender",
            "device_id",
            "user_active_level",
            "location.province_id",
            "location.city_id",
            "user_profile_v1.click_list.photo_id",
            "user_profile_v1.like_list.photo_id",
            "user_profile_v1.follow_list.photo_id",
          ],
        ) \
        .delegate_enrich(
          kess_service = "{{fountain_effective_interact_rank_model_service}}",
          send_common_attrs = [
            { "name": "fullrank_effective_interact_trimmed_user_info", "as": "user_info_str" }
          ],
          send_item_attrs = [
            { "name": "fullrank_sim_pevtr", "as": "pctr"},
            { "name": "fullrank_sim_pltr", "as": "pltr"},
            { "name": "fullrank_sim_pwtr", "as": "pwtr"},
            { "name": "fullrank_sim_pftr", "as": "pftr"},
            { "name": "fullrank_sim_plvtr", "as": "plvtr"},
            { "name": "fullrank_sim_pvtr", "as": "pvtr"},
            { "name": "fullrank_sim_pptr", "as": "pptr"},
            { "name": "fullrank_sim_pcmtr", "as": "pcmtr"},
            "cascade_pctr",
            "cascade_pltr",
            "cascade_pwtr",
            "cascade_pftr",
            "cascade_plvtr",
            {"name": "cascade_psvtr", "as": "cascade_psvr"},
          ],
          recv_item_attrs = [
            {"name": "is_effective_follow", "as": "effective_follow_rate_score"},
            {"name": "effective_follow_value", "as": "effective_follow_value_score"},
            {"name": "is_effective_interact", "as": "effective_interact_rate_score"},
            {"name": "effective_interact_value", "as": "effective_interact_value_score"},
          ],
          timeout_ms = 100,
          request_type = "{{fountain_effective_interact_rank_model_request_type}}",
          partition_size = "{{fountain_effective_interact_rank_model_partition_size}}",
        ) \
      .end_()
    return self

  def fullrank_effective_follow_ua_predict(self):
    self \
      .if_("enable_fountain_effective_follow_ua_model_predict == 1") \
        .explore_custom_trim_user_info(
          user_info_attr = "userInfo",
          save_trimed_user_info_to_attr = "fullrank_effective_follow_ua_trimmed_user_info",
          trim_user_info = [
            "device_stat.human_action",
            "device_stat.device_status_flags",
            "id",
            "device_id",
            "gender"
            "user_active_level",
            "location.province_id",
            "location.city_id",
            "basic_info.age_segment",
          ],
        ) \
        .delegate_enrich(
          kess_service = "{{fountain_effective_follow_ua_model_service}}",
          send_common_attrs = [
            { "name": "fullrank_effective_follow_ua_trimmed_user_info", "as": "user_info_str" }
          ],
          send_item_attrs = [
            { "name": "fullrank_sim_pevtr", "as": "pctr"},
            { "name": "fullrank_sim_pltr", "as": "pltr"},
            { "name": "fullrank_sim_pwtr", "as": "pwtr"},
            { "name": "fullrank_sim_pftr", "as": "pftr"},
            { "name": "fullrank_sim_plvtr", "as": "plvtr"},
            { "name": "fullrank_sim_pvtr", "as": "pvtr"},
            { "name": "fullrank_sim_pptr", "as": "pptr"},
            { "name": "fullrank_sim_pcmtr", "as": "pcmtr"},
            "cascade_pctr",
            "cascade_pltr",
            "cascade_pwtr",
            "cascade_pftr",
            "cascade_plvtr",
            {"name": "cascade_psvtr", "as": "cascade_psvr"}
          ],
          recv_item_attrs = [
            {"name": "is_effective_follow", "as": "effective_follow_ua_score"}
          ],
          timeout_ms = 100,
          request_type = "{{fountain_effective_follow_ua_model_request_type}}",
          partition_size = "{{fountain_effective_follow_ua_model_partition_size}}"
        ) \
      .end_()
    return self

  def fullrank_effective_follow_ua_pfd_predict(self):
    self \
      .if_("enable_fountain_effective_follow_ua_pfd_model_predict == 1") \
        .explore_custom_trim_user_info(
          user_info_attr = "userInfo",
          save_trimed_user_info_to_attr = "fullrank_effective_follow_ua_pfd_trimmed_user_info",
          trim_user_info = [
            "device_stat.human_action",
            "device_stat.device_status_flags",
            "id",
            "device_id",
            "gender"
            "user_active_level",
            "location.province_id",
            "location.city_id",
            "basic_info.age_segment",
          ],
        ) \
        .delegate_enrich(
          kess_service = "{{fountain_effective_follow_ua_pfd_model_service}}",
          send_common_attrs = [
            { "name": "fullrank_effective_follow_ua_pfd_trimmed_user_info", "as": "user_info_str" }
          ],
          send_item_attrs = [
            { "name": "fullrank_sim_pevtr", "as": "pctr"},
            { "name": "fullrank_sim_pltr", "as": "pltr"},
            { "name": "fullrank_sim_pwtr", "as": "pwtr"},
            { "name": "fullrank_sim_pftr", "as": "pftr"},
            { "name": "fullrank_sim_plvtr", "as": "plvtr"},
            { "name": "fullrank_sim_pvtr", "as": "pvtr"},
            { "name": "fullrank_sim_pptr", "as": "pptr"},
            { "name": "fullrank_sim_pcmtr", "as": "pcmtr"},
            "cascade_pctr",
            "cascade_pltr",
            "cascade_pwtr",
            "cascade_pftr",
            "cascade_plvtr",
            {"name": "cascade_psvtr", "as": "cascade_psvr"}
          ],
          recv_item_attrs = [
            {"name": "is_effective_follow", "as": "effective_follow_ua_pfd_score"}
          ],
          timeout_ms = 100,
          request_type = "{{fountain_effective_follow_ua_pfd_model_request_type}}",
          partition_size = "{{fountain_effective_follow_ua_pfd_model_partition_size}}"
        ) \
      .end_()
    return self

  def fullrank_ua_tracking_predict(self):
    self \
      .if_("enable_fountain_ua_tracking_model_predict == 1") \
        .explore_custom_trim_user_info(
          user_info_attr = "userInfo",
          save_trimed_user_info_to_attr = "fullrank_ua_tracking_trimmed_user_info",
          trim_user_info = [
            "id",
            "device_id",
            "gender",
            "user_profile_v1.like_list.photo_id",
            "user_profile_v1.follow_list.photo_id",
            "user_profile_v1.click_list.photo_id",
          ],
        ) \
        .delegate_enrich(
          kess_service = "{{fountain_ua_tracking_model_service}}",
          send_common_attrs = [
            { "name": "fullrank_ua_tracking_trimmed_user_info", "as": "user_info_str" }
          ],
          send_item_attrs = [
            { "name": "fullrank_sim_pevtr", "as": "pctr"},
            { "name": "fullrank_sim_pltr", "as": "pltr"},
            { "name": "fullrank_sim_pwtr", "as": "pwtr"},
            { "name": "fullrank_sim_pftr", "as": "pftr"},
            { "name": "fullrank_sim_plvtr", "as": "plvtr"},
            { "name": "fullrank_sim_pvtr", "as": "pvtr"},
            { "name": "fullrank_sim_pptr", "as": "pptr"},
            { "name": "fullrank_sim_pcmtr", "as": "pcmtr"},
            "cascade_pctr",
            "cascade_pltr",
            "cascade_pwtr",
            "cascade_pftr",
            "cascade_plvtr",
            {"name": "cascade_psvtr", "as": "cascade_psvr"}
          ],
          recv_item_attrs = [
            {"name": "active_tracking_score", "as": "ua_active_tracking_score"},
            {"name": "author_follow_long_reward_value", "as": "ua_tracking_score"},
          ],
          timeout_ms = 100,
          request_type = "{{fountain_ua_tracking_model_request_type}}",
          partition_size = "{{fountain_ua_tracking_model_partition_size}}"
        ) \
      .end_()
    return self
  
  def fullrank_wtd_by_frac_predict(self):
    self \
      .if_("enable_fullrank_wtd_by_frac_predict == 1") \
        .explore_custom_trim_user_info(
          user_info_attr = "userInfo",
          save_trimed_user_info_to_attr = "fr_wtd_by_frac_trimmed_user_info",
          trim_user_info = [
            "active_days",
            "basic_info.age_segment",
            "location.city_id",
            "location.region_type",
            "client_id",
            "device_id",
            "gender",
            "infer_gender",
            "true_gender",
            "request_location.poi_type",
            "request_location.province_id",
            "request_location.city_id",
            "visit_mod",
            "user_profile.exp_stat.exp_click",
            "user_profile.exp_stat.exp_like",
            "user_profile.exp_stat.exp_follow",
            "user_profile.exp_stat.exp_realshow",
            "user_profile.exp_stat.exp_long_view",
            "user_profile.user_level",
            "fountain_reco_user_profile.click_list.author_id",
            "fountain_reco_user_profile.click_list.photo_id",
            "fountain_reco_user_profile.comment_list.author_id",
            "fountain_reco_user_profile.comment_list.photo_id",
            "fountain_reco_user_profile.follow_list.author_id",
            "fountain_reco_user_profile.follow_list.photo_id",
            "fountain_reco_user_profile.like_list.author_id",
            "fountain_reco_user_profile.like_list.photo_id",
            "fountain_reco_user_profile.video_play_stat.photo_id",
            "fountain_reco_user_profile.video_play_stat.author_id",
            "fountain_reco_user_profile.video_play_stat.video_duration",
            "fountain_reco_user_profile.video_play_stat.playing_time",
            "user_profile_v1.click_list.author_id",
            "user_profile_v1.click_list.photo_id",
            "user_profile_v1.follow_list.author_id",
            "user_profile_v1.follow_list.photo_id",
            "user_profile_v1.like_list.author_id",
            "user_profile_v1.like_list.photo_id",
            "user_profile_v1.video_playing_stat.playing_time",
            "user_profile_v1.video_playing_stat.author_id",
            "user_profile_v1.video_playing_stat.photo_id",
            "realtime_click_list",
            "realtime_follow_list",
            "realtime_forward_list",
            "realtime_like_list",
          ],
        ) \
        .delegate_enrich(
          kess_service = "{{fountain_fullrank_wtd_by_frac_kess_service}}",
          recv_item_attrs = [
            {"name": "fountain_wtd_raw", "as": "fullrank_wtd_by_frac_score"},
            {"name": "gwtd_evtr", "as": "fullrank_wtd_by_frac_evtr_score"},
            {"name": "gwtd_lvtr", "as": "fullrank_wtd_by_frac_lvtr_score"},
            {"name": "gwtd_svtr", "as": "fullrank_wtd_by_frac_svtr_score"},
            {"name": "gwtd_cpr", "as": "fullrank_wtd_by_frac_cpr_score"},
          ],
          timeout_ms = 100,
          send_item_attrs = [
            "cascade_pctr",
            "cascade_pltr",
            "cascade_pwtr",
            "cascade_plvtr",
            "cascade_psvtr",
            "fullrank_detail_pctr",
            "fullrank_detail_pltr",
            "fullrank_detail_pwtr",
            "fullrank_detail_pftr",
            "fullrank_detail_plvtr",
            "fullrank_detail_pvtr",
            "fullrank_detail_psvr",
            "fullrank_detail_pcmtr",
            "fullrank_detail_pptr",
            "fullrank_detail_pwtd",
          ],
          send_common_attrs = [
            { "name": "fr_wtd_by_frac_trimmed_user_info", "as": "user_info_str" },
            { "name": "featureSourcePId", "as": "source_pid" },
            { "name": "sourcePidDuration", "as": "source_duration_ms" },
            { "name": "sourcePidTagId", "as": "source_tag" },
            { "name": "sourcePidAuthorId", "as": "source_aid" },
            { "name": "sourcePidHetuLevelOneList", "as": "source_hetu_tag_level1_list" },
            { "name": "sourcePidHetuLevelTwoList", "as": "source_hetu_tag_level2_list" },
            { "name": "featureSimilarUserList", "as": "similar_user_list" },
          ],
          request_type = "{{fountain_wtd_by_frac_request_type}}",
          partition_size = "{{fountain_wtd_by_frac_partition_size}}",
        ) \
      .end_()

    return self

  def _timestamp_begin(self, name: str):
    return self \
      .gen_common_attr_by_lua(
        attr_map = {
          name + "_begin_ts": "util.GetTimestamp()",
        },
      )

  def explore_sphinx_param(self):
    return self \
    .if_("fullrank_enable_jarvis_param > 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .enrich_attr_by_lua(
        import_common_attr = ["featureUId"],
        export_common_attr = ["user_stat_redis_key", "user_app_redis_key"],
        function_for_common = "gen_redis_key_for_sphinx",
        lua_script_file = "fountain/full_rank/lua/trans_reason_to_str.lua"
      ) \
      .get_common_attr_from_redis(
        cluster_name = "recoHotUserStatForLtr",
        redis_params = [
          {
            "redis_key": "{{user_stat_redis_key}}",
            "output_attr_name": "user_stat_xtr_attr_from_redis"
          }
        ]
      ) \
      .get_common_attr_from_redis(
        cluster_name = "recoExploreDegradeLeaf",
        redis_params = [
          {
            "redis_key": "{{user_app_redis_key}}",
            "output_attr_name": "user_app_attr_from_redis"
          }
        ]
      ) \
      .explore_sphinx_param_enrich(
        user_info_pb_name = "userInfoPb",
        session_id_attr = "sessionId",
        user_stat_attr = "user_stat_xtr_attr_from_redis",
        user_app_attr = "user_app_attr_from_redis",
        use_emp_play="{{fountain_fullrank_param_jarvis_use_emp_play}}",
        request_based_jarvis_enabled=True,
        jarvis_kess_service="{{fullrank_param_jarvis_service}}",
        jarvis_model_name="{{fullrank_param_jarvis_model_name}}",
        app_name="{{fullrank_param_jarvis_app_name}}",
        action_type="{{fullrank_param_jarvis_action_type}}",
        jarvis_time_out=150,
        use_app_cat=True,
        use_expxtr=True,
        item_attrs={
          "author_id": "author__id",
          "pevtr": "fullrank_sim_pevtr",
          "plvtr": "fullrank_sim_plvtr",
          "psvr": "fullrank_sim_psvr",
          "pvtr": "fullrank_sim_pvtr",
          "pltr": "fullrank_sim_pltr",
          "phtr": "fullrank_sim_phtr",
          "pwtr": "fullrank_sim_pwtr",
          "pftr": "fullrank_sim_pftr",
          "pptr": "fullrank_sim_pptr",
          "pcmtr": "fullrank_sim_pcmtr",
          "pcmef": "fullrank_sim_pcmef",
          "pepstr": "fullrank_sim_pepstr",
          "plsr": "fullrank_sim_lstr",
          "pcltr": "fullrank_sim_pcltr",
          "pfr_score1": "fullrank_sim_pfintr",
          "duration_ms": "duration_ms",
          "hetu_level_one": "hetu_tag_level_info__hetu_level_one",
          "hetu_level_two": "hetu_tag_level_info__hetu_level_two",
          "hetu_level_three": "hetu_tag_level_info__hetu_level_three"
        },
        queues=[
          {
            "name": "adjust:rl_rerank_fr_score1",
            "param_attr": "fullrank_vtr_adjust_ratio_attr"
          },
          {
            "name": "adjust:rl_rerank_fr_score2",
            "param_attr": "fullrank_lvtr_adjust_ratio_attr"
          },
          {
            "name": "adjust:rl_rerank_ltr",
            "param_attr": "fullrank_ltr_adjust_ratio_attr"
          },
          {
            "name": "adjust:rl_rerank_wtr",
            "param_attr": "fullrank_wtr_adjust_ratio_attr"
          },
          {
            "name": "adjust:rl_rerank_ftr",
            "param_attr": "fullrank_ftr_adjust_ratio_attr"
          },
          {
            "name": "adjust:rl_rerank_cmtr",
            "param_attr": "fullrank_cmtr_adjust_ratio_attr"
          },
          {
            "name": "adjust:rl_rerank_cmef",
            "param_attr": "fullrank_cmef_adjust_ratio_attr"
          },
          {
            "name": "adjust:rl_rerank_epstr",
            "param_attr": "fullrank_epstr_adjust_ratio_attr"
          },
          {
            "name": "adjust:rl_rerank_lstr",
            "param_attr": "fullrank_lstr_adjust_ratio_attr"
          },
          {
            "name": "adjust:rl_rerank_ctr",
            "param_attr": "fullrank_ctr_adjust_ratio_attr"
          },
          {
            "name": "adjust:rl_rerank_evtr_v2",
            "param_attr": "fullrank_evtr_v2_adjust_ratio_attr"
          },
          {
            "name": "adjust:rl_rerank_swptr",
            "param_attr": "fullrank_swptr_adjust_ratio_attr"
          },
          {
            "name": "adjust:rl_rerank_next",
            "param_attr": "fullrank_next_adjust_ratio_attr"
          },
          {
            "name": "adjust:rl_rerank_finish",
            "param_attr": "fullrank_finish_adjust_ratio_attr"
          },
          {
            "name": "adjust:rl_rerank_l2r",
            "param_attr": "fullrank_l2r_adjust_ratio_attr"
          },
          {
            "name": "adjust:rl_rerank_fintr",
            "param_attr": "fullrank_fintr_adjust_ratio_attr"
          },
          {
            "name": "adjust:rl_rerank_cpr",
            "param_attr": "fullrank_cpr_adjust_ratio_attr"
          },
          {
            "name": "adjust:rl_rerank_vtr_ori",
            "param_attr": "fullrank_vtr_ori_adjust_ratio_attr"
          },
          {
            "name": "adjust:rl_rerank_lvtr_ori",
            "param_attr": "fullrank_lvtr_ori_adjust_ratio_attr"
          },
        ]
        ) \
      .enrich_attr_by_lua(
        import_common_attr = [
          "fullrank_ctr_adjust_ratio_attr",
          "fullrank_vtr_adjust_ratio_attr",
          "fullrank_lvtr_adjust_ratio_attr",
          "fullrank_ltr_adjust_ratio_attr",
          "fullrank_wtr_adjust_ratio_attr",
          "fullrank_cmtr_adjust_ratio_attr",
          "fullrank_cmef_adjust_ratio_attr",
          "fullrank_epstr_adjust_ratio_attr",
          "fullrank_lstr_adjust_ratio_attr",
          "fullrank_cpr_adjust_ratio_attr",
          "fullrank_fintr_adjust_ratio_attr",
          "fullrank_l2r_adjust_ratio_attr",
          "fullrank_finish_adjust_ratio_attr",
          "fullrank_next_adjust_ratio_attr",
          "fullrank_swptr_adjust_ratio_attr",
          "fullrank_evtr_v2_adjust_ratio_attr",
          "fullrank_vtr_ori_adjust_ratio_attr",
          "fullrank_lvtr_ori_adjust_ratio_attr",
          "fullrank_ftr_adjust_ratio_attr",
          "fountain_ensemble_power_weight_fullrank_click_score",
          "fountain_ensemble_power_weight_fullrank_like_score",
          "fountain_ensemble_power_weight_fullrank_follow_score",
          "fountain_ensemble_power_weight_fullrank_pcmtr_score",
          "fountain_ensemble_power_weight_fullrank_pptr_score",
          "fountain_ensemble_power_weight_fullrank_pepstr_score",
          "fountain_ensemble_power_weight_fullrank_lstr_score",
          "fountain_ensemble_power_weight_fullrank_pvtr_multi_pwtr",
          "fountain_ensemble_power_weight_fullrank_pvtr_multi_pptr",
          "fountain_ensemble_weight_fullrank_sim_plvtr",
          "fountain_ensemble_weight_fullrank_sim_pwatchtime_no_bias_debias",
          "fountain_ensemble_power_weight_fullrank_cmef_score",
          "fountain_ensemble_power_weight_fullrank_pcpr_score",
          "fountain_ensemble_power_weight_fullrank_pfintr_score",
          "fountain_ensemble_power_weight_fullrank_ltr_score",
          "fountain_ensemble_power_weight_fullrank_ltr_v4_fountain_finish_rate",
          "fountain_ensemble_power_weight_fullrank_ltr_v4_next",
          "fountain_ensemble_weight_fullrank_ori_pswptr",
          "fountain_ensemble_weight_fullrank_detail_new_pevtr_v2",
          "fountain_ensemble_power_weight_fullrank_pvtr_score",
          "fountain_ensemble_power_weight_fullrank_longview_score",
          "fountain_ensemble_weight_forward_score"
        ],
        export_common_attr = [
          "fountain_ensemble_power_weight_fullrank_click_score",
          "fountain_ensemble_power_weight_fullrank_like_score",
          "fountain_ensemble_power_weight_fullrank_follow_score",
          "fountain_ensemble_power_weight_fullrank_pcmtr_score",
          "fountain_ensemble_power_weight_fullrank_pptr_score",
          "fountain_ensemble_power_weight_fullrank_pepstr_score",
          "fountain_ensemble_power_weight_fullrank_pvtr_multi_pwtr",
          "fountain_ensemble_power_weight_fullrank_pvtr_multi_pptr",
          "fountain_ensemble_weight_fullrank_sim_plvtr",
          "fountain_ensemble_weight_fullrank_sim_pwatchtime_no_bias_debias",
          "fountain_ensemble_power_weight_fullrank_cmef_score",
          "fountain_ensemble_power_weight_fullrank_lstr_score",
          "fountain_ensemble_power_weight_fullrank_pcpr_score",
          "fountain_ensemble_power_weight_fullrank_pfintr_score",
          "fountain_ensemble_power_weight_fullrank_ltr_score",
          "fountain_ensemble_power_weight_fullrank_ltr_v4_fountain_finish_rate",
          "fountain_ensemble_power_weight_fullrank_ltr_v4_next",
          "fountain_ensemble_weight_fullrank_ori_pswptr",
          "fountain_ensemble_weight_fullrank_detail_new_pevtr_v2",
          "fountain_ensemble_power_weight_fullrank_pvtr_score",
          "fountain_ensemble_power_weight_fullrank_longview_score",
          "fountain_ensemble_weight_forward_score"
        ],
        function_for_common = "cal_fullrank_adaptive_weights_rl",
        lua_script_file = "fountain/full_rank/lua/cal_adaptive_weight.lua"
      ) \
        .log_debug_info(
        common_attrs = ["fullrank_param_jarvis_service",
                        "fullrank_param_jarvis_model_name",
                        "fullrank_param_jarvis_app_name",
                        "fullrank_param_jarvis_action_type",
                        "sessionId",
                        "featureUId",
                        "user_stat_redis_key",
                        "user_app_redis_key",
                        "user_stat_xtr_attr_from_redis",
                        "user_app_attr_from_redis",
                        "fullrank_vtr_adjust_ratio_attr",
                        "fullrank_lvtr_adjust_ratio_attr",
                        "fullrank_ltr_adjust_ratio_attr",
                        "fullrank_wtr_adjust_ratio_attr",
                        "fullrank_ftr_adjust_ratio_attr",
                        "fullrank_cmtr_adjust_ratio_attr",
                        "fullrank_cmef_adjust_ratio_attr",
                        "fullrank_epstr_adjust_ratio_attr",
                        "fullrank_lstr_adjust_ratio_attr"],
        for_debug_request_only = True
      ) \
    .end_if_() \

  def fetch_similar_user_list(self):
    self \
      .retrieve_by_ann_embedding(
        reason = 1000000,
        kess_service = "{{fountain_similar_user_list_ann_server}}",
        timeout_ms = 300,
        items_from_attr = ["_USER_ID_"],
        # embeddings_from_attr = ["user_embedding_list"],
        bound_type = {
          "top_k": 100
        },
        algo_type = {
          "faiss": {}
        },
        src_data_type = "user",
        src_bucket = "user",
        dest_bucket = "user",
        save_result_to_common_attr = "featureSimilarUserList",
        skip = "{{skip_fetch_similar_user_list_by_ann}}"
      ) \
      .if_("skip_fetch_similar_user_list_by_redis == 0") \
        .enrich_attr_by_lua(
          import_common_attr = [
            "featureUId",
            "fountain_similar_user_list_redis_prefix",
          ],
          export_common_attr = [
            "fetch_similar_user_list_redis_key"
          ],
          function_for_common = "gen_similar_users_redis_key",
          lua_script_file = "fountain/full_rank/lua/trans_reason_to_str.lua"
        ) \
        .get_common_attr_from_redis(
          cluster_name = "recoColossusTriggers",
          redis_params = [
            {
              "redis_key": "{{fetch_similar_user_list_redis_key}}",
              "output_attr_name": "featureSimilarUserList_str"
            }
          ]
        ) \
        .split_string(
          input_common_attr = "featureSimilarUserList_str",
          output_common_attr = "featureSimilarUserList_str_list",
          delimiters=",",
        ) \
        .merchant_split_string_list(
          input_common_attr = "featureSimilarUserList_str_list",
          output_attr_configs = [
            {
              "export_common_attr": "featureSimilarUserList",
              "parse_to_type": "list_int64",
              "pos_in_splitted": 0,
              "default_value": "-1"
            },
            {
              "export_common_attr": "featureSimilarUserScoreList",
              "parse_to_type": "list_double",
              "pos_in_splitted": 1,
              "default_value": "0.0"
            }],
          delimiters=":"
        ) \
      .end_if_() \
      .log_debug_info(
        common_attrs = [
          "featureSimilarUserList",
          "featureSimilarUserScoreList"
        ],
        item_num_limit = 10,
        for_debug_request_only = True,
      )
    return self

  def boost_comment_weights(self):
    return self \
    .if_("enable_fr_boost_teenager_comment_weights > 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .enrich_with_protobuf(
        from_extra_var = "userInfoPb",
        attrs = [
          dict(name="age_segment", path="basic_info.age_segment"),
        ],
      ) \
      .enrich_attr_by_lua(
        import_common_attr = [
          "fountain_ensemble_power_weight_fullrank_pcmtr_score",
          "fountain_ensemble_power_weight_fullrank_cmef_score",
          "fountain_comment_targeted_teenager_age_segment_upper_bound",
          "age_segment",
          "fountain_ensemble_power_weight_fullrank_pcmtr_score_teenager_coeff",
          "fountain_ensemble_power_weight_fullrank_cmef_score_teenager_coeff",
        ],
        export_common_attr = [
          "fountain_ensemble_power_weight_fullrank_pcmtr_score",
          "fountain_ensemble_power_weight_fullrank_cmef_score"
        ],
        function_for_common = "boost_teenager_comment_weights",
        lua_script_file = "fountain/full_rank/lua/cal_adaptive_weight.lua"
      ) \
      .log_debug_info(
        common_attrs = [
          "age_segment",
          "fountain_ensemble_power_weight_fullrank_pcmtr_score",
          "fountain_ensemble_power_weight_fullrank_cmef_score"
        ],
        item_num_limit = 1,
        for_debug_request_only = True,
      ) \
    .end_if_() \

  def boost_low_follow_user_follow_weight(self):
    return self \
    .if_("fountain_fullrank_weight_low_follow == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_fullrank_low_follow_pwtr_weight", "as": "rank_low_follow_pwtr_weight"},
          {"name": "fountain_fullrank_low_follow_thres_s", "as": "rank_low_follow_thres_s"},
          {"name": "fountain_fullrank_enable_no_follow_boost", "as": "enable_no_follow_boost"},
          "follow_timestamps",
          {"name": "fountain_ensemble_power_weight_fullrank_pvtr_multi_pwtr", "as": "input_pwtr_score"},
          {"name": "fountain_fullrank_enable_low_follow_boost", "as": "enable_low_follow_boost"},
          {"name": "fountain_fullrank_low_follow_boost_threshold", "as": "low_follow_boost_threshold"},
          {"name": "user_follow_type", "as": "user_follow_type"},
          {"name": "fountain_fullrank_no_follow_pwtr_weight", "as": "no_follow_pwtr_weight"},
          {"name": "fountain_fullrank_valid_follow_pwtr_weight", "as": "valid_follow_pwtr_weight"},
          {"name": "fountain_fullrank_valid_low_follow_pwtr_weight", "as": "valid_low_follow_pwtr_weight"},
          {"name": "fountain_fullrank_valid_media_follow_pwtr_weight", "as": "valid_media_follow_pwtr_weight"},
          {"name": "fountain_fullrank_valid_high_follow_pwtr_weight", "as": "valid_high_follow_pwtr_weight"}
        ],
        export_common_attr = [
          {"name": "output_pwtr_score", "as": "fountain_ensemble_power_weight_fullrank_pvtr_multi_pwtr"},
        ],
        function_name = "UserSortWeightLowFollow",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("enable_fountain_fullrank_effective_follow_ua_adjust == 1 and (followAids == nil or #followAids < fountain_fullrank_effective_follow_ua_adjust_require_follow_num)") \
      .gen_common_attr_by_lua(
        attr_map={
          "fountain_ensemble_power_weight_effective_follow_ua_score": "fountain_ensemble_power_weight_effective_follow_ua_score * fountain_fullrank_effective_follow_ua_adjust_weight",
        }
      ) \
    .end_if_() \
    .if_("enable_fountain_fullrank_effective_follow_ua_pfd_adjust == 1 and (followAids == nil or #followAids < fountain_fullrank_effective_follow_ua_pfd_adjust_require_follow_num)") \
      .gen_common_attr_by_lua(
        attr_map={
          "fountain_ensemble_power_weight_effective_follow_ua_pfd_score": "fountain_ensemble_power_weight_effective_follow_ua_pfd_score * fountain_fullrank_effective_follow_ua_pfd_adjust_weight",
        }
      ) \
    .end_if_() \

  def fr_discount_single_pic(self):
    return self \
    .if_("enable_fr_discount_single_pic == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "cascading_support_author_discount_coef", "as": "boost_discount_coeff"},
        ],
        import_item_attr = [
          "duration_ms",
          "upload_type",
          "picture_type",
          "photo_picture_count",
          {"name": "fullrank_ensemble_score", "as": "boost_discount_score"},
        ],
        export_item_attr = [
          {"name": "boost_discount_score", "as": "fullrank_ensemble_score"},
        ],
        function_name = "BoostOrDiscountSinglePic",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_if_()

  def fr_ensemble_score_multiply_gate(self):
    self \
    .if_("enable_fr_ensemble_score_multiply_gate == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_fr_psvtr_gate_alpha", "as": "svtr_alpha"},
          {"name": "fountain_fr_psvtr_gate_beta", "as": "svtr_beta"},
          {"name": "fountain_fr_pctr_gate_alpha", "as": "ctr_alpha"},
          {"name": "fountain_fr_pctr_gate_beta", "as": "ctr_beta"},
        ],
        import_item_attr = [
          {"name": "fullrank_ensemble_score", "as": "es_score"},
          {"name": "fullrank_sim_psvr", "as": "svtr_score"},
          {"name": "fullrank_sim_click_score", "as": "ctr_score"},
        ],
        export_item_attr = [
          {"name": "es_score", "as": "fullrank_ensemble_score"},
        ],
        function_name = "EsScoreMultiplyGate",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_if_()

    return self

  def get_simple_ltr_feature(self):
    self \
    .get_kconf_params(
      kconf_configs = [
        {
          "kconf_key": "reco.offline.recoAllUserEmpXtr1",
          "value_type": "json",
          "json_path": "emp_ltr",
          "export_common_attr": "all_emp_ltr"
        },
        {
          "kconf_key": "reco.offline.recoAllUserEmpXtr1",
          "value_type": "json",
          "json_path": "emp_wtr",
          "export_common_attr": "all_emp_wtr"
        },
        {
          "kconf_key": "reco.offline.recoAllUserEmpXtr1",
          "value_type": "json",
          "json_path": "emp_ftr",
          "export_common_attr": "all_emp_ftr"
        },
        {
          "kconf_key": "reco.offline.recoAllUserEmpXtr1",
          "value_type": "json",
          "json_path": "emp_cmtr",
          "export_common_attr": "all_emp_cmtr"
        },
        {
          "kconf_key": "reco.offline.recoAllUserEmpXtr1",
          "value_type": "json",
          "json_path": "emp_eps",
          "export_common_attr": "all_emp_eps"
        },
        {
          "kconf_key": "reco.offline.recoAllUserEmpXtr1",
          "value_type": "json",
          "json_path": "emp_htr",
          "export_common_attr": "all_emp_htr"
        }
      ]
    ) \
    .enrich_attr_by_lua(
      import_common_attr = [
        "user_emp_ltr",
        "user_emp_wtr",
        "user_emp_cmtr",
        "user_emp_ftr",
        "user_emp_eptr",
        "user_emp_htr",
        "all_emp_ltr",
        "all_emp_wtr",
        "all_emp_ftr",
        "all_emp_cmtr",
        "all_emp_eps",
        "all_emp_htr"
      ],
      export_common_attr = [
        "featureUserLtrNew",
        "featureUserWtrNew",
        "featureUserFtrNew",
        "featureUserCmtrNew",
        "featureUserEptrNew",
        "featureUserHtrNew"
      ],
      function_for_common = "get_simple_ltr_feature",
      lua_script_file = "fountain/full_rank/lua/fullrank_feature_trans.lua"
    ) \
    .log_debug_info(
      common_attrs = [
        "user_emp_ltr",
        "user_emp_wtr",
        "user_emp_cmtr",
        "user_emp_ftr",
        "user_emp_eptr",
        "user_emp_htr",
        "featureUserLtrNew",
        "featureUserWtrNew",
        "featureUserFtrNew",
        "featureUserCmtrNew",
        "featureUserEptrNew",
        "featureUserHtrNew",
        "all_emp_ltr",
        "all_emp_wtr",
        "all_emp_ftr",
        "all_emp_cmtr",
        "all_emp_eps",
        "all_emp_htr"
      ],
      for_debug_request_only = True
    )

    return self

  def fetch_duration_group_id(self):
    self \
      .if_("skip_fountain_fetch_duration_group_id == 0") \
        .get_kconf_params(
          kconf_configs = [
            {
              "kconf_key": "reco.fountain.durationGroupId",
              "value_type": "list_double",
              "default_value": [],
              "export_common_attr": "faActionL2rV4DurationId_threshold_list"
            },
          ]
        )\
        .get_kconf_params(
          kconf_configs = [
            {
              "kconf_key": "reco.fountain.fountainActionV4VtrMaxList",
              "value_type": "list_double",
              "default_value": [],
              "export_common_attr": "fountain_fullrank_ltr_v4_vtr_max_list"
            },
          ]
        )\
        .enrich_attr_by_lua(
          import_common_attr = [
            "faActionL2rV4DurationId_threshold_list",
            "fountain_duration_s_id_max",
            "fountain_fullrank_ltr_v4_vtr_max_list",
          ],
          import_item_attr = [
            "duration_ms",
          ],
          export_item_attr = [
            "faActionL2rV4DurationId",
            "featureDurationSId",
            "fountain_act_vtr_max",
          ],
          function_for_item = "fetch_duration_group_id",
          lua_script_file = "fountain/full_rank/lua/trans_item_attr.lua"
        ) \
      .end_if_()
    return self

  def fountain_sl_only_fast(self):
    self \
    .enrich_attr_by_lua(
      import_common_attr = [
        "fountain_ensemble_weight_fullrank_sl_interactive_score",
        "fountain_sl_only_fast_v1",
        "page"
      ],
      export_common_attr = [
        "fountain_ensemble_weight_fullrank_sl_interactive_score"
      ],
      function_for_common = "change_sl_interactive_score",
      lua_script_file = "fountain/full_rank/lua/cal_sl_interactive_score.lua"
    ) \

    return self

  def fountain_pure_value_only_fast(self):
    self \
    .enrich_attr_by_lua(
      import_common_attr = [
        "fountain_ensemble_weight_fullrank_pure_value_score",
        "skip_fullrank_pure_value_es_rank_score",
        "skip_fullrank_pure_value_es_value_score",
        "fountain_pure_value_only_fast_v1",
        "page"
      ],
      export_common_attr = [
        "fountain_ensemble_weight_fullrank_pure_value_score",
        "skip_fullrank_pure_value_es_rank_score",
        "skip_fullrank_pure_value_es_value_score"
      ],
      function_for_common = "change_pure_value_score",
      lua_script_file = "fountain/full_rank/lua/cal_sl_interactive_score.lua"
    ) \

    return self

  def fullrank_duration_debias(self):
    self \
      .if_("enable_fountain_fullrank_duration_debias_f1 == 1") \
        .calc_by_formula1(
          kconf_key = "formula.scenarioKey93.fountain_duration_follow_score_debias",
          import_item_attr = [
            { "name": "fullrank_sim_pevtr", "as": "pctr"},
            { "name": "fullrank_sim_pltr", "as": "pltr"},
            { "name": "fullrank_sim_pwtr", "as": "pwtr"},
            { "name": "fullrank_sim_pftr", "as": "pftr"},
            { "name": "fullrank_sim_plvtr", "as": "plvtr"},
            { "name": "fullrank_sim_pvtr", "as": "pvtr"},
            { "name": "fullrank_sim_pptr", "as": "pptr"},
            "effective_follow_rate_score",
            "duration_ms"
          ],
          export_formula_value = [
            "duration_debias_score"
          ],
          abtest_biz_name = "KUAISHOU_APPS"
        ) \
      .end_()
    return self
  
  def fullrank_emp_xtr_debias(self):
    self \
    .calc_by_formula1(
      kconf_key = "formula.scenarioKey70.FountainEmpxtrDebiasRankSort",
      import_item_attr = [
        "fountain_stats__follow_count",
        "fountain_stats__like_count",
        "fountain_stats__real_show_count",
        "fountain_stats__view_length_sum",
        "fullrank_sim_plvtr",
        "fullrank_sim_pltr",
        "fullrank_sim_pwtr"
      ],
      import_common_attr = [
      ],
      export_formula_value = [
        {"name": "final_score", "as": "rank_es_score_f1"}
      ],
      abtest_biz_name = "KUAISHOU_APPS"
    ) \
    .item_attr_operation(
      item_attr_a = "fullrank_ensemble_score",
      item_attr_b = "rank_es_score_f1",
      operator = "*",
      output_attr = "fullrank_ensemble_score"
    )
    return self

  def fullrank_upload_time_debias(self):
    self \
      .if_("enable_fountain_fullrank_upload_time_debias_f1 == 1") \
        .calc_by_formula1(
          kconf_key = "formula.scenarioKey35.fountain_upload_time_follow_score_debias",
          import_item_attr = [
            { "name": "fullrank_sim_pevtr", "as": "pctr"},
            { "name": "fullrank_sim_pltr", "as": "pltr"},
            { "name": "fullrank_sim_pwtr", "as": "pwtr"},
            { "name": "fullrank_sim_pftr", "as": "pftr"},
            { "name": "fullrank_sim_plvtr", "as": "plvtr"},
            { "name": "fullrank_sim_pvtr", "as": "pvtr"},
            { "name": "fullrank_sim_pptr", "as": "pptr"},
            "effective_follow_rate_score",
            "upload_time"
          ],
          export_formula_value = [
            "upload_time_debias_score"
          ],
          abtest_biz_name = "KUAISHOU_APPS"
        ) \
      .end_()
    return self

  def fullrank_pxtr_debais(self):
      self \
        .explore_attrs_adjust_enricher(
          skip = "{{fountain_skip_fullrank_debais}}",
          item_attr_configs = fullrank_pxtr_debais_cfgs,
          common_attr_configs = [],
        ) \
        .enrich_attr_by_lua(
          skip = "{{fountain_fullrank_pxtr_sample_debais}}",
          import_common_attr = [
            "fountain_fullrank_follow_upsample_weight",
            "fountain_fullrank_commont_upsample_weight",
          ],
          import_item_attr = [
            "fullrank_sim_follow_score",
            "fullrank_sim_pcmtr",
          ],
          export_item_attr = [
            "fullrank_sim_follow_score",
            "fullrank_sim_pcmtr",
          ],
          function_for_item = "pxtr_sample_debais",
          lua_script_file = "fountain/full_rank/lua/calc_fullrank_score.lua",
        )
      return self

  def calc_debias_mix_score(self):
    self \
      .if_("enable_fountain_rank_debias_mix_score == 1 and (fountain_debias_pxtr_only_fast_v1 == 0 or page > 1)") \
        .explore_user_debias_xtr_v2_enricher(
          colossus_v2_attr_name = "colossus_resp_v2",
          user_info_ptr_attr = "userInfoPb",
          xtr_weight_str = "{{fountain_debias_mix_xtr_weight_str}}",
          shortterm_stat_show_count = "{{fountain_debias_mix_shortterm_stat_show_count}}",
          longterm_stat_click_count = "{{fountain_debias_mix_longterm_stat_click_count}}",
          hetu_tag_attr = "hetu_tag_level_info__hetu_level_one",
          duration_ms_attr = "duration_ms",
          ctr_attr = "fullrank_sim_click_score",
          ltr_attr = "fullrank_sim_pltr",
          wtr_attr = "fullrank_sim_pwtr",
          ftr_attr = "fullrank_sim_pftr",
          cmtr_attr = "fullrank_sim_pcmtr",
          pptr_attr = "fullrank_sim_pptr",
          playtime_attr = "fullrank_trans_pvtr_score",
          ctr_debias_attr = "fullrank_sim_click_score_debias_hetu",
          ltr_debias_attr = "fullrank_sim_pltr_debias_hetu",
          wtr_debias_attr = "fullrank_sim_pwtr_debias_hetu",
          ftr_debias_attr = "fullrank_sim_pftr_debias_hetu",
          cmtr_debias_attr = "fullrank_sim_pcmtr_debias_hetu",
          pptr_debias_attr = "fullrank_sim_pptr_debias_hetu",
          playtime_debias_attr = "fullrank_trans_pvtr_score_debias_duration",
          debias_mix_score_attr = "fullrank_debias_mix_score",
          page_type_attr = "fountain",
          stat_only_page = "{{fountain_debias_mix_stat_only_page}}",
          adjust_playtime_score = "{{fountain_debias_mix_adjust_playtime_score}}",
          playtime_score_coeff_str = "{{fountain_debias_mix_playtime_score_coeff_str}}",
          playtime_debias_by_duration = "{{fountain_debias_mix_playtime_debias_by_duration}}",
          debias_version = "{{fountain_debias_mix_debias_version}}",
          default_debias_value = "{{fountain_debias_mix_debias_default_debias_value}}",
          hetu_debias_value_str = "{{fountain_debias_mix_debias_hetu_debais_value_str}}"
        ) \
      .end_()
    return self

  def fountain_cal_hetu_second_debias_score_fr_s2(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "hetu_tag_level_info__hetu_level_two", "as": "extract_hetu_tag_list"},
      ],
      export_item_attr = [
        {"name": "first_hetu_tag", "as": "hetu_level_two_top1"},
      ],
      function_name = "ExtractFirstHetuTag",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .split_string(
      input_common_attr = "fountain_hetu_second_debias_xtr_weight_fr_s2_str",
      output_common_attr = "fountain_hetu_second_debias_xtr_weight_fr_s2_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "fountain_hetu_second_debias_xtr_power_fr_s2_str",
      output_common_attr = "fountain_hetu_second_debias_xtr_power_fr_s2_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "fountain_hetu_second_debias_xtr_buttom_fr_s2_str",
      output_common_attr = "fountain_hetu_second_debias_xtr_buttom_fr_s2_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "fountain_hetu_second_debias_xtr_upper_fr_s2_str",
      output_common_attr = "fountain_hetu_second_debias_xtr_upper_fr_s2_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .set_attr_value( 
      no_overwrite=True,
      common_attrs=[
        {
          "name": "fountain_fr_s2_hetu_second_debias_xtr_name_list",
          "type": "string_list",
          "value": self.debias_xtr_name()
        }
      ]
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_hetu_second_debias_xtr_weight_fr_s2_list", "as": "id_debias_xtr_weight_list"},
        {"name": "fountain_hetu_second_debias_xtr_power_fr_s2_list", "as": "id_debias_xtr_power_list"},
        {"name": "fountain_hetu_second_debias_xtr_buttom_fr_s2_list", "as": "id_debias_xtr_buttom_list"},
        {"name": "fountain_hetu_second_debias_xtr_upper_fr_s2_list", "as": "id_debias_xtr_upper_list"},
        {"name": "fountain_fr_s2_hetu_second_debias_xtr_name_list", "as": "fix_xtr_list"},
        {"name": "fountain_hetu_second_debias_default_bar_value", "as": "default_bar_value"},
        {"name": "fountain_hetu_second_debias_enable_set_default_score", "as": "enable_set_default_score"},
        {"name": "fountain_fr_s2_hetu_second_debias_default_score_coef", "as": "default_score_coef"},
      ],
      import_item_attr = [
        {"name": "hetu_level_two_top1", "as": "debias_id_feature"},
        "fullrank_sim_pevtr",
        "fullrank_sim_plvtr",
        "fullrank_sim_pltr",
        "fullrank_sim_pwtr",
        "fullrank_sim_pcltr",
        "fullrank_sim_pcmtr",
        "fullrank_sim_pfintr",
        "fullrank_sim_pwatchtime_no_bias",
        "fullrank_sim_pcpr",
        "fullrank_sim_psvr",
        "fullrank_sim_phtr"
      ],
      export_item_attr = [
        {"name": "debias_score", "as": "fullrank_hetu_second_debias_score"}
      ],
      function_name = "GenXtrScoreByIdFeature",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .if_("enable_fountain_fullrank_hetu_second_debias_high_report_adjust == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_fullrank_hetu_second_debias_high_report_adjust_negative_count_alpha", "as": "negative_count_alpha"},
          {"name": "fountain_fullrank_hetu_second_debias_high_report_adjust_negative_count_beta", "as": "negative_count_beta"},
          {"name": "fountain_fullrank_hetu_second_debias_high_report_adjust_negative_rate_alpha", "as": "negative_rate_alpha"},
          {"name": "fountain_fullrank_hetu_second_debias_high_report_adjust_negative_rate_beta", "as": "negative_rate_beta"},
          {"name": "fountain_fullrank_hetu_second_debias_high_report_adjust_report_count_alpha", "as": "report_count_alpha"},
          {"name": "fountain_fullrank_hetu_second_debias_high_report_adjust_report_count_beta", "as": "report_count_beta"},
          {"name": "fountain_fullrank_hetu_second_debias_high_report_adjust_report_rate_alpha", "as": "report_rate_alpha"},
          {"name": "fountain_fullrank_hetu_second_debias_high_report_adjust_report_rate_beta", "as": "report_rate_beta"},
          {"name": "fountain_fullrank_hetu_second_debias_high_report_adjust_nps_rate_alpha", "as": "nps_rate_alpha"},
          {"name": "fountain_fullrank_hetu_second_debias_high_report_adjust_nps_rate_beta", "as": "nps_rate_beta"}
        ],
        import_item_attr = [
          {"name": "explore_stat__show_count", "as": "show_count"},
          {"name": "explore_stat__negative_count", "as": "negative_count"},
          {"name": "explore_stat__report_detail__total_report_count", "as": "report_count"},
          "comment_nps_rate"
        ],
        export_item_attr = [
          {"name": "adjust_score", "as": "fullrank_hetu_second_debias_adjust_score"}
        ],
        function_name = "GetReportAdjustScore",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .item_attr_operation(
        item_attr_a = "fullrank_hetu_second_debias_score",
        item_attr_b = "fullrank_hetu_second_debias_adjust_score",
        operator = "*",
        output_attr = "fullrank_hetu_second_debias_score"
      ) \
    .end_if_() \
    .if_("enable_fountain_fr_bad_comment_pids_score_adjust == 1") \
      .fr_bad_comment_pids_hetu_second_debias_score_adjust("fullrank_hetu_second_debias_score") \
    .end_if_()
    return self
  
  def debias_xtr_name(self):
    update_fix_xtrs = [
      "fullrank_sim_pevtr",
      "fullrank_sim_plvtr",
      "fullrank_sim_pltr",
      "fullrank_sim_pwtr",
      "fullrank_sim_pcltr",
      "fullrank_sim_pcmtr",
      "fullrank_sim_pfintr",
      "fullrank_sim_pwatchtime_no_bias",
      "fullrank_sim_pcpr",
      "fullrank_sim_psvr",
      "fullrank_sim_phtr"
    ]
    return update_fix_xtrs

  def calc_debias_score(self):
    self \
      .if_("enable_fountain_rank_debias_pctr_score == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_rank_debias_pctr_default_debias_value", "as": "default_debias_value"},
            {"name": "fountain_rank_debias_pctr_debias_value_str", "as": "debias_value_str"},
            {"name": "fountain_rank_debias_pctr_debias_version", "as": "debias_version"},
          ],
          import_item_attr = [
            {"name": "fullrank_sim_click_score", "as": "xtr_input"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_list_input"},
            {"name": "duration_ms", "as": "duration_ms_input"},
          ],
          export_item_attr = [
            {"name": "xtr_output", "as": "fullrank_sim_click_score_debias_hetu"},
          ],
          function_name = "CalcDebiasXtr",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_fountain_rank_debias_pwatchtime_score == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_rank_debias_pwatchtime_default_debias_value", "as": "default_debias_value"},
            {"name": "fountain_rank_debias_pwatchtime_debias_value_str", "as": "debias_value_str"},
            {"name": "fountain_rank_debias_pwatchtime_debias_version", "as": "debias_version"},
          ],
          import_item_attr = [
            {"name": "fullrank_trans_pvtr_score", "as": "xtr_input"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_list_input"},
            {"name": "duration_ms", "as": "duration_ms_input"},
          ],
          export_item_attr = [
            {"name": "xtr_output", "as": "fullrank_trans_pvtr_score_debias_duration"},
          ],
          function_name = "CalcDebiasXtr",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_fountain_rank_debias_pltr_score == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_rank_debias_pltr_default_debias_value", "as": "default_debias_value"},
            {"name": "fountain_rank_debias_pltr_debias_value_str", "as": "debias_value_str"},
            {"name": "fountain_rank_debias_pltr_debias_version", "as": "debias_version"},
          ],
          import_item_attr = [
            {"name": "fullrank_sim_pltr", "as": "xtr_input"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_list_input"},
            {"name": "duration_ms", "as": "duration_ms_input"},
          ],
          export_item_attr = [
            {"name": "xtr_output", "as": "fullrank_sim_pltr_debias_hetu"},
          ],
          function_name = "CalcDebiasXtr",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_fountain_rank_debias_pwtr_score == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_rank_debias_pwtr_default_debias_value", "as": "default_debias_value"},
            {"name": "fountain_rank_debias_pwtr_debias_value_str", "as": "debias_value_str"},
            {"name": "fountain_rank_debias_pwtr_debias_version", "as": "debias_version"},
          ],
          import_item_attr = [
            {"name": "fullrank_sim_pwtr", "as": "xtr_input"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_list_input"},
            {"name": "duration_ms", "as": "duration_ms_input"},
          ],
          export_item_attr = [
            {"name": "xtr_output", "as": "fullrank_sim_pwtr_debias_hetu"},
          ],
          function_name = "CalcDebiasXtr",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_fountain_rank_debias_pftr_score == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_rank_debias_pftr_default_debias_value", "as": "default_debias_value"},
            {"name": "fountain_rank_debias_pftr_debias_value_str", "as": "debias_value_str"},
            {"name": "fountain_rank_debias_pftr_debias_version", "as": "debias_version"},
          ],
          import_item_attr = [
            {"name": "fullrank_sim_pftr", "as": "xtr_input"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_list_input"},
            {"name": "duration_ms", "as": "duration_ms_input"},
          ],
          export_item_attr = [
            {"name": "xtr_output", "as": "fullrank_sim_pftr_debias_hetu"},
          ],
          function_name = "CalcDebiasXtr",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_fountain_rank_debias_pcmtr_score == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_rank_debias_pcmtr_default_debias_value", "as": "default_debias_value"},
            {"name": "fountain_rank_debias_pcmtr_debias_value_str", "as": "debias_value_str"},
            {"name": "fountain_rank_debias_pcmtr_debias_version", "as": "debias_version"},
          ],
          import_item_attr = [
            {"name": "fullrank_sim_pcmtr", "as": "xtr_input"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_list_input"},
            {"name": "duration_ms", "as": "duration_ms_input"},
          ],
          export_item_attr = [
            {"name": "xtr_output", "as": "fullrank_sim_pcmtr_debias_hetu"},
          ],
          function_name = "CalcDebiasXtr",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_fountain_rank_debias_pptr_score == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_rank_debias_pptr_default_debias_value", "as": "default_debias_value"},
            {"name": "fountain_rank_debias_pptr_debias_value_str", "as": "debias_value_str"},
            {"name": "fountain_rank_debias_pptr_debias_version", "as": "debias_version"},
          ],
          import_item_attr = [
            {"name": "fullrank_sim_pptr", "as": "xtr_input"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_list_input"},
            {"name": "duration_ms", "as": "duration_ms_input"},
          ],
          export_item_attr = [
            {"name": "xtr_output", "as": "fullrank_sim_pptr_debias_hetu"},
          ],
          function_name = "CalcDebiasXtr",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_fountain_rank_debias_pfintr_score == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_rank_debias_pfintr_default_debias_value", "as": "default_debias_value"},
            {"name": "fountain_rank_debias_pfintr_debias_value_str", "as": "debias_value_str"},
            {"name": "fountain_rank_debias_pfintr_debias_version", "as": "debias_version"},
          ],
          import_item_attr = [
            {"name": "fullrank_sim_pfintr", "as": "xtr_input"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_list_input"},
            {"name": "duration_ms", "as": "duration_ms_input"},
          ],
          export_item_attr = [
            {"name": "xtr_output", "as": "fullrank_sim_pfintr_debias"},
          ],
          function_name = "CalcDebiasXtr",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()
    return self

  def calc_fr_cdf_mapping(self):
    self \
      .if_("skip_fountain_fullrank_fr_cdf_mapping == 0") \
        .explore_memory_data_enrich(
          data_key = "{{fountain_fullrank_pfr_debias_map}}",
          data_type = "string_double_vector_map",
          save_data_ptr_to_attr = "emp_fr_debias_map_ptr",
        ) \
        .explore_trans_fintr_enricher(
          enable_transfer_sigmoid = "{{fountain_fullrank_enable_transfer_fintr_sigmoid}}",
          get_fintr_quantile_mode = "{{fountain_fullrank_get_fintr_quantile_mode}}",
          fintr_debias_map_attr = "emp_fr_debias_map_ptr",
          fintr_redis_key_prefix = "{{fountain_fullrank_fintr_redis_key_prefix}}",
          fintr_short_photo_cluster_dist = "{{fountain_fullrank_fintr_photo_cluster_dist}}",
          duration_ms_attr = "duration_ms",
          fintr_attr = "fullrank_ltr_v4_fountain_finish_rate",
          max_fintr_limit = "{{fountain_fullrank_fintr_max_cluster_limit}}",
          max_dura_limit = "{{fountain_fullrank_fintr_max_dura_seconds}}",
          fintr_dist_reciprocal = "{{fountain_fullrank_fintr_dist_reciprocal}}",
          save_fintr_quantile_to_attr = "fullrank_dura_cdf_pfr"
        ) \
      .end_()
    return self

  def ensemble_filter(self, traceback_name: str):
    self \
      .count_reco_result(
        save_count_to="fountain_fullrank_result_count_before_stage1"
      ) \
      .explore_ensemble_filter_score_enricher(
        queues = fullrank_ensemble_filter_queues,
        filter_function = "{{fountain_ensemble_filter_function}}",
        score_with_rank = "{{fountain_ensemble_filter_score_with_rank}}",
        save_score_to_attr = "fullrank_ensemble_filter_score",
      ) \
      .sort(
        score_from_attr = "fullrank_ensemble_filter_score",
        stable_sort = True,
        desc = False
      ) \
      ._dump_attr_to_kafka( # filter 截断之前, 将全部item的重要 item attr 落盘
        stage_name = "fr_s1_score",
        dump_item_attr_list = [
          "fullrank_sim_pcmtr",
          "fullrank_sim_pfintr",
          "fullrank_sim_pevtr",
          "fullrank_sim_pltr",
          "fullrank_sim_pwtr",
          "fullrank_detail_new_pevtr_v2",
          "fullrank_sim_pvtr",
          "fullrank_ensemble_filter_score",
        ]
      ) \
      .gen_common_attr_by_lua(
        attr_map={
          "fullrank_stage1_limit_size": "math.floor(fountain_fullrank_result_count_before_stage1 * (1 - fountain_ensemble_filter_coeff))",
        }
      ) \
      .if_("fountain_enable_cascade_distill_full_link_sample == 1") \
        .__rank_stage1_full_link_sample_log() \
      .end_if_() \
      .limit(
        name = traceback_name,
        traceback = True,
        size = "{{fullrank_stage1_limit_size}}"
      )
    return self
  
  def fetch_sim_gsu_feature(self):
    self \
      .if_("enable_fountain_fr_fetch_sim_gsu_feature == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .get_remote_embedding(
          kess_service = "grpc_MMUHetuContentEmbeddingV2",
          shard_num = 4,
          query_source_type = 'item_key',
          save_to_common_attr = False,
          output_attr_name = "target_photo_embedding",
          id_converter = dict(type_name="kuibaEmbeddingIdConverter"),
          client_side_shard = True,
          timeout_ms = 50
        ) \
        .gsu_retriever_with_colossus_resp_v2(
          colossus_resp_attr = "colossus_resp_v2",
          save_author_id_to_attr = "gsu_aid",
          save_duration_to_attr = "gsu_duration",
          save_play_time_to_attr = "gsu_play_for_sim",
          save_tag_to_attr = "gsu_tag",
          save_timestamp_to_attr = "gsu_time",
          save_channel_to_attr = "gsu_page_type",
          save_label_to_attr = "gsu_action_label",
          save_result_to_common_attr = "gsu_colossus_pid_v2",
          filter_future_attr = True,
          parse_from_pb = False
        ) \
        .enrich_attr_by_light_function(
          import_item_attr = [
            {"name": "target_photo_embedding", "as": "emb"}
          ],
          export_item_attr = [
            {"name": "is_valid", "as": "is_target_embedding_hit"}
          ],
          function_name = "IsValidEmb",
          class_name = "ExploreLightFunctionSetV2"
        ) \
        .pack_item_attr(
          item_source = {"reco_results": True},
          mappings = [
            {"from_item_attr": "target_photo_embedding", "to_common_attr": "gsu_all_target_embedding"}
          ]
        ) \
        .fetch_tower_topn_dot_product_pxtr(
          user_embedding_attr = "gsu_all_target_embedding",
          item_list_from_attr = "gsu_colossus_pid_v2",
          use_item_key_as_embed_key = True,
          predict_labels = ["dp"],
          kess_service = "grpc_ExporeSimEmbDPCalcServer",
          shards = 4,
          timeout_ms = 50,
          sub_req_num_in_shard = 1,
          server_request_type = "calc_topn_dot_product",
          req_common_embedding_attr = "req_item_emb",
          return_pxtr_value_attr = "distance",
          sorted_item_idx_attr = "gsu_filter_sorted_item_index",
          pxtr_type = 1,
          emb_dim = 64,
          output_type = 4,
          return_sorted_item_ids_attr = "sorted_item_ids_vec",
          top_n = 50
        ) \
        .gsu_with_index(
          target_item = {"is_target_embedding_hit": 1},
          colossus_pid_attr = "gsu_colossus_pid_v2",
          author_id_attr = "gsu_aid",
          tag_attr = "gsu_tag",
          play_time_attr = "gsu_play_for_sim",
          duration_attr = "gsu_duration",
          timestamp_attr = "gsu_time",
          channel_attr = "gsu_page_type",
          label_attr = "gsu_action_label",
          output_sign_attr = "gsu_signs",
          output_slot_attr = "gsu_slots",
          sorted_item_idx_attr = "gsu_filter_sorted_item_index",
          slots_id = [701, 702, 703, 704, 705, 706, 707],
          mio_slots_id = [701, 702, 703, 704, 705, 706, 707],
          top_n = 50
        ) \
        .gsu_with_index(
          target_item = {"is_target_embedding_hit": 1},
          colossus_pid_attr = "gsu_colossus_pid_v2",
          author_id_attr = "gsu_aid",
          tag_attr = "gsu_tag",
          play_time_attr = "gsu_play_for_sim",
          duration_attr = "gsu_duration",
          timestamp_attr = "gsu_time",
          channel_attr = "gsu_page_type",
          label_attr = "gsu_action_label",
          output_sign_attr = "gsu_bias_signs",
          output_slot_attr = "gsu_bias_slots",
          sorted_item_idx_attr = "gsu_filter_sorted_item_index",
          slots_id = [711, 712, 713, 714, 715, 716, 717],
          mio_slots_id = [711, 712, 713, 714, 715, 716, 717],
          top_n = 50
        ) \
      .end_()
    return self

  def fullrank_get_hetu_behavior_score(self):
    """
    行为期望 & 类目多样性
    """
    self \
      .pack_item_attr(
        item_source = {
          "reco_results": True,
        },
        mappings = [
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_pevtr",
            "to_common_attr": "fullrank_pevtr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_plvtr",
            "to_common_attr": "fullrank_plvtr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_ltr_v4_fountain_next",
            "to_common_attr": "fullrank_next_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_pvtr",
            "to_common_attr": "fullrank_pvtr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_pfintr",
            "to_common_attr": "fullrank_pfintr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_pltr",
            "to_common_attr": "fullrank_pltr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_pwtr",
            "to_common_attr": "fullrank_pwtr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_pepstr",
            "to_common_attr": "fullrank_pepstr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_pcmtr",
            "to_common_attr": "fullrank_pcmtr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_pcltr",
            "to_common_attr": "fullrank_pcltr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_pftr",
            "to_common_attr": "fullrank_pftr_avg"
          },
        ]
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "colossus_hetu_distribution_hetu_stat",
          {"name": "fountain_fullrank_behaviour_hetu_diversity_hetu_coef_beta", "as": "hetu_coef_beta"},
          {"name": "fountain_fullrank_behaviour_hetu_diversity_enable_unknown_hetu_adjust", "as": "enable_unknown_hetu_adjust"},
          {"name": "fullrank_pevtr_avg", "as": "pctr_avg"},
          {"name": "fullrank_plvtr_avg", "as": "plvtr_avg"},
          {"name": "fullrank_next_avg", "as": "pslide_avg"},
          {"name": "fullrank_pvtr_avg", "as": "pwatch_time_avg"},
          {"name": "fullrank_pfintr_avg", "as": "pwtd_avg"},
          {"name": "fullrank_pltr_avg", "as": "pltr_avg"},
          {"name": "fullrank_pwtr_avg", "as": "pwtr_avg"},
          {"name": "fullrank_pepstr_avg", "as": "pepstr_avg"},
          {"name": "fullrank_pcmtr_avg", "as": "pcmtr_avg"},
          {"name": "fullrank_pcltr_avg", "as": "pcltr_avg"},
          {"name": "fullrank_pftr_avg", "as": "pftr_avg"},
          {"name": "fountain_fullrank_behaviour_hetu_diversity_pevtr_alpha", "as": "pctr_alpha"},
          {"name": "fountain_fullrank_behaviour_hetu_diversity_plvtr_alpha", "as": "plvtr_alpha"},
          {"name": "fountain_fullrank_behaviour_hetu_diversity_next_alpha", "as": "pslide_alpha"},
          {"name": "fountain_fullrank_behaviour_hetu_diversity_pvtr_alpha", "as": "pwatch_time_alpha"},
          {"name": "fountain_fullrank_behaviour_hetu_diversity_pfintr_alpha", "as": "pwtd_alpha"},
          {"name": "fountain_fullrank_behaviour_hetu_diversity_pltr_alpha", "as": "pltr_alpha"},
          {"name": "fountain_fullrank_behaviour_hetu_diversity_pwtr_alpha", "as": "pwtr_alpha"},
          {"name": "fountain_fullrank_behaviour_hetu_diversity_pepstr_alpha", "as": "pepstr_alpha"},
          {"name": "fountain_fullrank_behaviour_hetu_diversity_pcmtr_alpha", "as": "pcmtr_alpha"},
          {"name": "fountain_fullrank_behaviour_hetu_diversity_pcltr_alpha", "as": "pcltr_alpha"},
          {"name": "fountain_fullrank_behaviour_hetu_diversity_pftr_alpha", "as": "pftr_alpha"},
          {"name": "fountain_fullrank_behaviour_hetu_diversity_pevtr_beta", "as": "pctr_beta"},
          {"name": "fountain_fullrank_behaviour_hetu_diversity_plvtr_beta", "as": "plvtr_beta"},
          {"name": "fountain_fullrank_behaviour_hetu_diversity_next_beta", "as": "pslide_beta"},
          {"name": "fountain_fullrank_behaviour_hetu_diversity_pvtr_beta", "as": "pwatch_time_beta"},
          {"name": "fountain_fullrank_behaviour_hetu_diversity_pfintr_beta", "as": "pwtd_beta"},
          {"name": "fountain_fullrank_behaviour_hetu_diversity_pltr_beta", "as": "pltr_beta"},
          {"name": "fountain_fullrank_behaviour_hetu_diversity_pwtr_beta", "as": "pwtr_beta"},
          {"name": "fountain_fullrank_behaviour_hetu_diversity_pepstr_beta", "as": "pepstr_beta"},
          {"name": "fountain_fullrank_behaviour_hetu_diversity_pcmtr_beta", "as": "pcmtr_beta"},
          {"name": "fountain_fullrank_behaviour_hetu_diversity_pcltr_beta", "as": "pcltr_beta"},
          {"name": "fountain_fullrank_behaviour_hetu_diversity_pftr_beta", "as": "pftr_beta"},
        ],
        import_item_attr = [
          {"name": "fullrank_sim_pevtr", "as": "pctr"},
          {"name": "fullrank_sim_plvtr", "as": "plvtr"},
          {"name": "fullrank_ltr_v4_fountain_next", "as": "pslide"},
          {"name": "fullrank_sim_pvtr", "as": "pwatch_time"},
          {"name": "fullrank_sim_pfintr", "as": "pwtd"},
          {"name": "fullrank_sim_pltr", "as": "pltr"},
          {"name": "fullrank_sim_pwtr", "as": "pwtr"},
          {"name": "fullrank_sim_pepstr", "as": "pepstr"},
          {"name": "fullrank_sim_pcmtr", "as": "pcmtr"},
          {"name": "fullrank_sim_pcltr", "as": "pcltr"},
          {"name": "fullrank_sim_pftr", "as": "pftr"},
          {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_level_one_list"},
        ],
        export_item_attr = [
          {"name": "behaviour_hetu_diversity_boost_coeff", "as": "fullrank_hetu_behavior_score"}
        ],
        function_name = "GetBehaviourHetuDiversityBoostCoeff",
        class_name = "ExploreLightFunctionSetV2",
      ) 
    return self
  
  def fr_high_value_pic_boost(self):
    self \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_fr_hv_pic_boost_coef", "as": "boost_discount_coeff"},
        ],
        import_item_attr = [
          {"name": "fullrank_ensemble_score_after_adjust", "as": "score"},
        ],
        export_item_attr = [
          {"name": "score", "as": "fullrank_ensemble_score_after_adjust"},
        ],
        function_name = "BoostOrDiscountV2",
        class_name = "ExploreLightFunctionSetV2",
        target_item = { "high_value_pic_flag": 1 },
      )
    return self
  
  def fr_interact_similarity_score_boost(self):
    self \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "fountain_interact_similarity_score_boost_coef",
          "fountain_interact_similarity_score_threshold"
        ],
        import_item_attr = [
          "fullrank_ensemble_score_after_adjust",
          "fullrank_interact_similarity_score"
        ],
        export_item_attr = [
          "fullrank_ensemble_score_after_adjust"
        ],
        function_name = "BoostByInteractSimilarityScore",
        class_name = "ExploreLightFunctionSetV2",
      )
    return self
  
  def fullrank_hetu_level1_discount(self):
    self \
      .get_kconf_params(
        kconf_configs = [{
          "kconf_key": "reco.eyeshot.fountainLifeHetuDiscount",
          "value_type": "list_int64",
          "default_value": [9, 32],
          "export_common_attr": "fountain_life_discount_hetu",
        }]
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          {"name": "hetu_tag_level_info_v2__hetu_level_one", "as": "extract_hetu_tag_list"},
        ],
        export_item_attr = [
          {"name": "first_hetu_tag_id", "as": "hetu_level1"}
        ],
        function_name = "ExtractFirstHetuV2Tag",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_ranking_hetu_level1_discount_coef", "as": "boost_discount_coeff"},
        ],
        import_item_attr = [
          {"name": "fullrank_ensemble_score_after_adjust", "as": "score"},
        ],
        export_item_attr = [
          {"name": "score", "as": "fullrank_ensemble_score_after_adjust"},
        ],
        function_name = "BoostOrDiscountV2",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          "hetu_level1": "{{fountain_life_discount_hetu}}"
        },
      )
    return self

  def _dump_attr_to_kafka(self, stage_name : str, dump_item_attr_list : list):
    """
    dump item attr to kafka
    """
    dump_attr_to_kafka(self, stage_name, dump_item_attr_list)
    return self

  def audit_adjust_score(self):
    self \
      .if_("fountain_rank_enable_audit_hot_cover_level_adjust == 1") \
        .transform_item_attr(
          mappings = [
            {
              "check_attr_name": "audit_hot_cover_level",
              "check_attr_type": "int",
              "output_attr_name": "is_audit_hot_cover_level_discount",
              "output_attr_type": "int",
              "rules": [{
                "check_values": [2023746], # 劣质
                "output_value": 1
              }]
            },
            {
              "check_attr_name": "audit_hot_cover_level",
              "check_attr_type": "int",
              "output_attr_name": "is_audit_hot_cover_level_discount_soft",
              "output_attr_type": "int",
              "rules": [{
                "check_values": [2023742, 2023743, 2023744, 2023745], # 灰劣
                "output_value": 1
              }]
            }
          ]
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_rank_audit_hot_cover_level_discount_coeff", "as": "boost_discount_coeff"}
          ],
          import_item_attr = [
            {"name": "is_audit_hot_cover_level_discount", "as": "need_item_attr"},
            {"name": "fullrank_ensemble_score_after_adjust", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "fullrank_ensemble_score_after_adjust"},
          ],
          function_name = "BoostOrDiscount",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_rank_audit_hot_cover_level_discount_soft_coeff", "as": "boost_discount_coeff"}
          ],
          import_item_attr = [
            {"name": "is_audit_hot_cover_level_discount_soft", "as": "need_item_attr"},
            {"name": "fullrank_ensemble_score_after_adjust", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "fullrank_ensemble_score_after_adjust"},
          ],
          function_name = "BoostOrDiscount",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("fountain_rank_enable_impression_audit_adjust == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .transform_item_attr( # 观感审二级字段大于0才是已审核
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
            {"name": "fountain_rank_impression_audit_adjust_coeff_map_str", "as": "adjust_coeff_map_str_attr"}
          ],
          import_item_attr = [
            {"name": "content_safety_level_with_namespace__level_hot_online", "as": "audit_level_attr"},
            {"name": "fullrank_ensemble_score_after_adjust", "as": "ensemble_score_attr"},
            "upload_time"
          ],
          export_item_attr = [
            {"name": "ensemble_score_attr", "as": "fullrank_ensemble_score_after_adjust"},
          ],
          function_name = "AuditAdjustScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_impression_audit": 1,
          },
        ) \
      .end_() \
      .if_("fountain_rank_enable_high_hot_audit_adjust == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_rank_high_hot_audit_adjust_coeff_map_str", "as": "adjust_coeff_map_str_attr"}
          ],
          import_item_attr = [
            {"name": "audit_hot_high_tag_level", "as": "audit_level_attr"},
            {"name": "fullrank_ensemble_score_after_adjust", "as": "ensemble_score_attr"},
            "upload_time"
          ],
          export_item_attr = [
            {"name": "ensemble_score_attr", "as": "fullrank_ensemble_score_after_adjust"},
          ],
          function_name = "AuditAdjustScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("fountain_rank_enable_topk_audit_adjust == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_rank_topk_audit_adjust_coeff_map_str", "as": "adjust_coeff_map_str_attr"}
          ],
          import_item_attr = [
            {"name": "topk_audit_level", "as": "audit_level_attr"},
            {"name": "fullrank_ensemble_score_after_adjust", "as": "ensemble_score_attr"},
            "upload_time"
          ],
          export_item_attr = [
            {"name": "ensemble_score_attr", "as": "fullrank_ensemble_score_after_adjust"},
          ],
          function_name = "AuditAdjustScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()
    return self

  def calc_topk_mgs_expected_score(self):
    self \
      .explore_embedding_candidates_attr_enricher(
        trans_type = "fountain_candidates",
        enable_fix_low_hit_rate = True,
        enable_not_click = False,
        enable_play_stat = True,
        enable_hate = False,
        enable_explore_not_click = False,
        enable_source_photo = True,
        source_pid_attr = "featureSourcePId",
        session_history_max_size = "{{fountain_mgs_diversity_max_size}}",
        user_info_ptr_attr = "userInfoPb",
        export_common_attr = "topk_mgs_embedding_source_pids",
        check_point = "cascade",
      ) \
      .get_remote_embedding_lite(
        kess_service = "{{fountain_topk_mgs_expected_score_service}}",
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
        dim_size = "{{fountain_topk_mgs_expected_score_emb_dim_size}}",
        export_common_attr = "topk_mgs_pid_embedding_map",
      ) \
      .explore_diversity_update_enricher(
        user_info_ptr_attr = "userInfoPb",
        pid_embedding_common_attr = "topk_mgs_pid_embedding_map",
        export_item_attr = "topk_mgs_expected_score",
        history_feed_back_version = 3,
        dim_size = "{{fountain_topk_mgs_expected_score_emb_dim_size}}",
        expected_score_cand_size = "{{fountain_topk_mgs_expected_score_cand_num}}",
        max_interval_second = "{{fountain_topk_mgs_expected_score_max_interval_second}}",
        min_duration_threshold = "{{fountain_topk_mgs_expected_score_min_duration_threshold}}",
        dpp_diversity_mgs_topk = "{{fountain_topk_mgs_expected_score_topk_num}}",
        max_playtime_threshold = "{{fountain_topk_mgs_expected_score_max_playtime_threshold}}",
        enable_use_weight = "{{fountain_topk_mgs_expected_score_enable_use_weight}}",
        weight_version = "{{fountain_topk_mgs_expected_score_weight_version}}",
        ratio_scale = "{{fountain_topk_mgs_expected_score_ratio_scale}}",
        ratio_pow_weight = "{{fountain_topk_mgs_expected_score_ratio_pow_weight}}",
      ) \
      .explore_diversity_update_enricher(
        user_info_ptr_attr = "userInfoPb",
        pid_embedding_common_attr = "topk_mgs_pid_embedding_map",
        export_item_attr = "topk_mgs_neg_score",
        history_feed_back_version = 3,
        dim_size = "{{fountain_topk_mgs_neg_score_emb_dim_size}}",
        expected_score_cand_size = "{{fountain_topk_mgs_neg_score_cand_num}}",
        dpp_diversity_mgs_topk = "{{fountain_topk_mgs_neg_score_topk_num}}",
        max_playtime_threshold = "{{fountain_topk_mgs_neg_score_max_playtime_threshold}}",
        min_playtime_threshold = "{{fountain_topk_mgs_neg_score_min_playtime_threshold}}",
        enable_use_weight = "{{fountain_topk_mgs_neg_score_enable_use_weight}}",
        weight_version = "{{fountain_topk_mgs_neg_score_weight_version}}",
        ratio_scale = "{{fountain_topk_mgs_neg_score_ratio_scale}}",
        ratio_pow_weight = "{{fountain_topk_mgs_neg_score_ratio_pow_weight}}",
      )
    return self
  
  def calc_interact_similarity_score(self):
    self \
      .explore_diversity_update_enricher(
        user_info_ptr_attr = "userInfoPb",
        pid_embedding_common_attr = "topk_mgs_pid_embedding_map",
        export_item_attr = "fullrank_interact_similarity_score",
        history_feed_back_version = 4,
        dim_size = "{{fountain_interact_similarity_score_emb_dim_size}}",
        finish_rate_threshold = "{{fountain_interact_similarity_score_finish_rate_threshold}}",
        session_interact_history_max_size = "{{fountain_interact_similarity_score_history_max_size}}",
        enable_calc_matrix_similarity_score = "{{enable_calc_matrix_similarity_score}}",
      )
    return self
  
  def fountain_cal_quantile_relative_score(self):
    """
    Owner: xuwei09
    Date: 2024-06-03
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_pctr_quantile_threshold", "as": "quantile_threshold"},
        {"name": "fountain_pctr_quantile_k", "as": "quantile_k"},
        {"name": "fountain_pctr_quantile_alpha", "as": "quantile_alpha"},
      ],
      import_item_attr = [
        {"name": "fullrank_sim_pevtr", "as": "pxtr"},
      ],
      export_item_attr = [
        {"name": "quantile_score", "as": "pctr_quantile_score"},
      ],
      function_name = "QuantileScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_pltr_quantile_threshold", "as": "quantile_threshold"},
        {"name": "fountain_pltr_quantile_k", "as": "quantile_k"},
        {"name": "fountain_pltr_quantile_alpha", "as": "quantile_alpha"},
      ],
      import_item_attr = [
        {"name": "fullrank_sim_pltr", "as": "pxtr"},
      ],
      export_item_attr = [
        {"name": "quantile_score", "as": "pltr_quantile_score"},
      ],
      function_name = "QuantileScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_pwtr_quantile_threshold", "as": "quantile_threshold"},
        {"name": "fountain_pwtr_quantile_k", "as": "quantile_k"},
        {"name": "fountain_pwtr_quantile_alpha", "as": "quantile_alpha"},
      ],
      import_item_attr = [
        {"name": "fullrank_sim_pwtr", "as": "pxtr"},
      ],
      export_item_attr = [
        {"name": "quantile_score", "as": "pwtr_quantile_score"},
      ],
      function_name = "QuantileScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_pftr_quantile_threshold", "as": "quantile_threshold"},
        {"name": "fountain_pftr_quantile_k", "as": "quantile_k"},
        {"name": "fountain_pftr_quantile_alpha", "as": "quantile_alpha"},
      ],
      import_item_attr = [
        {"name": "fullrank_sim_pftr", "as": "pxtr"},
      ],
      export_item_attr = [
        {"name": "quantile_score", "as": "pftr_quantile_score"},
      ],
      function_name = "QuantileScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_pcmtr_quantile_threshold", "as": "quantile_threshold"},
        {"name": "fountain_pcmtr_quantile_k", "as": "quantile_k"},
        {"name": "fountain_pcmtr_quantile_alpha", "as": "quantile_alpha"},
      ],
      import_item_attr = [
        {"name": "fullrank_sim_pcmtr", "as": "pxtr"},
      ],
      export_item_attr = [
        {"name": "quantile_score", "as": "pcmtr_quantile_score"},
      ],
      function_name = "QuantileScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_pfintr_quantile_threshold", "as": "quantile_threshold"},
        {"name": "fountain_pfintr_quantile_k", "as": "quantile_k"},
        {"name": "fountain_pfintr_quantile_alpha", "as": "quantile_alpha"},
      ],
      import_item_attr = [
        {"name": "fullrank_sim_pfintr", "as": "pxtr"},
      ],
      export_item_attr = [
        {"name": "quantile_score", "as": "pfintr_quantile_score"},
      ],
      function_name = "QuantileScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_pevtr_v2_quantile_threshold", "as": "quantile_threshold"},
        {"name": "fountain_pevtr_v2_quantile_k", "as": "quantile_k"},
        {"name": "fountain_pevtr_v2_quantile_alpha", "as": "quantile_alpha"},
      ],
      import_item_attr = [
        {"name": "fullrank_detail_new_pevtr_v2", "as": "pxtr"},
      ],
      export_item_attr = [
        {"name": "quantile_score", "as": "pevtr_v2_quantile_score"},
      ],
      function_name = "QuantileScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_plvtr_quantile_threshold", "as": "quantile_threshold"},
        {"name": "fountain_plvtr_quantile_k", "as": "quantile_k"},
        {"name": "fountain_plvtr_quantile_alpha", "as": "quantile_alpha"},
      ],
      import_item_attr = [
        {"name": "fullrank_sim_plvtr", "as": "pxtr"},
      ],
      export_item_attr = [
        {"name": "quantile_score", "as": "plvtr_quantile_score"},
      ],
      function_name = "QuantileScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .calc_weighted_sum(
      fomula_version = 1,
      channels = [
        { "name": "pctr_quantile_score", "weight": "{{fountain_final_pctr_quantile_score_weight}}" },
        { "name": "pltr_quantile_score", "weight": "{{fountain_final_pltr_quantile_score_weight}}" },
        { "name": "pwtr_quantile_score", "weight": "{{fountain_final_pwtr_quantile_score_weight}}" },
        { "name": "pftr_quantile_score", "weight": "{{fountain_final_pftr_quantile_score_weight}}" },
        { "name": "pcmtr_quantile_score", "weight": "{{fountain_final_pcmtr_quantile_score_weight}}" },
        { "name": "pfintr_quantile_score", "weight": "{{fountain_final_pfintr_quantile_score_weight}}" },
        { "name": "pevtr_v2_quantile_score", "weight": "{{fountain_final_pevtr_v2_quantile_score_weight}}" },
        { "name": "plvtr_quantile_score", "weight": "{{fountain_final_plvtr_quantile_score_weight}}" },
      ],
      output_item_attr = "fullrank_quantile_relative_score",
    )
    return self

  def get_ranking_index(self):
    self \
    .get_index("fullrank_sim_pevtr", "pctr_index") \
    .get_index("fullrank_sim_plvtr", "plvtr_index") \
    .get_index("fullrank_sim_pvtr", "pvtr_index") \
    .get_index("fullrank_sim_pltr", "pltr_index") \
    .get_index("fullrank_sim_pftr", "pftr_index") \
    .get_index("fullrank_sim_pwtr", "pwtr_index") \
    .get_index("fullrank_sim_pepstr", "pesptr_index") \
    .get_index("fullrank_sim_psvr", "psvr_index") \
    .get_index("fullrank_sim_pfintr", "pfintr_index") \
    .get_index("fullrank_sim_pcmtr", "pcmtr_index") \
    .get_index("fullrank_sim_pcltr", "pcltr_index")

    return self

  def send_stage_sample(self, stage: str):
    self \
      .explore_feature_feedback_log_enricher(
        stage = stage,
        custom_common_attrs = [
          { "name": "uGender",  "as": "user_gender" },
          { "name": "featureAgeSegment",  "as": "user_age_segment" },
          { "name": "featureFollowCount",  "as": "user_follow_count" },
          { "name": "featureFansCount",  "as": "user_fans_count" },
          { "name": "featureCityId",  "as": "user_city_id" },
          { "name": "featureProvinceId",  "as": "user_province_id" },
        ],
        custom_item_attrs = [
          { "name": "featurePHetuTagLevel2",  "as": "hetu_l2" },
          { "name": "featurePHetuTagLevel3",  "as": "hetu_l3" },
          "duration_ms",
          { "name": "fullrank_sim_pevtr", "as": "pctr" },
          { "name": "fullrank_sim_pcmtr", "as": "pcmtr" },
          { "name": "fullrank_sim_pftr", "as": "pftr" },
          { "name": "fullrank_sim_pltr", "as": "pltr" },
          { "name": "fullrank_sim_plvtr", "as": "plvtr" },
          { "name": "fullrank_sim_pvtr", "as": "pvtr" },
          { "name": "fullrank_sim_pwtr", "as": "pwtr" },
          { "name": "fullrank_sim_pepstr", "as": "pepstr" },
          { "name": "fullrank_sim_pcltr", "as": "pcltr" },
          { "name": "fullrank_sim_psvr", "as": "psvr" },
          { "name": "fullrank_sim_pptr", "as": "pptr" },
          { "name": "fullrank_sim_pcmef", "as": "pcmef" },
          { "name": "fullrank_sim_phtr", "as": "phtr" },
          { "name": "fullrank_sim_pfintr", "as": "pwtd" },
          "pctr_index",
          "plvtr_index",
          "pvtr_index",
          "pltr_index",
          "pftr_index",
          "pwtr_index",
          "pesptr_index",
          "psvr_index",
          "pfintr_index",
          "pcmtr_index",
          "pcltr_index",
          "fullrank_ensemble_score",
        ],
        author_id_attr = "author__id",
        save_to_attr = "fr_stage_sample_str",
      ) \
      .send_with_kafka(
        common_attr = "fr_stage_sample_str",
        topic_name = "reco_explore_leaf_stage_sample",
      )

    return self

  def get_cascade_index(self):
    self \
    .get_index("cascade_pctr", "cascade_pctr_index") \
    .get_index("cascade_plvtr", "cascade_plvtr_index") \
    .get_index("cascade_pwatch_time", "cascade_pvtr_index") \
    .get_index("cascade_pltr", "cascade_pltr_index") \
    .get_index("cascade_pftr", "cascade_pftr_index") \
    .get_index("cascade_pwtr", "cascade_pwtr_index") \
    .get_index("cascade_pepstr", "cascade_pesptr_index") \
    .get_index("cascade_psvtr", "cascade_psvr_index") \

    return self

  def get_index(self, input_item_attr, output_item_attr):
    self \
    .sort(
      score_from_attr = input_item_attr
    ) \
    .enrich_attr_by_light_function(
      export_item_attr = [
        {"name": "item_attr_index", "as": output_item_attr}
      ],
      function_name = "SaveItemSeqAddOne",
      class_name = "ExploreLightFunctionSetV2",
    )

    return self

  def __rank_stage1_full_link_sample_log(self):
    self \
    .get_kconf_params(
      kconf_configs=[
        {
          "kconf_key": "reco.offline.fountainFulllinkDistillRankParam",
          "value_type": "json",
          "json_path": "rank_s1_full_link_distill_sample_num",
          "export_common_attr": "fountain_rank_s1_full_link_distill_sample_num"
        },
        {
          "kconf_key": "reco.offline.fountainFulllinkDistillRankParam",
          "value_type": "json",
          "json_path": "rank_s1_full_link_distill_sample_ratio",
          "export_common_attr": "fountain_rank_s1_full_link_distill_sample_ratio"
        },
      ]
    ) \
    .explore_full_link_context_sample_reco_log_enricher(
      sample_config = [
        {
          "sample_begin": "fullrank_stage1_limit_size",
          "sample_end": "fountain_fullrank_result_count_before_stage1",
          "sample_num": "fountain_rank_s1_full_link_distill_sample_num",
          "label_name": "rank_neg_stage1",
        },
      ],
      sample_ratio = "fountain_rank_s1_full_link_distill_sample_ratio",
      user_info_attr = "userInfoPb",
      load_attr = "fountain_full_link_reco_log_message",
      output_attr = "fountain_full_link_reco_log_message",
      cascade_pctr = "cascade_pctr",
      cascade_pltr = "cascade_pltr",
      cascade_pwtr = "cascade_pwtr",
      cascade_pftr = "cascade_pftr",
      cascade_pptr = "cascade_ptr",
      cascade_pcmtr = "cascade_pcmtr",
      cascade_plvtr = "cascade_plvtr",
      cascade_pvtr = "cascade_pwatch_time",
      pctr = "fullrank_sim_pevtr",
      pltr = "fullrank_sim_pltr",
      pwtr = "fullrank_sim_pwtr",
      pftr = "fullrank_sim_pftr",
      pptr = "fullrank_sim_pptr",
      pcmtr = "fullrank_sim_pcmtr",
      plvtr = "fullrank_sim_plvtr",
      pvtr = "fullrank_sim_pvtr",
    ) \
    
    return self

  def _refinement_boost_personified_author(self):
    """
    Module: full_rank_v44_flow
    功能: 细分用户和视频维度，精细化对人格化账号提权
    Owner: xubaoquan
    Date: 2023-07-19
    :return:
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "basic_info_age_segment_v2", "as": "basic_info_age_segment_v2"},
        {"name": "basic_info_gender_v2", "as": "basic_info_gender_v2"},
        {"name": "explore_personifed_author_boost_ptr", "as": "boost_map_ptr"},
        {"name": "refinement_boost_personified_author_redis_prefix", "as": "redis_prefix"},
        {"name": "fountain_rank_refinement_boost_personified_author_power_weight", "as": "power_weight"},
      ],
      import_item_attr = [
        {"name": "author__gender", "as": "author__gender"},
        {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_level_one_list"},
        {"name": "fullrank_ensemble_score_after_adjust", "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "fullrank_ensemble_score_after_adjust"},
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
    Module: full_rank_v44_flow
    功能: 将精排尾部结果写入redis进行过滤
    Owner: liuhao07
    Date: 2023-08-04
    :return:
    """
    self.pack_item_attr(
      item_source = {
        "reco_results": True
      },
      mappings = [{
        "from_item_attr": "photo_id",
        "to_common_attr": "rank_pos_photo_id_list",
        "aggregator": "concat"
      }],
      range_end = "{{fountain_rank_neg_photo_index}}"
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "rank_candidate_photo_id_list", "as": "universal_set_list"},
        {"name": "rank_pos_photo_id_list", "as": "sub_set_list"}
      ],
      export_common_attr = [
        {"name": "difference_list", "as": "rank_neg_photo_id_list"}
      ],
      function_name = "GetDifferenceSet",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .pack_common_attr(
      input_common_attrs = ["rank_neg_photo_id_list", "rank_neg_photo_id_filter_list"],
      output_common_attr = "rank_neg_photo_id_list",
      deduplicate = True,
      limit_num = "{{fountain_rank_neg_photo_size}}",
    ) \
    .write_to_redis(
      kcc_cluster = "recoExploreNegPhoto",
      timeout = 10,
      expire_second = "{{fountain_rank_neg_photo_redis_expire_seconds}}",
      key_prefix = "{{fountain_rank_neg_photo_key_prefix}}",
      key = "{{_DEVICE_ID_}}",
      value = "{{rank_neg_photo_id_list}}"
    )
    return self

  def write_rank_pos_result_to_redis(self):
    """
    Module: full_rank_v44_flow
    功能: 将精排头部结果写入redis进行召回
    Owner: liuhao07
    Date: 2023-09-04
    :return:
    """
    self.pack_item_attr(
      item_source = {
        "reco_results": True
      },
      mappings = [{
        "from_item_attr": "photo_id",
        "to_common_attr": "rank_pos_photo_id_list",
        "aggregator": "concat"
      }],
      range_end = "{{fountain_rank_pos_photo_end_index}}"
    ) \
    .pack_common_attr(
      input_common_attrs = [
        "rank_pos_photo_id_list",
        "fountain_rank_pos_photo_id_retrieval_list"
      ],
      output_common_attr = "rank_pos_photo_id_list",
      deduplicate = True,
      limit_num = "{{fountain_rank_pos_photo_size}}",
    ) \
    .write_to_redis(
      kcc_cluster = "recoExploreNegPhoto",
      timeout = 10,
      expire_second = "{{fountain_rank_pos_photo_redis_expire_seconds}}",
      key_prefix = "{{fountain_rank_pos_photo_key_prefix}}",
      key = "{{_DEVICE_ID_}}",
      value = "{{rank_pos_photo_id_list}}"
    )
    return self

  def high_photo_count_author_adjust(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "high_photo_count_author_map_ptr",
        {"name": "fountain_rank_high_photo_count_author_photo_coeff", "as": "boost_discount_coeff"},
        {"name": "fountain_rank_high_photo_count_author_post_num_base", "as": "post_num_base"},
      ],
      import_item_attr = [
        "author__id",
        {"name": "fullrank_ensemble_score_after_adjust", "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "fullrank_ensemble_score_after_adjust"},
      ],
      function_name = "HighPhotoCountAuthorPhotoAdjust",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def high_photo_count_author_adjust_v2(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "high_upload_photo_author_map_ptr",
        {"name": "fountain_rank_high_photo_count_author_photo_coeff", "as": "boost_discount_coeff"},
        {"name": "fountain_rank_high_photo_count_author_pos_neg_ratio_coeff", "as": "pos_neg_ratio_coeff"},
      ],
      import_item_attr = [
        "author__id",
        {"name": "fullrank_ensemble_score_after_adjust", "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "fullrank_ensemble_score_after_adjust"},
      ],
      function_name = "HighPhotoCountAuthorPhotoAdjustV2",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def _fullrank_htr_weight_adjust_by_uv_htr(self):
    self \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_ensemble_power_weight_fullrank_phtr_in_order_score", "as": "xtr_weight"},
          {"name": "fountain_recent_hate_count", "as": "user_vv"},
          {"name": "fountain_fr_htr_weight_adjust_alpha", "as": "alpha"},
          {"name": "fountain_fr_htr_weight_adjust_beta", "as": "beta"},
          {"name": "fountain_fr_htr_weight_adjust_omega", "as": "omega"},
          {"name": "fountain_fr_htr_weight_adjust_max", "as": "coeff_max"},
          {"name": "fountain_fr_htr_weight_adjust_min", "as": "coeff_min"},
        ],
        export_common_attr = [
          {"name": "xtr_weight", "as": "fountain_ensemble_power_weight_fullrank_phtr_in_order_score"},
        ],
        function_name = "AdjustWeightByUserVv",
        class_name = "ExploreLightFunctionSetV2",
      )
    return self
  
  def share_pull_ftr_weight_adjust_coef(self):
    self \
      .split_string(
        input_common_attr = "fountain_user_ftr_weight_adjust_score_align_list",
        output_common_attr = "user_ftr_weight_adjust_score_align_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_double = True,
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "uShareidCntKV", "as": "shareid_cnt"},
          {"name": "uOpenShareidCntKV", "as": "open_shareid_cnt"},
          {"name": "uOpenDeviceCntKV", "as": "open_device_cnt"},
          {"name": "uPullNumKV", "as": "pull_num"},
          {"name": "uShareBringNewDeviceNumKV", "as": "share_bring_new_device_num"},
          {"name": "uAttributionPerShareKV", "as": "attribution_per_share"},
          {"name": "user_ftr_weight_adjust_score_align_list", "as": "score_align_list"},
          {"name": "fountain_user_ftr_weight_adjust_upper", "as": "upper"},
          {"name": "fountain_user_ftr_weight_adjust_lower", "as": "lower"},
          {"name": "fountain_user_ftr_weight_adjust_score_avg", "as": "score_avg"},
        ],
        export_common_attr = [
          {"name": "coef", "as": "share_pull_ftr_adjust_coef"},
        ],
        function_name = "UserFtrWeightAdjustCoef",
        class_name = "ExploreLightFunctionSetV2",
      )
    return self

  def cal_share_pull_ftr_full_rank(self):
    self \
      .gen_common_attr_by_lua(
        attr_map={
          "fountain_ensemble_weight_forward_score": "fountain_ensemble_weight_forward_score * share_pull_ftr_adjust_coef",
        }
      )
    return self

  def fr_s2_collection_type_boost(self):
    """
    内流精排 s2 合集作品提权
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_fr_collection_type_boost_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "fullrank_ensemble_score_after_adjust", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "fullrank_ensemble_score_after_adjust"},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = { "is_collection": 1 }
    )
    return self

  def fr_pos_neg_ratio_boost(self):
    self \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_fr_like_hate_ratio_boost_alpha", "as": "like_hate_ratio_alpha"},
        {"name": "fountain_fr_like_hate_ratio_boost_weight", "as": "like_hate_ratio_weight"},
        {"name": "fountain_fr_long_short_view_ratio_boost_alpha", "as": "long_short_view_ratio_alpha"},
        {"name": "fountain_fr_long_short_view_ratio_boost_weight", "as": "long_short_view_ratio_weight"},
      ],
      import_item_attr = [
        {"name": "fullrank_sim_pltr", "as": "pltr_attr"},
        {"name": "fullrank_sim_phtr", "as": "phtr_attr"},
        {"name": "fullrank_sim_plvtr", "as": "plvtr_attr"},
        {"name": "fullrank_sim_psvr", "as": "psvtr_attr"},
      ],
      export_item_attr = [
        {"name": "boost_coeff", "as": "fountain_fr_pos_neg_ratio_boost_coeff"},
      ],
      function_name = "CalcPosNegRatioBoostCoeff",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "fountain_fr_pos_neg_ratio_boost_coeff", "as": "boost_discount_coeff"},
        {"name": "fullrank_ensemble_score_after_adjust", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "fullrank_ensemble_score_after_adjust"},
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def fr_watch_time_boost(self):
    self \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_fr_watch_time_boost_alpha", "as": "alpha"},
        {"name": "fountain_fr_watch_time_boost_upper_bound", "as": "upper_bound"},
      ],
      import_item_attr = [
        {"name": "fullrank_sim_pfintr", "as": "pwatch_time_attr"},
      ],
      export_item_attr = [
        {"name": "boost_coeff", "as": "fr_watch_time_boost_coeff"},
      ],
      function_name = "CalcWatchTimeBoostCoeff",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "fr_watch_time_boost_coeff", "as": "boost_discount_coeff"},
        {"name": "fullrank_ensemble_score_after_adjust", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "fullrank_ensemble_score_after_adjust"},
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def bid_follow_boost(self):
    self \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "friendAids", "as": "attr_list"},
      ],
      import_item_attr = [
        {"name": "author__id", "as": "attr"},
      ],
      export_item_attr = [
        {"name": "is_in_set", "as": "is_bid_follow_author"},
      ],
      function_name = "AttrIsInSet",
      class_name = "ExploreLightFunctionSetV2"
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_fullrank_bid_follow_boost_weight", "as": "boost_weight"},
      ],
      import_item_attr = [
        {"name": "fullrank_ensemble_score_after_adjust", "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "fullrank_ensemble_score_after_adjust"},
      ],
      function_name = "EnsembleScoreBoost",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {"is_bid_follow_author": 1}
    )
    return self

  def high_share_boost(self):
    self \
    .enrich_attr_by_light_function(
      import_item_attr = [
        "if_title_share",
        "upload_time"
      ],
      export_item_attr = [
        {"name": "is_high_share", "as": "is_high_share_photo"}
      ],
      function_name = "ItemIsHighShare",
      class_name = "ExploreLightFunctionSetV2"
    ) \
    .if_("bid_follow_num ~= 0 and (user_msg_cnt_ssm_today + user_msg_cnt_gsm_today <= 0)") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_fullrank_high_share_boost_weight", "as": "boost_weight"},
        ],
        import_item_attr = [
          {"name": "fullrank_ensemble_score_after_adjust", "as": "ensemble_score"},
        ],
        export_item_attr = [
          {"name": "ensemble_score", "as": "fullrank_ensemble_score_after_adjust"},
        ],
        function_name = "EnsembleScoreBoost",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {"is_high_share_photo": 1}
      ) \
    .end_() \
    
    return self

  def rank_marketing_compensation_adjust(self):
    self \
    .if_("enable_fountain_rank_calc_marketing_compensation_coeff == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_rank_marketing_compensation_adjust_ctr_weight", "as": "ctr_weight"},
          {"name": "fountain_rank_marketing_compensation_adjust_watchtime_weight", "as": "watchtime_weight"},
          {"name": "fountain_rank_marketing_compensation_adjust_score_base", "as": "score_base"},
          {"name": "fountain_rank_marketing_compensation_adjust_adjust_version", "as": "adjust_version"},
          {"name": "fountain_rank_marketing_compensation_adjust_score_base_ratio", "as": "score_base_ratio"},
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
        {"name": "fountain_rank_marketing_compensation_adjust_scale_factor", "as": "scale_factor"},
        {"name": "fountain_rank_marketing_compensation_adjust_base_coeff", "as": "base_coeff"},
      ],
      import_item_attr = [
        {"name": "fullrank_ensemble_score_after_adjust", "as": "old_coeff"},
        {"name": "rank_marketing_compensation_coeff", "as": "reward_coeff"},
      ],
      export_item_attr = [
        {"name": "new_coeff", "as": "fullrank_ensemble_score_after_adjust"},
      ],
      function_name = "MarketingCompensationPhotoAdjust",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {"is_marketing_compensation_photo": 1}
    )
    return self

  def rank_protogenetic_advertise_adjust(self):
    self \
    .enrich_attr_by_light_function(
      import_common_attr = [
        "fountain_protogenetic_advertise_type_list_str"
      ],
      import_item_attr = [
        "data_set_tags_bit"
      ],
      export_item_attr = [
        "is_protogenetic_advertise_photo"
      ],
      function_name = "IsProtogeneticAdvertisePhoto",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_rank_protogenetic_advertis_discount_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "fullrank_ensemble_score_after_adjust", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "fullrank_ensemble_score_after_adjust"}
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "is_protogenetic_advertise_photo" : 1
      }
    )
    return self

  def calc_plc_rank_model_attr_score(self):
    self \
    .if_("enable_fountain_plc_rank_model_predict == 1") \
      .explore_memory_data_enrich(
        data_key = "plc_frac_score",
        data_type = "string_double_vector_map",
        save_data_ptr_to_attr = "plc_frac_score_map_ptr",
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          { "name": "is_plc_item", "as": "is_plc_item" },
          { "name": "fountain_plc_ctr", "as": "plc_ctr" },
          { "name": "plc_business_type", "as" : "plc_business_type"},
        ],
        import_common_attr = [
          { "name": "plc_frac_score_map_ptr", "as": "plc_frac_score_map_ptr" },
          { "name": "fountain_plc_business_type_predict_str", "as": "plc_business_type_predict_str"},
          { "name": "fountain_default_plc_ctr_score", "as": "default_plc_ctr_score"},
          { "name": "fountain_disable_plc_mode", "as": "disable_plc_mode"},
          { "name": "fountain_plc_ctr_score_mode", "as": "plc_ctr_score_mode"},
          { "name": "fountain_disable_plc_random_ratio", "as": "disable_plc_random_ratio"},
          { "name": "fountain_disable_plc_plc_ctr_thresh", "as": "disable_plc_plc_ctr_thresh"},
          { "name": "fountain_plc_frac_score_model_name", "as": "plc_frac_score_model_name"},
          { "name": "fountain_plc_frac_score_weights_alpha", "as": "frac_alpha"},
          { "name": "fountain_plc_frac_score_weights_beta", "as": "frac_beta"},
          { "name": "fountain_plc_frac_score_weights_bias", "as": "frac_bias"},
          { "name": "fountain_plc_frac_score_boost_quantile", "as": "plc_frac_score_boost_quantile"},
          { "name": "fountain_plc_frac_score_boost_biz_quantile_str", "as": "plc_frac_score_boost_biz_quantile_str"},
          { "name": "fountain_plc_frac_score_disable_quantile", "as": "plc_frac_score_disable_quantile"},
          { "name": "fountain_plc_frac_score_disable_biz_quantile_str", "as": "plc_frac_score_disable_biz_quantile_str"},
        ],
        export_item_attr = [
          { "name": "plc_ctr_score", "as": "fountain_plc_ctr_score" },
          { "name": "disable_plc", "as": "disable_plc" },
        ],
        function_name = "CalPlcScore",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_()
    return self

  def llm_negative_photo_adjust(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_rank_llm_negative_photo_adjust_tag_coeff_map_str", "as": "tag_coeff_map_str"},
      ],
      import_item_attr = [
        "hetu_tag_level_info_v2__hetu_tag",
        "explore_stat__click_count",
        "explore_stat__report_count",
        "fountain_stats__real_show_count",
        "fountain_stats__report_count",
        {"name": "fullrank_ensemble_score_after_adjust", "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "fullrank_ensemble_score_after_adjust"}
      ],
      function_name = "LlmNegativePhotoAdjust",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def fr_llm_negative_photo_personal_adjust(self):
    self.calc_by_formula1(
      kconf_key = "formula.scenarioKey55.FrFountainLlmNeagtivePhotoDeboost",
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
        {"name": "final_score", "as": "final_llm_personal_score"}
      ],
      abtest_biz_name = "KUAISHOU_APPS"
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "fullrank_ensemble_score_after_adjust", "as": "score"},
        {"name": "final_llm_personal_score", "as": "boost_discount_coeff"},
      ],
      export_item_attr = [
        {"name": "score", "as": "fullrank_ensemble_score_after_adjust"},
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2"
    )
    return self

  def disable_forward_social_queue(self):
    self \
    .if_("fountain_fullrank_disable_forward_social_queue_condition == 1 and (bid_follow_num == 0 or (u_inside_share_active_degree_detail_code > 3))") \
      .gen_common_attr_by_lua(
        attr_map={
          "fountain_ensemble_weight_forward_score_social": "0.0",
          "fountain_fullrank_ensemble_pftr_hyperbolic_raw_pow_weight_attr_social": "0.0"
        }
      ) \
    .end_() \
    .if_("fountain_fullrank_disable_forward_social_queue_condition == 2 and (bid_follow_num == 0 or (u_share_num_30d == 0 and u_message_active_degree ~= 5 and u_message_active_degree ~= 6))") \
      .gen_common_attr_by_lua(
        attr_map={
          "fountain_ensemble_weight_forward_score_social": "0.0",
          "fountain_fullrank_ensemble_pftr_hyperbolic_raw_pow_weight_attr_social": "0.0"
        }
      ) \
    .end_() \
    .if_("fountain_fullrank_disable_forward_social_queue_condition == 3 and (bid_follow_num == 0 or (user_msg_cnt_ssm_today + user_msg_cnt_gsm_today > 0))") \
      .gen_common_attr_by_lua(
        attr_map={
          "fountain_ensemble_weight_forward_score_social": "0.0",
          "fountain_fullrank_ensemble_pftr_hyperbolic_raw_pow_weight_attr_social": "0.0"
        }
      ) \
    .end_()
    return self

  def user_intrest_adjust(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "output_intrest_key_list", "as": "intrest_key_list"},
        {"name": "output_intrest_value_list", "as": "intrest_value_list"},
        {"name": "fountain_fr_user_intrest_adjust_boost_coef", "as": "boost_coef"},
        {"name": "fountain_fr_user_intrest_adjust_discount_coef", "as": "discount_coef"},
        {"name": "fountain_enable_hetu1_user_intrest_adjust", "as": "enable_hetu1"},
      ],
      import_item_attr = [
        {"name": "fullrank_ensemble_score_after_adjust", "as": "input_score"},
        "hetu_tag_level_info__hetu_level_two",
        "hetu_tag_level_info__hetu_level_one",
      ],
      export_item_attr = [
        {"name": "output_score", "as": "fullrank_ensemble_score_after_adjust"}
      ],
      function_name = "IntrestAdjustScore",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def fr_bad_comment_pids_hetu_second_debias_score_adjust(self, score_name):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_fr_bad_comment_pids_discount_coef", "as": "boost_discount_coeff"},
        {"name": "bad_comment_pids_ptr", "as": "boost_set"},
      ],
      import_item_attr = [
        {"name": score_name, "as": "boost_score"},
      ],
      export_item_attr = [
        {"name": "boost_score", "as": score_name}
      ],
      function_name = "PidBoost",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def fr_sideinfo_retargeting_score_adjust(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "sidinfo_retargeting_score", "as": "boost_discount_coeff"},
        {"name": "fullrank_ensemble_score_after_adjust", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "fullrank_ensemble_score_after_adjust"}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def fr_marketing_compensation_photo_personal_adjust(self):
    self.calc_by_formula1(
      kconf_key = "formula.scenarioKey25.FrFountainMarketingPhotoDeboost",
      import_item_attr = [
        "fountain_ecology_positive_score"
      ],
      import_common_attr = [
        "colossus_user_info_fountain_positive_size"
      ],
      export_formula_value = [
        {"name": "final_score", "as": "final_fr_marketing_compensation_photo_score"}
      ],
      abtest_biz_name = "KUAISHOU_APPS",
      target_item = {"is_marketing_compensation_photo": 1}
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "final_fr_marketing_compensation_photo_score", "as": "boost_discount_coeff"},
        {"name": "fullrank_ensemble_score_after_adjust", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "fullrank_ensemble_score_after_adjust"},
      ],
      function_name = "BoostOrDiscountWithItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {"is_marketing_compensation_photo": 1}
    )
    return self

  def fr_consecutive_nonclick_tag_exit(self):
    self \
    .explore_colossus_nonclick_enricher(
      colossus_v2_attr_name = "colossus_resp_v2",
      save_consecutive_nonclick_exit_intensity_to_attr = "consecutive_nonclick_exit_intensity",
      hetu_attr = "hetu_tag_level_info__hetu_level_two",
      enable_only_fountain_nonclick = "{{enable_fountain_rank_only_fountain_nonclick}}",
      enable_handle_hetu_two_missing_item = "{{enable_fountain_rank_handle_hetu_two_missing_item}}",
      enable_time_aware_nonclick_weight = "{{enable_fountain_rank_time_aware_nonclick_weight}}",
      colossus_profile_recent_minutes = "{{consecutive_nonclick_colossus_profile_recent_minutes}}",
      positive_interact_nonclick_handle_mode = "{{fountain_rank_positive_interact_nonclick_handle_mode}}",
      hetu_two_missing_item_scaling_ratio = "{{fountain_rank_hetu_two_missing_item_scaling_ratio}}",
      time_aware_nonclick_weight_slope = "{{fountain_rank_time_aware_nonclick_weight_slope}}",
      time_aware_nonclick_weight_bias = "{{fountain_rank_time_aware_nonclick_weight_bias}}",
      short_view_play_time_threshold = "{{fountain_rank_short_view_play_time_threshold}}"
    ) \
    .calc_by_formula1(
      kconf_key = "formula.scenarioKey71.FrFountainNonclickDeboostV2",
      import_item_attr = [
        "consecutive_nonclick_exit_intensity"
      ],
      export_formula_value = [
        {"name": "final_score", "as": "final_fr_consecutive_nonclick_score"}
      ],
      abtest_biz_name = "KUAISHOU_APPS"
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "final_fr_consecutive_nonclick_score", "as": "boost_discount_coeff"},
        {"name": "fullrank_ensemble_score_after_adjust", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "fullrank_ensemble_score_after_adjust"},
      ],
      function_name = "BoostOrDiscountWithItemCoeff",
      class_name = "ExploreLightFunctionSetV2"
    )

    return self

  def fullrank_batch_similar_predict(self):
    self \
      .if_("enable_fountain_fullrank_batch_similar_predict == 1 and enable_fountain_cascade_batch_similar_model_predict == 0") \
        .set_attr_default_value(
          item_attrs = [
            {
              "name": "explore_fountain_fullrank_batch_similar_pc12h",
              "type": "int",
              "value": 6000
            }
          ],
        ) \
        .delegate_enrich(
          kess_service = "{{fountain_fullrank_dynamic_i2i_kess_service}}",
          recv_item_attrs=[
            {"name": "batch_similar_score_attr", "as": "batch_similar_score"},
          ],
          timeout_ms = 100,
          send_item_attrs=[
            {"name": "explore_fountain_fullrank_batch_similar_pc12h", "as": "vv_cnt_g"},
            {"name": "explore_fountain_fullrank_batch_similar_pc12h", "as": "vv_cnt_n"},
          ],
          send_common_attrs = [
            { "name": "fountain_retarget_interest_colossus_trigger_list", "as": "pid_list"},
          ],
          request_type="{{fountain_i2i_dynamic_request_type}}",
          partition_size = "{{fountain_i2i_dynamic_partition_size}}",
          use_packed_item_attr = True,
          infer_output_type = 2
        ) \
      .end_()
    
    return self

  def fountain_fullrank_dynamic_i2i_score(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "batch_similar_score", "as": "batch_score"},
        {"name": "sidinfo_retargeting_discount_score", "as": "retargeting_score"},
      ],
      import_common_attr = [
        {"name": "fountain_fr_dynamic_i2i_score_switch", "as": "cal_method"},
        {"name": "fountain_fr_dynamic_i2i_score_threshold", "as": "score_threshold"},
      ],
      export_item_attr = [
        {"name": "max_score", "as": "i2i_dynamic_score"},
      ],
      function_name = "CalI2IDynamicScore",
      class_name = "ExploreLightFunctionSetV2"
    )
    
    return self

  def fr_eco_living_good_author_boost(self):
    self.explore_memory_data_enrich(
      data_key = "ecommerce_good_author_show_case",
      data_type = "uint64_set",
      save_data_ptr_to_attr = "ecommerce_good_author_show_case_ptr"
    ) \
    .explore_memory_data_enrich(
      data_key = "ecommerce_good_author_e_commerce",
      data_type = "uint64_set",
      save_data_ptr_to_attr = "ecommerce_good_author_e_commerce_ptr"
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        "author__id",
        {"name": "fullrank_sim_pwtr", "as": "pwtr"},
        {"name": "fullrank_ensemble_score_after_adjust", "as": "old_score"}
      ],
      import_common_attr = [
        {"name": "ecommerce_good_author_show_case_ptr", "as": "live_aid_ptr"},
        {"name": "ecommerce_good_author_e_commerce_ptr", "as": "eco_aid_ptr"},
        {"name": "fountain_fr_eco_living_good_author_boost_raw_weight", "as": "raw_weight"},
        {"name": "fountain_fr_eco_living_good_author_boost_raw_pow_weight", "as": "raw_pow_weight"}
      ],
      export_item_attr = [
        {"name": "new_score", "as": "fullrank_ensemble_score_after_adjust"},
      ],
      function_name = "EcoLivingGoodAuthorBoost",
      class_name = "ExploreLightFunctionSetV2"
    )
    return self

  def calc_fountain_fullrank_iput_score(self):
    # IPUT 单位时间行为概率
    self \
      .item_attr_operation(
        item_attr_a="fullrank_ltr_v4_fountain_next",
        item_attr_b="fullrank_sim_pwtd_v2_playtime",
        operator="/",
        output_attr="fullrank_sim_next_iput"
      ) \
      .item_attr_operation(
        item_attr_a="fullrank_sim_like_score",
        item_attr_b="fullrank_sim_pwtd_v2_playtime",
        operator="/",
        output_attr="fullrank_sim_like_iput"
      ) \
      .item_attr_operation(
        item_attr_a="fullrank_action_once_interact_score",
        item_attr_b="fullrank_sim_pwtd_v2_playtime",
        operator="/",
        output_attr="fullrank_action_once_interact_iput"
      )
    return self

  def rank_stage1_interact_playtime_adjust(self):
    """
    s1阶段低互动人群提权时长队列
    """
    self \
    .if_("enable_fountain_rank_ensemble_filter_playtime_adjust == 1 and user_is_low_interact == 1") \
      .gen_common_attr_by_lua(
        attr_map={
          "fountain_ensemble_filter_weight_fullrank_pfintr": "fountain_ensemble_filter_weight_fullrank_pfintr * fountain_ensemble_filter_weight_fullrank_pfintr_coeff",
          "fountain_ensemble_filter_weight_fullrank_pvtr": "fountain_ensemble_filter_weight_fullrank_pvtr * fountain_ensemble_filter_weight_fullrank_pvtr_coeff",
          "fountain_ensemble_filter_weight_fullrank_pevtr": "fountain_ensemble_filter_weight_fullrank_pevtr * fountain_ensemble_filter_weight_fullrank_pevtr_coeff",
          "fountain_ensemble_filter_weight_fullrank_pevtr_v2": "fountain_ensemble_filter_weight_fullrank_pevtr_v2 * fountain_ensemble_filter_weight_fullrank_pevtr_v2_coeff"
        }
      ) \
    .end_if_()
    return self

  def fullrank_s2_interact_playtime_adjust(self):
    """
    s2阶段低互动人群提权时长队列
    """
    self \
    .if_("enable_fountain_fullrank_s2_playtime_adjust == 1 and user_is_low_interact == 1") \
      .gen_common_attr_by_lua(
        attr_map={
          "fountain_ensemble_power_weight_fullrank_pvtr_score": "fountain_ensemble_power_weight_fullrank_pvtr_score * fountain_ensemble_power_weight_fullrank_pvtr_score_coeff",
          "fountain_ensemble_power_weight_fullrank_pcpr_score": "fountain_ensemble_power_weight_fullrank_pcpr_score * fountain_ensemble_power_weight_fullrank_pcpr_score_coeff",
          "fountain_ensemble_power_weight_fullrank_pfintr_score": "fountain_ensemble_power_weight_fullrank_pfintr_score * fountain_ensemble_power_weight_fullrank_pfintr_score_coeff",
          "fountain_ensemble_weight_fullrank_action_watchtime_once_score": "fountain_ensemble_weight_fullrank_action_watchtime_once_score * fountain_ensemble_weight_fullrank_action_watchtime_once_score_coeff",
          "fountain_ensemble_power_weight_fullrank_ltr_v4_fountain_finish_rate": "fountain_ensemble_power_weight_fullrank_ltr_v4_fountain_finish_rate * fountain_ensemble_power_weight_fullrank_ltr_v4_fountain_finish_rate_coeff",
          "fountain_ensemble_power_weight_fullrank_svr_in_order_score": "fountain_ensemble_power_weight_fullrank_svr_in_order_score * fountain_ensemble_power_weight_fullrank_svr_in_order_score_coeff",
          "fountain_ensemble_weight_fullrank_sim_plvtr": "fountain_ensemble_weight_fullrank_sim_plvtr * fountain_ensemble_weight_fullrank_sim_plvtr_coeff",
          "fountain_ensemble_power_weight_fullrank_click_score": "fountain_ensemble_power_weight_fullrank_click_score * fountain_ensemble_power_weight_fullrank_click_score_coeff"
        }
      ) \
    .end_if_()
    return self
  
  def fullrank_follow_touch_high_adjust(self):
    """
    精排阶段双列和关注涨关摸高合作
    """
    self \
    .if_("enable_fountain_fullrank_follow_touch_high_adjust == 1") \
      .gen_common_attr_by_lua(
        attr_map={
          "fountain_ensemble_power_weight_fullrank_pvtr_multi_pwtr": "fountain_ensemble_power_weight_fullrank_pvtr_multi_pwtr * fountain_ensemble_power_weight_fullrank_pvtr_multi_pwtr_rise_follow_coeff"
        }
      ) \
    .end_if_()
    return self

  def request_feature_server(self):
    '''
    请求 feature server, 目前仅用于离线抽特征
    '''
    self.gen_common_attr_by_lua(
        attr_map={
            "fs_flow_control_id": "util.CityHash64(request_id)",
        }
    )
    self.check_tail_number(
        kconf_key='reco.arch.enableFountainRequestFs',
        test_value='{{fs_flow_control_id}}',
        output_to='enable_request_feature_server',
    )
    self.if_('enable_request_feature_server == 1 and (fr_sample_discard or 0) == 0')
    self.set_attr_value(
        common_attrs=[dict(name='fs_caller_biz', type='string', value='fountain'),
                      dict(name='fs_caller_stage', type='string', value='ranking'),
                      ]
    )
    self.str_format(
        format_string="%s%s",
        input_attrs=["request_id", "device_id"],
        output_attr="feature_server_hash_id",
    )
    self.delegate_enrich(
        kess_service="reco-feature-server-ranking",
        request_type="fountain-ranking-infer",
        consistent_hash=True,
        hash_id="{{feature_server_hash_id}}",
        send_item_attrs=[
            # FS 必需参数
            {"name": "reco_photo_info", "as": "context_info_str"},
            {"name": "reason_str", "as": "_reason"},
            {"name": "live_photo_info__is_living", "as": "_is_living"},

            {"name": "cascade_pctr_index", "as": "cascade_pctr_index"},
            {"name": "cascade_plvtr_index", "as": "cascade_plvtr_index"},
            {"name": "cascade_pvtr_index", "as": "cascade_pvtr_index"},
            {"name": "cascade_pltr_index", "as": "cascade_pltr_index"},
            {"name": "cascade_pftr_index", "as": "cascade_pftr_index"},
            {"name": "cascade_pwtr_index", "as": "cascade_pwtr_index"},
        ],
        send_common_attrs=[
            {"name": "fs_caller_biz", "as": "caller_biz"},
            {"name": "fs_caller_stage", "as": "caller_stage"},

            # FS 必需参数
            {"name": "tab_id", "as": "_tab_id"},
            {"name": "userInfo", "as": "user_info_str"},
            {"name": "page", "as": "page_common"},
            {"name": "featureSourcePId", "as": "source_photo_id"},

            # 目前仅用于离线抽特征，以下暂时不需要
            # {"name": "", "as": "infer_kess_service"},
            # {"name": "", "as": "infer_request_type"},
            # {"name": "", "as": "infer_partition_size"},
            # {"name": "", "as": "infer_timeout_ms"},
            # {"name": "", "as": "enable_infer_via_fs"},
        ],
        recv_common_attrs=["fs_res"],  # 无实际作用，仅用于成功请求
        use_packed_item_attr=True,
    )
    self.end_()
    return self
  
  def fr_interest_tagnex_tgi_adjust(self, interest_type):
    score_attr = f"fountain_user_{interest_type}_interest_tagnex_tgi_score"
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": score_attr, "as": "boost_discount_coeff"},
        {"name": "fullrank_ensemble_score_after_adjust", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "fullrank_ensemble_score_after_adjust"}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self