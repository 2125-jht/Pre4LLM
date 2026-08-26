#!/usr/bin/env python3
# coding=utf-8

from dragonfly.common_leaf_dsl import LeafFlow
from dragonfly.ext.subdivision.subdivision_api_mixin import subdivisionApiMixin
from dump_attr_to_kafka import dump_attr_to_kafka
from util import enrich_ab_param
from rerank.ab_params import get_rerank_v3_ab_params
from rerank.rerank_v3_mixin import RerankV3Mixin

class RerankFastV1Flow(LeafFlow, RerankV3Mixin, subdivisionApiMixin):
  def __init__(self):
    LeafFlow.__init__(self, "rerank_fast_v1")
    self \
      .namespace_(ns = "rerank_fast_v1", nest = True) \
      .get_abtest_params(
        biz_name = "RECO_RPC",
        ab_params = enrich_ab_param(get_rerank_v3_ab_params()),
        prioritized_suffix = "{{_ABTEST_SUFFIX_LIST_}}",
      )

    self.if_("enable_use_fountain_rerank == 1")
    self \
      .rerank_v3() \
      .copy_attr(
        attrs = [
          {
            "from_item": "rerank_list_score",
            "to_item": "virtual_rerank_score",
          },
        ],
      )
      
    dump_attr_to_kafka(
      self,
      stage_name = "rerank",
      dump_item_attr_list = [
        "virtual_rerank_score",
      ],
      range_end = 60,
    )

    self.end_()

    self \
      .if_("fountain_enable_cascade_distill_sample == 1") \
        .distill_sample() \
      .end_() \
      .if_("fountain_enable_cascade_distill_full_link_sample == 1") \
        .full_link_distill_sample() \
      .end_()

    self \
      .truncate(size_limit = "{{request_num}}") \
      .calc_result_count_to_ab_metric() \
      .perflog_reason_count(
        check_point = "rerank_finish",
      )

    cpu_cost_debug_info = [
      "retrieval_fast_cpu_cost_ts",
      "filter_fast_cpu_cost_ts",
      "cascade_fast_cpu_cost_ts",
      "full_rank_fast_cpu_cost_ts",
      "post_process_fast_cpu_cost_ts",
      "rerank_fast_cpu_cost_ts",
    ]

    self \
      .perflog_attr_value(
        check_point="rerank_top_6",
        item_attrs=["duration_perf_id"],
        aggregator="count",
        range_end = 6,
      ) \
      .copy_attr(
        attrs = [
          {
            "from_common": "fountain_reco_leaf_retrieval_default_ts",
            "to_common": "retrieval_fast_ts"
          },
          {
            "from_common": "fountain_reco_leaf_retrieval_default_cpu_cost_ts",
            "to_common": "retrieval_fast_cpu_cost_ts"
          },
          {
            "from_common": "fountain_reco_leaf_filter_default_ts",
            "to_common": "filter_fast_ts"
          },
          {
            "from_common": "fountain_reco_leaf_filter_default_cpu_cost_ts",
            "to_common": "filter_fast_cpu_cost_ts"
          }
        ]
      ) \
      .copy_user_meta_info(
        save_flow_cpu_cost_to_attr = "rerank_fast_cpu_cost_ts",
      ) \
      .gen_common_attr_by_lua(
        attr_map = {
          "rank_fast_ts": "util.GetTimestamp() - rank_fast_begin_ts",
          "total_fast_ts": "util.GetTimestamp() - prepare_begin_ts",
          "request_fast_count": "1",
        },
      ) \
      ._count_photo_type_distribution("leaf_show_fast") \
      .explore_environment_type_enrich(
        type_map = {
          "fountainRecoLeaf": "prod",
          "fountainLeaf_2022Q1combo": "prod", # 扩索引实验 hold leaf
          "fountainRecoLeafGray": "gray",
          "fountainRecoLeafAbGray": "gray",
          "fountainRecoLeafV2": "gray",
          "fountainRecoLeafTest": "gray",
          "default": "other",
        },
        save_type_to_attr = "env_type",
      ) \
      .gen_common_attr_by_lua(
        attr_map = {
          "_IS_ONLINE_SERVICE_": "env_type == \"prod\" or env_type == \"gray\"",
        }
      ) \
      .send_abtest_metrics(
        skip = "{{return _IS_ONLINE_SERVICE_ == 0}}",
        metrics = [
          "retrieval_fast_ts",
          "filter_fast_ts",
          "cascade_fast_ts",
          "rank_fast_ts",
          "total_fast_ts",
          "retrieval_fast_cpu_cost_ts",
          "filter_fast_cpu_cost_ts",
          "cascade_fast_cpu_cost_ts",
          "full_rank_fast_cpu_cost_ts",
          "post_process_fast_cpu_cost_ts",
          "rerank_fast_cpu_cost_ts",
          "filter_finish_fast_single_picture_count",
          "filter_finish_fast_long_picture_count",
          "filter_finish_fast_cluster_picture_count",
          "leaf_show_fast_single_picture_count",
          "leaf_show_fast_long_picture_count",
          "leaf_show_fast_cluster_picture_count",
          "filter_finish_fast_item_num",
          "request_fast_count",
          "rank_model_input_count",
          "fountain_prerank_hot_content_count",
          "fountain_prerank_authority_content_count",
          "fountain_prerank_personified_author_count",
          "fountain_prerank_blacklist_author_count",
          "fountain_prerank_duration_0_7s_count",
          "fountain_prerank_duration_7_9s_count",
          "fountain_prerank_duration_9_12s_count",
          "fountain_prerank_duration_12_17s_count",
          "fountain_prerank_duration_17_20s_count",
          "fountain_prerank_duration_20_58s_count",
          "fountain_prerank_duration_58_120s_count",
          "fountain_prerank_duration_gt_120s_count",
          "cascade_prerank_truncate",
          "fountain_mc_stage1_hot_content_count",
          "fountain_mc_stage1_authority_content_count",
          "fountain_mc_stage1_personified_author_count",
          "fountain_mc_stage1_duration_0_7s_count",
          "fountain_mc_stage1_duration_7_9s_count",
          "fountain_mc_stage1_duration_9_12s_count",
          "fountain_mc_stage1_duration_12_17s_count",
          "fountain_mc_stage1_duration_17_20s_count",
          "fountain_mc_stage1_duration_20_58s_count",
          "fountain_mc_stage1_duration_58_120s_count",
          "fountain_mc_stage1_duration_gt_120s_count",
          "fountain_mc_stage1_collection_count",
          "cascade_combine_variant",
          "fountain_mc_stage2_hot_content_count",
          "fountain_mc_stage2_authority_content_count",
          "fountain_mc_stage2_personified_author_count",
          "fountain_mc_stage2_duration_0_7s_count",
          "fountain_mc_stage2_duration_7_9s_count",
          "fountain_mc_stage2_duration_9_12s_count",
          "fountain_mc_stage2_duration_12_17s_count",
          "fountain_mc_stage2_duration_17_20s_count",
          "fountain_mc_stage2_duration_20_58s_count",
          "fountain_mc_stage2_duration_58_120s_count",
          "fountain_mc_stage2_duration_gt_120s_count",
          "fountain_mc_stage2_collection_count",
          "cascade_final",
          "fountain_fr_stage1_hot_content_count",
          "fountain_fr_stage1_authority_content_count",
          "fountain_fr_stage1_personified_author_count",
          "fountain_fr_stage1_duration_0_7s_count",
          "fountain_fr_stage1_duration_7_9s_count",
          "fountain_fr_stage1_duration_9_12s_count",
          "fountain_fr_stage1_duration_12_17s_count",
          "fountain_fr_stage1_duration_17_20s_count",
          "fountain_fr_stage1_duration_20_58s_count",
          "fountain_fr_stage1_duration_58_120s_count",
          "fountain_fr_stage1_duration_gt_120s_count",
          "fountain_fr_stage1_collection_count",
          "fountain_fr_stage1_count",
          "fountain_fr_stage2_hot_content_count",
          "fountain_fr_stage2_authority_content_count",
          "fountain_fr_stage2_personified_author_count",
          "fountain_fr_stage2_duration_0_7s_count",
          "fountain_fr_stage2_duration_7_9s_count",
          "fountain_fr_stage2_duration_9_12s_count",
          "fountain_fr_stage2_duration_12_17s_count",
          "fountain_fr_stage2_duration_17_20s_count",
          "fountain_fr_stage2_duration_20_58s_count",
          "fountain_fr_stage2_duration_58_120s_count",
          "fountain_fr_stage2_duration_gt_120s_count",
          "fountain_fr_stage2_collection_count",
          "fountain_fr_stage2_count",
          "fountain_rerank_duration_0_7s_top6_count",
          "fountain_rerank_duration_7_9s_top6_count",
          "fountain_rerank_duration_9_12s_top6_count",
          "fountain_rerank_duration_12_17s_top6_count",
          "fountain_rerank_duration_17_20s_top6_count",
          "fountain_rerank_duration_20_58s_top6_count",
          "fountain_rerank_duration_58_120s_top6_count",
          "fountain_rerank_duration_gt_120s_top6_count",
          "rerank_list_source_es_add_dpp",
          "rerank_list_source_es_mul_dpp",
          "rerank_list_source_hetu_dpp",
          "rerank_list_source_hetu_fusion_dpp",
          "rerank_list_source_gen_model_dpp",
          "rerank_list_source_fr_ensemble_dpp",
          "rerank_list_source_gen_nar_model_dpp",
          "rerank_list_source_gen_ar_model_dpp",
          "rerank_list_source_gen_ar_pinrec_model_dpp",
          "rerank_list_source_gen_rankmixer_model_dpp",
          "rerank_list_source_es_add_ssd",
          "rerank_list_source_es_mul_ssd",
          "rerank_list_source_hetu_ssd",
          "rerank_list_source_hetu_fusion_ssd",
          "rerank_list_source_gen_model_ssd",
          "rerank_list_source_fr_ensemble_ssd",
          "rerank_list_source_gen_ar_model_ssd",
          "rerank_list_source_gen_nar_model_ssd",
          "rerank_list_source_gen_ar_pinrec_model_ssd",
          "rerank_list_source_gen_rankmixer_model_ssd",
          "fountain_rr_list_reason_10_count",
          "fountain_rank_top60_mc_index_avg",
          "fountain_rerank_top10_rank_index_avg",
          "rerank_list_score_avg",
          "rerank_list_score_max",
        ],
        metric_name_prefix = "fountain_reco_leaf_"
      ) \
      .log_debug_info(
        common_attrs = cpu_cost_debug_info,
        for_debug_request_only = True,
      ) \
      .namespace_()

  def _count_photo_type_distribution(self, stage):
    self \
      .count_reco_result(
        save_count_to = "%s_single_picture_count" % stage,
        target_item = {"picture_type": 1}
      ) \
      .count_reco_result(
        save_count_to = "%s_long_picture_count" % stage,
        target_item = {"picture_type": 2}
      ) \
      .count_reco_result(
        save_count_to = "%s_cluster_picture_count" % stage,
        target_item = {"picture_type": 3}
      ) \
    
    return self

  def distill_sample(self):
    self \
    .get_kconf_params(
      kconf_configs=[
      {
        "kconf_key": "reco.fountain.sampleUserFeatures",
        "value_type": "list_string",
        "default_value": [],
        "export_common_attr": "sample_user_features_rerank"
      },
      {
        "kconf_key": "reco.fountain.sampleItemFeatures",
        "value_type": "list_string",
        "default_value": [],
        "export_common_attr": "sample_item_features_rerank"
      },
      {
        "kconf_key": "reco.offline.fountainMcDistillRankParamRerank",
        "value_type": "json",
        "json_path": "fullrank_seg1_sample_begin",
        "export_common_attr": "fountain_rerank_seg1_sample_begin"
      },
      {
        "kconf_key": "reco.offline.fountainMcDistillRankParamRerank",
        "value_type": "json",
        "json_path": "fullrank_seg1_sample_end",
        "export_common_attr": "fountain_rerank_seg1_sample_end"
      },
      {
        "kconf_key": "reco.offline.fountainMcDistillRankParamRerank",
        "value_type": "json",
        "json_path": "fullrank_seg1_sample_num",
        "export_common_attr": "fountain_rerank_seg1_sample_num"
      },
      {
        "kconf_key": "reco.offline.fountainMcDistillRankParamRerank",
        "value_type": "json",
        "json_path": "fullrank_seg2_sample_begin",
        "export_common_attr": "fountain_rerank_seg2_sample_begin"
      },
      {
        "kconf_key": "reco.offline.fountainMcDistillRankParamRerank",
        "value_type": "json",
        "json_path": "fullrank_seg2_sample_end",
        "export_common_attr": "fountain_rerank_seg2_sample_end"
      },
      {
        "kconf_key": "reco.offline.fountainMcDistillRankParamRerank",
        "value_type": "json",
        "json_path": "fullrank_seg2_sample_num",
        "export_common_attr": "fountain_rerank_seg2_sample_num"
      },
      {
        "kconf_key": "reco.offline.fountainMcDistillRankParamRerank",
        "value_type": "json",
        "json_path": "fullrank_seg3_sample_begin",
        "export_common_attr": "fountain_rerank_seg3_sample_begin"
      },
      {
        "kconf_key": "reco.offline.fountainMcDistillRankParamRerank",
        "value_type": "json",
        "json_path": "fullrank_seg3_sample_end",
        "export_common_attr": "fountain_rerank_seg3_sample_end"
      },
      {
        "kconf_key": "reco.offline.fountainMcDistillRankParamRerank",
        "value_type": "json",
        "json_path": "fullrank_seg3_sample_num",
        "export_common_attr": "fountain_rerank_seg3_sample_num"
      },
      {
        "kconf_key": "reco.offline.fountainMcDistillRankParamRerank",
        "value_type": "json",
        "json_path": "fullrank_seg_all_sample_begin",
        "export_common_attr": "fountain_rerank_seg_all_sample_begin"
      },
      {
        "kconf_key": "reco.offline.fountainMcDistillRankParamRerank",
        "value_type": "json",
        "json_path": "fullrank_seg_all_sample_end",
        "export_common_attr": "fountain_rerank_seg_all_sample_end"
      },
      {
        "kconf_key": "reco.offline.fountainMcDistillRankParamRerank",
        "value_type": "json",
        "json_path": "fullrank_seg_all_sample_num",
        "export_common_attr": "fountain_rerank_seg_all_sample_num"
      },
      {
        "kconf_key": "reco.offline.fountainMcDistillRankParamRerank",
        "value_type": "json",
        "json_path": "rerank_all_distill_sample_ratio",
        "export_common_attr": "rerank_all_distill_sample_ratio"
      },
    ]) \
    .fountain_enrich_sample_package(  # TODO(xuwei09) 级联全样本采样，资源原因，暂保留，预计 Q2 末删除
      item_attrs = [
        "item_seq"
      ],
      item_attrs_from_kconf = "sample_item_features_rerank",
      common_attrs = [
        "featureUId"
      ],
      common_attrs_from_kconf = "sample_user_features_rerank",
      sample_ratio = "rerank_all_distill_sample_ratio",
      sample_config = [
        {
          "sample_begin": "fountain_rerank_seg_all_sample_begin",
          "sample_end": "fountain_rerank_seg_all_sample_end",
          "sample_num": "fountain_rerank_seg_all_sample_num",
          "label_name": "rerank_distill_all",
        },
      ],
      load_attr = "cascadeSamplePackage",
      output_attr = "cascadeSamplePackage",
      check_point = "rerank_all",
      size_limit = "{{fountain_rerank_seg1_sample_size_limit}}",
    ) \
    .fountain_enrich_sample_package(
      item_attrs=[
        "item_seq"
      ],
      item_attrs_from_kconf="sample_item_features_rerank",
      common_attrs=[
        "featureUId"
      ],
      common_attrs_from_kconf="sample_user_features_rerank",
      sample_config=[
        {
          "sample_begin": "fountain_rerank_seg1_sample_begin",
          "sample_end": "fountain_rerank_seg1_sample_end",
          "sample_num": "fountain_rerank_seg1_sample_num",
          "label_name": "rerank_rank_seg1",
        },
        {
          "sample_begin": "fountain_rerank_seg2_sample_begin",
          "sample_end": "fountain_rerank_seg2_sample_end",
          "sample_num": "fountain_rerank_seg2_sample_num",
          "label_name": "rerank_rank_seg2",
        },
        {
          "sample_begin": "fountain_rerank_seg3_sample_begin",
          "sample_end": "fountain_rerank_seg3_sample_end",
          "sample_num": "fountain_rerank_seg3_sample_num",
          "label_name": "rerank_rank_seg3",
        }
      ],
      load_attr="cascadeSamplePackage",
      output_attr="cascadeSamplePackage",
      check_point="rerank_positive",
      size_limit="{{fountain_rerank_seg1_sample_size_limit}}",
    ) \
    .send_with_kafka(
      common_attr="cascadeSamplePackage",
      topic_name="fountain_cascade_sample_log"
    )

    return self
  
  def full_link_distill_sample(self):
    self \
    .get_kconf_params(
      kconf_configs=[
        {
          "kconf_key": "reco.offline.fountainFulllinkDistillRankParam",
          "value_type": "json",
          "json_path": "rerank_full_link_distill_sample_begin",
          "export_common_attr": "fountain_rerank_full_link_distill_sample_begin"
        },
        {
          "kconf_key": "reco.offline.fountainFulllinkDistillRankParam",
          "value_type": "json",
          "json_path": "rerank_full_link_distill_sample_end",
          "export_common_attr": "fountain_rerank_full_link_distill_sample_end"
        },
        {
          "kconf_key": "reco.offline.fountainFulllinkDistillRankParam",
          "value_type": "json",
          "json_path": "rerank_full_link_distill_sample_num",
          "export_common_attr": "fountain_rerank_full_link_distill_sample_num"
        },
        {
          "kconf_key": "reco.offline.fountainFulllinkDistillRankParam",
          "value_type": "json",
          "json_path": "rerank_full_link_distill_sample_ratio",
          "export_common_attr": "fountain_rerank_full_link_distill_sample_ratio"
        },
        {
          "kconf_key": "reco.offline.fountainFulllinkDistillRankParam",
          "value_type": "json",
          "json_path": "send_rerank_eval_list",
          "export_common_attr": "fl_send_rerank_eval_list"
        },
        {
          "kconf_key": "reco.hot.frCarmPidListSampleCount",
          "json_path": "fountain",
          "default_value": 0,
          "export_common_attr": "fountain_fr_carm_pid_sample_count"
        },
      ]
    ) \
    .if_("fountain_fr_carm_pid_sample_count > 0") \
      .explore_mc_distill_sample_enrich(  # 精排主模型 carm 特征回流
        candidate_list_attr = "cascade_output_item_id_list",
        sample_num = "{{fountain_fr_carm_pid_sample_count}}",
        save_sample_result_to = "fr_carm_pid_list",
      ) \
    .end_() \
    .explore_full_link_context_sample_reco_log_enricher(
      sample_config = [
        {
          "sample_begin": "fountain_rerank_full_link_distill_sample_begin",
          "sample_end": "fountain_rerank_full_link_distill_sample_end",
          "sample_num": "fountain_rerank_full_link_distill_sample_num",
          "label_name": "final_pos",
        },
      ],
      sample_ratio = "fountain_rerank_full_link_distill_sample_ratio",
      user_info_attr = "userInfoPb",
      send_rerank_eval_list = "{{fl_send_rerank_eval_list}}",
      rerank_list_item_idx_flat_list_attr = "rerank_list_item_idx_flat_list",
      rerank_list_score_list_attr = "rerank_list_score_list",
      load_attr = "fountain_full_link_reco_log_message",
      save_result_to = "fountain_full_link_reco_log_message_final",
      rank_index = "rank_index_after_es",
      final_index = "item_seq",
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
      psvr = "fullrank_sim_psvr",
      pepstr = "fullrank_sim_pepstr",
      pcltr = "fullrank_sim_pcltr",
      pwtd = "fullrank_sim_pfintr",
      pcpr = "fullrank_sim_pcpr",
      fullrank_ltr_score = "fullrank_ltr_score",
      fullrank_act_wtd = "fullrank_act_wtd",
      fullrank_ltr_v4_fountain_next = "fullrank_ltr_v4_fountain_next",
      fountain_related_score_v2 = "fountain_related_score_v2",
      size_limit = 60,
    ) \
    .send_with_kafka(
      common_attr = "fountain_full_link_reco_log_message_final",
      topic_name = "full_link_samples",
    )

    return self

  def _rerank_count_distribution(self):
    """
    rerank 之后统计视频分布
    """
    self \
    .count_reco_result(
      save_count_to = "fountain_rerank_duration_0_7s_top6_count",
      target_item = {"duration_0_7s": 1},
      range_end = 6
    ) \
    .count_reco_result(
      save_count_to = "fountain_rerank_duration_7_9s_top6_count",
      target_item = {"duration_7_9s": 1},
      range_end = 6
    ) \
    .count_reco_result(
      save_count_to = "fountain_rerank_duration_9_12s_top6_count",
      target_item = {"duration_9_12s": 1},
      range_end = 6
    ) \
    .count_reco_result(
      save_count_to = "fountain_rerank_duration_12_17s_top6_count",
      target_item = {"duration_12_17s": 1},
      range_end = 6
    ) \
    .count_reco_result(
      save_count_to = "fountain_rerank_duration_17_20s_top6_count",
      target_item = {"duration_17_20s": 1},
      range_end = 6
    ) \
    .count_reco_result(
      save_count_to = "fountain_rerank_duration_20_58s_top6_count",
      target_item = {"duration_20_58s": 1},
      range_end = 6
    ) \
    .count_reco_result(
      save_count_to = "fountain_rerank_duration_58_120s_top6_count",
      target_item = {"duration_58_120s": 1},
      range_end = 6
    ) \
    .count_reco_result(
      save_count_to = "fountain_rerank_duration_gt_120s_top6_count",
      target_item = {"duration_gt_120s": 1},
      range_end = 6
    )

    return self

  def calc_result_count_to_ab_metric(self):
    return self \
      .cast_attr_type(
        attr_type_cast_configs=[
          {
            "to_type": "double",
            "from_item_attr": "rank_final_index",
            "to_item_attr": "rank_final_index_double"
          }
        ]
      ) \
      .pack_item_attr(
        item_source = {
          "reco_results": True,
          "total_limit": 10,
        },
        mappings = [
          {
            "aggregator": "avg",
            "from_item_attr": "rank_final_index_double",
            "to_common_attr": "fountain_rerank_top10_rank_index_avg"
          },
        ]
      ) \
      ._rerank_count_distribution()
