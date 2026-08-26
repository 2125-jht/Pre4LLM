from dragonfly.ext.explore.explore_api_mixin import ExploreApiMixin
from dragonfly.ext.retrieval.retrieval_api_mixin import RetrievalApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from dragonfly.ext.embed_calc.embed_calc_api_mixin import EmbedCalcApiMixin
from dragonfly.ext.kgnn.kgnn_api_mixin import KgnnApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin

from common import CommonRecoFlow
from retrieval.module.retrieval_perf_module import RetrievalPerfModule
from retrieval.module.photo_info_fetching_module import PhotoInfoFetchingModule

class RetrievalFlow(CommonRecoFlow, ExploreApiMixin, RetrievalApiMixin, GsuApiMixin, EmbedCalcApiMixin, KgnnApiMixin, KuibaApiMixin):
  def __init__(self, name: str, is_sub_flow: bool = False) -> None:
    super().__init__(name, "explore", "retrieval", "config", "module", "config/module", "lua/module", is_sub_flow)

  def _flow_end(self):
    reason_list = []
    for module in self.reco_stage.modules:
      if module.is_retrieval() and module.reason > 0:
        reason_list.append(module.reason)

    self \
      .do_nothing(  # 这个 processor 是为了先知打点出所有召回的原始结果
        name = "explore_retr",
        traceback = True,
      ) \
      ._perf_result(
        step_name = "origin",
        perf_sampling_attr = "_IS_PERF_SAMPLING_REQUEST_",
      ) \
      .get_abtest_params(
        biz_name = "RECO_RPC",
        ab_params = [
          {
            "attr_name": "explore_retr_reason_limit_num",
            "default_value": -1,
            "param_name": "explore_retr_reason_limit_num",
            "param_type": "int",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "explore_retr_reason_limit_default_weight",
            "default_value": 1.0,
            "param_name": "explore_retr_reason_limit_default_weight",
            "param_type": "double"
          },
          {
            "attr_name": "explore_retr_reason_limit_reason_weight_str",
            "default_value": "",
            "param_name": "explore_retr_reason_limit_reason_weight_str",
            "param_type": "string",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "explore_retr_reason_limit_skip_fillback_result",
            "default_value": False,
            "param_name": "explore_retr_reason_limit_skip_fillback_result",
            "param_type": "bool",
          },
          {
            "attr_name": "skip_explore_retr_score_enricher",
            "default_value": 1,
            "param_name": "skip_explore_retr_score_enricher",
            "param_type": "int",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "explore_retr_rank_smooth",
            "default_value": 0,
            "param_name": "explore_retr_rank_smooth",
            "param_type": "int",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "enable_explore_xhs_install_adjust_retr_num",
            "default_value": 0,
            "param_name": "enable_explore_xhs_install_adjust_retr_num",
            "param_type": "int",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "enable_explore_xhs_install_adjust_retr_boost_weight",
            "default_value": 1.0,
            "param_name": "enable_explore_xhs_install_adjust_retr_boost_weight",
            "param_type": "double",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "enable_explore_new_retr_quota_limit",
            "default_value": False,
            "param_name": "enable_explore_new_retr_quota_limit",
            "param_type": "bool",
          },
          {
            "attr_name": "enable_explore_tnu_retr_quota_limit",
            "default_value": False,
            "param_name": "enable_explore_tnu_retr_quota_limit",
            "param_type": "bool",
          },
          {
            "attr_name": "explore_tnu_retr_reason_limit_reason_weight_str",
            "default_value": "",
            "param_name": "explore_tnu_retr_reason_limit_reason_weight_str",
            "param_type": "string",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "explore_tnu_retr_reason_limit_num",
            "default_value": -1,
            "param_name": "explore_tnu_retr_reason_limit_num",
            "param_type": "int",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "explore_enable_dedup_on_author_id",
            "default_value": False,
            "param_name": "explore_enable_dedup_on_author_id",
            "param_type": "bool",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "explore_enable_write_audit_low_quality_pids_to_redis",
            "default_value": False,
            "param_name": "explore_enable_write_audit_low_quality_pids_to_redis",
            "param_type": "bool",
          },
          {
            "attr_name": "explore_audit_low_quality_pids_exp_tags_str",
            "default_value": "",
            "param_name": "explore_audit_low_quality_pids_exp_tags_str",
            "param_type": "string",
          },
          {
            "attr_name": "audit_low_quality_pids_target_count",
            "default_value": 500,
            "param_name": "audit_low_quality_pids_target_count",
            "param_type": "int",
          },
          {
            "attr_name": "audit_low_quality_pids_show_limit",
            "default_value": 0,
            "param_name": "audit_low_quality_pids_show_limit",
            "param_type": "int",
          },
          {
            "attr_name": "explore_audit_low_quality_pids_redis_expire_seconds",
            "default_value": 600,
            "param_name": "explore_audit_low_quality_pids_redis_expire_seconds",
            "param_type": "int",
          },
          {
            "attr_name": "explore_audit_low_quality_pids_key_prefix",
            "default_value": "e_audit_emp_",
            "param_name": "explore_audit_low_quality_pids_key_prefix",
            "param_type": "string",
          },
          {
            "attr_name": "explore_enable_low_pass_rate_global_filter",
            "default_value": False,
            "param_name": "explore_enable_low_pass_rate_global_filter",
            "param_type": "bool",
          },
          {
            "attr_name": "enable_explore_swing_i2i_filter",
            "default_value": 0,
            "param_name": "enable_explore_swing_i2i_filter",
            "param_type": "int",
          },
          {
            "attr_name": "enable_explore_swing_u2u_filter",
            "default_value": 0,
            "param_name": "enable_explore_swing_u2u_filter",
            "param_type": "int",
          },
          {
            "attr_name": "explore_enable_similary_neg_feedback_limit",
            "param_name": "explore_enable_similary_neg_feedback_limit",
            "param_type": "bool",
            "default_value": False
          },
          {
            "attr_name": "enable_collect_actionlist_version",
            "param_name": "enable_collect_actionlist_version",
            "param_type": "bool",
            "default_value": True,
          },
          {
            "attr_name": "enable_explore_truncate_topk",
            "param_name": "enable_explore_truncate_topk",
            "param_type": "bool",
            "default_value": False,
          },
          {
            "attr_name": "enable_explore_extra_no_click_stat",
            "param_name": "enable_explore_extra_no_click_stat",
            "param_type": "bool",
            "default_value": False,
          },
          {
            "attr_name": "enable_explore_retr_fix_low_hit_rate",
            "param_name": "enable_explore_retr_fix_low_hit_rate",
            "param_type": "bool",
            "default_value": False,
          },
          {
            "attr_name": "enable_explore_hetu_topk",
            "param_name": "enable_explore_hetu_topk",
            "param_type": "bool",
            "default_value": False,
          },
          {
            "attr_name": "explore_retr_emb_kess_name_for_similary_neg_feedback",
            "param_name": "explore_retr_emb_kess_name_for_similary_neg_feedback",
            "param_type": "string",
            "default_value": "grpc_MMUHetuContentEmbedding",
          },
          {
            "attr_name": "explore_retr_hate_similary_score_not_click_hour_limit",
            "param_name": "explore_retr_hate_similary_score_not_click_hour_limit",
            "param_type": "double",
            "default_value": 0.5
          },
          {
            "attr_name": "explore_retr_hate_similary_score_play_stat_hour_limit",
            "param_name": "explore_retr_hate_similary_score_play_stat_hour_limit",
            "param_type": "double",
            "default_value": 0.5
          },
          {
            "attr_name": "explore_retr_hate_similary_score_extra_not_click_hour_limit",
            "param_name": "explore_retr_hate_similary_score_extra_not_click_hour_limit",
            "param_type": "double",
            "default_value": 0.5
          },
          {
            "attr_name": "explore_retr_hate_similary_score_not_hetu_limit_topk",
            "param_name": "explore_retr_hate_similary_score_not_hetu_limit_topk",
            "param_type": "int",
            "default_value": 10
          },
          {
            "attr_name": "explore_retr_hate_similary_score_not_pid_limit_topk",
            "param_name": "explore_retr_hate_similary_score_not_pid_limit_topk",
            "param_type": "int",
            "default_value": 10
          },
          {
            "attr_name": "explore_retr_hate_similary_score_not_click_topk_limit",
            "param_name": "explore_retr_hate_similary_score_not_click_topk_limit",
            "param_type": "int",
            "default_value": 10
          },
          {
            "attr_name": "explore_retr_hate_similary_score_play_stat_topk_limit",
            "param_name": "explore_retr_hate_similary_score_play_stat_topk_limit",
            "param_type": "int",
            "default_value": 10
          },
          {
            "attr_name": "explore_retr_similary_neg_score_not_click_weight",
            "param_name": "explore_retr_similary_neg_score_not_click_weight",
            "param_type": "double",
            "default_value": 1.0
          },
          {
            "attr_name": "explore_retr_similary_neg_score_short_view_weight",
            "param_name": "explore_retr_similary_neg_score_short_view_weight",
            "param_type": "double",
            "default_value": 1.0
          },
          {
            "attr_name": "explore_retr_similary_neg_score_extra_not_click_weight",
            "param_name": "explore_retr_similary_neg_score_extra_not_click_weight",
            "param_type": "double",
            "default_value": 1.0
          },
          {
            "attr_name": "explore_retr_similary_neg_score_short_view_threshold",
            "param_name": "explore_retr_similary_neg_score_short_view_threshold",
            "param_type": "int",
            "default_value": 3000
          },
          {
            "attr_name": "explore_similary_neg_score_thres",
            "param_name": "explore_similary_neg_score_thres",
            "param_type": "double",
            "default_value": 30.0
          },
          {
            "attr_name": "explore_enable_similary_neg_feedback_filter_by_rank",
            "param_name": "explore_enable_similary_neg_feedback_filter_by_rank",
            "param_type": "bool",
            "default_value": False,
          },
          {
            "attr_name": "explore_similary_neg_feedback_limit_percent",
            "param_name": "explore_similary_neg_feedback_limit_percent",
            "param_type": "double",
            "default_value": 1.0
          },
          {
            "attr_name": "explore_retr_enable_fountain_play_stat",
            "param_name": "explore_retr_enable_fountain_play_stat",
            "param_type": "bool",
            "default_value": False,
          },
          {
            "attr_name": "enable_explore_retr_judge_next_photo_stat",
            "param_name": "enable_explore_retr_judge_next_photo_stat",
            "param_type": "bool",
            "default_value": False,
          },
          {
            "attr_name": "explore_retr_similary_neg_score_enable_avg_pooling",
            "param_name": "explore_retr_similary_neg_score_enable_avg_pooling",
            "param_type": "bool",
            "default_value": False,
          },
          {
            "attr_name": "explore_pic_retr_reason_limit_num",
            "default_value": 0,
            "param_name": "explore_pic_retr_reason_limit_num",
            "param_type": "int",
          },
          {
            "attr_name": "explore_pic_retr_reason_limit_reason_weight_str",
            "default_value": "",
            "param_name": "explore_pic_retr_reason_limit_reason_weight_str",
            "param_type": "string",
          },
          {
            "attr_name": "explore_retr_reason_limit_non_personlized_recall_max_weight",
            "default_value": 5,
            "param_name": "explore_retr_reason_limit_non_personlized_recall_max_weight",
            "param_type": "int",
          },
          {
            "attr_name": "explore_retr_reason_limit_high_active_user_avg_vv",
            "default_value": 10,
            "param_name": "explore_retr_reason_limit_high_active_user_avg_vv",
            "param_type": "int",
          },
          {
            "attr_name": "explore_retr_reason_limit_non_personalized_recall_reasons",
            "default_value": "",
            "param_name": "explore_retr_reason_limit_non_personalized_recall_reasons",
            "param_type": "string",
          },
          {
            "attr_name": "explore_retr_reason_limit_skip_fillback_reasons",
            "default_value": "",
            "param_name": "explore_retr_reason_limit_skip_fillback_reasons",
            "param_type": "string",
          },
          {
            "attr_name": "enable_explore_bad_item_list_similarity_score",
            "default_value": False,
            "param_name": "enable_explore_bad_item_list_similarity_score",
            "param_type": "bool",
          },
          {
            "attr_name": "explore_retr_only_bad_item_save_count",
            "default_value": 100,
            "param_name": "explore_retr_only_bad_item_save_count",
            "param_type": "int",
          },
          {
            "attr_name": "enable_explore_sense_bad_item_list_similarity_score",
            "default_value": False,
            "param_name": "enable_explore_sense_bad_item_list_similarity_score",
            "param_type": "bool",
          },
          {
            "attr_name": "explore_retr_only_sense_bad_item_save_count",
            "default_value": 100,
            "param_name": "explore_retr_only_sense_bad_item_save_count",
            "param_type": "int",
          },
          {
            "attr_name": "enable_explore_hot_audit_bad_item_list_similarity_score",
            "default_value": False,
            "param_name": "enable_explore_hot_audit_bad_item_list_similarity_score",
            "param_type": "bool",
          },
          {
            "attr_name": "explore_retr_only_hot_audit_bad_item_save_count",
            "default_value": 100,
            "param_name": "explore_retr_only_sense_hot_audit_item_save_count",
            "param_type": "int",
          },
          {
            "attr_name": "enable_explore_topk_audit_bad_item_list_similarity_score",
            "default_value": False,
            "param_name": "enable_explore_topk_audit_bad_item_list_similarity_score",
            "param_type": "bool",
          },
          {
            "attr_name": "explore_retr_only_topk_audit_bad_item_save_count",
            "default_value": 60,
            "param_name": "explore_retr_only_sense_topk_audit_item_save_count",
            "param_type": "int",
          },
          {
            "attr_name": "enable_explore_questionnaire_good_item_list",
            "default_value": False,
            "param_name": "enable_explore_questionnaire_good_item_list",
            "param_type": "bool",
          },
          {
            "attr_name": "explore_questionnaire_satisfaction_rate_threshold",
            "default_value": 0.8,
            "param_name": "explore_questionnaire_satisfaction_rate_threshold",
            "param_type": "double",
          },
          {
            "attr_name": "explore_questionnaire_satisfaction_min_exposure_count",
            "default_value": 50,
            "param_name": "explore_questionnaire_satisfaction_min_exposure_count",
            "param_type": "int",
          },
          {
            "attr_name": "explore_questionnaire_satisfaction_rate_denominator_bias",
            "default_value": 10.0,
            "param_name": "explore_questionnaire_satisfaction_rate_denominator_bias",
            "param_type": "double",
          },
          {
            "attr_name": "explore_retr_questionnaire_good_item_save_count",
            "default_value": 100,
            "param_name": "explore_retr_questionnaire_good_item_save_count",
            "param_type": "int",
          },
          {
            "attr_name": "explore_questionnaire_age_threshold",
            "default_value": 7,
            "param_name": "explore_questionnaire_age_threshold",
            "param_type": "int",
          },
        ],
        prioritized_suffix = "{{_ABTEST_SUFFIX_LIST_}}"
      ) \
      .explore_item_reason_score_enricher(
        mappings = [{
          "reason": 3073,
          "to_item_attr": "pdn_rank_score",
        },
        {
          "reason": 3099,
          "to_item_attr": "comirec_rank_score",
        },
        {
          "reason": 3061,
          "to_item_attr": "colossus_ann_rank_score",
        }],
        smoothing = "{{explore_retr_rank_smooth}}",
        skip = "{{skip_explore_retr_score_enricher}}"
      ) \
      .deduplicate(
        name = "explore_retr_dedup",
        traceback = True,
        reason_priority_list = reason_list,
        save_dup_count_to = "retrieval_dup_count",
        append_reason_to = "multi_retr_reason"
      ) \
      .enrich_attr_by_light_function(
        export_item_attr = [
          "retr_rank",
        ],
        function_name = "SetRetrRank",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .filter_by_browse_set(
        name = "explore_retr_browse_set_filter",
        traceback = True,
      ) \
      .if_("mmu_sim_emb_dup_pid_list ~= nil and #mmu_sim_emb_dup_pid_list > 0") \
        .filter_by_common_attr(
          name = "explore_mmu_sim_emb_dup_pid_list_filter",
          traceback = True,
          common_attr = ["mmu_sim_emb_dup_pid_list"]
        ) \
      .end_() \
      .if_("hate_photo_i2i_neg_pid_list ~= nil and #hate_photo_i2i_neg_pid_list > 0") \
        .filter_by_common_attr(
          name = "explore_hate_i2i_retr_pid_list_filter",
          traceback = True,
          common_attr = ["hate_photo_i2i_neg_pid_list"]
        ) \
      .end_() \
      .if_("mmu_sim_emb_neg_pid_list ~= nil and #mmu_sim_emb_neg_pid_list > 0") \
        .filter_by_common_attr(
          name = "explore_mmu_sim_emb_neg_pid_list_filter",
          traceback = True,
          common_attr = ["mmu_sim_emb_neg_pid_list"]
        ) \
      .end_() \
      .filter_by_common_attr(
        name = "explore_cascade_neg_photo_id_list_filter",
        traceback = True,
        common_attr = ["cascade_neg_photo_id_list"]
      ) \
      .if_("enable_explore_swing_i2i_filter == 1 and swing_user_hate_i2i_list ~= nil and #swing_user_hate_i2i_list > 0") \
        .filter_by_common_attr(
          name = "swing_i2i_neg_photo_id_list_filter",
          traceback = True,
          common_attr = ["swing_user_hate_i2i_list"]
        ) \
      .end_() \
      .if_("enable_explore_swing_u2u_filter == 1 and swing_user_hate_u2u2i_list ~= nil and #swing_user_hate_u2u2i_list > 0") \
        .filter_by_common_attr(
          name = "swing_u2u_neg_photo_id_list_filter",
          traceback = True,
          common_attr = ["swing_user_hate_u2u2i_list"]
        ) \
      .end_() \
      .if_("mc_i2i_neg_pid_list ~= nil and #mc_i2i_neg_pid_list > 0") \
        .filter_by_common_attr(
          name = "explore_mc_i2i_neg_pid_list_filter",
          traceback = True,
          common_attr = ["mc_i2i_neg_pid_list"]
        ) \
      .end_() \
      .if_("explore_enable_low_pass_rate_global_filter == 1") \
        .explore_memory_data_ptr_filter(
          name = "explore_low_pass_rate_photo_set_filter",
          traceback = True,
          memory_data_ptr_attr = "explore_low_pass_rate_photo_set"
        ) \
      .end_() \
      .if_("explore_enable_similary_neg_feedback_limit == 1") \
        .similary_neg_feedback_limit() \
      .end_() \
      .if_("enable_explore_new_retr_quota_limit == 1") \
        .explore_retr_quota_limit(
          name = "explore_new_retr_quota_limit",
          traceback = True,
          limit_num = "{{return explore_retr_reason_limit_num + explore_pic_retr_reason_limit_num}}",
          quota_groups = [
            {
              "name": "default",
              "as_default": True,
              "quota_num": "{{explore_retr_reason_limit_num}}",
              "quota_weight": "{{explore_retr_reason_limit_reason_weight_str}}",
              "default_weight": 1.0,
            },
            {
              "name": "picture",
              "as_default": False,
              "quota_num": "{{explore_pic_retr_reason_limit_num}}",
              "quota_weight": "{{explore_pic_retr_reason_limit_reason_weight_str}}",
              "default_weight": 1.0,
            },
          ],
        ) \
      .else_if_("enable_explore_tnu_retr_quota_limit == 1 and uIsExploreTnuCrowdUser == 1") \
        .if_("enable_retr_filter_downgrade == 1") \
          .string_format( # 当外流降级生效时，tnu召回白名单新增9999 topk审核通过召回
            is_common_attr = True,
            format_string = "%s;9999:1.0",
            input_attrs = ["explore_tnu_retr_reason_limit_reason_weight_str"],
            output_attr = "explore_tnu_retr_reason_limit_reason_weight_str",
          ) \
        .end_() \
        .explore_retr_reason_limit(
          name = "explore_tnu_retr_quota_limit",
          traceback = True,
          size_limit = "{{explore_tnu_retr_reason_limit_num}}",
          default_weight = 0.0,
          reason_weight_str = "{{explore_tnu_retr_reason_limit_reason_weight_str}}",
          skip_fillback_result = True
        ) \
      .else_() \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "explore_retr_reason_limit_non_personlized_recall_max_weight",
            "explore_retr_reason_limit_high_active_user_avg_vv",
            "active_days_avg_vv",
            "explore_retr_reason_limit_non_personalized_recall_reasons"
          ],
          export_common_attr = [
            "explore_retr_non_personlized_reason_weight_str",
          ],
          function_name = "GenNonPersonalizedRecallWeight",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .str_format(
          format_string = "%s%s",
          input_attrs = [
            "explore_retr_reason_limit_reason_weight_str",
            "explore_retr_non_personlized_reason_weight_str"
          ],
          output_attr = "explore_retr_reason_limit_reason_weight_str_all",
        ) \
        .explore_retr_reason_limit(
          name = "explore_retr_quota_limit",
          traceback = True,
          skip_fillback_reason_str = "{{explore_retr_reason_limit_skip_fillback_reasons}}",
          size_limit = "{{explore_retr_reason_limit_num}}",
          default_weight = "{{explore_retr_reason_limit_default_weight}}",
          reason_weight_str = "{{explore_retr_reason_limit_reason_weight_str_all}}",
          skip_fillback_result = "{{explore_retr_reason_limit_skip_fillback_result}}"
        ) \
      .end_()

    self.namespace_(ns = "photo_info", nest = True)
    photo_info_module = PhotoInfoFetchingModule("photo_info")
    photo_info_module.set_flow(self)
    photo_info_module.process()
    self.namespace_()

    self \
      .pack_item_attr(  # 保存进入召回封面灰劣结果集，单独发送样本流
        item_source = {
          "reco_results": True
        },
        mappings = [{
          "aggregator": "concat",
          "from_item_attr": "photo_id",
          "to_common_attr": "retrieval_bad_cover_input_item_key_list"
        }],
        target_item = { "audit_hot_cover_level": [2023742, 2023743, 2023744, 2023745, 2023746, 2231037] }
      ) \
      .if_("enable_explore_bad_item_list_similarity_score == 1") \
        .pack_item_attr(  # 保存进入召回封面劣质结果集，计算相似度
          item_source = {
            "reco_results": True,
            "total_limit" : "{{explore_retr_only_bad_item_save_count}}",
          },
          mappings = [{
            "aggregator": "concat",
            "from_item_attr": "photo_id",
            "to_common_attr": "retrieval_only_bad_cover_input_item_key_list",
          }],
          target_item = { "audit_hot_cover_level": [2023746] }
        ) \
      .end_() \
      .if_("enable_explore_sense_bad_item_list_similarity_score == 1") \
        .pack_item_attr(  # 保存进入召回封面劣质结果集，计算相似度
          item_source = {
            "reco_results": True,
            "total_limit" : "{{explore_retr_only_sense_bad_item_save_count}}",
          },
          mappings = [{
            "aggregator": "concat",
            "from_item_attr": "photo_id",
            "to_common_attr": "retrieval_only_bad_sense_input_item_key_list",
          }],
          target_item = { "content_safety_level_with_namespace__level_hot_online": [0, 1] }
        ) \
      .end_() \
      .if_("enable_explore_hot_audit_bad_item_list_similarity_score == 1") \
        .pack_item_attr(  # 保存进入召回高热审劣质结果集，计算相似度
          item_source = {
            "reco_results": True,
            "total_limit" : "{{explore_retr_only_hot_audit_bad_item_save_count}}",
          },
          mappings = [{
            "aggregator": "concat",
            "from_item_attr": "photo_id",
            "to_common_attr": "retrieval_only_bad_hot_audit_input_item_key_list",
          }],
          target_item = { "audit_hot_high_tag_level": [1] }
        ) \
      .end_() \
      .if_("enable_explore_topk_audit_bad_item_list_similarity_score == 1") \
        .pack_item_attr(  # 保存进入召回高热审劣质结果集，计算相似度
          item_source = {
            "reco_results": True,
            "total_limit" : "{{explore_retr_only_topk_audit_bad_item_save_count}}",
          },
          mappings = [{
            "aggregator": "concat",
            "from_item_attr": "photo_id",
            "to_common_attr": "retrieval_only_bad_topk_audit_input_item_key_list",
          }],
          target_item = { "topk_audit_level": [1] }
        ) \
      .end_() \
      .if_("enable_explore_questionnaire_good_item_list == 1 and user_age_segment > 0 and user_age_segment <= explore_questionnaire_age_threshold") \
        .questionnaire_good_item_list() \
      .end_() \

    self \
      .if_("explore_enable_write_audit_low_quality_pids_to_redis == 1", to_be_delete = "date=2024-05-29;committer=liuhao07") \
        .split_string( # (liuhao07) Todo 10.08 召回源写死
          input_common_attr = "explore_audit_low_quality_pids_exp_tags_str",
          output_common_attr = "explore_audit_low_quality_pids_exp_tags",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "audit_low_quality_pids_show_limit", "as": "show_limit"},
            {"name": "audit_low_quality_pids_target_count", "as": "target_count"},
            {"name": "explore_audit_low_quality_pids_exp_tags", "as": "exp_tag_list"}
          ],
          import_item_attr = [
            "explore_stat__real_show_count",
            "explore_stat__view_length_sum",
            "topk_audit_level",
            "audit_hot_high_tag_level",
            "photo_id"
          ],
          export_common_attr = [
            {"name": "target_pids", "as": "audit_low_quality_pids"}
          ],
          function_name = "SelectAuditLowQualityPids",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .write_to_redis(
          kcc_cluster = "recoExploreNegPhoto",
          timeout = 10,
          expire_second = "{{explore_audit_low_quality_pids_redis_expire_seconds}}",
          key_prefix = "{{explore_audit_low_quality_pids_key_prefix}}",
          key = "{{_DEVICE_ID_}}",
          value = "{{audit_low_quality_pids}}"
        ) \
      .end_() \

    self.namespace_(ns = "retrieval_perf", nest = True)
    retrieval_perf_module = RetrievalPerfModule("retrieval_perf")
    retrieval_perf_module.set_flow(self)
    retrieval_perf_module.process()
    self.namespace_()
    
    # item attr 落盘
    self._dump_attr_to_kafka(
      stage_name = "retr", 
      dump_item_attr_list = [
        "i2i_retr__trigger_pid",
        "multi_retr_reason"
      ]
    ) \

    self \
      .pack_item_attr(
        item_source = {
          "reco_results": True,
        },
        mappings = [
          {
            "aggregator": "sum",
            "from_item_attr": "retrieval_dup_count",
            "default_val": 0,
            "to_common_attr": "retrieval_dup_count",
          }
        ]
      ) \
      .perflog_attr_value(
        check_point = "retrieval",
        common_attrs = [
          "retrieval_dup_count",
        ]
      )

    self._perf_result(
      attr_map = {
        "is_picture": ["pic", "count"],
        "is_support_author_picture": ["sp_aid_pic", "count"],
        "high_value_pic_flag": ["high_value_pic", "count"]
      },
      perf_sampling_attr = "_IS_PERF_SAMPLING_REQUEST_",
    )
    super()._flow_end()

  def similary_neg_feedback_limit(self):
    return self \
      .explore_embedding_candidates_attr_enricher(
        trans_type = "embedding_candidates",
        enable_fix_low_hit_rate = "{{enable_explore_retr_fix_low_hit_rate}}",
        user_info_ptr_attr = "user_info_ptr",
        export_common_attr = "embedding_source_pids" 
      ) \
      .get_remote_embedding_lite(
        kess_service = "{{explore_retr_emb_kess_name_for_similary_neg_feedback}}",
        shard_num = 4,
        timeout_ms = 25,
        id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
        input_attr_name = "embedding_source_pids",
        output_attr_name = "mmu_embeddings",
        query_source_type = "common_attr",
        size = 64,
        client_side_shard = True
      ) \
      .explore_custom_embedding_score_enricher(
        enable_fountain_version = "{{enable_collect_actionlist_version}}",
        enable_explore_truncate_topk = "{{enable_explore_truncate_topk}}",
        enable_extra_no_click_stat = "{{enable_explore_extra_no_click_stat}}",
        enable_fix_low_hit_rate = "{{enable_explore_retr_fix_low_hit_rate}}",
        enable_explore_hetu_topk = "{{enable_explore_hetu_topk}}",
        user_info_ptr_attr = "user_info_ptr",
        embedding_list_attr = "mmu_embeddings",
        source_pids_list_attr = "embedding_source_pids",
        calc_type = "action_bucket_dot",
        not_click_limit_hour = "{{explore_retr_hate_similary_score_not_click_hour_limit}}",
        play_stat_limit_hour = "{{explore_retr_hate_similary_score_play_stat_hour_limit}}",
        extra_not_click_limit_hour = "{{explore_retr_hate_similary_score_extra_not_click_hour_limit}}",
        not_click_limit_topk = "{{explore_retr_hate_similary_score_not_click_topk_limit}}",
        play_stat_limit_topk = "{{explore_retr_hate_similary_score_play_stat_topk_limit}}",
        not_hetu_limit_topk = "{{explore_retr_hate_similary_score_not_hetu_limit_topk}}",
        not_pid_limit_topk = "{{explore_retr_hate_similary_score_not_pid_limit_topk}}",
        not_click_weight = "{{explore_retr_similary_neg_score_not_click_weight}}",
        short_view_weight = "{{explore_retr_similary_neg_score_short_view_weight}}",
        extra_not_click_weight = "{{explore_retr_similary_neg_score_extra_not_click_weight}}",
        short_view_threshold = "{{explore_retr_similary_neg_score_short_view_threshold}}",
        enable_fountain_play_stat = "{{explore_retr_enable_fountain_play_stat}}",
        enable_judge_next_photo_stat = "{{enable_explore_retr_judge_next_photo_stat}}",
        enable_avg_pooling = "{{explore_retr_similary_neg_score_enable_avg_pooling}}",
        export_item_attr = "retr_similary_neg_score",
        dim_size = 64,
        check_point_ = "explore_retrieval",
      ) \
      .if_("explore_enable_similary_neg_feedback_filter_by_rank == 1") \
        .sort(
          score_from_attr = "retr_similary_neg_score",
          desc = False
        ) \
        .count_reco_result(
          save_count_to = "explore_retr_item_num"
        ) \
        .gen_common_attr_by_lua(
          attr_map = {
            "explore_similary_neg_feedback_limit_num": "math.floor(explore_retr_item_num * explore_similary_neg_feedback_limit_percent)"
          }
        ) \
        .limit(
          name = "explore_retr_similary_neg_filter_by_rank",
          traceback = True,
          size = "{{explore_similary_neg_feedback_limit_num}}"
        ) \
      .else_() \
        .filter_by_attr(
          name = "explore_retr_similary_neg_filter",
          traceback = True,
          attr_name = "retr_similary_neg_score",
          remove_if = ">=",
          compare_to = "{{explore_similary_neg_score_thres}}",
          remove_if_attr_missing = False,
        ) \
      .end_()

  def questionnaire_good_item_list(self):
    return self \
      .item_attr_operation(
        item_attr_a = "explore_questionnaire_info__exposure_count",
        common_attr_b = "{{explore_questionnaire_satisfaction_rate_denominator_bias}}",
        operator = "+",
        output_attr = "questionnaire_exposure_count_with_bias"
      ) \
      .item_attr_operation(
        item_attr_a = "explore_questionnaire_info__positive_count",
        item_attr_b = "questionnaire_exposure_count_with_bias",
        operator = "/",
        output_attr = "questionnaire_satisfaction_rate"
      ) \
      .pack_item_attr(
        item_source = {
          "reco_results": True,
          "total_limit" : "{{explore_retr_questionnaire_good_item_save_count}}",
        },
        mappings = [{
          "aggregator": "concat",
          "from_item_attr": "photo_id",
          "to_common_attr": "retrieval_questionnaire_good_input_item_key_list",
        }],
        select_item = {
          "join": "and",
          "filters": [{
            "attr_name": "explore_questionnaire_info__exposure_count",
            "compare_to": "{{explore_questionnaire_satisfaction_min_exposure_count}}",
            "select_if": ">",
            "select_if_attr_missing": False,
          }, {
            "attr_name": "questionnaire_satisfaction_rate",
            "compare_to": "{{explore_questionnaire_satisfaction_rate_threshold}}",
            "select_if": ">",
            "select_if_attr_missing": False,
          }],
        }
      )