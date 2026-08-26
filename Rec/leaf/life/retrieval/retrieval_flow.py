from dragonfly.ext.explore.explore_api_mixin import ExploreApiMixin
from dragonfly.ext.explore_life.explore_life_api_mixin import ExploreLifeApiMixin
from dragonfly.ext.retrieval.retrieval_api_mixin import RetrievalApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from dragonfly.ext.embed_calc.embed_calc_api_mixin import EmbedCalcApiMixin
from dragonfly.ext.kgnn.kgnn_api_mixin import KgnnApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin

from common import CommonRecoFlow
from retrieval.module.retrieval_perf_module import RetrievalPerfModule
from retrieval.module.photo_info_fetching_module import PhotoInfoFetchingModule

class RetrievalFlow(CommonRecoFlow, ExploreApiMixin, ExploreLifeApiMixin, RetrievalApiMixin, GsuApiMixin, EmbedCalcApiMixin, KgnnApiMixin, KuibaApiMixin):
  def __init__(self, name: str, is_sub_flow: bool = False) -> None:
    super().__init__(name, "life", "retrieval", "config", "module", "config/module", "lua/module", is_sub_flow)

  def _flow_end(self):

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
            "attr_name": "explore_retr_reason_limit_reason_weight_str",
            "default_value": "",
            "param_name": "explore_retr_reason_limit_reason_weight_str",
            "param_type": "string",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "life_new_user_retr_reason_limit_reason_weight_str",
            "default_value": "",
            "param_name": "life_new_user_retr_reason_limit_reason_weight_str",
            "param_type": "string",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
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
            "attr_name": "skip_explore_personal_quota",
            "default_value": 1,
            "param_name": "skip_explore_personal_quota",
            "param_type": "int",
            "report_ab_hit": "{{_ABTEST_REPORT_HIT_}}"
          },
          {
            "attr_name": "explore_retr_reason_limit_skip_fillback_result",
            "default_value": False,
            "param_name": "explore_retr_reason_limit_skip_fillback_result",
            "param_type": "bool",
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
            "attr_name": "enable_life_tnu_retr_quota_limit",
            "default_value": False,
            "param_name": "enable_life_tnu_retr_quota_limit",
            "param_type": "bool",
          },
          {
            "attr_name": "enable_life_new_user_first_page_project",
            "default_value": False,
            "param_name": "enable_life_new_user_first_page_project",
            "param_type": "bool",
          },
          {
            "attr_name": "enable_life_tnu_retr_strict_limit",
            "default_value": True,
            "param_name": "enable_life_tnu_retr_strict_limit",
            "param_type": "bool",
          },
          {
            "attr_name": "life_tnu_retr_reason_limit_num",
            "default_value": 10000,
            "param_name": "life_tnu_retr_reason_limit_num",
            "param_type": "int",
          },
          {
            "attr_name": "life_tnu_retr_reason_limit_reason_weight_str",
            "default_value": "2800:0.75;4050:0.125;3131:0.125",
            "param_name": "life_tnu_retr_reason_limit_reason_weight_str",
            "param_type": "string",
          },
          {
            "attr_name": "life_tnu_retr_reason_limit_reason_weight_str_strict",
            "default_value": "2800:1.0",
            "param_name": "life_tnu_retr_reason_limit_reason_weight_str_strict",
            "param_type": "string",
          },
          {
            "attr_name": "life_tnu_retr_reason_limit_reason_str",
            "default_value": "2800",
            "param_name": "life_tnu_retr_reason_limit_reason_str",
            "param_type": "string",
          },
          {
            "attr_name": "life_tnu_nice_photo_result_num_thr",
            "default_value": "2000",
            "param_name": "life_tnu_nice_photo_result_num_thr",
            "param_type": "int",
          }
        ],
        prioritized_suffix = "{{_ABTEST_SUFFIX_LIST_}}"
      ) \
      .if_("enable_explore_xhs_install_adjust_retr_num == 1 and is_la_correct_user == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_retr_reason_limit_num", "as": "value"},
            {"name": "enable_explore_xhs_install_adjust_retr_boost_weight", "as": "weight"},
          ],
          export_common_attr = [
            {"name": "new_value", "as": "explore_retr_reason_limit_num"},
          ],
          function_name = "CalExploreIntMultiDouble",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
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
      .if_("enable_life_tnu_retr_quota_limit ~= 1 or uIsTnuCrowdUser ~= 1 or request_type ~= \"life\"") \
        .deduplicate(
          name = "explore_retr_dedup",
          traceback = True,
          save_dup_count_to = "retrieval_dup_count",
          append_reason_to = "multi_retr_reason"
        ) \
      .end_() \
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
      .filter_by_common_attr(
        common_attr = ["cascade_neg_photo_id_list"]
      ) \
      .if_("dup_pid_list ~= nil and #dup_pid_list > 0") \
        .filter_by_common_attr(
          common_attr = ["dup_pid_list"]
        ) \
      .end_() \
      .if_("enable_second_tab < 1") \
        .if_("enable_life_tnu_retr_quota_limit == 1 and uIsTnuCrowdUser == 1 and request_type == \"life\"") \
          .split_string(
            input_common_attr = "life_tnu_retr_reason_limit_reason_str",
            output_common_attr = "life_tnu_retr_reason_limit_reason_list",
            delimiters = ",",
            trim_spaces = True,
            skip_empty_tokens = True,
            parse_to_int = True
          ) \
          .copy_item_meta_info(
            save_reason_to_attr="reason",
          ) \
          .count_reco_result(
            save_count_to="nice_photo_result_num",
            target_item = {
              "reason": "{{life_tnu_retr_reason_limit_reason_list}}"
            }
          ) \
          .if_("enable_life_tnu_retr_strict_limit == 1 and nice_photo_result_num >= life_tnu_nice_photo_result_num_thr") \
            .copy_attr(
              attrs = [{
                "from_common": "life_tnu_retr_reason_limit_reason_weight_str_strict",
                "to_common": "life_tnu_retr_reason_limit_reason_weight_str",
              }]
            ) \
          .end_() \
          .explore_retr_reason_limit(
            name = "life_tnu_retr_quota_limit",
            traceback = True,
            size_limit = "{{life_tnu_retr_reason_limit_num}}",
            default_weight = 0.0,
            reason_weight_str = "{{life_tnu_retr_reason_limit_reason_weight_str}}",
            skip_fillback_result = True
          ) \
          .deduplicate(
            name = "explore_retr_dedup_tnu",
            traceback = True,
            save_dup_count_to = "retrieval_dup_count",
            append_reason_to = "multi_retr_reason"
          ) \
        .else_if_("enable_life_new_user_first_page_project == 1 and (((uNebulaXlifeVisitDays30dKV or 0)+(uNebulaDoubleFindVisitDays30dKV or 0)) <= 1) and page == 1 and request_type == \"life\"") \
          .explore_retr_reason_limit(
            name = "life_new_user_first_page_retr_quota_limit",
            traceback = True,
            size_limit = "{{explore_retr_reason_limit_num}}",
            default_weight = 1.0,
            reason_weight_str = "{{life_new_user_retr_reason_limit_reason_weight_str}}",
            personal_weight_map_attr = "reason_ratio_map_attr",
            skip_personal_weight = "{{skip_explore_personal_quota}}",
            skip_fillback_result = "{{explore_retr_reason_limit_skip_fillback_result}}"
          ) \
        .else_() \
          .explore_retr_reason_limit(
            name = "explore_retr_quota_limit",
            traceback = True,
            size_limit = "{{explore_retr_reason_limit_num}}",
            default_weight = 1.0,
            reason_weight_str = "{{explore_retr_reason_limit_reason_weight_str}}",
            personal_weight_map_attr = "reason_ratio_map_attr",
            skip_personal_weight = "{{skip_explore_personal_quota}}",
            skip_fillback_result = "{{explore_retr_reason_limit_skip_fillback_result}}"
          ) \
        .end_() \
        .log_debug_info(
          common_attrs = ["uNebulaXlifeVisitDays30dKV", "uNebulaDoubleFindVisitDays30dKV", "page", "refreshTimes", "request_type"],
          for_debug_request_only = True,
        ) \
      .end_() \

    self.namespace_(ns = "photo_info", nest = True)
    photo_info_module = PhotoInfoFetchingModule("photo_info")
    photo_info_module.set_flow(self)
    photo_info_module.process()
    self.namespace_()

    self \
      .if_("explore_enable_dedup_on_author_id == 1") \
        .deduplicate(
          on_item_attr = "author__id",
        ) \
      .end_() \
      .if_("explore_enable_write_audit_low_quality_pids_to_redis == 1") \
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
