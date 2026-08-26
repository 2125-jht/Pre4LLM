from dragonfly.common_leaf_dsl import LeafFlow
from dragonfly.ext.explore.explore_api_mixin import ExploreApiMixin

class PostprocFlow(LeafFlow, ExploreApiMixin):
  def __init__(self, name: str, return_common_attrs: list, return_item_attrs: list, traceback_item_attrs: list):
    super().__init__(name)

    calc_total_cpu_cost = \
      "explore_reco_leaf_retrieval_default_cpu_cost_ts + " + \
      "explore_reco_leaf_filter_default_cpu_cost_ts + " + \
      "explore_reco_leaf_cascading_default_cpu_cost_ts + " + \
      "explore_reco_leaf_ranking_default_cpu_cost_ts + " + \
      "explore_reco_leaf_rerank_default_cpu_cost_ts"

    cpu_cost_debug_info = [
      "explore_reco_leaf_retrieval_default_cpu_cost_ts", 
      "explore_reco_leaf_filter_default_cpu_cost_ts",
      "explore_reco_leaf_cascading_default_cpu_cost_ts",
      "explore_reco_leaf_ranking_default_cpu_cost_ts",
      "explore_reco_leaf_rerank_default_cpu_cost_ts",
      "explore_reco_leaf_total_cpu_cost_ts",
    ]

    self \
      .namespace_(ns = name, nest = True) \
      .get_abtest_params(
        biz_name = "RECO_RPC",
        ab_params = [
          ("skip_explore_leaf_sleep", 1, "skip_explore_leaf_sleep"),
          ("explore_leaf_sleep_ms", 0, "explore_leaf_sleep_ms")
        ],
      ) \
      .sleep(
        sleep_ms = "{{explore_leaf_sleep_ms}}",
        skip = "{{skip_explore_leaf_sleep}}"
      ) \
      .if_("enable_explore_cascading_v2_flow == 1") \
        .copy_attr(
          attrs = [
            {"from_common": "explore_reco_leaf_cascading_v2_default_ts", "to_common": "explore_reco_leaf_cascading_default_ts"},
            {"from_common": "explore_reco_leaf_cascading_v2_default_cpu_cost_ts", "to_common": "explore_reco_leaf_cascading_default_cpu_cost_ts"},
          ]
        ) \
      .end_() \
      .gen_common_attr_by_lua(
        attr_map = {
          "explore_reco_leaf_total_ts": "util.GetTimestamp() - prepare_begin_ts",
          "explore_reco_leaf_request_count": "1",
          "explore_reco_leaf_total_cpu_cost_ts": calc_total_cpu_cost,
        },
      ) \
      .send_abtest_metrics(
        skip = "{{return _IS_ABTEST_METRICS_SAMPLING_REQUEST_ == 0 or _IS_ONLINE_SERVICE_ == 0 or _IS_NOT_BACKUP_ == 0}}",
        metrics = [
          {"name": "retrieval_origin_result_count", "as": "explore_reco_leaf_retrieval_origin_result_count"},
          {"name": "filter_result_count", "as": "explore_reco_leaf_filter_result_count"},
          {"name": "explore_reco_leaf_retrieval_default_ts", "as": "explore_reco_leaf_retrieval_ts"},
          {"name": "explore_reco_leaf_filter_default_ts", "as": "explore_reco_leaf_filter_ts"},
          {"name": "explore_reco_leaf_cascading_default_ts", "as": "explore_reco_leaf_cascading_ts"},
          {"name": "explore_reco_leaf_ranking_default_ts", "as": "explore_reco_leaf_ranking_ts"},
          {"name": "explore_reco_leaf_rerank_default_ts", "as": "explore_reco_leaf_rerank_ts"},
          "explore_reco_leaf_total_ts",
          {"name": "explore_reco_leaf_retrieval_default_cpu_cost_ts", "as": "explore_reco_leaf_retrieval_cpu_cost_ts"},
          {"name": "explore_reco_leaf_filter_default_cpu_cost_ts", "as": "explore_reco_leaf_filter_cpu_cost_ts"},
          {"name": "explore_reco_leaf_cascading_default_cpu_cost_ts", "as": "explore_reco_leaf_cascading_cpu_cost_ts"},
          {"name": "explore_reco_leaf_ranking_default_cpu_cost_ts", "as": "explore_reco_leaf_ranking_cpu_cost_ts"},
          {"name": "explore_reco_leaf_rerank_default_cpu_cost_ts", "as": "explore_reco_leaf_rerank_cpu_cost_ts"},
          "explore_reco_leaf_total_cpu_cost_ts",
          "explore_reco_leaf_request_count",
          "explore_reco_leaf_rank_model_input_count",
          "explore_reco_leaf_cascade_model_pic_input_result_count",
          "explore_reco_leaf_cascade_model_input_result_count",
          { "name": "enable_explore_pic_cluster_counter", "as": "explore_reco_leaf_pic_cluster_counter_hit_flag" },
          { "name": "retr_pic_has_recent_search", "as": "explore_reco_leaf_has_recent_search" },
          { "name": "retr_pic_count", "as": "explore_reco_leaf_retr_pic_count" },
          { "name": "retr_pic_hetu_count", "as": "explore_reco_leaf_retr_pic_hetu_count" },
          { "name": "retr_pic_short_term_interest_count", "as": "explore_reco_leaf_retr_pic_short_term_interest_count" },
          { "name": "retr_pic_long_term_interest_count", "as": "explore_reco_leaf_retr_pic_long_term_interest_count" },
          { "name": "retr_pic_explore_interest_count", "as": "explore_reco_leaf_retr_pic_explore_interest_count" },
          { "name": "retr_pic_unknown_interest_count", "as": "explore_reco_leaf_retr_pic_unknown_interest_count" },
          { "name": "retr_pic_cluster_count", "as": "explore_reco_leaf_retr_pic_cluster_count" },
          { "name": "retr_pic_long_interest_count", "as": "explore_reco_leaf_retr_pic_long_interest_count" },
          { "name": "retr_pic_valid_interest_count", "as": "explore_reco_leaf_retr_pic_valid_interest_count" },
          { "name": "retr_pic_single_valid_interest_count", "as": "explore_reco_leaf_retr_pic_single_valid_interest_count" },
          { "name": "retr_pic_double_valid_interest_count", "as": "explore_reco_leaf_retr_pic_double_valid_interest_count" },
          { "name": "retr_pic_recent_search_interest_count", "as": "explore_reco_leaf_retr_pic_recent_search_interest_count" },
          { "name": "prerank_pic_count", "as": "explore_reco_leaf_prerank_pic_count" },
          { "name": "prerank_pic_hetu_count", "as": "explore_reco_leaf_prerank_pic_hetu_count" },
          { "name": "prerank_pic_short_term_interest_count", "as": "explore_reco_leaf_prerank_pic_short_term_interest_count" },
          { "name": "prerank_pic_long_term_interest_count", "as": "explore_reco_leaf_prerank_pic_long_term_interest_count" },
          { "name": "prerank_pic_explore_interest_count", "as": "explore_reco_leaf_prerank_pic_explore_interest_count" },
          { "name": "prerank_pic_unknown_interest_count", "as": "explore_reco_leaf_prerank_pic_unknown_interest_count" },
          { "name": "prerank_pic_cluster_count", "as": "explore_reco_leaf_prerank_pic_cluster_count" },
          { "name": "prerank_pic_long_interest_count", "as": "explore_reco_leaf_prerank_pic_long_interest_count" },
          { "name": "prerank_pic_valid_interest_count", "as": "explore_reco_leaf_prerank_pic_valid_interest_count" },
          { "name": "prerank_pic_single_valid_interest_count", "as": "explore_reco_leaf_prerank_pic_single_valid_interest_count" },
          { "name": "prerank_pic_double_valid_interest_count", "as": "explore_reco_leaf_prerank_pic_double_valid_interest_count" },
          { "name": "prerank_pic_recent_search_interest_count", "as": "explore_reco_leaf_prerank_pic_recent_search_interest_count" },
          { "name": "mc_s1_pic_count", "as": "explore_reco_leaf_mc_s1_pic_count" },
          { "name": "mc_s1_pic_hetu_count", "as": "explore_reco_leaf_mc_s1_pic_hetu_count" },
          { "name": "mc_s1_pic_short_term_interest_count", "as": "explore_reco_leaf_mc_s1_pic_short_term_interest_count" },
          { "name": "mc_s1_pic_long_term_interest_count", "as": "explore_reco_leaf_mc_s1_pic_long_term_interest_count" },
          { "name": "mc_s1_pic_explore_interest_count", "as": "explore_reco_leaf_mc_s1_pic_explore_interest_count" },
          { "name": "mc_s1_pic_unknown_interest_count", "as": "explore_reco_leaf_mc_s1_pic_unknown_interest_count" },
          { "name": "mc_s1_pic_cluster_count", "as": "explore_reco_leaf_mc_s1_pic_cluster_count" },
          { "name": "mc_s1_pic_long_interest_count", "as": "explore_reco_leaf_mc_s1_pic_long_interest_count" },
          { "name": "mc_s1_pic_valid_interest_count", "as": "explore_reco_leaf_mc_s1_pic_valid_interest_count" },
          { "name": "mc_s1_pic_single_valid_interest_count", "as": "explore_reco_leaf_mc_s1_pic_single_valid_interest_count" },
          { "name": "mc_s1_pic_double_valid_interest_count", "as": "explore_reco_leaf_mc_s1_pic_double_valid_interest_count" },
          { "name": "mc_s1_pic_recent_search_interest_count", "as": "explore_reco_leaf_mc_s1_pic_recent_search_interest_count" },
          { "name": "mc_s2_pic_count", "as": "explore_reco_leaf_mc_s2_pic_count" },
          { "name": "mc_s2_pic_hetu_count", "as": "explore_reco_leaf_mc_s2_pic_hetu_count" },
          { "name": "mc_s2_pic_short_term_interest_count", "as": "explore_reco_leaf_mc_s2_pic_short_term_interest_count" },
          { "name": "mc_s2_pic_long_term_interest_count", "as": "explore_reco_leaf_mc_s2_pic_long_term_interest_count" },
          { "name": "mc_s2_pic_explore_interest_count", "as": "explore_reco_leaf_mc_s2_pic_explore_interest_count" },
          { "name": "mc_s2_pic_unknown_interest_count", "as": "explore_reco_leaf_mc_s2_pic_unknown_interest_count" },
          { "name": "mc_s2_pic_cluster_count", "as": "explore_reco_leaf_mc_s2_pic_cluster_count" },
          { "name": "mc_s2_pic_long_interest_count", "as": "explore_reco_leaf_mc_s2_pic_long_interest_count" },
          { "name": "mc_s2_pic_valid_interest_count", "as": "explore_reco_leaf_mc_s2_pic_valid_interest_count" },
          { "name": "mc_s2_pic_single_valid_interest_count", "as": "explore_reco_leaf_mc_s2_pic_single_valid_interest_count" },
          { "name": "mc_s2_pic_double_valid_interest_count", "as": "explore_reco_leaf_mc_s2_pic_double_valid_interest_count" },
          { "name": "mc_s2_pic_recent_search_interest_count", "as": "explore_reco_leaf_mc_s2_pic_recent_search_interest_count" },
          { "name": "rank_pic_count", "as": "explore_reco_leaf_rank_pic_count" },
          { "name": "rank_pic_hetu_count", "as": "explore_reco_leaf_rank_pic_hetu_count" },
          { "name": "rank_pic_short_term_interest_count", "as": "explore_reco_leaf_rank_pic_short_term_interest_count" },
          { "name": "rank_pic_long_term_interest_count", "as": "explore_reco_leaf_rank_pic_long_term_interest_count" },
          { "name": "rank_pic_explore_interest_count", "as": "explore_reco_leaf_rank_pic_explore_interest_count" },
          { "name": "rank_pic_unknown_interest_count", "as": "explore_reco_leaf_rank_pic_unknown_interest_count" },
          { "name": "rank_pic_cluster_count", "as": "explore_reco_leaf_rank_pic_cluster_count" },
          { "name": "rank_pic_long_interest_count", "as": "explore_reco_leaf_rank_pic_long_interest_count" },
          { "name": "rank_pic_valid_interest_count", "as": "explore_reco_leaf_rank_pic_valid_interest_count" },
          { "name": "rank_pic_single_valid_interest_count", "as": "explore_reco_leaf_rank_pic_single_valid_interest_count" },
          { "name": "rank_pic_double_valid_interest_count", "as": "explore_reco_leaf_rank_pic_double_valid_interest_count" },
          { "name": "rank_pic_recent_search_interest_count", "as": "explore_reco_leaf_rank_pic_recent_search_interest_count" },
          { "name": "rerank_pic_count", "as": "explore_reco_leaf_rerank_pic_count" },
          { "name": "rerank_pic_hetu_count", "as": "explore_reco_leaf_rerank_pic_hetu_count" },
          { "name": "rerank_pic_short_term_interest_count", "as": "explore_reco_leaf_rerank_pic_short_term_interest_count" },
          { "name": "rerank_pic_long_term_interest_count", "as": "explore_reco_leaf_rerank_pic_long_term_interest_count" },
          { "name": "rerank_pic_explore_interest_count", "as": "explore_reco_leaf_rerank_pic_explore_interest_count" },
          { "name": "rerank_pic_unknown_interest_count", "as": "explore_reco_leaf_rerank_pic_unknown_interest_count" },
          { "name": "rerank_pic_cluster_count", "as": "explore_reco_leaf_rerank_pic_cluster_count" },
          { "name": "rerank_pic_long_interest_count", "as": "explore_reco_leaf_rerank_pic_long_interest_count" },
          { "name": "rerank_pic_valid_interest_count", "as": "explore_reco_leaf_rerank_pic_valid_interest_count" },
          { "name": "rerank_pic_single_valid_interest_count", "as": "explore_reco_leaf_rerank_pic_single_valid_interest_count" },
          { "name": "rerank_pic_double_valid_interest_count", "as": "explore_reco_leaf_rerank_pic_double_valid_interest_count" },
          { "name": "rerank_pic_has_recent_search_interest_realshow", "as": "explore_reco_leaf_rerank_pic_has_recent_search_interest_realshow" },
        ],
        metric_name_prefix = "",
      ) \
      .log_debug_info(  # 为支持多 request type ，由上游指定 return attrs ，这里是对依赖检测的一个 tricky
        common_attrs = return_common_attrs + cpu_cost_debug_info,
        item_attrs = return_item_attrs + traceback_item_attrs,
        for_debug_request_only = True,
      ) \
      .namespace_()
