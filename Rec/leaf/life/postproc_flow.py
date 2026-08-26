from dragonfly.common_leaf_dsl import LeafFlow
from dragonfly.ext.explore.explore_api_mixin import ExploreApiMixin

class PostprocFlow(LeafFlow, ExploreApiMixin):
  def __init__(self, name: str, return_common_attrs: list, return_item_attrs: list, traceback_item_attrs: list):
    super().__init__(name)

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
      .gen_common_attr_by_lua(
        attr_map = {
          "explore_reco_leaf_total_ts": "util.GetTimestamp() - prepare_begin_ts",
          "explore_reco_leaf_request_count": "1",
        },
      ) \
      .send_abtest_metrics(
        skip = "{{return _IS_ONLINE_SERVICE_ == 0}}",
        metrics = [
          "explore_reco_leaf_retrieval_ts",
          "explore_reco_leaf_filter_ts",
          "explore_reco_leaf_cascading_ts",
          "explore_reco_leaf_ranking_ts",
          "explore_reco_leaf_rerank_ts",
          "explore_reco_leaf_total_ts",
          "explore_reco_leaf_request_count",
          "explore_reco_leaf_rank_model_input_count",
          "explore_reco_leaf_cascade_model_pic_input_result_count",
          "explore_reco_leaf_cascade_model_input_result_count",
          "explore_reco_leaf_retrieval_cpu_cost_ts",
          "explore_reco_leaf_filter_cpu_cost_ts",
          "explore_reco_leaf_cascading_cpu_cost_ts",
          "explore_reco_leaf_ranking_cpu_cost_ts",
          "explore_reco_leaf_rerank_cpu_cost_ts",
          "explore_reco_leaf_total_cpu_cost_ts",
        ],
        metric_name_prefix = "",
      ) \
      .log_debug_info(  # 为支持多 request type ，由上游指定 return attrs ，这里是对依赖检测的一个 tricky
        common_attrs = return_common_attrs,
        item_attrs = return_item_attrs + traceback_item_attrs,
        for_debug_request_only = True,
      ) \
      .namespace_()
