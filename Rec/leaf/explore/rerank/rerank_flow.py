from common import CommonRecoFlow
from dragonfly.ext.subdivision.subdivision_api_mixin import subdivisionApiMixin
from dragonfly.ext.explore.explore_api_mixin import ExploreApiMixin

class RerankFlow(CommonRecoFlow, subdivisionApiMixin, ExploreApiMixin):
  def __init__(self, name: str) -> None:
    super().__init__(name, "explore", "rerank", "config", "module", "config/module", "lua/module", optimize_processor_order = True)

  def _flow_begin(self):
    super()._flow_begin()
    self.get_abtest_params(
      biz_name = "RECO_RPC",
      ab_params = [
        ("dpp_diversity_candidate_size", 60, "rerank_candidate_size")
      ],
    )
    self._perf_result(
      step_name = "candidate",
      attr_map = {
        "is_picture": ["pic", "count"],
        "is_follow_author": ["follow_author", "count"],
        "shuffle_policy": ["shuffle", "count"],
        "content_safety_level_with_namespace__level_hot_online": ["", "value_count"],
        "topk_audit_level": ["", "value_count"],
        "audit_hot_high_tag_level": ["", "value_count"],
        "audit_hot_cover_level": ["", "value_count"],
        "is_support_author_picture": ["sp_aid_pic", "count"],
        "high_value_pic_flag": ["high_value_pic", "count"]
      },
      range_end = "{{rerank_candidate_size}}",
      perf_sampling_attr = "_IS_PERF_SAMPLING_REQUEST_",
    )

  def _flow_end(self):
    self.get_abtest_params(
      biz_name = "RECO_RPC",
      ab_params = [
        ("dpp_diversity_list_size", 10, "rerank_result_size")
      ],
    )
    self._perf_result(
      attr_map = {
        "is_picture": ["pic", "count"],
        "is_follow_author": ["follow_author", "count"],
        "shuffle_policy": ["shuffle", "count"],
        "content_safety_level_with_namespace__level_hot_online": ["", "value_count"],
        "topk_audit_level": ["", "value_count"],
        "audit_hot_high_tag_level": ["", "value_count"],
        "audit_hot_cover_level": ["", "value_count"],
        "audit_b_second_tag" : ["", "value_count"],
        "is_support_author_picture": ["sp_aid_pic", "count"],
        "high_value_pic_flag": ["high_value_pic", "count"],
        "is_explore_photo": ["explore", "count"],
        "is_high_quality_explore_photo": ["high_quality_explore", "count"]
      },
      range_end = "{{rerank_result_size}}",
      perf_sampling_attr = "_IS_PERF_SAMPLING_REQUEST_",
    )
    super()._flow_end()
