from common import CommonRecoFlow
from .ranking_strategy_mixin import ExploreRankingStrategyMixin
from dragonfly.ext.subdivision.subdivision_api_mixin import subdivisionApiMixin
from dragonfly.ext.explore.explore_api_mixin import ExploreApiMixin
from dragonfly.ext.explore_life.explore_life_api_mixin import ExploreLifeApiMixin
from dragonfly.ext.retrieval.retrieval_api_mixin import RetrievalApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from dragonfly.ext.merchant.merchant_api_mixin import MerchantApiMixin


class RankingFlow(CommonRecoFlow, subdivisionApiMixin, RetrievalApiMixin, GsuApiMixin, ExploreRankingStrategyMixin, ExploreApiMixin, ExploreLifeApiMixin, MerchantApiMixin):
  def __init__(self, name: str) -> None:
    super().__init__(name, "life", "ranking", "config", "module", "config/module", "lua/module", optimize_processor_order = True)

  def _flow_end(self):
    self._perf_result(
      attr_map = {
        "is_picture": ["pic", "count"],
        "is_follow_author": ["follow_author", "count"],
        "shuffle_policy": ["shuffle", "count"],
        "content_safety_level_with_namespace__level_hot_online": ["", "value_count"],
        "topk_audit_level": ["", "value_count"],
        "audit_hot_high_tag_level": ["", "value_count"],
        "audit_hot_cover_level": ["", "value_count"],
        "is_support_author_picture": ["sp_aid_pic", "count"],
        "high_value_pic_flag": ["high_value_pic", "count"],
        "is_explore_photo": ["explore", "count"],
        "is_high_quality_explore_photo": ["high_quality_explore", "count"]
      },
      perf_sampling_attr = "_IS_PERF_SAMPLING_REQUEST_",
    )
    super()._flow_end()
