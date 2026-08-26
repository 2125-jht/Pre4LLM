from common import CommonRecoFlow
from .cascade_strategy_mixin import ExploreCascadeStrategyMixin
from dragonfly.ext.subdivision.subdivision_api_mixin import subdivisionApiMixin
from dragonfly.ext.explore.explore_api_mixin import ExploreApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.retrieval.retrieval_api_mixin import RetrievalApiMixin
from dragonfly.ext.embedding.embedding_api_mixin import EmbeddingApiMixin

class CascadingFlow(CommonRecoFlow, ExploreApiMixin, subdivisionApiMixin, GsuApiMixin, MioApiMixin, RetrievalApiMixin, ExploreCascadeStrategyMixin, EmbeddingApiMixin):
  def __init__(self, name: str) -> None:
    super().__init__(name, "explore", "cascading", "config", "module", "config/module", "lua/module", optimize_processor_order = True)

  def _flow_begin(self):
    self.if_("enable_explore_cascading_v2_flow == 0")
    super()._flow_begin()

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
    ) \
    .copy_item_meta_info(
      save_item_seq_to_attr = "cascade_final_index",
    )

    super()._flow_end()
    self.end_()
