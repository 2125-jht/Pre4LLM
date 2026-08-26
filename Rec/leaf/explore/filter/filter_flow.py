from common import CommonRecoFlow
from dragonfly.ext.explore.explore_api_mixin import ExploreApiMixin

class FilterFlow(CommonRecoFlow, ExploreApiMixin):
  def __init__(self, name: str) -> None:
    super().__init__(name, "explore", "filter", "config", "module", "config/module", "lua/module")

  def _flow_end(self):
    self._perf_result(
      attr_map = {
        "is_picture": ["pic", "count"],
        "is_support_author_picture": ["sp_aid_pic", "count"],
        "high_value_pic_flag": ["high_value_pic", "count"]
      },
      perf_sampling_attr = "_IS_PERF_SAMPLING_REQUEST_",
    )
    super()._flow_end()
