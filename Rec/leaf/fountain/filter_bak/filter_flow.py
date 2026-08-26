from common import CommonRecoFlow
from dragonfly.ext.subdivision.subdivision_api_mixin import subdivisionApiMixin
from dragonfly.ext.explore.explore_api_mixin import ExploreApiMixin

class FilterFlow(CommonRecoFlow, subdivisionApiMixin, ExploreApiMixin):
  def __init__(self, name: str) -> None:
    super().__init__(name, "fountain", "filter", "config", "module", "config/module", "lua/module")

  def _flow_end(self):
    self._perf_result(
      perf_sampling_attr = "_IS_PERF_SAMPLING_REQUEST_",
    )
    super()._flow_end()
