from cascading import CommonModule
from cascading.cascade_util import hot_sim_fc_features

class CascadingFinalSortPredictModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    """leave empty function by AutoDelete"""

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = ["enable_hot_mc_fc_s2_predict"],
        for_debug_request_only = True
      )