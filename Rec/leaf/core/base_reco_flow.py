from dragonfly.common_leaf_dsl import LeafFlow

class BaseRecoFlow(LeafFlow):
  def __init__(self, name: str, config: dict = None, optimize_processor_order: bool = False) -> None:
    super().__init__(name, optimize_processor_order = optimize_processor_order)
    self._reco_stage = None

    self.__config = config

  @property
  def config(self) -> dict:
    return self.__config

  @property
  def reco_stage(self) -> dict:
    return self._reco_stage
 