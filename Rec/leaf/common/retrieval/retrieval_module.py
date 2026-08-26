from common import JsonConfigRecoModule

class RetrievalModule(JsonConfigRecoModule):
  def __init__(self, name: str, config_dir: str) -> None:
    super().__init__(name, config_dir)
    assert "dragon_params" in self.config and \
      ("abtest_retrieval_switch_attr" in self.config["dragon_params"] or \
      ("abtest" in self.config["dragon_params"] and "enable_retrieval" in self.config["dragon_params"]["abtest"])), \
      "召回 module 必须包含 \"enable_retrieval\" 开关"

  @classmethod
  def is_retrieval(cls) -> bool:
    return True

  @property
  def reason(self) -> int:
    assert "reason" in self.config
    return self.config["reason"]
