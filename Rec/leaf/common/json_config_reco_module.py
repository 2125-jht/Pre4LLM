import json
import os
from core import BaseRecoModule

class JsonConfigRecoModule(BaseRecoModule):
  def __init__(self, name: str, config_dir: str) -> None:
    config_path = os.path.join(config_dir, name + ".json")
    assert os.path.exists(config_path)
    with open(config_path, "r") as f:
      config = json.load(f)

    if self.is_retrieval():
      if "dragon_params" in config and "abtest_retrieval_switch_attr" not in config["dragon_params"]:
        config["dragon_params"]["abtest_retrieval_switch_attr"] = "enable_retrieval"
      if "dragon_params" in config and "abtest_retrieval_num_attr" not in config["dragon_params"]:
        config["dragon_params"]["abtest_retrieval_num_attr"] = "_ABTEST_RETRIEVAL_LIMIT_NUM_"

    # 每个 module 可以配置自己的 suffix list ，若未配置则统一配置
    if "dragon_params" in config and "abtest" in config["dragon_params"] \
        and "abtest_suffix_list_attr" not in config["dragon_params"]:
      config["dragon_params"]["abtest_suffix_list_attr"] = "_ABTEST_SUFFIX_LIST_"

    # 每个 module 可以配置自己的染色上报，若未配置则统一配置，只有当 module 的染色上报和参数自身的染色上报同时为真时，才会实际上报
    if "dragon_params" in config and "abtest" in config["dragon_params"] \
        and "abtest_report_hit_attr" not in config["dragon_params"]:
      config["dragon_params"]["abtest_report_hit_attr"] = "_ABTEST_REPORT_HIT_"

    super().__init__(name, config)
