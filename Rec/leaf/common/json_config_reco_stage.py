import importlib
import json
import os
from core import BaseRecoStage

class JsonConfigRecoStage(BaseRecoStage):
  def __init__(self, name: str, service_name: str, config_path: str, module_dir: str, module_config_dir: str) -> None:
    with open(config_path, "r") as f:
      config = json.load(f)
    common_module_dir = os.path.join("common", name, module_dir)
    if os.path.exists(common_module_dir):
      common_modules = importlib.import_module(common_module_dir.replace("/", "."))
    service_module_dir = os.path.join(service_name, name, module_dir)
    service_modules = importlib.import_module(service_module_dir.replace("/", "."))
    if "modules" in config:
      module_config_items = config.pop("modules")
      assert isinstance(module_config_items, list)
      config["modules"] = []
      for item in module_config_items:
        if not isinstance(item, dict):
          continue
        assert "name" in item
        assert "type_name" in item
        module_class = getattr(service_modules, item["type_name"], None)
        if module_class:
          module = module_class(item["name"])
        elif common_modules:
          final_module_config_dir = os.path.join(service_name, name, module_config_dir)
          module_class = getattr(common_modules, item["type_name"])
          module = module_class(item["name"], final_module_config_dir)

        assert module
        config["modules"].append(module)

    super().__init__(name, config)
