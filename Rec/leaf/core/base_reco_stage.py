from abc import ABC
from core.base_reco_module import BaseRecoModule
from core.base_reco_flow import BaseRecoFlow

class BaseRecoStage(ABC):
  RETRIEVAL_SWITCH_PREFIX = "enable_"

  __retrieval_flow_map = dict()

  def __init__(self, name: str, config: dict) -> None:
    self.__name = name
    self.__config = config
    self.__modules = []
    assert "modules" in self.__config
    module_items = self.__config["modules"]
    assert isinstance(module_items, list)
    for module_item in module_items:
      assert issubclass(type(module_item), BaseRecoModule)
      module_item.stage_name = name
      self.add_module(module_item)

  @property
  def name(self) -> str:
    return self.__name

  @property
  def config(self) -> dict:
    return self.__config

  @property
  def modules(self) -> list:
    return self.__modules

  def add_module(self, module: BaseRecoModule) -> None:
    self.__modules.append(module)

  def append_flow(self, flow: BaseRecoFlow) -> None:
    topological_modules = self.__get_topological_modules()
    topological_module_count = 0
    for level_modules in topological_modules:
      topological_module_count += len(level_modules)
    assert topological_module_count == len(self.__modules), "请检查 module 之间的 import ，export 的依赖关系"
    flow.namespace_(ns = flow.name, nest = True)

    self.__append_retrieval_params_processor(flow, topological_modules)
    for level_modules in topological_modules:
      for module in level_modules:
        if not module.is_retrieval():
          flow.namespace_(ns = module.name, nest = True)
          module.set_flow(flow)
          module.append_abtest_params_processor()
          module.append_kconf_params_processor()
          module.append_constant_params_processor()
          flow.namespace_()

    for level_modules in topological_modules:
      for module in level_modules:
        if module.is_retrieval():
          if module.name in self.__retrieval_flow_map:
            sub_flow = self.__retrieval_flow_map[module.name]
          else:
            sub_flow = type(flow)(name = module.name, is_sub_flow = True)
            sub_flow.namespace_(ns = module.name, nest = True)
            module.set_flow(sub_flow)
            module.append_abtest_params_processor()
            module.append_kconf_params_processor()
            module.append_constant_params_processor()
            module.read_retrieval_cache()
            module.process()
            module.post_process()
            module.write_retrieval_cache()
            sub_flow.namespace_()
            self.__retrieval_flow_map[module.name] = sub_flow

          if module.dragon_retrieval_switch:
            retrieval_switch = BaseRecoStage.RETRIEVAL_SWITCH_PREFIX + module.name
            flow.if_(retrieval_switch + " > 0", retrieval_switch)
          flow.retrieve_by_sub_flow(
            name = module.name,
            sub_flow = sub_flow,
            retrieve_num = "{{" + module.dragon_retrieval_num_attr + "}}",
            deep_copy = False,
            pass_browse_set = True if module.dragon_retrieval_pass_browse_set else False,
            pass_common_attrs_in_request = False,
            pass_common_attrs = [param["attr_name"] for param in module.dragon_import_params if param["attr_type"] == "common_attr"]
                + ([module.dragon_abtest_suffix_list_attr] if module.dragon_abtest_suffix_list_attr else [])
                + ([module.dragon_abtest_report_hit_attr] if module.dragon_abtest_report_hit_attr else []),
            merge_common_attrs = [ { "name": param["attr_name"], "as": param["attr_output_name"] } for param in module.dragon_export_params if param["attr_type"] == "common_attr"],
            merge_item_attrs = [ { "name": param["attr_name"], "as": param["attr_output_name"] } for param in module.dragon_export_params if param["attr_type"] == "item_attr"],
          )
          if module.dragon_retrieval_switch:
            flow.end_()

      for module in level_modules:
        if not module.is_retrieval():
          flow.namespace_(ns = module.name, nest = True)
          module.process()
          module.post_process()
          flow.namespace_()
    flow.namespace_()

  def __get_topological_modules(self) -> list:
    attr_key_2_modules_map = {}
    for module in self.__modules:
      if not module.dragon_import_params:
        continue
      for param in module.dragon_import_params:
        attr_key = "{}@{}".format(param["attr_name"], param["attr_type"])
        if attr_key not in attr_key_2_modules_map:
          attr_key_2_modules_map[attr_key] = []
        attr_key_2_modules_map[attr_key].append(module)

    module_name_2_modules_map = {}
    module_name_2_num_map = {}
    for module in self.__modules:
      if not module.dragon_export_params:
        continue
      for param in module.dragon_export_params:
        attr_key = "{}@{}".format(param["attr_name"], param["attr_type"])
        if attr_key not in attr_key_2_modules_map:
          continue
        for module2 in attr_key_2_modules_map[attr_key]:
          if module.name not in module_name_2_modules_map:
            module_name_2_modules_map[module.name] = []
          module_name_2_modules_map[module.name].append(module2)
          if module2.name not in module_name_2_num_map:
            module_name_2_num_map[module2.name] = 0
          module_name_2_num_map[module2.name] += 1

    level_modules = []
    for module in self.__modules:
      if not module.dragon_import_params or module.name not in module_name_2_num_map:
        level_modules.append(module)

    topological_modules = [ level_modules ]
    for level_modules in topological_modules:
      next_level_modules = []
      for module in level_modules:
        if module.name not in module_name_2_modules_map:
          continue
        for module2 in module_name_2_modules_map[module.name]:
          module_name_2_num_map[module2.name] -= 1
          if module_name_2_num_map[module2.name] == 0:
            next_level_modules.append(module2)
      if next_level_modules:
        topological_modules.append(next_level_modules)

    return topological_modules

  def __append_retrieval_params_processor(self, flow: BaseRecoFlow, topological_modules: list) -> None:
    ab_key_2_params_map = {}
    for level_modules in topological_modules:
      for module in level_modules:
        if module.is_retrieval() and module.dragon_retrieval_switch:
          ab_key = (module.dragon_retrieval_switch.pop("biz_name"), module.dragon_abtest_suffix_list_attr)
          if ab_key not in ab_key_2_params_map:
            ab_key_2_params_map[ab_key] = []
          module.dragon_retrieval_switch["attr_name"] = BaseRecoStage.RETRIEVAL_SWITCH_PREFIX + module.name
          ab_key_2_params_map[ab_key].append(module.dragon_retrieval_switch)

    for item in ab_key_2_params_map.items():
      param_dict = dict(biz_name = item[0][0], ab_params = item[1])
      if item[0][1]:
        param_dict["prioritized_suffix"] = "{{" + item[0][1] + "}}"
      flow.get_abtest_params(**param_dict)
