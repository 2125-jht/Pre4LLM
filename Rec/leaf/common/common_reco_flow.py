import os
import json
from core import BaseRecoFlow
from .json_config_reco_stage import JsonConfigRecoStage
from .check_rule import LuaNameCheckRule

class CommonRecoFlow(BaseRecoFlow):
  def __init__(self, name: str, service_name: str, stage_name: str, stage_config_dir: str, module_dir: str, module_config_dir: str, module_lua_dir: str,
      is_sub_flow: bool = False, optimize_processor_order: bool = False) -> None:
    if is_sub_flow:
      super().__init__(name, optimize_processor_order = optimize_processor_order)
      return

    final_stage_config_dir = os.path.join(service_name, stage_name, stage_config_dir)
    final_stage_config_dir = os.path.join(final_stage_config_dir, name + ".json")

    assert os.path.exists(final_stage_config_dir)
    with open(final_stage_config_dir, "r") as f:
      config = json.load(f)
    # NOTE(huzengyi): 使用 stage 配置作为 flow 配置
    super().__init__(name, config = config, optimize_processor_order = optimize_processor_order)
    self.__stage_name = stage_name
    self.__name = name
    self.__service_name = service_name
       
    final_module_config_dir = os.path.join(service_name, stage_name, module_config_dir)
    final_module_lua_dir = os.path.join(service_name, stage_name, module_lua_dir)
    check_rules = [
      LuaNameCheckRule(final_module_config_dir, final_module_lua_dir),
    ]
    for check_rule in check_rules:
      check_rule.check()

    self.namespace_(ns = self.name, nest = True)
    self._flow_begin()
    self.namespace_()
    self._reco_stage = JsonConfigRecoStage(stage_name, service_name, final_stage_config_dir, module_dir, module_config_dir)
    self._reco_stage.append_flow(self)
    self.namespace_(ns = self.name, nest = True)
    self._flow_end()
    self.namespace_()

  def _flow_begin(self):
    self \
      .gen_common_attr_by_lua(
        REORDER_BARRIER = True,
        attr_map = {
          self.__stage_name + "_begin_ts": "util.GetTimestamp()",
        },
      )

  def _flow_end(self):
    abtest_metrix_name_prefix = self.__service_name + "_reco_leaf_" + self.__name

    self \
      .gen_common_attr_by_lua(
        OPT_REORDER = False,
        attr_map = {
          abtest_metrix_name_prefix + "_ts": "util.GetTimestamp() - " + self.__stage_name + "_begin_ts",
        },
      ) \
      .copy_user_meta_info(
        OPT_REORDER = False,
        save_flow_cpu_cost_to_attr = abtest_metrix_name_prefix + "_cpu_cost_ts",
      )

  def _perf_result(self, step_name: str = None, attr_map: map = None, range_end: str = None, perf_sampling_attr: str = None):
    if perf_sampling_attr:
      self.if_(perf_sampling_attr + " == 1")

    if step_name:
      result_count_attr_name = self.__stage_name + "_" + step_name + "_result_count"
      reason_check_point = self.__stage_name + "_" + step_name
    else:
      result_count_attr_name = self.__stage_name + "_result_count"
      reason_check_point = self.__stage_name

    self \
      .count_reco_result(
        save_count_to = result_count_attr_name,
        range_end = range_end,
      )
    perf_common_attr_list = [result_count_attr_name]
    perf_item_attr_list = []

    if attr_map:
      for attr_name, perf_info in attr_map.items():
        if len(perf_info) == 2:
          perf_name, perf_type = perf_info
          if perf_type == "count":
            if step_name:
              result_count_attr_name = "_".join([self.__stage_name, step_name, perf_name, "result_count"])
            else:
              result_count_attr_name = "_".join([self.__stage_name, perf_name, "result_count"])
            self \
              .count_reco_result(
                save_count_to = result_count_attr_name,
                target_item = { attr_name: 1 },
                range_end = range_end,
              )
            perf_common_attr_list.append(result_count_attr_name)
          elif perf_type == "value_count":
            perf_item_attr_list.append(attr_name)

    self \
      .perflog_attr_value(
        check_point = self.__stage_name,
        common_attrs = perf_common_attr_list,
      ) \
      .perflog_reason_count(
        check_point = reason_check_point,
      )

    if len(perf_item_attr_list) > 0:
      self \
        .perflog_attr_value(
          check_point = reason_check_point,
          item_attrs = perf_item_attr_list,
          aggregator = "count",
          range_end = range_end,
        )

    if perf_sampling_attr:
      self.end_()

    return self

  def _dump_attr_to_kafka(self, stage_name : str, dump_item_attr_list : list, dump_common_attr_list : list = [], range_end : str = None):
    """
    在该阶段的关键位置, 将全部 item 的重要 item attr 落盘
    """
    self \
      .if_("enable_dump_attrs_to_kafka == 1")\
        .set_attr_value(
          common_attrs = [
            {
              "name": "dump_stage_name",
              "type": "string",
              "value": stage_name,
            }
          ],
        ) \
        .dump_context(
          common_attrs = [
            "_USER_ID_",
            "_DEVICE_ID_",
            "_REQ_ID_",
            "_REQ_TYPE_",
            "_REQ_TIME_",
            "dump_stage_name"
          ] + dump_common_attr_list,
          include_item_results = True,
          item_attrs = dump_item_attr_list,
          dump_to_attr = "dump_context_str",
          range_end = range_end,
        ) \
        .send_with_kafka(
          common_attr = "dump_context_str",
          topic_name = "reco_explore_leaf_dump_log",
        ) \
      .end_()

    return self