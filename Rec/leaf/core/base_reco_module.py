from abc import ABC, abstractmethod
from core.base_reco_flow import BaseRecoFlow

class BaseRecoModule(ABC):
  def __init__(self, name: str, config: dict) -> None:
    self.__name = name
    self.__stage_name = None
    self.__config = config
    self.__flow = None

    self.__dragon_retrieval_switch_attr = None
    self.__dragon_retrieval_switch = None
    self.__dragon_retrieval_num_attr = None
    self.__dragon_retrieval_pass_browse_set = None
    self.__dragon_abtest_suffix_list_attr = None
    self.__dragon_abtest_report_hit_attr = None
    self.__dragon_abtest_params = []
    self.__dragon_kconf_params = []
    self.__dragon_constant_params = []
    self.__dragon_import_params = []
    self.__dragon_export_params = []
    self.__extract_dragon_params()

  @property
  def name(self) -> str:
    return self.__name

  @property
  def stage_name(self) -> str:
    return self.__stage_name

  @stage_name.setter
  def stage_name(self, stage_name: str) -> None:
    self.__stage_name = stage_name

  @property
  def config(self) -> dict:
    return self.__config

  @property
  def flow(self) -> BaseRecoFlow:
    return self.__flow

  @property
  def dragon_retrieval_switch(self) -> str:
    return self.__dragon_retrieval_switch

  @property
  def dragon_retrieval_num_attr(self) -> str:
    return self.__dragon_retrieval_num_attr

  @property
  def dragon_retrieval_pass_browse_set(self) -> bool:
    return self.__dragon_retrieval_pass_browse_set

  @property
  def dragon_abtest_suffix_list_attr(self) -> str:
    return self.__dragon_abtest_suffix_list_attr

  @property
  def dragon_abtest_report_hit_attr(self) -> str:
    return self.__dragon_abtest_report_hit_attr

  @property
  def dragon_import_params(self) -> list:
    return self.__dragon_import_params

  @property
  def dragon_export_params(self) -> list:
    return self.__dragon_export_params

  @classmethod
  def is_retrieval(cls) -> bool:
    return False

  def set_flow(self, flow: BaseRecoFlow) -> None:
    self.__flow = flow

  @abstractmethod
  def process(self) -> None:
    pass

  def post_process(self) -> None:
    pass

  def append_abtest_params_processor(self) -> None:
    if not self.__dragon_abtest_params:
      return

    biz_name_2_params_map = {}
    for param in self.__dragon_abtest_params:
      biz_name = param.pop("biz_name")
      if biz_name not in biz_name_2_params_map:
        biz_name_2_params_map[biz_name] = []
      biz_name_2_params_map[biz_name].append(param)

    for item in biz_name_2_params_map.items():
      param_dict = dict(biz_name = item[0], ab_params = item[1])
      if self.__dragon_abtest_suffix_list_attr:
        param_dict["prioritized_suffix"] = "{{" + self.__dragon_abtest_suffix_list_attr + "}}"
      self.__flow.get_abtest_params(**param_dict)

  def append_kconf_params_processor(self) -> None:
    if not self.__dragon_kconf_params:
      return

    self.__flow.get_kconf_params(
      kconf_configs = self.__dragon_kconf_params
    )

  def append_constant_params_processor(self) -> None:
    if not self.__dragon_constant_params:
      return

    common_attr_map = {}
    for param in self.__dragon_constant_params:
      if param["attr_type"] == "common_attr":
        if isinstance(param["value"], str):
          common_attr_map[param["attr_name"]] = "\"{}\"".format(param["value"])
        elif isinstance(param["value"], int) or isinstance(param["value"], float):
          common_attr_map[param["attr_name"]] = str(param["value"])
        elif isinstance(param["value"], bool):
          common_attr_map[param["attr_name"]] = str(1 if param["value"] else 0)

    if common_attr_map:
      self.__flow.gen_common_attr_by_lua(
        attr_map = common_attr_map
      )

  def add_dragon_abtest_param(self, biz_name: str, param_name: str, default_value, param_type: str = None, attr_name: str = None,
      report_ab_hit: bool = None) -> None:
    param = {
      "biz_name": biz_name,
      "param_name": param_name,
      "default_value": default_value,
    }
    if param_type:
      param["param_type"] = param_type
    if attr_name:
      param["attr_name"] = attr_name
    if report_ab_hit:
      param["report_ab_hit"] = "{{" + self.__dragon_abtest_report_hit_attr + "}}"

    if attr_name == self.__dragon_retrieval_switch_attr:
      self.__dragon_retrieval_switch = param
    else:
      self.__dragon_abtest_params.append(param)

  def add_dragon_kconf_param(self, kconf_key: str, export_common_attr: str, default_value = None, value_type: str = None, json_path: str = None) -> None:
    param = {
      "kconf_key": kconf_key,
      "export_common_attr": export_common_attr,
    }
    if default_value is not None:
      param["default_value"] = default_value
    if value_type is not None:
      param["value_type"] = value_type
    if json_path is not None:
      param["json_path"] = json_path

    self.__dragon_kconf_params.append(param)

  def add_dragon_constant_param(self, attr_name: str, attr_type: str, value):
    self.__dragon_constant_params.append({
      "attr_name": attr_name,
      "attr_type": attr_type,
      "value": value,
    })

  def read_retrieval_cache(self):
    self.__flow \
      .if_("retrieval_cache_config ~= nil and #retrieval_cache_config > 0 and _USER_ID_ > 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            { "name": "retrieval_cache_config", "as": "expired_time_str" },
          ],
          export_common_attr = [
            { "name": "expired_time", "as": "retrieval_cache_expired_time" },
          ],
          function_name = "ParseRetrCacheExpiredTime",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .if_("retrieval_cache_expired_time ~= nil and retrieval_cache_expired_time > 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
          .gen_common_attr_by_lua(
            attr_map = {
              "retrieval_cache_key": "\"" + self.__name + "-uid=\" .. _USER_ID_",
            },
          ) \
          .get_common_attr_from_redis(
            cluster_name = "recoCollectUserData",
            timeout_ms = 10,
            redis_params = [
              {
                "redis_key": "{{retrieval_cache_key}}",
                "output_attr_name": "retrieval_cache_string",
              },
            ],
          ) \
          .if_("retrieval_cache_string ~= nil and #retrieval_cache_string > 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
            .explore_retrieve_by_cache_string(
              cache_string_attr = "retrieval_cache_string",
            ) \
            .return_() \
          .end_() \
        .end_() \
      .end_()

  def write_retrieval_cache(self):
    self.__flow \
      .if_("retrieval_cache_expired_time ~= nil and retrieval_cache_expired_time > 0 and (retrieval_cache_string == nil or #retrieval_cache_string == 0)") \
        .explore_build_cache_string(
          save_cache_string_to_attr = "retrieval_cache_string",
        ) \
        .write_to_redis(
          kcc_cluster = "recoCollectUserData",
          timeout_ms = 10,
          key_prefix = self.__name + "-uid=",
          key = "{{_USER_ID_}}",
          value = "{{retrieval_cache_string}}",
          expire_second = "{{return retrieval_cache_expired_time or 600}}",
        ) \
      .end_()

  def __add_dragon_import_params(self, attr_name: str, attr_type: str):
    self.__dragon_import_params.append({
      "attr_name": attr_name,
      "attr_type": attr_type,
    })

  def __add_dragon_export_params(self, attr_name: str, attr_type: str, attr_output_name: str = None):
    if not attr_output_name:
      attr_output_name = attr_name
    self.__dragon_export_params.append({
      "attr_output_name": attr_output_name,
      "attr_name": attr_name,
      "attr_type": attr_type,
    })

  def __extract_dragon_params(self) -> None:
    if self.__config is None or "dragon_params" not in self.__config:
      return

    dragon_params = self.__config["dragon_params"]
    assert isinstance(dragon_params, dict)
    if self.is_retrieval():
      self.__dragon_retrieval_switch_attr = dragon_params.get("abtest_retrieval_switch_attr")
      self.__dragon_retrieval_num_attr = dragon_params.get("abtest_retrieval_num_attr")
      self.__dragon_retrieval_pass_browse_set = self.__config.get("pass_browse_set")
    self.__dragon_abtest_suffix_list_attr = dragon_params.get("abtest_suffix_list_attr")
    self.__dragon_abtest_report_hit_attr = dragon_params.get("abtest_report_hit_attr")
    self.__extract_dragon_abtest_params(dragon_params)
    self.__extract_dragon_kconf_params(dragon_params)
    self.__extract_dragon_constant_params(dragon_params)
    self.__extract_dragon_import_params(dragon_params)
    self.__extract_dragon_export_params(dragon_params)

  def __extract_dragon_abtest_params(self, dragon_params: dict) -> None:
    if "abtest" not in dragon_params:
      return

    params = dragon_params["abtest"]
    assert isinstance(params, dict)
    for param in params.items():
      assert isinstance(param[0], str), "属性名必须是 str 类型"
      biz_name = None
      param_name = None
      param_type = None
      default_value = None
      report_ab_hit = None
      if isinstance(param[1], dict):
        assert "default_value" in param[1], "参数必须包含默认值"
        biz_name = param[1].get("biz_name")
        param_name = param[1].get("param_name")
        param_type = param[1].get("param_type")
        report_ab_hit = param[1].get("report_ab_hit")
        default_value = param[1]["default_value"]
      elif isinstance(param[1], list):
        assert len(param[1]) == 2, "定义必须是 (param_name, default_value) 的形式"
        assert isinstance(param[1][0], str), "参数名必须是 str 类型"
        param_name = param[1][0]
        default_value = param[1][1]
      else:
        default_value = param[1]

      assert type(default_value) in [str, int, float, bool], "默认值必须是 int, double, bool, string 类型中的一种"
      if biz_name is None:
        biz_name = "RECO_RPC"
      if param_name is None:
        param_name = param[0]

      self.add_dragon_abtest_param(
        biz_name = biz_name,
        param_name = param_name,
        default_value = default_value,
        param_type = param_type,
        attr_name = param[0],
        report_ab_hit = report_ab_hit,
      )

  def __extract_dragon_kconf_params(self, dragon_params: dict) -> None:
    if "kconf" not in dragon_params:
      return

    params = dragon_params["kconf"]
    assert isinstance(params, dict)
    for param in params.items():
      assert isinstance(param[0], str)
      kconf_key = None
      default_value = None
      value_type = None
      json_path = None
      if isinstance(param[1], dict):
        assert "kconf_key" in param[1]
        kconf_key = param[1]["kconf_key"]
        default_value = param[1].get("default_value")
        value_type = param[1].get("value_type")
        json_path = param[1].get("json_path")
      elif isinstance(param[1], list):
        assert len(param[1]) == 2, "定义必须是 (kconf_key, default_value) 的形式"
        assert isinstance(param[1][0], str), "kconf_key 必须是 str 类型"
        kconf_key = param[1][0]
        default_value = param[1][1]

      self.add_dragon_kconf_param(
        kconf_key = kconf_key,
        export_common_attr = param[0],
        default_value = default_value,
        value_type = value_type,
        json_path = json_path,
      )

  def __extract_dragon_constant_params(self, dragon_params: dict) -> None:
    if "constant" not in dragon_params:
      return

    params = dragon_params["constant"]
    assert isinstance(params, dict)
    for param in params.items():
      assert isinstance(param[0], str)
      attr_type = None
      value_type = None
      value = None
      if isinstance(param[1], dict):
        assert "value" in param[1]
        value = param[1]["value"]
        attr_type = param[1].get("attr_type")
      else:
        value = param[1]

      if attr_type is None:
        attr_type = "common_attr"
      assert attr_type in ["common_attr", "item_attr"]
      assert type(value) in [str, int, float, bool], "默认值必须是 int, double, bool, string 类型中的一种"

      self.add_dragon_constant_param(
        attr_name = param[0],
        attr_type = attr_type,
        value = value,
      )

  def __extract_dragon_import_params(self, dragon_params: dict) -> None:
    if "import" not in dragon_params:
      return

    params = dragon_params["import"]
    assert isinstance(params, dict)
    self.__extract_dragon_import_params_by_type(params, "common_attrs", "common_attr")
    self.__extract_dragon_import_params_by_type(params, "item_attrs", "item_attr")
    self.__extract_dragon_import_params_by_type(params, "results", "result")

  def __extract_dragon_import_params_by_type(self, import_params: dict, param_key: str, attr_type: str) -> None:
    if param_key not in import_params:
      return

    attrs = import_params[param_key]
    assert isinstance(attrs, list)
    for attr in attrs:
      assert isinstance(attr, str)
      self.__add_dragon_import_params(attr, attr_type)

  def __extract_dragon_export_params(self, dragon_params: dict) -> None:
    if "export" not in dragon_params:
      return

    params = dragon_params["export"]
    assert isinstance(params, dict)
    self.__extract_dragon_export_params_by_type(params, "common_attrs", "common_attr")
    self.__extract_dragon_export_params_by_type(params, "item_attrs", "item_attr")
    self.__extract_dragon_export_params_by_type(params, "results", "result")

  def __extract_dragon_export_params_by_type(self, export_params: dict, param_key: str, attr_type: str) -> None:
    if param_key not in export_params:
      return

    attrs = export_params[param_key]
    assert isinstance(attrs, list)
    for attr in attrs:
      if isinstance(attr, str):
        self.__add_dragon_export_params(attr, attr_type)
      elif isinstance(attr, list):
        assert len(attr) == 2 and isinstance(attr[0], str) and isinstance(attr[1], str), "必须是 [attr_name, attr_rename] 的形式"
        self.__add_dragon_export_params(attr[0], attr_type, attr[1])
