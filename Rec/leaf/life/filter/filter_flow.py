from common import CommonRecoFlow
from dragonfly.ext.explore.explore_api_mixin import ExploreApiMixin
from dragonfly.ext.explore_life.explore_life_api_mixin import ExploreLifeApiMixin


class FilterFlow(CommonRecoFlow, ExploreApiMixin, ExploreLifeApiMixin):
  def __init__(self, name: str) -> None:
    super().__init__(name, "life", "filter", "config", "module", "config/module", "lua/module")

  def _flow_end(self):
    self \
    .get_abtest_params(
      biz_name = "RECO_RPC",
      ab_params = [
        {
          "attr_name": "enable_xlife_not_life_filter",
          "default_value": False,
          "param_name": "enable_xlife_not_life_filter",
          "param_type": "bool"
        },
        {
          "attr_name": "enable_life_target_hetu_new",
          "default_value": False,
          "param_name": "enable_life_target_hetu_new",
          "param_type": "bool"
        },
        {
          "attr_name": "life_target_hetu_version",
          "default_value": "v1",
          "param_name": "life_target_hetu_version",
          "param_type": "string"
        },
        {
          "attr_name": "life_enable_nonlife_nice_author_skip_filter",
          "default_value": False,
          "param_name": "life_enable_nonlife_nice_author_skip_filter",
          "param_type": "bool"
        },
        {
          "attr_name": "life_nice_author_grade_thd",
          "default_value": 10,
          "param_name": "life_nice_author_grade_thd",
          "param_type": "int"
        }
      ]
    ) \
    .if_("enable_xlife_not_life_filter == 1") \
      .if_("enable_life_target_hetu_new == 1") \
        .get_kconf_params(
          kconf_configs = [
            {
              "kconf_key": "reco.eyeshot.LifeTabTargetHetuL2Json",
              "json_path": "{{life_target_hetu_version}}",
              "export_common_attr": "target_hetu_l2_list"
            },
            {
              "kconf_key": "reco.eyeshot.LifeTabGrayHetuL2Json",
              "json_path": "{{life_target_hetu_version}}",
              "export_common_attr": "gray_hetu_l2_list"
            },
            {
              "kconf_key": "reco.eyeshot.LifeTabTargetHetuL1Json",
              "json_path": "{{life_target_hetu_version}}",
              "export_common_attr": "target_hetu_l1_list"
            },
            {
              "kconf_key": "reco.eyeshot.LifeTabGrayHetuL1Json",
              "json_path": "{{life_target_hetu_version}}",
              "export_common_attr": "gray_hetu_l1_list"
            },
          ]
        ) \
        .enrich_attr_by_light_function(
          import_item_attr = [
            "photo_id",
            "hetu_tag_level_info__hetu_level_one",
            "hetu_tag_level_info__hetu_level_two"
          ],
          import_common_attr = [
            "target_hetu_l2_list",
            "gray_hetu_l2_list",
            "target_hetu_l1_list",
            "gray_hetu_l1_list"
          ],
          export_item_attr = [
            "gray_target", # 灰度 + 非生活打散，生活设为pid，灰度 + 非生活设为1
            "not_life_target" # 非生活打散，灰度 + 生活设置为pid，非生活设为1
          ],
          function_name = "ContentControlDiversifyTagV2",
          class_name = "ExploreLifeLightFunctionSet"
        ) \
      .else_() \
        .get_kconf_params(
          kconf_configs = [{
            "kconf_key": "reco.eyeshot.LifeTabTargetHetu",
            "value_type": "list_int64",
            "export_common_attr": "target_hetu_list",
            "default_value": []
          }]
        ) \
        .get_kconf_params(
          kconf_configs = [{
            "kconf_key": "reco.eyeshot.LifeTabNotTargetHetu",
            "value_type": "list_int64",
            "export_common_attr": "not_target_hetu_list",
            "default_value": []
          }]
        ) \
        .enrich_attr_by_light_function(
          import_item_attr = [
            "photo_id",
            "hetu_tag_level_info__hetu_level_one",
            "hetu_tag_level_info__hetu_level_two"
          ],
          import_common_attr = [
            "target_hetu_list",
            "not_target_hetu_list"
          ],
          export_item_attr = [
            "gray_target",
            "not_life_target"
          ],
          function_name = "ContentControlDiversifyTag",
          class_name = "ExploreLifeLightFunctionSet"
        ) \
      .end_() \
      .if_("life_enable_nonlife_nice_author_skip_filter == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "life_nice_author_grade_thd", "as": "author_grade_thd"}
          ],
          import_item_attr = [
            "author_grade_key",
            "not_life_target"
          ],
          export_item_attr = [
            "not_life_filter"
          ],
          function_name = "CalcNotLifeCateNotNiceAuthor",
          class_name = "ExploreLifeLightFunctionSet"
        ) \
        .count_reco_result(
          save_count_to = "not_life_filter_count",
          target_item = {"not_life_filter": 1}
        ) \
        .filter_by_attr(
          attr_name = "not_life_filter",
          remove_if = "==",
          compare_to = 1
        ) \
      .else_() \
        .count_reco_result(
          save_count_to = "not_life_filter_count",
          target_item = {"not_life_target": 1}
        ) \
        .filter_by_attr(
          attr_name = "not_life_target",
          remove_if = "==",
          compare_to = 1
        ) \
      .end_() \
      .perflog_attr_value(
        check_point = "life",
        aggregator = "avg",
        common_attrs = [
          "not_life_filter_count"
        ],
      ) \
    .end_()
    self._perf_result(
      attr_map={
        "is_picture": ["pic", "count"],
        "is_support_author_picture": ["sp_aid_pic", "count"],
        "high_value_pic_flag": ["high_value_pic", "count"]
      },
      perf_sampling_attr="_IS_PERF_SAMPLING_REQUEST_",
    )
    super()._flow_end()
