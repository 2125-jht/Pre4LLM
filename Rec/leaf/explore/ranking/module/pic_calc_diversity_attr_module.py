from ranking import CommonModule

class PicCalcDiversityAttrModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_calc_pic_top_category == 1") \
        .split_string(
          input_common_attr = "explore_pic_top_category_hetu_str",
          output_common_attr = "explore_pic_top_category_hetu_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_pic_top_category_hetu_list", "as": "attr_list"}
          ],
          import_item_attr = [
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "attrs"}
          ],
          export_item_attr = [
            {"name": "is_in_set", "as": "is_pic_top_category"}
          ],
          function_name = "AttrListIsInSet",
          class_name = "ExploreLightFunctionSetV2",
          select_item = {
            "join": "and",
            "filters": [{
              "attr_name": "is_picture",
              "select_if": "==",
              "compare_to": 1,
            }, {
              "join": "or",
              "filters": [{
                  "attr_name": "corr_pctr",
                  "select_if": "<",
                  "compare_to": "{{explore_pic_top_category_pctr_thd}}",
                  "select_if_attr_missing": True
              }, {
                  "attr_name": "pltr",
                  "select_if": "<",
                  "compare_to": "{{explore_pic_top_category_pltr_thd}}",
                  "select_if_attr_missing": True
              }]
            }],
          }
        ) \
      .end_() \
      .if_("enable_explore_calc_pic_is_not_long_interest_hetu == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "pic_user_long_interest_hetu_list", "as": "attr_list"}
          ],
          import_item_attr = [
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "attrs"}
          ],
          export_item_attr = [
            {"name": "is_not_in_set", "as": "is_not_pic_long_interest_hetu"}
          ],
          function_name = "AttrListIsNotInSet",
          class_name = "ExploreLightFunctionSetV2",
          select_item = {
            "join": "and",
            "filters": [{
              "attr_name": "is_picture",
              "select_if": "==",
              "compare_to": 1,
            }, {
              "join": "or",
              "filters": [{
                  "attr_name": "corr_pctr",
                  "select_if": "<",
                  "compare_to": "{{explore_pic_is_not_long_interest_hetu_pctr_thd}}",
                  "select_if_attr_missing": True
              }, {
                  "attr_name": "pltr",
                  "select_if": "<",
                  "compare_to": "{{explore_pic_is_not_long_interest_hetu_pltr_thd}}",
                  "select_if_attr_missing": True
              }]
            }],
          }
        ) \
      .end_()

