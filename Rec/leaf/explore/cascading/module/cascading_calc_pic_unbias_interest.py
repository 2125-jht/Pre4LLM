from cascading import CommonModule

class CascadingCalcPicUnbiasInterestModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_mc_calc_pic_unbias_interest == 1") \
        .if_("enable_explore_pic_unbias_interest_key_use_status == 1") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "explore_unbias_interest_cluster_prefix", "as": "key_prefix"},
              "basic_info_gender_v2",
              "uMarriageLabelKV",
              "uBirthLabelKV",
              "uEduLabelKV",
              "uMinorLabelKV",
              "uStudentLabelKV",
            ],
            export_common_attr = [
              {"name": "user_gender_status_key", "as": "user_age_gender_key"},
            ],
            function_name = "GetUserGenderStatusKey",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .else_() \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "explore_unbias_interest_cluster_prefix", "as": "key_prefix"},
              "basic_info_age_segment_v2",
              "basic_info_gender_v2",
            ],
            export_common_attr = [
              "user_age_gender_key"
            ],
            function_name = "GetUserAgeGenderKey",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .end_() \
        .get_kconf_params(
          kconf_configs = [{
            "kconf_key": "{{explore_mc_pic_unbias_interest_kconf_key}}",
            "json_path": "{{user_age_gender_key}}",
            "default_value": "",
            "export_common_attr": "explore_unbias_interest_str"
          }]
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "explore_unbias_interest_str",
          ],
          import_item_attr = [
            {"name": "hetu_tag_level_info_v2__hetu_level_two", "as": "hetu_level_two"},
          ],
          export_item_attr = [
            "pic_unbias_interset_score"
          ],
          function_name = "GetPicUnbiasIntersetScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()