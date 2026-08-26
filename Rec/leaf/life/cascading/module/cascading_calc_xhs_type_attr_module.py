from cascading import CommonModule

class CascadingCalcXhsTypeAttrModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_calc_xhs_type_picture == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "xhs_hetu_set",
            "xhs_hetu_memorydata_set",
            "calc_xhs_type_mode",
          ],
          import_item_attr = [
            "hetu_tag_level_info__hetu_level_two",
            "hetu_tag_level_info__hetu_level_three",
            "hetu_tag_level_info__hetu_level_four",
          ],
          export_item_attr = [
            "is_xhs_type_photo",
          ],
          function_name = "IsXhsTypePhoto",
          class_name = "ExploreLightFunctionSetV2",
          target_item = { "is_picture" : 1 }
        ) \
      .end_() \
      .if_("enable_calc_xhs_target_qualified_photo == 1") \
        .if_("enable_calc_xhs_target_photo_new_category == 1") \
          .copy_attr(
            attrs = [{"from_common": "xhs_target_hetu_set_new", "to_common": "final_xhs_target_hetu_set"}]
          ) \
        .else_() \
          .copy_attr(
            attrs = [{"from_common": "xhs_target_hetu_set", "to_common": "final_xhs_target_hetu_set"}]
          ) \
        .end_() \
        .enrich_attr_by_light_function(
          import_common_attr= [
            {"name": "final_xhs_target_hetu_set", "as": "target_hetu_set"},
            {"name": "xhs_target_audit_b_set", "as": "target_audit_b_set"},
            {"name": "xhs_target_photo_remove_author_bucket", "as": "remove_author_bucket"}
          ],
          import_item_attr = [
            {"name": "audit_b_second_tag", "as": "audit_b_result"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_tag_list"},
            {"name": "author__id", "as": "author_id"}
          ],
          export_item_attr = [
            "is_xhs_target_qualified_photo"
          ],
          function_name = "IsXhsTargetQualifiedPhoto",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_()
