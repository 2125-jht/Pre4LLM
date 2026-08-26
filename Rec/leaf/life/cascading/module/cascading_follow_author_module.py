from cascading import CommonModule

class CascadingFollowAuthorModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .enrich_with_protobuf(
        from_extra_var = "user_info_ptr",
        attrs = [
          dict(name="follow_author_list", path="follow_list.user.id"),
        ],
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "follow_author_list", "as": "attr_list"}
        ],
        import_item_attr = [
          {"name": "author__id", "as": "attr"}
        ],
        export_item_attr = [
          {"name": "is_in_set", "as": "is_follow_author"}
        ],
        function_name = "AttrIsInSet",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_la_long_view_author_list", "as": "attr_list"}
        ],
        import_item_attr = [
          {"name": "author__id", "as": "attr"}
        ],
        export_item_attr = [
          {"name": "is_in_set", "as": "is_long_view_author"}
        ],
        function_name = "AttrIsInSet",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .switch_("picture_follow_author_boost_version") \
      .case_(1) \
        .copy_attr(
          attrs = [{
            "from_item": "is_follow_author",
            "to_item": "is_picture_follow_author",
          }],
          target_item = {
            "is_picture": 1,
            "is_follow_author": 1
          }
        ) \
      .case_(2) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "colossus_follow_author__trigger_list", "as": "attr_list"}
          ],
          import_item_attr = [
            {"name": "author__id", "as": "attr"}
          ],
          export_item_attr = [
            {"name": "is_in_set", "as": "is_picture_follow_author"}
          ],
          function_name = "AttrIsInSet",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture": 1,
            "is_follow_author": 1
          }
        ) \
      .end_() \
      .copy_item_meta_info(
        save_reason_to_attr = "reason",
      ) \

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = [
          "colossus_follow_author__trigger_list"
        ],
        item_attrs = [
          "is_picture_follow_author",
          "author__id"
        ],
        target_item = {
          "is_picture_follow_author": 1
        }
      )
