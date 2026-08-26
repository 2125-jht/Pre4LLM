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
      .if_("enable_explore_get_high_value_author == 1", to_be_delete = "date=2024-05-29;committer=wangziqi05") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "uFollowHighValueAuthorList", "as": "attr_list"}
          ],
          import_item_attr = [
            {"name": "author__id", "as": "attr"}
          ],
          export_item_attr = [
            {"name": "is_in_set", "as": "is_high_value_author"}
          ],
          function_name = "AttrIsInSet",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .enrich_attr_by_light_function(
          item_list_from_attr = "explore_recent_play_list",
          import_common_attr = [
            {"name": "uFollowHighValueAuthorList", "as": "attr_list"}
          ],
          import_item_attr = [
            {"name": "author__id", "as": "attr"}
          ],
          export_item_attr = [
            {"name": "is_in_set", "as": "is_high_value_author"}
          ],
          function_name = "AttrIsInSet",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_ranking_diversity_enable_standard_explore_realshow_pid_list == 1") \
        .enrich_attr_by_light_function(
          item_list_from_attr = "standard_explore_realshow_pid_list",
          import_common_attr = [
            {"name": "uFollowHighValueAuthorList", "as": "attr_list"}
          ],
          import_item_attr = [
            {"name": "author__id", "as": "attr"}
          ],
          export_item_attr = [
            {"name": "is_in_set", "as": "is_high_value_author"}
          ],
          function_name = "AttrIsInSet",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
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
      .copy_item_meta_info(
        save_reason_to_attr = "reason",
      ) \

  def post_process(self) -> None:
    ""
