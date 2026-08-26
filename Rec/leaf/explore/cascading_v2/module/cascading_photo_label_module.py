from cascading_v2 import CommonModule

class CascadingPhotoLabelModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .copy_item_meta_info(
        save_item_key_to_attr = "item_key",
        save_reason_to_attr = "reason",
      ) \
      .pack_item_attr(
        item_source = {
          "reco_results": True
        },
        mappings = [{
          "aggregator": "concat",
          "from_item_attr": "item_key",
          "to_common_attr": "cascade_input_item_key_list",
        }],
      )

    self._is_marketing_compensation_label()
    self._userfulness_author_tag()
    self._interest_cluster_id()
    self._is_protogenetic_advertise_label()

  def _is_marketing_compensation_label(self):
    self.flow \
      .if_("explore_enable_adjust_marketing_compensation_photo == 1") \
        .split_string(
          input_common_attr = "explore_marketing_compensation_photo_tags_list_str",
          output_common_attr = "explore_marketing_compensation_photo_tags_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_marketing_compensation_photo_tags_list", "as": "tags_list"},
            {"name": "explore_marketing_compensation_high_value_author_ignore", "as": "high_value_author_ignore"},
            {"name": "explore_marketing_compensation_open_reason_thres", "as": "open_reason_thres"},
            "high_value_black_author_map_ptr"
          ],
          import_item_attr = [
            "sirius_distribution_info__mark_cod",
            "author__id"
          ],
          export_item_attr = [
            "is_marketing_compensation_photo"
          ],
          function_name = "GenIsMarketingCompensationPhoto",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()
    return self

  def _userfulness_author_tag(self) -> None:
    self.flow \
      .if_("enable_explore_transform_photo_proinc_type == 1") \
        .item_attr_operation(
          item_attr_a = "photo_proinc_type",
          common_attr_b = 8,
          operator = "&",
          output_attr = "userfulness_author_tag",
        ) \
      .end_()

  def _interest_cluster_id(self) -> None:
    self.flow \
      .if_("enable_explore_use_interest_cluster_id_632 == 1") \
        .copy_attr(
          attrs=[{
            "from_item": "cluster_id_632",
            "to_item": "interest_cluster_id"
          }]
        ) \
      .else_() \
        .copy_attr(
          attrs=[{
            "from_item": "hetu_sim_cluster_id",
            "to_item": "interest_cluster_id"
          }]
        ) \
      .end_()

  def _is_protogenetic_advertise_label(self):  # 精排用
    self.flow \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "protogenetic_advertise_type_list_str"
        ],
        import_item_attr = [
          "data_set_tags_bit"
        ],
        export_item_attr = [
          "is_protogenetic_advertise_photo"
        ],
        function_name = "IsProtogeneticAdvertisePhoto",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .if_("enable_explore_prev_items_gen_is_protogenetic_advertise_photo == 1") \
        .enrich_attr_by_light_function(
          item_list_from_attr = "explore_recent_play_list",
          import_common_attr = [
            "protogenetic_advertise_type_list_str"
          ],
          import_item_attr = [
            "data_set_tags_bit"
          ],
          export_item_attr = [
            "is_protogenetic_advertise_photo"
          ],
          function_name = "IsProtogeneticAdvertisePhoto",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_ranking_diversity_enable_standard_explore_realshow_pid_list == 1") \
        .enrich_attr_by_light_function(
          item_list_from_attr = "standard_explore_realshow_pid_list",
          import_common_attr = [
            "protogenetic_advertise_type_list_str"
          ],
          import_item_attr = [
            "data_set_tags_bit"
          ],
          export_item_attr = [
            "is_protogenetic_advertise_photo"
          ],
          function_name = "IsProtogeneticAdvertisePhoto",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()
