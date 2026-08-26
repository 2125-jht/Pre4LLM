from cascading import CommonModule

class CascadingU2cMapModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_u2c_map == 1") \
        .pack_item_attr(
          item_source = {
            "common_attr": ["candidate_cids"],
          },
          mappings = [{
            "from_item_attr": "explore_u2c",
            "to_common_attr": "cascade_explore_u2c_score_set",
            "default_val": "{{cascade_explore_u2c_default_score}}"
          }]
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "candidate_cids", "as": "key_list"},
            {"name": "cascade_explore_u2c_score_set", "as": "value_list"},
            {"name": "cascade_explore_u2c_default_score", "as": "default_value"},
          ],
          import_item_attr = [
            {"name": "hetu_sim_cluster_id", "as": "item_key_attr"}
          ],
          export_item_attr = [
            {"name": "target_item_attr", "as": "cascade_explore_u2c_score"}
          ],
          function_name = "AddItemAttrByCommonMap",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .else_if_("enable_interest_cid_collaborative_filter == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "uOldMmuClusterId300ListList", "as": "uOldMmuClusterId300ListList"},
            {"name": "interest_cid_collaborative_score_map", "as": "cid_score_map"}
          ],
          import_item_attr = [
            {"name": "hetu_sim_cluster_id", "as": "hetu_sim_cluster_id862"}
          ],
          export_item_attr = [
            {"name": "collaborative_score", "as": "cascade_explore_u2c_score"}
          ],
          function_name = "InterestCidCollaborativeFilter",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()
