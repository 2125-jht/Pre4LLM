from dragonfly.common_leaf_dsl import LeafFlow
from dragonfly.ext.explore.explore_api_mixin import ExploreApiMixin

class ExploreDiversifyingFlow(LeafFlow, ExploreApiMixin):
  def __init__(self, name: str):
    super().__init__(name)

    self \
      .get_abtest_params(
        biz_name = "KUAISHOU_APPS",
        ab_params = [
          ("enable_explore_degrading_pic_diversity", 1),
          ("explore_degrading_pic_winsize", 10),
          ("explore_degrading_pic_max", 3),
          ("enable_explore_degrading_hetu_cluster_diversity", 1),
          ("explore_degrading_hetu_cluster_winsize", 4),
          ("explore_degrading_hetu_cluster_max", 1),
          ("enable_explore_degrading_hetu5_diversity", 1),
          ("explore_degrading_hetu5_winsize", 4),
          ("explore_degrading_hetu5_max", 1),
          ("enable_explore_degrading_hetu2_diversity", 1),
          ("explore_degrading_hetu2_winsize", 10),
          ("explore_degrading_hetu2_max", 4),
          ("enable_explore_degrading_hetu1_diversity", 1),
          ("explore_degrading_hetu1_winsize", 10),
          ("explore_degrading_hetu1_max", 3),
        ],
      ) \
      .get_item_attr_by_distributed_flat_index(
        photo_store_kconf_key = "reco.distributedIndex.hotPhotoInfoCommonIndex",
        use_dynamic_photo_store = True,
        attrs = [
          "picture_type",
          "hetu_tag_level_info_v2__hetu_level_one",
          "hetu_tag_level_info_v2__hetu_level_two",
          "hetu_tag_level_info__hetu_level_five",
          "hetu_sim_cluster_id",
        ],
      ) \
      .copy_attr(
        attrs = [
          {
            "from_item": "picture_type",
            "to_item": "picture_type_rule_tag",
          },
        ],
        select_item = {
          "attr_name": "picture_type",
          "compare_to": 0,
          "select_if": ">",
          "select_if_attr_missing": False,
        },
      ) \
      .explore_transform_hetu_tag(
        output_attrs = [
          "hetu_level_one_v2",
          "hetu_level_two_v2",
        ],
        hetu_tag_attrs = [
          "hetu_tag_level_info_v2__hetu_level_one",
          "hetu_tag_level_info_v2__hetu_level_two",
        ],
      ) \
      .diversify_by_rules(
        max_satisfied_pick = "{{page_size}}",
        rules = [
          dict(attr_name = "picture_type_rule_tag",
                enabled = "{{enable_explore_degrading_pic_diversity}}",
                window_size = "{{explore_degrading_pic_winsize}}",
                max_num = "{{explore_degrading_pic_max}}",
                priority = 0),
          dict(attr_name = "hetu_sim_cluster_id",
                enabled = "{{enable_explore_degrading_hetu_cluster_diversity}}",
                window_size = "{{explore_degrading_hetu_cluster_winsize}}",
                max_num = "{{explore_degrading_hetu_cluster_max}}",
                priority = 0),
          dict(attr_name = "hetu_tag_level_info__hetu_level_five",
                enabled ="{{enable_explore_degrading_hetu5_diversity}}",
                window_size = "{{explore_degrading_hetu5_winsize}}",
                max_num = "{{explore_degrading_hetu5_max}}",
                priority = 0),
          dict(attr_name = "hetu_level_two_v2",
                enabled = "{{enable_explore_degrading_hetu2_diversity}}",
                window_size = "{{explore_degrading_hetu2_winsize}}",
                max_num = "{{explore_degrading_hetu2_max}}",
                priority = 0),
          dict(attr_name = "hetu_level_one_v2",
                enabled = "{{enable_explore_degrading_hetu1_diversity}}",
                window_size = "{{explore_degrading_hetu1_winsize}}",
                max_num = "{{explore_degrading_hetu1_max}}",
                priority = 0)
        ],
      )
