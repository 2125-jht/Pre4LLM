from cascading import CommonModule

class CascadingCalcClusterInterestAttrModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_life_calc_cluster_interest_attr == 1") \
        .set_attr_value(
          no_overwrite=True,
          item_attrs=[
            {
              "name": "cluster_interest_vv",
              "type": "int",
              "value": 0
            },
            {
              "name": "is_eff_interest",
              "type": "int",
              "value": 0
            },
            {
              "name": "is_longterm_interest",
              "type": "int",
              "value": 0
            },
            {
              "name": "cluster_interest_score",
              "type": "double",
              "value": 0.01
            },
            {
              "name": "combine_version",
              "type": "int",
              "value": 0
            },
            {
              "name": "longterm_cluster_interest_score",
              "type": "double",
              "value": 0.01
            },
          ]
        ) \
        .log_debug_info(
          item_attrs = [
            "photo_id",
          ],
          common_attrs = [
          "colossus_photo_id_list",
            "colossus_play_time_list",
            "copy_colossus_photo_id_list",
            "colossus_label_list",
            "colossus_duration_list",
            "colossus_timestamp_list",
            "colossus_channel_list",
            "colossus_tag_list",
            "last_danlie_photo_id_list",
        "last_tired_photo_id_list",
          ],
          for_debug_request_only = True,
          respect_sample_loggging = True,
        ) \
        .explore_life_cluster_interest_enricher(
        colossus_resp_attr = "colossus_resp_v2",
        remap_cluster_id_632_list = "remap_cluster_id_632_list",
        colossus_photo_id_list = "copy_colossus_photo_id_list",
        colossus_cluster_id_list = "colossus_cluster_id_list",
        hetu_sim_cluster_id = "hetu_sim_cluster_id",
        # save_score_to_attr = "yyl_test",
        save_cluster_interest_vv_to_attr = "cluster_interest_vv",
        save_is_eff_interest_to_attr = "is_eff_interest",
        save_is_longterm_interest_to_attr = "is_longterm_interest",
        save_cluster_interest_score_to_attr = "cluster_interest_score",
        save_lt_interest_score_to_attr = "longterm_cluster_interest_score",
        save_eff_interest_num_to_common_attr = "eff_interest_num",
        save_user_cluster_eff_interest_list = "cluster_eff_interest_list",
        save_user_cluster_interest_score_list = "cluster_interest_score_list",
        save_user_cluster_interest_vv_list = "cluster_interest_vv_list",
        save_user_cluster_interest_vv_ratio_list_ = "cluster_interest_vv_ratio_list",
        save_user_cluster_shortterm_interest_list = "cluster_shortterm_interest_list",
        save_user_cluster_longterm_interest_list = "cluster_longterm_interest_list",
        ) \
      .end_() \
      .normalize_attr(
            input_attr="cluster_interest_score",
            output_attr="norm_cluster_interest_score",
            mode="min_max_scale",
            default_val=0.0,
        ) \
      .if_("enable_cluster_interest_score_combine == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "xlife_cascading_cluster_interest_score_compile_version", "as": "combine_version1"},
          ],
          import_item_attr = [
            "cascade_plvtr2",
            "cluster_interest_vv",
            "norm_cluster_interest_score",
            "cluster_interest_score",
            "is_eff_interest",
            "is_longterm_interest",
            "longterm_cluster_interest_score",
            "combine_version",
          ],
          export_item_attr = [
            "nonneg_intere_score"
            # {"name": "nonneg_intere_score", "as": "nonneg_intere_score2"},
          ],
          function_name = "GetClusterInterestScore",
          class_name = "ExploreLifeLightFunctionSet",
        )\
          .pack_item_attr(
            item_source={
                'reco_results': True,
            },
          mappings = [{
            "from_item_attr": "cluster_interest_score",
            "to_common_attr": "cluster_interest_score_list",
          },
                      {
            "from_item_attr": "cluster_interest_vv",
            "to_common_attr": "cluster_interest_vv_list",
          },
                      ]
        ) \
        .log_debug_info(
          common_attrs = [
             "cluster_interest_vv_list",
            "cluster_interest_score_list",
            "colossus_cluster_id_list",
            "remap_cluster_id_632_list",
            "colossus_photo_id_list",
            "colossus_play_time_list",
            "colossus_label_list",
            "colossus_author_id_list",
            "colossus_channel_list",
            "colossus_duration_list",
            "colossus_timestamp_list",
            "colossus_tag_list",
             "colossus_tag_list",
            "eff_interest_num",
          ],
          item_attrs = [
            # "yyl_test",
            "norm_cluster_interest_score",
            "cascade_plvtr2",
            "hetu_sim_cluster_id",
            "cluster_interest_vv",
            "is_eff_interest",
            "is_longterm_interest",
            "cluster_interest_score",
            "nonneg_intere_score",
            "longterm_cluster_interest_score",
            "eff_interest_num",
          ],
          for_debug_request_only = True,
          respect_sample_loggging = True,
        ) \
      .end_() \
     