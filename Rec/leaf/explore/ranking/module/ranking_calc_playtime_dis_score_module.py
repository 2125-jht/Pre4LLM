from ranking import CommonModule

class RankingCalcPlaytimeDisScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_fr_cal_playtime_dis_score == 1 and uExploreActiveDays < explore_fr_playtime_dis_score_days_threshold") \
        .get_remote_embedding_lite_v2(
          protocol = 1,
          colossusdb_embd_service_name = "grpc_clsdb_ps-hate-embedv1-1",
          colossusdb_embd_table_name = "grpc_wtdGmmEmbV1",
          id_converter = {"type_name": "mioEmbeddingIdConverter"},
          slot = 4004,
          input_attr_name = "photo_id",
          output_attr_name = "dis_embedding",
          query_source_type = "item_attr",
          raw_data_type = "float32",
          colossusdb_use_kconf_client = False,
          size = 1,
          client_side_shard = True
        ) \
        .set_attr_default_value(
          item_attrs = [
            {
              "name": "temp_prefer_score_index",
              "type": "int",
              "value": 0
            }
          ]
        ) \
        .select_list_values(
          index_attr = "temp_prefer_score_index",
          list_values = [
            {"from": "dis_embedding", "to": "playtime_dis_score"},
          ],
          select_item = {
            "attr_name": "dis_embedding",
            "select_if": "not null"
          },
          is_common_attr = False
        ) \
        .if_("enable_fr_cal_wtd_merge_playtime_dis_score == 1") \
          .calc_by_simple_formula(
            formulas = [
              dict(
                expr = "[[playtime_dis_score]] * (1.0 + {{explore_fr_wtd_merge_playtime_dis_score_alpha}} * [[awesome_wtd]]) ^ {{explore_fr_wtd_merge_playtime_dis_score_beta}}",
                output_attr = "playtime_dis_score"
              )
            ]
          ) \
        .end_() \
      .end_() \
      .if_("enable_fr_cal_inorder_playtime_dis_score == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_fr_cal_inorder_playtime_dis_score_sub_coeff", "as": "sub_coeff"},
          ],
          import_item_attr = [
            {"name": "playtime_dis_score", "as": "score"},
          ],
          export_item_attr = [
            {"name": "score", "as": "inorder_playtime_dis_score"}
          ],
          function_name = "CalExploreDoubleMinusDouble",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_()