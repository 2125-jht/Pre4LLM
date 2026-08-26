from ranking import CommonModule

class RankingGetTagNexSimilarScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_fr_cal_tagnex_embedding_similar_score == 1") \
        .explore_custom_embedding_score_enricher(
          user_info_ptr_attr = "user_info_ptr",
          embedding_list_attr = "tagnex_embeddings_ranking",
          source_pids_list_attr = "tagnex_embeddings_source_pids",
          calc_type = "list_similarity",
          target_pids_list_attr = "videoPlayingPid",
          export_item_attr = "tagnex_embedding_similiar_score",
          dim_size = 128,
          check_point_ = "fr",
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_tagnex_embedding_similiar_score_average_ratio", "as": "average_ratio"},
          ],
          import_item_attr = [
            {"name": "tagnex_embedding_similiar_score", "as": "input_target_item_attr"},
          ],
          export_item_attr = [
            {"name": "output_target_item_attr", "as": "tagnex_embedding_similiar_score"},
          ],
          function_name = "FillItemAttrByAverage",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()
