from cascading import CommonModule

class CascadingHateSimilarScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_mc_calc_hate_similar_score == 1") \
        .explore_get_embedding_map_enricher(
          embedding_list_attr = "mmu_embeddings",
          source_pids_list_attr = "embedding_source_pids",
          dim_size = 64, 
          export_common_attr = "pid_embedding_map",
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "user_info_ptr",
          ],
          export_common_attr = [
            "real_show_no_click_pids",
          ],
          function_name = "GetRealShowNoClickPids",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name" : "real_show_no_click_pids", "as" : "history_pid_list"},
            "pid_embedding_map",
            {"name" : "enable_mc_calc_hate_history_limit", "as" : "history_limit_cnt"},
          ],
          import_item_attr = [
            "photo_id",
          ],
          export_item_attr = [
            {"name" : "similar_score", "as" : "hate_similar_score_by_mc_embedding"},
          ],
          function_name = "CalcHistorySimilarScoreByEmbedding",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
