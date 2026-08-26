from ranking import CommonModule

class RankingGetHateCoverSimilarScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_fr_cal_mm_cover_similar_score == 1") \
        .explore_diversity_update_enricher(
          user_info_ptr_attr = "user_info_ptr",
          pid_embedding_common_attr = "pid_mmu_cover_embedding_map_ranking",
          export_item_attr = "hate_cover_diversity_score",
          history_feed_back_version = 2,
          dim_size = "{{fr_mgs_hate_dim_size}}",
          max_interval_second = "{{fr_mgs_max_hate_interval_second}}",
          diversity_history_size = "{{fr_mgs_history_hate_size}}",
          dpp_diversity_mgs_topk = "{{fr_mgs_hate_topk}}",
          enable_dpp_diversity_mgs_ratio = "{{enable_fr_hate_mgs_ratio}}",
          dpp_diversity_mgs_ratio = "{{fr_mgs_hate_similar_ratio}}",
          enable_only_user_explore_hate = "{{enable_fr_mgs_only_user_explore_hate}}",
        ) \
        .enrich_attr_by_light_function(
          import_item_attr = [
            {"name": "hate_cover_diversity_score",  "as": "input_score"},
          ],
          export_item_attr = [
            {"name": "output_score",  "as": "hate_cover_similar_score"},
          ],
          function_name = "CalcReverseSimilarScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()