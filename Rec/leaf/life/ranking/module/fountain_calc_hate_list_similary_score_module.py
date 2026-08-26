from ranking import CommonModule

class FountainCalcHateListSimilaryScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("fountain_mc_enable_calc_neg_feedback_sim_score == 1 and enable_use_cascade_hate_similary_score_for_fullrank == 1") \
        .copy_attr(
          attrs=[{
              "from_item": "hate_similary_score",
              "to_item": "fullrank_hate_similary_score"
          }]
        ) \
      .else_() \
        .if_("skip_fullrank_hate_similary_score_in_ensemble_sort == 0") \
          .explore_embedding_candidates_attr_enricher(
            trans_type = "fountain_candidates",
            enable_fix_low_hit_rate = "{{fountain_fullrank_enable_fix_mmu_embedding_low_hit_rate}}",
            user_info_ptr_attr = "user_info_ptr",
            export_common_attr = "embedding_source_pids",
            check_point = "fullrank",
          ) \
          .get_remote_embedding_lite(
            kess_service = "{{fullrank_emb_kess_name_for_hate_similary_score}}",
            shard_num = 4,
            id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
            input_attr_name = "embedding_source_pids",
            output_attr_name = "mmu_embeddings",
            query_source_type = "common_attr",
            size = 64,
            client_side_shard = True
          ) \
          .explore_custom_embedding_score_enricher(
            check_point_ = "fullrank",
            enable_fountain_version = True,
            enable_fix_low_hit_rate = "{{fountain_fullrank_enable_fix_mmu_embedding_low_hit_rate}}",
            user_info_ptr_attr = "user_info_ptr",
            embedding_list_attr = "mmu_embeddings",
            source_pids_list_attr = "embedding_source_pids",
            calc_type = "action_bucket_dot",
            not_click_limit_hour = "{{fullrank_hate_similary_score_not_click_hour_limit}}",
            play_stat_limit_hour = "{{fullrank_hate_similary_score_play_stat_hour_limit}}",
            extra_not_click_limit_hour = "{{fullrank_hate_similary_score_extra_not_click_hour_limit}}",
            not_click_weight = "{{fullrank_hate_similary_score_not_click_weight}}",
            short_view_weight = "{{fullrank_hate_similary_score_short_view_weight}}",
            extra_not_click_weight = "{{fullrank_hate_similary_score_extra_not_click_weight}}",
            export_item_attr = "fullrank_hate_similary_score",
            dim_size = 64
          ) \
        .end_if_() \
      .end_if_()
  
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = [
          "fullrank_emb_kess_name_for_hate_similary_score"
        ],
        item_attrs = [
          "fullrank_hate_similary_score",
        ],
        for_debug_request_only = True
      )