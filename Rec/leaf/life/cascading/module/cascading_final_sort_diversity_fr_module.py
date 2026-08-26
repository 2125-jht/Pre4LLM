from cascading import CommonModule

class CascadingFinalSortDiversityFrModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    # 此 module 主要用来生成在粗排第二阶段及之后使用的 diversity fr
    self.flow \
      .if_("enable_skip_xlife_get_embedding_map == 0 or enable_skip_xlife_get_embedding_map_for_mgs_hate == 0") \
        .explore_get_embedding_map_enricher(
          embedding_list_attr = "mmu_embeddings",
          source_pids_list_attr = "embedding_source_pids",
          dim_size = 64,
          export_common_attr = "pid_mmu_embedding_map",
        ) \
      .end_() \
      .if_("enable_skip_xlife_diversity_update == 0") \
        .explore_life_diversity_update_enricher(
          user_info_ptr_attr = "user_info_ptr",
          pid_embedding_common_attr = "pid_mmu_embedding_map",
          export_item_attr = "diversity_fr",
          export_common_attr = "xlife_unclk_pid_list",
          history_matrix_set_mode = "{{xlife_history_list_set_mode}}",
          topk_select_mod = "{{xlife_topk_select_mod}}",
          dim_size = "{{xlife_embedding_dim_size}}",
          diversity_history_size = "{{xlife_diversity_history_size}}",
          dpp_diversity_mgs_topk = "{{xlife_diversity_mgs_topk}}",
          enable_use_explore_history = "{{xlife_diversity_enable_use_explore_history}}"
        ) \
      .end_() \
      .if_("enable_skip_xlife_topkmsg_hate_score == 0") \
        .explore_life_diversity_update_enricher(
          user_info_ptr_attr = "user_info_ptr",
          pid_embedding_common_attr = "pid_mmu_embedding_map",
          export_item_attr = "hate_fr",
          history_feed_back_version = 2,
          max_interval_second = "{{xlife_topk_mgs_max_hate_interval_second}}",
          dim_size = "{{xlife_embedding_dim_size}}",
          diversity_history_size = "{{xlife_topk_msg_hate_history_size}}",
          dpp_diversity_mgs_topk = "{{xlife_hate_mgs_topk}}",
          enable_dpp_diversity_mgs_ratio = "{{enable_xlife_dpp_diversity_mgs_ratio}}",
          dpp_diversity_mgs_ratio = "{{xlife_dpp_diversity_mgs_ratio}}",
          enable_only_user_explore_hate = "{{enable_only_user_xlife_hate}}",
          enable_only_user_explore_cover_hate = "{{enable_only_user_xlife_cover_hate}}" 
        ) \
      .end_()

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_attrs = ["diversity_fr"],
        common_attrs = ["xlife_unclk_pid_list"],
        for_debug_request_only = True
      )