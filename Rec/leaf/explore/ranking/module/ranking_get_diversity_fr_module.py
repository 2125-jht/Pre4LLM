from ranking import CommonModule

class RankingGetDiversityFrModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    # 此 module 主要用来生成在粗排第二阶段及之后使用的 diversity fr
    self.flow \
      .if_("enable_explore_diversity_update_ranking == 1") \
        .explore_diversity_update_enricher(
          user_info_ptr_attr = "user_info_ptr",
          pid_embedding_common_attr = "pid_mmu_embedding_map_ranking",
          prev_item_from_attr = "explore_realshow_click_common_list",
          cluster_id_attr = "hetu_sim_cluster_id",
          export_item_attr = "diversity_fr_ranking",
          dim_size = "{{explore_embedding_dim_size_ranking}}",
          diversity_history_size = "{{diversity_history_size_ranking}}",
          dpp_diversity_mgs_topk = "{{explore_diversity_mgs_topk_ranking}}",
          history_valid_timestamp_gap = "{{explore_diversity_history_valid_timestamp_gap}}",
          history_matrix_set_mode = "{{explore_diversity_history_matrix_set_mode}}",
          topk_select_mod = "{{explore_diversity_fr_topk_select_mod}}",
        ) \
      .end_()
