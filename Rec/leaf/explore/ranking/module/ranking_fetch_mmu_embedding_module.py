from ranking import CommonModule

class RankingFetchMMUEmbeddingModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_get_embedding_map_ranking == 1") \
        .explore_embedding_candidates_attr_enricher(
          trans_type = "embedding_candidates",
          user_info_ptr_attr = "user_info_ptr",
          export_common_attr = "embedding_source_pids_ranking",
          check_point = "ranking"
        ) \
        .get_remote_embedding_lite(
          kess_service = "{{explore_fr_mgs_emb_kess_name}}",
          shard_num = 4,
          id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
          input_attr_name = "embedding_source_pids_ranking",
          output_attr_name = "mmu_embeddings_ranking",
          query_source_type = "common_attr",
          size = 64,
          client_side_shard = True
        ) \
        .explore_get_embedding_map_enricher(
          embedding_list_attr = "mmu_embeddings_ranking",
          source_pids_list_attr = "embedding_source_pids_ranking",
          dim_size = 64,
          export_common_attr = "pid_mmu_embedding_map_ranking",
        ) \
      .end_()
