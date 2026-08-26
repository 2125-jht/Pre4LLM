from ranking import CommonModule

class RankingFetchCoverEmbeddingModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_fr_get_embedding_map_ranking == 1") \
        .explore_embedding_candidates_attr_enricher(
          trans_type = "embedding_candidates",
          user_info_ptr_attr = "user_info_ptr",
          export_common_attr = "source_pids_for_cover_embedding",
          check_point = "ranking"
        ) \
        .get_remote_embedding_lite(
          kess_service = "grpc_MMUExploreCoverEmbedding",
          shard_num = 4,
          id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
          input_attr_name = "source_pids_for_cover_embedding",
          output_attr_name = "mmu_cover_embeddings",
          query_source_type = "common_attr",
          size = 64,
          client_side_shard = True
        ) \
        .explore_get_embedding_map_enricher(
          embedding_list_attr = "mmu_cover_embeddings",
          source_pids_list_attr = "source_pids_for_cover_embedding",
          dim_size = 64,
          export_common_attr = "pid_mmu_cover_embedding_map_ranking",
        ) \
      .end_()