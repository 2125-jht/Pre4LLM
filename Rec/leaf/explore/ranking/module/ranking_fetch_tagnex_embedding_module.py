from ranking import CommonModule

class RankingFetchTagNexEmbeddingModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_get_tagnex_embedding_map_ranking == 1") \
        .explore_embedding_candidates_attr_enricher(
          trans_type = "embedding_candidates",
          user_info_ptr_attr = "user_info_ptr",
          export_common_attr = "tagnex_embeddings_source_pids",
          check_point = "ranking"
        ) \
        .get_remote_embedding_lite_v2(
          protocol = 1,
          colossusdb_embd_service_name = "grpc_clsdb_ps-mmu-tagnex-emb",
          colossusdb_embd_table_name = "emb_mmu_tagnex_explore",
          id_converter = {"type_name": "mioEmbeddingIdConverter"},
          slot = 0,
          input_attr_name = "tagnex_embeddings_source_pids",
          output_attr_name = "tagnex_embeddings_ranking",
          query_source_type = "common_attr",
          raw_data_type = "float32",
          colossusdb_use_kconf_client = False,
          size = 128,
          client_side_shard = True
        ) \
      .end_()
