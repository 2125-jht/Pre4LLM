from rerank import CommonModule

class RerankCalPage1TriggerScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow.if_("explore_enable_sim_matrix_combo == 1") \
      .switch_("explore_rerank_hetu_emb_switch") \
        .case_(0) \
          .set_attr_value(
            common_attrs = [
              {
                "name": "explore_rerank_dpp_sim_matrix_dim",
                "type": "int",
                "value": 64
              }
            ]
          ) \
          .get_remote_embedding_lite(
            kess_service = "{{explore_emb_kess_name_for_dpp_emb}}",
            shard_num = 4,
            id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
            output_attr_name = "explore_dpp_emb",
            size = 64,
            client_side_shard = True
          ) \
        .case_(1) \
          .set_attr_value(
            common_attrs = [
              {
                "name": "explore_rerank_dpp_sim_matrix_dim",
                "type": "int",
                "value": 128
              }
            ]
          ) \
          .get_remote_embedding_lite(
            protocol = 1,
            shard_num = 8,
            colossusdb_embd_service_name = "explore_reco_hetu_emb_v4",
            colossusdb_embd_table_name = "explore_reco_hetu_emb_v4",
            id_converter = {"type_name": "mioEmbeddingIdConverter"},
            output_attr_name = "explore_dpp_emb",
            size = 128,
            client_side_shard = True,
          ) \
        .case_(2) \
          .set_attr_value(
            common_attrs = [
              {
                "name": "explore_rerank_dpp_sim_matrix_dim",
                "type": "int",
                "value": 128
              }
            ]
          ) \
          .get_remote_embedding_lite(
            protocol = 1,
            shard_num = 8,
            colossusdb_embd_service_name = "explore_hetu_emb_server_v51",
            colossusdb_embd_table_name = "explore_hetu_emb_server_v51",
            id_converter = {"type_name": "mioEmbeddingIdConverter"},
            output_attr_name = "explore_dpp_emb",
            size = 128,
            client_side_shard = True,
          ) \
      .end_() \
    .end_() \

    self.flow.if_("explore_rerank_hetu_emb_switch == 0 and explore_enable_calc_page1_trigger_similarity_score == 1 and page_index == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "user_info_ptr",
        ],
        export_common_attr = [
          "page1_trigger_ids",
        ],
        function_name = "GenExplorePageOneTriggers",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .limit(
        size = "{{explore_rerank_page1_trigger_ids_size}}",
        item_list_from_attr = "page1_trigger_ids",
      ) \
      .get_remote_embedding_lite(
        kess_service = "{{explore_emb_kess_name_for_dpp_emb}}",
        id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
        input_attr_name = "page1_trigger_ids",
        output_attr_name = "page1_trigger_embeddings",
        query_source_type = "common_attr",
        size = 64,
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "page1_trigger_ids", "as": "trigger_list"},
          {"name": "page1_trigger_embeddings", "as": "trigger_embedding_list"},
          {"name": "explore_rerank_dpp_emb_dim", "as": "dim"}
        ],
        export_common_attr = [
          {"name": "trigger_list", "as": "valid_page1_trigger_ids"},
          {"name": "trigger_embedding_list", "as": "valid_page1_trigger_embeddings"}
        ],
        function_name = "GetValidEmbeddings",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .explore_rerank_calc_page1_trigger_score_enricher(
        embedding_list_common_attr = "valid_page1_trigger_embeddings",
        pid_list_common_attr = "valid_page1_trigger_ids",
        export_item_attr = "page1_trigger_score",
        candidates_embedding_item_attr = "explore_dpp_emb",
        trigger_size = "{{explore_rerank_page1_trigger_ids_final_size}}",
        sim_weight = "{{explore_rerank_page1_trigger_sim_weight}}",
        boost_top_n = "{{explore_rerank_page1_trigger_boost_top_n}}",
        boost_score = "{{explore_rerank_page1_trigger_boost_score}}",
      ) \
    .end_()