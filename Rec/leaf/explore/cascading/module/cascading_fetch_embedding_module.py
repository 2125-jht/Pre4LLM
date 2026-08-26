from cascading import CommonModule

class CascadingFetchEmbeddingModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.fetch_mmu_hetu_sim_content_embedding()

  def fetch_mmu_hetu_sim_content_embedding(self):
    return self.flow \
      .if_("enable_explore_mc_fetch_mmu_embedding == 1") \
        .if_("enable_explore_mc_fetch_mmu_embedding_marketing_compensation_positive_trigger == 1") \
          .pack_item_attr(
            item_source = {
              "reco_results": True
            },
            mappings = [{
              "from_item_attr": "photo_id",
              "to_common_attr": "explore_mc_mmu_embedding_marketing_source_pids",
              "aggregator": "concat"
            }],
            target_item = {
              "is_marketing_compensation_photo": 1
            }
          ) \
          .pack_common_attr(
            input_common_attrs = [
              "explore_mc_mmu_embedding_marketing_source_pids",
              "explore_marketing_compensation_positive_trigger"
            ],
            output_common_attr = "explore_mc_mmu_embedding_marketing_source_pids",
            deduplicate = True
          ) \
        .end_() \
        .pack_common_attr(
          input_common_attrs = [
            "explore_mc_mmu_embedding_marketing_source_pids"
          ],
          output_common_attr = "explore_mc_mmu_embedding_source_pids",
          deduplicate = True
        ) \
        .if_("#(explore_mc_mmu_embedding_source_pids or {}) > 0") \
          .fetch_remote_embedding(
            kess_service = "grpc_MMUHetuSimContentEmbedding",
            shard_num = 4,
            id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
            input_attr_name = "explore_mc_mmu_embedding_source_pids",
            output_attr_name = "explore_mc_mmu_embeddings",
            query_source_type = "common_attr",
            size = 64,
            client_side_shard = True
          ) \
        .end_() \
      .end_()
