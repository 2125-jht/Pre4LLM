from retrieval import RetrievalModule

class LifeActiveCollectInterestRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  @classmethod
  def is_retrieval(cls) -> bool:
    return True
    
  def process(self) -> None:
    self.flow \
      .pack_common_attr(
        input_common_attrs = [
          "collect_list"
        ],
        output_common_attr = "seed_photos"
      ) \
      .get_remote_embedding(
        kess_service = "{{life_mc_active_interest_embedding_service_name}}",
        shard_num = self.emb_server_shard_num,
        timeout_ms = 20,
        slot = 0,
        id_converter = {
          "type_name": "kuibaEmbeddingIdConverter"
        },
        item_list_from_attr = "seed_photos",
        save_to_common_attr = True,
        output_item_list_attr = "life_mc_trigger_list",
        output_embedding_list_attr = "life_mc_trigger_embedding",
        query_source_type = "item_key",
        client_side_shard = True,
        raw_data_type = "uint16",
        is_raw_data = False,
        is_raw_data_list = False,
      ) \
      .retrieve_by_ann_embedding(
        kess_service = "{{life_mc_active_interest_ann_service_name}}",
        space = "cosine",
        timeout_ms = 100,
        reason = self.reason,
        shard_num = 1,
        items_from_attr = ["life_mc_trigger_list"],
        embeddings_from_attr = ["life_mc_trigger_embedding"],
        bound_type = {
          "total_limit": "{{life_active_interest_mc_retrieval_count}}",
        },
        algo_type = {
          "scann": {},
        },
        src_bucket = "photo",
        dest_bucket = "{{life_active_interest_mc_retrieval_bucket}}",
      ) \
      .filter_by_common_attr(
        common_attr = ["seed_photos"]
      )

  @property
  def emb_server_shard_num(self) -> int:
    return self.config.get("emb_server_shard_num", 8)