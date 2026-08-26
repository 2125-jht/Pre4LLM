from retrieval.retrieval_module import RetrievalModule

class LifeMcU2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .get_remote_embedding_lite(
        kess_service = "{{fountain_life_user_embedding_service_name}}",
        shard_num = 1,
        timeout_ms = 10,
        id_converter = {
          "type_name": "kuibaEmbeddingIdConverter"
        },
        size = 128,
        output_attr_name = "fountain_mc_u2i_user_embedding_list",
        query_source_type = "user_id",
        client_side_shard = True
      ) \
      .retrieve_by_ann_embedding(
        kess_service = "{{fountain_life_mc_u2i_ann_service_name}}",
        space = "ip",
        timeout_ms = 100,
        reason = self.reason,
        shard_num = 1,
        items_from_attr = ["_USER_ID_"],
        embeddings_from_attr = ["fountain_mc_u2i_user_embedding_list"],
        bound_type = {
        "total_limit": "{{fountain_life_mc_u2i_ann_service_retr_count}}",
        },
        algo_type = {
        "scann": {},
        },
        src_bucket = "{{fountain_life_mc_u2i_ann_service_bucket}}",
        dest_bucket = "{{fountain_life_mc_u2i_ann_service_bucket}}",
        dest_bucket_item_type = 0
      )