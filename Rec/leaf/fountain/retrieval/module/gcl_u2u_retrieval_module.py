from retrieval.retrieval_module import RetrievalModule

class GclU2uRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .get_remote_embedding_lite(
        kess_service = "{{fountain_gcl_u2u_retr_emb_service_name}}",
        shard_num = 1,
        timeout_ms = 15,
        id_converter = {
          "type_name": "mioEmbeddingIdConverter"
        },
        size = 128,
        slot = 6,
        query_source_type = "user_id",
        output_attr_name = "gcl_user_embedding_list",
        skip = "{{skip_fountain_gcl_u2u_retr}}"
      ) \
      .retrieve_by_ann_embedding(
        reason = 1, # 级联召回第一步统一reason统一修改为1
        kess_service = "{{fountain_gcl_u2u_retr_ann_service_name}}",
        space = "cosine",
        timeout_ms = 50,
        items_from_attr = ["_USER_ID_"],
        embeddings_from_attr = ["gcl_user_embedding_list"],
        bound_type = {
          "top_k": "{{fountain_gcl_u2u_retr_sim_user_num}}"
        },
        algo_type = {
          "scann": {}
        },
        src_data_type = "user",
        src_bucket = "user",
        dest_bucket = "user_bucket",
        save_result_to_common_attr = "gcl_sim_user_list",
        skip = "{{return (not gcl_user_embedding_list) or skip_fountain_gcl_u2u_retr}}"
      ) \
      .shuffle_list_attr(
        common_attr = "gcl_sim_user_list",
        skip = "{{return (not gcl_user_embedding_list) or skip_fountain_gcl_u2u_retr}}"
      ) \
      .retrieve_by_redis(
        reason = self.reason,
        cluster_name = "recoUserPreferAuthor",
        timeout_ms = 30,
        retrieve_num = "{{fountain_gcl_u2u_retr_result_num}}",
        key_from_attr = "gcl_sim_user_list",
        key_prefix = "{{fountain_gcl_u2u_retr_key_prefix}}",
        retrieve_num_per_key = "{{fountain_gcl_u2u_retr_retrieve_num_per_key}}",
        item_separator = ",",
        skip = "{{return (not gcl_user_embedding_list) or skip_fountain_gcl_u2u_retr}}"
      )
