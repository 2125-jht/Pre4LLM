from retrieval.retrieval_module import RetrievalModule

class FocalU2uRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .get_remote_embedding(
        kess_service = "{{fountain_focal_u2u_emb_service}}",
        timeout_ms = 10,
        id_converter = {
          "type_name": "mioEmbeddingIdConverter"
        },
        slot = 100,
        save_to_common_attr = True,
        output_item_list_attr = "focal_user_id",
        output_embedding_list_attr = "focal_user_embedding",
        query_source_type = "user_id",
        is_raw_data = False,
        is_raw_data_list = False,
        skip = "{{skip_fountain_focal_u2u_retr}}"
      ) \
      .retrieve_by_ann_embedding(
        reason = 1, # 级联召回第一步统一reason统一为1
        kess_service = "{{fountain_focal_u2u_ann_service}}",
        space = "cosine",
        timeout_ms = 40,
        items_from_attr = ["focal_user_id"],
        embeddings_from_attr = ["focal_user_embedding"],
        bound_type = {
          "top_k": "{{fountain_focal_u2u_user_num}}"
        },
        algo_type = {
          "scann": {}
        },
        src_data_type = "user",
        src_bucket = "user",
        dest_bucket = "user",
        save_result_to_common_attr = "focal_user_list",
        skip = "{{skip_fountain_focal_u2u_retr}}"
      ) \
      .explore_retrieve_by_redis_list_range(
        reason = self.reason,
        key_attr = "focal_user_list",
        cluster_name = "{{fountain_user_photo_redis_cluster_name}}",
        key_prefix = "{{fountain_user_photo_redis_key_prefix}}",
        timeout_ms = 50,
        retrieve_num_per_key = "{{fountain_focal_u2u_num_per_user}}",
        retrieve_num = "{{fountain_focal_u2u_cand_num}}",
        skip = "{{skip_fountain_focal_u2u_retr}}"
      )
