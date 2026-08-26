from retrieval.retrieval_module import RetrievalModule

class AnnU2uRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("enable_emb_server > 0") \
        .get_remote_embedding(
          kess_service = "{{embedding_service}}",
          timeout_ms = 10,
          id_converter = {
            "type_name": "mioEmbeddingIdConverter"
          },
          slot = self.slot,
          save_to_common_attr = True,
          output_item_list_attr = "valid_user_list",
          output_embedding_list_attr = "user_embedding_list",
          query_source_type = "user_id",
          is_raw_data = False,
          is_raw_data_list = False
        ) \
      .else_() \
        .copy_attr(
          attrs = [{"from_common": "mc_u2i_user_embedding_list", "to_common": "user_embedding_list"}]
        ) \
      .end_() \
      .retrieve_by_ann_embedding(
        reason = 1,
        kess_service = "{{ann_service}}",
        space = "cosine",
        timeout_ms = 50,
        items_from_attr = ["_USER_ID_"],
        embeddings_from_attr = ["user_embedding_list"],
        bound_type = {
          "top_k": "{{sim_user_num}}"
        },
        algo_type = {
          "scann": {}
        },
        src_data_type = "user",
        src_bucket = "user",
        dest_bucket = "{{ann_dest_bucket}}",
        save_result_to_common_attr = "sim_user_list"
      ) \
      .if_("shuffle_sim_user == 1") \
        .shuffle_list_attr(
          common_attr = "sim_user_list"
        ) \
      .end_() \
      .if_("user_limit_num and user_limit_num > 0") \
        .truncate(
          size_limit = "{{user_limit_num}}",
          item_list_from_attr = "sim_user_list"
        ) \
      .end_() \
      .explore_retrieve_by_redis_list_range(
        reason = self.reason,
        key_attr = "sim_user_list",
        save_score_to_attr = "user_photo_score",
        cluster_name = "{{user_photo_redis_cluster_name}}",
        retrieve_num_per_key = "{{photo_num_per_user}}",
        timeout_ms = 50,
        key_prefix = "{{user_photo_redis_key_prefix}}"
      ) \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"]
      ) \
      .filter_by_browse_set(
        skip = "{{skip_browse_set}}"
      ) \
      .sort(
        score_from_attr = "user_photo_score"
      ) \
      .limit("{{retrieve_num}}")
    
  @property
  def slot(self) -> int:
    assert "slot" in self.config
    return self.config["slot"]