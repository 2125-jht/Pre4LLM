from retrieval.retrieval_module import RetrievalModule

class GclU2uRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .get_remote_embedding_lite(
        kess_service = "{{embedding_service_name}}",
        shard_num = 1,
        timeout_ms = 15,
        id_converter = {
          "type_name": "mioEmbeddingIdConverter"
        },
        size = 128,
        slot = 6,
        query_source_type = "user_id",
        output_attr_name = "user_embedding_list"
      ) \
      .if_("user_embedding_list == nil") \
        .return_() \
      .end_() \
      .retrieve_by_ann_embedding(
        reason = 1,
        kess_service = "{{ann_service_name}}",
        space = "cosine",
        timeout_ms = "{{ann_timeout_ms}}",
        items_from_attr = ["_USER_ID_"],
        embeddings_from_attr = ["user_embedding_list"],
        bound_type = {
          "top_k": "{{ann_user_top_k}}"
        },
        algo_type = {
          "scann": {}
        },
        src_data_type = "{{ann_src_data_type}}",
        src_bucket = "{{ann_src_data_type}}",
        dest_bucket = "{{ann_dest_bucket}}",
        save_result_to_common_attr = "sim_user_list_gcl_u2u"
      ) \
      .if_("enable_sim_user_shuffle > 0 and sim_user_list_gcl_u2u ~= nil and #sim_user_list_gcl_u2u > 0") \
        .shuffle_list_attr(
          common_attr = "sim_user_list_gcl_u2u"
        ) \
        .limit(
          size = "{{sim_user_limit_num}}",
          item_list_from_attr = "sim_user_list_gcl_u2u"
        ) \
      .end_() \
      .if_("gcl_u2u_use_new_index == 1") \
        .retrieve_by_remote_index(
          kess_service = "{{explore_gcl_u2i_index_service}}",
          timeout_ms = 50,
          reason = self.reason,
          querys = [
            {
              "query": "usim:{{sim_user_list_gcl_u2u}}",
              "search_num": 30,
              "expire_second": 1,
              "random_search": 1
            }
        ]) \
      .else_() \
        .if_("enable_u2i_list_v2 > 0") \
          .retrieve_by_redis(
            reason = self.reason,
            cluster_name = "recoUserPreferAuthor",
            timeout_ms = 20,
            retrieve_num = "{{user_photo_retrieve_num}}",
            key_from_attr = "sim_user_list_gcl_u2u",
            key_prefix = "{{user_photo_redis_key_prefix_v2}}",
            retrieve_num_per_key = "{{user_photo_retrieve_num_per_key}}",
            item_separator = ","
          ) \
        .else_() \
          .explore_retrieve_by_redis_list_range(
            reason = self.reason,
            key_attr = "sim_user_list_gcl_u2u",
            save_score_to_attr = "user_photo_score",
            cluster_name = "{{user_photo_redis_cluster_name}}",
            timeout_ms = "{{user_photo_redis_timeout_ms}}",
            key_prefix = "{{user_photo_redis_key_prefix}}"
          ) \
        .end_() \
      .end_() \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"]
      ) \
      .deduplicate() \
      .filter_by_browse_set(
        skip = "{{skip_browse_set}}"
      ) \
      .if_("enable_result_shuffle > 0") \
        .shuffle() \
      .else_() \
        .sort(
          score_from_attr = "user_photo_score"
        ) \
      .end_() \
      .limit(size = "{{retrieve_num}}")
