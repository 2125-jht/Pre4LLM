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
      .retrieve_by_redis(
        reason = self.reason,
        cluster_name = "recoUserPreferAuthor",
        timeout_ms = 20,
        retrieve_num = "{{user_photo_retrieve_num}}",
        key_from_attr = "sim_user_list_gcl_u2u",
        key_prefix = "{{user_photo_redis_key_prefix_v2}}",
        retrieve_num_per_key = "{{user_photo_retrieve_num_per_key}}",
        item_separator = ",",
        save_src_key_to_attr = "sim_user_id_string"
      ) \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"]
      ) \
      .deduplicate() \
      .filter_by_browse_set(
        skip = "{{skip_browse_set}}"
      ) \
      .if_("enable_result_shuffle > 0") \
        .shuffle() \
      .end_() \
      .if_("enable_snake_merge == 1") \
        .cast_attr_type(
          attr_type_cast_configs = [
            {
              "to_type": "int",
              "from_item_attr": "sim_user_id_string",
              "to_item_attr": "sim_user_id"
            }
          ]
        ) \
        .explore_snake_merge(
          cluster_attr_name = "sim_user_id",
          max_item_num = "{{retrieve_num}}"
        ) \
      .else_() \
        .limit(size = "{{retrieve_num}}") \
      .end_()