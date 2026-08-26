from retrieval.retrieval_module import RetrievalModule

class McU2uRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("enable_emb_server > 0") \
        .get_remote_embedding(
          kess_service = "{{embedding_service_name}}",
          shard_num = 1,
          timeout_ms = 10,
          id_converter = {
            "type_name": "mioEmbeddingIdConverter"
          },
          slot = 578,
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
      .if_("enable_ucf_user > 0") \
        .retrieve_by_redis(
          reason = 1,
          retrieve_num = "{{ucf_user_top_k}}",
          cluster_name = "recoColossusTriggers",
          timeout_ms = 10,
          key_from_attr = "_USER_ID_",
          key_prefix = "{{ucf_redis_key_prefix}}",
          item_separator = ",",
          attr_separator = ":",
          extra_item_attrs = [
            {"name": "score", "type": "double"}
          ],
          save_result_to_common_attr = "ucf_user_list"
        ) \
      .end_() \
      .retrieve_by_ann_embedding(
        reason = 2,
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
        save_result_to_common_attr = "sim_user_list_mc_u2u"
      ) \
      .if_("ucf_user_list ~= nil and #ucf_user_list > 0") \
        .pack_common_attr(
          input_common_attrs = ["ucf_user_list", "sim_user_list_mc_u2u"],
          output_common_attr = "sim_user_list_mc_u2u",
          deduplicate = True
        ) \
      .end_() \
      .if_("mc_u2u_use_new_index == 1", to_be_delete = "date=2024-05-29;committer=liubaoan") \
        .retrieve_by_remote_index(
          kess_service = "{{explore_mc_u2i_index_service}}",
          timeout_ms = 50,
          reason = self.reason,
          querys = [
            {
              "query": "usim:{{sim_user_list_mc_u2u}}",
              "search_num": 30,
              "expire_second": 1,
              "random_search": 1
            }
        ]) \
      .else_() \
        .explore_retrieve_by_redis_list_range(
          reason = self.reason,
          key_attr = "sim_user_list_mc_u2u",
          save_score_to_attr = "user_photo_score",
          cluster_name = "{{user_photo_redis_cluster_name}}",
          timeout_ms = "{{user_photo_redis_timeout_ms}}",
          key_prefix = "{{user_photo_redis_key_prefix}}"
        ) \
      .end_() \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"]
      ) \
      .filter_by_browse_set(
        skip = "{{skip_browse_set}}"
      ) \
      .sort(
        score_from_attr = "user_photo_score"
      ) \
      .limit(size = "{{retrieve_num}}")
  
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = [
          "valid_user_list"
        ],
        item_attrs = [
          "score@ucf_user_list"
        ]
      )
