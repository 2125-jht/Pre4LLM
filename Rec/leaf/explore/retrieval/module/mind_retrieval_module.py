from retrieval import RetrievalModule

class MindRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .get_remote_embedding(
        kess_service = "{{remote_embedding_kess_name}}",
        shard_num = 1,
        timeout_ms = "{{service_timeout_ms}}",
        id_converter = {
          "type_name": "kuibaEmbeddingIdConverter"
        },
        save_to_common_attr = True,
        output_item_list_attr = "",
        output_embedding_list_attr = "k_user_vec_",
        query_source_type = "user_id",
        client_side_shard = False,
        raw_data_type = "uint64",
        is_raw_data=False,
        is_raw_data_list=False
      ) \
      .if_("k_user_vec_ == nil or #k_user_vec_ <= 0 or #k_user_vec_ % 64 ~= 0") \
        .return_() \
      .end_() \
      .enrich_attr_by_lua(
        import_common_attr = ["_USER_ID_", "k_user_vec_"],
        export_common_attr = ["mind_emb_uid_list"],
        function_for_common = "calculate",
        lua_script_file = "explore/retrieval/lua/module/mind_retr__gen_uid_list.lua"
      ) \
      .retrieve_by_ann_embedding(
        kess_service = "{{ann_kess_service}}",
        space = "ip",
        timeout_ms = "{{service_timeout_ms}}",
        reason = self.reason,
        shard_num = 1,
        items_from_attr = ["mind_emb_uid_list"],
        embeddings_from_attr = ["k_user_vec_"], 
        bound_type = {
          "total_limit": "{{service_request_num}}",
        },
        algo_type = {
          "faiss": {}
        },
        src_bucket = "item",
        dest_bucket = "item",
        dest_bucket_item_type = 0
      ) \
      .deduplicate(
      ) \
  
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = [
          "mind_emb_uid_list",
          "k_user_vec_",
        ],
        for_debug_request_only = True,
      )