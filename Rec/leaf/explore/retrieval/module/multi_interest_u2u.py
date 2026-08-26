from retrieval.retrieval_module import RetrievalModule

class MultiInterestU2uRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("_USER_ID_ == 0") \
        .return_() \
      .end_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "_USER_ID_", "as": "user_id"},
          "interest_num",
        ],
        export_common_attr = [
          "encoded_uid_lists",
        ],
        function_name = "MultiInterestEncodeUid",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .if_("enable_splited_emb > 0") \
        .get_remote_embedding_lite(
          kess_service="{{embedding_service}}",
          timeout_ms=10,
          query_source_type="common_attr",
          input_attr_name="encoded_uid_lists",
          output_attr_name="user_embedding_list",
          id_converter={"type_name": "kuibaEmbeddingIdConverter"},
          slot=101,
          size=64,
        ) \
      .else_() \
        .get_remote_embedding_lite(
          kess_service="{{embedding_service}}",
          timeout_ms=10,
          query_source_type="user_id",
          output_attr_name="user_embedding_list",
          id_converter={"type_name": "kuibaEmbeddingIdConverter"},
          slot=7,
          size=512,
        ) \
      .end_() \
      .if_ ("user_embedding_list == nil") \
        .return_() \
      .end_() \
      .retrieve_by_ann_embedding(
        reason = self.reason,
        kess_service = "{{ann_service}}",
        space = "cosine",
        timeout_ms = 50,
        items_from_attr = ["encoded_uid_lists"],
        embeddings_from_attr = ["user_embedding_list"],
        bound_type = {
          "top_k": "{{sim_user_num}}"
        },
        algo_type = {
          "scann": {}
        },
        src_data_type = "user",
        dest_bucket = "user",
        src_bucket = "user",
        save_result_to_common_attr = "sim_user_list"
      ) \
      .if_ ("sim_user_list == nil") \
        .return_() \
      .end_() \
      .deduplicate(
        item_list_from_attr = "sim_user_list",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "sim_user_list", 
        ],
        export_common_attr = [
          "decoded_sim_user_list",
        ],
        function_name = "MultiInterestDecodeUid",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .if_("enable_sim_user_shuffle > 0 and decoded_sim_user_list ~= nil and #decoded_sim_user_list > 0") \
        .shuffle_list_attr(
          common_attr = "decoded_sim_user_list"
        ) \
        .limit(
          size = "{{sim_user_limit_num}}",
          item_list_from_attr = "decoded_sim_user_list"
        ) \
      .end_() \
      .switch_("redis_type") \
      .case_(0) \
        .explore_retrieve_by_redis_list_range(
          reason = self.reason,
          key_attr = "decoded_sim_user_list",
          save_score_to_attr = "user_photo_score",
          cluster_name = "{{user_photo_redis_cluster_name}}",
          retrieve_num_per_key = "{{photo_num_per_user}}",
          timeout_ms = 50,
          key_prefix = "{{user_photo_redis_key_prefix}}"
        ) \
        .sort(
          score_from_attr = "user_photo_score"
        ) \
      .case_(1) \
        .retrieve_by_redis(
          cluster_name = "recoUserPreferAuthor",
          retrieve_num = "{{redis_retrieve_num}}",
          timeout_ms = 50,
          key_from_attr = "decoded_sim_user_list",
          key_prefix = "{{user_photo_redis_key_prefix_v1}}",
          retrieve_num_per_key = "{{photo_num_per_user}}",
          item_separator = ",",
          reason = self.reason,
        ) \
      .case_(2) \
        .return_() \
      .end_() \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"]
      ) \
      .deduplicate() \
      .if_("enable_result_shuffle > 0") \
        .shuffle() \
      .end_() \
      .limit("{{retrieve_num}}")
