from retrieval.retrieval_module import RetrievalModule

class McU2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  @classmethod
  def is_retrieval(cls) -> bool:
    return True
    
  def process(self) -> None:
    self.flow \
      .if_ ("user_risk_level and user_risk_level < risk_level_min or mc_u2i_user_embedding_list == nil") \
        .return_() \
      .end_() \
      .if_ ("enable_retriveal_add_neg_feedback == 0 or mc_u2i_neg_photo_embedding_list == nil or neg_pid_list == nil") \
        .retrieve_by_ann_embedding(
          kess_service = "{{ann_service_name}}",
          space = "ip",
          timeout_ms = "{{ann_service_timeout}}",
          reason = self.reason,
          shard_num = 1,
          items_from_attr = ["_USER_ID_"],
          embeddings_from_attr = ["mc_u2i_user_embedding_list"],
          bound_type = {
            "total_limit": "{{ann_service_retr_count}}",
          },
          algo_type = {
            "scann": {},
          },
          src_bucket = "{{ann_service_bucket}}",
          dest_bucket = "{{ann_service_bucket}}",
          dest_bucket_item_type = 0,
          save_distance_to_attr = "ann_dist_list"
        ) \
      .else_() \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "mc_u2i_neg_photo_embedding_list", 
            "mc_u2i_user_embedding_list", 
            "retriveal_add_neg_feedback_alpha",
            "neg_pid_list",
            "weight_list",
          ],
          export_common_attr = [
            "transformed_mc_u2i_user_embedding_list",
          ],
          function_name = "TransformMcU2iUserEmbeddings",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .retrieve_by_ann_embedding(
          kess_service = "{{ann_service_name}}",
          space = "ip",
          timeout_ms = "{{ann_service_timeout}}",
          reason = self.reason,
          shard_num = 1,
          items_from_attr = ["_USER_ID_"],
          embeddings_from_attr = ["transformed_mc_u2i_user_embedding_list"],
          bound_type = {
            "total_limit": "{{ann_service_retr_count}}",
          },
          algo_type = {
            "scann": {},
          },
          src_bucket = "{{ann_service_bucket}}",
          dest_bucket = "{{ann_service_bucket}}",
          dest_bucket_item_type = 0,
          save_distance_to_attr = "ann_dist_list"
        ) \
      .end_() \
      .deduplicate(
      ) \
      .enrich_attr_by_lua(
        import_item_attr = ["ann_dist_list"],
        export_item_attr = ["ann_dist"],
        function_for_item = "calculate",
        lua_script_file = "explore/retrieval/lua/module/mc_u2i_ann__gen_ann_dist.lua"
      ) \
      .filter_by_attr(
        attr_name = "ann_dist",
        remove_if = "<",
        compare_to = "{{ann_dist_threshold}}"
      )

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = [
          "ann_service_name",
          "mc_u2i_uid_list",
          "neg_pid_list",
          "weight_list",
          "mc_u2i_user_embedding_list",
          "transformed_mc_u2i_user_embedding_list"
        ]
      )
