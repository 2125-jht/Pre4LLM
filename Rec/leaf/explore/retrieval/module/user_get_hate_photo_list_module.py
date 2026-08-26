from retrieval import CommonModule

class UserGetHatePhotoListModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("enable_explore_sim_user_hate_item_list_similarity_score == 1 and recent_hate_count <= explore_koc_htr_count_threshold") \
        .delegate_retrieve(
          kess_service = "grpc_LLMU2URetrAndFixedMiddleLowUserActionListV2",
          request_type = "default",
          timeout_ms = 20,
          send_browse_set = False,
          send_common_attrs_in_request = False,
          recv_common_attrs = [{"name": "llm_sim_user_list", "as": "llm_sim_hate_user_list"}],
          send_common_attrs = [
            {"name": "explore_llm_hate_u2u_similar_user_num", "as": "llm_u2u_sim_top_k"},
            {"name": "explore_llm_hate_u2u_recall_sim_score_thres", "as": "explore_u2u_recall_sim_score_thres"},
          ],
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "user_hate_photo_id_map_ptr",
            "llm_sim_hate_user_list",
          ],
          export_common_attr = [
            "sim_user_hate_photo_id_list",
          ],
          function_name = "GetSimUserHatePhotoIdList",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()
