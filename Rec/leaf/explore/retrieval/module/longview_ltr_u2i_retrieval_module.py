from retrieval.retrieval_module import RetrievalModule

class LongviewLtrU2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .if_("tag_click_retrieval_use_kai == 1") \
        .explore_common_user_feature_enricher(
          user_info_attr = "user_info_ptr",
          user_uid_attr = "uId",
          user_did_attr = "deviceId",
          user_city_attr = "cityId",
          user_profilev1_click_pids_attr = "clickList",
          user_profilev1_like_pids_attr = "likeList",
          user_profilev1_follow_aids_attr = "followAuthorList",
          user_gender_attr = "uGender",
          user_basic_age_attr = "uAgeSeg",
          user_city_level_attr = "cityLevel",
          user_is_douyin_attr = "uIsDouYin",
          user_visit_mod_attr = "uMod",
          user_longview_action_attr = "longviewList"
        ) \
        .delegate_retrieve(
          kess_service = "{{tower_ltr_u2i_retr_kess_name}}",
          timeout_ms = "{{retr_timeout_ms}}",
          reason = self.reason,
          request_num = "{{tower_ltr_u2i_retr_request_num}}",
          send_browse_set = True,
          send_common_attrs_in_request = False,
          send_common_attrs = ["uId", "deviceId", "uGender", "cityId", "cityLevel", "uAgeSeg",
                           "clickList", "likeList", "followAuthorList", "longviewList", "uMod", "uIsDouYin"]
        ) \
      .else_() \
        .explore_user_feature_common_attr_enrich(
          user_info_ptr_attr = "user_info_ptr",
          output_common_attr_list_attr = "user_feature_name_list"
        ) \
        .get_kuiba_user_embedding(
          tensor_request_layer = "{{infer_request_layer}}",
          kess_service = "{{infer_service_name}}",
          timeout_ms = 50,
          input_common_attr = "{{user_feature_name_list}}",
          output_tensor_attr = "ltr_user_embedding"
        ) \
        .retrieve_by_ann_embedding(
          reason = self.reason,
          kess_service = "{{retr_service_name}}",
          space = "{{retr_space}}",
          timeout_ms = "{{retr_timeout_ms}}",
          items_from_attr = ["_USER_ID_"],
          embeddings_from_attr = ["ltr_user_embedding"],
          bound_type = {
            "total_limit": "{{retr_total_limit}}"
          },
          algo_type = {
            "scann": {},
          },
          src_data_type = "{{retr_src_data_type}}",
          src_bucket = "{{retr_src_data_type}}",
          dest_bucket = "{{retr_dest_bucket}}"
        ) \
        .deduplicate() \
      .end_()

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = ["ltr_user_embedding"]
      )
