from retrieval.retrieval_module import RetrievalModule

class GraphEmbeddingRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .gen_common_attr_by_lua(
        attr_map = {
          "cand_num": "retrieve_num * 2",
        }
      ) \
      .explore_custom_trim_user_info(
        user_info_attr = "userInfo",
        save_trimed_user_info_to_attr = "trimedUserInfo",
        trim_user_info = [
          "user_profile_v1.real_show_list",
          "user_profile_v1.click_list"
        ]
      ) \
      .if_("trimedUserInfo ~= nil") \
        .delegate_retrieve(
          kess_service = "{{service_name}}",
          timeout_ms = "{{timeout_ms}}",
          reason = self.reason,
          request_type = "default",
          request_num = "{{cand_num}}",
          send_common_attrs = [
            {"name": "trimedUserInfo", "as": "user"},
            {"name": "bucket_name", "as": "dest_bucket"},
            {"name": "browse_screen__pid_list", "as": "browsed_pids"}
          ]
        ) \
      .else_() \
        .delegate_retrieve(
          kess_service = "{{service_name}}",
          timeout_ms = "{{timeout_ms}}",
          reason = self.reason,
          request_type = "default",
          request_num = "{{cand_num}}",
          send_common_attrs = [
            {"name": "userInfo", "as": "user"},
            {"name": "bucket_name", "as": "dest_bucket"},
            {"name": "browse_screen__pid_list", "as": "browsed_pids"}
          ]
        ) \
      .end_() \
      .deduplicate()