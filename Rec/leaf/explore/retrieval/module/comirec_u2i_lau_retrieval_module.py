from retrieval import RetrievalModule

class ComirecU2ILauRetrievaluModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .enrich_attr_by_light_function(
        import_common_attr = ["user_info_ptr"],
        export_common_attr = ["user_play_pid_list", "user_play_uid_list"],
        function_name = "CalcUserPlayGateList",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .delegate_retrieve(
        kess_service = "{{comirec_u2i_lau_retr_service_name}}",
        timeout_ms = 100,
        reason = 10048,
        request_num = "{{comirec_u2i_lau_retr_request_num}}",
        send_browse_set = True,
        send_common_attrs_in_request = False,
        send_common_attrs = [
          {"name": "comirec_u2i_lau_search_num", "as": "search_num"},
          {"name": "comirec_u2i_lau_interest_depth", "as": "interest_depth"},
          {"name": "comirec_u2i_lau_diversity_boost", "as": "diversity_boost"},
          {"name": "comirec_u2i_lau_photo_score_boost", "as": "photo_score_boost"},
          {"name": "comirec_u2i_lau_ann_src", "as": "ann_src"},
          {"name": "comirec_u2i_lau_ann_dst", "as": "ann_dst"},
          {"name": "_USER_ID_", "as": "uId"},
          {"name": "_DEVICE_ID_", "as": "dId"},
          {"name": "follow_aids", "as": "uFollowAuthors"},
          {"name": "user_play_pid_list", "as": "uProfileV1PlayTopKPidList"},
          {"name": "user_play_uid_list", "as": "uProfileV1PlayTopKAidList"}
        ],
        skip = "{{skip_comirec_u2i_lau_retr_server}}")
 
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = ["_USER_ID_"],
        print_all_item_keys = True
      )

