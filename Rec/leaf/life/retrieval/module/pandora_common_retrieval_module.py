from retrieval.retrieval_module import RetrievalModule

class PandoraCommonRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "retrieve_num", "as": "origin_size"},
          {"name": "increase_quota_status", "as": "increase_quota_status"},
          {"name": "increase_quota_factor", "as": "factor"}
        ],
        export_common_attr = [
          {"name": "final_size", "as": "retrieve_num"}
        ],
        function_name = "IncreaseQuotaProcess",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .set_attr_value(
        common_attrs=[
          {
            "name": "reason",
            "type": "int",
            "value": self.reason
          }
        ],
        skip = "{{skip_personal_quota}}"
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "retrieve_num", "as": "retr_size"},
          "reason_ratio_map_attr",
          "reason"
        ],
        export_common_attr = [
          {"name": "retr_size", "as": "retrieve_num"}
        ],
        function_name = "DynamicRetrQuota",
        class_name = "ExploreLightFunctionSetV2",
        skip = "{{skip_personal_quota}}"
      )
    self.flow \
      .split_string(
        input_common_attr = "auxiliary_arg",
        output_common_attr = "auxiliary_arg_list",
        delimiters = ",",
        trim_spaces = True
      ) \
      .enrich_attr_by_lua(
        import_common_attr = ["auxiliary_arg_list"],
        export_common_attr = ["interest_depth", "diversity_boost", "photo_score_boost", "ann_src", "ann_dst"],
        function_for_common = "prepare_request_params",
        lua_script_file = "life/retrieval/lua/module/comirec_retr__preprocess.lua"
      ) \
      .explore_custom_trim_user_info(
        user_info_attr = "userInfo",
        save_trimed_user_info_to_attr = "trimedUserInfo",
        trim_user_info = [
          "id",
          "device_id",
          "browsed_photo_ids", 
          "slide_browsed_photo_ids",
          "user_profile_v1.click_list",
          "user_profile_v1.like_list",
          "user_profile_v1.follow_list",
          "user_profile_v1.forward_list",
          "user_profile_v1.collect_list",
          "user_profile_v1.comment_list",
          "user_profile_v1.profile_enter_list",
          "user_profile_v1.video_playing_stat"
        ]
      ) \
      .if_("trimedUserInfo ~= nil") \
        .delegate_retrieve(
          kess_service = "{{service_name}}",
          timeout_ms = "{{service_timeout_ms}}",
          reason = self.reason,
          request_num = "{{retrieve_num}}",
          send_common_attrs_in_request = False,
          send_common_attrs = [
            {"name": "interest_depth", "as": "interest_depth"},
            {"name": "diversity_boost", "as": "diversity_boost"},
            {"name": "photo_score_boost", "as": "photo_score_boost"},
            {"name": "ann_src", "as": "ann_src"},
            {"name": "ann_dst", "as": "ann_dst"},
            {"name": "trimedUserInfo", "as": "user"},
          ],
          reset_item_type = 0
        ) \
      .else_() \
        .delegate_retrieve(
          kess_service = "{{service_name}}",
          timeout_ms = "{{service_timeout_ms}}",
          reason = self.reason,
          request_num = "{{retrieve_num}}",
          send_common_attrs_in_request = False,
          send_common_attrs = [
            {"name": "interest_depth", "as": "interest_depth"},
            {"name": "diversity_boost", "as": "diversity_boost"},
            {"name": "photo_score_boost", "as": "photo_score_boost"},
            {"name": "ann_src", "as": "ann_src"},
            {"name": "ann_dst", "as": "ann_dst"},
            {"name": "userInfo", "as": "user"},
          ],
          reset_item_type = 0
        ) \
      .end_() \
      .deduplicate()
