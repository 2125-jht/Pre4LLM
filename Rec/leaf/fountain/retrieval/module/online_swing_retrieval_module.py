from retrieval.retrieval_module import RetrievalModule

class OnlineSwingRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .explore_custom_trim_user_info(
        user_info_attr = self.user_info_attr,
        save_trimed_user_info_to_attr = "trimed_user_info",
        trim_user_info = self.trimed_user_info_config,
        skip = "{{skip_fountain_sim_lr_user_info}}"
      ) \
      .pack_common_attr(
        input_common_attrs = ["featureFountainProfileEffViewPidList"],
        output_common_attr = "featureFountainProfileEffViewPidListSub",
        limit_num = "{{fountain_effective_view_limit_num}}",
        skip = "{{skip_fountain_effective_view_limit_num}}"
      ) \
      .pack_common_attr(
        input_common_attrs = ["featureFountainProfileLongViewPidList"],
        output_common_attr = "featureFountainProfileLongViewPidListSubs",
        limit_num = "{{fountain_longview_limit_num}}",
        skip = "{{skip_fountain_longview_limit_num}}"
      ) \
      .pack_common_attr(
        input_common_attrs = ["colossusRetrievalTrigger"],
        output_common_attr = "colossusRetrievalTriggerSub",
        limit_num = "{{fountain_colossus_trigger_limit_num}}",
        skip = "{{skip_fountain_colossus_limit_num}}"
      ) \
      .if_("skip_fountain_online_swing_retr == 0") \
        .pack_common_attr(
          input_common_attrs = ["featureFountainProfileEffViewPidListSub",   
                                "featureFountainProfileLongViewPidListSubs",
                                "colossusRetrievalTriggerSub"],
          output_common_attr = "online_swing_trigger",
          limit_num = "{{fountain_online_swing_trigger_limit}}",
          deduplicate = True
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "online_swing_trigger", "as": "origin_int_list"},
            {"name": "fountain_online_swing_trigger_seperator", "as": "list_seperator"}
          ],
          export_common_attr = [
            {"name": "joined_str", "as": "swing_trigger"}
          ],
          function_name = "IntListCommonAttr2String",
          class_name = "ExploreLightFunctionSetV2") \
        .delegate_retrieve(
          kess_service = "{{fountain_online_swing_retr_kess_service_name}}",
          timeout_ms = 100,
          request_num = "{{fountain_online_swing_retr_request_num}}",
          send_common_attrs_in_request = False,
          send_common_attrs = [
            {"name": "trimed_user_info", "as": "user"},
            {"name": "fountain_online_swing_retr_custom_kconf_key", "as": "custom_kconf_key"},
            {"name": "swing_trigger", "as": "custom_trigger"}
          ],
          reason = self.reason
        ) \
      .end_()

  @property
  def trimed_user_info_config(self) -> list:
    return self.config.get("trimed_user_info")
  
  @property
  def user_info_attr(self) -> str:
    return self.config.get("user_info_attr", "userInfo")