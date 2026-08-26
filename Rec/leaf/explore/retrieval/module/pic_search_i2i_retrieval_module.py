from retrieval import RetrievalModule

class PicSearchI2IRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
    
  def process(self) -> None:
    self.flow \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "search_click_list",
          "search_click_list_timestamps",
          "pic_search_i2i_time_min_thres",
          "pic_search_i2i_trigger_size",
          "search_play_pid_list",
          "search_play_timestamp_list",
          "search_play_video_duration_list",
          "search_play_time_list",
          {"name": "enable_explore_pic_search_i2i_add_search_play_trigger", "as": "enable_add_search_play_trigger"},
          {"name": "explore_pic_search_i2i_play_time_thresh", "as": "play_time_thresh"}
        ],
        export_common_attr = [
          "trigger_list"
        ],
        function_name = "GenPicSearchTriggerList",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .copy_user_meta_info(
        save_request_type_to_attr="request_type",
      ) \
      .delegate_retrieve(
        reason=self.reason,
        kess_service="{{service_name}}",
        timeout_ms=50,
        request_type="{{request_type}}",
        request_num="{{retrieve_num}}",
        send_common_attrs_in_request=False,
        send_common_attrs=["trigger_list"]
      ) \
      .set_attr_value(
        item_attrs = [
          {
            "name": "is_pic_search",
            "type": "int",
            "value": 1
          }
        ]
      )
