from retrieval import CommonModule

class ReTargetTrigger(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .if_("fountain_enable_get_retarget_trigger == 1") \
        .split_string(
          input_common_attr = "fountain_retarget_interest_positive_weights_str",
          output_common_attr = "fountain_retarget_interest_positive_weights_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_double = True,
        ) \
        .explore_colossus_v2_trigger_enrich(
          colossus_resp_attr = "colossus_resp_v2",
          enable_default_select_triggers = "{{fountain_retarget_enable_default_select_triggers}}",
          enable_interest_score_triggers = "{{fountain_retarget_enable_interest_score_triggers}}",
          enable_select_explore_colossus_list = "{{fountain_retarget_enable_select_explore_colossus_list}}",
          enable_select_bottom_colossus_list = "{{fountain_retarget_enable_select_bottom_colossus_list}}",
          enable_select_outer_colossus_list = "{{fountain_retarget_enable_select_outer_colossus_list}}",
          enable_select_fountain_colossus_list = "{{fountain_retarget_enable_select_fountain_colossus_list}}",
          enable_not_get_shortview_trigger = "{{fountain_retarget_enable_not_get_shortview_trigger}}",
          enable_only_get_effview_trigger = "{{fountain_retarget_enable_only_get_effview_trigger}}",
          enable_get_interact_trigger = "{{fountain_retarget_enable_get_interact_trigger}}",
          different_signals_triggers_min_minutes_ago = "{{fountain_retarget_different_signals_triggers_min_minutes_ago}}",
          different_signals_triggers_max_minutes_ago = "{{fountain_retarget_different_signals_triggers_max_minutes_ago}}",
          different_signals_interact_triggers_min_minutes_ago = "{{fountain_retarget_different_signals_interact_triggers_min_minutes_ago}}",
          different_signals_interact_triggers_max_minutes_ago = "{{fountain_retarget_different_signals_interact_triggers_max_minutes_ago}}",
          different_signals_interest_triggers_select_num = "{{fountain_retarget_different_signals_interest_triggers_select_num}}",
          different_signals_interest_interact_triggers_select_num = "{{fountain_retarget_different_signals_interest_interact_triggers_select_num}}",
          enable_filter_hate = "{{fountain_retarget_enable_filter_hate}}",
          retarget_play_time_ratio = "{{fountain_retarget_play_time_ratio}}",
          positive_weights = "{{fountain_retarget_interest_positive_weights_list}}",
          output_colossus_trigger_attr = "fountain_retarget_interest_colossus_trigger_list",
          output_colossus_trigger_positive_score_attr = "fountain_retarget_interest_colossus_trigger_weight_list"
        ) \
        .copy_item_meta_info(
          save_item_id_to_attr = "item_id",
          item_list_from_attr = "fountain_retarget_interest_colossus_trigger_list"
        ) \
      .end_() \
      .if_("fountain_enable_get_retarget_history_list == 1") \
        .split_string(
          input_common_attr = "fountain_retatget_history_vv_list_channel_str",
          output_common_attr = "fountain_retatget_history_vv_list_channel_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "colossus_resp_v2",
            {"name": "fountain_retatget_history_vv_list_limit", "as": "limit_num"},
            {"name": "fountain_retatget_history_vv_list_begin_minute", "as": "begin_minute"},
            {"name": "fountain_retatget_history_vv_list_end_minute", "as": "end_minute"},
            {"name": "fountain_retatget_history_vv_list_channel_list", "as": "channel_list"},
          ],
          export_common_attr = [
            {"name": "history_pid_list", "as": "fountain_retatget_history_vv_list"}
          ],
          function_name = "GetHistoryPidList",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .copy_item_meta_info(
          save_item_id_to_attr = "item_id",
          item_list_from_attr = "fountain_retatget_history_vv_list"
        ) \
      .end_()
