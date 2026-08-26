from retrieval import CommonModule

class ColossusParseModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("enable_parse == 1") \
        .reverse_list_attr( #注意reverse后action是按照timestamp由大到小取max_len个
          common_attrs = [
            "colossus_all_photo_id_list",
            "colossus_all_play_time_list",
            "colossus_all_label_list",
            "colossus_all_author_id_list",
            "colossus_all_channel_list",
            "colossus_all_duration_list",
            "colossus_all_timestamp_list",
            "colossus_all_tag_list"
          ]
        ) \
        .pack_common_attr(
          input_common_attrs = [
            "colossus_all_photo_id_list",
          ],
          output_common_attr = "colossus_photo_id_list",
          limit_num = "{{max_len}}",
        ) \
        .pack_common_attr(
          input_common_attrs = [
            "colossus_all_play_time_list",
          ],
          output_common_attr = "colossus_play_time_list",
          limit_num = "{{max_len}}",
        ) \
        .pack_common_attr(
          input_common_attrs = [
            "colossus_all_label_list",
          ],
          output_common_attr = "colossus_label_list",
          limit_num = "{{max_len}}",
        ) \
        .pack_common_attr(
          input_common_attrs = [
            "colossus_all_author_id_list",
          ],
          output_common_attr = "colossus_author_id_list",
          limit_num = "{{max_len}}",
        ) \
        .pack_common_attr(
          input_common_attrs = [
            "colossus_all_channel_list",
          ],
          output_common_attr = "colossus_channel_list",
          limit_num = "{{max_len}}",
        ) \
        .pack_common_attr(
          input_common_attrs = [
            "colossus_all_duration_list",
          ],
          output_common_attr = "colossus_duration_list",
          limit_num = "{{max_len}}",
        ) \
        .pack_common_attr(
          input_common_attrs = [
            "colossus_all_timestamp_list",
          ],
          output_common_attr = "colossus_timestamp_list",
          limit_num = "{{max_len}}",
        ) \
        .pack_common_attr(
          input_common_attrs = [
            "colossus_all_tag_list",
          ],
          output_common_attr = "colossus_tag_list",
          limit_num = "{{max_len}}",
        ) \
      .end_() \
      .if_("enable_explore_marketing_compensation_positive_trigger == 1") \
        .explore_colossus_v2_trigger_enrich(
          colossus_resp_attr = "colossus_resp_v2",
          output_colossus_trigger_attr = "explore_marketing_compensation_positive_trigger",
          enable_default_select_triggers = "{{enable_explore_default_select_marketing_compensation_positive_triggers}}",
          enable_different_signals_triggers = "{{enable_explore_different_signals_marketing_compensation_positive_triggers}}",
          different_signals_triggers_select_num = "{{explore_different_signals_marketing_compensation_positive_triggers_select_num}}",
          different_signals_triggers_min_play_time = "{{explore_different_signals_marketing_compensation_positive_triggers_min_play_time}}",
          different_signals_triggers_play_time_ratio = "{{explore_different_signals_marketing_compensation_positive_triggers_play_time_ratio}}",
          different_signals_triggers_min_days_ago = "{{explore_different_signals_marketing_compensation_positive_triggers_min_days_ago}}",
          different_signals_triggers_max_days_ago = "{{explore_different_signals_marketing_compensation_positive_triggers_max_days_ago}}",
          enable_different_signals_triggers_action_explore_list = "{{enable_different_signals_marketing_compensation_positive_triggers_action_explore_list}}",
          enable_different_signals_triggers_action_completion_list = "{{enable_explore_different_signals_marketing_compensation_positive_triggers_action_completion_list}}",
          enable_different_signals_triggers_action_interact_list = "{{enable_explore_different_signals_marketing_compensation_positive_triggers_action_interact_list}}",
          enable_different_signals_triggers_action_timestamp_order = "{{enable_different_signals_marketing_compensation_positive_triggers_action_timestamp_order}}",
          enable_not_select_bottom_selection_page = "{{enable_explore_marketing_compensation_positive_triggers_not_select_bottom_selection_page}}",
          enable_only_select_explore_colossus_list = "{{enable_explore_marketing_compensation_positive_triggers_only_select_explore_colossus_list}}",
          enable_only_select_high_interest_tab = "{{enable_explore_marketing_compensation_positive_triggers_only_select_high_interest_tab}}",
          enable_select_high_interest_and_profile_tab = "{{enable_explore_marketing_compensation_positive_triggers_select_high_interest_and_profile_tab}}",
          enable_only_select_fountain_colossus_list =  "{{enable_explore_marketing_compensation_positive_triggers_only_select_fountain_colossus_list}}",
          enable_only_unselect_explore_colossus_list =  "{{enable_explore_marketing_compensation_positive_triggers_only_unselect_explore_colossus_list}}",
          enable_only_unselect_fountain_colossus_list =  "{{enable_explore_marketing_compensation_positive_triggers_only_unselect_fountain_colossus_list}}",
          enable_get_longview_trigger = "{{enable_explore_get_longview_marketing_compensation_positive_trigger}}",
        ) \
        .gen_common_attr_by_lua(
          attr_map={
            "explore_marketing_compensation_positive_trigger_size": "#(explore_marketing_compensation_positive_trigger or {})",
          }
        ) \
      .end_() \
      .if_("enable_explore_get_recent_interest_trigger == 1") \
        .split_string(
          input_common_attr = "explore_recent_interest_trigger_label_weights_str",
          output_common_attr = "explore_recent_interest_trigger_label_weights_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_double = True,
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "colossus_all_photo_id_list", "colossus_all_play_time_list", "colossus_all_label_list",
            "colossus_all_channel_list", "colossus_all_duration_list", "colossus_all_timestamp_list",
            {"name": "explore_recent_interest_trigger_min_minutes_ago", "as": "recent_interest_min_minutes_ago"},
            {"name": "explore_recent_interest_trigger_max_minutes_ago", "as": "recent_interest_max_minutes_ago"},
            {"name": "explore_recent_interest_trigger_max_history_size", "as": "recent_interest_max_history_size"},
            {"name": "enable_explore_recent_interest_trigger_select_explore_colossus_list", "as": "enable_select_explore"},
            {"name": "enable_explore_recent_interest_trigger_select_fountain_colossus_list", "as": "enable_select_fountain"},
            {"name": "enable_explore_recent_interest_trigger_select_bottom_colossus_list", "as": "enable_select_bottom"},
            {"name": "enable_explore_recent_interest_trigger_select_outer_colossus_list", "as": "enable_select_outer"},
            {"name": "enable_explore_recent_interest_trigger_filter_shortview", "as": "enable_filter_shortview"},
            {"name": "enable_explore_recent_interest_trigger_filter_hate", "as": "enable_filter_hate"},
            {"name": "enable_explore_recent_interest_trigger_only_select_picture", "as": "enable_only_select_picture"},
            {"name": "explore_recent_interest_trigger_trigger_select_num", "as": "trigger_select_num"},
            {"name": "explore_recent_interest_trigger_label_weights_list", "as": "label_weights"},
          ],
          export_common_attr = [
            {"name": "recent_interest_trigger_pid_list", "as": "explore_recent_interest_colossus_trigger_list"},
            {"name": "recent_interest_trigger_weight_list", "as": "explore_recent_interest_colossus_trigger_weight_list"}
          ],
          function_name = "GetRecentInterestList",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()