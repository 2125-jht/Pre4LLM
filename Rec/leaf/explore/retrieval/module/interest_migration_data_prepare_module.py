from retrieval import CommonModule

class InterestMigrationDataPrepareModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .if_("enable_frist_screen_customization_interest_migration_data_prerare == 1 and is_first_refresh == 1") \
        .copy_attr(
          attrs=[
            {
              "from_common": "enable_interest_migration_data_prepare_frist_screen",
              "to_common": "enable_interest_migration_data_prepare"
            },
            {
              "from_common": "interest_migration_time_second_upper_frist_screen",
              "to_common": "interest_migration_time_second_upper"
            },
            {
              "from_common": "interest_migration_realshow_num_limit_frist_screen",
              "to_common": "interest_migration_realshow_num_limit"
            },
            {
              "from_common": "interest_migration_ignore_channel_list_str_frist_screen",
              "to_common": "interest_migration_ignore_channel_list_str"
            },
            {
              "from_common": "interest_migration_hot_cnt_threshold_frist_screen",
              "to_common": "interest_migration_hot_cnt_threshold"
            },
            {
              "from_common": "interest_migration_hot_rate_threshold_frist_screen",
              "to_common": "interest_migration_hot_rate_threshold"
            },
            {
              "from_common": "interest_migration_playtime_weight_frist_screen",
              "to_common": "interest_migration_playtime_weight"
            },
            {
              "from_common": "interest_migration_not_effective_view_weight_frist_screen",
              "to_common": "interest_migration_not_effective_view_weight"
            },
            {
              "from_common": "interest_migration_like_weight_frist_screen",
              "to_common": "interest_migration_like_weight"
            },
            {
              "from_common": "interest_migration_follow_weight_frist_screen",
              "to_common": "interest_migration_follow_weight"
            },
            {
              "from_common": "interest_migration_forward_weight_frist_screen",
              "to_common": "interest_migration_forward_weight"
            },
            {
              "from_common": "interest_migration_comment_weight_frist_screen",
              "to_common": "interest_migration_comment_weight"
            },
            {
              "from_common": "interest_migration_profile_weight_frist_screen",
              "to_common": "interest_migration_profile_weight"
            },
            {
              "from_common": "interest_migration_follow_weight_frist_screen",
              "to_common": "interest_migration_follow_weight"
            },
            {
              "from_common": "interest_migration_collection_weight_frist_screen",
              "to_common": "interest_migration_collection_weight"
            },
            {
              "from_common": "interest_migration_long_rate_vv_threshold_frist_screen",
              "to_common": "interest_migration_long_rate_vv_threshold"
            },
            {
              "from_common": "interest_migration_vv_rate_weight_frist_screen",
              "to_common": "interest_migration_vv_rate_weight"
            },
            {
              "from_common": "interest_migration_play_time_rate_weight_attr_frist_screen",
              "to_common": "interest_migration_play_time_rate_weight_attr"
            }
          ]
        ) \
      .end_() \
      .if_("enable_interest_migration_data_prepare == 1") \
        .split_string(
          input_common_attr = "interest_migration_ignore_channel_list_str",
          output_common_attr = "interest_migration_ignore_channel_list",
          delimiters = ",", 
          skip_empty_tokens = True,
          trim_spaces = True,
          parse_to_int = True 
        ) \
        .explore_interest_migration_history_prepare_enricher(
          time_second_upper = "{{interest_migration_time_second_upper}}",
          colossus_v2_attr_name = "colossus_resp_v2",
          user_info_ptr_name = "user_info_ptr",
          ignore_channel_name = "interest_migration_ignore_channel_list",
          output_id_list_name = "interest_migration_pids",
          output_score_list_name = "interest_migration_scores",
          output_realshow_list_name = "explore_realshow_pids",
          output_is_degraded_flag_name = "interest_migration_is_degraded",
          colossus_num_limit_attr = "{{interest_migration_colossus_num_limit}}",
          realshow_num_limit_attr = "{{interest_migration_realshow_num_limit}}",
          hot_cnt_threshold_attr = "{{interest_migration_hot_cnt_threshold}}",
          hot_rate_threshold_attr = "{{interest_migration_hot_rate_threshold}}",
          playtime_weight = "{{interest_migration_playtime_weight}}",
          not_effective_view_weight = "{{interest_migration_not_effective_view_weight}}",
          like_weight_attr = "{{interest_migration_like_weight}}",
          follow_weight_attr = "{{interest_migration_follow_weight}}",
          forward_weight_attr = "{{interest_migration_forward_weight}}",
          comment_weight_attr = "{{interest_migration_comment_weight}}",
          profile_weight_attr = "{{interest_migration_profile_weight}}",
          collection_weight_attr = "{{interest_migration_collection_weight}}",
          short_rate_hour_upper_attr = "{{interest_migration_short_hour_upper}}",
          short_rate_cnt_threshold_attr = "{{interest_migration_short_rate_cnt_threshold}}",
          long_rate_vv_threshold_attr = "{{interest_migration_long_rate_vv_threshold}}",
          vv_rate_weight_attr = "{{interest_migration_vv_rate_weight}}",
          play_time_rate_weight_attr = "{{interest_migration_play_time_rate_weight_attr}}",
          active_rate_weight_attr = "{{interest_migration_active_rate_weight_attr}}",
          output_user_page_prefer_score_name = "user_page_prefer_score",
          enable_collection_list_attr = "{{interest_migration_enable_collection_list}}",
          bs_short_view_mins_upper = "{{interest_migration_bs_short_view_mins_upper}}",
          bs_short_view_num_upper = "{{interest_migration_bs_short_view_num_upper}}",
        ) \
      .end_() \
      .if_("explore_enable_partial_time_based_interest_triggers == 1") \
        .explore_colossus_v2_trigger_enrich(
          colossus_resp_attr = "colossus_resp_v2",
          output_colossus_trigger_attr = "partial_time_based_selected_pids",
          enable_default_select_triggers = False,
          enable_partial_time_based_interest_triggers = "{{explore_enable_partial_time_based_interest_triggers}}",
          max_days_ago = "{{explore_partial_time_based_interest_range_max_days}}",
          trigger_select_num = "{{explore_partial_time_based_interest_range_max_num}}",
          trigger_min_play_time = "{{explore_partial_time_based_interest_play_time_ths}}",
          enable_only_explore_data = "{{explore_partial_time_based_interest_enable_only_explore_data}}",
          cur_time_mode = "{{explore_partial_time_based_interest_cur_time_mode}}",
        ) \
      .end_() \
      .set_attr_value(
        common_attrs=[
          {
            "name": "interest_score_based_valid_user",
            "type": "int",
            "value": 1
          }
        ]
      ) \
      .if_("enable_explore_interest_score_based_select_valid_user == 1 and active_days_gt_5min_rate > interest_score_based_valid_user_5min_rate") \
        .set_attr_value(
          common_attrs=[
            {
              "name": "interest_score_based_valid_user",
              "type": "int",
              "value": 0
            }
          ]
        ) \
      .end_() \
      .if_("enable_explore_interest_score_based_interest_triggers == 1 and interest_score_based_valid_user == 1") \
        .split_string(
          input_common_attr = "explore_interest_score_based_positive_weights_str",
          output_common_attr = "explore_interest_score_based_positive_weights",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_double = True,
        ) \
        .explore_colossus_v2_trigger_enrich(
          colossus_resp_attr = "colossus_resp_v2",
          output_colossus_trigger_attr = "interest_score_based_pids",
          output_colossus_trigger_positive_score_attr = "interest_score_based_scores",
          output_colossus_trigger_channel_attr = "interest_score_based_channels",
          enable_default_select_triggers = "{{enable_interest_score_based_colossus_default_select}}",
          enable_interest_score_triggers = "{{enable_interest_score_based_triggers}}",
          enable_select_explore_colossus_list = "{{explore_interest_score_based_enable_select_explore_colossus_list}}",
          enable_select_fountain_colossus_list = "{{explore_interest_score_based_enable_select_fountain_colossus_list}}",
          enable_select_bottom_colossus_list = "{{explore_interest_score_based_enable_select_bottom_colossus_list}}",
          enable_select_outer_colossus_list = "{{explore_interest_score_based_enable_select_outer_colossus_list}}",
          enable_not_get_shortview_trigger = "{{explore_interest_score_based_enable_not_get_shortview_trigger}}",
          enable_only_get_effview_trigger = "{{explore_interest_score_based_enable_only_get_effview_trigger}}",
          enable_get_interact_trigger = "{{explore_interest_score_based_enable_get_interact_trigger}}",
          enable_filter_hate = "{{explore_interest_score_based_enable_filter_hate}}",
          different_signals_triggers_min_minutes_ago = "{{explore_interest_score_based_different_signals_triggers_min_minutes_ago}}",
          different_signals_triggers_max_minutes_ago = "{{explore_interest_score_based_different_signals_triggers_max_minutes_ago}}",
          different_signals_interact_triggers_min_minutes_ago = "{{explore_interest_score_based_different_signals_interact_triggers_min_minutes_ago}}",
          different_signals_interact_triggers_max_minutes_ago = "{{explore_interest_score_based_different_signals_interact_triggers_max_minutes_ago}}",
          different_signals_interest_triggers_select_num = "{{explore_interest_score_based_different_signals_interest_triggers_select_num}}",
          different_signals_interest_interact_triggers_select_num = "{{explore_interest_score_based_different_signals_interest_interact_triggers_select_num}}",
          colossus_limit_num = "{{explore_interest_score_based_colossus_limit_num}}",
          retarget_play_time_ratio = "{{explore_interest_score_based_retarget_play_time_ratio}}",
          positive_weights = "{{explore_interest_score_based_positive_weights}}"
        ) \
      .end_()
