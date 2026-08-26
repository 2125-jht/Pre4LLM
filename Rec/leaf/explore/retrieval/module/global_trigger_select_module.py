from retrieval import CommonModule

class GlobalTriggerSelectModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .if_("_USER_ID_ > 0 and enable_only_hot_list == 1 and enbable_only_hot_list_freq_control == 1") \
        .gen_common_attr_by_lua(
          attr_map = {
            "redis_key": "'i2i_hc_' .. _USER_ID_"
          }
        ) \
        .get_common_attr_from_redis(
          cluster_name = "recoExploreUserStatKiwi",
          redis_params = [
            {
              "redis_key": "{{redis_key}}",
              "output_attr_name": "count",
              "output_attr_type": "int"
            }
          ],
          is_async = True,
        ) \
        .write_to_redis(
          kcc_cluster = "recoExploreUserStatKiwi",
          key = "{{redis_key}}",
          value = "{{return (count + 1) % only_hot_list_freq_control_window_size}}",
          timeout_ms = 10,
          expire_second = "{{only_hot_list_freq_control_expire_seconds}}",
        ) \
        .if_("count ~= nil and count == only_hot_list_freq_control_nth_request") \
          .set_attr_value(
            common_attrs = [
              {
                "name": "enable_only_hot_list",
                "type": "int",
                "value": 0
              }
            ]
          ) \
        .end_() \
      .end_() \
      .explore_global_trigger_select_enricher(
        user_info_ptr_attr = "user_info_ptr",
        colossus_v2_attr = "colossus_resp_v2",
        max_playstat_trigger_num = "{{max_playstat_trigger_num}}",
        max_colossus_trigger_num = "{{max_colossus_trigger_num}}",
        max_actionlist_trigger_num = "{{max_actionlist_trigger_num}}",
        high_value_interval_size = "{{high_value_interval_size}}",
        normal_interval_size = "{{normal_interval_size}}",
        hetu_low_level_tags = "{{hetu_low_level_tags}}",
        hetu_level2_tags = "{{hetu_level2_tags}}",
        enable_click_list = "{{enable_click_list}}",
        colossus_trigger_play_time_ths = "{{colossus_trigger_play_time_ths}}",
        normal_trigger_list_attr = "global_normal_trigger_list",
        high_value_trigger_list_attr = "global_high_value_trigger_list",
        normal_trigger_weight_list_attr = "global_normal_trigger_weight_list",
        high_value_trigger_weight_list_attr = "global_high_value_trigger_weight_list",
        negative_trigger_ids = "{{negative_trigger_ids}}",
        negative_trigger_weights = "{{negative_trigger_weights}}",
        negative_trigger_min_weight = "{{negative_trigger_min_weight}}",
        prefer_trigger_ids = "{{prefer_trigger_ids}}",
        prefer_trigger_weights = "{{prefer_trigger_weights}}",
        append_prefer_trigger_num = "{{append_prefer_trigger_num}}",
        unbias_trigger_num = "{{unbias_trigger_num}}",
        unbias_trigger_ids = "{{uUnbiasTriggerPidsList}}",
        unbias_trigger_weight = "{{unbias_trigger_weight}}",
        enable_weight_adjust = "{{enable_global_trigger_weight_adjust}}",
        weight_adjust_coef = "{{trigger_weight_adjust_coef}}",
        weight_max_value = "{{trigger_weight_max_value}}",
        use_fountain_playstat = "{{use_fountain_playstat}}",
        use_fountain_action_list = "{{use_fountain_action_list}}",
        playstat_trigger_play_time_ths = "{{playstat_trigger_play_time_ths}}",
        enable_real_show_list = "{{enable_real_show_list}}",
        enable_hot_list = "{{enable_hot_list}}",
        enable_colossus_fountain_list = "{{enable_colossus_fountain_list}}",
        enable_slide_list = "{{enable_slide_list}}",
        enable_only_hot_list = "{{enable_only_hot_list}}",
        append_other_tabs_threshold = "{{append_other_tabs_threshold}}",
        append_play_stat_num = "{{append_play_stat_num}}",
        append_fountain_play_stat_num = "{{append_fountain_play_stat_num}}",
        append_only_slide_list = "{{append_only_slide_list}}",
        non_hot_trigger_ratio = "{{non_hot_trigger_ratio}}",
        playstat_same_period_trigger_ratio = "{{playstat_same_period_trigger_ratio}}",
        colossus_same_period_trigger_ratio = "{{colossus_same_period_trigger_ratio}}",
        actionlist_same_period_trigger_ratio = "{{actionlist_same_period_trigger_ratio}}",
        enable_same_period_trigger = "{{enable_same_period_trigger}}",
        same_period_time_before_present = "{{same_period_time_before_present}}",
        same_period_time_after_present = "{{same_period_time_after_present}}",
        boost_same_period_trigger_weight = "{{boost_same_period_trigger_weight}}",
        enable_search_click_photo_list = "{{enable_search_click_photo_list}}"
      )
