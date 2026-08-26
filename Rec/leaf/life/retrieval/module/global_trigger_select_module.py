from retrieval import CommonModule

class GlobalTriggerSelectModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .if_("enable_global_trigger_select > 0") \
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
          enable_weight_adjust = "{{enable_global_trigger_weight_adjust}}",
          weight_adjust_coef = "{{trigger_weight_adjust_coef}}",
          weight_max_value = "{{trigger_weight_max_value}}",
          use_fountain_playstat = "{{use_fountain_playstat}}",
          use_fountain_action_list = "{{use_fountain_action_list}}",
          playstat_trigger_play_time_ths = "{{playstat_trigger_play_time_ths}}",
          enable_real_show_list = "{{enable_real_show_list}}"
        ) \
      .end_() \
      .if_("enable_global_trigger_select <= 0 and enable_global_trigger_select_v2 > 0") \
        .explore_global_trigger_select_v2_enricher(
          user_info_attr = "user_info_ptr",
          colossus_resp_attr = "colossus_resp_v2",
          hetu_map_ptr_attr = "hetu_v1_id_mapping_ptr",
          min_play_time = "{{min_trigger_play_time_s}}",
          play_time_weight = "{{trigger_play_time_weight}}",
          max_play_time_limit = "{{trigger_max_play_time_limit}}",
          play_ratio_weight = "{{trigger_play_ratio_weight}}",
          time_decay_weight = "{{trigger_time_decay_weight}}",
          label_weight_map = "{{trigger_label_weight_map}}",
          min_cluster_size = "{{trigger_min_cluster_size}}",
          normal_trigger_num = "{{normal_trigger_num}}",
          high_value_trigger_num = "{{high_value_trigger_num}}",
          enable_normal_shuffle = "{{enable_normal_shuffle}}",
          normal_trigger_attr = "global_normal_trigger_list",
          high_value_trigger_attr = "global_high_value_trigger_list",
          normal_trigger_weight_attr = "global_normal_trigger_weight_list",
          high_value_trigger_weight_attr = "global_high_value_trigger_weight_list"
        ) \
      .end_() \
      .if_("use_explore_trigger_selected == 1") \
        .explore_trigger_selected_enricher(
          user_info_ptr_attr = "user_info_ptr",
          colossus_resp_attr = "colossus_resp_v2",
          enable_life_hetu_restrict = "{{enable_life_hetu_restrict}}",
          final_item_list_attr = "explore_selected_trigger_list",
          life_restrict_hetu_level1 = "{{life_restrict_hetu_level1}}"
        ) \
        .log_debug_info(
          common_attrs = [
            "explore_selected_trigger_list"
          ]
        ) \
      .end_()