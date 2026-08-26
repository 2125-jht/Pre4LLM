from retrieval import CommonModule

class PicGlobalTriggerSelectModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .if_("enable_pic_global_trigger_select > 0") \
        .explore_global_trigger_select_enricher(
          user_info_ptr_attr = "user_info_ptr",
          colossus_v2_attr = "colossus_resp_v2",
          max_playstat_trigger_num = "{{pic_max_playstat_trigger_num}}",
          max_colossus_trigger_num = "{{pic_max_colossus_trigger_num}}",
          max_actionlist_trigger_num = "{{pic_max_actionlist_trigger_num}}",
          high_value_interval_size = "{{pic_high_value_interval_size}}",
          normal_interval_size = "{{pic_normal_interval_size}}",
          hetu_low_level_tags = "{{hetu_low_level_tags}}",
          hetu_level2_tags = "{{hetu_level2_tags}}",
          enable_click_list = "{{pic_enable_click_list}}",
          colossus_trigger_play_time_ths = "{{pic_colossus_trigger_play_time_ths}}",
          normal_trigger_list_attr = "pic_global_normal_trigger_list",
          high_value_trigger_list_attr = "pic_global_high_value_trigger_list",
          normal_trigger_weight_list_attr = "pic_global_normal_trigger_weight_list",
          high_value_trigger_weight_list_attr = "pic_global_high_value_trigger_weight_list",
          negative_trigger_ids = "{{negative_trigger_ids}}",
          negative_trigger_weights = "{{negative_trigger_weights}}",
          negative_trigger_min_weight = "{{pic_negative_trigger_min_weight}}",
          prefer_trigger_ids = "{{prefer_trigger_ids}}",
          prefer_trigger_weights = "{{prefer_trigger_weights}}",
          append_prefer_trigger_num = "{{pic_append_prefer_trigger_num}}",
          unbias_trigger_num = "{{pic_unbias_trigger_num}}",
          unbias_trigger_ids = "{{uUnbiasTriggerPidsList}}",
          unbias_trigger_weight = "{{pic_unbias_trigger_weight}}",
          enable_weight_adjust = "{{pic_enable_global_trigger_weight_adjust}}",
          weight_adjust_coef = "{{pic_trigger_weight_adjust_coef}}",
          weight_max_value = "{{pic_trigger_weight_max_value}}",
          use_fountain_playstat = "{{pic_use_fountain_playstat}}",
          use_fountain_action_list = "{{pic_use_fountain_action_list}}",
          playstat_trigger_play_time_ths = "{{pic_playstat_trigger_play_time_ths}}",
          enable_real_show_list = "{{pic_enable_real_show_list}}",
          enable_hot_list = "{{pic_enable_hot_list}}",
          enable_colossus_fountain_list = "{{pic_enable_colossus_fountain_list}}",
          enable_slide_list = "{{pic_enable_slide_list}}",
          enable_only_hot_list = "{{pic_enable_only_hot_list}}"
        ) \
      .end_()