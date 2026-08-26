from retrieval import CommonModule

class GlobalTriggerSelectModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .if_("fountain_enable_global_trigger_select > 0 and fountain_global_trigger_select_use_v2 == 0") \
        .explore_global_trigger_select_enricher(
          user_info_ptr_attr = "userInfoPb",
          colossus_v2_attr = "colossus_resp_v2",
          max_playstat_trigger_num = "{{fountain_max_playstat_trigger_num}}",
          max_colossus_trigger_num = "{{fountain_max_colossus_trigger_num}}",
          max_actionlist_trigger_num = "{{fountain_max_actionlist_trigger_num}}",
          hetu_low_level_tags = "{{hetu_low_level_tags}}",
          hetu_level2_tags = "{{hetu_level2_tags}}",
          normal_trigger_list_attr = "global_normal_trigger_list",
          high_value_trigger_list_attr = "global_high_value_trigger_list",
          normal_trigger_weight_list_attr = "global_normal_trigger_weight_list",
          high_value_trigger_weight_list_attr = "global_high_value_trigger_weight_list",
          use_hot_playstat = "{{fountain_golbal_trigger_use_hot_playstat}}",
          use_hot_action_list = "{{fountain_golbal_trigger_use_hot_action_list}}",
          use_fountain_playstat = "{{fountain_golbal_trigger_use_fountain_playstat}}",
          use_fountain_action_list = "{{fountain_golbal_trigger_use_fountain_action_list}}",
          playstat_trigger_play_time_ths = "{{fountain_playstat_trigger_play_time_ths}}",
          colossus_trigger_recent_preserve_num = "{{fountain_colossus_trigger_recent_preserve_num}}",
          enable_hot_list = "{{fountain_global_trigger_enable_hot_list}}",
          enable_colossus_fountain_list = "{{fountain_global_trigger_enable_colossus_fountain_list}}",
          enable_slide_list = "{{fountain_global_trigger_enable_slide_list}}"
        ) \
      .end_() \
      .if_("fountain_enable_global_trigger_select > 0 and fountain_global_trigger_select_use_v2 > 0", to_be_delete = "date=2024-05-29;committer=shaolei") \
        .explore_memory_data_enrich(
          data_key = "hetu_v1_id_mapping",
          data_type = "uint64_uint64_map",
          save_data_ptr_to_attr = "hetu_v1_id_mapping_ptr"
        ) \
        .explore_global_trigger_select_v2_enricher(
          user_info_attr = "userInfoPb",
          colossus_resp_attr = "colossus_resp_v2",
          hetu_map_ptr_attr = "hetu_v1_id_mapping_ptr",
          user_fountain_behv = "{{fountain_global_trigger_enable_user_fountain_behv}}",
          min_play_time = "{{fountain_global_trigger_min_play_time_s}}",
          play_time_weight = "{{fountain_global_trigger_play_time_weight}}",
          max_play_time_limit = "{{fountain_global_trigger_max_play_time_limit}}",
          play_ratio_weight = "{{fountain_global_trigger_play_ratio_weight}}",
          time_decay_weight = "{{fountain_global_trigger_time_decay_weight}}",
          label_weight_map = "{{fountain_global_trigger_label_weight_map}}",
          min_cluster_size = "{{fountain_global_trigger_min_cluster_size}}",
          normal_trigger_num = "{{fountain_global_trigger_normal_trigger_num}}",
          high_value_trigger_num = "{{fountain_global_trigger_high_value_trigger_num}}",
          enable_normal_shuffle = "{{fountain_global_trigger_enable_normal_shuffle}}",
          normal_trigger_attr = "global_normal_trigger_list",
          high_value_trigger_attr = "global_high_value_trigger_list",
          normal_trigger_weight_attr = "global_normal_trigger_weight_list",
          high_value_trigger_weight_attr = "global_high_value_trigger_weight_list"
        ) \
      .end_()
