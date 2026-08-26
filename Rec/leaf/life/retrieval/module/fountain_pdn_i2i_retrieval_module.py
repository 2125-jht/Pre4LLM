from retrieval.retrieval_module import RetrievalModule

class FountainPdnI2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .switch_("request_type") \
        .case_("fountain_fast_v1_life") \
          .explore_global_trigger_select_v2_enricher(
            user_info_attr = "user_info_ptr",
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
          .enrich_attr_by_light_function(
            import_common_attr = [
              "global_normal_trigger_list",
              "global_high_value_trigger_list",
              "global_normal_trigger_weight_list",
              "global_high_value_trigger_weight_list",
              {"name": "fountain_pdn_sample_trigger_num", "as": "sample_trigger_num"},
              {"name": "fountain_pdn_start_idx", "as": "start_idx"},
            ],
            export_common_attr = [
              {"name": "final_trigger_list", "as": "pdn_total_triggers"},
            ],
            function_name = "SampleTriggers",
            class_name = "ExploreLightFunctionSetV2"
          ) \
          .retrieve_by_redis(
            cluster_name = "recoUserPreferAuthor",
            item_separator = ",",
            attr_separator = ":",
            extra_item_attrs = [
              {"name": "redis_score"}
            ],
            key_from_attr = "pdn_total_triggers",
            key_prefix = "{{fountain_total_pdn_retr_key_prefix}}",
            retrieve_num = "{{fountain_total_pdn_retr_num}}",
            retrieve_num_per_key = "{{fountain_total_pdn_retr_num_per_key}}",
            save_src_key_to_attr = "i2i_trigger_id",
            reason = self.reason,
            timeout_ms = 50
          ) \
          .deduplicate() \
          .filter_by_common_attr(
            common_attr=["browse_screen__pid_list"]
          ) \
          .log_debug_info(
            item_attrs = [
              "i2i_trigger_id",
              "redis_score"
            ]
          ) \
      .end_()