from retrieval import CommonModule

class SplashUserRelevancePreferenceCalcModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .if_("enable_fountain_splash_cacl_long_term_relevance_preference_weight == 1") \
        .get_kconf_params(
          kconf_configs = [
            {
              "kconf_key": "reco.fountain.relatedScoreConfig",
              "json_path": "relevance_preference_score_by_hetu",
              "default_value": [
                0.522122,0.554315,0.667171,0.579611,0.430956,0.678762,0.675772,0.246868,0.728291,0.537961,
                0.608603,0.609079,0.666419,0.474237,0.634214,0.617335,0.665180,0.566264,0.473103,0.601703,
                0.723034,0.660349,0.507028,0.651612,0.680084,0.369982,0.537818,0.761581,0.713508,0.645662,
                0.5,0.5,0.5,0.356862,0.764598,0.585746,0.695644,0.62538,0.5,0.6
              ], # hetu v1
              "export_common_attr": "fountain_splash_relevance_preference_hetu1_adjust_weight_list"
            },
            {
              "kconf_key": "reco.fountain.relatedScoreConfig",
              "json_path": "relevance_preference_score_global",
              "default_value": 0.5957,
              "export_common_attr": "fountain_splash_relevance_preference_global_adjust_weight"
            },
          ]
        ) \
        .gen_common_attr_by_lua(
          attr_map={
            "source_hetu_level_one_first_index": "(source_hetu_level_one and #source_hetu_level_one > 0) and (source_hetu_level_one[1] - 1) or -1",
          }
        ) \
        .select_list_values(
          index_attr = "source_hetu_level_one_first_index",
          list_values = [
            {"from": "fountain_splash_relevance_preference_hetu1_adjust_weight_list", "to": "fountain_splash_relevance_preference_hetu1_adjust_weight"},
          ],
          is_common_attr=True
        ) \
        .explore_memory_data_enrich(
          data_key = "highest_level_hetu_tag_map_to_level_one",
          data_type = "uint64_uint64_map",
          save_data_ptr_to_attr = "highest_level_hetu_tag_map_to_level_one_ptr",
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_splash_cacl_long_term_relevance_preference_seq_size", "as": "seq_size"},
            {"name": "fountain_splash_relevance_preference_global_adjust_weight", "as": "global_relevance_preference_adjust_weight"},
            {"name": "fountain_splash_relevance_preference_hetu1_adjust_weight", "as": "global_hetu1_relevance_preference_adjust_weight"},
            {"name": "fountain_splash_cacl_long_term_relevance_preference_play_cnt_threshold", "as": "play_cnt_threshold"},
            {"name": "fountain_splash_cacl_long_term_relevance_preference_fusion_type", "as": "fusion_type"},
            {"name": "fountain_splash_cacl_long_term_relevance_preference_alpha", "as": "alpha"},
            {"name": "fountain_splash_cacl_long_term_relevance_preference_beta", "as": "beta"},
            {"name": "fountain_splash_relevance_preference_adjust_weight_lower_bound", "as": "relevance_preference_weight_lower_bound"},
            {"name": "fountain_splash_relevance_preference_adjust_weight_upper_bound", "as": "relevance_preference_weight_upper_bound"},
            "colossus_photo_id_list",
            "colossus_tag_list",
            "colossus_play_time_list",
            "colossus_label_list",
            "colossus_channel_list",
            "colossus_duration_list",
            "colossus_timestamp_list",
            "highest_level_hetu_tag_map_to_level_one_ptr",
            "source_hetu_level_one",
          ],
          export_common_attr = [
            {"name": "preference_score", "as": "user_long_term_relevance_preference_weight"},
          ],
          function_name = "CalcSplashLongTermRelevancePreference",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()