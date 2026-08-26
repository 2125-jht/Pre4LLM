from ranking import CommonModule

class GenDebiasXtrModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("enable_explore_rank_debias_mix_score == 1") \
        .explore_user_debias_xtr_v2_enricher(
          colossus_v2_attr_name = "colossus_resp_v2",
          user_info_ptr_attr = "user_info_ptr",
          xtr_weight_str = "{{explore_debias_mix_xtr_weight_str}}",
          shortterm_stat_show_count = "{{explore_debias_mix_shortterm_stat_show_count}}",
          longterm_stat_click_count = "{{explore_debias_mix_longterm_stat_click_count}}",
          hetu_tag_attr = "hetu_tag_level_info__hetu_level_one",
          duration_ms_attr = "duration_ms",
          ctr_attr = "pctr",
          ltr_attr = "pltr",
          wtr_attr = "pwtr",
          ftr_attr = "pftr",
          cmtr_attr = "pcmtr",
          pptr_attr = "pptr",
          playtime_attr = "fr_score2",
          ctr_debias_attr = "pctr_debias_hetu",
          ltr_debias_attr = "pltr_debias_hetu",
          wtr_debias_attr = "pwtr_debias_hetu",
          ftr_debias_attr = "pftr_debias_hetu",
          cmtr_debias_attr = "pcmtr_debias_hetu",
          pptr_debias_attr = "pptr_debias_hetu",
          playtime_debias_attr = "fr_score2_debias_duration",
          debias_mix_score_attr = "debias_mix_score",
          page_type_attr = "explore",
          stat_only_page = "{{explore_debias_mix_stat_only_page}}",
          adjust_playtime_score = "{{explore_debias_mix_adjust_playtime_score}}",
          playtime_score_coeff_str = "{{explore_debias_mix_playtime_score_coeff_str}}",
          playtime_debias_by_duration = "{{explore_debias_mix_playtime_debias_by_duration}}",
          debias_version = "{{explore_debias_mix_debias_version}}",
          default_debias_value = "{{explore_debias_mix_debias_default_debias_value}}",
          hetu_debias_value_str = "{{explore_debias_mix_debias_hetu_debias_value_str}}"
        ) \
      .end_() \
      .if_("enable_explore_rank_debias_pctr_score == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rank_debias_pctr_default_debias_value", "as": "default_debias_value"},
            {"name": "explore_rank_debias_pctr_debias_value_str", "as": "debias_value_str"},
            {"name": "explore_rank_debias_pctr_debias_version", "as": "debias_version"},
          ],
          import_item_attr = [
            {"name": "pctr", "as": "xtr_input"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_list_input"},
            {"name": "duration_ms", "as": "duration_ms_input"},
          ],
          export_item_attr = [
            {"name": "xtr_output", "as": "pctr_debias_hetu"},
          ],
          function_name = "CalcDebiasXtr",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_rank_debias_pwatchtime_score == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rank_debias_pwatchtime_default_debias_value", "as": "default_debias_value"},
            {"name": "explore_rank_debias_pwatchtime_debias_value_str", "as": "debias_value_str"},
            {"name": "explore_rank_debias_pwatchtime_debias_version", "as": "debias_version"},
          ],
          import_item_attr = [
            {"name": "fr_score2", "as": "xtr_input"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_list_input"},
            {"name": "duration_ms", "as": "duration_ms_input"},
          ],
          export_item_attr = [
            {"name": "xtr_output", "as": "fr_score2_debias_duration"},
          ],
          function_name = "CalcDebiasXtr",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_rank_debias_pltr_score == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rank_debias_pltr_default_debias_value", "as": "default_debias_value"},
            {"name": "explore_rank_debias_pltr_debias_value_str", "as": "debias_value_str"},
            {"name": "explore_rank_debias_pltr_debias_version", "as": "debias_version"},
          ],
          import_item_attr = [
            {"name": "pltr", "as": "xtr_input"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_list_input"},
            {"name": "duration_ms", "as": "duration_ms_input"},
          ],
          export_item_attr = [
            {"name": "xtr_output", "as": "pltr_debias_hetu"},
          ],
          function_name = "CalcDebiasXtr",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_rank_debias_pwtr_score == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rank_debias_pwtr_default_debias_value", "as": "default_debias_value"},
            {"name": "explore_rank_debias_pwtr_debias_value_str", "as": "debias_value_str"},
            {"name": "explore_rank_debias_pwtr_debias_version", "as": "debias_version"},
          ],
          import_item_attr = [
            {"name": "pwtr", "as": "xtr_input"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_list_input"},
            {"name": "duration_ms", "as": "duration_ms_input"},
          ],
          export_item_attr = [
            {"name": "xtr_output", "as": "pwtr_debias_hetu"},
          ],
          function_name = "CalcDebiasXtr",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_rank_debias_pftr_score == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rank_debias_pftr_default_debias_value", "as": "default_debias_value"},
            {"name": "explore_rank_debias_pftr_debias_value_str", "as": "debias_value_str"},
            {"name": "explore_rank_debias_pftr_debias_version", "as": "debias_version"},
          ],
          import_item_attr = [
            {"name": "pftr", "as": "xtr_input"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_list_input"},
            {"name": "duration_ms", "as": "duration_ms_input"},
          ],
          export_item_attr = [
            {"name": "xtr_output", "as": "pftr_debias_hetu"},
          ],
          function_name = "CalcDebiasXtr",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_rank_debias_pcmtr_score == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rank_debias_pcmtr_default_debias_value", "as": "default_debias_value"},
            {"name": "explore_rank_debias_pcmtr_debias_value_str", "as": "debias_value_str"},
            {"name": "explore_rank_debias_pcmtr_debias_version", "as": "debias_version"},
          ],
          import_item_attr = [
            {"name": "pcmtr", "as": "xtr_input"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_list_input"},
            {"name": "duration_ms", "as": "duration_ms_input"},
          ],
          export_item_attr = [
            {"name": "xtr_output", "as": "pcmtr_debias_hetu"},
          ],
          function_name = "CalcDebiasXtr",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_rank_debias_pptr_score == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rank_debias_pptr_default_debias_value", "as": "default_debias_value"},
            {"name": "explore_rank_debias_pptr_debias_value_str", "as": "debias_value_str"},
            {"name": "explore_rank_debias_pptr_debias_version", "as": "debias_version"},
          ],
          import_item_attr = [
            {"name": "pptr", "as": "xtr_input"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_list_input"},
            {"name": "duration_ms", "as": "duration_ms_input"},
          ],
          export_item_attr = [
            {"name": "xtr_output", "as": "pptr_debias_hetu"},
          ],
          function_name = "CalcDebiasXtr",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_rank_debias_awesome_wtd_v2 == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rank_debias_awesome_wtd_default_debias_value", "as": "default_debias_value"},
            {"name": "explore_rank_debias_awesome_wtd_debias_value_str", "as": "debias_value_str"},
            {"name": "explore_rank_debias_awesome_wtd_debias_version", "as": "debias_version"},
          ],
          import_item_attr = [
            {"name": "awesome_wtd", "as": "xtr_input"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_list_input"},
            {"name": "duration_ms", "as": "duration_ms_input"},
          ],
          export_item_attr = [
            {"name": "xtr_output", "as": "awesome_wtd_debias_v2"},
          ],
          function_name = "CalcDebiasXtr",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_rank_debias_awesome_wtd == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "user_gender", "as": "basic_gender"},
          ],
          import_item_attr = [
            "hetu_tag_level_info__hetu_level_one"
          ],
          export_item_attr = [
            "awesome_wtd_debias_bucket_name",
          ],
          function_name = "GetXtrDebiasBucketName",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .get_kconf_params(
          kconf_configs = [{
            "kconf_key": "reco.offline.hetuLevelOnePlaytimeConfig",
            "json_path": "{{awesome_wtd_debias_bucket_name}}",
            "default_value": 34.64,
            "export_item_attr": "awesome_wtd_debias_bucket_score"
          }]
        ) \
        .enrich_attr_by_light_function(
          import_item_attr = [
            "pctr",
            {"name": "awesome_wtd", "as": "xtr_name"},
            {"name": "awesome_wtd_debias_bucket_score", "as": "debias_bucket_score_name"},
          ],
          import_common_attr = [
            {"name": "explore_awesome_wtd_bias_pctr_weight", "as": "xtr_bias_pctr_weight"},
            {"name": "explore_awesome_wtd_bias_alpha_weight", "as": "xtr_bias_alpha_weight"},
            {"name": "explore_awesome_wtd_bias_beta_weight", "as": "xtr_bias_beta_weight"},
            {"name": "explore_awesome_wtd_bias_max_limit", "as": "xtr_bias_max_limit"},
            {"name": "explore_awesome_wtd_bias_min_limit", "as": "xtr_bias_min_limit"}
          ],
          export_item_attr = [
            {"name": "target_queue_name", "as": "awesome_wtd_debias_score"}
          ],
          function_name = "CalculateXtrDebiasScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
    .end_() \
    .if_("enable_ranking_hetu_gender_debias_v2 == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "hetu_gender_to_norm_ctr_map_ptr",
          {"name": "basic_info_gender_v2", "as": "gender"},
        ],
        import_item_attr = [
          {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu1"},
          {"name": "corr_pctr", "as": "ctr"},
          {"name": "pltr", "as": "ltr"},
          {"name": "pftr", "as": "ftr"},
          {"name": "pcltr", "as": "cltr"},
          {"name": "pcmtr", "as": "cmtr"},
          {"name": "pwtr", "as": "wtr"},
          {"name": "duration_ms", "as": "avg_play_time_ms"},
        ],
        export_item_attr = [
          {"name": "ctr", "as": "hetu_gender_debias_ctr"},
          {"name": "ltr", "as": "hetu_gender_debias_ltr"},
          {"name": "ftr", "as": "hetu_gender_debias_ftr"},
          {"name": "cltr", "as": "hetu_gender_debias_cltr"},
          {"name": "cmtr", "as": "hetu_gender_debias_cmtr"},
          {"name": "wtr", "as": "hetu_gender_debias_wtr"},
          {"name": "avg_play_time_ms_d", "as": "hetu_gender_debias_avg_play_time_ms"}
        ],
        function_name = "HetuGenderNormDebias",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_()
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_attrs = [
          "pctr_debias_hetu",
          "pltr_debias_hetu",
          "pwtr_debias_hetu",
          "pftr_debias_hetu",
          "pcmtr_debias_hetu",
          "pptr_debias_hetu",
          "fr_score2_debias_duration",
          "awesome_wtd_debias_v2"
        ],
        for_debug_request_only = True,
      )
