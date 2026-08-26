from ranking import CommonModule

class RankingCprHandleModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_change_cpr_value == 1") \
        .if_("enable_cpr_frac_debias == 1") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "explore_pftr_fractile_score_attr_from_redis_ptr", "as": "cpr_map_ptr"},
              {"name": "explore_fr_cpr_fracs_pctr_weight", "as": "pctr_weight"},
              {"name": "explore_fr_cpr_fracs_redis_prefix", "as": "prefix"},
              {"name": "active_days_gt_5min_rate", "as": "active_days"},
              {"name": "explore_fr_active_days_split_conf", "as": "active_days_split_conf"},
              {"name": "explore_fr_cpr_fracs_duration_ms_upper_bound", "as": "duration_ms_upper_bound"}
            ],
            import_item_attr = [
              "duration_ms",
              "cpr",
              "corr_pctr",
            ],
            export_item_attr = [
              "corr_cpr",
            ],
            function_name = "GetCorrCprByFrac",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .else_() \
          .enrich_attr_by_light_function(
            import_item_attr = [
              "duration_ms"
            ],
            export_item_attr = [
              "duration_str"
            ],
            function_name = "DurationToStr",
            class_name = "ExploreLightFunctionSetV2"
          ) \
          .get_kconf_params(
            kconf_configs = [{
              "kconf_key": "reco.offline.cprDurationHandle",
              "json_path": "{{duration_str}}",
              "default_value": 0.05,
              "export_item_attr": "cpr_duration_multi"
            }]
          ) \
          .enrich_attr_by_light_function(
            import_item_attr = [
              "cpr",
              "cpr_duration_multi",
              "corr_pctr"
            ],
            import_common_attr = [
              "explore_cpr_corr_alpha",
              "explore_cpr_corr_beta",
              "explore_cpr_corr_pctr_weight",
              "explore_cpr_corr_cpr_weight"
            ],
            export_item_attr = [
              "corr_cpr"
            ],
            function_name = "GetCorrCpr",
            class_name = "ExploreLightFunctionSetV2"
          ) \
        .end_() \
      .end_()