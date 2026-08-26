from ranking import CommonModule

class RankingCprHandleModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
    .if_("enable_change_cpr_value == 1") \
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
    .end_()