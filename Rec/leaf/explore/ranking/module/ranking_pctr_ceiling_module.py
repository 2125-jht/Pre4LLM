from ranking import CommonModule

class RankingPctrCeilingModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
    .enrich_attr_by_light_function(
      import_common_attr = [
        "user_emp_ctr",
        {"name":"explore_ranking_pctr_upper_bound_bias", "as": "pctr_upper_bound_bias"}
      ],
      import_item_attr = [
        "corr_pctr"
      ],
      export_item_attr = [
        "ceiled_pctr_score"
      ],
      function_name = "PctrCeilingCal",
      class_name = "ExploreLightFunctionSetV2",
    )
