from retrieval import CommonModule

class SplashIcfEnlargeModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .enrich_attr_by_lua(
        import_common_attr = [
          "featureFountainProfileEffViewPidList",
          "featureUserHateList",
          "skip_fountain_eff_view_filter_splash",
          "colossusRetrievalTrigger",
          "skip_fountain_colossus_filter_splash"],
        export_common_attr = [
          "featureFountainProfileEffViewPidList",
          "colossusRetrievalTrigger"
        ],
        function_for_common = "filter_trigger_list_splash",
        lua_script_file = "fountain/retrieval/lua/module/icf_enlarge__calc_icf_enlarge_trigger.lua",
      )