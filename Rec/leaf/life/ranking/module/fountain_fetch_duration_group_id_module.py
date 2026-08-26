from ranking import CommonModule

class FountainFetchDurationGroupIdModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def process(self) -> None:
      self.flow \
        .if_("skip_fountain_fetch_duration_group_id == 0") \
          .get_kconf_params(
            kconf_configs = [
              {
                "kconf_key": "reco.fountain.durationGroupId",
                "value_type": "list_double",
                "default_value": [],
                "export_common_attr": "faActionL2rV4DurationId_threshold_list"
              },
            ]
          ) \
          .get_kconf_params(
            kconf_configs = [
              {
                "kconf_key": "reco.fountain.fountainActionV4VtrMaxList",
                "value_type": "list_double",
                "default_value": [],
                "export_common_attr": "fountain_fullrank_ltr_v4_vtr_max_list"
              },
            ]
          ) \
          .enrich_attr_by_lua(
            import_common_attr = [
              "faActionL2rV4DurationId_threshold_list",
              "fountain_duration_s_id_max",
              "fountain_fullrank_ltr_v4_vtr_max_list",
            ],
            import_item_attr = [
              "duration_ms",
            ],
            export_item_attr = [
              "faActionL2rV4DurationId",
              "featureDurationSId",
              "fountain_act_vtr_max",
            ],
            function_for_item = "fetch_duration_group_id",
            lua_script_file = "life/ranking/lua/module/fountain_ranking_score__trans_item_attr.lua"
          ) \
        .end_if_()

