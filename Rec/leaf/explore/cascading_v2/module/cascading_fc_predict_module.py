from cascading_v2 import CommonModule

class CascadingFcPredictModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_mc_fc_predict == 1") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            "photo_id",
            "author__id",
            "tag",
            "duration_ms",
            "upload_time"
          ],
          export_item_attr = [
            {"name":"context_slots", "as":"hot_fc_car_slots"},
            {"name":"context_signs", "as":"hot_fc_car_signs"}
          ],
          function_name = "GenCARSigns",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .extract_with_ks_sign_feature(
          extractor_kconf_path = "reco.hot.fountainLeafMcFeature",
          caller_model = "{{mc_fc_predict_service}}",
          user_info_attr = "user_info_ptr",
          common_slots_output = "fc_sign_common_slots",
          common_parameters_output = "fc_sign_common_parameters",
        ) \
        .delegate_enrich(
          kess_service = "{{mc_fc_predict_service}}",
          request_type = "{{mc_fc_request_type}}",
          timeout_ms = 100,
          send_common_attrs = [
            { "name": "fc_sign_common_slots", "as": "user_feature_slots" },
            { "name": "fc_sign_common_parameters", "as": "user_feature_signs" },
          ],
          send_item_attrs = [
            "hot_fc_car_slots",
            "hot_fc_car_signs"
          ], 
          recv_item_attrs = [
            {"name": "fc_pctr_value", "as": "cascade_pctr"},
            {"name": "fc_plvr_value", "as": "cascade_plvtr"},
            {"name": "fc_psvr_value", "as": "cascade_psvtr"},
            {"name": "fc_pvtr_value", "as": "cascade_pwatch_time"},
            {"name": "fc_pwtd_value", "as": "cascade_pcptr"},
            {"name": "fc_pevr_value", "as": "cascade_fc_pevr"}, #evtr目标
            {'name':'fc_ltr_value', 'as': 'cascade_fc_pltr'},
            {'name':'fc_wtr_value', 'as': 'cascade_fc_pwtr'},
            {'name':'fc_ftr_value', 'as': 'cascade_fc_pftr'},
          ],
          use_packed_item_attr = True,
        ) \
      .end_()
