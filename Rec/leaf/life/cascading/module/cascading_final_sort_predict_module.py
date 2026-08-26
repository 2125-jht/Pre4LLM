from cascading import CommonModule
from cascading.cascade_util import hot_sim_fc_features

class CascadingFinalSortPredictModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_hot_mc_fc_s2_predict == 1") \
        .extract_with_ks_sign_feature(
          feature_list = hot_sim_fc_features, # base feature
          user_info_attr = "user_info_ptr",
          common_slots_output = "fc_sign_common_slots",
          common_parameters_output = "fc_sign_common_parameters",
        ) \
        .delegate_enrich(
          kess_service = "{{hot_mc_s2_fc_predict_service}}",
          request_type = "{{hot_mc_s2_fc_request_type}}",
          timeout_ms = 100,
          send_common_attrs = [
            { "name": "fc_sign_common_slots", "as": "common_slots" },
            { "name": "fc_sign_common_parameters", "as": "common_parameters" },
          ],
          recv_item_attrs = [
            {'name':'fc_pctr_value', 'as': 'cascade_fc_s2_pctr'},
            {'name':'fc_plvr_value', 'as': 'cascade_fc_s2_plvtr'},
            {'name':'fc_psvr_value', 'as': 'cascade_fc_s2_psvtr'},
            {'name':'fc_pvtr_value', 'as': 'cascade_fc_s2_pvtr'},
          ],
          use_packed_item_attr = True,
        ) \
      .end_()

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = ["enable_hot_mc_fc_s2_predict"],
        for_debug_request_only = True
      )