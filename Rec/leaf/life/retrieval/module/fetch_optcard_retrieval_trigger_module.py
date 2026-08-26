from retrieval import CommonModule

# TODO:liucong03(2月份删除，后面到统一trigger里调整)

class FetchOptcardRetrievalTriggerModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .if_("mc_enable_calculate_opt_card_trigger == 1") \
        .explore_intrest_adjust_enricher(
          gamora_hetu_adjust_history_list_attr = "gamora_hetu_adjust_history_list",
          opt_card_like_list_attr = "opt_card_like_list",
          opt_card_dis_like_list_attr = "opt_card_dis_like_list",
          adjust_mode = "{{adjust_mode}}",
          output_intrest_key_list_attr = "output_intrest_key_list",
          output_intrest_value_list_attr = "output_intrest_value_list",
          optcard_like_trigger_id_list_attr = "optcard_like_trigger_id_list",
          optcard_dislike_trigger_id_list_attr = "optcard_dislike_trigger_id_list"
        ) \
        .if_("optcard_like_trigger_id_list or optcard_dislike_trigger_id_list") \
          .get_abtest_params(
            biz_name = "MOBILE",
            ab_params = [{
              "param_name": "enable_opt_card_report",
              "param_type": "bool",
              "default_value": False,
              "report_ab_hit": True
            }],
          ) \
          .if_("enable_opt_card_report == 1") \
            .set_attr_value(
                common_attrs = [
                  {
                    "name": "optcard_like_trigger_id_list",
                    "type": "int_list",
                    "value": []
                  },
                  {
                    "name": "optcard_dislike_trigger_id_list",
                    "type": "int_list",
                    "value": []
                  }
                ]
              ) \
          .end_() \
        .end_() \
      .end_()
