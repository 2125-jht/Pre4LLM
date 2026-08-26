from cascading_v2.module.channel.base_channel import BaseChannelPartitioner
from cascading_v2.module.channel.base_channel import BaseChannelScorer

class PrerankPhotoChannelParitioner(BaseChannelPartitioner):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _calculate_flag(self):
    pass

class PrerankPhotoChannelScorer(BaseChannelScorer):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _caculate_score(self, flag_attr, weight_attr, left_count_attr):
    self._calc_prerank_score(flag_attr, weight_attr)

  def _calc_prerank_score(self, flag_attr, weight_attr):
    self.flow \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "cascade_prerank_calc_type",
          "cascade_prerank_pctr_weight",
          "cascade_prerank_pltr_weight",
          "cascade_prerank_prstr_weight",
          "cascade_emp_watchtime_score_weight",
        ],
        import_item_attr = [
          "cascade_prerank_pctr",
          "cascade_prerank_pltr",
          "cascade_prerank_prstr",
          "cascade_emp_watchtime_score",
        ],
        export_item_attr = [
          {"name": "cascade_prerank_score", "as": self._score_attr}
        ],
        function_name = "CalPreRankScore",
        class_name = "ExploreLightFunctionSetV2",
        target_item = { flag_attr: 1 }
      ) \
      .if_("enable_explore_prerank_boost == 1") \
        .if_("enable_prerank_not_audit_discount == 1") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              "prerank_not_audit_discount_coef",
            ],
            import_item_attr = [
              {"name": self._score_attr, "as": "cascade_prerank_score"},
              "audit_b_second_tag",
            ],
            export_item_attr = [
              {"name": "cascade_prerank_score", "as": self._score_attr},
            ],
            function_name = "DiscountNotAuditPhotos",
            class_name = "ExploreLightFunctionSetV2",
            target_item = { flag_attr: 1 }
          ) \
        .end_() \
        .if_("explore_prerank_low_cost_photo_discount == 1") \
          .prerank_low_cost_photo_discount(self._score_attr, flag_attr) \
        .end_() \
        .if_("enable_explore_cs_photo_boost_prerank == 1") \
          .mc_cs_boost(self._score_attr, flag_attr, "prerank") \
        .end_() \
      .end_() \
      .sort(
        score_from_attr = self._score_attr,
        target_item = { flag_attr: 1 }
      ) \
      .if_("enable_prerank_select_photo_by_interest == 1") \
        .prerank_select_photo_by_interest(self._score_attr, flag_attr) \
      .end_() \
      .copy_item_meta_info(
        save_item_seq_to_attr = "prerank_final_index_photo",
        target_item = { flag_attr: 1 }
      )
