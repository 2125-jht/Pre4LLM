from cascading_v2.module.channel.base_channel import BaseChannelPartitioner
from cascading_v2.module.channel.base_channel import BaseChannelScorer

class PrerankFollowAuthorChannelParitioner(BaseChannelPartitioner):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _calculate_flag(self):
    self.flow \
      .copy_attr(
        attrs = [{
          "from_item": "is_follow_author",
          "to_item": self._flag_attr,
        }],
      )

class PrerankFollowAuthorChannelScorer(BaseChannelScorer):
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
          "cascade_emp_watchtime_score_weight",
        ],
        import_item_attr = [
          "cascade_prerank_pctr",
          "cascade_prerank_pltr",
          "cascade_emp_watchtime_score",
        ],
        export_item_attr = [
          {"name": "cascade_prerank_score", "as": self._score_attr}
        ],
        function_name = "CalPreRankScore",
        class_name = "ExploreLightFunctionSetV2",
        target_item = { flag_attr: 1 }
      )
