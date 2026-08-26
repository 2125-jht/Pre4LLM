from cascading.module.queue.cascading_channel_sort_queue import ChannelSortQueuePartitioner
from cascading.module.queue.cascading_channel_sort_queue import ChannelSortQueueScorer


class FollowAuthorQueueParitioner(ChannelSortQueuePartitioner):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _calculate_flag(self):
    self.flow \
      .if_("explore_prerank_enable_use_follow_author_bucket == 1") \
      .copy_attr(
        attrs=[{
          "from_item": "is_follow_author",
          "to_item": self._flag_attr,
        }]
      ) \
      .end_()

class FollowAuthorQueuePrerankScorer(ChannelSortQueueScorer):
  def __init__(self, name, flow, config):
    super().__init__(name, flow, config)

  def _caculate_score(self, flag_attr, weight_attr, left_count_attr):
    self._calc_prerank_score(flag_attr, weight_attr)

  def _calc_prerank_score(self, flag_attr, weight_attr):
    self.flow.enrich_attr_by_light_function(
      import_common_attr = [
        "prerank_duration_debias_bucket",
        "prerank_duration_debias_prefix",
        "prerank_short_duration_debias_interval",
        "prerank_mid_duration_debias_interval",
        "prerank_long_duration_debias_interval",
      ],
      import_item_attr = [
        "duration_ms",
        "explore_stat__view_length_sum",
        "explore_stat__click_count",
      ],
      export_item_attr = [
        "cascade_emp_watchtime_score",
      ],
      function_name = "CalEmpWatchTimeScore",
      class_name = "ExploreLightFunctionSetV2",
      target_item={ flag_attr: 1 }
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        "cascade_prerank_pctr_weight",
        "cascade_prerank_pltr_weight",
        "cascade_emp_watchtime_score_weight",
        "cascade_prerank_calc_type",
        "prerank_ltr_weight",
        "prerank_ctr_weight",
        "prerank_wtd_weight",
        "prerank_life_ctr_weight",
      ],
      import_item_attr = [
        "cascade_prerank_pctr",
        "cascade_prerank_pltr",
        "cascade_emp_watchtime_score",
        "prerank_ltr",
        "prerank_ctr",
        "prerank_wtd",
        "prerank_life_ctr",
      ],
      export_item_attr = [
        {"name": "cascade_prerank_score", "as": self._score_attr}
      ],
      function_name = "CalPreRankScore",
      class_name = "ExploreLightFunctionSetV2",
      target_item={ flag_attr: 1 }
    ) \
    .copy_attr(
      attrs=[{
        "from_item": self._score_attr,
        "to_item": "cascade_prerank_score"
      }]
    )  # copy_attr 放在 prerank 算分最后