#!/usr/bin/env python3
# coding=utf-8

from cascading import CommonModule

class CascadingEmpiricalXtrModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
    .if_("enable_prerank_ensemble_sort == 0") \
    .enrich_attr_by_light_function(
      import_item_attr = [
        "explore_stat__real_show_count",
        "explore_stat__click_count",
        "explore_stat__like_count",
        "explore_stat__follow_count",
        "explore_stat__forward_count",
        "explore_stat__long_play_count",
        "explore_stat__short_play_count",
        "explore_stat__profile_enter_count",
        "explore_stat__negative_count",
        "explore_stat__comment_count",
        "explore_stat__view_length_sum",
        "is_picture",
      ],
      export_item_attr = [
        "empirical_ctr",
        "empirical_ltr",
        "empirical_wtr",
        "empirical_ftr",
        "empirical_lvtr",
        "empirical_svtr",
        "empirical_ptr",
        "empirical_htr",
        "empirical_cmtr",
        "empirical_watch_time",
      ],
      function_name = "McCalEmpiricalXtr",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .end_()
