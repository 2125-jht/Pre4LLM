from cascading_v2 import CommonModule

class CascadingPrerankPhotoScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self._calc_hetu_level_one_ratio()
    self._calc_emp_watchtime_score()
    self._calc_upload_time_second()
    self._calc_photo_quality_score()
    self._calc_pic_recent_interest_cluster_score()

  def _calc_hetu_level_one_ratio(self) -> None:  # TODO: 应该挪到精排
    self.flow \
      .if_("enable_cal_prerank_adjust_diversity_distribution == 1") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            "hetu_level_one_top1",
          ],
          export_item_attr = [
            {"name": "hetu_level_one_ratio", "as": "explore_prerank_hetu_level_one_ratio"},
          ],
          function_name = "CalHetuOneRatio",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()

  def _calc_emp_watchtime_score(self) -> None:
    self.flow \
      .enrich_attr_by_light_function(
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
      )

  def _calc_upload_time_second(self):
    self.flow \
      .if_("enable_explore_gen_upload_time_second == 1") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            {"name": "upload_time", "as": "upload_time"}
          ],
          export_item_attr = [
            {"name": "upload_time_second", "as": "item_upload_second"},
          ],
          function_name = "GenUploadTimeSecond",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_()

  def _calc_photo_quality_score(self):
    self.flow \
      .if_("explore_enable_gen_photo_quality_score == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "cold_item_quality_score_map_ptr", "as": "map_ptr"}
          ],
          import_item_attr = [
            {"name": "photo_id", "as": "key_attr"}
          ],
          export_item_attr = [
            {"name": "target_item_attr", "as": "cold_item_quality_score"},
          ],
          function_name = "GetItemAttrByIntToDoubleMapPtr",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_()

  def _calc_pic_recent_interest_cluster_score(self):
    self.flow \
      .if_("enable_explore_calc_pic_recent_interest_cluster_score == 1") \
        .enrich_attr_by_light_function(
          item_list_from_attr = "explore_recent_interest_colossus_trigger_list",
          import_common_attr = [
            {"name": "enable_explore_calc_recent_interest_cluster_score_weight_decay", "as": "enable_weight_decay"},
            {"name": "explore_calc_recent_interest_cluster_score_weight_decay_power", "as": "weight_decay_power"},
            {"name": "enable_explore_calc_recent_interest_cluster_score_click_num_thres", "as": "enable_click_num_thres"},
            {"name": "explore_recent_interest_colossus_trigger_weight_list", "as": "weight_list"},
          ],
          import_item_attr = [
            "cluster_id_632",
          ],
          export_common_attr = [
            {"name": "interest_cluster_id_list", "as": "explore_pic_recent_interest_cluster_id_list"},
            {"name": "interest_cluster_score_list", "as": "explore_pic_recent_interest_cluster_score_list"},
          ],
          function_name = "GetPicRecentInterestScoreList",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()
