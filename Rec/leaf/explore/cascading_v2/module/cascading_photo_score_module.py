from cascading_v2 import CommonModule

class CascadingPhotoScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self._calc_empirical_xtr()
    self._calc_avg_watch_time()
    self._calc_explore_u2c_score()
    self._calc_global_emphtr_score() # 应该挪到精排
    self._calc_pic_search_interest_tagnex_score()
    self._calc_cover_sense_view_score_trans()
    self._cal_user_career_interest_tagnex_tgi_score() # 应该挪到精排

  def _calc_empirical_xtr(self) -> None:
    self.flow \
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
      )

  def _calc_avg_watch_time(self) -> None:
    self.flow \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "mille_avg_watch_time_upper_bound"
        ],
        import_item_attr = [
          "duration_ms",
          "explore_stat__click_count",
          "explore_stat__view_length_sum",
          "is_picture",
        ],
        export_item_attr = [
          "avg_watch_time"
        ],
        function_name = "McCalAvgWatchTime",
        class_name = "ExploreLightFunctionSetV2",
      )

  def _calc_explore_u2c_score(self) -> None:
    self.flow \
      .if_("explore_mc_enable_mc_cluster_862_uninterest_cluster_by_u2c == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "uOldMmuClusterId300ListList", "as": "uOldMmuClusterId300ListList"},
            {"name": "interest_cid_collaborative_score_map", "as": "cid_score_map"}
          ],
          import_item_attr = [
            {"name": "hetu_sim_cluster_id", "as": "hetu_sim_cluster_id862"}
          ],
          export_item_attr = [
            {"name": "collaborative_score", "as": "cascade_explore_u2c_score"}
          ],
          function_name = "InterestCidCollaborativeFilter",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .sort(
          score_from_attr = "cascade_explore_u2c_score",
          update_score = False
        ) \
        .pack_item_attr(
          item_source = {
            "reco_results": True,
          },
          mappings = [
            {
              "from_item_attr": "hetu_sim_cluster_id",
              "to_common_attr": "user_cluster862_sorted_list",
              "aggregator": "concat",
              "dedup_to_common_attr": True,
            },
          ],
        ) \
      .end_()

  def _calc_global_emphtr_score(self) -> None:
    self.flow \
      .if_("enable_mc_cal_global_emphtr_score == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_hate_like_rate_report_weight", "as": "report_weight"},
            {"name": "explore_global_realshow_thres_for_emphtr_score", "as": "global_realshow_thres"},
          ],
          import_item_attr = [
            "explore_stat__real_show_count",
            "explore_stat__negative_count",
            "explore_stat__like_count",
            "explore_stat__report_count",
          ],
          export_item_attr = [
            "global_emphtr_score"
          ],
          function_name = "CalcGlobalEmphtrScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()

  def _calc_pic_search_interest_tagnex_score(self):
    self.flow \
      .if_("enable_explore_pic_search_interest_tagnex_chase == 1") \
        .set_attr_default_value(
          item_attrs = [{
            "name": "pic_search_interest_tagnex_score",
            "type": "double",
            "value": "{{explore_pic_search_interest_tagnex_chase_default_score}}"
          }]
        ) \
        .gen_common_attr_by_lua(
          attr_map = { # 搜索有效播放率越高 图文有效兴趣数越少 追打力度越大
            "pic_search_interest_user_weight": "user_search_valid_play_rate * explore_pic_search_interest_user_weight_boost_weight ^ math.min(math.max(explore_pic_search_interest_user_weight_boost_interest_thresh - (uDoubleOutsideValidPicClusterCnt7dKV or 0), 0), explore_pic_search_interest_user_weight_boost_power_weight_upbound)",
          },
        ) \
        .explore_interest_tagnex_chase_enricher( # 搜索兴趣 tagnex 追打分数
          user_positive_pid_list_attr = "user_search_valid_play_pid_list",
          user_weight_attr = "pic_search_interest_user_weight",
          tagnex_min_val = "{{explore_pic_search_interest_tagnex_chase_tagnex_min_val}}",
          tagnex_max_val = "{{explore_pic_search_interest_tagnex_chase_tagnex_max_val}}",
          adjust_coeff = "{{explore_pic_search_interest_tagnex_chase_adjust_coeff}}",
          scale_bound = "{{explore_pic_search_interest_tagnex_chase_scale_bound}}",
          base_weight = "{{explore_pic_search_interest_tagnex_chase_base_weight}}",
          tagnex_attr = "hetu_tag_level_info__hetu_tag",
          output_score_attr = "pic_search_interest_tagnex_score",
          target_item = {
            "is_picture": 1
          },
        ) \
      .end_()

  def _calc_cover_sense_view_score_trans(self):
    self.flow.if_("explore_enable_cover_sense_view_score_trans == 1")
    self.flow.set_attr_default_value(
      item_attrs=[{
        "name": "sense_view_predict_trans_score",
        "type": "double",
        "value": 0.0
      }, {
        "name": "cover_view_predict_trans_score",
        "type": "double",
        "value": 0.0       
      }],
    )
    self.flow.switch_("explore_cover_sense_view_score_version")
    self.flow.case_(2)
    self.__explore_cover_sense_view_score_unreview_trans(sense_view_predict_score = "sense_view_predict_score_v2",
                                                        cover_view_predict_score = "cover_view_predict_score_v2")
    self.flow.default_()
    self.__explore_cover_sense_view_score_unreview_trans(sense_view_predict_score = "sense_view_predict_score",
                                                        cover_view_predict_score = "cover_view_predict_score")
    self.flow.end_()
    self.flow.end_()

  def __explore_cover_sense_view_score_unreview_trans(self, sense_view_predict_score, cover_view_predict_score):
    self.flow \
      .copy_attr(
        attrs=[{
          "from_item": sense_view_predict_score,
          "to_item": "sense_view_predict_trans_score"
        }],
        select_item = {
          "attr_name": "audit_b_second_tag",
          "compare_to": 0,
          "select_if": "<=",
          "select_if_attr_missing": True,
        }
      ) \
      .copy_attr(
        attrs=[{
          "from_item": cover_view_predict_score,
          "to_item": "cover_view_predict_trans_score"
        }],
        select_item = {
          "attr_name": "audit_hot_cover_level",
          "compare_to": 0,
          "select_if": "<=",
          "select_if_attr_missing": True,
        }
      )

  def _cal_user_career_interest_tagnex_tgi_score(self):
    self.flow \
      .if_("explore_enable_cal_user_career_interest_tagnex_tgi == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_user_career_interest_tagnex_tgi_list", "as": "match_list"},
            {"name": "explore_user_career_interest_tagnex_tgi_coeff", "as": "coeff"},
            {"name": "explore_user_career_interest_tagnex_tgi_bias", "as": "bias"},
            {"name": "explore_user_career_interest_tagnex_circle_attr_min", "as": "attr_min"},
            {"name": "explore_user_career_interest_tagnex_circle_attr_max", "as": "attr_max"},
            {"name": "explore_user_career_interest_tagnex_circle_use_single_match_item", "as": "use_single_match_item"},
          ],
          import_item_attr = [
            {"name" : "hetu_tag_level_info__hetu_tag", "as" : "hetu_tag"},
            {"name" : "cluster_id_632", "as" : "cluster_id_632"}
          ],
          export_item_attr = [
            {"name": "match_score", "as": "user_career_interest_tagnex_tgi_score"}
          ],
          function_name = "CalMatchScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()
