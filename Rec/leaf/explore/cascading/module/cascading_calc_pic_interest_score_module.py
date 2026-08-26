from cascading import CommonModule

class CascadingCalcPicInterestScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_calc_mc_pic_search_score == 1") \
        .pack_item_attr(
          item_source = {
            "reco_results": True,
          },
          mappings = [
            {
              "aggregator": "avg",
              "from_item_attr": "mc_ensemble_pctr",
              "to_common_attr": "mc_pic_pctr_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "mc_ensemble_pltr",
              "to_common_attr": "mc_pic_pltr_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "mc_ensemble_pwtr",
              "to_common_attr": "mc_pic_pwtr_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "mc_ensemble_pcmtr",
              "to_common_attr": "mc_pic_pcmtr_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "mc_ensemble_pcltr",
              "to_common_attr": "mc_pic_pcltr_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "mc_ensemble_pftr",
              "to_common_attr": "mc_pic_pftr_avg"
            },
          ],
          target_item = {
            "is_picture" : 1
          },
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_mc_pic_search_cluster_score_thresh", "as": "search_cluster_score_thresh"},
            {"name": "explore_mc_pic_search_cluster_score_max", "as": "search_cluster_score_max"},
            {"name": "explore_mc_pic_search_cluster_score_weight", "as": "search_cluster_score_weight"},
            {"name": "explore_mc_pic_search_score_pxtr_score_max", "as": "pxtr_score_max"},
            {"name": "explore_mc_pic_search_score_pctr_thresh", "as": "pctr_thresh"},
            {"name": "mc_pic_pctr_avg", "as": "pctr_avg"},
            {"name": "mc_pic_pltr_avg", "as": "pltr_avg"},
            {"name": "mc_pic_pwtr_avg", "as": "pwtr_avg"},
            {"name": "mc_pic_pcmtr_avg", "as": "pcmtr_avg"},
            {"name": "mc_pic_pcltr_avg", "as": "pcltr_avg"},
            {"name": "mc_pic_pftr_avg", "as": "pftr_avg"},
            {"name": "explore_mc_pic_search_score_pctr_alpha", "as": "pctr_alpha"},
            {"name": "explore_mc_pic_search_score_pltr_alpha", "as": "pltr_alpha"},
            {"name": "explore_mc_pic_search_score_pwtr_alpha", "as": "pwtr_alpha"},
            {"name": "explore_mc_pic_search_score_pcltr_alpha", "as": "pcltr_alpha"},
            {"name": "explore_mc_pic_search_score_pcmtr_alpha", "as": "pcmtr_alpha"},
            {"name": "explore_mc_pic_search_score_pftr_alpha", "as": "pftr_alpha"},
            {"name": "explore_mc_pic_search_score_pctr_beta", "as": "pctr_beta"},
            {"name": "explore_mc_pic_search_score_pltr_beta", "as": "pltr_beta"},
            {"name": "explore_mc_pic_search_score_pwtr_beta", "as": "pwtr_beta"},
            {"name": "explore_mc_pic_search_score_pcltr_beta", "as": "pcltr_beta"},
            {"name": "explore_mc_pic_search_score_pcmtr_beta", "as": "pcmtr_beta"},
            {"name": "explore_mc_pic_search_score_pftr_beta", "as": "pftr_beta"},
          ],
          import_item_attr = [
            {"name": "pic_search_interest_cluster_score", "as": "search_cluster_score"},
            {"name": "mc_ensemble_pctr", "as": "pctr"},
            {"name": "mc_ensemble_pltr", "as": "pltr"},
            {"name": "mc_ensemble_pwtr", "as": "pwtr"},
            {"name": "mc_ensemble_pcltr", "as": "pcltr"},
            {"name": "mc_ensemble_pcmtr", "as": "pcmtr"},
            {"name": "mc_ensemble_pftr", "as": "pftr"},
          ],
          export_item_attr = [
            {"name": "search_interest_score", "as": "mc_pic_search_score"}
          ],
          function_name = "CaclPicSearchInterestScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture" : 1
          },
        ) \
      .end_() \
      .if_("enable_explore_calc_pic_valid_interest_tag_score == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "user_valid_tag_id_list", "as": "tag_id_list"},
            {"name": "user_valid_tag_score_list", "as": "tag_score_list"},
            {"name": "explore_pic_valid_interest_tag_score_scale_coeff", "as": "scale_coeff"},
            {"name": "explore_pic_valid_interest_tag_score_bias_coeff", "as": "bias_coeff"},
          ],
          import_item_attr = [
            {"name": "hetu_tag_level_info__hetu_tag", "as": "item_tag_list"},
          ],
          export_item_attr = [
            {"name": "item_tag_score", "as": "pic_valid_interest_tag_score"}
          ],
          function_name = "GetItemTagScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture" : 1
          },
        ) \
      .end_()

    self._search_interest_tagnex_chase()

  def _search_interest_tagnex_chase(self):
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
