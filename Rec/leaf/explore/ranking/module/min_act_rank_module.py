from ranking import CommonModule

class MinActRankModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
    .if_("enable_min_act_rank_score == 1") \
      .sort(
        score_from_attr = "fr_score2",
      ) \
      .copy_item_meta_info(
        save_item_seq_to_attr = "fr_score2_rank",
      ) \
      .sort(
        score_from_attr = "awesome_wtd",
      ) \
      .copy_item_meta_info(
        save_item_seq_to_attr = "fr_wtd_rank",
      ) \
      .sort(
        score_from_attr = "corr_pctr",
      ) \
      .copy_item_meta_info(
        save_item_seq_to_attr = "fr_ctr_rank",
      ) \
      .sort(
        score_from_attr = "pltr",
      ) \
      .copy_item_meta_info(
        save_item_seq_to_attr = "fr_ltr_rank",
      ) \
      .sort(
        score_from_attr = "pwtr",
      ) \
      .copy_item_meta_info(
        save_item_seq_to_attr = "fr_wtr_rank",
      ) \
      .sort(
        score_from_attr = "pftr",
      ) \
      .copy_item_meta_info(
        save_item_seq_to_attr = "fr_ftr_rank",
      ) \
      .sort(
        score_from_attr = "pcmtr",
      ) \
      .copy_item_meta_info(
        save_item_seq_to_attr = "fr_cmtr_rank",
      ) \
      .sort(
        score_from_attr = "pcltr",
      ) \
      .copy_item_meta_info(
        save_item_seq_to_attr = "fr_cltr_rank",
      ) \
      .if_("enable_watch_time_rank == 1") \
        .sort(
          score_from_attr = "pvtr",
        ) \
        .copy_item_meta_info(
          save_item_seq_to_attr = "fr_vtr_rank",
        ) \
        .sort(
          score_from_attr = "plvtr",
        ) \
        .copy_item_meta_info(
          save_item_seq_to_attr = "fr_lvtr_rank",
        ) \
        .sort(
          score_from_attr = "pfvtr",
        ) \
        .copy_item_meta_info(
          save_item_seq_to_attr = "fr_fvtr_rank",
        ) \
        .sort(
          score_from_attr = "pevtr",
        ) \
        .copy_item_meta_info(
          save_item_seq_to_attr = "fr_evtr_rank",
        ) \
      .end_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "explore_fr_min_act_rank_fscore2_weight",
          "explore_fr_min_act_rank_wtd_weight",
          "explore_fr_min_act_rank_fscore2_watch_time_weight",
          "explore_fr_min_act_rank_wtd_watch_time_weight",
          "explore_fr_min_act_rank_ctr_weight",
          "explore_fr_min_act_rank_ltr_weight",
          "explore_fr_min_act_rank_wtr_weight",
          "explore_fr_min_act_rank_ftr_weight",
          "explore_fr_min_act_rank_cmtr_weight",
          "explore_fr_min_act_rank_cltr_weight",
          "explore_fr_min_act_rank_pic_weight",
          "explore_fr_min_act_rank_vtr_weight",
          "explore_fr_min_act_rank_lvtr_weight",
          "explore_fr_min_act_rank_fvtr_weight",
          "explore_fr_min_act_rank_evtr_weight",
          "enable_avg_rank",
          "enable_watch_time_rank",
          "enable_second_act_rank",
          "enable_watch_time_second_act_rank"
        ],
        import_item_attr = [
          "fr_score2_rank",
          "fr_wtd_rank",
          "fr_ctr_rank",
          "fr_ltr_rank",
          "fr_wtr_rank",
          "fr_ftr_rank",
          "fr_cmtr_rank",
          "fr_cltr_rank",
          "fr_vtr_rank",
          "fr_lvtr_rank",
          "fr_fvtr_rank",
          "fr_evtr_rank",
          "is_picture"
        ],
        export_item_attr = [
          "min_act_rank_score",
          "min_watch_time_rank_score",
        ],
        function_name = "CalMinActRankScore",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .if_("enable_explore_topk_action_rank_score == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "explore_fr_topk_act_rank_fscore2_weight",
            "explore_fr_topk_act_rank_wtd_weight",
            "explore_fr_topk_act_rank_ctr_weight",
            "explore_fr_topk_act_rank_ltr_weight",
            "explore_fr_topk_act_rank_wtr_weight",
            "explore_fr_topk_act_rank_ftr_weight",
            "explore_fr_topk_act_rank_cmtr_weight",
            "explore_fr_topk_act_rank_cltr_weight",
            "explore_fr_topk_act_rank_vtr_weight",
            "explore_fr_topk_act_rank_lvtr_weight",
            "explore_fr_topk_act_rank_fvtr_weight",
            "explore_fr_topk_act_rank_evtr_weight",
            "explore_fr_topk_act_rank_pic_weight",
            "explore_fr_act_rank_topk"
          ],
          import_item_attr = [
            "fr_score2_rank",
            "fr_wtd_rank",
            "fr_ctr_rank",
            "fr_ltr_rank",
            "fr_wtr_rank",
            "fr_ftr_rank",
            "fr_cmtr_rank",
            "fr_cltr_rank",
            "fr_vtr_rank",
            "fr_lvtr_rank",
            "fr_fvtr_rank",
            "fr_evtr_rank",
            "is_picture"
          ],
          export_item_attr = [
            "topk_act_rank_score",
          ],
          function_name = "CalTopKActRankScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
    .end_()
