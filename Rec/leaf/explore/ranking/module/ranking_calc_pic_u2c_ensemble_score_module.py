from ranking import CommonModule

class RankingCalcPicU2CEnsembleScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_calc_fr_pic_u2c_ensemble_score == 1") \
        .if_("enable_explore_calc_mc_pic_u2c_ensemble_score == 0") \
          .set_attr_default_value(
            item_attrs=[{
              "name": "pic_u2c_score",
              "type": "double",
              "value": "{{explore_pic_u2c_score_default_value}}"
            }]
          ) \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "uPicU2CTopkCidList", "as": "cluster_id_list"},
              {"name": "uPicU2CTopkProbList", "as": "cluster_score_list"},
            ],
            import_item_attr = [
              {"name": "cluster_id_632", "as": "cluster_id"},
            ],
            export_item_attr = [
              {"name": "score", "as": "pic_u2c_score"}
            ],
            function_name = "CalcPicU2CScore",
            class_name = "ExploreLightFunctionSetV2",
            target_item = {
              "is_picture": 1
            },
          ) \
        .end_() \
        .enrich_attr_by_light_function( # 结合精排 pxtr 计算精排 fr_pic_u2c_ensemble_score 队列
          import_common_attr = [
            {"name": "explore_fr_pic_u2c_ensemble_score_u2c_alpha", "as": "u2c_alpha"},
            {"name": "explore_fr_pic_u2c_ensemble_score_pctr_alpha", "as": "pctr_alpha"},
            {"name": "explore_fr_pic_u2c_ensemble_score_pltr_alpha", "as": "pltr_alpha"},
            {"name": "explore_fr_pic_u2c_ensemble_score_pwtr_alpha", "as": "pwtr_alpha"},
            {"name": "explore_fr_pic_u2c_ensemble_score_pcltr_alpha", "as": "pcltr_alpha"},
            {"name": "explore_fr_pic_u2c_ensemble_score_pcmtr_alpha", "as": "pcmtr_alpha"},
            {"name": "explore_fr_pic_u2c_ensemble_score_pftr_alpha", "as": "pftr_alpha"},
            {"name": "explore_fr_pic_u2c_ensemble_score_u2c_beta", "as": "u2c_beta"},
            {"name": "explore_fr_pic_u2c_ensemble_score_pctr_beta", "as": "pctr_beta"},
            {"name": "explore_fr_pic_u2c_ensemble_score_pltr_beta", "as": "pltr_beta"},
            {"name": "explore_fr_pic_u2c_ensemble_score_pwtr_beta", "as": "pwtr_beta"},
            {"name": "explore_fr_pic_u2c_ensemble_score_pcltr_beta", "as": "pcltr_beta"},
            {"name": "explore_fr_pic_u2c_ensemble_score_pcmtr_beta", "as": "pcmtr_beta"},
            {"name": "explore_fr_pic_u2c_ensemble_score_pftr_beta", "as": "pftr_beta"},
          ],
          import_item_attr = [
            {"name": "pic_u2c_score", "as": "u2c_score"},
            "pctr",
            "pltr",
            "pwtr",
            "pcltr",
            "pcmtr",
            "pftr",
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "fr_pic_u2c_ensemble_score"}
          ],
          function_name = "CalcPicU2CEnsembleScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture" : 1
          },
        ) \
      .end_()

