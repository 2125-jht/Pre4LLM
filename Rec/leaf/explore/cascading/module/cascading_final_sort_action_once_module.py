from cascading import CommonModule

class CascadingFinalSortActionOnceModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    # 此 module 主要用来生成在粗排第二阶段及之后使用的 diversity fr
    self.flow \
      .if_("enable_interact_fusion_score_cascade == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "mc_act_fusion_score_htr_weight", "as": "fr_act_fusion_score_htr_weight"},
            {"name": "mc_act_fusion_score_ctr_weight", "as": "fr_act_fusion_score_ctr_weight"},
            {"name": "mc_act_fusion_score_interact_ctr_weight", "as": "fr_act_fusion_score_interact_ctr_weight"},
            {"name": "mc_act_fusion_score_ftr_weight", "as": "fr_act_fusion_score_ftr_weight"},
            {"name": "mc_act_fusion_score_dtr_weight", "as": "fr_act_fusion_score_dtr_weight"},
            {"name": "mc_act_fusion_score_cmtr_weight", "as": "fr_act_fusion_score_cmtr_weight"},
            {"name": "mc_act_fusion_score_ltr_weight", "as": "fr_act_fusion_score_ltr_weight"},
            {"name": "mc_act_fusion_score_cltr_weight", "as": "fr_act_fusion_score_cltr_weight"},
            {"name": "mc_act_fusion_score_wtr_weight", "as": "fr_act_fusion_score_wtr_weight"},
            {"name": "mc_act_fusion_score_evtr_weight", "as": "fr_act_fusion_score_evtr_weight"},
            {"name": "mc_act_fusion_score_lvtr_ctr_weight", "as": "fr_act_fusion_score_lvtr_ctr_weight"},
            {"name": "mc_act_fusion_score_lvtr_weight", "as": "fr_act_fusion_score_lvtr_weight"},
            {"name": "mc_act_fusion_score_lvtr2_ctr_weight", "as": "fr_act_fusion_score_lvtr2_ctr_weight"},
            {"name": "mc_act_fusion_score_lvtr2_weight", "as": "fr_act_fusion_score_lvtr2_weight"},
            {"name": "mc_act_fusion_score_fvtr_weight", "as": "fr_act_fusion_score_fvtr_weight"},
            {"name": "mc_act_fusion_score_epstr_weight", "as": "fr_act_fusion_score_epstr_weight"},
            {"name": "mc_act_fusion_score_cmef_weight", "as": "fr_act_fusion_score_cmef_weight"},
            {"name": "mc_act_fusion_score_fetr_weight", "as": "fr_act_fusion_score_fetr_weight"},
            {"name": "mc_act_fusion_score_fr_score1_ctr_weight", "as": "fr_act_fusion_score_fr_score1_ctr_weight"},
            {"name": "mc_act_fusion_score_fr_score1_weight", "as": "fr_act_fusion_score_fr_score1_weight"},
            {"name": "mc_act_fusion_score_fr_score2_weight", "as": "fr_act_fusion_score_fr_score2_weight"},
            {"name": "mc_act_fusion_score_fr_score2_ctr_weight", "as": "fr_act_fusion_score_fr_score2_ctr_weight"},
            {"name": "mc_act_fusion_score_awesome_wtd_weight", "as": "fr_act_fusion_score_awesome_wtd_weight"},
            {"name": "mc_act_fusion_score_wtd_weight", "as": "fr_act_fusion_score_wtd_weight"},
            {"name": "mc_act_fusion_score_pvtr_weight", "as": "fr_act_fusion_score_pvtr_weight"},
            {"name": "mc_act_max_watchtime_threshold", "as": "fr_act_max_watchtime_threshold"},
            {"name": "mc_enable_pure_interact_fusion", "as": "enable_pure_interact_fusion"}
          ],
          import_item_attr = [
            {"name": "cascade_phtr", "as": "phtr"},
            {"name": "cascade_pctr", "as": "pctr"},
            {"name": "cascade_pftr", "as": "pftr"},
            {"name": "cascade_pdtr", "as": "pdtr"},
            {"name": "cascade_pcmtr", "as": "pcmtr"},
            {"name": "cascade_pltr", "as": "pltr"},
            {"name": "cascade_pcltr", "as": "pcltr"},
            {"name": "cascade_pwtr", "as": "pwtr"},
            {"name": "cascade_plvtr", "as": "plvtr"},
            {"name": "cascade_plvtr2", "as": "plvtr2"},
            {"name": "cascade_pefctr", "as": "pfvtr"},
            {"name": "cascade_pepstr", "as": "pepstr"},
            {"name": "cascade_pcestr", "as": "pcmef"},
            {"name": "cascade_peftr", "as": "fetr"},
            {"name": "cascade_pwatch_time", "as": "fr_score1"},
            {"name": "pptime", "as": "fr_score2"},
            {"name": "cascade_plivingtr", "as": "pvtr"},
            {"name": "mc_ensemble_pwtd", "as": "awesome_wtd"}
          ],
          export_item_attr = [
            {"name": "interact_fusion_score", "as": "mc_interact_fusion_score"},
            {"name": "watch_time_fusion_score", "as": "mc_watch_time_fusion_score"}
          ],
          function_name = "CalInteractFusionScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()