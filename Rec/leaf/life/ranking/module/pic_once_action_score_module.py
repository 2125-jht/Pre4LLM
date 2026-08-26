from ranking import CommonModule

class PicOnceActionScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("skip_pic_once_action_score == 0") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "pic_fr_act_fusion_score_htr_weight", "as": "fr_act_fusion_score_htr_weight"},
            {"name": "pic_fr_act_fusion_score_ctr_weight", "as": "fr_act_fusion_score_ctr_weight"},
            {"name": "pic_fr_act_fusion_score_ftr_weight", "as": "fr_act_fusion_score_ftr_weight"},
            {"name": "pic_fr_act_fusion_score_dtr_weight", "as": "fr_act_fusion_score_dtr_weight"},
            {"name": "pic_fr_act_fusion_score_cmtr_weight", "as": "fr_act_fusion_score_cmtr_weight"},
            {"name": "pic_fr_act_fusion_score_ltr_weight", "as": "fr_act_fusion_score_ltr_weight"},
            {"name": "pic_fr_act_fusion_score_cltr_weight", "as": "fr_act_fusion_score_cltr_weight"},
            {"name": "pic_fr_act_fusion_score_wtr_weight", "as": "fr_act_fusion_score_wtr_weight"},
            {"name": "pic_fr_act_fusion_score_evtr_weight", "as": "fr_act_fusion_score_evtr_weight"},
            {"name": "pic_fr_act_fusion_score_lvtr_weight", "as": "fr_act_fusion_score_lvtr_weight"},
            {"name": "pic_fr_act_fusion_score_fvtr_weight", "as": "fr_act_fusion_score_fvtr_weight"},
            {"name": "pic_fr_act_fusion_score_epstr_weight", "as": "fr_act_fusion_score_epstr_weight"},
            {"name": "pic_fr_act_fusion_score_cmef_weight", "as": "fr_act_fusion_score_cmef_weight"},
            {"name": "pic_fr_act_fusion_score_fetr_weight", "as": "fr_act_fusion_score_fetr_weight"},
            {"name": "pic_fr_act_fusion_score_fr_score1_weight", "as": "fr_act_fusion_score_fr_score1_weight"},
            {"name": "enable_pic_pure_interact_fusion", "as": "enable_pure_interact_fusion"},
          ],
          import_item_attr = [
            "phtr",
            "pctr",
            "pftr",
            "pdtr",
            "pcmtr",
            "pltr",
            "pcltr",
            "pwtr",
            "pevtr",
            "plvtr",
            {"name": "plvtr", "as": "pfvtr"},
            "pepstr",
            "pcmef",
            "fetr",
            "fr_score1",
          ],
          export_item_attr = [
            {"name": "interact_fusion_score", "as": "pic_interact_fusion_score"},
            {"name": "watch_time_fusion_score", "as": "pic_watch_time_fusion_score"},
          ],
          function_name = "CalInteractFusionScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture" : 1
          }
        ) \
      .end_()
  
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_attrs = [
          "pic_interact_fusion_score",
          "pic_watch_time_fusion_score"
        ],
        for_debug_request_only = True,
        target_item = {
          "is_picture" : 1
        }
      )
