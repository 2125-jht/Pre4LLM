from cascading_v2 import CommonModule

class CascadingWaitedPhotoScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self._calc_pic_diversity_mgs_score()
    self._calc_interact_fusion_score()
    self._calc_hetu_one_debias_score()

  def _calc_pic_diversity_mgs_score(self) -> None:
    self.flow \
      .if_("enable_explore_mc_calc_pic_diversity == 1") \
        .explore_get_embedding_map_enricher(
          embedding_list_attr = "pic_embeddings",
          source_pids_list_attr = "embedding_source_pic_ids",
          dim_size = 64,
          export_common_attr = "pic_pid_embedding_map",
        ) \
        .explore_picture_diversity_enricher(
          export_item_attr = "pic_diversity_mgs_score",
          history_pic_ids_attr = "history_pic_ids",
          pid_embedding_common_attr = "pic_pid_embedding_map",
          dim_size = 64,
          dpp_diversity_mgs_topk = "{{explore_pic_mc_diversity_history_mgs_topk}}",
          calc_diversity_method = "{{explore_pic_mc_diversity_calc_diversity_method}}",
          target_item = {"is_picture": 1}
        ) \
      .end_()
    return self

  def _calc_interact_fusion_score(self) -> None:
    self.flow \
      .if_("enable_interact_fusion_score_cascade == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "mc_act_fusion_score_htr_weight", "as": "fr_act_fusion_score_htr_weight"},
            {"name": "mc_act_fusion_score_interact_ctr_weight", "as": "fr_act_fusion_score_interact_ctr_weight"},
            {"name": "mc_act_fusion_score_ftr_weight", "as": "fr_act_fusion_score_ftr_weight"},
            {"name": "mc_act_fusion_score_dtr_weight", "as": "fr_act_fusion_score_dtr_weight"},
            {"name": "mc_act_fusion_score_cmtr_weight", "as": "fr_act_fusion_score_cmtr_weight"},
            {"name": "mc_act_fusion_score_ltr_weight", "as": "fr_act_fusion_score_ltr_weight"},
            {"name": "mc_act_fusion_score_cltr_weight", "as": "fr_act_fusion_score_cltr_weight"},
            {"name": "mc_act_fusion_score_wtr_weight", "as": "fr_act_fusion_score_wtr_weight"},
            {"name": "mc_act_fusion_score_epstr_weight", "as": "fr_act_fusion_score_epstr_weight"},
            {"name": "mc_act_fusion_score_cmef_weight", "as": "fr_act_fusion_score_cmef_weight"},
            {"name": "mc_enable_pure_interact_fusion", "as": "enable_pure_interact_fusion"},
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
            {"name": "cascade_pepstr", "as": "pepstr"},
            {"name": "cascade_pcestr", "as": "pcmef"},
          ],
          export_item_attr = [
            {"name": "interact_fusion_score", "as": "mc_interact_fusion_score"},
          ],
          function_name = "CalInteractFusionScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()

  def _calc_hetu_one_debias_score(self) -> None:
    update_fix_xtrs = [
      "mc_ensemble_pctr",
      "mc_ensemble_pltr",
      "mc_ensemble_pwtr",
      "mc_ensemble_pcmtr",
      "mc_ensemble_pcltr",
      "mc_ensemble_pwtd_inverse",
      "mc_ensemble_pwatch_time",
      "mc_ensemble_peftr",
      "mc_ensemble_plvtr2",
      "cascade_pctr",
      "mc_ensemble_pftr"
    ]

    self.flow \
      .if_("explore_enable_hetu_one_xtr_debias_cal_mc_s2 == 1") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "extract_hetu_tag_list"},
          ],
          export_item_attr = [
            {"name": "first_hetu_tag", "as": "hetu_level_one_top1"},
          ],
          function_name = "ExtractFirstHetuTag",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .split_string(
          input_common_attr = "explore_hetu_one_debias_xtr_weight_mc_s2_str",
          output_common_attr = "explore_hetu_one_debias_xtr_weight_mc_s2_list",
          delimiters = ",",
          parse_to_double = True
        ) \
        .split_string(
          input_common_attr = "explore_hetu_one_debias_xtr_power_mc_s2_str",
          output_common_attr = "explore_hetu_one_debias_xtr_power_mc_s2_list",
          delimiters = ",",
          parse_to_double = True
        ) \
        .split_string(
          input_common_attr = "explore_hetu_one_debias_xtr_buttom_mc_s2_str",
          output_common_attr = "explore_hetu_one_debias_xtr_buttom_mc_s2_list",
          delimiters = ",",
          parse_to_double = True
        ) \
        .split_string(
          input_common_attr = "explore_hetu_one_debias_xtr_upper_mc_s2_str",
          output_common_attr = "explore_hetu_one_debias_xtr_upper_mc_s2_list",
          delimiters = ",",
          parse_to_double = True
        ) \
        .set_attr_value( 
          no_overwrite=True,
          common_attrs=[
            {
              "name": "explore_mc_s2_hetu_one_debias_xtr_name_list",
              "type": "string_list",
              "value": update_fix_xtrs,
            }
          ]
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_hetu_one_debias_xtr_weight_mc_s2_list", "as": "id_debias_xtr_weight_list"},
            {"name": "explore_hetu_one_debias_xtr_power_mc_s2_list", "as": "id_debias_xtr_power_list"},
            {"name": "explore_hetu_one_debias_xtr_buttom_mc_s2_list", "as": "id_debias_xtr_buttom_list"},
            {"name": "explore_hetu_one_debias_xtr_upper_mc_s2_list", "as": "id_debias_xtr_upper_list"},
            {"name": "explore_mc_s2_hetu_one_debias_xtr_name_list", "as": "fix_xtr_list"},
          ],
          import_item_attr = [
            {"name": "hetu_level_one_top1", "as": "debias_id_feature"},
          ] + update_fix_xtrs,
          export_item_attr = [
            {"name": "debias_score", "as": "cascade_hetu_one_xtr_debias_score"}
          ],
          function_name = "GenXtrScoreByIdFeature",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_()
