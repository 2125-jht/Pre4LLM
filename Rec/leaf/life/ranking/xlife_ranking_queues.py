#!/usr/bin/env python3
# coding=utf-8

#精排 ensemble_sort 队列
# alpha -> raw_weight_attr
# beta -> raw_power_weight_attr
#图文队列

picture_queues = [
  {
    "name": "is_picture",
    "raw_weight_attr": "xlife_is_picture_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_is_picture_beta_raw_power_weight"
  },
]

#多样性队列
diversity_queues = [
  {
    "name": "diversity_fr",
    "raw_weight_attr": "xlife_diversity_fr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_diversity_fr_beta_raw_power_weight"
  },
  {
    "name": "diversity_fr_ranking",
    "raw_weight_attr": "xlife_diversity_fr_ranking_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_diversity_fr_ranking_beta_raw_power_weight"
  },
  {
    "name": "fr_mmu_embedding_score",
    "raw_weight_attr": "xlife_fr_mmu_embedding_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_fr_mmu_embedding_score_beta_raw_power_weight"
  },
  {
    "name": "fr_mc_embedding_score",
    "raw_weight_attr": "xlife_fr_mc_embedding_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_fr_mc_embedding_score_beta_raw_power_weight"
  },
  {
    "name": "hate_cover_similar_score",
    "raw_weight_attr": "xlife_hate_cover_similar_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_hate_cover_similar_score_beta_raw_power_weight"
  },
  {
    "name": "longterm_cluster_score",
    "raw_weight_attr": "xlife_longterm_cluster_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_longterm_cluster_score_beta_raw_power_weight"
  },
] 

#时长队列
time_queues = [
  {
    "name": "duration_ms",
    "value_type": "int",
    "raw_weight_attr": "xlife_duration_ms_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_duration_ms_beta_raw_power_weight"
  },
  {
    "name": "score_psvr",
    "raw_weight_attr": "xlife_score_psvr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_score_psvr_beta_raw_power_weight"
  },
  {
    "name": "ann_hetu_lvtr_score",
    "raw_weight_attr": "xlife_ann_hetu_lvtr_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_ann_hetu_lvtr_score_beta_raw_power_weight"
  },
  {
    "name": "corr_pctr_psvr",
    "raw_weight_attr": "xlife_corr_pctr_psvr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_corr_pctr_psvr_beta_raw_power_weight"
  },
  {
    "name": "pevtr",
    "raw_weight_attr": "xlife_pevtr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_pevtr_beta_raw_power_weight"
  },
  {
    "name": "dis_fr_score1",
    "raw_weight_attr": "xlife_dis_fr_score1_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_dis_fr_score1_beta_raw_power_weight"
  },
  {
    "name": "dis_fr_score2",
    "raw_weight_attr": "xlife_dis_fr_score2_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_dis_fr_score2_beta_raw_power_weight"
  },
  {
    "name": "fr_score1",
    "raw_weight_attr": "xlife_fr_score1_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_fr_score1_beta_raw_power_weight"
  },
  {
    "name": "fr_score2",
    "raw_weight_attr": "xlife_fr_score2_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_fr_score2_beta_raw_power_weight"
  },
  {
    "name": "score_pepstr",
    "raw_weight_attr": "xlife_score_pepstr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_score_pepstr_beta_raw_power_weight"
  },
  {
    "name": "corr_fetr",
    "raw_weight_attr": "xlife_corr_fetr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_corr_fetr_beta_raw_power_weight"
  },
  {
    "name": "corr_fountain_eff",
    "raw_weight_attr": "xlife_corr_fountain_eff_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_corr_fountain_eff_beta_raw_power_weight"
  },
  {
    "name": "awesome_wtd_score",
    "raw_weight_attr": "xlife_awesome_wtd_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_awesome_wtd_score_beta_raw_power_weight"
  },
  {
    "name": "score_consume_time_ltr",
    "raw_weight_attr": "xlife_score_consume_time_ltr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_score_consume_time_ltr_beta_raw_power_weight"
  },
  {
    "name": "consume_time_pf2r_score",
    "raw_weight_attr": "xlife_consume_time_pf2r_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_consume_time_pf2r_score_beta_raw_power_weight"
  },
  {
    "name": "adaptive_wtd_v2",
    "raw_weight_attr": "xlife_adaptive_wtd_v2_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_adaptive_wtd_v2_beta_raw_power_weight"
  },
  {
    "name": "watch_time_fusion_score",
    "raw_weight_attr": "xlife_watch_time_fusion_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_watch_time_fusion_score_beta_raw_power_weight"
  },
  {
    "name": "corr_cpr",
    "raw_weight_attr": "xlife_corr_cpr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_corr_cpr_beta_raw_power_weight"
  },
  {
    "name": "corr_wtd_evtr",
    "raw_weight_attr": "xlife_corr_wtd_evtr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_corr_wtd_evtr_beta_raw_power_weight"
  },
  {
    "name": "corr_wtd_lvtr",
    "raw_weight_attr": "xlife_corr_wtd_lvtr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_corr_wtd_lvtr_beta_raw_power_weight"
  },
  {
    "name": "awesome_wtd_debias_score",
    "raw_weight_attr": "xlife_awesome_wtd_debias_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_awesome_wtd_debias_score_beta_raw_power_weight"
  },
  {
    "name": "debias_mix_score",
    "raw_weight_attr": "xlife_debias_mix_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_debias_mix_score_beta_raw_power_weight"
  },
  {
    "name": "lte_ltr",
    "raw_weight_attr": "xlife_lte_ltr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_lte_ltr_beta_raw_power_weight"
  },
  {
    "name": "fetr",
    "raw_weight_attr": "xlife_fetr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_fetr_beta_raw_power_weight"
  },
  {
    "name": "fountain_eff",
    "raw_weight_attr": "xlife_fountain_eff_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_fountain_eff_beta_raw_power_weight"
  },
  {
    "name": "explore_fr_wtd_from_frac_score",
    "raw_weight_attr": "xlife_explore_fr_wtd_from_frac_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_explore_fr_wtd_from_frac_score_beta_raw_power_weight"
  },
  {
    "name": "explore_fr_wtd_frac_5_score",
    "raw_weight_attr": "xlife_explore_fr_wtd_frac_5_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_explore_fr_wtd_frac_5_score_beta_raw_power_weight"
  },
  {
    "name": "svtr_adapt_wtd_score",
    "raw_weight_attr": "xlife_svtr_adapt_wtd_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_svtr_adapt_wtd_score_beta_raw_power_weight"
  },
  {
    "name": "pairwise_rank_score",
    "raw_weight_attr": "xlife_pairwise_rank_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_pairwise_rank_score_beta_raw_power_weight"
  },
  {
    "name": "pairwise_rank_raw_score",
    "raw_weight_attr": "xlife_pairwise_rank_raw_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_pairwise_rank_raw_score_beta_raw_power_weight"
  },
  {
    "name": "ctr_multy_wtd_sharpe_ratio_score",
    "raw_weight_attr": "xlife_ctr_multy_wtd_sharpe_ratio_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_ctr_multy_wtd_sharpe_ratio_score_beta_raw_power_weight"
  },
  {
    "name": "fr_score2_debias_duration",
    "raw_weight_attr": "xlife_fr_score2_debias_duration_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_fr_score2_debias_duration_beta_raw_power_weight"
  },
  {
    "name": "hetu_gender_debias_avg_play_time_ms",
    "raw_weight_attr": "xlife_hetu_gender_debias_avg_play_time_ms_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_hetu_gender_debias_avg_play_time_ms_beta_raw_power_weight"
  },
  {
    "name": "mc_ensemble_pwatch_time",
    "raw_weight_attr": "xlife_mc_ensemble_pwatch_time_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_mc_ensemble_pwatch_time_beta_raw_power_weight"
  },
  {
    "name": "corr_pwtr",
    "raw_weight_attr": "xlife_corr_pwtr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_corr_pwtr_beta_raw_power_weight"
  },
  {
    "name": "cascade_prerank_pltr",
    "raw_weight_attr": "xlife_cascade_prerank_pltr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_cascade_prerank_pltr_beta_raw_power_weight"
  },
  {
    "name": "score_pctr_x_fr_score2",
    "raw_weight_attr": "xlife_score_pctr_x_fr_score2_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_score_pctr_x_fr_score2_beta_raw_power_weight"
  },
  {
    "name": "score_pctr_x_awesome_wtd",
    "raw_weight_attr": "xlife_score_pctr_x_awesome_wtd_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_score_pctr_x_awesome_wtd_beta_raw_power_weight"
  },
  {
    "name": "gen_l2r_score",
    "raw_weight_attr": "xlife_gen_l2r_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_gen_l2r_score_beta_raw_power_weight"
  },
  {
    "name": "ordinal_wtd_ltr_score",
    "raw_weight_attr": "xlife_ordinal_wtd_ltr_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_ordinal_wtd_ltr_score_beta_raw_power_weight"
  },
  {
    "name": "gen_l2r_fusion_score",
    "raw_weight_attr": "xlife_gen_l2r_fusion_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_gen_l2r_fusion_score_beta_raw_power_weight"
  },
  {
    "name": "consume_time_wtd",
    "raw_weight_attr": "xlife_consume_time_wtd_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_consume_time_wtd_beta_raw_power_weight"
  },
  {
    "name": "cpr",
    "raw_weight_attr": "xlife_cpr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_cpr_beta_raw_power_weight"
  },
]

#互动队列
interactive_queues = [
  {
    "name": "score_pctr",
    "raw_weight_attr": "xlife_score_pctr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_score_pctr_beta_raw_power_weight"
  },
  {
    "name": "score_pltr",
    "raw_weight_attr": "xlife_score_pltr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_score_pltr_beta_raw_power_weight"
  },
  {
    "name": "score_pwtr",
    "raw_weight_attr": "xlife_score_pwtr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_score_pwtr_beta_raw_power_weight"
  },
  {
    "name": "score_pftr",
    "raw_weight_attr": "xlife_score_pftr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_score_pftr_beta_raw_power_weight"
  },
  {
    "name": "score_pcmtr",
    "raw_weight_attr": "xlife_score_pcmtr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_score_pcmtr_beta_raw_power_weight"
  },
  {
    "name": "score_pptr",
    "raw_weight_attr": "xlife_score_pptr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_score_pptr_beta_raw_power_weight"
  },
  {
    "name": "score_pcmef",
    "raw_weight_attr": "xlife_score_pcmef_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_score_pcmef_beta_raw_power_weight"
  },
  {
    "name": "score_pdtr",
    "raw_weight_attr": "xlife_score_pdtr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_score_pdtr_beta_raw_power_weight"
  },
  {
    "name": "score_pcltr",
    "raw_weight_attr": "xlife_score_pcltr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_score_pcltr_beta_raw_power_weight"
  },
  {
    "name": "score_phtr",
    "reverse_order": True,
    "raw_weight_attr": "xlife_score_phtr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_score_phtr_beta_raw_power_weight"
  },
  {
    "name": "score_phtr",
    "raw_weight_attr": "xlife_score_phtr2_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_score_phtr2_beta_raw_power_weight"
  },
  {
    "name": "svr_act_score",
    "raw_weight_attr": "xlife_svr_act_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_svr_act_score_beta_raw_power_weight"
  },
  {
    "name": "act_combo_vtr_score",
    "raw_weight_attr": "xlife_act_combo_vtr_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_act_combo_vtr_score_beta_raw_power_weight"
  },
  {
    "name": "deep_ltr_score",
    "raw_weight_attr": "xlife_deep_ltr_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_deep_ltr_score_beta_raw_power_weight"
  },
  {
    "name": "watchtime_interact_score",
    "raw_weight_attr": "xlife_watchtime_interact_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_watchtime_interact_score_beta_raw_power_weight"
  },
  {
    "name": "ewatch_score",
    "raw_weight_attr": "xlife_ewatch_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_ewatch_score_beta_raw_power_weight"
  },
  {
    "name": "pcmef_debias_score",
    "raw_weight_attr": "xlife_pcmef_debias_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_pcmef_debias_score_beta_raw_power_weight"
  },
  {
    "name": "interact_fusion_score",
    "raw_weight_attr": "xlife_interact_fusion_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_interact_fusion_score_beta_raw_power_weight"
  },
  {
    "name": "corr_future_xtr",
    "raw_weight_attr": "xlife_corr_future_xtr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_corr_future_xtr_beta_raw_power_weight"
  },
  {
    "name": "ada_xtr_score",
    "raw_weight_attr": "xlife_ada_xtr_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_ada_xtr_score_beta_raw_power_weight"
  },
  {
    "name": "cascade_linear_score",
    "raw_weight_attr": "xlife_cascade_linear_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_cascade_linear_score_beta_raw_power_weight"
  },
  {
    "name": "click_cost_score",
    "raw_weight_attr": "xlife_click_cost_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_click_cost_score_beta_raw_power_weight"
  },
  {
    "name": "highorder_interact_score",
    "raw_weight_attr": "xlife_highorder_interact_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_highorder_interact_score_beta_raw_power_weight"
  },
  {
    "name": "absolute_xtr_score_que",
    "raw_weight_attr": "xlife_absolute_xtr_score_que_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_absolute_xtr_score_que_beta_raw_power_weight"
  },
  {
    "name": "xhs_meta_ltr",
    "raw_weight_attr": "xlife_xhs_meta_ltr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_xhs_meta_ltr_beta_raw_power_weight"
  },
  {
    "name": "pctr_pfr2r",
    "raw_weight_attr": "xlife_pctr_pfr2r_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_pctr_pfr2r_beta_raw_power_weight"
  },
  {
    "name": "pcltr_pfr2r",
    "raw_weight_attr": "xlife_pcltr_pfr2r_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_pcltr_pfr2r_beta_raw_power_weight"
  },
  {
    "name": "lte_ctr",
    "raw_weight_attr": "xlife_lte_ctr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_lte_ctr_beta_raw_power_weight"
  },
  {
    "name": "pctr_debias_hetu",
    "raw_weight_attr": "xlife_pctr_debias_hetu_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_pctr_debias_hetu_beta_raw_power_weight"
  },
  {
    "name": "pltr_debias_hetu",
    "raw_weight_attr": "xlife_pltr_debias_hetu_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_pltr_debias_hetu_beta_raw_power_weight"
  },
  {
    "name": "pwtr_debias_hetu",
    "raw_weight_attr": "xlife_pwtr_debias_hetu_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_pwtr_debias_hetu_beta_raw_power_weight"
  },
  {
    "name": "pftr_debias_hetu",
    "raw_weight_attr": "xlife_pftr_debias_hetu_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_pftr_debias_hetu_beta_raw_power_weight"
  },
  {
    "name": "pcmtr_debias_hetu",
    "raw_weight_attr": "xlife_pcmtr_debias_hetu_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_pcmtr_debias_hetu_beta_raw_power_weight"
  },
  {
    "name": "pptr_debias_hetu",
    "raw_weight_attr": "xlife_pptr_debias_hetu_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_pptr_debias_hetu_beta_raw_power_weight"
  },
  {
    "name": "awesome_wtd_debias_v2",
    "raw_weight_attr": "xlife_awesome_wtd_debias_v2_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_awesome_wtd_debias_v2_beta_raw_power_weight"
  },
  {
    "name": "corr_pltr_formula",
    "raw_weight_attr": "xlife_corr_pltr_formula_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_corr_pltr_formula_beta_raw_power_weight"
  },
  {
    "name": "corr_fr_score2_formula",
    "raw_weight_attr": "xlife_corr_fr_score2_formula_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_corr_fr_score2_formula_beta_raw_power_weight"
  },
  {
    "name": "corr_pfvtr_formula",
    "raw_weight_attr": "xlife_corr_pfvtr_formula_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_corr_pfvtr_formula_beta_raw_power_weight"
  },
  {
    "name": "plstr",
    "raw_weight_attr": "xlife_plstr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_plstr_beta_raw_power_weight"
  },
  {
    "name": "plsst",
    "raw_weight_attr": "xlife_plsst_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_plsst_beta_raw_power_weight"
  },
  {
    "name": "min_act_rank_score",
    "raw_weight_attr": "xlife_min_act_rank_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_min_act_rank_score_beta_raw_power_weight"
  },
  {
    "name": "fr_ctcvr_score",
    "raw_weight_attr": "xlife_fr_ctcvr_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_fr_ctcvr_score_beta_raw_power_weight"
  },
  {
    "name": "fr_ctcvr_gmv_score",
    "raw_weight_attr": "xlife_fr_ctcvr_gmv_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_fr_ctcvr_gmv_score_beta_raw_power_weight"
  },
  {
    "name": "fr_elive_ctcvr_score",
    "raw_weight_attr": "xlife_fr_elive_ctcvr_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_fr_elive_ctcvr_score_beta_raw_power_weight"
  },
  {
    "name": "fr_elive_ctcvr_gmv_score",
    "raw_weight_attr": "xlife_fr_elive_ctcvr_gmv_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_fr_elive_ctcvr_gmv_score_beta_raw_power_weight"
  },
  {
    "name": "produce_mtctr",
    "raw_weight_attr": "xlife_produce_mtctr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_produce_mtctr_beta_raw_power_weight"
  },
  {
    "name": "produce_twhtr",
    "raw_weight_attr": "xlife_produce_twhtr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_produce_twhtr_beta_raw_power_weight"
  },
  {
    "name": "produce_mfctr",
    "raw_weight_attr": "xlife_produce_mfctr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_produce_mfctr_beta_raw_power_weight"
  },
  {
    "name": "produce_mtcotr",
    "raw_weight_attr": "xlife_produce_mtcotr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_produce_mtcotr_beta_raw_power_weight"
  },
  {
    "name": "produce_mtjtr",
    "raw_weight_attr": "xlife_produce_mtjtr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_produce_mtjtr_beta_raw_power_weight"
  },
  {
    "name": "produce_mtm1",
    "raw_weight_attr": "xlife_produce_mtm1_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_produce_mtm1_beta_raw_power_weight"
  },
  {
    "name": "produce_upload_sum_score",
    "raw_weight_attr": "xlife_produce_upload_sum_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_produce_upload_sum_score_beta_raw_power_weight"
  },
  {
    "name": "produce_uploads",
    "raw_weight_attr": "xlife_produce_uploads_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_produce_uploads_beta_raw_power_weight"
  },
  {
    "name": "produce_consuv_sum_score",
    "raw_weight_attr": "xlife_produce_consuv_sum_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_produce_consuv_sum_score_beta_raw_power_weight"
  },
  {
    "name": "score_pctr_x_pcltr",
    "raw_weight_attr": "xlife_score_pctr_x_pcltr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_score_pctr_x_pcltr_beta_raw_power_weight"
  },
  {
    "name": "score_pctr_x_pepstr",
    "raw_weight_attr": "xlife_score_pctr_x_pepstr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_score_pctr_x_pepstr_beta_raw_power_weight"
  },
  {
    "name": "score_pctr_x_pptr",
    "raw_weight_attr": "xlife_score_pctr_x_pptr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_score_pctr_x_pptr_beta_raw_power_weight"
  },
  {
    "name": "score_pctr_x_pdtr",
    "raw_weight_attr": "xlife_score_pctr_x_pdtr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_score_pctr_x_pdtr_beta_raw_power_weight"
  },
  {
    "name": "score_pctr_x_pftr",
    "raw_weight_attr": "xlife_score_pctr_x_pftr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_score_pctr_x_pftr_beta_raw_power_weight"
  },
  {
    "name": "life_ltr_pctr",
    "raw_weight_attr": "xlife_life_ltr_pctr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_life_ltr_pctr_beta_raw_power_weight"
  },
  {
    "name": "life_pctr",
    "raw_weight_attr": "xlife_life_pctr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_life_pctr_beta_raw_power_weight"
  },
  {
    "name": "life_psvtr",
    "raw_weight_attr": "xlife_life_psvtr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_life_psvtr_beta_raw_power_weight"
  },
  {
    "name": "life_truth_pctr",
    "raw_weight_attr": "xlife_life_truth_pctr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_life_truth_pctr_beta_raw_power_weight"
  },
  {
    "name": "global_emphtr_score",
    "raw_weight_attr": "xlife_global_emphtr_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_global_emphtr_score_beta_raw_power_weight"
  },
  {
    "name": "global_empwtr_score",
    "raw_weight_attr": "xlife_global_empwtr_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_global_empwtr_score_beta_raw_power_weight"
  },
  {
    "name": "hetu_gender_debias_ctr",
    "raw_weight_attr": "xlife_hetu_gender_debias_ctr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_hetu_gender_debias_ctr_beta_raw_power_weight"
  },
  {
    "name": "hetu_gender_debias_ltr",
    "raw_weight_attr": "xlife_hetu_gender_debias_ltr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_hetu_gender_debias_ltr_beta_raw_power_weight"
  },
  {
    "name": "hetu_gender_debias_ftr",
    "raw_weight_attr": "xlife_hetu_gender_debias_ftr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_hetu_gender_debias_ftr_beta_raw_power_weight"
  },
  {
    "name": "hetu_gender_debias_cltr",
    "raw_weight_attr": "xlife_hetu_gender_debias_cltr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_hetu_gender_debias_cltr_beta_raw_power_weight"
  },
  {
    "name": "hetu_gender_debias_cmtr",
    "raw_weight_attr": "xlife_hetu_gender_debias_cmtr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_hetu_gender_debias_cmtr_beta_raw_power_weight"
  },
  {
    "name": "hetu_gender_debias_wtr",
    "raw_weight_attr": "xlife_hetu_gender_debias_wtr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_hetu_gender_debias_wtr_beta_raw_power_weight"
  },
  {
    "name": "action_diff_score",
    "raw_weight_attr": "xlife_action_diff_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_action_diff_score_beta_raw_power_weight"
  },
  {
    "name": "post_follow_score",
    "raw_weight_attr": "xlife_post_follow_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_post_follow_score_beta_raw_power_weight"
  },
  {
    "name": "rank_distill_ltr",
    "raw_weight_attr": "xlife_rank_distill_ltr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_rank_distill_ltr_beta_raw_power_weight"
  },
  {
    "name": "rank_distill_ctr",
    "raw_weight_attr": "xlife_rank_distill_ctr_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_rank_distill_ctr_beta_raw_power_weight"
  },
  {
    "name": "coordinated_watchtime_score",
    "raw_weight_attr": "xlife_coordinated_watchtime_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_coordinated_watchtime_score_beta_raw_power_weight"
  },
  {
    "name": "ctr_pairwise_rank_score",
    "raw_weight_attr": "xlife_ctr_pairwise_rank_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_ctr_pairwise_rank_score_beta_raw_power_weight"
  },
  {
    "name": "ctr_pairwise_rank_raw_score",
    "raw_weight_attr": "xlife_ctr_pairwise_rank_raw_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_ctr_pairwise_rank_raw_score_beta_raw_power_weight"
  },
  {
    "name": "intere_score",
    "raw_weight_attr": "xlife_cintere_score_alpha_raw_weight",
    "raw_power_weight_attr": "xlife_intere_score_beta_raw_power_weight"
  }
]

zero_play_queues = [
  "explore_ensemble_power_weight_fullrank_pltr_score",
  "explore_ensemble_power_weight_fullrank_pwtr_score",
  "explore_ensemble_power_weight_fullrank_pftr_score",
  "fr_pmctr_rank_weight",
  "explore_ensemble_power_weight_fullrank_pptr_score",
  "explore_ensemble_power_weight_fullrank_pcmef_score",
  "explore_ensemble_power_weight_fullrank_pdtr_score",
  "explore_ensemble_power_weight_fullrank_pcltr_score",
  "explore_ensemble_weight_duration_ms",
  "explore_ensemble_power_weight_fullrank_l2r_score"
]

zero_play_ctr_queues = [
  "explore_ensemble_power_weight_fullrank_pctr_score",
]

all_ensemble_queues_xlife = picture_queues + diversity_queues + time_queues + interactive_queues


