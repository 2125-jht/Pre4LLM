from ranking import CommonModule

class PicRerankModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  # 精排模型产出的队列
  def fr_queues(self):
    queues = [
      {
        "name": "pctr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pctr_score",
        "enable_norm_attr": "explore_pic_fullrank_pctr_enable_norm",
        "func_mode_attr": "explore_pic_fullrank_pctr_func_mode",
        "use_exp_base_attr": "explore_pic_fullrank_pctr_use_exp_base",
        "raw_score_b_attr": "explore_pic_fullrank_pctr_raw_score_b",
        "raw_score_k_attr": "explore_pic_fullrank_pctr_raw_score_k",
        "raw_score_p_attr": "explore_pic_fullrank_pctr_raw_score_p",
        "raw_weight_attr": "explore_pic_fullrank_pctr_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pctr_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pctr_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_pctr_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_pctr_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_pctr_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_pctr_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_pctr_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_pctr_rank_weight",
      },
      {
        "name": "pltr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pltr_score",
        "enable_norm_attr": "explore_pic_fullrank_pltr_enable_norm",
        "func_mode_attr": "explore_pic_fullrank_pltr_func_mode",
        "use_exp_base_attr": "explore_pic_fullrank_pltr_use_exp_base",
        "raw_score_b_attr": "explore_pic_fullrank_pltr_raw_score_b",
        "raw_score_k_attr": "explore_pic_fullrank_pltr_raw_score_k",
        "raw_score_p_attr": "explore_pic_fullrank_pltr_raw_score_p",
        "pow_alpha_attr": "pltr_pow_alpha_attr",
        "pow_beta_attr": "pltr_pow_beta_attr",
        "pow_bias_attr": "pltr_pow_bias_attr",
        "raw_weight_attr": "explore_pic_fullrank_pltr_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pltr_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pltr_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_pltr_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_pltr_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_pltr_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_pltr_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_pltr_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_pltr_rank_weight",
      },
      {
        "name": "pwtr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pwtr_score",
        "enable_norm_attr": "explore_pic_fullrank_pwtr_enable_norm",
        "func_mode_attr": "explore_pic_fullrank_pwtr_func_mode",
        "use_exp_base_attr": "explore_pic_fullrank_pwtr_use_exp_base",
        "raw_score_b_attr": "explore_pic_fullrank_pwtr_raw_score_b",
        "raw_score_k_attr": "explore_pic_fullrank_pwtr_raw_score_k",
        "raw_score_p_attr": "explore_pic_fullrank_pwtr_raw_score_p",
        "pow_alpha_attr": "pwtr_pow_alpha_attr",
        "pow_beta_attr": "pwtr_pow_beta_attr",
        "pow_bias_attr": "pwtr_pow_bias_attr",
        "raw_weight_attr": "explore_pic_fullrank_pwtr_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pwtr_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pwtr_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_pwtr_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_pwtr_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_pwtr_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_pwtr_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_pwtr_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_pwtr_rank_weight",
      },
      {
        "name": "pftr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pftr_score",
        "enable_norm_attr": "explore_pic_fullrank_pftr_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_pftr_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pftr_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pftr_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_pftr_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_pftr_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_pftr_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_pftr_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_pftr_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_pftr_rank_weight",
      },
      {
        "name": "pcmtr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pcmtr_score",
        "enable_norm_attr": "explore_pic_fullrank_pcmtr_enable_norm",
        "func_mode_attr": "explore_pic_fullrank_pcmtr_func_mode",
        "use_exp_base_attr": "explore_pic_fullrank_pcmtr_use_exp_base",
        "raw_score_b_attr": "explore_pic_fullrank_pcmtr_raw_score_b",
        "raw_score_k_attr": "explore_pic_fullrank_pcmtr_raw_score_k",
        "raw_score_p_attr": "explore_pic_fullrank_pcmtr_raw_score_p",
        "pow_alpha_attr": "pcmtr_pow_alpha_attr",
        "pow_beta_attr": "pcmtr_pow_beta_attr",
        "pow_bias_attr": "pcmtr_pow_bias_attr",
        "raw_weight_attr": "explore_pic_fullrank_pcmtr_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pcmtr_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pcmtr_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_pcmtr_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_pcmtr_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_pcmtr_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_pcmtr_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_pcmtr_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_pcmtr_rank_weight",
      },
      {
        "name": "pptr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pptr_score",
        "enable_norm_attr": "explore_pic_fullrank_pptr_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_pptr_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pptr_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pptr_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_pptr_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_pptr_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_pptr_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_pptr_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_pptr_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_pptr_rank_weight",
      },
      {
        "name": "pcmef",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pcmef_score",
        "enable_norm_attr": "explore_pic_fullrank_pcmef_enable_norm",
        "pow_alpha_attr": "pcmef_pow_alpha_attr",
        "pow_beta_attr": "pcmef_pow_beta_attr",
        "pow_bias_attr": "pcmef_pow_bias_attr",
        "raw_weight_attr": "explore_pic_fullrank_pcmef_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pcmef_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pcmef_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_pcmef_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_pcmef_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_pcmef_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_pcmef_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_pcmef_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_pcmef_rank_weight",
      },
      {
        "name": "pevtr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pevtr_score",
        "enable_norm_attr": "explore_pic_fullrank_pevtr_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_pevtr_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pevtr_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pevtr_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_pevtr_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_pevtr_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_pevtr_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_pevtr_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_pevtr_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_pevtr_rank_weight",
      },
      {
        "name": "psvr",
        "reverse_value": "{{explore_pic_fullrank_psvr_reverse_value}}",
        "power_weight_attr": "explore_pic_power_weight_fullrank_psvr_score",
        "enable_norm_attr": "explore_pic_fullrank_psvr_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_psvr_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_psvr_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_psvr_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_psvr_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_psvr_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_psvr_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_psvr_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_psvr_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_psvr_rank_weight",
      },
      {
        "name": "fr_score1",
        "power_weight_attr": "explore_pic_power_weight_fullrank_fr_score1_score",
        "enable_norm_attr": "explore_pic_fullrank_fr_score1_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_fr_score1_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_fr_score1_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_fr_score1_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_fr_score1_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_fr_score1_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_fr_score1_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_fr_score1_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_fr_score1_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_fr_score1_rank_weight",
      },
      {
        "name": "fr_score2",
        "power_weight_attr": "explore_pic_power_weight_fullrank_fr_score2_score",
        "enable_norm_attr": "explore_pic_fullrank_fr_score2_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_fr_score2_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_fr_score2_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_fr_score2_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_fr_score2_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_fr_score2_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_fr_score2_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_fr_score2_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_fr_score2_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_fr_score2_rank_weight",
      },
      {
        "name": "pepstr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pepstr_score",
        "enable_norm_attr": "explore_pic_fullrank_pepstr_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_pepstr_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pepstr_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pepstr_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_pepstr_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_pepstr_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_pepstr_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_pepstr_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_pepstr_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_pepstr_rank_weight",
      },
      {
        "name": "pdtr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pdtr_score",
        "enable_norm_attr": "explore_pic_fullrank_pdtr_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_pdtr_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pdtr_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pdtr_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_pdtr_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_pdtr_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_pdtr_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_pdtr_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_pdtr_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_pdtr_rank_weight",
      },
      {
        "name": "pcltr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pcltr_score",
        "enable_norm_attr": "explore_pic_fullrank_pcltr_enable_norm",
        "func_mode_attr": "explore_pic_fullrank_pcltr_func_mode",
        "use_exp_base_attr": "explore_pic_fullrank_pcltr_use_exp_base",
        "raw_score_b_attr": "explore_pic_fullrank_pcltr_raw_score_b",
        "raw_score_k_attr": "explore_pic_fullrank_pcltr_raw_score_k",
        "raw_score_p_attr": "explore_pic_fullrank_pcltr_raw_score_p",
        "raw_weight_attr": "explore_pic_fullrank_pcltr_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pcltr_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pcltr_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_pcltr_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_pcltr_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_pcltr_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_pcltr_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_pcltr_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_pcltr_rank_weight",
      },
      {
        "name": "phtr",
        "reverse_order": "{{explore_pic_fullrank_phtr_enable_reverse_order}}",
        "power_weight_attr": "explore_pic_power_weight_htr_score",
        "enable_norm_attr": "explore_pic_fullrank_phtr_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_phtr_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_phtr_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_phtr_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_phtr_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_phtr_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_phtr_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_phtr_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_phtr_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_phtr_rank_weight",
      },
      {
        "name": "awesome_wtd",
        "power_weight_attr": "explore_pic_power_weight_fullrank_awesome_wtd",
        "enable_norm_attr": "explore_pic_fullrank_pwtd_enable_norm",
        "func_mode_attr": "explore_pic_fullrank_pwtd_func_mode",
        "use_exp_base_attr": "explore_pic_fullrank_pwtd_use_exp_base",
        "raw_score_b_attr": "explore_pic_fullrank_pwtd_raw_score_b",
        "raw_score_k_attr": "explore_pic_fullrank_pwtd_raw_score_k",
        "raw_score_p_attr": "explore_pic_fullrank_pwtd_raw_score_p",
      },
      {
        "name": "fetr",
        "power_weight_attr": "explore_pic_power_weight_fetr",
      },
      {
        "name": "fountain_eff",
        "power_weight_attr": "explore_pic_power_weight_fountain_eff",
      },
      {
        "name": "pic_corr_pwtr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_corr_pwtr_score",
      },
      {
        "name": "pic_corr_pcmtr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_corr_pcmtr_score",
      },
      {
        "name": "pic_corr_pctr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_corr_pctr_score",
      },
      {
        "name": "pic_corr_pltr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_corr_pltr_score",
      },
      {
        "name": "pic_corr_pftr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_corr_pftr_score",
      },
      {
        "name": "pic_corr_pptr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_corr_pptr_score",
      },
      {
        "name": "pic_corr_pcltr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_corr_pcltr_score",
      },
      {
        "name": "pic_emp_corr_pwtr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_emp_corr_pwtr_score",
      },
      {
        "name": "pic_emp_corr_pcmtr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_emp_corr_pcmtr_score",
      },
      {
        "name": "pic_emp_corr_pctr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_emp_corr_pctr_score",
        "enable_norm_attr": "explore_pic_fullrank_pic_emp_corr_pctr_enable_norm",
        "raw_weight_attr": "explore_pic_emp_corr_pctr_raw_weight",
        "raw_bias_attr": "explore_pic_emp_corr_pctr_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_emp_corr_pctr_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_emp_corr_pctr_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_emp_corr_pctr_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_emp_corr_pctr_rank_temperature",
        "rank_smooth_attr": "explore_pic_emp_corr_pctr_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_emp_corr_pctr_rank_base_number",
        "rank_weight_attr": "explore_pic_emp_corr_pctr_rank_weight",
      },
      {
        "name": "pic_emp_corr_pltr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_emp_corr_pltr_score",
      },
      {
        "name": "pic_emp_corr_pftr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_emp_corr_pftr_score",
      },
      {
        "name": "pic_emp_corr_pptr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_emp_corr_pptr_score",
      },
      {
        "name": "pic_variety_score",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pic_variety_score",
        "enable_norm_attr": "explore_pic_fullrank_pic_variety_score_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_pic_variety_score_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pic_variety_score_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pic_variety_score_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_pic_variety_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_pic_variety_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_pic_variety_score_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_pic_variety_score_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_variety_score_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_pic_variety_score_rank_weight",
      },
      {
        "name": "pic_behaviour_diversity_score",
        "power_weight_attr": "explore_pic_power_weight_behaviour_diversity_score",
        "enable_norm_attr": "explore_pic_fullrank_pic_behaviour_diversity_score_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_pic_behaviour_diversity_score_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pic_behaviour_diversity_score_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pic_behaviour_diversity_score_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_pic_behaviour_diversity_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_pic_behaviour_diversity_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_pic_behaviour_diversity_score_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_pic_behaviour_diversity_score_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_pic_behaviour_diversity_score_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_pic_behaviour_diversity_score_rank_weight",
      }
    ]
    return queues

  # LTR队列和提权队列等非精排队列
  def ltr_and_manual_queues(self):
    queues = [
      {
        "name": "consume_time_ltr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_l2r_score",
        "raw_weight_attr": "explore_pic_fullrank_consume_time_ltr_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_consume_time_ltr_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_consume_time_ltr_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_consume_time_ltr_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_consume_time_ltr_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_consume_time_ltr_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_consume_time_ltr_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_consume_time_ltr_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_consume_time_ltr_rank_weight",
      },
      {
        "name": "deep_ltr_score",
        "power_weight_attr": "explore_pic_power_weight_deep_ltr_score"
      },
      {
        "name": "explore_fr_ensemble_score",
        "power_weight_attr": "explore_pic_power_weight_explore_fr_ensemble_score"
      },
      {
        "name": "corr_pic_wtd",
        "power_weight_attr": "explore_pic_power_weight_corr_pic_wtd",
      },
      {
        "name": "corr_pic_lvtr",
        "power_weight_attr": "explore_pic_power_weight_corr_pic_lvtr",
      },
      {
        "name": "corr_pic_cpr",
        "power_weight_attr": "explore_pic_power_weight_corr_pic_cpr",
      },
      {
        "name": "pic_ltr_weighted_ctr",
        "power_weight_attr": "explore_pic_power_weight_pic_weighted_ctr",
        "enable_norm_attr": "explore_pic_fullrank_pic_ltr_weighted_ctr_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_pic_ltr_weighted_ctr_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pic_ltr_weighted_ctr_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pic_ltr_weighted_ctr_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_pic_ltr_weighted_ctr_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_pic_ltr_weighted_ctr_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_pic_ltr_weighted_ctr_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_pic_ltr_weighted_ctr_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_pic_ltr_weighted_ctr_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_pic_ltr_weighted_ctr_rank_weight",
      },
      {
        "name": "pic_ltr_lvtr",
        "power_weight_attr": "explore_pic_power_weight_pic_lvtr",
      },
      {
        "name": "pic_ltr_fvtr",
        "power_weight_attr": "explore_pic_power_weight_pic_fvtr",
        "enable_norm_attr": "explore_pic_fullrank_pic_ltr_fvtr_enable_norm",
        "personalized_power_weight_attr": "pic_ltr_fvtr_personalized_weight",
        "raw_weight_attr": "explore_pic_fullrank_pic_ltr_fvtr_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pic_ltr_fvtr_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pic_ltr_fvtr_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_pic_ltr_fvtr_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_pic_ltr_fvtr_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_pic_ltr_fvtr_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_pic_ltr_fvtr_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_pic_ltr_fvtr_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_pic_ltr_fvtr_rank_weight",
      },
      {
        "name": "pic_ltr_wtd",
        "power_weight_attr": "explore_pic_power_weight_pic_wtd",
      },
      {
        "name": "pic_ltr_acttr",
        "power_weight_attr": "explore_pic_power_weight_pic_acttr",
      },
      {
        "name": "fr_ensemble_pic_oppo_cost_score",
        "power_weight_attr": "explore_pic_power_weight_oppo_cost_score",
      },
      {
        "name": "fr_single_pic_demote_score",
        "power_weight_attr": "explore_pic_power_weight_single_pic_demote_score",
        "enable_norm_attr": "explore_pic_fullrank_fr_single_pic_demote_score_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_fr_single_pic_demote_score_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_fr_single_pic_demote_score_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_fr_single_pic_demote_score_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_fr_single_pic_demote_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_fr_single_pic_demote_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_fr_single_pic_demote_score_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_fr_single_pic_demote_score_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_fr_single_pic_demote_score_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_fr_single_pic_demote_score_rank_weight",
        "default": 1.0,
      },
      {
        "name": "fr_follow_author_pic_boost_score",
        "power_weight_attr": "explore_pic_power_weight_follow_author_pic_boost_score",
        "enable_norm_attr": "explore_pic_fullrank_fr_follow_author_pic_boost_score_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_fr_follow_author_pic_boost_score_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_fr_follow_author_pic_boost_score_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_fr_follow_author_pic_boost_score_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_fr_follow_author_pic_boost_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_fr_follow_author_pic_boost_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_fr_follow_author_pic_boost_score_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_fr_follow_author_pic_boost_score_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_fr_follow_author_pic_boost_score_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_fr_follow_author_pic_boost_score_rank_weight",
        "default": 1.0
      },
      {
        "name": "fr_target_hetu_pic_boost_score",
        "power_weight_attr": "explore_pic_power_weight_target_hetu_pic_boost_score",
        "enable_norm_attr": "explore_pic_fullrank_fr_target_hetu_pic_boost_score_enable_norm",
        "raw_weight_attr": "explore_fr_target_hetu_pic_boost_score_raw_weight",
        "raw_bias_attr": "explore_fr_target_hetu_pic_boost_score_raw_bias",
        "pow_raw_score_attr": "explore_pic_fr_target_hetu_pic_boost_score_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_fr_target_hetu_pic_boost_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_fr_target_hetu_pic_boost_score_max_val",
        "rank_temperature_attr": "explore_fr_target_hetu_pic_boost_score_rank_temperature",
        "rank_smooth_attr": "explore_fr_target_hetu_pic_boost_score_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_target_hetu_pic_boost_score_rank_base_number",
        "rank_weight_attr": "explore_fr_target_hetu_pic_boost_score_rank_weight",
        "default": 1.0
      },
      {
        "name": "pic_ltr_young_photo_score",
        "power_weight_attr": "young_photo_pic_score_weight",
        "default": 0.0
      },
      {
        "name": "pic_ltr_hv_picture_score",
        "power_weight_attr": "pic_ltr_hv_pic_score_weight",
        "default": 0.1
      },
      {
        "name": "fr_pic_ensemble_long_caption_score",
        "power_weight_attr": "explore_pic_power_weight_long_caption_score",
        "enable_norm_attr": "explore_pic_fullrank_fr_pic_ensemble_long_caption_score_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_long_caption_score_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_long_caption_score_raw_bias",
        "pow_raw_score_attr": "explore_pic_fr_pic_ensemble_long_caption_score_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_long_caption_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_long_caption_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_long_caption_score_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_long_caption_score_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_pic_ensemble_long_caption_score_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_long_caption_score_rank_weight",
        "default": 1.0
      },
      {
        "name": "pic_interact_fusion_score",
        "power_weight_attr": "explore_pic_power_weight_pic_interact_fusion_score",
      },
      {
        "name": "pic_watch_time_fusion_score",
        "power_weight_attr": "explore_pic_power_weight_pic_watch_time_fusion_score",
      },
      {
        "name": "pic_ltr_bpr_ctr",
        "power_weight_attr": "explore_pic_power_weight_pic_ltr_bpr_ctr",
      },
      {
        "name": "pic_ltr_bpr_cltr",
        "power_weight_attr": "explore_pic_power_weight_pic_ltr_bpr_cltr",
      },
      {
        "name": "pic_ltr_bpr_revisittr",
        "power_weight_attr": "explore_pic_power_weight_pic_ltr_bpr_revisittr",
      },
      {
        "name": "pic_diversity_score",
        "power_weight_attr": "explore_pic_power_weight_pic_diversity_score",
        "enable_norm_attr": "explore_pic_fullrank_pic_diversity_score_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_pic_diversity_score_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pic_diversity_score_raw_bias",
        "pow_raw_score_attr": "explore_pic_fr_pic_diversity_score_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_pic_diversity_score_min_val",
        "raw_score_max_val_attr": "explore_pic_pic_diversity_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_pic_diversity_score_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_pic_diversity_score_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_pic_diversity_score_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_pic_diversity_score_rank_weight",
      },
      {
        "name": "pic_ranking_heat_boost_output_heat",
        "power_weight_attr": "explore_pic_power_weight_pic_ranking_heat_boost_output_heat",
      },
      {
        "name": "pic_cascade_fc_interact_score",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pic_fc_interact_score",
      },
      {
        "name": "pic_cascade_fc_d2q",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pic_fc_d2q",
      },
      {
        "name": "pic_diversity_mgs_score",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pic_diversity_mgs_score",
      }
    ]
    return queues

  def ensemble_queues(self):
    queues = self.fr_queues()
    queues.extend(self.ltr_and_manual_queues())
    return queues

  def quantile_trans_queues(self):
    queues = self.fr_queues()
    trans_queues = []
    for queue in queues:
        new_trans = {}
        new_trans["name"] = queue["name"]
        new_trans["quantile_name"] = queue["name"] + "_quantile"
        new_trans["enable_quantile"] = "explore_pic_enable_quantile_" + queue["name"]
        trans_queues.append(new_trans)
    return trans_queues

  def fine_quantile_queues(self):
    queues = {
      "pic_pwtr" : "pwtr",
      "pic_pcmtr": "pcmtr",
      "pic_corr_pctr": "pic_corr_pctr",
      "pic_pftr": "pftr",
      "pic_pptr": "pptr",
      "pic_pcltr": "pcltr",
      "pic_pltr": "pltr",
    }
    trans_queues = []
    for key in queues:
      new_trans = {}
      new_trans["name"] = "fr_" + key + "_pct"
      new_trans["xtr_attr"] = queues[key]
      new_trans["fractile_value_attr"] = new_trans["name"]
      trans_queues.append(new_trans)
    return trans_queues

  def quantile_ensemble_queues(self):
    queues = self.fr_queues()
    for queue in queues:
      queue["name"] += "_quantile"
      queue["power_weight_attr"] = "explore_pic_power_weight_" + queue["name"]
    queues.extend(self.ltr_and_manual_queues())
    return queues

  def process(self) -> None:
    self.flow.if_("skip_explore_pic_rerank == 0")

    self._score_enricher()

    self.flow \
      .explore_pic_rerank(
        save_score_to_attr = "fr_pic_ensemble_score",
        limit = "{{explore_pic_rerank_limit}}",
        picture_attr = "is_picture",
        queues = self.ensemble_queues(),
        unify_rank_mode = "{{explore_pic_rerank_unify_rank_mode}}",
        unify_queue_calc_mode = "{{explore_pic_rerank_unify_queue_calc_mode}}",
        queue_max_raw_score = "{{explore_pic_rerank_queue_max_raw_score}}",
        queue_min_raw_score = "{{explore_pic_rerank_queue_min_raw_score}}",
        queue_max_rank_score = "{{explore_pic_rerank_queue_max_rank_score}}",
        queue_min_rank_score = "{{explore_pic_rerank_queue_min_rank_score}}",
        enable_normalization_item_score = "{{explore_pic_rerank_enable_normalization_item_score}}",
        enable_2sigma_overall_ori_pxtr = "{{explore_pic_rerank_enable_2sigma_overall_ori_pxtr}}",
      )

    self.flow.if_("skip_explore_pic_rerank_variant == 0")
    self._variant()
    self.flow.end_()

    self.flow.end_()
    self.flow.log_debug_info(
      common_attrs = [
        'explore_pic_enable_quantile_fetr', 'explore_pic_enable_quantile_fountain_eff', 'explore_pic_enable_quantile_fr_score1', 'explore_pic_enable_quantile_fr_score2', 'explore_pic_enable_quantile_pcltr',
        'explore_pic_enable_quantile_pcmef', 'explore_pic_enable_quantile_pcmtr', 'explore_pic_enable_quantile_pctr', 'explore_pic_enable_quantile_pdtr', 'explore_pic_enable_quantile_pepstr',
        'explore_pic_enable_quantile_pevtr', 'explore_pic_enable_quantile_pftr', 'explore_pic_enable_quantile_phtr', 'explore_pic_enable_quantile_pltr', 'explore_pic_enable_quantile_pptr', 'explore_pic_enable_quantile_pwtr',
        'explore_pic_fr_adjust_queue_weight_k_cltr', 'explore_pic_fr_adjust_queue_weight_k_cmtr', 'explore_pic_fr_adjust_queue_weight_k_ftr', 'explore_pic_fr_adjust_queue_weight_k_ltr', 'explore_pic_fr_adjust_queue_weight_k_wtr',
        'explore_pic_fr_adjust_queue_weight_max', 'explore_pic_fr_adjust_queue_weight_mode', 'explore_pic_fr_adjust_queue_weight_p', 'explore_pic_power_weight_fetr_quantile', 'explore_pic_power_weight_fountain_eff_quantile',
        'explore_pic_power_weight_fr_score1_quantile', 'explore_pic_power_weight_fr_score2_quantile', 'explore_pic_power_weight_pcltr_quantile', 'explore_pic_power_weight_pcmef_quantile', 'explore_pic_power_weight_pcmtr_quantile',
        'explore_pic_power_weight_pctr_quantile', 'explore_pic_power_weight_pdtr_quantile', 'explore_pic_power_weight_pepstr_quantile', 'explore_pic_power_weight_pevtr_quantile', 'explore_pic_power_weight_pftr_quantile',
        'explore_pic_power_weight_phtr_quantile', 'explore_pic_power_weight_pltr_quantile', 'explore_pic_power_weight_pptr_quantile', 'explore_pic_power_weight_pwtr_quantile', 'explore_pic_rerank_rank_mode',
        'explore_pxtr_quantile_map', "explore_pic_enable_fr_adjust_weights_by_emp_xtr"
      ]
    )

  def _score_enricher(self):
    return self.flow \
      .if_("skip_calc_pic_ltr_fvtr_with_pic_count == 0") \
          .enrich_attr_by_light_function(
              import_common_attr = [
                  "pic_ltr_fvtr_pic_count_max",
              ],
              import_item_attr = [
                  "photo_picture_count",
                  "pic_ltr_fvtr",
              ],
              export_item_attr = [
                  "pic_ltr_fvtr",
              ],
              function_name = "CalcPicLtrFvtr",
              class_name = "ExploreLightFunctionSetV2",
              target_item = { "is_picture": 1 }
          ) \
      .end_() \
      .if_("enable_pic_ltr_skip_single_pic == 1") \
          .set_attr_value(
              item_attrs=[{
                  "name": "pic_ltr_fvtr",
                  "type": "double",
                  "value": 0.0
              }],
              target_item = {
                  "is_picture": 1,
                  "picture_type": 1
              }
          ) \
      .end_() \
      .if_("enable_pic_ltr_fvtr_personalized_weight == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
              {"name": "explore_pic_power_weight_pic_fvtr", "as": "picture_weight"},
              {"name": "explore_pic_power_weight_pic_fvtr_single", "as": "single_picture_weight"},
              {"name": "explore_pic_power_weight_pic_fvtr_long", "as": "long_picture_weight"},
              {"name": "explore_pic_power_weight_pic_fvtr_set", "as": "picture_set_weight"},
          ],
          import_item_attr = [
              "picture_type",
          ],
          export_item_attr = [
              {"name": "weight_by_type", "as": "pic_ltr_fvtr_personalized_weight"},
          ],
          function_name = "CalEnsembleWeightByPicType",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
              "is_picture" : 1
          }
        ) \
      .end_() \
      .if_("enable_pic_ltr_enable_oppo_cost == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "explore_pic_fr_oppo_cost_q_weights",
            "explore_pic_fr_oppo_cost_q_pwatch_time_power",
            "explore_pic_fr_oppo_cost_q_pwatch_time_power2",
          ],
          import_item_attr = [
            "pctr",
            "pltr",
            "pwtr",
            "pftr",
            "pcltr",
            "pcmtr",
            "fr_score2",
          ],
          export_item_attr = [
            "fr_ensemble_pic_oppo_cost_score",
          ],
          function_name = "CalcFrPicOpportunityCostScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
              "is_picture" : 1
          }
        ) \
      .end_() \
      .if_("expl_pic_enable_fr_queue_weight_adjust == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "use_video_xtr", "as": "use_video_xtr"},
            "pic_stat_pic_like_cnt",
            "pic_stat_pic_follow_cnt",
            "pic_stat_pic_forward_cnt",
            "pic_stat_pic_comment_cnt",
            "pic_stat_pic_play_cnt",
            "pic_stat_video_like_cnt",
            "pic_stat_video_follow_cnt",
            "pic_stat_video_forward_cnt",
            "pic_stat_video_comment_cnt",
            "pic_stat_video_play_cnt",
          ],
          export_common_attr = [
            {"name": "pic_user_emp_ltr", "as": "pic_fr_user_emp_ltr"},
            {"name": "pic_user_emp_wtr", "as": "pic_fr_user_emp_wtr"},
            {"name": "pic_user_emp_ftr", "as": "pic_fr_user_emp_ftr"},
            {"name": "pic_user_emp_cmtr", "as": "pic_fr_user_emp_cmtr"},
            {"name": "pic_user_emp_cltr", "as": "pic_fr_user_emp_cltr"},
          ],
          function_name = "CalcPicUserEmpXtr",
          class_name = "ExploreLightFunctionSetV2",
          target_item = { "is_picture": 1 }
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_pic_fr_pxtr_attr_config_str", "as": "pxtr_attr_config_str"},
            {"name": "explore_pic_fr_avg_top_num", "as": "avg_top_num"},
          ],
          export_common_attr = [
            {"name": "pxtr_topn_avg_corr_pctr", "as": "pic_fr_pxtr_topn_avg_corr_pctr"},
            {"name": "pxtr_topn_avg_pltr", "as": "pic_fr_pxtr_topn_avg_pltr"},
            {"name": "pxtr_topn_avg_pwtr", "as": "pic_fr_pxtr_topn_avg_pwtr"},
            {"name": "pxtr_topn_avg_pcmtr", "as": "pic_fr_pxtr_topn_avg_pcmtr"},
            {"name": "pxtr_topn_avg_pcltr", "as": "pic_fr_pxtr_topn_avg_pcltr"},
          ],
          import_item_attr = ["corr_pctr", "pltr", "pwtr", "pcmtr", "pcltr"],
          function_name = "CalcPxtrStatScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item = { "is_picture": 1 }
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "expl_pic_fr_wgt_adj_mem_data_key", "as": "mem_data_key"},
            {"name": "expl_pic_fr_wgt_adj_pxtr_mem_data_suffix", "as": "pic_pxtr_mem_data_suffix"},
            {"name": "expl_pic_fr_wgt_adj_corr_pctr_boost_coef", "as": "corr_pctr_boost_coef"},
            {"name": "expl_pic_fr_wgt_adj_boost_comb_method", "as": "boost_comb_method"},
            {"name": "expl_pic_fr_wgt_adj_use_random", "as": "use_random"},
            {"name": "expl_pic_fr_wgt_adj_random_scale_base", "as": "random_scale_base"},
            {"name": "expl_pic_fr_wgt_adj_pltr_boost_coef", "as": "pltr_boost_coef"},
            {"name": "expl_pic_fr_wgt_adj_pwtr_boost_coef", "as": "pwtr_boost_coef"},
            {"name": "expl_pic_fr_wgt_adj_pcmtr_boost_coef", "as": "pcmtr_boost_coef"},
            {"name": "expl_pic_fr_wgt_adj_pcltr_boost_coef", "as": "pcltr_boost_coef"},

            {"name": "expl_pic_fr_wgt_adj_corr_pctr_boost_max", "as": "corr_pctr_boost_max"},
            {"name": "expl_pic_fr_wgt_adj_pltr_boost_max", "as": "pltr_boost_max"},
            {"name": "expl_pic_fr_wgt_adj_pwtr_boost_max", "as": "pwtr_boost_max"},
            {"name": "expl_pic_fr_wgt_adj_pcmtr_boost_max", "as": "pcmtr_boost_max"},
            {"name": "expl_pic_fr_wgt_adj_pcltr_boost_max", "as": "pcltr_boost_max"},
            {"name": "expl_pic_fr_wgt_adj_corr_pctr_boost_min", "as": "corr_pctr_boost_min"},
            {"name": "expl_pic_fr_wgt_adj_pltr_boost_min", "as": "pltr_boost_min"},
            {"name": "expl_pic_fr_wgt_adj_pwtr_boost_min", "as": "pwtr_boost_min"},
            {"name": "expl_pic_fr_wgt_adj_pcmtr_boost_min", "as": "pcmtr_boost_min"},
            {"name": "expl_pic_fr_wgt_adj_pcltr_boost_min", "as": "pcltr_boost_min"},
            {"name": "expl_pic_fr_wgt_adj_corr_pctr_pxtr_wgt", "as": "corr_pctr_pxtr_wgt"},
            {"name": "expl_pic_fr_wgt_adj_pltr_pxtr_wgt", "as": "pltr_pxtr_wgt"},
            {"name": "expl_pic_fr_wgt_adj_pwtr_pxtr_wgt", "as": "pwtr_pxtr_wgt"},
            {"name": "expl_pic_fr_wgt_adj_pcmtr_pxtr_wgt", "as": "pcmtr_pxtr_wgt"},
            {"name": "expl_pic_fr_wgt_adj_pcltr_pxtr_wgt", "as": "pcltr_pxtr_wgt"},
            {"name": "expl_pic_fr_wgt_adj_corr_pctr_emp_wgt", "as": "corr_pctr_emp_wgt"},
            {"name": "expl_pic_fr_wgt_adj_pltr_emp_wgt", "as": "pltr_emp_wgt"},
            {"name": "expl_pic_fr_wgt_adj_pwtr_emp_wgt", "as": "pwtr_emp_wgt"},
            {"name": "expl_pic_fr_wgt_adj_pcmtr_emp_wgt", "as": "pcmtr_emp_wgt"},
            {"name": "expl_pic_fr_wgt_adj_pcltr_emp_wgt", "as": "pcltr_emp_wgt"},

            {"name": "pic_fr_pxtr_topn_avg_corr_pctr", "as": "topk_avg_corr_pctr"},
            {"name": "pic_fr_pxtr_topn_avg_pltr", "as": "topk_avg_pltr"},
            {"name": "pic_fr_pxtr_topn_avg_pwtr", "as": "topk_avg_pwtr"},
            {"name": "pic_fr_pxtr_topn_avg_pcmtr", "as": "topk_avg_pcmtr"},
            {"name": "pic_fr_pxtr_topn_avg_pcltr", "as": "topk_avg_pcltr"},
            {"name": "pic_fr_user_emp_ltr", "as": "user_emp_ltr"},
            {"name": "pic_fr_user_emp_wtr", "as": "user_emp_wtr"},
            {"name": "pic_fr_user_emp_ftr", "as": "user_emp_ftr"},
            {"name": "pic_fr_user_emp_cmtr", "as": "user_emp_cmtr"},
            {"name": "pic_fr_user_emp_cltr", "as": "user_emp_cltr"},

            "explore_pic_fr_pxtr_pcts_ptr",
            "pic_stat_pic_play_cnt",
            "explore_pic_power_weight_fullrank_pctr_score",
            "explore_pic_power_weight_fullrank_pltr_score",
            "explore_pic_power_weight_fullrank_pwtr_score",
            "explore_pic_power_weight_fullrank_pcmtr_score",
            "explore_pic_power_weight_fullrank_pcltr_score",
          ],
          export_common_attr = [
            "explore_pic_power_weight_fullrank_pctr_score",
            "explore_pic_power_weight_fullrank_pltr_score",
            "explore_pic_power_weight_fullrank_pwtr_score",
            "explore_pic_power_weight_fullrank_pcmtr_score",
            "explore_pic_power_weight_fullrank_pcltr_score",
          ],
          function_name = "AdjustPicQueueWeights",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {"is_picture" : 1},
        ) \
      .end_() \
      .if_("explore_fr_pic_enable_single_pic_demote == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
              {"name": "explore_fr_single_pic_demote_score", "as": "explore_single_pic_demote_score"},
          ],
          import_item_attr = [
              "duration_ms",
              "upload_type",
              "picture_type",
              "photo_picture_count",
          ],
          export_item_attr = [
              {"name": "single_pic_demote_score", "as": "fr_single_pic_demote_score"},
          ],
          function_name = "DemoteSinglePic",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
              "is_picture" : 1
          }
        ) \
      .end_() \
      .if_("enable_fullrank_follow_author_pic_boost_v2 == 1") \
        .dispatch_common_attr(
          from_common_attr="fullrank_follow_author_pic_boost_coeff_v2",
          to_item_attr="fr_follow_author_pic_boost_score",
          target_item=
            {
              "is_picture_follow_author": 1,
              "is_picture": 1
            }
        ) \
      .end_() \
      .if_("enable_fullrank_target_hetu_pic_boost_v2 == 1") \
        .dispatch_common_attr(
          from_common_attr="fullrank_target_hetu_pic_boost_coeff_v2",
          to_item_attr="fr_target_hetu_pic_boost_score",
          target_item=
            {
              "is_boost_hetu_pic": 1,
              "is_picture": 1
            }
        ) \
      .end_() \
      .if_("enable_young_photo_ltr_boost == 1") \
        .set_attr_default_value(
          item_attrs=[
            {
              "name": "pic_ltr_young_photo_score",
              "type": "double",
              "value": "{{young_photo_pic_score}}"
            }
          ],
          target_item={
            "is_picture": 1,
            "is_young_photo": 1
          }
        ) \
      .end_() \
      .if_("enable_hv_pic_ltr_es_queue == 1") \
        .dispatch_common_attr(
          from_common_attr="hv_pic_score_fixed",
          to_item_attr="pic_ltr_hv_picture_score",
          target_item={
            "is_picture": 1,
            "high_value_pic_flag": 1
          }
        ) \
      .end_() \
      .if_("enable_fullrank_caption_pic_queue == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fullrank_caption_pic_queue_coef", "as": "caption_boost_coef"},
            {"name": "fullrank_caption_boost_len_thresh", "as": "caption_boost_len_thresh"},
            {"name": "fullrank_caption_boost_len_max", "as": "caption_boost_len_max"},
          ],
          import_item_attr = [
            "caption_length",
            "is_xhs_type_photo",
          ],
          export_item_attr = [
            {"name": "score", "as": "fr_pic_ensemble_long_caption_score"},
          ],
          function_name = "EnsembleCaptionSetScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item = { "is_picture": 1 }
        ) \
      .end_() \
      .if_("enable_explore_pic_fr_pxtr_calib == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_pic_fr_pwtr_calib_w", "as": "fr_pic_pwtr_calib_w"},
            {"name": "explore_pic_fr_pcmtr_calib_w", "as": "fr_pic_pcmtr_calib_w"},
            {"name": "explore_pic_fr_pctr_calib_w", "as": "fr_pic_pctr_calib_w"},
            {"name": "explore_pic_fr_pltr_calib_w", "as": "fr_pic_pltr_calib_w"},
            {"name": "explore_pic_fr_pftr_calib_w", "as": "fr_pic_pftr_calib_w"},
            {"name": "explore_pic_fr_pptr_calib_w", "as": "fr_pic_pptr_calib_w"},
            {"name": "explore_pic_fr_pcltr_calib_w", "as": "fr_pic_pcltr_calib_w"},
          ],
          import_item_attr = [
            "pwtr",
            "pcmtr",
            "pctr",
            "pltr",
            "pftr",
            "pptr",
            "pcltr",
          ],
          export_item_attr = [
            "pic_corr_pwtr",
            "pic_corr_pcmtr",
            "pic_corr_pctr",
            "pic_corr_pltr",
            "pic_corr_pftr",
            "pic_corr_pptr",
            "pic_corr_pcltr",
          ],
          function_name = "PicPxtrCalib",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
              "is_picture" : 1
          }
        ) \
      .end_() \
      .if_("enable_explore_pic_fr_pxtr_calib_by_emp_xtr == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_pic_fr_pwtr_emp_calib_w", "as": "fr_pic_pwtr_calib_w"},
            {"name": "explore_pic_fr_pcmtr_emp_calib_w", "as": "fr_pic_pcmtr_calib_w"},
            {"name": "explore_pic_fr_pctr_emp_calib_w", "as": "fr_pic_pctr_calib_w"},
            {"name": "explore_pic_fr_pltr_emp_calib_w", "as": "fr_pic_pltr_calib_w"},
            {"name": "explore_pic_fr_pftr_emp_calib_w", "as": "fr_pic_pftr_calib_w"},
            {"name": "explore_pic_fr_pptr_emp_calib_w", "as": "fr_pic_pptr_calib_w"},
            {"name": "explore_pic_fr_pwtr_emp_free_w", "as": "fr_pic_pwtr_free_w"},
            {"name": "explore_pic_fr_pcmtr_emp_free_w", "as": "fr_pic_pcmtr_free_w"},
            {"name": "explore_pic_fr_pctr_emp_free_w", "as": "fr_pic_pctr_free_w"},
            {"name": "explore_pic_fr_pltr_emp_free_w", "as": "fr_pic_pltr_free_w"},
            {"name": "explore_pic_fr_pftr_emp_free_w", "as": "fr_pic_pftr_free_w"},
            {"name": "explore_pic_fr_pptr_emp_free_w", "as": "fr_pic_pptr_free_w"},
            {"name": "explore_fr_pic_calib_show_thresh", "as": "fr_pic_calib_show_thresh"},
            {"name": "explore_fr_pic_calib_click_thresh", "as": "fr_pic_calib_click_thresh"},
            {"name": "explore_pic_fr_emp_debias_thresh", "as": "debias_thresh"},
          ],
          import_item_attr = [
            {"name": "pic_corr_pwtr", "as": "pwtr"},
            {"name": "pic_corr_pcmtr", "as": "pcmtr"},
            {"name": "pic_corr_pctr", "as": "pctr"},
            {"name": "pic_corr_pltr", "as": "pltr"},
            {"name": "pic_corr_pftr", "as": "pftr"},
            {"name": "pic_corr_pptr", "as": "pptr"},
            {"name": "pic_corr_pcltr", "as": "pcltr"},
            {"name": "explore_stat__click_count", "as": "click_count"},
            {"name": "explore_stat__comment_count", "as": "comment_count"},
            {"name": "explore_stat__like_count", "as": "like_count"},
            {"name": "explore_stat__follow_count", "as": "follow_count"},
            {"name": "explore_stat__forward_count", "as": "forward_count"},
            {"name": "explore_stat__real_show_count", "as": "real_show_count"},
            {"name": "explore_stat__profile_enter_count", "as": "profile_enter_count"},
          ],
          export_item_attr = [
            "pic_emp_corr_pwtr",
            "pic_emp_corr_pcmtr",
            "pic_emp_corr_pctr",
            "pic_emp_corr_pltr",
            "pic_emp_corr_pftr",
            "pic_emp_corr_pptr",
          ],
          function_name = "PicPxtrEmpCalib",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture" : 1
          }
        ) \
      .end_() \
      .if_("enable_pic_ranking_heat_boost == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "pic_ranking_heat_boost_decay_coeff", "as": "decay_coeff"},
            {"name": "pic_ranking_heat_boost_init_heat", "as": "init_heat"},
            {"name": "pic_ranking_heat_boost_min_heat", "as": "min_heat"},
          ],
          import_item_attr = [
            {"name": "upload_time", "as": "upload_time_ms"},
          ],
          export_item_attr = [
            {"name": "output_heat", "as": "pic_ranking_heat_boost_output_heat"},
          ],
          function_name = "LawOfCooling",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_pic_fr_behav_div_score == 1") \
        .pack_item_attr(
          item_source = {
            "reco_results": True,
          },
          mappings = [
            {
              "aggregator": "avg",
              "from_item_attr": "pctr",
              "to_common_attr": "pic_pctr_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "pltr",
              "to_common_attr": "pic_pltr_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "pwtr",
              "to_common_attr": "pic_pwtr_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "pcltr",
              "to_common_attr": "pic_pcltr_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "pcmtr",
              "to_common_attr": "pic_pcmtr_avg"
            },
             {
               "aggregator": "avg",
               "from_item_attr": "awesome_wtd",
               "to_common_attr": "pic_pwtd_avg"
             },
          ],
          target_item = {
            "is_picture" : 1
          },
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "pic_pctr_avg", "as": "pctr_avg"},
            {"name": "pic_pltr_avg", "as": "pltr_avg"},
            {"name": "pic_pwtr_avg", "as": "pwtr_avg"},
            {"name": "pic_pcltr_avg", "as": "pcltr_avg"},
            {"name": "pic_pcmtr_avg", "as": "pcmtr_avg"},
            {"name": "pic_pwtd_avg", "as": "pwtd_avg"},
            {"name": "expl_pic_behav_div_pctr_alpha", "as": "pctr_alpha"},
            {"name": "expl_pic_behav_div_pltr_alpha", "as": "pltr_alpha"},
            {"name": "expl_pic_behav_div_pwtr_alpha", "as": "pwtr_alpha"},
            {"name": "expl_pic_behav_div_pcltr_alpha", "as": "pcltr_alpha"},
            {"name": "expl_pic_behav_div_pcmtr_alpha", "as": "pcmtr_alpha"},
            {"name": "expl_pic_behav_div_pwtd_alpha", "as": "pwtd_alpha"},
            {"name": "expl_pic_behav_div_pctr_beta", "as": "pctr_beta"},
            {"name": "expl_pic_behav_div_pltr_beta", "as": "pltr_beta"},
            {"name": "expl_pic_behav_div_pwtr_beta", "as": "pwtr_beta"},
            {"name": "expl_pic_behav_div_pcltr_beta", "as": "pcltr_beta"},
            {"name": "expl_pic_behav_div_pcmtr_beta", "as": "pcmtr_beta"},
            {"name": "expl_pic_behav_div_pwtd_beta", "as": "pwtd_beta"},
            {"name": "expl_pic_behav_div_pwtd_max", "as": "pwtd_max"},
          ],
          import_item_attr = [
            {"name": "pctr", "as": "pctr"},
            {"name": "pltr", "as": "pltr"},
            {"name": "pwtr", "as": "pwtr"},
            {"name": "pcltr", "as": "pcltr"},
            {"name": "pcmtr", "as": "pcmtr"},
            {"name": "awesome_wtd", "as": "pwtd"},
          ],
          export_item_attr = [
            {"name": "pic_behaviour_diversity_score", "as": "pic_behaviour_diversity_score"}
          ],
          function_name = "GetPicBehaviourDiversityScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture" : 1
          },
        ) \
      .end_()

  def _variant(self):
    return self.flow \
      .if_("add_is_long_word_conver == 1") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            "ocr_cover_text_word_count",
          ],
          export_item_attr = [
            "is_long_word_conver"
          ],
          function_name = "IsLongWordConver",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {"is_picture": 1},
        ) \
      .end_() \
      .variant(
        variant_config = {
          "default_decay_window_size": 6,
          # 设置为 2 表示不允许重复，只能出现 1 次
          "default_decay_occurrent_times": 2,
          # 打散衰减系数
          "default_decay_rate": 0.2,
          # 按 author_id 打散
          "author__id": {
            "decay_window_size": "{{explore_pic_rerank_author_id_decay_window_size}}",
            "decay_occurrent_times": "{{explore_pic_rerank_author_id_decay_occurrent_times}}",
            "decay_rate": "{{explore_pic_rerank_author_id_decay_rate}}",
          },
          "is_good_looking": {
            "enabled": "{{enable_fr_good_looking_diversity}}",
            "decay_window_size": "{{explore_pic_rerank_good_looking_decay_window_size}}",
            "decay_occurrent_times": "{{explore_pic_rerank_good_looking_decay_occurrent_times}}",
            "decay_rate": "{{explore_pic_rerank_good_looking_decay_rate}}",
          },
          "hetu_tag_level_info__hetu_level_one": {
            "decay_window_size": "{{explore_pic_rerank_hetu_one_decay_window_size}}",
            "decay_occurrent_times": "{{explore_pic_rerank_hetu_one_decay_occurrent_times}}",
            "decay_rate": "{{explore_pic_rerank_hetu_one_decay_rate}}",
          },
          "hetu_tag_level_info__hetu_level_two": {
            "decay_window_size": "{{explore_pic_rerank_hetu_two_decay_window_size}}",
            "decay_occurrent_times": "{{explore_pic_rerank_hetu_two_decay_occurrent_times}}",
            "decay_rate": "{{explore_pic_rerank_hetu_two_decay_rate}}",
          },
          "hetu_tag_level_info__hetu_level_five": {
            "decay_window_size": "{{explore_pic_rerank_hetu_five_decay_window_size}}",
            "decay_occurrent_times": "{{explore_pic_rerank_hetu_five_decay_occurrent_times}}",
            "decay_rate": "{{explore_pic_rerank_hetu_five_decay_rate}}",
          },
          "hetu_tag_level_info__hetu_tag": {
            "enabled": "{{explore_pic_rerank_enable_hetu_tag_variant}}",
            "decay_window_size": "{{explore_pic_rerank_hetu_tag_decay_window_size}}",
            "decay_occurrent_times": "{{explore_pic_rerank_hetu_tag_decay_occurrent_times}}",
            "decay_rate": "{{explore_pic_rerank_hetu_tag_decay_rate}}",
          },
          "is_follow_author": {
            "enabled": "{{explore_pic_rerank_enable_is_follow_author}}",
            "decay_window_size": "{{explore_pic_rerank_is_follow_author_decay_window_size}}",
            "decay_occurrent_times": "{{explore_pic_rerank_is_follow_author_decay_occurrent_times}}",
            "decay_rate": "{{explore_pic_rerank_is_follow_author_decay_rate}}",
          },
          "photo_dnn_cluster_id": {
            "enabled": "{{explore_pic_rerank_enable_photo_dnn_cluster_id_variant}}",
            "decay_window_size": "{{explore_pic_rerank_photo_dnn_cluster_id_decay_window_size}}",
            "decay_occurrent_times": "{{explore_pic_rerank_photo_dnn_cluster_id_decay_occurrent_times}}",
            "decay_rate": "{{explore_pic_rerank_photo_dnn_cluster_id_decay_rate}}",
          },
          "is_long_word_conver": {
            "enabled": "{{add_is_long_word_conver}}",
            "decay_window_size": "{{explore_rerank_long_word_conver_decay_window_size}}",
            "decay_occurrent_times": "{{explore_rerank_long_word_conver_decay_occurrent_times}}",
            "decay_rate": "{{explore_rerank_long_word_conver_decay_rate}}",
          }
        },
        target_item = {
          "is_picture" : 1
        }
      )

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_num_limit = 20,
        common_attrs = [
        ],
        item_attrs = [
          "pic_ltr_weighted_ctr",
          "pic_ltr_lvtr",
          "pic_ltr_fvtr",
          "pic_ltr_wtd",
          "fr_pic_ensemble_score",
          "is_picture",
          "duration_ms",
          "upload_type",
          "author__id",
          "hetu_tag_level_info__hetu_level_one",
          "hetu_tag_level_info__hetu_level_two",
          "hetu_tag_level_info__hetu_level_five",
          "pic_ltr_fvtr_personalized_weight",
          "pctr",
          "pltr",
          "pwtr",
          "pftr",
        ],
        for_debug_request_only = True
      )
