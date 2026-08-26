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
        "valid_rank_smooth_attr": "explore_pic_fullrank_pctr_valid_rank_smooth",
        "valid_rank_weight_attr": "explore_pic_fullrank_pctr_valid_rank_weight",
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
        "valid_rank_smooth_attr": "explore_pic_fullrank_pltr_valid_rank_smooth",
        "valid_ori_score_attr": "explore_pic_fullrank_pltr_valid_ori_score",
        "valid_rank_weight_attr": "explore_pic_fullrank_pltr_valid_rank_weight",
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
        "valid_rank_smooth_attr": "explore_pic_fullrank_pwtr_valid_rank_smooth",
        "valid_ori_score_attr": "explore_pic_fullrank_pwtr_valid_ori_score",
        "valid_rank_weight_attr": "explore_pic_fullrank_pwtr_valid_rank_weight",
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
        "valid_rank_smooth_attr": "explore_pic_fullrank_pftr_valid_rank_smooth",
        "valid_ori_score_attr": "explore_pic_fullrank_pftr_valid_ori_score",
        "valid_rank_weight_attr": "explore_pic_fullrank_pftr_valid_rank_weight",
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
        "valid_rank_smooth_attr": "explore_pic_fullrank_pcmtr_valid_rank_smooth",
        "valid_ori_score_attr": "explore_pic_fullrank_pcmtr_valid_ori_score",
        "valid_rank_weight_attr": "explore_pic_fullrank_pcmtr_valid_rank_weight",
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
        "valid_rank_smooth_attr": "explore_pic_fullrank_pptr_valid_rank_smooth",
        "valid_rank_weight_attr": "explore_pic_fullrank_pptr_valid_rank_weight",
      },
      {
        "name": "pcmef",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pcmef_score",
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
        "valid_rank_smooth_attr": "explore_pic_fullrank_pcmef_valid_rank_smooth",
        "valid_rank_weight_attr": "explore_pic_fullrank_pcmef_valid_rank_weight",
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
        "valid_rank_smooth_attr": "explore_pic_fullrank_pevtr_valid_rank_smooth",
        "valid_ori_score_attr": "explore_pic_fullrank_pevtr_valid_ori_score",
        "valid_rank_weight_attr": "explore_pic_fullrank_pevtr_valid_rank_weight",
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
        "valid_rank_smooth_attr": "explore_pic_fullrank_psvr_valid_rank_smooth",
        "valid_ori_score_attr": "explore_pic_fullrank_psvr_valid_ori_score",
        "valid_rank_weight_attr": "explore_pic_fullrank_psvr_valid_rank_weight",
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
        "valid_rank_smooth_attr": "explore_pic_fullrank_fr_score1_valid_rank_smooth",
        "valid_rank_weight_attr": "explore_pic_fullrank_fr_score1_valid_rank_weight",
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
        "valid_rank_smooth_attr": "explore_pic_fullrank_fr_score2_valid_rank_smooth",
        "valid_rank_weight_attr": "explore_pic_fullrank_fr_score2_valid_rank_weight",
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
        "valid_rank_smooth_attr": "explore_pic_fullrank_pepstr_valid_rank_smooth",
        "valid_rank_weight_attr": "explore_pic_fullrank_pepstr_valid_rank_weight",
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
        "valid_rank_smooth_attr": "explore_pic_fullrank_pdtr_valid_rank_smooth",
        "valid_ori_score_attr": "explore_pic_fullrank_pdtr_valid_ori_score",
        "valid_rank_weight_attr": "explore_pic_fullrank_pdtr_valid_rank_weight",
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
        "valid_rank_smooth_attr": "explore_pic_fullrank_pcltr_valid_rank_smooth",
        "valid_ori_score_attr": "explore_pic_fullrank_pcltr_valid_ori_score",
        "valid_rank_weight_attr": "explore_pic_fullrank_pcltr_valid_rank_weight",
      },
      {
        "name": "phtr",
        "reverse_order": "{{explore_pic_fullrank_phtr_enable_reverse_order}}",
        "power_weight_attr": "explore_pic_power_weight_fullrank_phtr_score",
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
        "valid_rank_smooth_attr": "explore_pic_fullrank_phtr_valid_rank_smooth",
        "valid_ori_score_attr": "explore_pic_fullrank_phtr_valid_ori_score",
        "valid_rank_weight_attr": "explore_pic_fullrank_phtr_valid_rank_weight",
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
        "enable_norm_attr": "explore_pic_fullrank_pic_emp_corr_pctr_enable_norm",
        "power_weight_attr": "explore_pic_power_weight_fullrank_emp_corr_pctr_score",
        "raw_weight_attr": "explore_pic_emp_corr_pctr_raw_weight",
        "raw_bias_attr": "explore_pic_emp_corr_pctr_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_emp_corr_pctr_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_emp_corr_pctr_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_emp_corr_pctr_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_emp_corr_pctr_rank_temperature",
        "rank_smooth_attr": "explore_pic_emp_corr_pctr_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_emp_corr_pctr_rank_base_number",
        "rank_weight_attr": "explore_pic_emp_corr_pctr_rank_weight",
        "valid_rank_smooth_attr": "explore_pic_fullrank_pic_emp_corr_pctr_valid_rank_smooth",
        "valid_rank_weight_attr": "explore_pic_fullrank_pic_emp_corr_pctr_valid_rank_weight",
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
        "name": "fr_pic_pwtr_pct",
        "power_weight_attr": "explore_pic_power_weight_fullrank_fine_quantile_pwtr_score",
        "enable_norm_attr": "explore_pic_fullrank_pwtr_pct_enable_norm",
        "use_reciprocal": True,
      },
      {
        "name": "fr_pic_pcmtr_pct",
        "power_weight_attr": "explore_pic_power_weight_fullrank_fine_quantile_pcmtr_score",
        "enable_norm_attr": "explore_pic_fullrank_pcmtr_pct_enable_norm",
        "use_reciprocal": True,
      },
      {
        "name": "fr_pic_corr_pctr_pct",
        "power_weight_attr": "explore_pic_power_weight_fullrank_fine_quantile_pctr_score",
        "enable_norm_attr": "explore_pic_fullrank_corr_pctr_pct_enable_norm",
        "use_reciprocal": True,
      },
      {
        "name": "fr_pic_pltr_pct",
        "power_weight_attr": "explore_pic_power_weight_fullrank_fine_quantile_pltr_score",
        "enable_norm_attr": "explore_pic_fullrank_pltr_pct_enable_norm",
        "use_reciprocal": True,
      },
      {
        "name": "fr_pic_pftr_pct",
        "power_weight_attr": "explore_pic_power_weight_fullrank_fine_quantile_pftr_score",
        "enable_norm_attr": "explore_pic_fullrank_pftr_pct_enable_norm",
        "use_reciprocal": True,
      },
      {
        "name": "fr_pic_pptr_pct",
        "power_weight_attr": "explore_pic_power_weight_fullrank_fine_quantile_pptr_score",
        "enable_norm_attr": "explore_pic_fullrank_pptr_pct_enable_norm",
        "use_reciprocal": True,
      },
      {
        "name": "fr_pic_pcltr_pct",
        "power_weight_attr": "explore_pic_power_weight_fullrank_fine_quantile_pcltr_score",
        "enable_norm_attr": "explore_pic_fullrank_pcltr_pct_enable_norm",
        "use_reciprocal": True,
      },
      {
        "name": "pic_cluster_debias_pcltr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_cluster_debias_pcltr",
        "enable_norm_attr": "explore_pic_fullrank_cluster_debias_pcltr_enable_norm",
      },
      {
        "name": "pic_cluster_debias_pcmtr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_cluster_debias_pcmtr",
        "enable_norm_attr": "explore_pic_fullrank_cluster_debias_pcmtr_enable_norm",
      },
      {
        "name": "pic_cluster_debias_pctr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_cluster_debias_pctr",
        "enable_norm_attr": "explore_pic_fullrank_cluster_debias_pctr_enable_norm",
      },
      {
        "name": "pic_cluster_debias_pltr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_cluster_debias_pltr",
        "enable_norm_attr": "explore_pic_fullrank_cluster_debias_pltr_enable_norm",
      },
      {
        "name": "pic_cluster_debias_pftr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_cluster_debias_pftr",
        "enable_norm_attr": "explore_pic_fullrank_cluster_debias_pftr_enable_norm",
      },
      {
        "name": "pic_cluster_debias_pptr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_cluster_debias_pptr",
        "enable_norm_attr": "explore_pic_fullrank_cluster_debias_pptr_enable_norm",
      },
      {
        "name": "pic_cluster_debias_pwtr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_cluster_debias_pwtr",
        "enable_norm_attr": "explore_pic_fullrank_cluster_debias_pwtr_enable_norm",
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
        "valid_rank_smooth_attr": "explore_pic_fullrank_pic_variety_score_valid_rank_smooth",
        "valid_rank_weight_attr": "explore_pic_fullrank_pic_variety_score_valid_rank_weight",
      },
      {
        "name": "pic_diversity_mgs_score",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pic_diversity_mgs_score",
        "enable_norm_attr": "explore_pic_fullrank_diversity_mgs_score_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_diversity_mgs_score_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_diversity_mgs_score_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_diversity_mgs_score_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_diversity_mgs_score_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_diversity_mgs_score_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_diversity_mgs_score_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_diversity_mgs_score_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_diversity_mgs_score_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_diversity_mgs_score_rank_weight",
        "valid_rank_smooth_attr": "explore_pic_fullrank_diversity_mgs_score_valid_rank_smooth",
        "valid_ori_score_attr": "explore_pic_fullrank_diversity_mgs_score_valid_ori_score",
        "valid_rank_weight_attr": "explore_pic_fullrank_diversity_mgs_score_valid_rank_weight",
      },
      {
        "name": "vid2pic_sim_score",
        "power_weight_attr": "explore_pic_power_weight_fullrank_vid2pic_sim_score",
        "enable_norm_attr": "explore_pic_fullrank_vid2pic_sim_score_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_vid2pic_sim_score_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_vid2pic_sim_score_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_vid2pic_sim_score_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_vid2pic_sim_score_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_vid2pic_sim_score_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_vid2pic_sim_score_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_vid2pic_sim_score_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_vid2pic_sim_score_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_vid2pic_sim_score_rank_weight",
        "valid_rank_smooth_attr": "explore_pic_fullrank_vid2pic_sim_score_valid_rank_smooth",
        "valid_ori_score_attr": "explore_pic_fullrank_vid2pic_sim_score_valid_ori_score",
        "valid_rank_weight_attr": "explore_pic_fullrank_vid2pic_sim_score_valid_rank_weight",
      },
      {
        "name": "pic_emp_debias_pcltr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_hv_emp_debias_pcltr",
        "enable_norm_attr": "explore_pic_fullrank_debias_pcltr_enable_norm",
      },
      {
        "name": "pic_emp_debias_pctr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_hv_emp_debias_pctr",
        "enable_norm_attr": "explore_pic_fullrank_debias_pctr_enable_norm",
      },
      {
        "name": "pic_emp_debias_pltr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_hv_emp_debias_pltr",
        "enable_norm_attr": "explore_pic_fullrank_debias_pltr_enable_norm",
      },
      {
        "name": "pic_emp_debias_pwtr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_hv_emp_debias_pwtr",
        "enable_norm_attr": "explore_pic_fullrank_debias_pwtr_enable_norm",
      },
      {
        "name": "pic_emp_debias_pcmtr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_hv_emp_debias_pcmtr",
        "enable_norm_attr": "explore_pic_fullrank_debias_pcmtr_enable_norm",
      },
      {
        "name": "pic_bucket_corr_pctr",
        "power_weight_attr": "explore_pic_power_weight_bucket_corr_pctr"
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
        "valid_rank_smooth_attr": "explore_pic_fullrank_pic_behaviour_diversity_score_valid_rank_smooth",
        "valid_ori_score_attr": "explore_pic_fullrank_pic_behaviour_diversity_score_valid_ori_score",
        "valid_rank_weight_attr": "explore_pic_fullrank_pic_behaviour_diversity_score_valid_rank_weight",
      },
      {
        "name": "pic_fr_min_act_rank_score",
        "power_weight_attr": "explore_pic_power_weight_fr_min_act_rank_score",
      },
      {
        "name": "pic_ltv1",
        "power_weight_attr": "explore_pic_power_weight_pic_ltv1",
        "enable_norm_attr": "explore_pic_fullrank_pic_ltv1_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_pic_ltv1_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pic_ltv1_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pic_ltv1_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_pic_ltv1_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_pic_ltv1_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_pic_ltv1_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_pic_ltv1_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_pic_ltv1_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_pic_ltv1_rank_weight",
        "valid_rank_smooth_attr": "explore_pic_fullrank_pic_ltv1_valid_rank_smooth",
        "valid_rank_weight_attr": "explore_pic_fullrank_pic_ltv1_valid_rank_weight",
      },
      {
        "name": "pic_ltv2",
        "power_weight_attr": "explore_pic_power_weight_pic_ltv2",
      },
      {
        "name": "pic_ui_ltv_over_show",
        "power_weight_attr": "explore_pic_power_weight_pic_ui_ltv_over_show",
        "enable_norm_attr": "explore_pic_fullrank_ui_ltv_over_show_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_ui_ltv_over_show_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_ui_ltv_over_show_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_ui_ltv_over_show_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_ui_ltv_over_show_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_ui_ltv_over_show_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_ui_ltv_over_show_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_ui_ltv_over_show_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_ui_ltv_over_show_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_ui_ltv_over_show_rank_weight",
        "valid_rank_smooth_attr": "explore_pic_fullrank_ui_ltv_over_show_valid_rank_smooth",
        "valid_rank_weight_attr": "explore_pic_fullrank_ui_ltv_over_show_valid_rank_weight",
      },
      {
        "name": "pic_ui_ltv_over_click",
        "power_weight_attr": "explore_pic_power_weight_pic_ui_ltv_over_click",
        "enable_norm_attr": "explore_pic_fullrank_ui_ltv_over_click_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_ui_ltv_over_click_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_ui_ltv_over_click_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_ui_ltv_over_click_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_ui_ltv_over_click_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_ui_ltv_over_click_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_ui_ltv_over_click_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_ui_ltv_over_click_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_ui_ltv_over_click_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_ui_ltv_over_click_rank_weight",
        "valid_rank_smooth_attr": "explore_pic_fullrank_ui_ltv_over_click_valid_rank_smooth",
        "valid_rank_weight_attr": "explore_pic_fullrank_ui_ltv_over_click_valid_rank_weight",
      },
      {
        "name": "pic_ltr_for_good_comment",
        "power_weight_attr": "explore_pic_power_weight_pic_ltr_for_good_comment",
        "enable_norm_attr": "explore_pic_fullrank_ltr_for_good_comment_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_ltr_for_good_comment_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_ltr_for_good_comment_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_ltr_for_good_comment_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_ltr_for_good_comment_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_ltr_for_good_comment_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_ltr_for_good_comment_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_ltr_for_good_comment_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_ltr_for_good_comment_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_ltr_for_good_comment_rank_weight",
        "valid_rank_smooth_attr": "explore_pic_fullrank_ltr_for_good_comment_valid_rank_smooth",
        "valid_rank_weight_attr": "explore_pic_fullrank_ltr_for_good_comment_valid_rank_weight",
      },
      {
        "name": "fr_pic_div_ctr",
        "power_weight_attr": "explore_pic_power_weight_pic_div_ctr",
      },
      {
        "name": "pic_ltr_ctr_db",
        "power_weight_attr": "explore_pic_power_weight_pic_ltr_ctr_db",
        "enable_norm_attr": "explore_pic_fullrank_pic_ltr_ctr_db_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_pic_ltr_ctr_db_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pic_ltr_ctr_db_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pic_ltr_ctr_db_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_pic_ltr_ctr_db_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_pic_ltr_ctr_db_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_pic_ltr_ctr_db_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_pic_ltr_ctr_db_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_ltr_ctr_db_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_pic_ltr_ctr_db_rank_weight",
        "valid_rank_smooth_attr": "explore_pic_fullrank_pic_ltr_ctr_db_valid_rank_smooth",
        "valid_rank_weight_attr": "explore_pic_fullrank_pic_ltr_ctr_db_valid_rank_weight",
      },
      {
        "name": "pic_ltr_acttr_db",
        "power_weight_attr": "explore_pic_power_weight_pic_ltr_acttr_db",
        "enable_norm_attr": "explore_pic_fullrank_pic_ltr_acttr_db_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_pic_ltr_acttr_db_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pic_ltr_acttr_db_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pic_ltr_acttr_db_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_pic_ltr_acttr_db_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_pic_ltr_acttr_db_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_pic_ltr_acttr_db_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_pic_ltr_acttr_db_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_ltr_acttr_db_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_pic_ltr_acttr_db_rank_weight",
        "valid_rank_smooth_attr": "explore_pic_fullrank_pic_ltr_acttr_db_valid_rank_smooth",
        "valid_ori_score_attr": "explore_pic_fullrank_ltr_acttr_db_valid_ori_score",
        "valid_rank_weight_attr": "explore_pic_fullrank_pic_ltr_acttr_db_valid_rank_weight",
      },
      {
        "name": "pic_ltr_collect",
        "power_weight_attr": "explore_pic_power_weight_pic_ltr_collect",
        "enable_norm_attr": "explore_pic_fullrank_pic_ltr_collect_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_pic_ltr_collect_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pic_ltr_collect_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pic_ltr_collect_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_pic_ltr_collect_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_pic_ltr_collect_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_pic_ltr_collect_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_pic_ltr_collect_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_pic_ltr_collect_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_pic_ltr_collect_rank_weight",
        "valid_rank_smooth_attr": "explore_pic_fullrank_pic_ltr_collect_valid_rank_smooth",
        "valid_ori_score_attr": "explore_pic_fullrank_pic_ltr_collect_valid_ori_score",
        "valid_rank_weight_attr": "explore_pic_fullrank_pic_ltr_collect_valid_rank_weight",
      },
      {
        "name": "fr_pic_interest_ctr",
        "power_weight_attr": "explore_pic_power_weight_fr_pic_interest_ctr",
        "enable_norm_attr": "explore_pic_fullrank_fr_pic_interest_ctr_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_fr_pic_interest_ctr_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_fr_pic_interest_ctr_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_fr_pic_interest_ctr_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_fr_pic_interest_ctr_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_fr_pic_interest_ctr_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_fr_pic_interest_ctr_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_fr_pic_interest_ctr_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_fr_pic_interest_ctr_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_fr_pic_interest_ctr_rank_weight",
        "valid_rank_smooth_attr": "explore_pic_fullrank_fr_pic_interest_ctr_valid_rank_smooth",
        "valid_ori_score_attr": "explore_pic_fullrank_fr_pic_interest_ctr_valid_ori_score",
        "valid_rank_weight_attr": "explore_pic_fullrank_fr_pic_interest_ctr_valid_rank_weight",
      },
      {
        "name": "fr_pic_interest_acttr",
        "power_weight_attr": "explore_pic_power_weight_fr_pic_interest_acttr",
        "enable_norm_attr": "explore_pic_fullrank_fr_pic_interest_acttr_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_fr_pic_interest_acttr_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_fr_pic_interest_acttr_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_fr_pic_interest_acttr_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_fr_pic_interest_acttr_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_fr_pic_interest_acttr_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_fr_pic_interest_acttr_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_fr_pic_interest_acttr_rank_smooth",
        "rank_base_number_attr": "explore_pic_fullrank_fr_pic_interest_acttr_rank_base_number",
        "rank_weight_attr": "explore_pic_fullrank_fr_pic_interest_acttr_rank_weight",
        "valid_rank_smooth_attr": "explore_pic_fullrank_fr_pic_interest_acttr_valid_rank_smooth",
        "valid_ori_score_attr": "explore_pic_fullrank_fr_pic_interest_acttr_valid_ori_score",
        "valid_rank_weight_attr": "explore_pic_fullrank_fr_pic_interest_acttr_valid_rank_weight",
      },
      {
        "name": "pic_unbias_interset_score",
        "power_weight_attr": "explore_pic_power_weight_pic_unbias_interset_score",
        "enable_norm_attr": "explore_pic_fullrank_pic_unbias_interset_score_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_pic_unbias_interset_score_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pic_unbias_interset_score_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pic_unbias_interset_score_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_pic_unbias_interset_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_pic_unbias_interset_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_pic_unbias_interset_score_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_pic_unbias_interset_score_rank_smooth",
      },
      {
        "name": "pic_search_interest_cluster_score",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pic_search_interest_cluster_score",
        "enable_norm_attr": "explore_pic_fullrank_pic_search_interest_cluster_score_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_pic_search_interest_cluster_score_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pic_search_interest_cluster_score_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pic_search_interest_cluster_score_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_pic_search_interest_cluster_score_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_pic_search_interest_cluster_score_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_pic_search_interest_cluster_score_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_pic_search_interest_cluster_score_rank_smooth",
      },
      {
        "name": "fr_pic_search_score",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pic_search_score",
        "enable_norm_attr": "explore_pic_fullrank_pic_search_score_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_pic_search_score_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pic_search_score_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pic_search_score_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_pic_search_score_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_pic_search_score_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_pic_search_score_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_pic_search_score_rank_smooth",
      },
      {
        "name": "pic_real_pctr",
        "power_weight_attr": "explore_pic_power_weight_pic_real_pctr",
        "raw_weight_attr": "explore_pic_real_pctr_raw_weight",
        "raw_bias_attr": 'explore_pic_real_pctr_raw_bias',
        "rank_smooth_attr": "explore_pic_real_pctr_smooth",
      },
      {
        "name": "pic_real_pltr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pic_real_pltr",
        "pow_raw_score_attr": "explore_pic_fullrank_pic_real_pltr_pow_raw_score",
        "rank_smooth_attr": "explore_pic_fullrank_pic_real_pltr_rank_smooth",
        "enable_norm_attr": "explore_pic_fullrank_pic_real_pltr_enable_norm",
      },
      {
        "name": "pic_real_pwtr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pic_real_pwtr",
        "pow_raw_score_attr": "explore_pic_fullrank_pic_real_pwtr_pow_raw_score",
        "rank_smooth_attr": "explore_pic_fullrank_pic_real_pwtr_rank_smooth",
        "enable_norm_attr": "explore_pic_fullrank_pic_real_pwtr_enable_norm",
      },
      {
        "name": "pic_real_pftr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pic_real_pftr",
        "pow_raw_score_attr": "explore_pic_fullrank_pic_real_pftr_pow_raw_score",
        "rank_smooth_attr": "explore_pic_fullrank_pic_real_pftr_rank_smooth",
        "enable_norm_attr": "explore_pic_fullrank_pic_real_pftr_enable_norm",
      },
      {
        "name": "pic_real_pcmtr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pic_real_pcmtr",
        "pow_raw_score_attr": "explore_pic_fullrank_pic_real_pcmtr_pow_raw_score",
        "rank_smooth_attr": "explore_pic_fullrank_pic_real_pcmtr_rank_smooth",
        "enable_norm_attr": "explore_pic_fullrank_pic_real_pcmtr_enable_norm",
      },
      {
        "name": "pic_real_pdtr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pic_real_pdtr",
        "pow_raw_score_attr": "explore_pic_fullrank_pic_real_pdtr_pow_raw_score",
        "rank_smooth_attr": "explore_pic_fullrank_pic_real_pdtr_rank_smooth",
        "enable_norm_attr": "explore_pic_fullrank_pic_real_pdtr_enable_norm",
      },
      {
        "name": "pic_real_pcltr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pic_real_pcltr",
        "pow_raw_score_attr": "explore_pic_fullrank_pic_real_pcltr_pow_raw_score",
        "rank_smooth_attr": "explore_pic_fullrank_pic_real_pcltr_rank_smooth",
        "enable_norm_attr": "explore_pic_fullrank_pic_real_pcltr_enable_norm",
      },
      {
        "name": "pic_real_pevtr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pic_real_pevtr",
        "pow_raw_score_attr": "explore_pic_fullrank_pic_real_pevtr_pow_raw_score",
        "rank_smooth_attr": "explore_pic_fullrank_pic_real_pevtr_rank_smooth",
        "enable_norm_attr": "explore_pic_fullrank_pic_real_pevtr_enable_norm",
      },
      {
        "name": "pic_real_ltr_acttr_db",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pic_real_ltr_acttr_db",
        "pow_raw_score_attr": "explore_pic_fullrank_pic_real_ltr_acttr_db_pow_raw_score",
        "rank_smooth_attr": "explore_pic_fullrank_pic_real_ltr_acttr_db_rank_smooth",
        "enable_norm_attr": "explore_pic_fullrank_pic_real_ltr_acttr_db_enable_norm",
      },
      {
        "name": "pic_real_ltr_fvtr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pic_real_ltr_fvtr",
        "pow_raw_score_attr": "explore_pic_fullrank_pic_real_ltr_fvtr_pow_raw_score",
        "rank_smooth_attr": "explore_pic_fullrank_pic_real_ltr_fvtr_rank_smooth",
        "enable_norm_attr": "explore_pic_fullrank_pic_real_ltr_fvtr_enable_norm",
      },
      {
        "name": "pic_real_corr_pctr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pic_real_corr_pctr",
      },
      {
        "name": "explore_pic_rfm_hetul1_debias_age_gender_pctr",
        "power_weight_attr": "explore_pic_rfm_hetul1_debias_age_gender_pctr_pow_weight",
        "enable_norm_attr": "explore_pic_rfm_hetul1_debias_age_gender_pctr_enable_norm",
        "raw_weight_attr": "explore_pic_rfm_hetul1_debias_age_gender_pctr_raw_weight",
        "raw_bias_attr": "explore_pic_rfm_hetul1_debias_age_gender_pctr_raw_bias",
        "pow_raw_score_attr": "explore_pic_rfm_hetul1_debias_age_gender_pctr_raw_pow_weight",
        "raw_score_min_val_attr": "explore_pic_rfm_hetul1_debias_age_gender_pctr_min_val",
        "raw_score_max_val_attr": "explore_pic_rfm_hetul1_debias_age_gender_pctr_max_val",
        "rank_temperature_attr": "explore_pic_rfm_hetul1_debias_age_gender_pctr_rank_temp",
        "rank_smooth_attr": "explore_pic_rfm_hetul1_debias_age_gender_pctr_rank_smooth",
      },
      {
        "name": "explore_pic_rfm_hetul1_debias_age_gender_pltr",
        "power_weight_attr": "explore_pic_rfm_hetul1_debias_age_gender_pltr_pow_weight",
        "enable_norm_attr": "explore_pic_rfm_hetul1_debias_age_gender_pltr_enable_norm",
        "raw_weight_attr": "explore_pic_rfm_hetul1_debias_age_gender_pltr_raw_weight",
        "raw_bias_attr": "explore_pic_rfm_hetul1_debias_age_gender_pltr_raw_bias",
        "pow_raw_score_attr": "explore_pic_rfm_hetul1_debias_age_gender_pltr_raw_pow_weight",
        "raw_score_min_val_attr": "explore_pic_rfm_hetul1_debias_age_gender_pltr_min_val",
        "raw_score_max_val_attr": "explore_pic_rfm_hetul1_debias_age_gender_pltr_max_val",
        "rank_temperature_attr": "explore_pic_rfm_hetul1_debias_age_gender_pltr_rank_temp",
        "rank_smooth_attr": "explore_pic_rfm_hetul1_debias_age_gender_pltr_rank_smooth",
      },
      {
        "name": "explore_pic_rfm_hetul1_debias_age_gender_pwtr",
        "power_weight_attr": "explore_pic_rfm_hetul1_debias_age_gender_pwtr_pow_weight",
        "enable_norm_attr": "explore_pic_rfm_hetul1_debias_age_gender_pwtr_enable_norm",
        "raw_weight_attr": "explore_pic_rfm_hetul1_debias_age_gender_pwtr_raw_weight",
        "raw_bias_attr": "explore_pic_rfm_hetul1_debias_age_gender_pwtr_raw_bias",
        "pow_raw_score_attr": "explore_pic_rfm_hetul1_debias_age_gender_pwtr_raw_pow_weight",
        "raw_score_min_val_attr": "explore_pic_rfm_hetul1_debias_age_gender_pwtr_min_val",
        "raw_score_max_val_attr": "explore_pic_rfm_hetul1_debias_age_gender_pwtr_max_val",
        "rank_temperature_attr": "explore_pic_rfm_hetul1_debias_age_gender_pwtr_rank_temp",
        "rank_smooth_attr": "explore_pic_rfm_hetul1_debias_age_gender_pwtr_rank_smooth",
      },
      {
        "name": "explore_pic_rfm_hetul1_debias_age_gender_pftr",
        "power_weight_attr": "explore_pic_rfm_hetul1_debias_age_gender_pftr_pow_weight",
        "enable_norm_attr": "explore_pic_rfm_hetul1_debias_age_gender_pftr_enable_norm",
        "raw_weight_attr": "explore_pic_rfm_hetul1_debias_age_gender_pftr_raw_weight",
        "raw_bias_attr": "explore_pic_rfm_hetul1_debias_age_gender_pftr_raw_bias",
        "pow_raw_score_attr": "explore_pic_rfm_hetul1_debias_age_gender_pftr_raw_pow_weight",
        "raw_score_min_val_attr": "explore_pic_rfm_hetul1_debias_age_gender_pftr_min_val",
        "raw_score_max_val_attr": "explore_pic_rfm_hetul1_debias_age_gender_pftr_max_val",
        "rank_temperature_attr": "explore_pic_rfm_hetul1_debias_age_gender_pftr_rank_temp",
        "rank_smooth_attr": "explore_pic_rfm_hetul1_debias_age_gender_pftr_rank_smooth",
      },
      {
        "name": "explore_pic_rfm_hetul1_debias_age_gender_pcmtr",
        "power_weight_attr": "explore_pic_rfm_hetul1_debias_age_gender_pcmtr_pow_weight",
        "enable_norm_attr": "explore_pic_rfm_hetul1_debias_age_gender_pcmtr_enable_norm",
        "raw_weight_attr": "explore_pic_rfm_hetul1_debias_age_gender_pcmtr_raw_weight",
        "raw_bias_attr": "explore_pic_rfm_hetul1_debias_age_gender_pcmtr_raw_bias",
        "pow_raw_score_attr": "explore_pic_rfm_hetul1_debias_age_gender_pcmtr_raw_pow_weight",
        "raw_score_min_val_attr": "explore_pic_rfm_hetul1_debias_age_gender_pcmtr_min_val",
        "raw_score_max_val_attr": "explore_pic_rfm_hetul1_debias_age_gender_pcmtr_max_val",
        "rank_temperature_attr": "explore_pic_rfm_hetul1_debias_age_gender_pcmtr_rank_temp",
        "rank_smooth_attr": "explore_pic_rfm_hetul1_debias_age_gender_pcmtr_rank_smooth",
      },
      {
        "name": "explore_pic_rfm_hetul1_debias_age_gender_pcltr",
        "power_weight_attr": "explore_pic_rfm_hetul1_debias_age_gender_pcltr_pow_weight",
        "enable_norm_attr": "explore_pic_rfm_hetul1_debias_age_gender_pcltr_enable_norm",
        "raw_weight_attr": "explore_pic_rfm_hetul1_debias_age_gender_pcltr_raw_weight",
        "raw_bias_attr": "explore_pic_rfm_hetul1_debias_age_gender_pcltr_raw_bias",
        "pow_raw_score_attr": "explore_pic_rfm_hetul1_debias_age_gender_pcltr_raw_pow_weight",
        "raw_score_min_val_attr": "explore_pic_rfm_hetul1_debias_age_gender_pcltr_min_val",
        "raw_score_max_val_attr": "explore_pic_rfm_hetul1_debias_age_gender_pcltr_max_val",
        "rank_temperature_attr": "explore_pic_rfm_hetul1_debias_age_gender_pcltr_rank_temp",
        "rank_smooth_attr": "explore_pic_rfm_hetul1_debias_age_gender_pcltr_rank_smooth",
      },
      {
        "name": "cascading_explore_gamora_interest_ptr",
        "power_weight_attr": "explore_pic_rank_gamora_interest_ptr_pow_weight",
        "enable_norm_attr": "explore_pic_rank_gamora_interest_ptr_enable_norm",
        "raw_weight_attr": "explore_pic_rank_gamora_interest_ptr_raw_weight",
        "raw_bias_attr": "explore_pic_rank_gamora_interest_ptr_raw_bias",
        "pow_raw_score_attr": "explore_pic_rank_gamora_interest_ptr_raw_pow_weight",
        "raw_score_min_val_attr": "explore_pic_rank_gamora_interest_ptr_min_val",
        "raw_score_max_val_attr": "explore_pic_rank_gamora_interest_ptr_max_val",
        "rank_temperature_attr": "explore_pic_rank_gamora_interest_ptr_rank_temp",
        "rank_smooth_attr": "explore_pic_rank_gamora_interest_ptr_rank_smooth",
      },
      {
        "name": "cascading_explore_gamora_interest_ltr",
        "power_weight_attr": "explore_pic_rank_gamora_interest_ltr_pow_weight",
        "enable_norm_attr": "explore_pic_rank_gamora_interest_ltr_enable_norm",
        "raw_weight_attr": "explore_pic_rank_gamora_interest_ltr_raw_weight",
        "raw_bias_attr": "explore_pic_rank_gamora_interest_ltr_raw_bias",
        "pow_raw_score_attr": "explore_pic_rank_gamora_interest_ltr_raw_pow_weight",
        "raw_score_min_val_attr": "explore_pic_rank_gamora_interest_ltr_min_val",
        "raw_score_max_val_attr": "explore_pic_rank_gamora_interest_ltr_max_val",
        "rank_temperature_attr": "explore_pic_rank_gamora_interest_ltr_rank_temp",
        "rank_smooth_attr": "explore_pic_rank_gamora_interest_ltr_rank_smooth",
      },
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
        "valid_rank_smooth_attr": "explore_pic_fullrank_consume_time_ltr_valid_rank_smooth",
        "valid_rank_weight_attr": "explore_pic_fullrank_consume_time_ltr_valid_rank_weight",
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
        "valid_rank_smooth_attr": "explore_pic_fullrank_pic_ltr_weighted_ctr_valid_rank_smooth",
        "valid_ori_score_attr": "explore_pic_fullrank_pic_ltr_weighted_ctr_valid_ori_score",
        "spec_rank_mode_attr": "explore_pic_fullrank_pic_ltr_weighted_ctr_spec_rank_mode",
        "valid_rank_weight_attr": "explore_pic_fullrank_pic_ltr_weighted_ctr_valid_rank_weight",
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
        "valid_rank_smooth_attr": "explore_pic_fullrank_pic_ltr_fvtr_valid_rank_smooth",
        "valid_rank_weight_attr": "explore_pic_fullrank_pic_ltr_fvtr_valid_rank_weight",
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
        "valid_rank_smooth_attr": "explore_pic_fullrank_fr_single_pic_demote_score_valid_rank_smooth",
        "valid_rank_weight_attr": "explore_pic_fullrank_fr_single_pic_demote_score_valid_rank_weight",
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
        "valid_rank_smooth_attr": "explore_pic_fullrank_follow_author_pic_boost_score_valid_rank_smooth",
        "valid_rank_weight_attr": "explore_pic_fullrank_follow_author_pic_boost_score_valid_rank_weight",
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
        "valid_rank_smooth_attr": "explore_pic_fullrank_target_hetu_pic_boost_score_valid_rank_smooth",
        "valid_rank_weight_attr": "explore_pic_fullrank_target_hetu_pic_boost_score_valid_rank_weight",
        "default": 1.0
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
        "valid_rank_smooth_attr": "explore_pic_fullrank_pic_ensemble_long_caption_score_valid_rank_smooth",
        "valid_rank_weight_attr": "explore_pic_fullrank_pic_ensemble_long_caption_score_valid_rank_weight",
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
        "valid_rank_smooth_attr": "explore_pic_fullrank_pic_diversity_score_valid_rank_smooth",
        "valid_rank_weight_attr": "explore_pic_fullrank_pic_diversity_score_valid_rank_weight",
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
        "name": "pic_ltr_like_timing",
        "power_weight_attr": "explore_pic_power_weight_pic_ltr_like_timing",
      },
      {
        "name": "pic_ltr_action_twice",
        "power_weight_attr": "explore_pic_power_weight_pic_ltr_action_twice",
      },
      {
        "name": "fr_pic_u2u_acttr",
        "power_weight_attr": "explore_pic_power_weight_u2u_acttr",
        "enable_norm_attr": "explore_pic_fullrank_u2u_acttr_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_u2u_acttr_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_u2u_acttr_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_u2u_acttr_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_u2u_acttr_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_u2u_acttr_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_u2u_acttr_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_u2u_acttr_rank_smooth",
      },
      {
        "name": "fr_pic_u2u_evtr",
        "power_weight_attr": "explore_pic_power_weight_u2u_evtr",
        "enable_norm_attr": "explore_pic_fullrank_u2u_evtr_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_u2u_evtr_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_u2u_evtr_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_u2u_evtr_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_u2u_evtr_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_u2u_evtr_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_u2u_evtr_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_u2u_evtr_rank_smooth",
      },
      {
        "name": "fr_pic_ua_action_score",
        "power_weight_attr": "explore_pic_power_weight_ua_action_score",
        "enable_norm_attr": "explore_pic_fullrank_ua_action_score_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_ua_action_score_raw_weight",
        "raw_bias_attr": 'explore_pic_fullrank_ua_action_score_raw_bias',
        "pow_raw_score_attr": "explore_pic_fullrank_ua_action_score_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_ua_action_score_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_ua_action_score_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_ua_action_score_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_ua_action_score_rank_smooth",
        "rank_weight_attr": "explore_pic_fullrank_ua_action_score_rank_weight",
      },
      {
        "name": "fr_pic_ua_click_score",
        "power_weight_attr": "explore_pic_power_weight_ua_click_score",
        "enable_norm_attr": "explore_pic_fullrank_ua_click_score_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_ua_click_score_raw_weight",
        "raw_bias_attr": 'explore_pic_fullrank_ua_click_score_raw_bias',
        "pow_raw_score_attr": "explore_pic_fullrank_ua_click_score_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_ua_click_score_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_ua_click_score_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_ua_click_score_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_ua_click_score_rank_smooth",
        "rank_weight_attr": "explore_pic_fullrank_ua_click_score_rank_weight",
      },
      {
        "name": "pic_pxtr_fusion_score",
        "power_weight_attr": "explore_pic_power_weight_pic_pxtr_fusion_score",
        "enable_norm_attr": "explore_pic_fullrank_pic_pxtr_fusion_score_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_pic_pxtr_fusion_score_raw_weight",
        "raw_bias_attr": 'explore_pic_fullrank_pic_pxtr_fusion_score_raw_bias',
        "pow_raw_score_attr": "explore_pic_fullrank_pic_pxtr_fusion_score_pow_raw_score",
        "raw_score_min_val_attr": "explore_pic_fullrank_pic_pxtr_fusion_score_raw_score_min_val",
        "raw_score_max_val_attr": "explore_pic_fullrank_pic_pxtr_fusion_score_raw_score_max_val",
        "rank_temperature_attr": "explore_pic_fullrank_pic_pxtr_fusion_score_rank_temperature",
        "rank_smooth_attr": "explore_pic_fullrank_pic_pxtr_fusion_score_rank_smooth",
      },
      {
        "name": "pic_valid_interest_tag_score",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pic_valid_interest_tag_score",
        "enable_norm_attr": "explore_pic_fullrank_pic_valid_interest_tag_score_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_pic_valid_interest_tag_score_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pic_valid_interest_tag_score_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pic_valid_interest_tag_score_pow_raw_score",
      },
      {
        "name": "fr_pic_u2c_ensemble_score",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pic_u2c_ensemble_score",
        "enable_norm_attr": "explore_pic_fullrank_pic_u2c_ensemble_score_enable_norm",
        "raw_weight_attr": "explore_pic_fullrank_pic_u2c_ensemble_score_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pic_u2c_ensemble_score_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pic_u2c_ensemble_score_pow_raw_score",
      },
      {
        "name": "pic_search_interest_tagnex_score",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pic_search_interest_tagnex_score",
        "enable_norm_attr": "explore_pic_fullrank_pic_search_interest_tagnex_score_enable_norm",
        "rank_smooth_attr": "explore_pic_fullrank_pic_search_interest_tagnex_score_rank_smooth",
        "raw_weight_attr": "explore_pic_fullrank_pic_search_interest_tagnex_score_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pic_search_interest_tagnex_score_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pic_search_interest_tagnex_score_pow_raw_score",
      },
      {
        "name": "pic_fr_comment_quality_score",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pic_fr_comment_quality_score",
        "enable_norm_attr": "explore_pic_fullrank_pic_fr_comment_quality_score_enable_norm",
        "rank_smooth_attr": "explore_pic_fullrank_pic_fr_comment_quality_score_rank_smooth",
        "raw_weight_attr": "explore_pic_fullrank_pic_fr_comment_quality_score_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pic_fr_comment_quality_score_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pic_fr_comment_quality_score_pow_raw_score",
      },
      {
        "name": "pic_u2c_collaborative_score",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pic_u2c_collaborative_score",
        "enable_norm_attr": "explore_pic_fullrank_pic_u2c_collaborative_score_enable_norm",
        "rank_smooth_attr": "explore_pic_fullrank_pic_u2c_collaborative_score_rank_smooth",
        "raw_weight_attr": "explore_pic_fullrank_pic_u2c_collaborative_score_raw_weight",
        "raw_bias_attr": "explore_pic_fullrank_pic_u2c_collaborative_score_raw_bias",
        "pow_raw_score_attr": "explore_pic_fullrank_pic_u2c_collaborative_score_pow_raw_score",
      },
      {
        "name": "pic_career_interest_tagnex_tgi_score", # 2026-06-03 by zhangziqian03
        "power_weight_attr": "explore_pic_power_weight_fullrank_pic_career_interest_tagnex_tgi_score",
      },
      {
        "name": "pic_age_interest_tagnex_tgi_score", # 2026-06-03 by zhangziqian03
        "power_weight_attr": "explore_pic_power_weight_fullrank_pic_age_interest_tagnex_tgi_score",
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

  def min_rank_queues(self):
    return [
        {
          "name" : "pctr",
          "enable_attr": "explore_pic_fr_min_rank_score_enable_pctr"
        },
        {
          "name" : "pltr",
          "enable_attr": "explore_pic_fr_min_rank_score_enable_pltr"
        },
        {
          "name" : "pwtr",
          "enable_attr": "explore_pic_fr_min_rank_score_enable_pwtr"
        },
        {
          "name" : "pcmtr",
          "enable_attr": "explore_pic_fr_min_rank_score_enable_pcmtr"
        },
        {
          "name" : "pcltr",
          "enable_attr": "explore_pic_fr_min_rank_score_enable_pcltr"
        },
        {
          "name" : "pftr",
          "enable_attr": "explore_pic_fr_min_rank_score_enable_pftr"
        },
        {
          "name" : "pptr",
          "enable_attr": "explore_pic_fr_min_rank_score_enable_pptr"
        },
        {
          "name": "awesome_wtd",
          "enable_attr": "explore_pic_fr_min_rank_score_enable_awesome_wtd"
        },
        {
          "name": "fr_score1",
          "enable_attr": "explore_pic_fr_min_rank_score_enable_fr_score1"
        },
        {
          "name": "fr_score2",
          "enable_attr": "explore_pic_fr_min_rank_score_enable_fr_score2"
        }
      ]


  def process(self) -> None:
    self.flow.if_("skip_explore_pic_rerank == 0") \

    self._score_enricher()

    self.flow \
        .if_("enable_explore_first_screen_ranking_refactor == 1 and is_first_refresh > 0") \
          .calc_by_formula1(
            kconf_key = "formula.scenarioKey37.explore_first_screen_rank_picture_value",
            target_item = {
              "is_picture": 1
            },
            import_item_attr = [
              "pltr",
              "pwtr",
              "pftr",
              "pcmtr",
              "pcltr",
              "pdtr",
              "pptr",
              "pevtr",
              "psvr",
              "fr_score1",
              "fr_score2",
              "pepstr",
              "phtr",
              "pic_ltr_weighted_ctr",
              "pic_ltr_fvtr",
              "pic_pxtr_fusion_score",
              "pic_ltr_for_good_comment",
              "pic_real_pctr",
              "pic_real_pltr",
              "pic_real_pwtr",
              "pic_real_pftr",
              "pic_real_pcltr",
              "fr_follow_author_pic_boost_score",
              "pic_ltr_ctr_db",
              "pic_ltr_acttr_db",
              "pic_emp_corr_pctr",
              "pic_ui_ltv_over_click",
              "explore_pic_rfm_hetul1_debias_age_gender_pctr",
              "explore_pic_rfm_hetul1_debias_age_gender_pltr",
              "explore_pic_rfm_hetul1_debias_age_gender_pwtr",
              "explore_pic_rfm_hetul1_debias_age_gender_pftr",
              "explore_pic_rfm_hetul1_debias_age_gender_pcmtr",
              "explore_pic_rfm_hetul1_debias_age_gender_pcltr",
              "consume_time_ltr"
            ],
            export_formula_value = [
                {"name": "first_screen_fr_pic_ensemble_score", "as": "fr_pic_ensemble_score"}
            ],
            abtest_biz_name = "KUAISHOU_APPS",
            perf_tag = "{{explore_first_screen_ranking_refactor_f1_perf_tag}}"
           ) \
        .else_() \
          .explore_pic_rerank(
            save_score_to_attr = "fr_pic_ensemble_score",
            limit = "{{explore_pic_rerank_limit}}",
            picture_attr = "is_picture",
            picture_type_attr = "picture_type",
            queues = self.ensemble_queues(),
            unify_rank_mode = "{{explore_pic_rerank_unify_rank_mode}}",
            unify_queue_calc_mode = "{{explore_pic_rerank_unify_queue_calc_mode}}",
            valid_rank_mode = "{{explore_pic_rerank_unify_valid_rank_mode}}",
            valid_rank_exclude_single_pic = "{{enable_rerank_valid_rank_exclude_single_pic}}",
            queue_max_raw_score = "{{explore_pic_rerank_queue_max_raw_score}}",
            queue_min_raw_score = "{{explore_pic_rerank_queue_min_raw_score}}",
            queue_max_rank_score = "{{explore_pic_rerank_queue_max_rank_score}}",
            queue_min_rank_score = "{{explore_pic_rerank_queue_min_rank_score}}",
            enable_normalization_item_score = "{{explore_pic_rerank_enable_normalization_item_score}}",
            enable_2sigma_overall_ori_pxtr = "{{explore_pic_rerank_enable_2sigma_overall_ori_pxtr}}",
          ) \
        .end_()

    
    # 图文一阶段 boost
    self.flow.if_("enable_explore_pic_fr_es_adjust == 1")
    self._fr_es_adjust()
    self.flow.end_()
    
    # 图文打散
    self.flow.if_("skip_explore_pic_rerank_variant == 0")
    self._variant()
    self.flow.end_()
    
    # 图文前置强插
    self.flow.if_("enable_explore_pic_fr_insert_before_overwrite_fr_es == 1")
    self._fr_insert()
    self.flow.end_()

    #  rerank 分改为 1 / (rank + smooth)
    self._fr_es_overwrite()
    
    # 图文后强插
    self.flow.if_("enable_explore_pic_fr_insert_before_overwrite_fr_es == 0")
    self._fr_insert()
    self.flow.end_()
    
    # 图文二阶段 boost (会改变 rerank 分)
    self.flow.if_("enable_explore_pic_fr_es_adjust_after_overwrite == 1")
    self._fr_es_adjust_after_overwrite()
    self.flow.end_()

    self.flow \
      .if_("enable_explore_pic_fr_final_sort == 1") \
        .sort(
          score_from_attr = "fr_pic_ensemble_score",
          target_item = {
            "is_picture": 1
          }
        ) \
      .end_()

    self.flow.end_()

  def _score_enricher(self):
    self.flow \
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
            "enable_explore_pic_real_pxtr_power_adjust",
            "explore_pic_power_weight_fullrank_pic_real_corr_pctr",
            "explore_pic_power_weight_fullrank_pic_real_pltr",
            "explore_pic_power_weight_fullrank_pic_real_pwtr",
            "explore_pic_power_weight_fullrank_pic_real_pcmtr",
            "explore_pic_power_weight_fullrank_pic_real_pcltr"
          ],
          export_common_attr = [
            "explore_pic_power_weight_fullrank_pctr_score",
            "explore_pic_power_weight_fullrank_pltr_score",
            "explore_pic_power_weight_fullrank_pwtr_score",
            "explore_pic_power_weight_fullrank_pcmtr_score",
            "explore_pic_power_weight_fullrank_pcltr_score",
            "explore_pic_power_weight_fullrank_pic_real_corr_pctr",
            "explore_pic_power_weight_fullrank_pic_real_pltr",
            "explore_pic_power_weight_fullrank_pic_real_pwtr",
            "explore_pic_power_weight_fullrank_pic_real_pcmtr",
            "explore_pic_power_weight_fullrank_pic_real_pcltr"
          ],
          function_name = "AdjustPicQueueWeights",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {"is_picture" : 1},
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
      .if_("enable_explore_hv_pic_fr_pxtr_calib_by_emp == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "explore_pic_xtr_emp_debias_map_ptr",
            "basic_info_gender_v2",
            {"name": "basic_info_age_segment_v2", "as":"age_segment"},
            {"name": "explore_hv_pic_fr_emp_debias_thresh", "as": "debias_thresh"},
            {"name": "explore_hv_pic_fr_emp_debias_redis_prefix", "as": "redis_prefix"}
          ],
          import_item_attr = [
            "pwtr",
            "pcmtr",
            "pltr",
            "pcltr",
            "high_value_pic_flag",
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_level_one_list"},
            {"name": "pic_ltr_weighted_ctr", "as": "pctr"},
          ],
          export_item_attr = [
            "pic_emp_debias_pwtr",
            "pic_emp_debias_pcmtr",
            "pic_emp_debias_pctr",
            "pic_emp_debias_pltr",
            "pic_emp_debias_pcltr"
          ],
          function_name = "HvPicPxtrEmpDebias",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture" : 1
          }
        ) \
      .end_() \
      .if_("enable_explore_pic_fr_pxtr_calib_by_bucket_cluster_emp_xtr == 1", to_be_delete = "date=2024-05-29;committer=chenluoxiang") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "user_age_segment", "as": "uAgeSeg"},
            {"name": "pic_bucket_calib_param_map", "as": "param_map"}
          ],
          import_item_attr = [
            {"name": "pctr", "as": "pPctr"},
          ],
          export_item_attr = [
            "pic_bucket_corr_pctr"
          ],
          function_name = "PicPxtrBucketCalib",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture" : 1
          }
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
      .end_() \
      .if_("enable_explore_pic_fr_min_act_rank_score == 1", to_be_delete = "date=2024-05-29;committer=zhuwenyong") \
        .explore_min_act_rank_score_enricher(
          target_item = { "is_picture": 1 },
          max_rank_ratio = "{{explore_pic_fr_max_rank_ratio}}",
          queues = self.min_rank_queues(),
          save_score_to_attr = "pic_fr_min_act_rank_score"
        ) \
      .end_() \
      .if_("enable_explore_pic_vv_control_fr_queue_weight_adjust == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_pic_rerank_top_pic_vv_recent_thre", "as": "top_pic_vv_recent_thre"},
            {"name": "explore_pic_rerank_tail_pic_vv_recent_thre", "as": "tail_pic_vv_recent_thre"},
            {"name": "pic_stat_pic_recent_play_cnt", "as": "pic_recent_play_cnt"},
            {"name": "explore_pic_power_weight_fullrank_pic_diversity_mgs_score", "as": "diversity_mgs_score_power_weight"},
            {"name": "explore_rerank_pic_diversity_mgs_score_power_coeff", "as": "power_coeff"},
          ],
          export_common_attr = [
            {"name": "diversity_mgs_score_power_weight", "as": "explore_pic_power_weight_fullrank_pic_diversity_mgs_score"},
          ],
          function_name = "CalcPicVVControlMgsWeight",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture": 1
          }
        ) \
      .end_() \
      .if_("enable_explore_pic_real_pctr == 1 and ((uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_pic_interest_thresh or explore_zero_show_pic_user_first_screen == 1)") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            {"name": "corr_pctr", "as": "pctr"},
            {"name": "psvr", "as": "psvr"},
          ],
          export_item_attr = [
            {"name": "real_pctr", "as": "pic_real_pctr"},
          ],
          function_name = "CalcRealPctr",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture" : 1
          }
        ) \
      .end_()\
      .if_("enable_explore_personalized_pic_real_pctr_filter_by_age > 0 and ((basic_info_age_segment_v2 > enable_explore_personalized_pic_real_pctr_age_max) or (basic_info_age_segment_v2 < enable_explore_personalized_pic_real_pctr_age_min))") \
        .set_attr_value(
          no_overwrite = True,
          common_attrs = [
            {
              "name": "explore_personalized_pic_real_pctr_target_age_user",
              "type": "int",
              "value": 0,
            }
          ]
        ) \
      .end_() \
      .if_("enable_explore_personalized_pic_real_pctr > 0 and (uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_personalized_pic_real_pctr_interest_thresh and (explore_personalized_pic_real_pctr_target_age_user or 1) > 0") \
        .enrich_attr_by_light_function(  # 计算精排个性化权重
          import_common_attr = [
            {"name": "explore_pic_real_pctr_weight_config_str", "as": "pxtr_attr_config_str"},
            {"name": "explore_pic_real_pctr_weight_avg_top_num", "as": "avg_top_num"},
            {"name": "explore_pic_real_pctr_weight_enable_trans", "as": "enable_trans"},
            {"name": "explore_pic_real_pctr_weight_trans_alpha", "as": "trans_alpha"},
            {"name": "explore_pic_real_pctr_weight_trans_bias", "as": "trans_bias"},
            {"name": "explore_pic_real_pctr_weight_trans_pow", "as": "trans_pow"},
            {"name": "explore_pic_real_pctr_weight_trans_min", "as": "trans_min"},
            {"name": "explore_pic_real_pctr_weight_trans_max", "as": "trans_max"},
          ],
          export_common_attr = [
            {"name": "pxtr_topn_avg_score", "as": "explore_pic_power_weight_pic_real_pctr"},
          ],
          import_item_attr = [
            "corr_pctr",
            "pltr",
            "pwtr",
            "pcltr",
            "pcmtr",
            "pftr",
            "pdtr",
          ],
          function_name = "CalcPxtrStatScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture": 1
          }
        ) \
      .end_() \
      .if_("enable_explore_pic_real_pctr_user_action_feedback_weight_adjust == 1 and (uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_pic_interest_thresh") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "pic_stat_pic_play_cnt", "as": "pic_play_count"},
            {"name": "pic_stat_pic_eff_play_cnt", "as": "pic_eff_play_count"},
            {"name": "pic_stat_video_play_cnt", "as": "video_play_count"},
            {"name": "pic_stat_video_eff_play_cnt", "as": "video_eff_play_count"},
            {"name": "explore_user_action_feedback_pic_play_percent_bias", "as": "pic_play_percent_bias"},
            {"name": "explore_user_action_feedback_pic_video_eff_ratio_bias", "as": "pic_video_eff_ratio_bias"},
            {"name": "explore_user_action_feedback_pic_play_percent_min", "as": "pic_play_percent_min"},
            {"name": "explore_user_action_feedback_pic_play_percent_max", "as": "pic_play_percent_max"},
            {"name": "explore_user_action_feedback_pic_video_eff_ratio_min", "as": "pic_video_eff_ratio_min"},
            {"name": "explore_user_action_feedback_pic_video_eff_ratio_max", "as": "pic_video_eff_ratio_max"},
          ],
          export_common_attr = [
            {"name": "action_feedback_weight", "as": "explore_pic_power_weight_pic_real_pctr"},
          ],
          function_name = "CalcUserPicActionFeedbackCtrWeight",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_()\
      .if_("enable_explore_pic_real_pxtr == 1") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            {"name": "pic_ltr_weighted_ctr", "as": "pctr"},
            "pltr",
            "pwtr",
            "pftr",
            "pcmtr",
            "pdtr",
            "pcltr",
            "pevtr",
            "pic_ltr_acttr_db",
            "pic_ltr_fvtr"
          ],
          import_common_attr = [
            {"name": "explore_pic_real_pxtr_adjust_power_coeff", "as": "power_coeff"},
            {"name": "explore_pic_real_pxtr_adjust_max_pctr_boost_score", "as": "max_pctr_boost_score"},
            {"name": "explore_pic_real_pxtr_adjust_min_pctr_boost_score", "as": "min_pctr_boost_score"},
            {"name": "uDoubleOutsideValidPicClusterCnt7dKV", "as": "pic_interest_cluster_cnt"},
            {"name": "explore_pic_calc_real_pxtr_interest_thresh", "as": "pic_interest_thresh"},
            {"name": "explore_pic_real_pxtr_adjust_pctr_bias_coeff", "as": "pctr_bias_coeff"},
            {"name": "explore_pic_real_pxtr_adjust_pctr_alpha_coeff", "as": "pctr_alpha_coeff"},
            {"name": "enable_explore_pic_real_pxtr_pctr_power_adjust", "as": "enable_pctr_power_adjust"},
            {"name": "enable_explore_pic_real_pxtr_alpha_adjust", "as": "enable_pctr_alpha_adjust"},
            {"name": "explore_pic_real_pxtr_adjust_avg_emp_ctr", "as": "avg_emp_ctr"},
            "user_emp_ctr",
            "explore_zero_show_pic_user_first_screen"
          ],
          export_item_attr = [
            "pic_real_pltr",
            "pic_real_pwtr",
            "pic_real_pftr",
            "pic_real_pcmtr",
            "pic_real_pdtr",
            "pic_real_pcltr",
            "pic_real_pevtr",
            "pic_real_ltr_acttr_db",
            "pic_real_ltr_fvtr",
            "pic_real_corr_pctr"
          ],
          function_name = "CalcPicRealPxtr",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture" : 1
          }
        ) \
      .end_() \
      .if_("enable_explore_pic_vid2pic_boost_few_cluster_user == 1 and (uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_pic_vid2pic_boost_user_cluster_cnt_thresh") \
        .gen_common_attr_by_lua(
          attr_map = {
            "explore_pic_power_weight_fullrank_vid2pic_sim_score": "explore_pic_power_weight_fullrank_vid2pic_sim_score * explore_fr_vid2pic_boost_weight * (explore_pic_vid2pic_boost_user_cluster_cnt_thresh - (uDoubleOutsideValidPicClusterCnt7dKV or 0)) ^ explore_fr_vid2pic_boost_pow_weight",
          }
        ) \
      .end_() \
      .if_("enable_explore_pic_u2c_boost_on_low_interest_user == 1 and (uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_pic_u2c_boost_interest_thresh") \
        .copy_attr(
          attrs = [
            {"from_common": "explore_pic_power_weight_fullrank_pic_u2c_ensemble_score_low_interest_user", "to_common": "explore_pic_power_weight_fullrank_pic_u2c_ensemble_score"},
            {"from_common": "explore_pic_fullrank_pic_u2c_ensemble_score_raw_weight_low_interest_user", "to_common": "explore_pic_fullrank_pic_u2c_ensemble_score_raw_weight"},
            {"from_common": "explore_pic_fullrank_pic_u2c_ensemble_score_pow_raw_score_low_interest_user", "to_common": "explore_pic_fullrank_pic_u2c_ensemble_score_pow_raw_score"},
          ],
        ) \
      .end_() \
      .if_("enable_explore_pic_low_vv_ctr_weight_adjust == 1 and (pic_stat_pic_recent_play_cnt or 0) < explore_pic_rerank_tail_pic_vv_recent_thre") \
        .gen_common_attr_by_lua(
          attr_map = {
            "explore_pic_power_weight_pic_weighted_ctr": "explore_pic_power_weight_pic_weighted_ctr * explore_fr_pic_low_vv_weighted_ctr_boost_weight",
          }
        ) \
      .end_() \
      .if_("enable_explore_pic_pxtr_fusion_score == 1") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            {"name": "pic_ltr_weighted_ctr", "as": "pctr"},
            "pltr",
            "pwtr",
            "pftr",
            "pcmtr",
            "pdtr",
            "pcltr",
            "psvr",
            "pevtr"
          ],
          import_common_attr = [
            {"name": "explore_pic_pxtr_fusion_pltr_weight", "as": "pltr_weight"},
            {"name": "explore_pic_pxtr_fusion_pwtr_weight", "as": "pwtr_weight"},
            {"name": "explore_pic_pxtr_fusion_pftr_weight", "as": "pftr_weight"},
            {"name": "explore_pic_pxtr_fusion_pcmtr_weight", "as": "pcmtr_weight"},
            {"name": "explore_pic_pxtr_fusion_pdtr_weight", "as": "pdtr_weight"},
            {"name": "explore_pic_pxtr_fusion_pcltr_weight", "as": "pcltr_weight"},
            {"name": "explore_pic_pxtr_fusion_psvr_weight", "as": "psvr_weight"},
            {"name": "explore_pic_pxtr_fusion_pevtr_weight", "as": "pevtr_weight"},
            {"name": "explore_pic_pxtr_fusion_pxtr_fusion_pow_weight", "as": "pxtr_fusion_pow_weight"}
          ],
          export_item_attr = [
            "pic_pxtr_fusion_score"
          ],
          function_name = "CalcPicPxtrFusionScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture" : 1
          }
        ) \
      .end_() \
      .if_("enable_explore_pic_fr_boost_pctr_on_not_click_user == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "pic_recent_realshow_not_click_cnt",
            {"name": "explore_pic_fr_boost_pctr_on_not_click_user_lower_bound", "as": "not_click_count_lower_bound"},
            {"name": "explore_pic_fr_boost_pctr_on_not_click_user_upper_bound", "as": "not_click_count_upper_bound"},
            {"name": "explore_pic_fr_boost_pctr_on_not_click_user_alpha", "as": "boost_pctr_alpha"},
            {"name": "explore_pic_fr_boost_pctr_on_not_click_user_beta", "as": "boost_pctr_beta"},
            {"name": "explore_pic_fr_boost_pctr_on_not_click_user_power_weight", "as": "boost_pctr_power_weight"},
          ],
          export_common_attr = [
            {"name": "pctr_power_weight_boost_coeff", "as": "fr_pic_not_click_pctr_power_weight_boost_coeff"},
          ],
          function_name = "CalPicNotClickBoostCoeff",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .gen_common_attr_by_lua(
          attr_map = {
            "explore_pic_power_weight_fullrank_pctr_score": "explore_pic_power_weight_fullrank_pctr_score * fr_pic_not_click_pctr_power_weight_boost_coeff",
            "explore_pic_power_weight_pic_weighted_ctr": "explore_pic_power_weight_pic_weighted_ctr * fr_pic_not_click_pctr_power_weight_boost_coeff"
          }
        ) \
      .end_()

    self.flow.if_("enable_explore_pic_rfm_hetul1_debias_age_gender_debias == 1")
    self._enrich_rfm_hetul1_debias_pxtr_score()
    self.flow.end_()

  def _enrich_rfm_hetul1_debias_pxtr_score(self):
    # 使用到的 pxtr
    pxtr_names = ['pctr', 'pltr', 'pwtr', 'pftr', 'pcmtr', 'pcltr']
    prefix = "explore_pic_rfm_hetul1_debias_age_gender"

    # 1. 得到纠偏用的 pxtr_avg_list
    # 1.1 产生 kconf json key
    self.flow.gen_common_attr_by_lua(
      attr_map = {
        f"{prefix}_{pxtr_name}_key" : f"'{pxtr_name}' .. '.' .. tostring(basic_info_age_segment_v2) .. '_' .. tostring(user_gender)"
        for pxtr_name in pxtr_names
      }
    )
    # 1.2 从 kconf 取
    self.flow.get_kconf_params(
      kconf_configs = [{
        "kconf_key": "{{explore_pic_rfm_hetul1_debias_age_gender_kconf}}",
        "value_type": "list_double",
        "json_path": "{{" + f"{prefix}_{pxtr_name}_key" + "}}",
        "default_value": [],
        "export_common_attr": f"{prefix}_{pxtr_name}_hetu_avg_list"
      } for pxtr_name in pxtr_names]
    )
    # 2. 产生纠偏分
    # 2.1 产生 pxtr_names attr 供 light function 遍历
    self.flow.set_attr_value(
      common_attrs=[
        {
          "name": f"{prefix}_pxtr_names",
          "type": "string_list",
          "value": pxtr_names
        }
      ]
    )
    # 2.2 将所有 pxtr 纠偏用的参数、input_attr、output_attr，都 concat 成一个 list
    import_common_attrs = [{"name": f"{prefix}_pxtr_names", "as": "pxtr_names"}]
    import_item_attrs = [{"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_l1"}] + pxtr_names
    export_item_attrs = [{"name": f"debias_{pxtr_name}", "as": f"{prefix}_{pxtr_name}"} for pxtr_name in pxtr_names]
    for pxtr_name in pxtr_names:
      import_common_attrs.extend([
        {"name": f"{prefix}_{pxtr_name}_hetu_avg_list", "as": f"{pxtr_name}_hetu_avg_list"},
        {"name": f"{prefix}_{pxtr_name}_default_score", "as": f"{pxtr_name}_default_score"},
        {"name": f"{prefix}_{pxtr_name}_smooth", "as": f"{pxtr_name}_smooth"},
        {"name": f"{prefix}_{pxtr_name}_bias", "as": f"{pxtr_name}_bias"},
        {"name": f"{prefix}_{pxtr_name}_is_pos_sample", "as": f"{pxtr_name}_is_pos_sample"},
        {"name": f"{prefix}_{pxtr_name}_sample_rate", "as": f"{pxtr_name}_sample_rate"},
      ])

    # 2.3 产生纠偏分
    self.flow.enrich_attr_by_light_function(
      import_common_attr = import_common_attrs,
      import_item_attr = import_item_attrs,
      export_item_attr = export_item_attrs,
      function_name = "EnrichPicRfmHetul1DebiasPxtrScore",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {"is_picture": 1}
    )

  def _fr_es_adjust(self):
    return self.flow \
      .if_("enable_explore_pic_hack_act_fr_es_decay == 1", to_be_delete = "date=2024-05-29;committer=zhuwenyong") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_pic_hack_act_fr_decay_weight", "as": "decay_weight"},
            {"name": "explore_pic_hack_act_fr_decay_only_single_pic", "as": "only_single_pic"},
          ],
          import_item_attr = [
            "picture_type",
            "high_value_pic_flag",
            "audit_b_second_tag",
            "author__fans_count",
            {"name": "fr_pic_ensemble_score", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "fr_pic_ensemble_score"},
          ],
          target_item = {
            "is_picture": 1
          },
          function_name = "PicHackActEsDecay",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_fr_pic_recent_search_cluster_boost == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_fr_pic_recent_search_cluster_boost_coef", "as": "boost_discount_coeff"},
            {"name": "explore_fr_pic_recent_search_cluster_boost_thres", "as": "boost_discount_thres"},
            {"name": "explore_fr_pic_recent_search_cluster_boost_topk", "as": "topk"},
          ],
          import_item_attr = [
            {"name": "corr_pctr", "as": "need_item_attr"},
            {"name": "fr_pic_ensemble_score", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "fr_pic_ensemble_score"}
          ],
          function_name = "BoostOrDiscountWithThres",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture": 1,
            "is_pic_recent_search_cluster": 1
          }
        ) \
      .end_() \
      .if_("enable_explore_fr_pic_valid_interest_cluster_boost == 1") \
        .if_("enable_explore_fr_pic_valid_interest_cluster_boost_low_interest_user_enhance == 1") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "explore_fr_pic_valid_interest_cluster_boost_coef", "as": "base_coeff"},
              {"name": "explore_fr_pic_valid_interest_cluster_boost_low_interest_user_enhance_base_coef", "as": "enhance_base_coeff"},
              {"name": "explore_fr_pic_valid_interest_cluster_boost_low_interest_user_enhance_weight", "as": "enhance_weight"},
              {"name": "explore_fr_pic_valid_interest_cluster_boost_low_interest_thres", "as": "low_interest_thres"},
              {"name": "pic_double_outside_valid_interest_num", "as": "user_interest_num"},
            ],
            export_common_attr = [
              {"name": "enhance_coeff", "as": "explore_fr_pic_valid_interest_cluster_boost_low_interest_user_enhance_coef"}
            ],
            function_name = "CalcPicLowInterestUserBoost",
            class_name = "ExploreLightFunctionSetV2"
          ) \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "explore_fr_pic_valid_interest_cluster_boost_low_interest_user_enhance_coef", "as": "boost_discount_coeff"},
              {"name": "explore_fr_pic_valid_interest_cluster_boost_thres", "as": "boost_discount_thres"},
              {"name": "explore_fr_pic_valid_interest_cluster_boost_topk", "as": "topk"},
            ],
            import_item_attr = [
              {"name": "corr_pctr", "as": "need_item_attr"},
              {"name": "fr_pic_ensemble_score", "as": "ensemble_score"},
            ],
            export_item_attr = [
              {"name": "ensemble_score", "as": "fr_pic_ensemble_score"}
            ],
            function_name = "BoostOrDiscountWithThres",
            class_name = "ExploreLightFunctionSetV2",
            target_item = {
              "is_picture": 1,
              "is_pic_valid_interest_cluster": 1
            }
          ) \
        .else_() \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "explore_fr_pic_valid_interest_cluster_boost_coef", "as": "boost_discount_coeff"},
              {"name": "explore_fr_pic_valid_interest_cluster_boost_thres", "as": "boost_discount_thres"},
              {"name": "explore_fr_pic_valid_interest_cluster_boost_topk", "as": "topk"},
            ],
            import_item_attr = [
              {"name": "corr_pctr", "as": "need_item_attr"},
              {"name": "fr_pic_ensemble_score", "as": "ensemble_score"},
            ],
            export_item_attr = [
              {"name": "ensemble_score", "as": "fr_pic_ensemble_score"}
            ],
            function_name = "BoostOrDiscountWithThres",
            class_name = "ExploreLightFunctionSetV2",
            target_item = {
              "is_picture": 1,
              "is_pic_valid_interest_cluster": 1
            }
          ) \
        .end_() \
      .end_() \
      .if_("enable_explore_fr_pic_long_interest_cluster_boost == 1") \
        .if_("enable_explore_fr_pic_long_interest_cluster_boost_low_interest_user_enhance == 1") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "explore_fr_pic_long_interest_cluster_boost_coef", "as": "base_coeff"},
              {"name": "explore_fr_pic_long_interest_cluster_boost_low_interest_user_enhance_coef", "as": "enhance_base_coeff"},
              {"name": "explore_fr_pic_long_interest_cluster_boost_low_interest_user_enhance_weight", "as": "enhance_weight"},
              {"name": "explore_fr_pic_long_interest_cluster_boost_low_interest_thres", "as": "low_interest_thres"},
              {"name": "pic_double_outside_valid_interest_num", "as": "user_interest_num"},
            ],
            export_common_attr = [
              {"name": "enhance_coeff", "as": "explore_fr_pic_long_interest_cluster_boost_low_interest_user_enhance_coef"}
            ],
            function_name = "CalcPicLowInterestUserBoost",
            class_name = "ExploreLightFunctionSetV2"
          ) \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "explore_fr_pic_long_interest_cluster_boost_low_interest_user_enhance_coef", "as": "boost_discount_coeff"},
              {"name": "explore_fr_pic_long_interest_cluster_boost_thres", "as": "boost_discount_thres"},
              {"name": "explore_fr_pic_long_interest_cluster_boost_topk", "as": "topk"},
            ],
            import_item_attr = [
              {"name": "corr_pctr", "as": "need_item_attr"},
              {"name": "fr_pic_ensemble_score", "as": "ensemble_score"},
            ],
            export_item_attr = [
              {"name": "ensemble_score", "as": "fr_pic_ensemble_score"}
            ],
            function_name = "BoostOrDiscountWithThres",
            class_name = "ExploreLightFunctionSetV2",
            target_item = {
              "is_picture": 1,
              "is_pic_long_interest_cluster": 1
            }
          ) \
        .else_() \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "explore_fr_pic_long_interest_cluster_boost_coef", "as": "boost_discount_coeff"},
              {"name": "explore_fr_pic_long_interest_cluster_boost_thres", "as": "boost_discount_thres"},
              {"name": "explore_fr_pic_long_interest_cluster_boost_topk", "as": "topk"},
            ],
            import_item_attr = [
              {"name": "corr_pctr", "as": "need_item_attr"},
              {"name": "fr_pic_ensemble_score", "as": "ensemble_score"},
            ],
            export_item_attr = [
              {"name": "ensemble_score", "as": "fr_pic_ensemble_score"}
            ],
            function_name = "BoostOrDiscountWithThres",
            class_name = "ExploreLightFunctionSetV2",
            target_item = {
              "is_picture": 1,
              "is_pic_long_interest_cluster": 1
            }
          ) \
        .end_() \
      .end_() \
      .if_("explore_enable_user_pic_growth_cluster_boost == 1 and ((uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_user_pic_growth_cluster_boost_interest_thresh)") \
        .count_reco_result( # 统计 item 数量
          save_count_to = "rank_pic_growth_target_item_count",
          target_item = {
            "is_picture": 1,
            "is_pic_growth_cluster": 1,
          }
        ) \
        .gen_common_attr_by_lua( # 计算要 boost 多少个, 同时控制最大比例和个数
          attr_map={
            "rank_pic_growth_target_item_boost_num": f"""
              math.min(math.ceil(
                rank_pic_growth_target_item_count * explore_rank_pic_growth_cluster_boost_num_ratio),
                explore_rank_pic_growth_cluster_boost_num_max)
            """,
          }
        ) \
        .enrich_attr_by_light_function( # boost
          import_common_attr=[
            {"name": "explore_rank_pic_growth_cluster_boost_coef", "as": "boost_discount_coeff"},
            {"name": "rank_pic_growth_target_item_boost_num", "as": "topk"},
          ],
          import_item_attr=[{"name": "fr_pic_ensemble_score", "as": "score"},],
          export_item_attr=[{"name": "score", "as": "fr_pic_ensemble_score"},],
          function_name="BoostOrDiscountV2",
          class_name="ExploreLightFunctionSetV2",
          target_item={
            "is_picture": 1,
            "is_pic_growth_cluster": 1,
          }
        ) \
      .end_() \
      .if_("enable_explore_picture_type_fr_es_boost == 1") \
        .if_("enable_explore_fr_pic_type_age_boost == 1") \
          .calc_by_formula1(
            kconf_key = "formula.scenarioKey91.FrExploreAgePictureTypeBoost",
            import_common_attr = ["basic_info_age_segment_v2"],
            import_item_attr = [
              "picture_type",
              "fr_pic_ensemble_score"
            ],
            export_formula_value = [
              {"name": "final_score", "as": "fr_pic_ensemble_score"}
            ],
            abtest_biz_name = "KUAISHOU_APPS",
            target_item = {
              "is_picture": 1
            },
          ) \
        .else_() \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "enable_explore_single_pic_fr_es_boost_weight", "as": "single_boost_weight"},
              {"name": "enable_explore_long_pic_fr_es_boost_weight", "as": "long_boost_weight"},
              {"name": "enable_explore_set_pic_fr_es_boost_weight", "as": "set_boost_weight"},
            ],
            import_item_attr = [
              "picture_type",
              {"name": "fr_pic_ensemble_score", "as": "ensemble_score"},
            ],
            export_item_attr = [
              {"name": "ensemble_score", "as": "fr_pic_ensemble_score"},
            ],
            target_item = {
              "is_picture": 1
            },
            function_name = "PictureTypeEsBoost",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .end_() \
      .end_() \
      .if_("enable_explore_picture_follow_author_boost == 1 and page_index == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rank_pic_follow_author_boost_coef", "as": "boost_discount_coeff"},
            {"name": "explore_fr_pic_follow_author_boost_topk", "as": "topk"},
          ],
          import_item_attr = [
            {"name": "fr_pic_ensemble_score", "as": "score"},
          ],
          export_item_attr = [
            {"name": "score", "as": "fr_pic_ensemble_score"},
          ],
          function_name = "BoostOrDiscountV2",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture": 1,
            "is_picture_follow_author": 1,
          }
        ) \
      .end_() \
      .if_("enable_explore_fr_pic_valid_interest_cluster_boost_first_screen_adjust == 1 and page_index == 1") \
        .gen_common_attr_by_lua(
          attr_map={
            "explore_pic_double_valid_interest_cluster_boost_interest_thresh": "explore_pic_double_valid_interest_cluster_boost_interest_thresh_first_screen",
            "explore_fr_pic_double_valid_interest_cluster_boost_coef": "explore_fr_pic_double_valid_interest_cluster_boost_coef_first_screen",
            "explore_pic_single_valid_interest_cluster_boost_interest_thresh": "explore_pic_single_valid_interest_cluster_boost_interest_thresh_first_screen",
            "explore_fr_pic_single_valid_interest_cluster_boost_coef": "explore_fr_pic_single_valid_interest_cluster_boost_coef_first_screen"
          }
        ) \
      .end_() \
      .if_("enable_explore_fr_pic_double_valid_interest_cluster_boost == 1 and (uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_pic_double_valid_interest_cluster_boost_interest_thresh") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_fr_pic_double_valid_interest_cluster_boost_coef", "as": "boost_discount_coeff"},
            {"name": "explore_fr_pic_double_valid_interest_cluster_boost_thres", "as": "boost_discount_thres"},
            {"name": "explore_fr_pic_double_valid_interest_cluster_boost_topk", "as": "topk"},
          ],
          import_item_attr = [
            {"name": "corr_pctr", "as": "need_item_attr"},
            {"name": "fr_pic_ensemble_score", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "fr_pic_ensemble_score"}
          ],
          function_name = "BoostOrDiscountWithThres",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_pic_double_valid_interest_cluster": 1
          }
        ) \
      .end_() \
      .if_("enable_explore_fr_pic_single_valid_interest_cluster_boost == 1 and (uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_pic_single_valid_interest_cluster_boost_interest_thresh") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_fr_pic_single_valid_interest_cluster_boost_coef", "as": "boost_discount_coeff"},
            {"name": "explore_fr_pic_single_valid_interest_cluster_boost_thres", "as": "boost_discount_thres"},
            {"name": "explore_fr_pic_single_valid_interest_cluster_boost_topk", "as": "topk"},
          ],
          import_item_attr = [
            {"name": "corr_pctr", "as": "need_item_attr"},
            {"name": "fr_pic_ensemble_score", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "fr_pic_ensemble_score"}
          ],
          function_name = "BoostOrDiscountWithThres",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_pic_single_valid_interest_cluster": 1
          }
        ) \
      .end_() \
      .if_("enable_explore_fr_pic_recent_interest_cluster_boost == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_pic_recent_interest_cluster_id_list", "as": "key_list"},
            {"name": "explore_pic_recent_interest_cluster_score_list", "as": "value_list"},
            {"name": "explore_fr_pic_recent_interest_cluster_score_alpha", "as": "score_alpha"},
            {"name": "explore_fr_pic_recent_interest_cluster_score_beta", "as": "score_beta"},
            {"name": "explore_fr_pic_recent_interest_cluster_score_sigma", "as": "score_sigma"},
            {"name": "enable_explore_fr_pic_recent_interest_cluster_score_range_limit", "as": "enable_score_range_limit"},
            {"name": "explore_fr_pic_recent_interest_cluster_score_lower_bound", "as": "score_lower_bound"},
            {"name": "explore_fr_pic_recent_interest_cluster_score_upper_bound", "as": "score_upper_bound"},
            {"name": "enable_explore_fr_pic_recent_interest_cluster_boost_only_low_interest_user", "as": "enable_only_low_interest_user"},
            {"name": "enable_explore_fr_pic_recent_interest_cluster_boost_low_interest_user_enhance", "as": "enable_low_interest_user_enhance"},
            {"name": "explore_fr_pic_recent_interest_cluster_boost_low_interest_user_enhance_base_coef", "as": "enhance_base_coeff"},
            {"name": "explore_fr_pic_recent_interest_cluster_boost_low_interest_user_enhance_weight", "as": "enhance_weight"},
            {"name": "explore_fr_pic_recent_interest_cluster_boost_low_interest_thres", "as": "low_interest_thres"},
            {"name": "uDoubleOutsideValidPicCluster7dList", "as": "user_interest"},
            {"name": "enable_explore_fr_pic_recent_interest_cluster_boost_only_not_interest_cluster", "as": "enable_only_not_interest_cluster"},
          ],
          import_item_attr = [
            "cluster_id_632",
          ],
          export_item_attr = [
            {"name": "recent_interest_score", "as": "explore_fr_pic_recent_interest_score"}
          ],
          function_name = "CalcPicRecentInterestScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture": 1
          }
        ) \
        .enrich_attr_by_light_function(
          import_item_attr = [
            {"name": "explore_fr_pic_recent_interest_score", "as": "boost_discount_coeff"},
            {"name": "fr_pic_ensemble_score", "as": "score"},
          ],
          export_item_attr = [
            {"name": "score", "as": "fr_pic_ensemble_score"}
          ],
          function_name = "BoostOrDiscountByItemCoeff",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture": 1
          }
        ) \
      .end_() \
      .if_("enable_explore_fr_pic_hot_content_topk_boost == 1 and (user_pic_recent_show_cnt or 0) < explore_fr_pic_recent_low_show_boost_thresh and (pic_stat_pic_eff_play_cnt or 0) > explore_fr_pic_recent_low_show_boost_history_play_thresh") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_fr_pic_hot_content_boost_coef", "as": "boost_discount_coeff"},
            {"name": "explore_fr_pic_hot_content_boost_thres", "as": "boost_discount_thres"},
            {"name": "explore_fr_pic_hot_content_boost_topk", "as": "topk"},
          ],
          import_item_attr = [
            {"name": "corr_pctr", "as": "need_item_attr"},
            {"name": "fr_pic_ensemble_score", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "fr_pic_ensemble_score"}
          ],
          function_name = "BoostOrDiscountWithThres",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_pic_hot_content": 1,
          }
        ) \
      .end_() \
      .if_("enable_explore_fr_pic_consecutive_nonclick_exit == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_fr_pic_consecutive_nonclick_deboost_coeff", "as": "boost_discount_coeff"},
          ],
          import_item_attr = [
            {"name": "is_short_uninterested_photo", "as": "need_item_attr"},
            {"name": "fr_pic_ensemble_score", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "fr_pic_ensemble_score"}
          ],
          function_name = "BoostOrDiscount",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture": 1,
          }
        ) \
      .end_() \
      .if_("enable_explore_fr_operation_pic_boost == 1 and user_has_pic_crowd_show == 0") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_fr_operation_pic_boost_coef", "as": "boost_discount_coeff"},
            {"name": "explore_fr_operation_pic_boost_thres", "as": "boost_discount_thres"},
            {"name": "explore_fr_operation_pic_boost_topk", "as": "topk"},
          ],
          import_item_attr = [
            {"name": "corr_pctr", "as": "need_item_attr"},
            {"name": "fr_pic_ensemble_score", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "fr_pic_ensemble_score"}
          ],
          function_name = "BoostOrDiscountWithThres",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "operation_pic": 1,
          }
        ) \
      .end_()

  def _variant(self):
    return self.flow \
      .if_("explore_pic_rerank_div_sort_by_ensemble_score == 1") \
        .sort(
          score_from_attr = "fr_pic_ensemble_score",
          target_item = {
            "is_picture": 1
          }
        ) \
      .end_() \
      .if_("explore_pic_rerank_div_enable_realshow_decay == 1") \
        .enrich_attr_by_light_function(
          name = "PicFrGetScoreByRank1",
          import_common_attr = [
            {"name": "explore_pic_rerank_div_recip_smooth", "as": "recip_smooth"},
          ],
          export_item_attr = [
            {"name": "output_score", "as": "fr_pic_rerank_score_for_div"},
          ],
          function_name = "GetScoreByRank",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture": 1
          }
        ) \
        .switch_("enable_pic_rerank_realshow_decay_hetu_level") \
          .case_(2) \
            .fr_pic_real_show_not_click_decay(
              hetu_attr_name = "hetu_tag_level_info__hetu_level_two",
            ) \
          .case_(5) \
            .fr_pic_real_show_not_click_decay(
              hetu_attr_name = "hetu_tag_level_info__hetu_level_five",
            ) \
          .default_() \
            .fr_pic_real_show_not_click_decay(
              hetu_attr_name = "hetu_tag_level_info__hetu_level_one",
            ) \
        .end_() \
        .sort(
          score_from_attr = "fr_pic_rerank_score_for_div",
          target_item = {
            "is_picture": 1
          },
        ) \
      .end_() \
      .if_("explore_pic_rerank_div_enable_div_adjust == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "_USER_ID_", "as": "user_id"},
            {"name": "explore_pic_rerank_hetu_one_decay_window_size", "as": "hetu_l1_win_size"},
            {"name": "explore_pic_rerank_hetu_two_decay_window_size", "as": "hetu_l2_win_size"},
            {"name": "explore_pic_rerank_hetu_five_decay_window_size", "as": "hetu_l5_win_size"},
            {"name": "explore_pic_rerank_div_adj_hetu_l1_max_id", "as": "hetu_l1_max_id"},
            {"name": "explore_pic_rerank_div_adj_top_hetu_num", "as": "top_hetu_num"},
            {"name": "explore_pic_rerank_div_adj_win_size_min", "as": "win_size_min"},
            {"name": "explore_pic_rerank_div_adj_win_size_max", "as": "win_size_max"},
            {"name": "explore_pic_rerank_div_adj_mode", "as": "mode"},
            {"name": "explore_pic_rerank_div_adj_alpha", "as": "alpha"},
            {"name": "explore_pic_rerank_div_adj_beta", "as": "beta"},
            {"name": "explore_pic_rerank_div_adj_gamma", "as": "gamma"},
            {"name": "explore_pic_rerank_div_adj_min_page_index", "as": "min_page_index"},
            "page_index",
          ],
          import_item_attr = [
            "hetu_tag_level_info__hetu_level_one",
            "corr_pctr",
          ],
          export_common_attr = [
            {"name":"hetu_l1_win_size", "as":"explore_pic_rerank_hetu_one_decay_window_size"},
            {"name":"hetu_l2_win_size", "as":"explore_pic_rerank_hetu_two_decay_window_size"},
            {"name":"hetu_l5_win_size", "as":"explore_pic_rerank_hetu_five_decay_window_size"},
          ],
          function_name = "UpdatePicVariantParam",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture": 1
          }
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
          },
          "picture_type": {
            "enabled": "{{explore_pic_rerank_enable_picture_type_variant}}",
            "decay_window_size": "{{explore_pic_rerank_picture_type_variant_decay_window_size}}",
            "decay_occurrent_times": "{{explore_pic_rerank_picture_type_variant_decay_occurrent_times}}",
            "decay_rate": "{{explore_pic_rerank_picture_type_variant_decay_rate}}",
          },
        },
        target_item = {
          "is_picture" : 1
        }
      ) \
      .if_("enable_explore_pic_fr_diversity == 1") \
        .gen_common_attr_by_lua(
          attr_map = {
            "enable_explore_pic_not_long_interest_hetu_diversity": "enable_explore_pic_not_long_interest_hetu_diversity == 1 or (enable_explore_pic_not_long_interest_hetu_diversity_few_cluster_user == 1 and (uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_pic_not_long_interest_hetu_diversity_user_cluster_cnt_thresh)",
            "enable_explore_pic_hetu_sim_cluster_id_diversity": "enable_explore_pic_hetu_sim_cluster_id_diversity == 1 and (pic_stat_pic_play_cnt >= explore_pic_hetu_sim_cluster_id_diversity_play_cnt_threshold or pic_stat_pic_recent_play_cnt >= explore_pic_hetu_sim_cluster_id_diversity_recent_play_cnt_threshold)"
          }
        ) \
        .diversify_by_rules(
          max_satisfied_pick = "{{explore_pic_fr_diversity_max_satisfied_num}}",
          rules = [
            dict(attr_name = "is_pic_top_category",
                  enabled = "{{enable_explore_pic_top_category_diversity}}",
                  window_size = "{{explore_pic_top_category_diversity_winsize}}",
                  max_num = "{{explore_pic_top_category_diversity_max_num}}",
                  priority = "{{explore_pic_top_category_diversity_priority}}"),
            dict(attr_name = "is_not_pic_long_interest_hetu",
                  enabled = "{{enable_explore_pic_not_long_interest_hetu_diversity}}",
                  window_size = "{{explore_pic_not_long_interest_hetu_diversity_winsize}}",
                  max_num = "{{explore_pic_not_long_interest_hetu_diversity_max_num}}",
                  priority = "{{explore_pic_not_long_interest_hetu_diversity_priority}}"),
            dict(attr_name = "hetu_sim_cluster_id",
                 enabled = "{{enable_explore_pic_hetu_sim_cluster_id_diversity}}",
                 window_size="{{explore_pic_hetu_sim_cluster_id_diversity_winsize}}",
                 max_num = "{{explore_pic_hetu_sim_cluster_id_diversity_max_num}}",
                 priority = "{{explore_pic_hetu_sim_cluster_id_diversity_priority}}"),
          ],
          target_item = {
            "is_picture" : 1
          }
        ) \
      .end_()

  def _fr_es_overwrite(self):
    return self.flow \
      .if_("explore_pic_rerank_div_enable_overwrite_pic_fr_score == 1") \
        .enrich_attr_by_light_function(
          name = "PicFrGetScoreByRank2",
          import_common_attr = [
            {"name": "explore_pic_rerank_div_recip_smooth", "as": "recip_smooth"},
          ],
          export_item_attr = [
            {"name": "output_score", "as": "fr_pic_rerank_score_for_div"},
          ],
          function_name = "GetScoreByRank",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture": 1
          }
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_pic_rerank_div_decay_coeff", "as": "boost_discount_coeff"},
          ],
          import_item_attr = [
            {"name": "photo_id", "as": "need_item_attr"},
            {"name": "fr_pic_rerank_score_for_div", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "fr_pic_ensemble_score"}
          ],
          function_name = "BoostOrDiscount",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture": 1
          },
        ) \
      .end_()

  def _fr_insert(self):
    return self.flow \
      .if_("enable_explore_rank_vid2pic_insert == 1 and util.Random() < explore_rank_vid2pic_insert_ratio") \
        .split_string(
          input_common_attr = "explore_vid2pic_sim_target_hetu_str_v2",
          output_common_attr = "explore_vid2pic_sim_target_hetu_list_v2",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
        .enrich_attr_by_light_function(
          import_item_attr = [
            {"name": "corr_pctr", "as": "pxtr_score"},
            "vid2pic_sim_score",
            "hetu_tag_level_info_v2__hetu_level_one"
          ],
          import_common_attr = [
            {"name": "explore_rank_vid2pic_insert_position", "as": "force_insert_position"},
            {"name": "explore_rank_vid2pic_insert_pxtr_score_coeff", "as": "pxtr_score_coeff"},
            {"name": "explore_rank_vid2pic_insert_sim_score_thresh", "as": "sim_score_thresh"},
            {"name": "explore_vid2pic_sim_target_hetu_list_v2", "as": "target_hetu_list_v2"},
          ],
          export_item_attr = [
            {"name": "promote_to_position", "as": "explore_vid2pic_promote_to_position"},
          ],
          function_name = "CalForceInsertPositionForVid2Pic",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture": 1
          }
        ) \
        .force_insert(
          position_from_attr = "explore_vid2pic_promote_to_position",
          target_item = {
            "is_picture": 1
          }
        ) \
      .end_() \
      .if_("enable_explore_pic_search_insert == 1 or enable_explore_fr_es_after_overwrite_pic_search_boost == 1") \
        .if_("explore_pic_search_insert_only_recent_search == 1") \
          .if_("enable_explore_pic_search_candicate_expand_by_cluster == 1") \
            .copy_attr(
              attrs = [{
                "from_item": "is_pic_recent_search_cluster",
                "to_item": "is_pic_search_insert_candicate",
              }]
            ) \
          .else_() \
            .copy_attr(
              attrs = [{
                "from_item": "is_pic_recent_search",
                "to_item": "is_pic_search_insert_candicate",
              }]
            ) \
          .end_() \
        .else_() \
          .if_("enable_explore_pic_search_candicate_expand_by_cluster == 1") \
            .copy_attr(
              attrs = [{
                "from_item": "is_pic_search_cluster",
                "to_item": "is_pic_search_insert_candicate",
              }]
            ) \
          .else_() \
            .copy_attr(
              attrs = [{
                "from_item": "is_pic_search",
                "to_item": "is_pic_search_insert_candicate",
              }]
            ) \
          .end_() \
        .end_() \
      .end_() \
      .if_("enable_explore_pic_search_insert == 1 and util.Random() < explore_pic_search_insert_ratio_thd and (pic_search_boost_user_degree or 0) >= explore_pic_search_boost_user_degree_thresh") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            {"name": "corr_pctr", "as": "score"},
            {"name": "is_pic_search_insert_candicate", "as": "is_insert_item_attr"},
          ],
          import_common_attr = [
            {"name": "explore_pic_search_insert_position", "as": "force_insert_position"},
            {"name": "explore_pic_search_insert_coeff", "as": "coeff"},
          ],
          export_item_attr = [
            {"name": "promote_to_position", "as": "explore_pic_search_promote_to_position"},
          ],
          function_name = "CalForceInsertPosition",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture": 1
          }
        ) \
        .force_insert(
          position_from_attr = "explore_pic_search_promote_to_position",
          target_item = {
            "is_picture": 1
          }
        ) \
      .end_()

  def _fr_es_adjust_after_overwrite(self):
    return self.flow \
      .if_("enable_explore_rerank_pic_recent_search_cluster_boost == 1 and util.Random() < explore_rerank_pic_recent_search_cluster_boost_ratio") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rerank_pic_recent_search_cluster_boost_coef", "as": "boost_discount_coeff"},
            {"name": "explore_rerank_pic_recent_search_cluster_boost_thres", "as": "boost_discount_thres"},
            {"name": "explore_rerank_pic_recent_search_cluster_boost_topk", "as": "topk"},
          ],
          import_item_attr = [
            {"name": "corr_pctr", "as": "need_item_attr"},
            {"name": "fr_pic_ensemble_score", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "fr_pic_ensemble_score"}
          ],
          function_name = "BoostOrDiscountWithThres",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture": 1,
            "is_pic_recent_search_cluster": 1
          }
        ) \
      .end_() \
      .if_("enable_explore_rerank_pic_valid_interest_cluster_boost == 1 and util.Random() < explore_rerank_pic_valid_interest_cluster_boost_ratio") \
        .if_("enable_explore_rerank_pic_valid_interest_cluster_boost_low_interest_user_enhance == 1") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "explore_rerank_pic_valid_interest_cluster_boost_coef", "as": "base_coeff"},
              {"name": "explore_rerank_pic_valid_interest_cluster_boost_low_interest_user_enhance_base_coef", "as": "enhance_base_coeff"},
              {"name": "explore_rerank_pic_valid_interest_cluster_boost_low_interest_user_enhance_weight", "as": "enhance_weight"},
              {"name": "explore_rerank_pic_valid_interest_cluster_boost_low_interest_thres", "as": "low_interest_thres"},
              {"name": "pic_double_outside_valid_interest_num", "as": "user_interest_num"},
            ],
            export_common_attr = [
              {"name": "enhance_coeff", "as": "explore_rerank_pic_valid_interest_cluster_boost_low_interest_user_enhance_coef"}
            ],
            function_name = "CalcPicLowInterestUserBoost",
            class_name = "ExploreLightFunctionSetV2"
          ) \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "explore_rerank_pic_valid_interest_cluster_boost_low_interest_user_enhance_coef", "as": "boost_discount_coeff"},
              {"name": "explore_rerank_pic_valid_interest_cluster_boost_thres", "as": "boost_discount_thres"},
              {"name": "explore_rerank_pic_valid_interest_cluster_boost_topk", "as": "topk"},
            ],
            import_item_attr = [
              {"name": "corr_pctr", "as": "need_item_attr"},
              {"name": "fr_pic_ensemble_score", "as": "ensemble_score"},
            ],
            export_item_attr = [
              {"name": "ensemble_score", "as": "fr_pic_ensemble_score"}
            ],
            function_name = "BoostOrDiscountWithThres",
            class_name = "ExploreLightFunctionSetV2",
            target_item = {
              "is_picture": 1,
              "is_pic_valid_interest_cluster": 1
            }
          ) \
        .else_() \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "explore_rerank_pic_valid_interest_cluster_boost_coef", "as": "boost_discount_coeff"},
              {"name": "explore_rerank_pic_valid_interest_cluster_boost_thres", "as": "boost_discount_thres"},
              {"name": "explore_rerank_pic_valid_interest_cluster_boost_topk", "as": "topk"},
            ],
            import_item_attr = [
              {"name": "corr_pctr", "as": "need_item_attr"},
              {"name": "fr_pic_ensemble_score", "as": "ensemble_score"},
            ],
            export_item_attr = [
              {"name": "ensemble_score", "as": "fr_pic_ensemble_score"}
            ],
            function_name = "BoostOrDiscountWithThres",
            class_name = "ExploreLightFunctionSetV2",
            target_item = {
              "is_picture": 1,
              "is_pic_valid_interest_cluster": 1
            }
          ) \
        .end_() \
      .end_() \
      .if_("enable_explore_rerank_pic_long_interest_cluster_boost == 1 and util.Random() < explore_rerank_pic_long_interest_cluster_boost_ratio") \
        .if_("enable_explore_rerank_pic_long_interest_cluster_boost_low_interest_user_enhance == 1") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "explore_rerank_pic_long_interest_cluster_boost_coef", "as": "base_coeff"},
              {"name": "explore_rerank_pic_long_interest_cluster_boost_low_interest_user_enhance_base_coef", "as": "enhance_base_coeff"},
              {"name": "explore_rerank_pic_long_interest_cluster_boost_low_interest_user_enhance_weight", "as": "enhance_weight"},
              {"name": "explore_rerank_pic_long_interest_cluster_boost_low_interest_thres", "as": "low_interest_thres"},
              {"name": "pic_double_outside_valid_interest_num", "as": "user_interest_num"},
            ],
            export_common_attr = [
              {"name": "enhance_coeff", "as": "explore_rerank_pic_long_interest_cluster_boost_low_interest_user_enhance_coef"}
            ],
            function_name = "CalcPicLowInterestUserBoost",
            class_name = "ExploreLightFunctionSetV2"
          ) \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "explore_rerank_pic_long_interest_cluster_boost_low_interest_user_enhance_coef", "as": "boost_discount_coeff"},
              {"name": "explore_rerank_pic_long_interest_cluster_boost_thres", "as": "boost_discount_thres"},
              {"name": "explore_rerank_pic_long_interest_cluster_boost_topk", "as": "topk"},
            ],
            import_item_attr = [
              {"name": "corr_pctr", "as": "need_item_attr"},
              {"name": "fr_pic_ensemble_score", "as": "ensemble_score"},
            ],
            export_item_attr = [
              {"name": "ensemble_score", "as": "fr_pic_ensemble_score"}
            ],
            function_name = "BoostOrDiscountWithThres",
            class_name = "ExploreLightFunctionSetV2",
            target_item = {
              "is_picture": 1,
              "is_pic_long_interest_cluster": 1
            }
          ) \
        .else_() \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "explore_rerank_pic_long_interest_cluster_boost_coef", "as": "boost_discount_coeff"},
              {"name": "explore_rerank_pic_long_interest_cluster_boost_thres", "as": "boost_discount_thres"},
              {"name": "explore_rerank_pic_long_interest_cluster_boost_topk", "as": "topk"},
            ],
            import_item_attr = [
              {"name": "corr_pctr", "as": "need_item_attr"},
              {"name": "fr_pic_ensemble_score", "as": "ensemble_score"},
            ],
            export_item_attr = [
              {"name": "ensemble_score", "as": "fr_pic_ensemble_score"}
            ],
            function_name = "BoostOrDiscountWithThres",
            class_name = "ExploreLightFunctionSetV2",
            target_item = {
              "is_picture": 1,
              "is_pic_long_interest_cluster": 1
            }
          ) \
        .end_() \
      .end_() \
      .if_("enable_explore_rerank_pic_recent_interest_cluster_boost == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_pic_recent_interest_cluster_id_list", "as": "key_list"},
            {"name": "explore_pic_recent_interest_cluster_score_list", "as": "value_list"},
            {"name": "explore_rerank_pic_recent_interest_cluster_score_alpha", "as": "score_alpha"},
            {"name": "explore_rerank_pic_recent_interest_cluster_score_beta", "as": "score_beta"},
            {"name": "explore_rerank_pic_recent_interest_cluster_score_sigma", "as": "score_sigma"},
            {"name": "enable_explore_rerank_pic_recent_interest_cluster_score_range_limit", "as": "enable_score_range_limit"},
            {"name": "explore_rerank_pic_recent_interest_cluster_score_lower_bound", "as": "score_lower_bound"},
            {"name": "explore_rerank_pic_recent_interest_cluster_score_upper_bound", "as": "score_upper_bound"},
            {"name": "enable_explore_rerank_pic_recent_interest_cluster_boost_only_low_interest_user", "as": "enable_only_low_interest_user"},
            {"name": "enable_explore_rerank_pic_recent_interest_cluster_boost_low_interest_user_enhance", "as": "enable_low_interest_user_enhance"},
            {"name": "explore_rerank_pic_recent_interest_cluster_boost_low_interest_user_enhance_base_coef", "as": "enhance_base_coeff"},
            {"name": "explore_rerank_pic_recent_interest_cluster_boost_low_interest_user_enhance_weight", "as": "enhance_weight"},
            {"name": "explore_rerank_pic_recent_interest_cluster_boost_low_interest_thres", "as": "low_interest_thres"},
            {"name": "uDoubleOutsideValidPicCluster7dList", "as": "user_interest"},
            {"name": "enable_explore_rerank_pic_recent_interest_cluster_boost_only_not_interest_cluster", "as": "enable_only_not_interest_cluster"},
          ],
          import_item_attr = [
            "cluster_id_632",
          ],
          export_item_attr = [
            {"name": "recent_interest_score", "as": "explore_rerank_pic_recent_interest_score"}
          ],
          function_name = "CalcPicRecentInterestScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture": 1
          }
        ) \
        .enrich_attr_by_light_function(
          import_item_attr = [
            {"name": "explore_rerank_pic_recent_interest_score", "as": "boost_discount_coeff"},
            {"name": "fr_pic_ensemble_score", "as": "score"},
          ],
          export_item_attr = [
            {"name": "score", "as": "fr_pic_ensemble_score"}
          ],
          function_name = "BoostOrDiscountByItemCoeff",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_picture": 1
          }
        ) \
      .end_() \
      .if_("enable_explore_fr_es_after_overwrite_pic_search_boost == 1 and (pic_search_boost_user_degree or 0) >= explore_pic_search_boost_user_degree_thresh") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_fr_es_after_overwrite_pic_search_boost_coef", "as": "boost_discount_coeff"},
            {"name": "explore_fr_es_after_overwrite_pic_search_boost_thres", "as": "boost_discount_thres"},
            {"name": "explore_fr_es_after_overwrite_pic_search_boost_topk", "as": "topk"},
          ],
          import_item_attr = [
            {"name": "corr_pctr", "as": "need_item_attr"},
            {"name": "fr_pic_ensemble_score", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "fr_pic_ensemble_score"}
          ],
          function_name = "BoostOrDiscountWithThres",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_pic_search_insert_candicate": 1
          }
        ) \
      .end_() \
      .if_("enable_explore_rerank_operation_pic_boost == 1 and user_has_pic_crowd_show == 0") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rerank_operation_pic_boost_coef", "as": "boost_discount_coeff"},
            {"name": "explore_rerank_operation_pic_boost_thres", "as": "boost_discount_thres"},
            {"name": "explore_rerank_operation_pic_boost_topk", "as": "topk"},
          ],
          import_item_attr = [
            {"name": "corr_pctr", "as": "need_item_attr"},
            {"name": "fr_pic_ensemble_score", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "fr_pic_ensemble_score"}
          ],
          function_name = "BoostOrDiscountWithThres",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "operation_pic": 1
          }
        ) \
      .end_() \
      .if_("enable_explore_rerank_pic_type_boost == 1") \
        .if_("enable_explore_rerank_pic_type_age_boost == 1") \
          .calc_by_formula1(
            kconf_key = "formula.scenarioKey90.RerankExploreAgePictureTypeBoost",
            import_common_attr = ["basic_info_age_segment_v2"],
            import_item_attr = [
              "picture_type",
              "fr_pic_ensemble_score"
            ],
            export_formula_value = [
              {"name": "final_score", "as": "fr_pic_ensemble_score"}
            ],
            abtest_biz_name = "KUAISHOU_APPS",
            target_item = {
              "is_picture": 1
            },
          ) \
        .else_() \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "explore_rerank_single_pic_boost_coeff", "as": "single_boost_weight"},
              {"name": "explore_rerank_long_pic_boost_coeff", "as": "long_boost_weight"},
              {"name": "explore_rerank_pic_set_boost_coeff", "as": "set_boost_weight"},
            ],
            import_item_attr = [
              "picture_type",
              {"name": "fr_pic_ensemble_score", "as": "ensemble_score"},
            ],
            export_item_attr = [
              {"name": "ensemble_score", "as": "fr_pic_ensemble_score"},
            ],
            target_item = {
              "is_picture": 1
            },
            function_name = "PictureTypeEsBoost",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .end_() \
      .end_()

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_num_limit = 20,
        common_attrs = [
          "pic_bucket_calib_param_map"
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
          "pic_bucket_corr_pctr"
        ],
        for_debug_request_only = True
      )
