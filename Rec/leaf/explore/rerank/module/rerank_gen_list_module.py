from rerank import CommonModule
from rerank.module.rerank_features import *

def gen_seed_ensemble_queues_dpp():
    res = []
    queues = ["ctr",
            "wtr",
            "ltr",
            "fr_score1_corr", #这个要改名字
            "fr_score2_corr", #这个要改名字
            "l2r_score",
            "ftr",
            "duration_gt_58s_corr", #这个需要改名字
            "ptr",
            "lvtr",
            "epstr",
            "ensemble_score",
            "cltr",
            "fetr_corr", #这个要改名字
            "feff",
            "cmtr",
            "cmef",
            "diversity",
            "ada_score",
            "diversity_fr",
            "diversity_fr_ranking",
            "explore_diversity_ltr_score",
            "effective_follow_rate_score",
            "effective_follow_value_score",
            "cascading_watch_comment_score",
            "cascading_comment_like_score",
            "cascading_comment_time_score",
            "cascading_valid_play_score",
            "cascading_explore_gamora_interest_ptr",
            "cascading_explore_gamora_interest_ltr",
            "sense_view_predict_trans_score",
            "cover_view_predict_trans_score",
            "short_develop_interest_score",
            "effective_follow_ua_score",
            "effective_follow_ua_pfd_score",
            "revisit_score",
            "revisit_score_author",
            "interact_cost",
            "awesome_wtd_score",
            "dtr",
            "pdbfrtr",
            "interact_fusion",
            "watch_time_fusion",
            "frctr_fusion",
            "frcltr_fusion",
            "corr_fetr",
            "corr_fountain_eff",
            "rerank_cpr_corr",
            "rerank_pevtr_corr",
            "pltr",
            "pftr",
            "posftr",
            "pcmtr",
            "pcmef",
            "pptr",
            "pepstr",
            "fetr",
            "fountain_eff",
            "fr_score1",
            "fr_score2",
            "gen_l2r_score",
            "gen_l2r_score_corr",
            "cascade_prerank_pltr",
            "fr_pic_ensemble_score",
            "min_act_rank",
            "score_phtr",
            "svr_act_score",
            "pcmef_debias_score",
            "pctr_debias_hetu",
            "pltr_debias_hetu",
            "pwtr_debias_hetu",
            "pftr_debias_hetu",
            "pcmtr_debias_hetu",
            "pptr_debias_hetu",
            "mc_ensemble_pwatch_time",
            "fr_elive_ctcvr_gmv_score",
            "produce_mtctr",
            "produce_twhtr",
            "produce_mfctr",
            "produce_mtcotr",
            "produce_mtjtr",
            "produce_mtm1",
            "produce_upload_sum_score",
            "produce_consuv_sum_score",
            "psvr",
            "gen_l2r_fusion_score",
            "listwise_distill_score",
            "rank_distill_ctr",
            "rank_distill_ltr",
            "coordinated_watchtime_score",
            "duration_ms",
            "consume_time_ctr",
            "consume_time_pltr",
            "consume_time_lph",
            "consume_time_pstd",
            "diversity_fr_ranking",
            "corr_pctr_psvr",
            "fullrank_sess_reward_score",
            "ctr_evtr_map_val",
            "quantile_relative_score",
            "page1_trigger_score",
            "explore_photo_trend_score",
            "good_cover_similary_score",
            "good_sense_similary_score",
            "good_hot_audit_similary_score",
            "fr_slide_pctr_score",
            "fr_slide_awesome_wtd_score",
            "fr_sort_ratio_slide_pctr_score",
            "fr_sort_ratio_slide_pltr_score",
            "fr_sort_ratio_slide_pwtr_score",
            "fr_sort_ratio_slide_pcmtr_score",
            "fr_sort_ratio_slide_pcltr_score",
            "fr_sort_ratio_slide_pftr_score",
            "fr_sort_ratio_slide_awesome_wtd_score",
            "fr_sort_diversity_slide_pcmtr_score",
            "fr_sort_diversity_slide_pcltr_score",
            "fr_sort_diversity_slide_awesome_wtd_score",
            "min_watch_time_rank_score",
            "user_positive_action_photo_similary_score",
            "click_and_future_revisit_value",
            "future_revisit_value",
            "click_and_future_open_app_rate",
            "future_open_app_rate",
            "reverse_koc_cover_htr",
            "reverse_koc_detail_htr",
            "ctr_multy_wtd_sharpe_ratio_score",
            "svtr_rid_ctr_score",
            "emp_sharpe_score",
            "reverse_hate_photo_id_similary_score",
            "user_group_interest_tgi_score",
            "corr_pctr_adjust_by_pcoc",
            "explore_diversity_interest_lma_score",
            "explore_correction_pctr_score",
            "explore_diversity_interest_group_ctr_score",
            "debias_pf2r_score",
            "duration_based_5min_calibrated_evtr",
            "duration_based_5min_calibrated_lvtr",
            "esnn_model_score",
            "cascading_cover_htr",
            "cascading_detail_htr",
            "score_consume_time_ltr",
            "consume_time_pf2r_score",
            "ctr_emp_action",
            "explore_uninterest_ctr_adjust_score",
            "explore_rerank_gen_score",
            "user_career_interest_tagnex_tgi_score",
            "topk_act_rank_score", # 2026-01-07 by huzongyao
            "user_stage_interest_tagnex_tgi_score", # 2026-01-21 by wuyichun
            "consume_time_lvtr", # 2026-03-09 by wuyichun
            "consume_time_evtr", # 2026-03-09 by wuyichun
            "user_age_interest_tagnex_tgi_score", # 2026-03-30 by wuyichun
            "photo_history_interest_score_with_fr_ctr", # 2026-04-12 by guohao
            "user_age_interest_tagnex_tgi_product_fr_pxtr_score", # 2026-04-28 by wuyichun
            "teenager_ltr", # 2026-05-07 by huzongyao
            "teenager_ctr", # 2026-05-07 by huzongyao
            "teenager_wtd",  # 2026-05-07 by huzongyao
            "user_no_bias_interest_tagnex_tgi_score", # 2026-05-09 by juyi
            "user_no_bias_interest_tagnex_tgi_product_fr_pxtr_score", # 2026-05-09 by juyi
            "user_stage_interest_tagnex_tgi_product_fr_pxtr_score", # 2026-05-12 by wuyichun
            "is_hot_ranking_retr_score", # 2026-06-16 by juyi
            "is_prefer_author_ranking_retr_score", # 2026-06-16 by juyi
            ]
    prefix = "{{explore_rerank_gen_seed_ensemble_"
    for i, q in enumerate(queues) :
      t = {"name" : q }
      t.update({"weight_base" : prefix + q + '_weight}}'})
      t.update({"rank_pow_weight" : prefix + q + '_rank_pow_weight}}'})
      t.update({"bias_range" : prefix + q + '_range}}'})
      t.update({"weight_lower_bound" : prefix + q + '_lower_bound}}'})
      t.update({"raw_pow_weight" : prefix + q + '_raw_pow_weight}}'})
      t.update({"raw_weight" : prefix + q + '_raw_weight}}'})
      t.update({"raw_weight_multiply" : prefix + q + '_raw_weight_multiply}}'})
      t.update({"boost_topk_coef" : prefix + q + '_boost_topk_coef}}'})
      t.update({"boost_topk_threshold" : prefix + q + '_boost_topk_threshold}}'})
      t.update({"raw_pow_weight_multiply" : prefix + q + '_raw_pow_weight_multiply}}'})
      t.update({"score_norm" : prefix + q + '_score_norm}}'})
      t.update({"que_discount_coef" : prefix + q + '_que_discount_coef}}'})
      t.update({"weight_addAndMul" : prefix + q + '_weight_addAndMul}}'})
      t.update({"raw_pow_weight_addAndMul" : prefix + q + '_raw_pow_weight_addAndMul}}'})
      t.update({"raw_weight_addAndMul" : prefix + q + '_raw_weight_addAndMul}}'})
      res.append(t)

    return res

def rerank_pxtr_combo_queues():
  pxtr_matrix_combo_queues = [
    {
      "pxtr_attr": "pctr",
      "weight_attr": "explore_rerank_pxtr_sim_matrix_fullrank_pctr_weight",
    },
    {
      "pxtr_attr": "pltr",
      "weight_attr": "explore_rerank_pxtr_sim_matrix_fullrank_pltr_weight",
    },
    {
      "pxtr_attr": "pwtr",
      "weight_attr": "explore_rerank_pxtr_sim_matrix_fullrank_pwtr_weight",
    },
    {
      "pxtr_attr": "pcmtr",
      "weight_attr": "explore_rerank_pxtr_sim_matrix_fullrank_pcmtr_weight",
    },
    {
      "pxtr_attr": "pftr",
      "weight_attr": "explore_rerank_pxtr_sim_matrix_fullrank_pftr_weight",
    },
    {
      "pxtr_attr": "pcmef",
      "weight_attr": "explore_rerank_pxtr_sim_matrix_fullrank_pcmef_weight",
    },
    {
      "pxtr_attr": "pcltr",
      "weight_attr": "explore_rerank_pxtr_sim_matrix_fullrank_pcltr_weight",
    },
    {
      "pxtr_attr": "pevtr",
      "weight_attr": "explore_rerank_pxtr_sim_matrix_fullrank_pevtr_weight",
    },
    {
      "pxtr_attr": "fr_score1",
      "weight_attr": "explore_rerank_pxtr_sim_matrix_fullrank_fr_score1_weight",
    },
    {
      "pxtr_attr": "psvr",
      "weight_attr": "explore_rerank_pxtr_sim_matrix_fullrank_psvr_weight",
    },
    {
      "pxtr_attr": "awesome_wtd",
      "weight_attr": "explore_rerank_pxtr_sim_matrix_fullrank_awesome_wtd_weight",
    },
    {
      "pxtr_attr": "fr_score2",
      "weight_attr": "explore_rerank_pxtr_sim_matrix_fullrank_fr_score2_weight",
    },
    {
      "pxtr_attr": "fetr",
      "weight_attr": "explore_rerank_pxtr_sim_matrix_fullrank_fetr_weight",
    },
    {
      "pxtr_attr": "fountain_eff",
      "weight_attr": "explore_rerank_pxtr_sim_matrix_fullrank_fountain_eff_weight",
    },
  ]
  return pxtr_matrix_combo_queues

def fr_pxtrs():
    pxtrs = [
      "pctr",
      "pltr",
      "pwtr",
      "pftr",
      "pcmtr",
      "pptr",
      "pcmef",
      "pevtr",
      "fr_score1",
      "fr_score2",
      "pepstr",
      "pdtr",
      "pcltr",
    ]
    return pxtrs

def fr_fountain_pxtrs():
    pxtrs = [
      "fetr",
      "fountain_eff",
      "consume_time_ltr",
    ]
    return pxtrs

def generate_mix_queues(pxtrs):
  queues = []
  for pxtr in pxtrs:
    queue = {}
    queue["name"] = pxtr
    queue["weight"] = 0.0
    queue["power_weight_attr"] = "dpp_mix_rerank_weight_" + pxtr
    queues.append(queue)
  return queues

def dpp_variant_rules():
  rules = [
    dict(attr_name = "is_minority_photo",
          enabled = "{{enable_minority_photo_diversity}}",
          window_size = "{{minority_photo_diversity_winsize}}",
          max_num = "{{minority_photo_diversity_max_num}}",
          min_num = "{{minority_photo_diversity_min_num}}",
          priority = "{{minority_photo_diversity_priority}}"),
    dict(attr_name = "is_protogenetic_advertise_photo",
          enabled = "{{enable_protogenetic_advertise_photo_diversity}}",
          window_size = "{{protogenetic_advertise_photo_diversity_winsize}}",
          min_num = "{{protogenetic_advertise_photo_diversity_min_num}}",
          max_num = "{{protogenetic_advertise_photo_diversity_max_num}}",
          priority = "{{protogenetic_advertise_photo_diversity_diversity_priority}}",
          consider_prev_items = "{{enable_protogenetic_advertise_photo_diversity_consider_prev_items}}"),
    dict(attr_name= "hetu_level_one_v2",
          enabled="{{enable_rerank_variety_hetu_level_one}}",
          window_size= "{{rerank_variety_hetu_level_one_winsize}}",
          max_num="{{rerank_variety_hetu_one_max_num}}",
          priority="{{rerank_variety_hetu_level_one_priority}}",
          consider_prev_items = "{{enable_rerank_variety_hetu_level_one_consider_prev_items}}"),
    dict(attr_name= "hetu_level_two_v2",
          enabled="{{rerank_variety_shuanglie_enable13}}",
          window_size="{{rerank_variety_shuanglie_winsize13}}",
          max_num="{{rerank_variety_shuanglie_max13}}",
          priority="{{rerank_variety_shuanglie_priority13}}",
          consider_prev_items = "{{rerank_variety_shuanglie_enable13_consider_prev_items}}"),
    dict(attr_name= "is_grpr_pron_photo",
          enabled="{{rerank_variety_shuanglie_enable12}}",
          window_size= "{{rerank_variety_shuanglie_winsize12}}",
          max_num="{{rerank_variety_shuanglie_max12}}",
          priority="{{rerank_variety_shuanglie_priority12}}"),
    dict(attr_name= "author__id",
          enabled="{{rerank_variety_shuanglie_enable11}}",
          window_size= "{{rerank_variety_shuanglie_winsize11}}",
          max_num="{{rerank_variety_shuanglie_max11}}",
          priority="{{rerank_variety_shuanglie_priority11}}",
          consider_prev_items = "{{rerank_variety_shuanglie_enable11_consider_prev_items}}"),
    dict(attr_name= "video_variant_attr",
          enabled="{{rerank_variety_shuanglie_enable10}}",
          window_size="{{rerank_variety_shuanglie_winsize10}}",
          max_num="{{rerank_variety_shuanglie_max10}}",
          priority="{{rerank_variety_shuanglie_priority10}}"),
    dict(attr_name= "duration_0_7s",
          enabled="{{rerank_variety_shuanglie_enable9}}",
          window_size="{{rerank_variety_shuanglie_winsize9}}",
          max_num="{{rerank_variety_shuanglie_max9}}",
          priority="{{rerank_variety_shuanglie_priority9}}"),
    dict(attr_name= "hetu_tag_level_info__hetu_level_one",
          enabled="{{rerank_variety_shuanglie_enable8}}",
          window_size="{{rerank_variety_shuanglie_winsize8}}",
          window_type="{{rerank_variety_shuanglie_wintype8}}",
          max_num="{{rerank_variety_shuanglie_max8}}",
          priority="{{rerank_variety_shuanglie_priority8}}",
          consider_prev_items = "{{rerank_variety_shuanglie_enable8_consider_prev_items}}"),
    dict(attr_name= "is_follow_author",
          enabled="{{rerank_variety_shuanglie_enable7}}",
          window_size="{{rerank_variety_shuanglie_winsize7}}",
          max_num="{{rerank_variety_shuanglie_max7}}",
          priority="{{rerank_variety_shuanglie_priority7}}",
          consider_prev_items = "{{rerank_variety_shuanglie_enable7_consider_prev_items}}"),
    dict(attr_name= "specified_hetu5_found",
          enabled="{{rerank_variety_shuanglie_enable6}}",
          window_size= "{{rerank_variety_shuanglie_winsize6}}",
          max_num="{{rerank_variety_shuanglie_max6}}",
          priority="{{rerank_variety_shuanglie_priority6}}"),
    dict(attr_name= "gr_policy_softcore",
          enabled="{{rerank_variety_shuanglie_enable5}}",
          window_size= "{{rerank_variety_shuanglie_winsize5}}",
          max_num="{{rerank_variety_shuanglie_max5}}",
          priority="{{rerank_variety_shuanglie_priority5}}"),
    dict(attr_name= "shuffle_policy_changed",
          enabled="{{rerank_variety_shuanglie_enable4}}",
          window_size= "{{rerank_variety_shuanglie_winsize4}}",
          max_num="{{rerank_variety_shuanglie_max4}}",
          priority="{{rerank_variety_shuanglie_priority4}}"),
    dict(attr_name= "hetu_tag_level_info__hetu_face_id",
          enabled="{{rerank_variety_shuanglie_enable3}}",
          window_size= "{{rerank_variety_shuanglie_winsize3}}",
          max_num="{{rerank_variety_shuanglie_max3}}",
          priority="{{rerank_variety_shuanglie_priority3}}"),
    dict(attr_name= "hetu_tag_level_info__hetu_level_five",
          enabled="{{rerank_variety_shuanglie_enable2}}",
          window_size= "{{rerank_variety_shuanglie_winsize2}}",
          max_num="{{rerank_variety_shuanglie_max2}}",
          priority="{{rerank_variety_shuanglie_priority2}}"),
    dict(attr_name="cluster_id_632",
          enabled="{{enable_rerank_hetu_cluster_diversity}}",
          window_size="{{rerank_hetu_cluster_refactoring_winsize}}",
          max_num="{{rerank_hetu_cluster_refactoring_max}}",
          min_num="{{rerank_hetu_cluster_refactoring_min}}",
          priority="{{rerank_hetu_cluster_refactoring_priority}}",
          consider_prev_items="{{enable_rerank_hetu_cluster_consider_prev}}"),
    dict(attr_name="is_new_interest_explore",
          enabled="{{enable_rerank_new_interest_explore_diversity}}",
          window_size="{{rerank_new_interest_explore_refactoring_winsize}}",
          max_num="{{rerank_new_interest_explore_refactoring_max}}",
          min_num="{{rerank_new_interest_explore_refactoring_min}}",
          priority="{{rerank_new_interest_explore_refactoring_priority}}",
          consider_prev_items="{{enable_rerank_new_interest_explore_consider_prev}}"),
    dict(attr_name="appearance_hetu_level_one",
          enabled="{{enable_rerank_appearance_hetu_level_one_diversity}}",
          window_size="{{rerank_appearance_hetu_level_one_winsize}}",
          max_num="{{rerank_appearance_hetu_level_one_max}}",
          min_num="{{rerank_appearance_hetu_level_one_min}}",
          priority="{{rerank_appearance_hetu_level_one_priority}}"),
    dict(attr_name="photo_source_type",
          enabled="{{enable_rerank_photo_source_type_diversity}}",
          window_size="{{rerank_photo_source_type_winsize}}",
          max_num="{{rerank_photo_source_type_max}}",
          min_num="{{rerank_photo_source_type_min}}",
          priority="{{rerank_photo_source_type_priority}}"),
    dict(attr_name="is_high_value_author",
          enabled="{{enable_rerank_high_value_author_explore_diversity_sample}}",
          window_size="{{rerank_high_value_author_explore_refactoring_winsize}}",
          max_num="{{rerank_high_value_authort_explore_refactoring_max}}",
          min_num="{{rerank_high_value_author_explore_refactoring_min}}",
          priority="{{rerank_high_value_author_explore_refactoring_priority}}",
          consider_prev_items="{{enable_rerank_high_value_author_explore_consider_prev_items}}"),
    dict(attr_name="is_hetu_beauty_cluster_id",
          enabled="{{enable_rerank_hetu_beauty_cluster_id_list_diversity}}",
          window_size="{{rerank_hetu_beauty_cluster_id_list_explore_refactoring_winsize}}",
          max_num="{{rerank_hetu_beauty_cluster_id_list_explore_refactoring_max}}",
          min_num="{{rerank_hetu_beauty_cluster_id_list_explore_refactoring_min}}",
          priority="{{rerank_hetu_beauty_cluster_id_list_explore_refactoring_priority}}",
          consider_prev_items="{{enable_rerank_hetu_beauty_cluster_id_list_explore_consider_prev_items}}"),
    dict(attr_name="is_recommend_by_friend",
          enabled="{{enable_rerank_friend_recommendation_explore_diversity_sample}}",
          window_size="{{rerank_friend_recommendation_explore_refactoring_winsize}}",
          max_num="{{rerank_friend_recommendation_explore_refactoring_max}}",
          min_num="{{rerank_friend_recommendation_explore_refactoring_min}}",
          priority="{{rerank_friend_recommendation_explore_refactoring_priority}}"),
    dict(attr_name= "is_wide_screen_photo",
          enabled="{{enable_wide_screen_photo_diversity}}",
          window_size= "{{wide_screen_photo_diversity_winsize}}",
          max_num="{{wide_screen_photo_diversity_max_num}}",
          min_num="{{wide_screen_photo_diversity_min_num}}",
          priority="{{wide_screen_photo_diversity_priority}}"),
    dict(attr_name = "continuous_hitting_hetu_level_five",
          enabled = "{{enable_rerank_is_continuous_hitting}}",
          window_size = "{{rerank_is_continuous_hitting_winsize}}",
          max_num = "{{rerank_is_continuous_hitting_max_num}}",
          priority = "{{rerank_is_continuous_hitting_priority}}"),
    dict(attr_name = "i2i_retr__trigger_pid",
          enabled = "{{enable_rerank_trigger_pid_diversity}}",
          window_size = "{{rerank_trigger_pid_diversity_winsize}}",
          max_num = "{{rerank_trigger_pid_diversity_max_num}}",
          priority = "{{rerank_trigger_pid_diversity_priority}}"),
    dict(attr_name = "is_merchant_hetu_tag_id",
          enabled = "{{enable_rerank_merchant_hetu_tag_id_diversity}}",
          window_size = "{{rerank_merchant_hetu_tag_id_diversity_winsize}}",
          max_num = "{{rerank_merchant_hetu_tag_id_diversity_max_num}}",
          priority = "{{rerank_merchant_hetu_tag_id_diversity_priority}}",
          consider_prev_items="{{enable_rerank_merchant_hetu_tag_id_explore_consider_prev_items}}"),
    dict(attr_name = "is_merchant_impress_id",
          enabled = "{{enable_rerank_merchant_impress_id_diversity}}",
          window_size = "{{rerank_merchant_impress_id_diversity_winsize}}",
          max_num = "{{rerank_merchant_impress_id_diversity_max_num}}",
          priority = "{{rerank_merchant_impress_id_diversity_priority}}",
          consider_prev_items="{{enable_rerank_merchant_impress_id_explore_consider_prev_items}}"),
    dict(attr_name = "is_marketing_compensation_photo",
          enabled = "{{enable_marketing_compensation_photo_diversity}}",
          window_size = "{{marketing_compensation_photo_diversity_winsize}}",
          max_num = "{{marketing_compensation_photo_diversity_max_num}}",
          min_num = "{{marketing_compensation_photo_diversity_min_num}}",
          priority = "{{marketing_compensation_photo_diversity_priority}}"),
    dict(attr_name = "is_olympic_latest", # 奥运新内容保量
          enabled = "{{enable_olympic_latest_diversity}}",
          window_size = "{{olympic_latest_diversity_winsize}}",
          min_num = "{{olympic_latest_diversity_min_num}}",
          max_num = "{{olympic_latest_diversity_max_num}}",
          priority = "{{olympic_latest_diversity_priority}}"),
    dict(attr_name = "is_olympic", # 奥运内容打散
          enabled = "{{enable_olympic_diversity}}",
          window_size = "{{olympic_diversity_winsize}}",
          min_num = "{{olympic_diversity_min_num}}",
          max_num = "{{olympic_diversity_max_num}}",
          priority = "{{olympic_diversity_priority}}"),
    dict(attr_name = "is_hot_list_flag",
          enabled = "{{enable_hot_list_diversity}}",
          window_size = "{{hot_list_diversity_winsize}}",
          min_num = "{{hot_list_diversity_min_num}}",
          max_num = "{{hot_list_diversity_max_num}}",
          priority = "{{hot_list_diversity_priority}}"),
    dict(attr_name="is_quality_singal_hot_list_topk",
          enabled="{{enable_rerank_hot_list_photo_source_type_boost}}",
          window_size="{{rerank_hot_list_photo_source_type_winsize}}",
          max_num="{{rerank_hot_list_photo_source_type_max}}",
          min_num="{{rerank_hot_list_photo_source_type_min}}",
          priority="{{rerank_hot_list_photo_source_type_priority}}"),
    dict(attr_name="is_quality_singal_prior_author_topk",
          enabled="{{enable_rerank_prior_author_photo_source_type_boost}}",
          window_size="{{rerank_prior_author_photo_source_type_winsize}}",
          max_num="{{rerank_prior_author_photo_source_type_max}}",
          min_num="{{rerank_prior_author_photo_source_type_min}}",
          priority="{{rerank_prior_author_photo_source_type_priority}}"),
    dict(attr_name="is_quality_singal_life_prior_topk",
          enabled="{{enable_rerank_life_prior_photo_source_type_boost}}",
          window_size="{{rerank_life_prior_photo_source_type_winsize}}",
          max_num="{{rerank_life_prior_photo_source_type_max}}",
          min_num="{{rerank_life_prior_photo_source_type_min}}",
          priority="{{rerank_life_prior_photo_source_type_priority}}"),
    dict(attr_name="is_quality_original_author_list_topk",
          enabled="{{enable_rerank_original_author_photo_source_type_boost}}",
          window_size="{{rerank_original_author_photo_source_type_winsize}}",
          max_num="{{rerank_original_author_photo_source_type_max}}",
          min_num="{{rerank_original_author_photo_source_type_min}}",
          priority="{{rerank_original_author_photo_source_type_priority}}"),
    dict(attr_name = "is_pid_for_similar_author",
          enabled = "{{enable_rerank_pid_for_similar_author_diversity}}",
          window_size = "{{rerank_similar_author_reason_retr_diversity_winsize}}",
          min_num = "{{rerank_similar_author_reason_retr_diversity_min_num}}",
          max_num = "{{rerank_similar_author_reason_retr_diversity_max_num}}",
          priority = "{{rerank_similar_author_reason_retr_diversity_priority}}"),
    dict(attr_name = "is_pid_for_long_worth_author",
          enabled = "{{enable_long_worth_author_diversity}}",
          window_size = "{{rerank_long_worth_author_reason_retr_diversity_winsize}}",
          min_num = "{{rerank_long_worth_author_reason_retr_diversity_min_num}}",
          max_num = "{{rerank_long_worth_author_reason_retr_diversity_max_num}}",
          priority = "{{rerank_long_worth_author_reason_retr_diversity_priority}}"),
     dict(attr_name = "is_unbias_interest_pid_for_crows",
          enabled = "{{enable_unbias_interest_diversity}}",
          window_size = "{{rerank_unbias_interest_reason_retr_diversity_winsize}}",
          min_num = "{{rerank_unbias_interest_reason_retr_diversity_min_num}}",
          max_num = "{{rerank_unbias_interest_reason_retr_diversity_max_num}}",
          priority = "{{rerank_unbias_interest_reason_retr_diversity_priority}}"),
    dict(attr_name = "is_meinv_photo",
          enabled = "{{enable_rerank_is_meinv_photo_diversity}}",
          window_size = "{{rerank_is_meinv_photo_diversity_winsize}}",
          min_num = "{{rerank_is_meinv_photo_diversity_min_num}}",
          max_num = "{{rerank_is_meinv_photo_diversity_max_num}}",
          priority = "{{rerank_is_meinv_photo_diversity_priority}}"),
    dict(attr_name="is_first_refresh_good_photo",
          enabled="{{enable_rerank_first_refresh_good_photo_diversity}}",
          window_size= "{{rerank_first_refresh_good_diversity_winsize}}",
          max_num="{{rerank_first_refresh_good_diversity_max_num}}",
          min_num = "{{rerank_first_refresh_good_diversity_min_num}}",
          priority="{{rerank_first_refresh_good_diversity_priority}}"),
    dict(attr_name="is_senseview_lowcost_photo",
          enabled="{{enable_rerank_is_senseview_lowcost_photo_diversity}}",
          window_size= "{{rerank_senseview_lowcost_photo_diversity_winsize}}",
          max_num="{{rerank_senseview_lowcost_photo_diversity_max_num}}",
          min_num = "{{rerank_senseview_lowcost_photo_diversity_min_num}}",
          priority="{{rerank_senseview_lowcost_photo_diversity_priority}}",
          consider_prev_items="{{enable_rerank_senseview_lowcost_photo_explore_consider_prev_items}}"),
    dict(attr_name = "is_sexually_photo",
          enabled = "{{enable_rerank_sexually_photo_diversity}}",
          window_size = "{{sexually_photo_diversity_winsize}}",
          max_num = "{{sexually_photo_diversity_max_num}}",
          min_num = "{{sexually_photo_diversity_min_num}}",
          priority = "{{sexually_photo_diversity_priority}}"),
    dict(attr_name = "reach_content",
          enabled = "{{enable_explore_rerank_reach_content_diversity}}",
          window_size = "{{explore_rerank_reach_content_diversity_winsize}}",
          min_num = "{{explore_rerank_reach_content_diversity_min_num}}",
          max_num = "{{explore_rerank_reach_content_diversity_max_num}}",
          priority = "{{explore_rerank_reach_content_diversity_priority}}"),
  ]
  return rules

def single_queues():
  queues = [
    {
      "name": "fullrank_neg_feedback_discount_score",
      "enabled": "{{dpp_generator_single_queue_fullrank_neg_feedback_discount_score}}"
    },
    {
      "name": "fullrank_l2r_score",
      "enabled": "{{dpp_generator_single_queue_fullrank_l2r_score}}"
    },
    {
      "name": "ensemble_score",
      "enabled": "{{dpp_generator_single_queue_ensemble_score}}"
    },
    {
      "name": "mix_ensemble_score",
      "enabled": "{{dpp_generator_single_queue_mix_ensemble_score}}"
    },
  ]
  return queues

def rerank_hetu_ensemble_queues():
  queues = [
    {
      "pxtr_attr": "ctr",
      "avg_attr": "avg_click_dev",
      "weight_attr": "explore_rerank_hetu_ensemble_fullrank_click_score",
    },
    {
      "pxtr_attr": "ltr",
      "avg_attr": "avg_like_dev",
      "weight_attr": "explore_rerank_hetu_ensemble_fullrank_like_score",
    },
    {
      "pxtr_attr": "wtr",
      "avg_attr": "avg_follow_dev",
      "weight_attr": "explore_rerank_hetu_ensemble_fullrank_follow_score",
    },
    {
      "pxtr_attr": "ftr",
      "avg_attr": "avg_forward_dev",
      "weight_attr": "explore_rerank_hetu_ensemble_fullrank_forward_score",
    },
    {
      "pxtr_attr": "cmtr",
      "avg_attr": "avg_comment_dev",
      "weight_attr": "explore_rerank_hetu_ensemble_fullrank_comment_score",
    },
    {
      "pxtr_attr": "cltr",
      "avg_attr": "avg_collect_dev",
      "weight_attr": "explore_rerank_hetu_ensemble_fullrank_collect_score",
    },
    {
      "pxtr_attr": "awesome_wtd_score",
      "avg_attr": "avg_awesome_wtd_score_dev",
      "weight_attr": "explore_rerank_hetu_ensemble_fullrank_awesome_wtd_score",
    }

  ]
  return queues

rerank_weight_param_dict = {
  "ctr": "explore_rerank_gen_seed_ensemble_ctr_weight",
  "watchtime": "explore_rerank_gen_seed_ensemble_awesome_wtd_score_weight",
  "ltr": "explore_rerank_gen_seed_ensemble_ltr_weight",
  "cmtr": "explore_rerank_gen_seed_ensemble_cmtr_weight",
  "wtr": "explore_rerank_gen_seed_ensemble_wtr_weight",
  "cltr": "explore_rerank_gen_seed_ensemble_cltr_weight",
  "ftr": "explore_rerank_gen_seed_ensemble_ftr_weight",
}

user_group_input_common_attrs = [
  {"name": "explore_all_user_consume_str", "as": "all_user_consume_stat_str"},
  {"name": "explore_user_group_consume_str", "as": "user_consume_stat_str"}
]
user_group_output_common_attrs = []
ratio_prefix = "explore_user_group_consume_weight_adjust_ratio_rerank_"
for xtr in rerank_weight_param_dict.keys():
  user_group_input_common_attrs.append({"name": rerank_weight_param_dict[xtr], "as": xtr + "_weight"})
  user_group_input_common_attrs.append({"name": ratio_prefix + xtr, "as": xtr + "_adjust_ratio"})
  user_group_output_common_attrs.append({"name": xtr + "_weight", "as": rerank_weight_param_dict[xtr]})

class RerankGenList(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def enrich_item_attr(self, target_item = {}):
    return self.flow \
        .copy_item_meta_info(
          save_item_seq_to_attr = "rerank_list_enter_index",
          target_item = target_item,
        ) \
        .enrich_attr_by_lua(
          import_item_attr = [
            "hetu_tag_level_info__hetu_level_one",
            "hetu_tag_level_info__hetu_level_two",
            "explore_stat__view_length_sum",
            "explore_stat__click_count",
            "duration_ms",
            "explore_fr_ensemble_score"
          ],
          export_item_attr = [
            "hetu_level_one_attr",
            "hetu_level_two_attr",
            "hetu_level_two_attr2",
            "hetu_level_two_attr3",
            "empirical_watchtime",
            "duration_0_7s",
            "duration_7_9s",
            "duration_9_12s",
            "duration_12_17s",
            "duration_17_20s",
            "duration_20_58s",
            "duration_gt_58s",
            "duration_gt_120s",
            "virtual_rerank_score"
          ],
          function_for_item = "convert_photo_info_attr",
          lua_script_file = "explore/rerank/lua/module/rerank_gen_list__multi_lua.lua",
          range_end = "{{dpp_diversity_candidate_size}}",
          target_item = target_item,
      ) \
      .enrich_attr_by_lua(
          import_item_attr = [
              "duration_ms",
              "photo_dnn_cluster_id",
              "view_length_sum",
              "explore_stat__real_show_count",
              "explore_stat__click_count",
              "explore_stat__like_count",
              "explore_stat__follow_count",
              "explore_stat__forward_count",
              "explore_stat__profile_enter_count",
              "explore_stat__comment_count",
              "explore_stat__negative_count",
              "explore_stat__report_detail__total_report_count",
              "upload_time",
              "is_picture"
          ],
          export_item_attr = [
              "dnn_cluster_variant_attr",
              "hetu_cluster_attr",
              "short_duration_variant_attr",
              "long_duration_variant_attr",
              "lt20s_duration_variant_attr",
              "empirical_ctr",
              "empirical_ltr",
              "empirical_wtr",
              "empirical_ftr",
              "empirical_ptr",
              "empirical_cmtr",
              "empirical_htr",
              "empirical_watchtime",
              "empirical_rrr",
              "photo_age_hour",
              "avg_watch_time_ms"
          ],
          function_for_item = "calculate",
          lua_script_file = "explore/rerank/lua/module/rerank_gen_list__multi_lua.lua",
          range_end = "{{dpp_diversity_candidate_size}}",
          target_item = target_item,
      ) \
      .split_string(
        input_common_attr="explore_photo_age_boost_weight_str",
        output_common_attr="age_weight_number",
        delimiters=":",
        trim_spaces=True,
        skip_empty_tokens=True,
        parse_to_double=True
      ) \
      .split_string(
        input_common_attr="explore_pic_age_boost_weight_str",
        output_common_attr="pic_age_weight_number",
        delimiters=":",
        trim_spaces=True,
        skip_empty_tokens=True,
        parse_to_double=True
      ) \
      .enrich_attr_by_lua(
          import_common_attr = [
              "dpp_rerank_picture_discount_param_new",
              "fr_rerank_photo_level_discount_param",
              "fr_rerank_duration_lt_58_discount_param",
              "explore_rerank_enable_no_pctr_multiply",
              "explore_rerank_enable_diversity_pfntr_multiply",
              "explore_rerank_enable_diversity_pfntr_multiply_coff",
              "fr_rerank_interest_explore_boost_param",
              "age_weight_number",
              "pic_age_weight_number",
              "explore_photo_age_boost_fans_threshold",
              "explore_rerank_ctr_power_adjust_weight",
              "explore_rerank_fetr_feff_power"
          ],
          import_item_attr = [
              "duration_ms",
              "is_picture",
              "content_safety_level_with_namespace__level_hot_online",
              "corr_pctr",
              "corr_pwtr",
              "pltr",
              "fr_score1",
              "fr_score2",
              "consume_time_ltr",
              "pftr",
              "pptr",
              "plvtr",
              "pepstr",
              "explore_fr_ensemble_score",
              "pcltr",
              "fetr",
              "fountain_eff",
              "pcmtr",
              "pcmef",
              "avg_watch_time_ms",
              "ada_xtr_score",
              "watchtime_interact_score",
              "awesome_wtd",
              "pdtr",
              "consume_time_pf2r_score",
              "interact_fusion_score",
              "watch_time_fusion_score",
              "rerank_pic_coff_attr_transfer",
              "is_explore_photo",
              "pctr_pfr2r",
              "pcltr_pfr2r",
              "photo_age_hour",
              "pctr_duration_debias_coffe",
              "pwtr_duration_debias_coffe",
              "pltr_duration_debias_coffe",
              "pftr_duration_debias_coffe",
              "plvtr_duration_debias_coffe",
              "pcltr_duration_debias_coffe",
              "pcmtr_duration_debias_coffe",
              "author__fans_count",
              "corr_cpr",
              "pevtr",
              "min_act_rank_score",
              "gen_l2r_score"
          ],
          export_item_attr = [
              "ctr",
              "wtr",
              "ltr",
              "fr_score1_corr", #这个要改名字
              "fr_score2_corr", #这个要改名字
              "l2r_score",
              "ftr",
              "duration_gt_58s_corr", #这个需要改名字
              "ptr",
              "lvtr",
              "epstr",
              "ensemble_score",
              "cltr",
              "fetr_corr", #这个要改名字
              "feff",
              "cmtr",
              "cmef",
              "diversity",
              "ada_score",
              "interact_cost",
              "awesome_wtd_score",
              "dtr",
              "pdbfrtr",
              "interact_fusion",
              "watch_time_fusion",
              "frctr_fusion",
              "frcltr_fusion",
              "rerank_cpr_corr",
              "rerank_pevtr_corr",
              "min_act_rank",
              "gen_l2r_score_corr",
          ],
          function_for_item = "full_rank_score_cal",
          lua_script_file = "explore/rerank/lua/module/rerank_gen_list__multi_lua.lua",
          range_end = "{{dpp_diversity_candidate_size}}",
          target_item = target_item,
      ) \
      .if_("explore_rerank_enable_ef_score_debias_by_picture_type == 1") \
        .set_attr_value(
          item_attrs=[{
            "name": "fetr_corr",
            "type": "double",
            "value": 0.0
          }, {
            "name": "feff",
            "type": "double",
            "value": 0.0
          }],
          target_item = {
            "picture_type": [2, 3]
          }
        ) \
      .end_() \
      .enrich_attr_by_lua(
        import_common_attr = [
          "rerank_picture_discount_param"
        ],
        import_item_attr = [
          "is_picture",
          "explore_fr_ensemble_score",
          "consume_time_ltr"
        ],
        export_item_attr = [
          "fullrank_neg_feedback_discount_score",
          "fullrank_l2r_score"
        ],
        function_for_item = "other_name",
        lua_script_file = "explore/rerank/lua/module/rerank_gen_list__multi_lua.lua",
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          "shuffle_policy"
        ],
        import_common_attr = [
          "is_shuffle",
          "rerank_variety_shuanglie_enable4",
          "is_tmp_risk_user"
        ],
        export_item_attr = [
          "shuffle_policy_changed"
        ],
        export_common_attr = [
          "rerank_variety_shuanglie_enable4"
        ],
        function_name = "ManNeedShuffle",
        class_name = "ExploreLightFunctionSetV2",
        range_end = "{{dpp_diversity_candidate_size}}",
        target_item = target_item,
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          "hetu_tag_level_info__hetu_level_one"
        ],
        import_common_attr = [
          "rerank_variety_shuanglie_enable5",
          "user_risk_level",
          "explore_user_risk_min"
        ],
        export_item_attr = [
          "gr_policy_softcore"
        ],
        export_common_attr = [
          "rerank_variety_shuanglie_enable5"
        ],
        function_name = "ManNeedShuffleSoftCore",
        class_name = "ExploreLightFunctionSetV2",
        range_end = "{{dpp_diversity_candidate_size}}",
        target_item = target_item,
      ) \
      .if_("explore_rerank_enable_cal_emp_action_score == 1") \
        .calc_weighted_sum( # todo 2025/12/10之前和精排合并
          channels = [
            { "name": "ctr", "weight": "{{explore_rerank_emp_action_pctr_weight}}" },
            { "name": "wtr", "weight": "{{explore_rerank_emp_action_pwtr_weight}}" },
            { "name": "ltr", "weight": "{{explore_rerank_emp_action_pltr_weight}}" },
            { "name": "lvtr", "weight": "{{explore_rerank_emp_action_plvtr_weight}}" },
            { "name": "fr_score1_corr", "weight": "{{explore_rerank_emp_action_pf1_weight}}" },
            { "name": "fr_score2_corr", "weight": "{{explore_rerank_emp_action_pf2_weight}}" },
            { "name": "empirical_ctr", "weight": "{{explore_rerank_emp_action_ctr_weight}}" },
            { "name": "empirical_ltr", "weight": "{{explore_rerank_emp_action_ltr_weight}}" },
            { "name": "empirical_wtr", "weight": "{{explore_rerank_emp_action_wtr_weight}}" },
            { "name": "empirical_ftr", "weight": "{{explore_rerank_emp_action_ftr_weight}}" },
            { "name": "empirical_cmtr", "weight": "{{explore_rerank_emp_action_cmtr_weight}}" },
            { "name": "empirical_watch_time", "weight": "{{explore_rerank_emp_action_watch_time_weight}}" },
          ],
          output_item_attr = "emp_action_score",
          target_item = target_item,
        ) \
        .pack_item_attr(
          item_source={
            "reco_results": True,
          },
          mappings=[
            {
              "aggregator": "avg",
              "from_item_attr": "emp_action_score",
              "to_common_attr": "ctr_empirical_action_emp_action_score_avg"
            },
            {
              "aggregator": "max",
              "from_item_attr": "emp_action_score",
              "to_common_attr": "ctr_empirical_action_emp_action_score_max"
            },
          ],
          target_item = target_item,
        ) \
      .end_() \
      .if_("explore_rerank_enable_ctr_empirical_action_score == 1") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            "ctr",
            "emp_action_score",
            "explore_stat__real_show_count",
            "thanos_stats__real_show_count",
            "explore_stat__click_count"
          ],
          import_common_attr = [
            "ctr_empirical_action_emp_action_score_max",
            "ctr_empirical_action_emp_action_score_avg",
            {"name": "explore_rerank_ctr_empirical_action_ctr_weight", "as": "ctr_weight"},
            {"name": "explore_rerank_ctr_empirical_action_bias_weight", "as": "bias_weight"},
            {"name": "explore_rerank_ctr_empirical_action_emp_action_score_weight", "as": "emp_action_score_weight"},
            {"name": "explore_rerank_ctr_empirical_action_show_limit_threshold", "as": "show_limit_threshold"},
            {"name": "explore_rerank_ctr_empirical_action_ctr_limit_threshold", "as": "ctr_limit_threshold"},
            {"name": "explore_rerank_ctr_empirical_action_percent_threshold", "as": "percent_threshold"},
          ],
          export_item_attr = [
            "ctr_emp_action"
          ],
          function_name = "CalCtrEmpiricalActionScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item = target_item,
        ) \
      .end_() \
      .if_("enable_explore_sim_user_hate_item_list_similarity_score == 1 and recent_hate_count <= explore_koc_htr_count_threshold") \
        .item_attr_operation(
          item_attr_a = "hate_photo_id_similary_score",
          common_attr_b = 1.0,
          operator = "-",
          output_attr = "reverse_hate_photo_id_similary_score",
          target_item = target_item,
        ) \
        .item_attr_operation(
          item_attr_a = "reverse_hate_photo_id_similary_score",
          common_attr_b = -1.0,
          operator = "*",
          output_attr = "reverse_hate_photo_id_similary_score",
          target_item = target_item,
        ) \
      .end_() \
      .if_("enable_rerank_gen_user_short_develop_interest_score == 1") \
        .cast_attr_type(
          attr_type_cast_configs=[
            {
              "to_type": "double",
              "from_item_attr": "is_user_short_develop_interest",
              "to_item_attr": "short_develop_interest_score"
            },
          ],
          target_item = target_item,
        ) \
      .end_() \
      .if_("explore_rerank_enable_gen_sexually_photo == 1") \
        .split_string(
          input_common_attr = "sexually_manjiao_markcode_str",
          output_common_attr = "sexually_manjiao_markcode_list",
          delimiters = ",",
          trim_spaces = True,
          parse_to_int = True,
        ) \
        .enrich_attr_by_light_function(
          import_item_attr = [
            {"name": "manjiao_markcode", "as": "attrs"},
          ],
          import_common_attr = [
            {"name": "sexually_manjiao_markcode_list", "as": "attr_list"},
          ],
          export_item_attr = [
            {"name": "is_in_set", "as": "is_sexually_photo"},
          ],
          function_name = "AttrListIsInSet",
          class_name = "ExploreLightFunctionSetV2",
          target_item = target_item,
        ) \
      .end_() \

  def enrich_common_attr(self):
    self.flow.if_("enable_explore_cocoon_rerank_diversity_control_dynamic == 1")
    self.multi_int_value_adjust(
      int_value_name_list = [
        "rerank_variety_shuanglie_winsize8",
        "rerank_variety_shuanglie_max8",
        "rerank_trigger_pid_diversity_winsize",
      ],
      strategy_name = "cocoon"
    )
    self.flow.end_()
    self.flow.if_("explore_enable_user_need_break_cocoon_rerank == 1 and user_need_break_cocoon_flag == 1")
    self.multi_int_value_adjust(
      int_value_name_list = [
        "rerank_variety_shuanglie_winsize8",
        "rerank_variety_shuanglie_max8",
        "rerank_variety_shuanglie_winsize2",
        "rerank_variety_shuanglie_max2",
      ],
      strategy_name = "user_need_break_cocoon"
    )
    self.flow.end_()
    self.flow \
      .if_("enable_rerank_explore_low_active_customization_view_score_weight == 1 and is_explore_new_la_user == 1") \
        .copy_attr(
          attrs=[{
            "from_common": "explore_rerank_gen_seed_ensemble_sense_view_predict_trans_score_weight_low_active",
            "to_common": "explore_rerank_gen_seed_ensemble_sense_view_predict_trans_score_weight"
          }, {
            "from_common": "explore_rerank_gen_seed_ensemble_cover_view_predict_trans_score_weight_low_active",
            "to_common": "explore_rerank_gen_seed_ensemble_cover_view_predict_trans_score_weight"
          }]
        ) \
      .end_() \
      .if_("enable_explore_rerank_weight_adjust_by_high_time_rate == 1") \
        .calc_by_formula1(
          kconf_key = "formula.scenarioKey49.explore_reranking_low_time_active_weight_adjust_f1",
          import_common_attr = [
            "explore_rerank_gen_seed_ensemble_ctr_weight",
            "explore_rerank_gen_seed_ensemble_wtr_weight",
            "explore_rerank_gen_seed_ensemble_ltr_weight",
            "explore_rerank_gen_seed_ensemble_ftr_weight",
            "explore_rerank_gen_seed_ensemble_cmtr_weight",
            "explore_rerank_gen_seed_ensemble_awesome_wtd_score_weight",
            "active_days_high_time_rate"
          ],
          export_formula_value = [
            {"name": "explore_rerank_gen_seed_ensemble_ctr_weight", "to_common": True},
            {"name": "explore_rerank_gen_seed_ensemble_wtr_weight", "to_common": True},
            {"name": "explore_rerank_gen_seed_ensemble_ltr_weight", "to_common": True},
            {"name": "explore_rerank_gen_seed_ensemble_ftr_weight", "to_common": True},
            {"name": "explore_rerank_gen_seed_ensemble_cmtr_weight", "to_common": True},
            {"name": "explore_rerank_gen_seed_ensemble_awesome_wtd_score_weight", "to_common": True}
          ],
          abtest_biz_name = "KUAISHOU_APPS"
        ) \
      .end_() \
      .if_("enable_rerank_watch_time_fusion_score_weight_divide_vv_adjust == 1 and active_days_avg_vv >= explore_rerank_watch_time_fusion_vv_threshold") \
        .gen_common_attr_by_lua( # watch_time_fusion_score 根据活跃天均vv划分
          attr_map = {
              "explore_rerank_gen_seed_ensemble_watch_time_fusion_weight" : "explore_rerank_watch_time_fusion_high_vv_weight * explore_rerank_gen_seed_ensemble_watch_time_fusion_weight",
              "explore_rerank_gen_seed_ensemble_watch_time_fusion_weight_addAndMul" : "explore_rerank_watch_time_fusion_high_vv_weight * explore_rerank_gen_seed_ensemble_watch_time_fusion_weight_addAndMul",
          }
        ) \
      .end_() \
      .if_("enable_rerank_watch_time_fusion_score_weight_divide_active_adjust == 1 and (find_user_active_degree == 3 or find_user_active_degree == 4)") \
        .gen_common_attr_by_lua( # watch_time_fusion_score 根据人群活跃度划分
          attr_map = {
              "explore_rerank_gen_seed_ensemble_watch_time_fusion_weight" : "explore_rerank_watch_time_fusion_high_active_weight * explore_rerank_gen_seed_ensemble_watch_time_fusion_weight",
              "explore_rerank_gen_seed_ensemble_watch_time_fusion_weight_addAndMul" : "explore_rerank_watch_time_fusion_high_active_weight * explore_rerank_gen_seed_ensemble_watch_time_fusion_weight_addAndMul",
          }
        ) \
      .end_() \
      .if_("enable_rerank_slide_pctr_score_weight_divide_vv_adjust == 1 and active_days_avg_vv >= explore_rerank_slide_pctr_score_vv_threshold") \
        .gen_common_attr_by_lua( # slide_pctr_score 根据活跃天均vv划分
          attr_map = {
              "explore_rerank_gen_seed_ensemble_fr_slide_pctr_score_weight" : "explore_rerank_slide_pctr_score_high_vv_weight * explore_rerank_gen_seed_ensemble_fr_slide_pctr_score_weight",
              "explore_rerank_gen_seed_ensemble_fr_slide_pctr_score_raw_pow_weight" : "explore_rerank_slide_pctr_score_high_vv_weight * explore_rerank_gen_seed_ensemble_fr_slide_pctr_score_raw_pow_weight",
              "explore_rerank_gen_seed_ensemble_fr_slide_pctr_score_raw_weight" : "explore_rerank_slide_pctr_score_high_vv_weight * explore_rerank_gen_seed_ensemble_fr_slide_pctr_score_raw_weight",
          }
        ) \
      .end_() \
      .if_("enable_rerank_slide_pctr_score_weight_divide_active_adjust == 1 and (find_user_active_degree == 3 or find_user_active_degree == 4)") \
        .gen_common_attr_by_lua( # slide_pctr_score 根据人群活跃度划分
          attr_map = {
              "explore_rerank_gen_seed_ensemble_fr_slide_pctr_score_weight" : "explore_rerank_fr_slide_pctr_score_high_active_weight * explore_rerank_gen_seed_ensemble_fr_slide_pctr_score_weight",
              "explore_rerank_gen_seed_ensemble_fr_slide_pctr_score_raw_pow_weight" : "explore_rerank_fr_slide_pctr_score_high_active_weight * explore_rerank_gen_seed_ensemble_fr_slide_pctr_score_raw_pow_weight",
              "explore_rerank_gen_seed_ensemble_fr_slide_pctr_score_raw_weight" : "explore_rerank_fr_slide_pctr_score_high_active_weight * explore_rerank_gen_seed_ensemble_fr_slide_pctr_score_raw_weight",
          }
        ) \
      .end_() \
      .if_("enable_explore_rerank_diversity_interest_lma_score_divide_active_adjust == 1 and find_user_active_degree ~= 1 and find_user_active_degree ~= 2") \
        .gen_common_attr_by_lua( # explore_diversity_interest_lma_score 根据人群活跃度划分
          attr_map = {
            "explore_rerank_gen_seed_ensemble_explore_diversity_interest_lma_score_weight" : "explore_rerank_diversity_interest_lma_score_adjust_coeff * explore_rerank_gen_seed_ensemble_explore_diversity_interest_lma_score_weight",
            "explore_rerank_gen_seed_ensemble_explore_diversity_interest_lma_score_raw_pow_weight" : "explore_rerank_diversity_interest_lma_score_adjust_coeff * explore_rerank_gen_seed_ensemble_explore_diversity_interest_lma_score_raw_pow_weight",
            "explore_rerank_gen_seed_ensemble_explore_diversity_interest_lma_score_raw_weight" : "explore_rerank_diversity_interest_lma_score_adjust_coeff * explore_rerank_gen_seed_ensemble_explore_diversity_interest_lma_score_raw_weight",
          }
        ) \
      .end_() \
      .if_("enable_rerank_user_age_tgi_product_first_refresh_weight_adjust == 1 and is_first_refresh == 1") \
        .gen_common_attr_by_lua( # user_age_interest_tagnex_tgi_product_fr_pxtr_score 首屏权重独立设置
          attr_map = {
            "explore_rerank_gen_seed_ensemble_user_age_interest_tagnex_tgi_product_fr_pxtr_score_weight" : "explore_rerank_user_age_tgi_product_first_refresh_weight",
            "explore_rerank_gen_seed_ensemble_user_age_interest_tagnex_tgi_product_fr_pxtr_score_raw_pow_weight" : "explore_rerank_user_age_tgi_product_first_refresh_raw_pow_weight",
            "explore_rerank_gen_seed_ensemble_user_age_interest_tagnex_tgi_product_fr_pxtr_score_raw_weight" : "explore_rerank_user_age_tgi_product_first_refresh_raw_weight",
          }
        ) \
      .end_() \
      .if_("enable_user_age_tgi_score_population_weight_adjust == 1 and basic_info_age_segment_v2 > user_age_tgi_score_population_age_segment_threshold and active_days_gt_5min_rate < user_age_tgi_score_population_active_days_threshold") \
        .copy_attr(
          attrs = [{
            "from_common": "explore_rerank_gen_seed_ensemble_user_age_interest_tagnex_tgi_product_fr_pxtr_score_population_weight",
            "to_common": "explore_rerank_gen_seed_ensemble_user_age_interest_tagnex_tgi_product_fr_pxtr_score_weight"
          },
          {
            "from_common": "explore_rerank_gen_seed_ensemble_user_age_interest_tagnex_tgi_product_fr_pxtr_score_population_raw_pow_weight",
            "to_common": "explore_rerank_gen_seed_ensemble_user_age_interest_tagnex_tgi_product_fr_pxtr_score_raw_pow_weight"
          },
          {
            "from_common": "explore_rerank_gen_seed_ensemble_user_age_interest_tagnex_tgi_product_fr_pxtr_score_population_raw_weight",
            "to_common": "explore_rerank_gen_seed_ensemble_user_age_interest_tagnex_tgi_product_fr_pxtr_score_raw_weight"
          }]
        ) \
      .end_() \
      .if_("enable_explore_rerank_interact_fusion_score_divide_age_adjust == 1 and user_age_segment >= explore_rerank_interact_fusion_score_age_min and user_age_segment <= explore_rerank_interact_fusion_score_age_max") \
        .gen_common_attr_by_lua( # interact_fusion_score 根据年龄划分
          attr_map = {
            "explore_rerank_gen_seed_ensemble_interact_fusion_weight" : "explore_rerank_interact_fusion_weight_adjust_bias + explore_rerank_gen_seed_ensemble_interact_fusion_weight",
            "explore_rerank_gen_seed_ensemble_interact_fusion_raw_pow_weight" : "explore_rerank_interact_fusion_raw_pow_weight_adjust_bias + explore_rerank_gen_seed_ensemble_interact_fusion_raw_pow_weight",
            "explore_rerank_gen_seed_ensemble_interact_fusion_raw_weight" : "explore_rerank_interact_fusion_raw_weight_adjust_bias + explore_rerank_gen_seed_ensemble_interact_fusion_raw_weight",
          }
        ) \
      .end_() \
      .if_("enable_explore_rerank_min_act_rank_score_divide_age_adjust == 1 and user_age_segment >= explore_rerank_min_act_rank_score_age_min and user_age_segment <= explore_rerank_min_act_rank_score_age_max") \
        .gen_common_attr_by_lua( # min_act_rank_score 根据年龄划分
          attr_map = {
            "explore_rerank_gen_seed_ensemble_min_act_rank_weight" : "explore_rerank_min_act_rank_weight_adjust_bias + explore_rerank_gen_seed_ensemble_min_act_rank_weight",
            "explore_rerank_gen_seed_ensemble_min_act_rank_raw_pow_weight" : "explore_rerank_min_act_rank_raw_pow_weight_adjust_bias + explore_rerank_gen_seed_ensemble_min_act_rank_raw_pow_weight",
            "explore_rerank_gen_seed_ensemble_min_act_rank_raw_weight" : "explore_rerank_min_act_rank_raw_weight_adjust_bias + explore_rerank_gen_seed_ensemble_min_act_rank_raw_weight",
          }
        ) \
      .end_() \
      .enrich_with_protobuf(
          from_extra_var = "user_info_ptr",
          attrs = [
            dict(name="is_shuffle", path="feature_collection.is_shuffle")
          ]
      ) \
      .if_("explore_rerank_enable_boost_user_group_emp_psvtr == 1") \
        .gen_common_attr_by_lua(
          attr_map={
            "explore_rerank_gen_seed_ensemble_psvr_weight": "explore_rerank_gen_seed_ensemble_psvr_weight * user_group_emp_svtr",
          }
        ) \
      .end_() \
      .if_("explore_user_group_consume_weight_adjust_rerank == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = user_group_input_common_attrs,
          export_common_attr = user_group_output_common_attrs,
          function_name = "UserGroupWeightAdjustCoef",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_rerank_enable_pic_queue_weight_adjust == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rerank_gen_seed_ensemble_fr_pic_ensemble_score_weight", "as": "origin_value"},
            {"name": "explore_rerank_pic_queue_weight_adjust_param", "as": "weight"},
          ],
          export_common_attr = [
            {"name": "new_weight", "as": "explore_rerank_gen_seed_ensemble_fr_pic_ensemble_score_weight"}
          ],
          function_name = "AdjustPicQueueWeight",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_enable_pic_rerank_weight_muted_boost == 1 and isMuted == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rerank_gen_seed_ensemble_fr_pic_ensemble_score_weight", "as": "origin_value"},
            {"name": "explore_pic_rerank_weight_muted_boost_coef", "as": "weight"},
          ],
          export_common_attr = [
            {"name": "new_weight", "as": "explore_rerank_gen_seed_ensemble_fr_pic_ensemble_score_weight"}
          ],
          function_name = "AdjustPicQueueWeight",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_rerank_enable_low_pic_play_user == 1 and enable_pic_explore_flag == 1") \
        .switch_("explore_rerank_enable_low_pic_play_user_mode") \
          .case_(1) \
            .gen_common_attr_by_lua(
              attr_map={
                "is_low_pic_play_boost_user": "(explore_rerank_enable_target_age_user_boost == 1 and (user_age_segment or 0) < explore_rerank_target_user_age_thresh) and ((explore_rerank_enable_xhs_user_boost == 1 and (uIsXhsUser or 0) == 1) or (explore_rerank_enable_history_pic_user_boost == 1 and (pic_stat_pic_eff_play_cnt or 0) > explore_rerank_history_pic_user_boost_thresh))",
              }
            ) \
          .default_() \
            .gen_common_attr_by_lua(
              attr_map={
                "is_low_pic_play_boost_user": "(explore_rerank_enable_target_age_user_boost == 1 and (user_age_segment or 0) < explore_rerank_target_user_age_thresh) or (explore_rerank_enable_xhs_user_boost == 1 and (uIsXhsUser or 0) == 1) or (explore_rerank_enable_history_pic_user_boost == 1 and (pic_stat_pic_eff_play_cnt or 0) > explore_rerank_history_pic_user_boost_thresh)",
              }
            ) \
        .end_() \
      .end_() \
      .if_("explore_rerank_enable_low_pic_play_user_weight_boost == 1 and is_low_pic_play_boost_user == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rerank_gen_seed_ensemble_fr_pic_ensemble_score_weight", "as": "origin_value"},
            {"name": "explore_rerank_target_user_pic_boost_coef", "as": "weight"},
          ],
          export_common_attr = [
            {"name": "new_weight", "as": "explore_rerank_gen_seed_ensemble_fr_pic_ensemble_score_weight"}
          ],
          function_name = "AdjustPicQueueWeight",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_rerank_enable_pic_first_screen_user == 1") \
        .gen_common_attr_by_lua( # 临时实验用，后续实验推全，需修改成is_first_refresh，实验eta 10/26
          attr_map={
            "is_pic_first_screen_boost_user": "is_first_refresh > 0",
          }
        ) \
      .end_() \
      .if_("explore_rerank_enable_pic_first_screen_user_weight_boost == 1 and is_pic_first_screen_boost_user == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rerank_gen_seed_ensemble_fr_pic_ensemble_score_weight", "as": "origin_value"},
            {"name": "explore_rerank_pic_first_screen_user_boost_coef", "as": "weight"},
          ],
          export_common_attr = [
            {"name": "new_weight", "as": "explore_rerank_gen_seed_ensemble_fr_pic_ensemble_score_weight"}
          ],
          function_name = "AdjustPicQueueWeight",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_rerank_enable_pic_userfeedback_weight_adjust == 1 and (uDoubleOutsideValidPicClusterCnt7dKV or 0) < explore_rerank_pic_interest_thresh and pic_stat_pic_play_cnt > explore_rerank_user_feedback_pic_limit_cnt and  pic_stat_video_play_cnt > explore_rerank_user_feedback_video_limit_cnt") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rerank_user_feedback_weight_adjust_mode", "as": "adjust_mode"},
            {"name": "explore_rerank_user_feedback_enbable_pic_play_cnt_adjust", "as": "enbable_pic_play_cnt_adjust"},
            {"name": "pic_stat_pic_play_cnt", "as": "pic_stat_pic_play_cnt"},
            {"name": "pic_stat_pic_eff_play_cnt", "as": "pic_stat_pic_eff_play_cnt"},
            {"name": "pic_stat_video_play_cnt", "as": "pic_stat_video_play_cnt"},
            {"name": "pic_stat_video_eff_play_cnt", "as": "pic_stat_video_eff_play_cnt"},
            {"name": "pic_stat_pic_like_cnt", "as": "pic_stat_pic_like_cnt"},
            {"name": "pic_stat_pic_follow_cnt", "as": "pic_stat_pic_follow_cnt"},
            {"name": "pic_stat_pic_forward_cnt", "as": "pic_stat_pic_forward_cnt"},
            {"name": "pic_stat_pic_comment_cnt", "as": "pic_stat_pic_comment_cnt"},
            {"name": "pic_stat_video_like_cnt", "as": "pic_stat_video_like_cnt"},
            {"name": "pic_stat_video_follow_cnt", "as": "pic_stat_video_follow_cnt"},
            {"name": "pic_stat_video_forward_cnt", "as": "pic_stat_video_forward_cnt"},
            {"name": "pic_stat_video_comment_cnt", "as": "pic_stat_video_comment_cnt"},
            {"name": "explore_rerank_user_feedback_pic_play_percent_bias", "as": "pic_play_percent_bias"},
            {"name": "explore_rerank_user_feedback_pic_weight_adjust_coef_min", "as": "explore_rerank_pic_weight_adjust_coef_min"},
            {"name": "explore_rerank_user_feedback_pic_weight_adjust_coef_max", "as": "explore_rerank_pic_weight_adjust_coef_max"},
          ],
          export_common_attr = [
            {"name": "explore_rerank_pic_weight_adjust_coeff", "as": "explore_rerank_pic_user_feedback_weight_adjust_coef"}
          ],
          function_name = "DynamicPicRerankWeightAdjust",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rerank_gen_seed_ensemble_fr_pic_ensemble_score_weight", "as": "origin_value"},
            {"name": "explore_rerank_pic_user_feedback_weight_adjust_coef", "as": "weight"},
          ],
          export_common_attr = [
            {"name": "new_weight", "as": "explore_rerank_gen_seed_ensemble_fr_pic_ensemble_score_weight"}
          ],
          function_name = "AdjustPicQueueWeight",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()\
      .if_("explore_rerank_enable_pic_weight_boost == 1") \
        .calc_by_formula1(
          kconf_key = "formula.scenarioKey07.RerankExplorePicWeightAdjustCtrCoeff",
          import_common_attr = [
            "pic_ctr_preference_coeff"
          ],
          export_formula_value = [
            {"name": "final_score", "as": "explore_rerank_pic_video_ctr_preference_coeff", "to_common": True}
          ],
          abtest_biz_name = "KUAISHOU_APPS"
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "pic_stat_pic_like_cnt",
            "pic_stat_pic_follow_cnt",
            "pic_stat_pic_forward_cnt",
            "pic_stat_pic_comment_cnt",
            "pic_stat_pic_eff_play_cnt",
            "pic_stat_pic_play_cnt",
            "pic_stat_video_like_cnt",
            "pic_stat_video_follow_cnt",
            "pic_stat_video_forward_cnt",
            "pic_stat_video_comment_cnt",
            "pic_stat_video_eff_play_cnt",
            "pic_stat_video_play_cnt",
            "explore_rerank_pic_weight_boost_low_bound",
            "explore_rerank_pic_weight_boost_up_bound",
            "explore_rerank_pic_weight_boost_mode",
            "explore_rerank_pic_weight_boost_pic_limit_cnt",
            "explore_rerank_pic_weight_boost_video_limit_cnt",
            "explore_rerank_pic_video_interact_emp_ratio",
            "explore_rerank_pic_video_eff_play_emp_ratio",
            "explore_rerank_pic_video_play_cnt_emp_ratio",
            "explore_rerank_pic_video_interact_ratio_weight",
            "explore_rerank_pic_video_eff_play_ratio_weight",
            "explore_rerank_pic_video_play_cnt_ratio_weight",
            "explore_rerank_pic_video_action_value_emp_ratio",
            "explore_rerank_pic_video_like_weight_boost_coef",
            "explore_rerank_pic_video_follow_weight_boost_coef",
            "explore_rerank_pic_video_forward_weight_boost_coef",
            "explore_rerank_pic_video_comment_weight_boost_coef",
            "explore_rerank_pic_video_eff_play_weight_boost_coef",
            "explore_rerank_enable_filter_low_play_user",
            "explore_rerank_pic_weight_uplift_action_coeff",
            "explore_rerank_pic_eff_play_smooth_alpha",
            "explore_rerank_pic_eff_play_smooth_beta",
            "explore_rerank_pic_interact_smooth_alpha",
            "explore_rerank_pic_interact_smooth_beta",
            "explore_rerank_video_eff_play_smooth_alpha",
            "explore_rerank_video_eff_play_smooth_beta",
            "explore_rerank_video_interact_smooth_alpha",
            "explore_rerank_video_interact_smooth_beta",
            "explore_rerank_pic_video_play_cnt_smooth_alpha",
            "explore_rerank_pic_video_play_cnt_smooth_beta",
            "explore_rerank_enable_pic_weight_uplift_adjust",
            "uExplorePicUpliftValuesKV",
            "explore_rerank_pic_weight_uplift_task_num",
            "explore_rerank_pic_weight_uplift_thresholds",
            "explore_rerank_pic_weight_uplift_coeff_alphas",
            "explore_rerank_pic_weight_uplift_coeff_betas",
            "explore_rerank_pic_weight_uplift_coeff_power_weights",
            "explore_rerank_pic_weight_uplift_upper_bound",
            "explore_rerank_pic_weight_uplift_lower_bound",
            "explore_rerank_pic_ctr_preference_weight",
            "explore_rerank_pic_video_ctr_preference_coeff"
          ],
          export_common_attr = [
            "explore_rerank_pic_weight_boost_coeff",
          ],
          function_name = "DynamicPicWeightBoost",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .gen_common_attr_by_lua(
          attr_map = {
            "explore_rerank_gen_seed_ensemble_fr_pic_ensemble_score_weight": "explore_rerank_gen_seed_ensemble_fr_pic_ensemble_score_weight * explore_rerank_pic_weight_boost_coeff"
          }
        ) \
      .end_() \
      .if_("explore_rerank_enable_pic_queue_raw_weight_adjust == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rerank_gen_seed_ensemble_fr_pic_ensemble_score_raw_weight", "as": "origin_value"},
            {"name": "explore_rerank_pic_queue_raw_weight_adjust_param", "as": "weight"},
          ],
          export_common_attr = [
            {"name": "new_weight", "as": "explore_rerank_gen_seed_ensemble_fr_pic_ensemble_score_raw_weight"}
          ],
          function_name = "AdjustPicQueueWeight",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_rerank_enable_boost_user_mau_emp_xtr == 1") \
        .gen_common_attr_by_lua(
          attr_map={
            "explore_rerank_gen_seed_ensemble_ctr_weight" : "user_mau_emp_evtr * explore_rerank_gen_seed_ensemble_ctr_weight",
            "explore_rerank_gen_seed_ensemble_ltr_weight" : "user_mau_emp_ltr * explore_rerank_gen_seed_ensemble_ltr_weight",
            "explore_rerank_gen_seed_ensemble_wtr_weight" : "user_mau_emp_wtr * explore_rerank_gen_seed_ensemble_wtr_weight",
            "explore_rerank_gen_seed_ensemble_ftr_weight" : "user_mau_emp_ftr * explore_rerank_gen_seed_ensemble_ftr_weight",
            "explore_rerank_gen_seed_ensemble_cmtr_weight" : "user_mau_emp_cmtr * explore_rerank_gen_seed_ensemble_cmtr_weight",
            "explore_rerank_gen_seed_ensemble_awesome_wtd_score_weight" : "user_mau_emp_rank_play * explore_rerank_gen_seed_ensemble_awesome_wtd_score_weight",
          }
        ) \
      .end_() \
      .if_("rerank_enable_user_group_dynamic_weight == 1", to_be_delete = "date=2024-05-29;committer=xuwei09") \
        .gen_common_attr_by_lua(
          attr_map={
            "explore_weight_adjust_avg_emp_ltr": "explore_weight_adjust_avg_emp_ltr * user_group_emp_ltr",
            "explore_weight_adjust_avg_emp_wtr": "explore_weight_adjust_avg_emp_wtr * user_group_emp_wtr",
            "explore_weight_adjust_avg_emp_ftr": "explore_weight_adjust_avg_emp_ftr * user_group_emp_ftr",
            "explore_weight_adjust_avg_emp_cmtr": "explore_weight_adjust_avg_emp_cmtr * user_group_emp_cmtr",
          }
        ) \
      .end_() \
      .if_("explore_rerank_enable_boost_user_timely_diversity_score == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "user_info_ptr",
            {"name": "explore_rerank_user_timely_diversity_num_threshold", "as": "num_threshold"},
            {"name": "explore_rerank_user_timely_diversity_time_ms_threshold", "as": "time_ms_threshold"},
            {"name": "explore_rerank_user_timely_diversity_hetu_rate_threshold", "as": "hetu_rate_threshold"},
            {"name": "explore_rerank_user_timely_diversity_max_hetu_weight", "as": "max_hetu_weight"},
            {"name": "explore_rerank_user_timely_diversity_min_hetu_weight", "as": "min_hetu_weight"},
            {"name": "explore_rerank_user_timely_diversity_click_hetu_weight", "as": "click_hetu_weight"},
          ],
          export_common_attr = [
            {"name": "output_weight", "as": "explore_rerank_user_timely_diversity_weight"},
            {"name": "output_click_weight", "as": "explore_rerank_user_timely_click_diversity_weight"}
          ],
          function_name = "GetTimelyTopHetuRate",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .switch_("explore_rerank_boost_user_timely_diversity_method") \
          .case_(1) \
            .gen_common_attr_by_lua(
              attr_map={
                "explore_rerank_gen_seed_ensemble_ctr_weight" : "(explore_rerank_user_timely_diversity_weight or 1.0) * explore_rerank_gen_seed_ensemble_ctr_weight",
              }
            ) \
          .case_(2) \
            .gen_common_attr_by_lua(
              attr_map={
                "explore_rerank_gen_seed_ensemble_ctr_weight" : "(explore_rerank_user_timely_click_diversity_weight or 1.0) * explore_rerank_gen_seed_ensemble_ctr_weight",
              }
            ) \
        .end_() \
      .end_() \
      .if_("enable_user_timely_diversity_entropy_rerank_pctr_weight_adjust == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_user_timely_diversity_entropy_score", "as": "weight"},
            {"name": "explore_rerank_gen_seed_ensemble_ctr_weight", "as": "value"},
          ],
          export_common_attr = [
            {"name": "new_value", "as": "explore_rerank_gen_seed_ensemble_ctr_weight"},
          ],
          function_name = "CalExploreDoubleMultiDouble",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_rerank_sort_weight_adjust == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "enable_explore_weight_adjust_v2",
            "explore_colossus_user_emp_xtr_map_ptr",
            "explore_weight_adjust_coeff_a",
            "explore_weight_adjust_coeff_b",
            "explore_weight_adjust_coeff_c",
            "explore_weight_adjust_coeff_d",
            "user_emp_ltr",
            "user_emp_wtr",
            "user_emp_ftr",
            "user_emp_cmtr",
            "user_emp_eptr",
            {"name": "explore_weight_adjust_avg_emp_ltr", "as": "all_user_emp_ltr"},
            {"name": "explore_weight_adjust_avg_emp_wtr", "as": "all_user_emp_wtr"},
            {"name": "explore_weight_adjust_avg_emp_ftr", "as": "all_user_emp_ftr"},
            {"name": "explore_weight_adjust_avg_emp_cmtr", "as": "all_user_emp_cmtr"},
            {"name": "explore_weight_adjust_avg_emp_eptr", "as": "all_user_emp_eptr"},
            {"name": "explore_rerank_gen_seed_ensemble_ltr_weight", "as": "user_ori_ltr_weight"},
            {"name": "explore_rerank_gen_seed_ensemble_wtr_weight", "as": "user_ori_wtr_weight"},
            {"name": "explore_rerank_gen_seed_ensemble_ftr_weight", "as": "user_ori_ftr_weight"},
            {"name": "explore_rerank_gen_seed_ensemble_cmtr_weight", "as": "user_ori_cmtr_weight"},
            {"name": "explore_rerank_gen_seed_ensemble_epstr_weight", "as": "user_ori_eptr_weight"},
            {"name": "explore_weight_adjust_coeff_min_rerank", "as": "explore_weight_adjust_coeff_min"},
            {"name": "explore_weight_adjust_coeff_max_rerank", "as": "explore_weight_adjust_coeff_max"}
          ],
          export_common_attr = [
            {"name": "user_ltr_weight", "as": "explore_rerank_gen_seed_ensemble_ltr_weight"},
            {"name": "user_wtr_weight", "as": "explore_rerank_gen_seed_ensemble_wtr_weight"},
            {"name": "user_ftr_weight", "as": "explore_rerank_gen_seed_ensemble_ftr_weight"},
            {"name": "user_cmtr_weight", "as": "explore_rerank_gen_seed_ensemble_cmtr_weight"},
            {"name": "user_eptr_weight", "as": "explore_rerank_gen_seed_ensemble_epstr_weight"},
          ],
          function_name = "UserSortWeightAdjust",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_rerank_adjust_hetu_five_window_size == 1") \
        .enrich_attr_by_light_function(
          item_list_from_attr = "explore_realshow_click_common_list",
          import_common_attr = [
            "explore_realshow_click_timestamp_common_list",
            "explore_click_common_list",
            "explore_realshow_hetu_five_common_list",
            {"name": "explore_rerank_hetu_five_window_size", "as": "timestamp_window_thred"},
            {"name": "explore_rerank_hetu_five_realshow_num_limit", "as": "realshow_num_limit"},
          ],
          export_common_attr = [
            {"name": "continuous_hitting_filter_hetu_id_common_attr", "as": "rerank_hetu_five_hetu_id_common_attr"},
            {"name": "continuous_hitting_filter_hetu_cnt_common_attr", "as": "rerank_hetu_five_hetu_cnt_common_attr"},
          ],
          function_name = "CalculateRealshowUnclickCnt",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "rerank_hetu_five_hetu_id_common_attr",
            "rerank_hetu_five_hetu_cnt_common_attr",
            "explore_rerank_hetu_five_enable_white_list",
            "explore_rerank_hetu_five_whitelist_str",
            "explore_rerank_hetu_five_hitting_thred"
          ],
          import_item_attr = [
            "hetu_tag_level_info__hetu_level_five",
          ],
          export_item_attr = [
            "continuous_hitting_hetu_level_five",
          ],
          function_name = "CalculateDynamicHetuFiveWindowSize",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_rerank_enable_request_pxtr_adjust == 1") \
        .gen_common_attr_by_lua(
          attr_map = {
            "explore_rerank_gen_seed_ensemble_ctr_weight" : "rerank_boost_pctr * explore_rerank_gen_seed_ensemble_ctr_weight",
            "explore_rerank_gen_seed_ensemble_ltr_weight" : "rerank_boost_pltr * explore_rerank_gen_seed_ensemble_ltr_weight",
            "explore_rerank_gen_seed_ensemble_wtr_weight" : "rerank_boost_pwtr * explore_rerank_gen_seed_ensemble_wtr_weight",
            "explore_rerank_gen_seed_ensemble_ftr_weight" : "rerank_boost_pftr * explore_rerank_gen_seed_ensemble_ftr_weight",
            "explore_rerank_gen_seed_ensemble_cltr_weight" : "rerank_boost_pcltr * explore_rerank_gen_seed_ensemble_cltr_weight",
            "explore_rerank_gen_seed_ensemble_ptr_weight" : "rerank_boost_pptr * explore_rerank_gen_seed_ensemble_ptr_weight",
            "explore_rerank_gen_seed_ensemble_cmtr_weight" : "rerank_boost_pcmtr * explore_rerank_gen_seed_ensemble_cmtr_weight",
            "explore_rerank_gen_seed_ensemble_fr_score1_corr_weight" : "rerank_boost_fr_score1 * explore_rerank_gen_seed_ensemble_fr_score1_corr_weight",
            "explore_rerank_gen_seed_ensemble_fr_score2_corr_weight" : "rerank_boost_fr_score2 * explore_rerank_gen_seed_ensemble_fr_score2_corr_weight",
            "explore_rerank_gen_seed_ensemble_awesome_wtd_score_weight" : "rerank_boost_awesome_wtd * explore_rerank_gen_seed_ensemble_awesome_wtd_score_weight",
            "explore_rerank_gen_seed_ensemble_fetr_corr_weight" : "rerank_boost_fetr * explore_rerank_gen_seed_ensemble_fetr_corr_weight",
            "explore_rerank_gen_seed_ensemble_feff_weight" : "rerank_boost_fountain_eff * explore_rerank_gen_seed_ensemble_feff_weight",
          }
        ) \
      .end_() \
      .if_("explore_enable_rerank_ef_weight_adjust == 1") \
        .gen_common_attr_by_lua(
          attr_map = {
            "explore_rerank_gen_seed_ensemble_fetr_corr_weight": "explore_rerank_gen_seed_ensemble_fetr_corr_weight * explore_fountain_view_weight",
            "explore_rerank_gen_seed_ensemble_fetr_corr_weight_addAndMul": "explore_rerank_gen_seed_ensemble_fetr_corr_weight_addAndMul * explore_fountain_view_weight",
            "explore_rerank_gen_seed_ensemble_feff_weight": "explore_rerank_gen_seed_ensemble_feff_weight * explore_fountain_view_weight",
            "explore_rerank_gen_seed_ensemble_feff_weight_addAndMul": "explore_rerank_gen_seed_ensemble_feff_weight_addAndMul * explore_fountain_view_weight",
          }
        ) \
      .end_() \
      .if_("explore_rerank_enable_pagesize_pxtr_adjust == 1 and page_index == 1 and refreshTimes ~= 0") \
        .if_("(explore_rerank_enable_gemini_refresh_scene_pxtr_adjust == 0) or (gemini_refresh_scene > 0 and gemini_refresh_scene < 4)") \
          .gen_common_attr_by_lua(
            attr_map = {
              "explore_rerank_gen_seed_ensemble_diversity_fr_ranking_weight" : "explore_rerank_weight_adjust_pagesize_boost_weight * explore_rerank_gen_seed_ensemble_diversity_fr_ranking_weight",
            }
          ) \
        .end_() \
      .end_() \
      .if_("explore_rerank_enable_hour_adjust_queue == 1") \
        .gen_common_attr_by_lua(
          attr_map = {
            "request_hour_str": "tostring(request_hour)"
          }
        ) \
        .get_kconf_params(
          kconf_configs = [
            {
              "kconf_key": "reco.author.exploreRerankHourAdjustQueueWeight",
              "value_type": "double",
              "json_path": "{{request_hour_str}}",
              "default_value": 1.0,
              "export_common_attr": "explore_rerank_hour_adjust_weight"
            },
          ]
        ) \
        .gen_common_attr_by_lua(
          attr_map = {
            "explore_rerank_gen_seed_ensemble_lvtr_weight": "explore_rerank_gen_seed_ensemble_lvtr_weight * explore_rerank_hour_adjust_lvtr_bias_weight * explore_rerank_hour_adjust_weight",
            "explore_rerank_gen_seed_ensemble_fr_score1_corr_weight": "explore_rerank_gen_seed_ensemble_fr_score1_corr_weight * explore_rerank_hour_adjust_f1_bias_weight * explore_rerank_hour_adjust_weight",
            "explore_rerank_gen_seed_ensemble_fr_score2_corr_weight": "explore_rerank_gen_seed_ensemble_fr_score2_corr_weight * explore_rerank_hour_adjust_f2_bias_weight * explore_rerank_hour_adjust_weight",
            "explore_rerank_gen_seed_ensemble_awesome_wtd_score_weight": "explore_rerank_gen_seed_ensemble_awesome_wtd_score_weight * explore_rerank_hour_adjust_wtd_bias_weight * explore_rerank_hour_adjust_weight"
          }
        ) \
      .end_() \
      .if_("explore_rerank_enable_request_pxtr_raw_weight_adjust == 1", to_be_delete = "date=2024-05-29;committer=xuwei09") \
        .gen_common_attr_by_lua(
          attr_map = {
            "explore_rerank_gen_seed_ensemble_ctr_raw_weight" : "rerank_boost_pctr * explore_rerank_gen_seed_ensemble_ctr_raw_weight",
            "explore_rerank_gen_seed_ensemble_ltr_raw_weight" : "rerank_boost_pltr * explore_rerank_gen_seed_ensemble_ltr_raw_weight",
            "explore_rerank_gen_seed_ensemble_wtr_raw_weight" : "rerank_boost_pwtr * explore_rerank_gen_seed_ensemble_wtr_raw_weight",
            "explore_rerank_gen_seed_ensemble_ftr_raw_weight" : "rerank_boost_pftr * explore_rerank_gen_seed_ensemble_ftr_raw_weight",
            "explore_rerank_gen_seed_ensemble_cltr_raw_weight" : "rerank_boost_pcltr * explore_rerank_gen_seed_ensemble_cltr_raw_weight",
            "explore_rerank_gen_seed_ensemble_ptr_raw_weight" : "rerank_boost_pptr * explore_rerank_gen_seed_ensemble_ptr_raw_weight",
            "explore_rerank_gen_seed_ensemble_cmtr_raw_weight" : "rerank_boost_pcmtr * explore_rerank_gen_seed_ensemble_cmtr_raw_weight",
            "explore_rerank_gen_seed_ensemble_fr_score1_corr_raw_weight" : "rerank_boost_fr_score1 * explore_rerank_gen_seed_ensemble_fr_score1_corr_raw_weight",
            "explore_rerank_gen_seed_ensemble_fr_score2_corr_raw_weight" : "rerank_boost_fr_score2 * explore_rerank_gen_seed_ensemble_fr_score2_corr_raw_weight",
            "explore_rerank_gen_seed_ensemble_awesome_wtd_score_raw_weight" : "rerank_boost_awesome_wtd * explore_rerank_gen_seed_ensemble_awesome_wtd_score_raw_weight",
            "explore_rerank_gen_seed_ensemble_fetr_corr_raw_weight" : "rerank_boost_fetr * explore_rerank_gen_seed_ensemble_fetr_corr_raw_weight",
            "explore_rerank_gen_seed_ensemble_feff_raw_weight" : "rerank_boost_fountain_eff * explore_rerank_gen_seed_ensemble_feff_raw_weight",
          }
        ) \
      .end_() \
      .if_("explore_rerank_enable_request_pxtr_weight_adjust == 1") \
        .pack_item_attr(
          item_source = {
            "reco_results": True,
          },
          mappings = [
            {
              "aggregator": "avg",
              "from_item_attr": "corr_pctr",
              "to_common_attr": "pctr_rerank_request_avg"
            },
          ],
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rerank_gen_seed_ensemble_ctr_weight", "as": "weight"},
            {"name": "pctr_rerank_request_avg", "as": "pxtr_avg"},
            {"name": "user_emp_ctr", "as": "user_emp_xtr"},
            {"name": "explore_rerank_request_pctr_weight_adjust_lower", "as": "lower"},
            {"name": "explore_rerank_request_pctr_weight_adjust_upper", "as": "upper"},
            {"name": "explore_rerank_request_pctr_weight_adjust_power_weight", "as": "power_weight"},
            {"name": "explore_rerank_request_pctr_weight_adjust_bias", "as": "bias"},
          ],
          export_common_attr = [
            {"name": "weight", "as": "explore_rerank_gen_seed_ensemble_ctr_weight"}
          ],
          function_name = "RequestPxtrWeightAdjust",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_rerank_enable_emp_fetr_adjust == 1") \
        .gen_common_attr_by_lua(
          attr_map = {
            "explore_rerank_gen_seed_ensemble_fetr_corr_weight" : "emp_rank_boost_fetr * explore_rerank_gen_seed_ensemble_fetr_corr_weight",
            "explore_rerank_gen_seed_ensemble_feff_weight" : "emp_rank_boost_fetr * explore_rerank_gen_seed_ensemble_feff_weight",
          }
        ) \
      .end_() \
      .if_("enable_rerank_high_value_author_explore_diversity == 1 or (explore_rerank_enable_only_cold_start == 1 and refreshTimes == 0) or (explore_rerank_enable_only_first_page == 1 and page_index == 1)") \
        .gen_common_attr_by_lua(
            attr_map = {
              "enable_rerank_high_value_author_explore_diversity_sample": "util.Random() < explore_rerank_high_value_author_freq_thred and 1 or 0"
            }
          ) \
      .end_() \
      .if_("enable_rerank_friend_recommendation_explore_diversity == 1 or (explore_rerank_enable_friend_recommendation_only_cold_start == 1 and refreshTimes == 0) or (explore_rerank_enable_friend_recommendation_only_first_page == 1 and page_index == 1)") \
        .gen_common_attr_by_lua(
            attr_map = {
              "enable_rerank_friend_recommendation_explore_diversity_sample": "util.Random() < explore_rerank_friend_recommendation_freq_thred and 1 or 0"
            }
          ) \
      .end_() \
      .if_("explore_la_rerank_ctr_adjust > 0") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_vv_3d", "as": "origin_value"},
            {"name": "explore_rerank_gen_seed_ensemble_ctr_weight", "as": "pctr_weight"},
            {"name": "explore_rerank_la_ensemble_sort_pctr_weight_max", "as": "weight_max"},
            {"name": "explore_rerank_la_ensemble_sort_pctr_weight_base", "as": "weight_base"}
          ],
          export_common_attr = [
            {"name": "new_pctr_weight", "as": "explore_rerank_gen_seed_ensemble_ctr_weight"}
          ],
          function_name = "AdjustFullRankPxtrWeight",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_cal_share_pull_ftr_rerank == 1") \
        .gen_common_attr_by_lua(
          attr_map={
            "explore_rerank_gen_seed_ensemble_ftr_weight": "explore_rerank_gen_seed_ensemble_ftr_weight * share_pull_ftr_adjust_coef",
          }
        ) \
      .end_() \
      .if_("explore_rerank_enable_user_active_days_weight_adjust == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rerank_gen_seed_ensemble_user_positive_action_photo_similary_score_weight", "as": "xtr_weight"},
            {"name": "uExploreActiveDays", "as": "user_vv"},
            {"name": "explore_rerank_ensemble_power_active_days_weight_adjust_exp_upper", "as": "exp_upper"},
            {"name": "explore_rerank_ensemble_power_active_days_weight_adjust_alpha", "as": "alpha"},
            {"name": "explore_rerank_ensemble_power_active_days_weight_adjust_beta", "as": "beta"},
            {"name": "explore_rerank_ensemble_power_active_days_weight_adjust_omega", "as": "omega"},
            {"name": "explore_rerank_ensemble_power_active_days_weight_adjust_max", "as": "coeff_max"},
            {"name": "explore_rerank_ensemble_power_active_days_weight_adjust_min", "as": "coeff_min"},
          ],
          export_common_attr = [
            {"name": "xtr_weight", "as": "explore_rerank_gen_seed_ensemble_user_positive_action_photo_similary_score_weight"},
          ],
          function_name = "AdjustWeightByUserVv",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_rerank_enable_request_pxtr_power_weight_adjust == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rerank_ctr_power_adjust_weight", "as": "xtr_weight"},
            {"name": "explore_recent_valid_click_count", "as": "user_vv"},
            {"name": "explore_rerank_ensemble_ctr_power_weight_adjust_exp_upper", "as": "exp_upper"},
            {"name": "explore_rerank_ensemble_ctr_power_weight_adjust_alpha", "as": "alpha"},
            {"name": "explore_rerank_ensemble_ctr_power_weight_adjust_beta", "as": "beta"},
            {"name": "explore_rerank_ensemble_ctr_power_weight_adjust_omega", "as": "omega"},
            {"name": "explore_rerank_ensemble_ctr_power_weight_adjust_max", "as": "coeff_max"},
            {"name": "explore_rerank_ensemble_ctr_power_weight_adjust_min", "as": "coeff_min"},
          ],
          export_common_attr = [
            {"name": "xtr_weight", "as": "explore_rerank_ctr_power_adjust_weight"},
          ],
          function_name = "AdjustWeightByUserVv",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_rerank_enable_user_recent_hate_count_power_weight_adjust == 1 and recent_hate_count > explore_koc_htr_count_threshold") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rerank_gen_seed_ensemble_reverse_koc_cover_htr_weight", "as": "xtr_weight"},
            {"name": "recent_hate_count", "as": "user_vv"},
            {"name": "explore_rerank_ensemble_reverse_koc_cover_htr_power_weight_adjust_exp_upper", "as": "exp_upper"},
            {"name": "explore_rerank_ensemble_reverse_koc_cover_htr_power_weight_adjust_alpha", "as": "alpha"},
            {"name": "explore_rerank_ensemble_reverse_koc_cover_htr_power_weight_adjust_beta", "as": "beta"},
            {"name": "explore_rerank_ensemble_reverse_koc_cover_htr_power_weight_adjust_omega", "as": "omega"},
            {"name": "explore_rerank_ensemble_reverse_koc_cover_htr_power_weight_adjust_max", "as": "coeff_max"},
            {"name": "explore_rerank_ensemble_reverse_koc_cover_htr_power_weight_adjust_min", "as": "coeff_min"},
          ],
          export_common_attr = [
            {"name": "xtr_weight", "as": "explore_rerank_gen_seed_ensemble_reverse_koc_cover_htr_weight"},
          ],
          function_name = "AdjustWeightByUserVv",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rerank_gen_seed_ensemble_reverse_koc_detail_htr_weight", "as": "xtr_weight"},
            {"name": "recent_hate_count", "as": "user_vv"},
            {"name": "explore_rerank_ensemble_reverse_koc_detail_htr_power_weight_adjust_exp_upper", "as": "exp_upper"},
            {"name": "explore_rerank_ensemble_reverse_koc_detail_htr_power_weight_adjust_alpha", "as": "alpha"},
            {"name": "explore_rerank_ensemble_reverse_koc_detail_htr_power_weight_adjust_beta", "as": "beta"},
            {"name": "explore_rerank_ensemble_reverse_koc_detail_htr_power_weight_adjust_omega", "as": "omega"},
            {"name": "explore_rerank_ensemble_reverse_koc_detail_htr_power_weight_adjust_max", "as": "coeff_max"},
            {"name": "explore_rerank_ensemble_reverse_koc_detail_htr_power_weight_adjust_min", "as": "coeff_min"},
          ],
          export_common_attr = [
            {"name": "xtr_weight", "as": "explore_rerank_gen_seed_ensemble_reverse_koc_detail_htr_weight"},
          ],
          function_name = "AdjustWeightByUserVv",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_pic_explore_flag == 1 and explore_rerank_enable_pic_explore_weight_adjust == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rerank_gen_seed_ensemble_fr_pic_ensemble_score_weight", "as": "origin_value"},
            {"name": "explore_rerank_pic_explore_raw_weight_adjust_param", "as": "weight"},
          ],
          export_common_attr = [
            {"name": "new_weight", "as": "explore_rerank_gen_seed_ensemble_fr_pic_ensemble_score_weight"}
          ],
          function_name = "AdjustPicQueueWeight",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .gen_common_attr_by_lua(
        attr_map = {
          "enable_rerank_first_refresh_good_photo_diversity":
            "is_first_refresh == 1 and enable_rerank_first_refresh_good_photo_diversity"
        }
      ) \
      .if_("is_growth_reflux ~= nil and is_growth_reflux == 1") \
        .gen_common_attr_by_lua(
          attr_map = {
            "enable_rerank_sexually_photo_diversity":
              "explore_rerank_enable_growth_reflux_sexually_diversity"
          }
        ) \
      .end_()
    self.limit_explore_rerank_es_weight_bound(# 必须放最后，统一调整 es 队列权重上下界
    )
    return self

  def sequence_generator(self):
    return self.flow \
      .split_string(
        input_common_attr="explore_rerank_pic_fixed_slot_config",
        output_common_attr="pic_fixed_slot_conf_list",
        delimiters=";",
      ) \
      .if_("explore_dpp_enable_es_score_hetu_cal == 1") \
        .pack_item_attr(
          item_source={
            "reco_results": True,
          },
          mappings=[
            {
              "aggregator": "avg",
              "from_item_attr": "ctr",
              "to_common_attr": "avg_click_dev"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "ltr",
              "to_common_attr": "avg_like_dev"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "wtr",
              "to_common_attr": "avg_follow_dev"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "ftr",
              "to_common_attr": "avg_forward_dev"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "cltr",
              "to_common_attr": "avg_collect_dev"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "cmtr",
              "to_common_attr": "avg_comment_dev"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "awesome_wtd_score",
              "to_common_attr": "avg_awesome_wtd_score_dev"
            },
          ]
        ) \
      .end_() \
      .if_("enable_open_infer_uvctr_handle_variety == 1", to_be_delete = "date=2024-05-29;committer=fengjingping") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "infer_uv_ctr",
            "infer_uv_ctr_high_threshold",
            "hetu2_adjust_formula_alpha",
            "hetu2_adjust_formula_beta",
            "hetu2_adjust_formula_omega",
            "hetu1_adjust_formula_alpha",
            "hetu1_adjust_formula_beta",
            "hetu1_adjust_formula_omega",
            {"name": "rerank_variety_shuanglie_max13", "as": "rerank_variety_shuanglie_max13_origin"},
            {"name": "rerank_variety_shuanglie_max8", "as": "rerank_variety_shuanglie_max8_origin"},
          ],
          export_common_attr = [
            {"name": "rerank_variety_shuanglie_max13_refactor", "as": "rerank_variety_shuanglie_max13"},
            {"name": "rerank_variety_shuanglie_max8_refactor", "as": "rerank_variety_shuanglie_max8"},
          ],
          function_name = "AdjustDiversityByInferUvCtr",
          class_name = "ExploreLightFunctionSetV2",
          ) \
      .end_() \
      .if_("enable_open_infer_uvctr_handle_variety_for_hetu_version2 == 1", to_be_delete = "date=2024-05-29;committer=fengjingping") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "infer_uv_ctr",
            {"name": "refactor_infer_uv_ctr_high_threshold", "as": "infer_uv_ctr_high_threshold"},
            {"name": "refactor_hetu2_adjust_formula_alpha", "as": "hetu2_adjust_formula_alpha"},
            {"name": "refactor_hetu2_adjust_formula_beta", "as": "hetu2_adjust_formula_beta"},
            {"name": "refactor_hetu2_adjust_formula_omega", "as": "hetu2_adjust_formula_omega"},
            {"name": "refactor_hetu1_adjust_formula_alpha", "as": "hetu1_adjust_formula_alpha"},
            {"name": "refactor_hetu1_adjust_formula_beta", "as": "hetu1_adjust_formula_beta"},
            {"name": "refactor_hetu1_adjust_formula_omega", "as": "hetu1_adjust_formula_omega"},
            {"name": "rerank_variety_shuanglie_max13", "as": "rerank_variety_shuanglie_max13_origin"},
            {"name": "rerank_variety_hetu_one_max_num", "as": "rerank_variety_shuanglie_max8_origin"},
          ],
          export_common_attr = [
            {"name": "rerank_variety_shuanglie_max13_refactor", "as": "rerank_variety_shuanglie_max13"},
            {"name": "rerank_variety_shuanglie_max8_refactor", "as": "rerank_variety_hetu_one_max_num"},
          ],
          function_name = "AdjustDiversityByInferUvCtr",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_explore_hierarchical_priority_insert_tag == 1") \
        .explore_hierarchical_priority_insert_tag_enricher(
          queues = [
            {"name": "is_follow_author", "weight_attr": "explore_tag_signal_follow_weight", "photo_source_type": 100},
            {"name": "eyeshot_source", "weight_attr": "explore_tag_signal_eyeshot_source_weight", "photo_source_type": 101},
            {"name": "is_high_value_author", "weight_attr": "explore_tag_signal_high_value_weight", "photo_source_type": 102},
            {"name": "is_long_view_author", "weight_attr": "explore_tag_signal_always_click_weight", "photo_source_type": 103},
            {"name": "is_recommend_by_friend", "weight_attr": "explore_tag_signal_friend_recommendation_weight", "photo_source_type": 104},
            {"name": "is_hetu_memory_rank_retrieval", "weight_attr": "explore_tag_signal_hetu_memory_rank_retrieval_weight", "photo_source_type": 105},
          ],
          enable_only_cold_start = "{{enable_explore_only_cold_start_hierarchical_priority_insert_tag}}",
          enable_only_first_page_show = "{{enable_explore_only_first_page_hierarchical_priority_insert_tag}}",
          enable_only_is_zero_play = "{{enable_explore_only_is_zero_play_hierarchical_priority_insert_tag}}",
          is_cold_start = "{{return refreshTimes == 0}}",
          is_first_page_show = "{{return page_index == 1}}",
          is_zero_play = "{{return zero_visit_gap == 0}}",
          save_tag_to_attr = "photo_source_type",
          seek_num = "{{explore_hierarchical_priority_insert_tag_seek_num}}"
        ) \
      .end_() \
      .if_("enable_explore_hetu_beauty_cluster_id == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "hetu_beauty_cluster_id_list", "as": "attr_list"}
          ],
          import_item_attr = [
            {"name": "hetu_sim_cluster_id", "as": "attr"}
          ],
          export_item_attr = [
            {"name": "is_in_set", "as": "is_hetu_beauty_cluster_id"}
          ],
          function_name = "AttrIsInSet",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_rerank_enable_adjust_xtr_playtime_map == 1", to_be_delete = "date=2024-05-29;committer=caoying03") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "explore_rr_ctr_list",
            "explore_rr_map_pevtr_list"
          ],
          import_item_attr = [
            "pctr",
          ],
          export_item_attr = [
            "ctr_evtr_map_val"
          ],
          function_name = "GenXtrPlayMapValue",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_adjust_rerank_dpp_diversity_weight == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "user_info_ptr",
            {"name": "enable_rerank_calcu_hetu_by_click_behavior", "as": "enable_utilize_click_action"},
            "action_once_hetu_threshold",
            "hetu_cnt_pow_weight",
            "dpp_diversity_rank_theta",
            "enable_limited_hetu_action_timeliness",
            "action_time_scope_threshold"
          ],
          export_common_attr = [
            {"name": "dpp_diversity_score", "as": "dpp_diversity_rank_theta"}
          ],
          function_name = "CalcDppDiversityIntensity",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_enable_user_need_break_cocoon_rerank == 1 and user_need_break_cocoon_flag == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "dpp_diversity_rank_theta", "as": "value"},
            {"name": "dpp_diversity_rank_theta_user_need_break_cocoon_adjust_coef", "as": "weight"},
          ],
          export_common_attr = [
            {"name": "new_value", "as": "dpp_diversity_rank_theta"},
          ],
          function_name = "CalExploreDoubleMultiDouble",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()

  def dpp_gen_sequence_add_prev_items(self, target_item = {}, prev_items_from_attr="explore_recent_play_list"):
    self.flow.if_("explore_rerank_pic_seed_random_escore_num ~= nil and explore_rerank_pic_seed_random_escore_num > 0")
    self.pic_seed_random_es_score(target_item)
    self.flow.end_()

    return  self.flow \
      .dpp_gen_sequence(
        max_sequence_num = "{{dpp_diversity_max_sequence_num}}",
        return_item_type = 3,
        is_explore = "{{dpp_use_weight_is_explore}}",
        queues = gen_seed_ensemble_queues_dpp(),
        embedding_service_name = "{{dpp_diversity_embedding_service_name}}",
        dpp_diversity_shard_num = "{{dpp_diversity_embedding_shard_num}}",
        embedding_slot_id = "{{dpp_diversity_embedding_slot_id}}",
        embedding_sign_format = "{{dpp_diversity_embedding_sign_format}}",
        embedding_timeout_ms = "{{dpp_diversity_embedding_timeout_ms}}",
        embedding_format = "{{dpp_diversity_embedding_format}}",
        keep_pre_size = "{{fr_rerank_keep_pre_size}}",
        the_temperature = "{{fr_rerank_proportion_temperature}}",
        use_power_rank = "{{fr_rerank_use_power_rank}}",
        use_proportion = "{{explore_rerank_dpp_use_proportion}}",
        the_temperature_addAndMul = "{{fr_rerank_proportion_temperature_addAndMul}}",
        use_power_rank_addAndMul = "{{fr_rerank_use_power_rank_addAndMul}}",
        use_proportion_addAndMul = "{{explore_rerank_dpp_use_proportion_addAndMul}}",
        use_sigmoid = "{{explore_rerank_enable_es_use_sigmoid}}",
        sigmoid_beta = "{{explore_rerank_es_sigmoid_beta}}",
        sigmoid_gamma = "{{explore_rerank_es_sigmoid_gamma}}",
        use_combine = "{{explore_rerank_enable_es_use_combine}}",
        combine_smooth = "{{explore_rerank_es_combine_smooth}}",
        combine_alpha = "{{explore_rerank_es_combine_alpha}}",
        sequence_num_multiple = "{{explore_rerank_dpp_sequence_num_multiple}}",
        diversity_list_size = "{{dpp_diversity_list_size}}",
        rank_theta = "{{dpp_diversity_rank_theta}}",
        dm_epsilon = "{{dpp_diversity_dm_epsilon}}",
        enable_dpp = "{{enable_dpp_diversity_new}}",
        enable_slot_num_minus_offset = "{{explore_enable_slot_num_minus_offset}}",
        filter_red_vertical_num = "{{explore_rerank_filter_red_vertical_num}}",
        enable_skip_sin_que = "{{explore_enable_skip_sin_que}}",
        enable_dpp_use_ssd = "{{enable_use_ssd_list_filter}}",
        enable_ssd_filter_skip_sin = "{{enable_ssd_filter_skip_sin}}",
        final_cnt = "{{fr_rerank_ssd_final_num}}",
        enable_relate_score_org = "{{explore_dpp_enable_relate_score_org}}",
        enable_relate_score_ensemble = "{{explore_dpp_enable_relate_score_ensemble}}",
        enable_relate_score_ltr = "{{explore_dpp_enable_relate_score_ltr}}",
        enable_ensemble_hetu_score = "{{explore_dpp_enable_ensemble_hetu_score}}",
        ensemble_hetu_score_method = "{{explore_dpp_ensemble_hetu_score_method}}",
        top_rank_threshold = "{{explore_dpp_top_rank_threshold}}",
        duration_threshold = "{{explore_dpp_duration_threshold}}",
        enable_ensemble_hetu_cal = "{{explore_dpp_enable_ensemble_hetu_cal}}",
        enable_es_score_hetu_cal = "{{explore_dpp_enable_es_score_hetu_cal}}",
        hetu_emsemble_attr = rerank_hetu_ensemble_queues(),
        enable_dpp_diversity_div_que = "{{enable_dpp_diversity_div_que}}",
        user_info_ptr_attr = "user_info_ptr",
        diversity_history_size = "{{dpp_diversity_history_size}}",
        diversity_queue_name = "diversity",
        # use rank or value
        enable_rank_value_detach = "{{explore_rerank_enable_rank_value_detach}}",
        # rerank action boost
        emp_action_score_attr = "emp_action_score",
        emp_action_score_max = "{{ctr_empirical_action_emp_action_score_max}}",
        emp_action_score_avg = "{{ctr_empirical_action_emp_action_score_avg}}",
        emp_action_bias_score_weight = "{{explore_rerank_emp_action_bias_score_weight}}",
        emp_action_top_num = "{{explore_rerank_emp_action_top_num}}",
        emp_action_threshold = "{{explore_rerank_emp_action_threshold}}",
        enable_emp_action_score = "{{explore_rerank_enable_emp_action_score}}",
        # rerank generation model
        rerank_gen_model_attr="explore_rerank_gen_score",
        enable_dpp_gen_model="{{explore_enable_dpp_gen_model}}",
        rerank_gen_model_beam_size="{{explore_rerank_gen_model_beam_size}}",
        # filter set
        use_set_filter = "{{explore_dpp_use_set_filter}}",
        set_max_filter_cnt = "{{explore_dpp_set_max_filter_cnt}}",
        related_score_power_weight = "{{explore_dpp_related_score_power_weight}}",
        dpp_beam_size = "{{explore_dpp_beam_size}}",
        rank_score_type = "{{explore_dpp_rank_score_type}}",
        cluster_id_attr = "hetu_tag_level_info__hetu_level_one",
        # matrix_combo
        enale_dpp_sim_matrix_norm = "{{explore_enale_dpp_sim_matrix_norm}}",
        enable_sim_matrix_combo = "{{explore_enable_sim_matrix_combo}}",
        sim_matrix_alpha = "{{explore_rerank_dpp_sim_matrix_alpha}}",
        dpp_emb_attr_name = "explore_dpp_emb",
        dpp_emb_dim = "{{explore_rerank_dpp_sim_matrix_dim}}",
        # pxtr_sim_matrix_combo
        enable_pxtr_based_sim_matrix = "{{explore_enable_pxtr_based_sim_matrix}}",
        enable_pxtr_matrix_cliff = "{{explore_enable_pxtr_matrix_cliff}}",
        pxtr_sim_matrix_alpha = "{{explore_pxtr_sim_matrix_alpha}}",
        pxtr_matrix_row_norm = "{{explore_pxtr_matrix_row_norm}}",
        pxtr_matrix_col_norm = "{{explore_pxtr_matrix_col_norm}}",
        pxtr_matrix_col_zero_centered = "{{explore_pxtr_matrix_col_zero_centered}}",
        sim_matrix_pxtr_emb = rerank_pxtr_combo_queues(),
        # theta personalized
        personalized_param_type = "{{explore_rerank_personalized_param_type}}",
        personalized_base = "{{explore_rerank_personalized_base}}",
        personalized_min = "{{explore_rerank_personalized_min}}",
        personalized_max = "{{explore_rerank_personalized_max}}",
        personalized_score_attr = "explore_colossus_hetu_personalized_score",
        personalized_timely_param_type = "{{explore_rerank_personalized_timely_param_type}}",
        personalized_timely_score_attr = "explore_user_timely_diversity_entropy_score",
        personalized_hetu_top1_param_type = "{{explore_rerank_personalized_hetu_top1_param_type}}",
        personalized_hetu_top1_attr = "explore_rerank_user_timely_diversity_weight",
        # ssd gen list
        enable_ssd_gen_list = "{{explore_enable_ssd_gen_list}}",
        enable_ori_gen_list = "{{explore_enable_ori_gen_list}}",
        ssd_diversity_factor = "{{explore_ssd_diversity_factor}}",
        ssd_score_fusion_type = "{{explore_ssd_score_fusion_type}}",
        ssd_rank_score_weight = "{{explore_ssd_rank_score_weight}}",
        # filter list by hutu1 num
        enable_hetu1_filter = "{{explore_rerank_enable_bottom_filtration_through_hetu1}}",
        hetu1_filter_num = "{{explore_rerank_bottom_filtration_through_hetu1_num}}",
        # discrete_space
        dpp_dynamic_action_space = "{{explore_rerank_dpp_dynamic_action_space}}",
        enable_discrete_action_space = "{{explore_rerank_enable_discrete_action_space}}",
        # matrix exp
        enable_sim_matrix_exp = "{{explore_dpp_enable_sim_matrix_exp}}",
        matrix_exp_param = "{{explore_dpp_matrix_exp_param}}",
        # multiply score
        sequence_num_for_multiply = "{{explore_rerank_sequence_num_for_multiply}}",
        multiply_use_power_rank = "{{explore_rerank_multiply_use_power_rank}}",
        enable_raw_weight_random = "{{explore_rerank_multiply_enable_raw_weight_random}}",
        enable_raw_pow_weight_random = "{{explore_rerank_multiply_enable_raw_pow_weight_random}}",
        # random replace
        enable_random_replace_topk = "{{explore_enable_random_replace_topk}}",
        random_replace_topk = "{{explore_random_replace_topk}}",
        # hetu1 limit
        enable_max_hetu1_dpp_regular = "{{explore_enable_max_hetu1_dpp_regular}}",
        max_hetu1_dpp_regular_num = "{{explore_max_hetu1_dpp_regular_num}}",
        # session hetu1 limit
        hetu_history_size = "{{explore_session_hetu_history_size}}",
        enable_session_hetu1_dpp_regular = "{{explore_enable_session_hetu1_dpp_regular}}",
        session_hetu1_dpp_regular_num = "{{explore_session_hetu1_dpp_regular_num}}",
        realshow_page_type = 1,
        # theta bias
        enable_theta_random = "{{explore_enable_theta_random}}",
        theta_bias_range = "{{explore_theta_bias_range}}",
        # que_tail_discount
        enable_que_tail_discount = "{{explore_enable_que_tail_discount}}",
        que_tail_discount_threshold = "{{explore_que_tail_discount_threshold}}",
        que_tail_discount_coef = "{{explore_que_tail_discount_coef}}",
        que_tail_discount_min = "{{explore_que_tail_discount_min}}",
        que_tail_boost_threshold = "{{explore_que_tail_boost_threshold}}",
        enable_que_tail_adaptive_discount = "{{explore_enable_que_tail_adaptive_discount}}",
        enable_que_tail_adaptive_boost = "{{explore_enable_que_tail_adaptive_boost}}",
        # relate score
        use_rank_div = "{{explore_relate_score_use_rank_div}}",
        related_score_smooth = "{{explore_relate_score_rank_div_smooth}}",
        duration_attr = "duration_ms",
        predict_play_time_attr = "awesome_wtd",
        # multiply
        use_multiply = "{{explore_dpp_ensemble_sort_use_multiply}}",
        # 打散相关
        enable_new_variety_engineer = "{{enable_new_variety_engineer}}",
        max_satisfied_pick="{{variety_engineer_slot_num_shuanglie}}",
        rules = dpp_variant_rules(),
        smooth_num = "{{rerank_smooth_num}}",
        action_day = "{{rerk_collect_queue_boost_active_day_num}}",
        use_div_prefer = "{{explore_dpp_use_div_prefer_cal}}",
        div_lower_bound = "{{explore_dpp_div_lower_bound}}",
        div_upper_bound = "{{explore_dpp_div_upper_bound}}",
        div_bias = "{{explore_dpp_div_bias}}",
        prev_items_from_attr = prev_items_from_attr,
        # 图文混排相关
        enable_pic_mix_generator = "{{explore_enable_pic_mix_generator}}",
        mix_score_attr = "mix_ensemble_score",
        picture_attr = "is_picture",
        pic_score_attr = "fr_pic_ensemble_score",
        top_slot = "{{dpp_mix_rerank_top_slot}}",
        min_gap = "{{dpp_mix_rerank_min_gap}}",
        enable_dynamic_pic_min_gap = "{{explore_enable_dynamic_pic_min_gap}}",
        pic_quota_attr = "dynamic_pic_quota",
        enable_pic_fixed_slots = "{{explore_rerank_enable_pic_fixed_slots}}",
        pic_fixed_slot_conf_attr = "pic_fixed_slot_conf_list",
        pic_fix_slot_skip_variety = "{{explore_rerank_pic_fix_slot_skip_variety}}",
        is_fresh_request_attr = "is_fresh_request",
        enable_fresh_request_dynamic_config = "{{enable_fresh_request_dynamic_config}}",
        fresh_request_pic_fixed_slot_conf_attr = "fresh_fixed_slot_conf_list",
        # dpp 前插入图文
        enable_pic_mix_insertion = "{{explore_enable_pic_mix_insertion}}",
        mix_insert_num_limit = "{{explore_mix_insert_num_limit}}",
        mix_insert_range_end = "{{explore_mix_insert_range_end}}",
        mix_insert_score_attr = "corr_pctr",
        mix_insert_pic_boost_coef = "{{explore_mix_insert_pic_boost_coef}}",
        pic_mix_insertion_skip_single_pic = "{{explore_pic_mix_insertion_skip_single_pic}}",
        picture_type_attr = "picture_type",
        # dpp 后插入图文
        pics_to_insert_after_dpp = "{{pic_list_to_insert_after_rerank_dpp}}",
        fixed_slots_after_dpp = "{{explore_pic_fixed_slots_after_dpp}}",
        # dpp 后做图文uv探索
        enable_pic_explore = "{{explore_pic_interest_explore__enable}}",
        item_key_for_pic_explore = "{{item_key_for_pic_explore}}",
        pic_explore_flag = "{{enable_pic_explore_flag}}",
        pic_explore_insert_pos_min = "{{pic_explore_insert_pos_min}}",
        pic_explore_insert_pos_max = "{{pic_explore_insert_pos_max}}",
        enable_pic_explore_pos_move_forward = "{{enable_pic_explore_pos_move_forward}}",
        # dpp 后做图文兴趣探索
        enable_pic_interest_explore = "{{enable_picture_interest_explore_rerank}}",
        item_key_for_pic_interest_explore = "{{target_pid_for_pic_interest_explore}}",
        pic_interest_explore_cluster_id_attr = "hetu_tag_level_info__hetu_level_one",
        pic_interest_explore_cluster_id_list_attr = "pic_interest_explore_hetu_list",
        # dpp 后做图文兴趣替换
        enable_pic_interest_expand = "{{explore_rerank_enable_pic_interest_expand}}",
        item_key_for_pic_interest_expand = "{{item_key_for_pic_interest_expand}}",
        # 单队列相关
        enable_new_single_queues = "{{explore_enable_new_single_queues}}",
        single_queues = single_queues(),
        range_end = "{{dpp_diversity_candidate_size}}",
        target_item = target_item,
        # 图文/视频 embedding
        embedding_orthogonal_method = "{{explore_embedding_orthogonal_method}}",
        embedding_orthogonal_bias = "{{explore_embedding_orthogonal_bias}}",
        # kl score 融合
        enable_get_user_longterm_interest = "{{explore_enable_get_user_longterm_interest}}",
        enable_get_session_hetu1 = "{{explore_enable_get_session_hetu1}}",
        enable_cal_kl_score = "{{enable_cal_kl_score}}",
        kl_fusion_real_show_size_max_threshold = "{{kl_fusion_real_show_size_max_threshold}}",
        kl_max_threshold = "{{explore_rerank_kl_max_threshold}}",
        enable_kl_fusion_real_show_hetu_cnt = "{{enable_kl_fusion_real_show_hetu_cnt}}",
        hetu_rate_min_threshold = "{{hetu_rate_min_threshold}}",
        kl_score_smooth_alpha = "{{kl_score_smooth_alpha}}",
        real_show_history_size_min_threshold = "{{real_show_history_size_min_threshold}}",
        real_show_unique_hetu_min_threshold = "{{real_show_unique_hetu_min_threshold}}",
        kl_score_power_weight = "{{kl_score_power_weight}}",
        enabl_kl_fusion_add_sigmod = "{{enabl_kl_fusion_add_sigmod}}",
        user_hetu_stat_attr = "colossus_hetu_distribution_hetu_stat",
        # 营销号代偿打压
        enable_marketing_compensation_photo_discount = "{{explore_enable_rerank_marketing_compensation_photo_discount}}",
        marketing_compensation_photo_discount_coeff = "{{explore_rerank_marketing_compensation_photo_discount_coeff}}",
        marketing_compensation_photo_attr = "is_marketing_compensation_photo",
        # 图文 es 内部替换实验
        enable_replace_pic_es_score_by_pic_rank = "{{explore_enable_replace_pic_es_score_by_pic_rank}}",
        replace_pic_es_score_by_pic_rank_item_prob = "{{explore_replace_pic_es_score_by_pic_rank_item_prob}}",
        replace_pic_es_score_by_pic_rank_seq_prob = "{{explore_replace_pic_es_score_by_pic_rank_seq_prob}}",
        # 新增外部 enrich 的扰动 es 分
        random_es_scores_num_list = ["pic_seed_random_es_scores_num"],
        random_es_scores_list = ["pic_seed_random_es_scores"],
      )

  def int_value_adjust(self, int_value_name, strategy_name):
    adjust_coef = int_value_name + "_" + strategy_name + "_adjust_coef"
    self.flow \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": adjust_coef, "as": "weight"},
          {"name": int_value_name, "as": "value"}
        ],
        export_common_attr = [
          {"name": "new_value", "as": int_value_name}
        ],
        function_name = "CalExploreIntMultiDouble",
        class_name = "ExploreLightFunctionSetV2",
      )
    return self

  def multi_int_value_adjust(self, int_value_name_list, strategy_name):
    for int_value_name in int_value_name_list:
      self.int_value_adjust(int_value_name, strategy_name)
    return self

  def pic_interest_explore(self):
    return self.flow \
      .if_("explore_pic_interest_explore__enable == 1 and enable_pic_explore_flag == 1 and util.Random() < pic_uv_explore_ratio_thd") \
        .enrich_attr_by_light_function(
          import_common_attr=[
            {"name": "video_calc_score_method_for_pic_explore", "as": "score_method"},
            {"name": "video_top_num_for_pic_explore", "as": "top_num"},
            {"name": "video_pxtr_weight_config_str_for_pic_explore", "as": "xtr_weight_config_str"},
          ],
          import_item_attr=[
            "corr_pctr",
            "pltr",
            "pwtr",
            "pcltr",
            "fetr",
            "awesome_wtd"
          ],
          export_common_attr=[
            {"name": "viceo_topk_avg", "as": "fr_viceo_topk_avg"},
          ],
          function_name="CalcVideoTopkScoreAvg",
          class_name="ExploreLightFunctionSetV2",
          target_item={"is_picture": 0},
        ) \
        .switch_("picture_type_select_mode_for_pic_uv") \
          .case_(1) \
            .enrich_attr_by_light_function(  # 长图 & 图集
              import_common_attr=[
                {"name": "pic_fr_rel_score_pct_map", "as": "pic_score_pct_map"},
                {"name": "fr_viceo_topk_avg", "as": "viceo_topk_avg"},
                {"name": "pic_pct_data_key_for_pic_explore", "as": "pct_data_key"},
                {"name": "pic_thd_pct_milli_for_pic_explore", "as": "thd_pct_milli"},
                {"name": "pic_range_end_for_pic_explore", "as": "range_end"},
                {"name": "pic_calc_score_method_for_pic_explore", "as": "score_method"},
                {"name": "video_topk_pow_weight_for_pic_explore", "as": "viceo_topk_pow"},
                {"name": "base_score_xtr_coeff_for_pic_explore", "as": "xtr_coeff_for_pic_explore"},
                {"name": "pic_pxtr_weight_config_str_for_pic_explore", "as": "xtr_weight_config_str"},
              ],
              import_item_attr=[
                "photo_id",
                "corr_pctr",
                "pltr",
                "pwtr",
                "pcltr",
                "fetr",
                "awesome_wtd"
              ],
              export_common_attr=[
                "item_key_for_pic_explore"
              ],
              function_name="CalcPicQuotaAndPids",
              class_name="ExploreLightFunctionSetV2",
              target_item={"picture_type": [2, 3]},
            ) \
          .default_() \
            .enrich_attr_by_light_function(  # 长图 & 图集 & 单图
              import_common_attr=[
                {"name": "pic_fr_rel_score_pct_map", "as": "pic_score_pct_map"},
                {"name": "fr_viceo_topk_avg", "as": "viceo_topk_avg"},
                {"name": "pic_pct_data_key_for_pic_explore", "as": "pct_data_key"},
                {"name": "pic_thd_pct_milli_for_pic_explore", "as": "thd_pct_milli"},
                {"name": "pic_range_end_for_pic_explore", "as": "range_end"},
                {"name": "pic_calc_score_method_for_pic_explore", "as": "score_method"},
                {"name": "video_topk_pow_weight_for_pic_explore", "as": "viceo_topk_pow"},
                {"name": "base_score_xtr_coeff_for_pic_explore", "as": "xtr_coeff_for_pic_explore"},
                {"name": "pic_pxtr_weight_config_str_for_pic_explore", "as": "xtr_weight_config_str"},
              ],
              import_item_attr=[
                "photo_id",
                "corr_pctr",
                "pltr",
                "pwtr",
                "pcltr",
                "fetr",
                "awesome_wtd"
              ],
              export_common_attr=[
                "item_key_for_pic_explore"
              ],
              function_name="CalcPicQuotaAndPids",
              class_name="ExploreLightFunctionSetV2",
              target_item={"is_picture": 1},
            ) \
        .end_() \
      .else_if_("enable_pic_explore_flag ~= 1 and enable_picture_interest_explore_rerank == 1 and enable_picture_interest_explore == 1 and pic_interest_explore_hetu_list ~= nil and #pic_interest_explore_hetu_list > 0 and util.Random() < pic_interest_explore_ratio_thd") \
        .enrich_attr_by_light_function(
          import_common_attr=[
            {"name": "video_calc_score_method_for_pic_interest_explore", "as": "score_method"},
            {"name": "video_top_num_for_pic_interest_explore", "as": "top_num"},
            {"name": "video_pxtr_weight_config_str_for_pic_interest_explore", "as": "xtr_weight_config_str"}
          ],
          import_item_attr=[
            "corr_pctr",
            "pltr",
            "pwtr",
            "pcltr",
            "fetr",
            "awesome_wtd"
          ],
          export_common_attr=[
            {"name": "viceo_topk_avg", "as": "fr_video_topk_avg"},
          ],
          function_name="CalcVideoTopkScoreAvg",
          class_name="ExploreLightFunctionSetV2",
          target_item={"is_picture": 0},
        ) \
        .switch_("picture_type_select_mode_for_pic_interest_explore") \
          .case_(1) \
            .enrich_attr_by_light_function(  # 长图 & 图集
              import_common_attr=[
                {"name": "pic_interest_explore_hetu_list", "as": "hetu_list"},
                {"name": "fr_video_topk_avg", "as": "video_topk_avg"},
                {"name": "pic_range_end_for_pic_interest_explore", "as": "range_end"},
                {"name": "base_score_xtr_coeff_for_pic_interest_explore", "as": "base_score_coeff"},
                {"name": "pic_pxtr_weight_config_str_for_pic_interest_explore", "as": "xtr_weight_config_str"},
              ],
              import_item_attr=[
                "photo_id",
                "corr_pctr",
                "pltr",
                "pwtr",
                "pcltr",
                "fetr",
                "awesome_wtd",
                "hetu_tag_level_info__hetu_level_one"
              ],
              export_common_attr=[
                "target_pid_for_pic_interest_explore"
              ],
              function_name="GenTargetPidForPictureInterestExplore",
              class_name="ExploreLightFunctionSetV2",
              target_item={"picture_type": [2, 3]},
            ) \
          .default_() \
            .enrich_attr_by_light_function(  # 长图 & 图集 & 单图
              import_common_attr=[
                {"name": "pic_interest_explore_hetu_list", "as": "hetu_list"},
                {"name": "fr_video_topk_avg", "as": "video_topk_avg"},
                {"name": "pic_range_end_for_pic_interest_explore", "as": "range_end"},
                {"name": "base_score_xtr_coeff_for_pic_interest_explore", "as": "base_score_coeff"},
                {"name": "pic_pxtr_weight_config_str_for_pic_interest_explore", "as": "xtr_weight_config_str"},
              ],
              import_item_attr=[
                "photo_id",
                "corr_pctr",
                "pltr",
                "pwtr",
                "pcltr",
                "fetr",
                "awesome_wtd",
                "hetu_tag_level_info__hetu_level_one"
              ],
              export_common_attr=[
                "target_pid_for_pic_interest_explore"
              ],
              function_name="GenTargetPidForPictureInterestExplore",
              class_name="ExploreLightFunctionSetV2",
              target_item={"is_picture": 1},
            ) \
        .end_() \
      .end_() \

  def limit_explore_rerank_es_weight_bound(self):
    self.flow \
      .if_("explore_rerank_gen_seed_ensemble_fr_pic_final_weight_upbound_limit == 1") \
        .gen_common_attr_by_lua(
          attr_map = {
            "explore_rerank_gen_seed_ensemble_fr_pic_ensemble_score_weight" : "math.min(explore_rerank_gen_seed_ensemble_fr_pic_ensemble_score_weight, explore_rerank_gen_seed_ensemble_fr_pic_weight_upbound)",
            "explore_rerank_gen_seed_ensemble_fr_pic_ensemble_score_raw_weight" : "math.min(explore_rerank_gen_seed_ensemble_fr_pic_ensemble_score_raw_weight, explore_rerank_gen_seed_ensemble_fr_pic_raw_weight_upbound)",
            "explore_rerank_gen_seed_ensemble_fr_pic_ensemble_score_raw_pow_weight" : "math.min(explore_rerank_gen_seed_ensemble_fr_pic_ensemble_score_raw_pow_weight, explore_rerank_gen_seed_ensemble_fr_pic_raw_pow_weight_upbound)",
          }
        ) \
      .end_()

  def random_es_score_generate_queue_configs(self, param_prefix, queues):
    queue_configs = []
    param_prefix = "{{" + param_prefix
    configs = [
      "enable",
      "rank_reverse_order",
      "rank_pow",
      "rank_smooth",
      "raw_alpha",
      "raw_bias",
      "raw_pow",
      "queue_pow",
      "queue_pow_range",
    ]
    for q in queues:
      q_config = {"name": q}
      for config in configs:
        q_config[config] = param_prefix + q + "_" + config + "}}"
      queue_configs.append(q_config)
    return queue_configs

  def pic_seed_random_es_score(self, target_item):
    """
    !!! 该 function 仅限图文组修改 !!!
    图文 dpp es 序列补充召回, 使用与原 dpp(dpp_gen_sequence) 不同的 seed 来得到不同比重的 es 分来生成序列, 新增的队列参数与公式不可与原 dpp 相同
    """
    # 1. 生成队列参数配置
    queue_configs = self.random_es_score_generate_queue_configs(
      param_prefix = "explore_pic_seed_rerank_random_es_",
      queues = [
        "corr_pctr_psvr",
        "pltr",
        "pcltr",
        "pwtr",
        "pftr",
        "pcmtr",
        "pdtr",
        "plvtr",
        "awesome_wtd",
        "phtr",
        "fetr",
        "fr_pic_ensemble_score",
    ])

    # 2. 执行 random es 打分
    self.flow.explore_random_es_score_enrich(
      num_es_score = "{{explore_rerank_pic_seed_random_escore_num}}",
      queues = queue_configs,
      output_item_random_es_scores = "pic_seed_random_es_scores",
      output_common_es_score_num = "pic_seed_random_es_scores_num",
      target_item = target_item
    )

  def process(self) -> None:
    self.flow.if_("enable_use_explore_rerank == 1")
    self.enrich_common_attr() # 填充 common attr

    self.flow.if_("enable_full_link_sample_package == 1")

    self.flow \
      .copy_item_meta_info(
        save_item_seq_to_attr = "rank_index_before_rerank",
      ) \

    self.flow.end_() \

     # 图文混排 generator
    self.flow \
      .if_("explore_enable_calc_pic_quota == 1 and explore_enable_insert_pic_after_rerank_dpp == 0 and explore_enable_calc_pic_quota_fixed_load == 0") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_pic_quota_pxtr_attr_config_str_pic", "as": "pxtr_attr_config_str"},
            {"name": "explore_pic_quota_avg_top_num_pic", "as": "avg_top_num"},
            {"name": "explore_pic_rerank_quota_pic_enable_trans", "as": "enable_trans"},
            {"name": "explore_pic_rerank_quota_pic_trans_alpha", "as": "trans_alpha"},
            {"name": "explore_pic_rerank_quota_pic_trans_bias", "as": "trans_bias"},
            {"name": "explore_pic_rerank_quota_pic_trans_pow", "as": "trans_pow"},
            {"name": "explore_pic_rerank_quota_pic_trans_min", "as": "trans_min"},
            {"name": "explore_pic_rerank_quota_pic_trans_max", "as": "trans_max"},
          ],
          export_common_attr = [
            {"name": "pxtr_topn_avg_score", "as": "pxtr_topn_avg_score_pic"},
          ],
          import_item_attr = ["corr_pctr", "pltr", "pwtr", "awesome_wtd", "fetr", "pcmtr", "pcltr", "pdtr", "pftr"],
          function_name = "CalcPxtrStatScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item={ "is_picture": 1 }
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_pic_quota_pxtr_attr_config_str_video", "as": "pxtr_attr_config_str"},
            {"name": "explore_pic_quota_avg_top_num_video", "as": "avg_top_num"},
            {"name": "explore_pic_rerank_quota_vid_enable_trans", "as": "enable_trans"},
            {"name": "explore_pic_rerank_quota_vid_trans_alpha", "as": "trans_alpha"},
            {"name": "explore_pic_rerank_quota_vid_trans_bias", "as": "trans_bias"},
            {"name": "explore_pic_rerank_quota_vid_trans_pow", "as": "trans_pow"},
            {"name": "explore_pic_rerank_quota_vid_trans_min", "as": "trans_min"},
            {"name": "explore_pic_rerank_quota_vid_trans_max", "as": "trans_max"},
          ],
          export_common_attr = [
            {"name": "pxtr_topn_avg_score", "as": "pxtr_topn_avg_score_video"},
          ],
          import_item_attr = ["corr_pctr", "pltr", "pwtr", "awesome_wtd", "fetr", "pcmtr", "pcltr", "pdtr", "pftr"],
          function_name = "CalcPxtrStatScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item={ "is_picture": 0 }
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_pic_quota_prop_recent_weight", "as": "pic_quota_prop_recent_weight"},
            {"name": "explore_pic_quota_prop_weight", "as": "pic_quota_prop_weight"},
            {"name": "explore_pic_quota_max", "as": "pic_quota_max"},
            {"name": "explore_pic_quota_min", "as": "pic_quota_min"},
            {"name": "explore_pic_quota_score_power_weight", "as": "score_power_weight"},
            {"name": "explore_pic_quota_score_power_min_base", "as": "score_power_min_base"},
            {"name": "explore_pic_quota_score_power_max_base", "as": "score_power_max_base"},
            {"name": "explore_pic_quota_score_coef", "as": "score_coef"},
            {"name": "pic_stat_pic_play_cnt", "as": "colossus_pic_cnt"},
            {"name": "pic_stat_video_play_cnt", "as": "colossus_video_cnt"},
            {"name": "explore_pic_quota_single_pic_rm_prob", "as": "pic_quota_single_pic_rm_prob"},
            {"name": "explore_pic_quota_pic_set_add_prob", "as": "pic_quota_pic_set_add_prob"},
            {"name": "explore_pic_quota_long_pic_add_prob", "as": "pic_quota_long_pic_add_prob"},
            {"name": "explore_pic_quota_recent_ctr_score_power", "as": "recent_ctr_score_power"},
            {"name": "explore_pic_prefer_score_rerank_coeff", "as": "pic_prefer_score_coeff"},
            {"name": "explore_pic_prefer_score_rerank_weight", "as": "pic_prefer_score_weight"},
            {"name": "explore_external_user_rerank_pic_quota_up_bound", "as": "rerank_pic_quota_up_bound"},
            "short_term_pic_cnt",
            "short_term_video_cnt",
            "pxtr_topn_avg_score_video",
            "pxtr_topn_avg_score_pic",
            "user_pic_recent_ctr_score",
            "explore_pic_quota_enable_calc_by_prefer_model",
            "dynamic_pic_quota",
            "external_prefer_user_flag",
            # 静音时对图文 boost 的参数
            {"name": "explore_enable_pic_rerank_quota_muted_boost", "as": "enable_muted_boost"},
            {"name": "explore_pic_rerank_quota_muted_boost_coef", "as": "muted_boost_coef"},
            "isMuted",
            {"name": "explore_enable_pic_rerank_quota_recent_search_boost", "as": "enable_recent_search_boost"},
            {"name": "explore_pic_rerank_quota_recent_search_boost_coef", "as": "recent_search_boost_coef"},
            "is_recent_search_user",
            "pic_search_boost_user_degree",
            {"name": "explore_pic_search_boost_user_degree_thresh", "as": "pic_search_boost_user_degree_thresh"},
            # 浅度兴趣人群提权
            {"name": "explore_rerank_enable_pic_low_interest_boost", "as": "enable_low_interest_boost"},
            {"name": "explore_rerank_pic_low_interest_boost_coef", "as": "low_interest_boost_coef"},
            {"name": "uDoubleOutsideValidPicClusterCnt7dKV", "as": "pic_eff_interest_cnt"},
            {"name": "explore_rerank_pic_interest_quota_boost_thresh", "as": "pic_eff_interest_thresh"},
            # 18-无图文曝光用户提权
            "is_low_pic_play_boost_user",
            {"name": "explore_rerank_enable_low_pic_play_quota_boost", "as": "enable_low_pic_play_boost"},
            {"name": "explore_rerank_low_pic_play_quota_boost_coef", "as": "low_pic_play_boost_coef"},
            # 图文首刷用户提权
            "is_pic_first_screen_boost_user",
            {"name": "explore_rerank_enable_pic_first_screen_quota_boost", "as": "enable_first_screen_boost"},
            {"name": "explore_rerank_pic_first_screen_quota_boost_coef", "as": "first_screen_boost_coef"},
            # 按年龄段调节quota
            "explore_pic_quota_enable_adjust_by_age",
            "basic_info_age_segment_v2",
            {"name": "explore_rerank_pic_quota_age_prop_weights", "as": "age_segment_prop_weights"},
          ],
          import_item_attr = [
            "picture_type",
          ],
          export_common_attr = [
            {"name": "pic_quota", "as": "dpp_mix_rerank_pic_queue_size"},
          ],
          function_name = "CalcPicQuotaScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item = { "is_picture": 1 },
        ) \
      .end_() \
      .if_("explore_enable_calc_pic_quota_fixed_load == 1", to_be_delete = "date=2024-05-29;committer=fenglei03") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_pic_quota_pxtr_attr_config_str_pic", "as": "pxtr_attr_config_str"},
            {"name": "explore_pic_quota_avg_top_num_pic", "as": "avg_top_num"},
          ],
          export_common_attr = [
            {"name": "pxtr_topn_avg_score", "as": "pxtr_topn_avg_score_pic"},
          ],
          import_item_attr = ["corr_pctr", "pltr", "pwtr", "awesome_wtd", "fetr"],
          function_name = "CalcPxtrStatScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item = { "picture_type": [2, 3] }
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_pic_quota_pxtr_attr_config_str_video", "as": "pxtr_attr_config_str"},
            {"name": "explore_pic_quota_avg_top_num_video", "as": "avg_top_num"},
          ],
          export_common_attr = [
            {"name": "pxtr_topn_avg_score", "as": "pxtr_topn_avg_score_video"},
          ],
          import_item_attr = ["corr_pctr", "pltr", "pwtr", "awesome_wtd", "fetr"],
          function_name = "CalcPxtrStatScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item={ "is_picture": 0 }
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "pic_rerank_pid_realshow_data_map", "as": "pid_realshow_data_map"},
            {"name": "expl_pic_rerank_long_set_pic_realshow_num_key", "as": "realshow_num_key"},
            {"name": "expl_pic_rerank_long_set_pic_realshow_ratio_key", "as": "realshow_ratio_key"},
            {"name": "expl_pic_rerank_long_set_pic_pid_set_point", "as": "pid_set_point"},
            {"name": "expl_pic_rerank_pid_scale_pow", "as": "scale_pow"},
          ],
          export_common_attr = [
            {"name": "pid_fractions", "as": "long_set_pic_pid_fractions"},
          ],
          function_name = "CalcRerankPicPIDFractions",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .auto_adjust(
          window_size = 1000,
          windows_num = 50,
          history_input_save_mod = "customize",
          fractions_attr = "long_set_pic_pid_fractions",
          adjust_output = "long_set_pic_pid_thd_adjust",
          adjust_function = "pid",
          set_point = "{{expl_pic_rerank_long_set_pic_pid_set_point}}",
          kp = "{{expl_pic_rerank_pid_long_set_pic_kp}}",
          ki = "{{expl_pic_rerank_pid_long_set_pic_ki}}",
          kd = "{{expl_pic_rerank_pid_long_set_pic_kd}}",
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_pic_quota_prop_recent_weight_fixed_load", "as": "pic_quota_prop_recent_weight"},
            {"name": "explore_pic_quota_prop_weight_fixed_load", "as": "pic_quota_prop_weight"},
            {"name": "explore_pic_quota_max_fixed_load", "as": "pic_quota_max"},
            {"name": "explore_pic_quota_min_fixed_load", "as": "pic_quota_min"},
            {"name": "explore_pic_quota_score_power_weight_fixed_load", "as": "score_power_weight"},
            {"name": "explore_pic_quota_score_power_min_base_fixed_load", "as": "score_power_min_base"},
            {"name": "explore_pic_quota_score_power_max_base_fixed_load", "as": "score_power_max_base"},
            {"name": "explore_pic_quota_score_coef_fixed_load", "as": "score_coef"},
            {"name": "pic_stat_pic_play_cnt", "as": "colossus_pic_cnt"},
            {"name": "pic_stat_video_play_cnt", "as": "colossus_video_cnt"},
            {"name": "explore_pic_quota_single_pic_rm_prob", "as": "pic_quota_single_pic_rm_prob"},
            {"name": "explore_pic_quota_pic_set_add_prob", "as": "pic_quota_pic_set_add_prob"},
            {"name": "explore_pic_quota_long_pic_add_prob", "as": "pic_quota_long_pic_add_prob"},
            {"name": "explore_pic_quota_recent_ctr_score_power", "as": "recent_ctr_score_power"},
            {"name": "long_set_pic_pid_thd_adjust", "as": "pid_adjust"},
            {"name": "explore_pic_quota_pid_adjust_method", "as": "pid_adjust_method"},
            {"name": "explore_pic_quota_pid_adjust_limit", "as": "pid_adjust_limit"},
            "short_term_pic_cnt",
            "short_term_video_cnt",
            "pxtr_topn_avg_score_video",
            "pxtr_topn_avg_score_pic",
            "user_pic_recent_ctr_score",
          ],
          import_item_attr = [
            "picture_type",
          ],
          export_common_attr = [
            {"name": "pic_quota", "as": "dpp_mix_rerank_pic_queue_size"},
          ],
          function_name = "CalcPicQuotaScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()
    self.pic_interest_explore()

    self.flow \
      .if_("explore_rerank_enable_add_pic_explore_into_candidate == 1") \
        .pack_common_attr( # 将图文强插的候选都放到这个list中统一处理,新增强插逻辑请务必遵守
          input_common_attrs = ["item_key_for_pic_explore", "target_pid_for_pic_interest_explore"],
          output_common_attr = "pic_explore_item_keys",
        ) \
        .set_attr_value(
          item_attrs = [
            {
              "name": "mix_mark",
              "type": "int",
              "value": 2,
            },
          ],
          item_list_from_attr = "pic_explore_item_keys"
        ) \
        .enrich_attr_by_light_function(
          item_list_from_attr = "pic_explore_item_keys",
          import_item_attr = [
            "mix_mark",
          ],
          function_name = "EmptyFunction",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "dpp_mix_rerank_pic_queue_size", "as": "pic_queue_size"},
          {"name": "dpp_mix_rerank_video_queue_size", "as": "video_queue_size"},
          {"name": "dpp_mix_rerank_single_pic_as_video", "as": "single_pic_as_video"},
          {"name": "dpp_mix_rerank_single_pic_max_num", "as": "single_pic_max_num"},
        ],
        import_item_attr = [
          "is_picture",
          "picture_type",
        ],
        export_item_attr = [
          "mix_mark", # 标记参与混排的 item, 视频为 1, 图片为2.
                      # 当 single_pic_as_video = 1 时, 当前function跳过单图并且由下面的 MarkMixItemSinglePic 专门处理单图
        ],
        function_name = "MarkMixItem",
        class_name = "ExploreLightFunctionSetV2",
        range_end = "{{dpp_mix_rerank_candidate_size}}",
      )

    self.enrich_item_attr( # 抽取dpp需要的 item attr
        target_item = {
          "mix_mark" : [1, 2]
        }
      )
    self.flow \
      .explore_calc_ensemble_score(
        save_score_to_attr = "mix_ensemble_score",
        user_power_calc = 1,
        queues = [
          {
            "name": "corr_pctr",
            "weight": 1.0,
            "power_weight_attr": "dpp_mix_rerank_power_weight_pctr",
          },
          {
            "name": "mix_reward",
            "weight": 1.0,
            "power_weight_attr": "dpp_mix_rerank_power_weight_mix_reward",
          },
          {
            "name": "explore_fr_ensemble_score",
            "weight": 0.0,
            "power_weight_attr": "dpp_mix_rerank_power_weight_explore_fr_ensemble_score",
          },
        ],
        target_item = {
          "mix_mark" : [1, 2]
        }
      )
    self.sequence_generator()
    self.flow.if_("explore_rerank_diversity_enable_standard_explore_realshow_pid_list == 1")
    self.flow \
      .limit(
        size = "{{explore_rerank_diversity_max_keep_realshow_photoid_size}}",
        item_list_from_attr = "standard_explore_realshow_pid_list"
      )
    self.dpp_gen_sequence_add_prev_items(
      target_item = {
        "mix_mark" : [1, 2]
      },
      prev_items_from_attr = "standard_explore_realshow_pid_list"
    )
    self.flow.else_()
    self.dpp_gen_sequence_add_prev_items(
      target_item = {
        "mix_mark" : [1, 2]
      },
      prev_items_from_attr = "explore_recent_play_list"
    )
    self.flow.end_()

    self.flow \
      .pack_common_attr(
        input_common_attrs = ["retrieval_list_keys_6", "retrieval_list_keys_7"],
        output_common_attr = 'retrieval_list_keys',
        deduplicate = True
      ) \
      .switch_("enable_use_new_rerank_features") \
        .case_(0) \
          .if_("enable_skip_old_list_fea == 0", to_be_delete = "date=2024-05-29;committer=xuwei09") \
            .list_wise_seq_attr(
              item_attrs_transform_map = rerank_features_new,
              seq_item_attr_name = "generated_diversity_lists",
              item_list_from_attr = "retrieval_list_keys"
            ) \
          .end_() \
          .if_("enable_skip_list_context_fea == 0") \
            .explore_listwise_attr(
              item_list_from_attr = "retrieval_list_keys",
              seq_item_attr_name = "generated_diversity_lists",
              hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
              hetu_level_two_attr = "hetu_tag_level_info__hetu_level_two",
              duration_attr = "duration_ms",
              hetu_level_one_count = "hetu_level_one_count",
              hetu_level_two_count = "hetu_level_two_count",
              duration_0_9s_num_attr = "0_9s_duration_photo_count",
              duration_9_15s_num_attr = "9_15s_duration_photo_count",
              duration_15_20s_num_attr = "15_20s_duration_photo_count",
              duration_20_58s_num_attr = "20_58s_duration_photo_count",
              duration_gt_58s_num_attr = "gt_58s_duration_photo_count",
              avg_duration_attr = "avg_duration_context",
              context_item_attr_map = rerank_list_fea(),
              item_attrs_transform_map = rerank_features_new,
              enable_context_attr = "{{explore_rerank_enable_context_attr}}"
            ) \
          .end_() \
        .case_(1, to_be_delete = "date=2024-05-29;committer=xuwei09") \
          .if_("enable_skip_old_list_fea == 0", to_be_delete = "date=2024-05-29;committer=xuwei09") \
            .list_wise_seq_attr(
              item_attrs_transform_map = rerank_features,
              seq_item_attr_name = "generated_diversity_lists",
              item_list_from_attr = "retrieval_list_keys"
            ) \
          .end_() \
          .if_("enable_skip_list_context_fea == 0") \
            .explore_listwise_attr(
              item_list_from_attr = "retrieval_list_keys",
              seq_item_attr_name = "generated_diversity_lists",
              hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
              hetu_level_two_attr = "hetu_tag_level_info__hetu_level_two",
              duration_attr = "duration_ms",
              hetu_level_one_count = "hetu_level_one_count",
              hetu_level_two_count = "hetu_level_two_count",
              duration_0_9s_num_attr = "0_9s_duration_photo_count",
              duration_9_15s_num_attr = "9_15s_duration_photo_count",
              duration_15_20s_num_attr = "15_20s_duration_photo_count",
              duration_20_58s_num_attr = "20_58s_duration_photo_count",
              duration_gt_58s_num_attr = "gt_58s_duration_photo_count",
              avg_duration_attr = "avg_duration_context",
              context_item_attr_map = rerank_list_fea(),
              item_attrs_transform_map = rerank_features,
              enable_context_attr = "{{explore_rerank_enable_context_attr}}"
            ) \
          .end_() \
        .case_(2) \
          .if_("enable_skip_old_list_fea == 0", to_be_delete = "date=2024-05-29;committer=xuwei09") \
            .list_wise_seq_attr(
              item_attrs_transform_map = rerank_features_new_v2,
              seq_item_attr_name = "generated_diversity_lists",
              item_list_from_attr = "retrieval_list_keys"
            ) \
          .end_() \
          .if_("enable_skip_list_context_fea == 0") \
            .explore_listwise_attr(
              item_list_from_attr = "retrieval_list_keys",
              use_attr_value_type = "{{explore_rerank_listwise_fea_use_attr_value_type}}",
              seq_item_attr_name = "generated_diversity_lists",
              hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
              hetu_level_two_attr = "hetu_tag_level_info__hetu_level_two",
              duration_attr = "duration_ms",
              hetu_level_one_count = "hetu_level_one_count",
              hetu_level_two_count = "hetu_level_two_count",
              duration_0_9s_num_attr = "0_9s_duration_photo_count",
              duration_9_15s_num_attr = "9_15s_duration_photo_count",
              duration_15_20s_num_attr = "15_20s_duration_photo_count",
              duration_20_58s_num_attr = "20_58s_duration_photo_count",
              duration_gt_58s_num_attr = "gt_58s_duration_photo_count",
              avg_duration_attr = "avg_duration_context",
              context_item_attr_map = rerank_list_fea_v2(),
              item_attrs_transform_map = rerank_features_new_v2,
              enable_context_attr = "{{explore_rerank_enable_context_attr}}"
            ) \
          .end_() \
      .end_()

    self.flow.end_()

  def post_process(self) -> None:
    self.flow \
      .if_("enable_use_explore_rerank == 1") \
        .log_debug_info(
          item_attrs = [
            "photo_id",
            "author__fans_count",
            "dnn_cluster_variant_attr",
            "hetu_cluster_attr",
            "short_duration_variant_attr",
            "long_duration_variant_attr",
            "lt20s_duration_variant_attr",
            "empirical_ctr",
            "empirical_ltr",
            "empirical_wtr",
            "empirical_ftr",
            "empirical_ptr",
            "empirical_cmtr",
            "empirical_htr",
            "hetu_level_one_attr",
            "hetu_level_two_attr",
            "hetu_level_two_attr2",
            "hetu_level_two_attr3",
            "empirical_watchtime",
            "duration_0_7s",
            "duration_7_9s",
            "duration_9_12s",
            "duration_12_17s",
            "duration_17_20s",
            "duration_20_58s",
            "duration_gt_58s",
            "duration_gt_120s",
            "reason",
            "tag",
            "music",
            "mod",
            "pliving_wtr",
            "mmu_img_cluster_v3",
            "live_photo_info__is_living",
            "pfvtr",
            "location__city_id",
            "author_age_info__age_segment",
            "mmu_content_id",
            "pliving_ctr",
            "empirical_rrr",
            "mmu_img_cluster_v1",
            "show_level_b",
            "show_level_a",
            "ocr_cover_text_word_count",
            "author__gender",
            "mmu_cluster_music_id",
            "location__province_id",
            "photo_age_hour",
            "music_info__music_combo_id",
            "fullrank_neg_feedback_discount_score",
            "fullrank_l2r_score",
            "diversity",
            "shuffle_policy_changed",
            "gr_policy_softcore",
            "specified_hetu5_found",
            "video_variant_attr",
          ],
          item_num_limit = 10,
          for_debug_request_only = True
        ) \
        .log_debug_info(
          common_attrs = [
            "long_set_pic_pid_fractions",
          ],
          item_attrs = [
            "ctr",
            "wtr",
            "ltr",
            "fr_score1_corr", #这个要改名字
            "fr_score2_corr", #这个要改名字
            "l2r_score",
            "ftr",
            "duration_gt_58s_corr", #这个需要改名字
            "ptr",
            "lvtr",
            "epstr",
            "ensemble_score",
            "cltr",
            "fetr_corr", #这个要改名字
            "feff",
            "cmtr",
            "cmef",
          ],
          item_num_limit = 20,
          for_debug_request_only = True
        ) \
      .end_() \
      .if_("enable_explore_pic_cluster_counter == 1") \
        .explore_pic_cluster_counter_enricher(
          save_pic_cluster_distr_str_attr = "rank_pic_cluster_distr_str",
          save_long_term_interest_cnt_attr = "rank_pic_long_term_interest_count",
          save_short_term_interest_cnt_attr = "rank_pic_short_term_interest_count",
          save_explore_interest_cnt_attr = "rank_pic_explore_interest_count",
          save_unknown_interest_cnt_attr = "rank_pic_unknown_interest_count",
          save_pic_cnt_attr = "rank_pic_count",
          save_hetu_cnt_attr = "rank_pic_hetu_count",
          long_term_interest_list_attr = "explore_pic_long_interest_list",
          short_term_interest_list_attr = "explore_pic_short_interest_list",
          explore_interest_list_attr = "explore_pic_explore_interest_list",
          hetu_list_attr = "hetu_tag_level_info__hetu_level_one",
          candidate_topk = "{{dpp_mix_rerank_pic_queue_size}}",
          target_item = {"is_picture": 1}
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "uPicLongInterestClusterIdList", "as": "long_interest_cluster_list"},
            {"name": "uPicValidInterestClusterIdList", "as": "valid_interest_cluster_list"},
            {"name": "uSingleValidPicCluster7dList", "as": "pic_single_valid_interest_cluster_list"},
            {"name": "uDoubleOutsideValidPicCluster7dList", "as": "pic_double_valid_interest_cluster_list"},
            {"name": "pic_recent_search_cluster_id_632_list", "as": "recent_search_cluster_list"},
            {"name": "dpp_mix_rerank_pic_queue_size", "as": "candidate_size"},
          ],
          import_item_attr = [
            "cluster_id_632"
          ],
          export_common_attr = [
            {"name": "cluster_count", "as": "rank_pic_cluster_count"},
            {"name": "long_interest_count", "as": "rank_pic_long_interest_count"},
            {"name": "valid_interest_count", "as": "rank_pic_valid_interest_count"},
            {"name": "pic_single_valid_interest_count", "as": "rank_pic_single_valid_interest_count"},
            {"name": "pic_double_valid_interest_count", "as": "rank_pic_double_valid_interest_count"},
            {"name": "recent_search_interest_count", "as": "rank_pic_recent_search_interest_count"},
          ],
          function_name = "CountPicInterestClusterDistribution",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {"is_picture": 1},
        ) \
      .end_()
