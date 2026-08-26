#!/usr/bin/env python3
# coding=utf-8

def gen_variant_config2():
    return {
        "default_decay_rate": 0.1,
        "default_decay_window_size": 6,
        "default_decay_occurrent_times": 3,
        "author__id": {
          "decay_occurrent_times": "{{fountain_rerank_author_id_times}}",
          "decay_window_size": "{{fountain_rerank_author_id_winsize}}"
        },
        "author__category_detail__fourth_level_id": {
          "decay_window_size": "{{fountain_rerank_author_fourth_level_id_winsize}}",
          "decay_occurrent_times": "{{fountain_rerank_author_fourth_level_id_times}}"
        },
        "author__category_detail__second_level_id": {
          "decay_window_size": "{{fountain_rerank_author_second_level_id_winsize}}",
          "decay_occurrent_times": "{{fountain_rerank_author_second_level_id_times}}"
        },
        "hetu_level_one_attr": {
          "decay_occurrent_times": "{{fountain_rerank_hetu_level_one_times}}",
          "decay_window_size": "{{fountain_rerank_hetu_level_one_winsize}}"
        },
        "hetu_level_two_attr": {
          "decay_occurrent_times": "{{fountain_rerank_hetu_level_two_times}}",
          "decay_window_size": "{{fountain_rerank_hetu_level_two_winsize}}"
        },
        "picture_variant_attr": {
          "decay_occurrent_times": "{{fountain_rerank_picture_times}}",
          "decay_window_size": "{{fountain_rerank_picture_winsize}}"
        },
        "is_photo_author_followed": {
          "decay_window_size": "{{fountain_rerank_follow_author_id_winsize}}",
          "decay_occurrent_times": "{{fountain_rerank_follow_author_id_times}}",
        },
        "duration_0_7s": {
          "decay_occurrent_times": "{{fountain_rerank_duration_0_7s_times}}",
          "decay_window_size": "{{fountain_rerank_duration_0_7s_winsize}}"
        },
        "duration_7_9s": {
          "decay_occurrent_times": "{{fountain_rerank_duration_7_9s_times}}",
          "decay_window_size": "{{fountain_rerank_duration_7_9s_winsize}}"
        },
        "duration_9_12s": {
          "decay_occurrent_times": "{{fountain_rerank_duration_9_12s_times}}",
          "decay_window_size": "{{fountain_rerank_duration_9_12s_winsize}}"
        },
        "duration_12_17s": {
          "decay_occurrent_times": "{{fountain_rerank_duration_12_17s_times}}",
          "decay_window_size": "{{fountain_rerank_duration_12_17s_winsize}}"
        },
        "duration_17_20s": {
          "decay_occurrent_times": "{{fountain_rerank_duration_17_20s_times}}",
          "decay_window_size": "{{fountain_rerank_duration_17_20s_winsize}}"
        },
        "duration_20_58s": {
          "decay_occurrent_times": "{{fountain_rerank_duration_20_58s_times}}",
          "decay_window_size": "{{fountain_rerank_duration_20_58s_winsize}}"
        },
        "duration_gt_58s": {
          "decay_occurrent_times": "{{fountain_rerank_duration_gt_58s_times}}",
          "decay_window_size": "{{fountain_rerank_duration_gt_58s_winsize}}"
        },
        "duration_gt_120s": {
          "decay_occurrent_times": "{{fountain_rerank_duration_gt_120s_times}}",
          "decay_window_size": "{{fountain_rerank_duration_gt_120s_winsize}}"
        },
        "hetu_level_one_6_type": {
          "decay_occurrent_times": "{{fountain_rerank_hetu_level_one_6_type_times}}",
          "decay_window_size": "{{fountain_rerank_hetu_level_one_6_type_winsize}}"
        },
        "hetu_level_one_9_type": {
          "decay_occurrent_times": "{{fountain_rerank_hetu_level_one_9_type_times}}",
          "decay_window_size": "{{fountain_rerank_hetu_level_one_9_type_winsize}}"
        },
        "hetu_level_one_28_type": {
          "decay_occurrent_times": "{{fountain_rerank_hetu_level_one_28_type_times}}",
          "decay_window_size": "{{fountain_rerank_hetu_level_one_28_type_winsize}}"
        },
        "hetu_level_one_39_type": {
          "decay_occurrent_times": "{{fountain_rerank_hetu_level_one_39_type_times}}",
          "decay_window_size": "{{fountain_rerank_hetu_level_one_39_type_winsize}}"
        }
    }

def gen_variant_config():
    return {
        "default_decay_rate": 0.1,
        "default_decay_window_size": 6,
        "default_decay_occurrent_times": 3,
        "author__id": {
          "decay_occurrent_times": 3,
          "decay_window_size": 6
        },
    }

def gen_beamsearch_filter_queues():
    res = []
    queues = ["click", "like", "follow", "forward", "comment", "profile", "longview",
              "watchtime", "shortview", "l2r", "pepstr", "outctr", "shortviewinorder", "neg_feedback_discount",
              "watchtime_ori", "fusion_pctr", "fusion_pcltr"]
    prefix = "{{fountain_rerank_beamsearch_filter_"
    for i, q in enumerate(queues) :
      t = {"name" : "fullrank_" + q + "_score"}
      t.update({"weight_base" : prefix + q + '_score}}'})
      res.append(t)
    return res

def gen_splash_seed_ensemble_queues():
    res = [
      {
        "name": "fullrank_click_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_click_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_click_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_click_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_click_use_proportion}}"
      },
      {
        "name": "fullrank_like_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_like_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_like_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_like_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_like_use_proportion}}"
      },
      {
        "name": "fullrank_follow_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_follow_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_follow_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_follow_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_follow_use_proportion}}"
      },
      {
        "name": "fullrank_forward_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_forward_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_forward_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_forward_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_forward_use_proportion}}"
      },
      {
        "name": "fullrank_comment_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_comment_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_comment_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_comment_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_comment_use_proportion}}"
      },
      {
        "name": "fullrank_profile_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_profile_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_profile_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_profile_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_profile_use_proportion}}"
      },
      {
        "name": "fullrank_longview_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_longview_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_longview_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_longview_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_longview_use_proportion}}"
      },
      {
        "name": "fullrank_watchtime_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_watchtime_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_watchtime_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_watchtime_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_watchtime_use_proportion}}"
      },
      {
        "name": "fullrank_shortview_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_shortview_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_shortview_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_shortview_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_shortview_use_proportion}}"
      },
      {
        "name": "fullrank_l2r_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_l2r_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_l2r_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_l2r_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_l2r_use_proportion}}"
      },
      {
        "name": "fullrank_pepstr_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_pepstr_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_pepstr_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_pepstr_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_pepstr_use_proportion}}"
      },
      {
        "name": "fullrank_outctr_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_outctr_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_outctr_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_outctr_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_outctr_use_proportion}}"
      },
      {
        "name": "fullrank_shortviewinorder_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_shortviewinorder_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_shortviewinorder_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_shortviewinorder_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_shortviewinorder_use_proportion}}"
      },
      {
        "name": "fullrank_neg_feedback_discount_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_neg_feedback_discount_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_neg_feedback_discount_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_neg_feedback_discount_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_neg_feedback_discount_use_proportion}}"
      },
      {
        "name": "fullrank_watchtime_ori_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_watchtime_ori_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_watchtime_ori_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_watchtime_ori_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_watchtime_ori_use_proportion}}"
      },
      {
        "name": "fullrank_evtr_v2_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_evtr_v2_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_evtr_v2_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_evtr_v2_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_evtr_v2_use_proportion}}"
      },
      {
        "name": "fullrank_lstr_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_lstr_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_lstr_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_lstr_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_lstr_use_proportion}}"
      },
      {
        "name": "fullrank_collect_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_collect_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_collect_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_collect_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_collect_use_proportion}}"
      },
      {
        "name": "fullrank_cmef_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_cmef_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_cmef_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_cmef_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_cmef_use_proportion}}"
      },
      {
        "name": "fullrank_lvtr_ori_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_lvtr_ori_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_lvtr_ori_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_lvtr_ori_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_lvtr_ori_use_proportion}}"
      },
      {
        "name": "fullrank_pfintr_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_pfintr_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_pfintr_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_pfintr_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_pfintr_use_proportion}}"
      },
      {
        "name": "fullrank_finish_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_finish_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_finish_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_finish_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_finish_use_proportion}}"
      },
      {
        "name": "fullrank_next_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_next_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_next_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_next_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_next_use_proportion}}"
      },
      {
        "name": "fullrank_slide_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_slide_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_slide_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_slide_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_slide_use_proportion}}"
      },
      {
        "name": "fullrank_cl_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_cl_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_cl_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_cl_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_cl_use_proportion}}"
      },
      {
        "name": "fullrank_act_ctr_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_act_ctr_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_act_ctr_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_act_ctr_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_act_ctr_use_proportion}}"
      },
      {
        "name": "fullrank_fusion_pctr_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_fusion_pctr_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_fusion_pctr_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_fusion_pctr_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_fusion_pctr_use_proportion}}"
      },
      {
        "name": "fullrank_fusion_pcltr_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_fusion_pcltr_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_fusion_pcltr_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_fusion_pcltr_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_fusion_pcltr_use_proportion}}"
      },
      {
        "name": "fullrank_cl_play_time_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_cl_play_time_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_cl_play_time_range}}",
        "weight_lower_bound": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_cl_play_time_lower_bound}}",
        "enable_use_proportion": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_cl_play_time_use_proportion}}"
      },
      {
        "name": "fullrank_pcpr_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_pcpr_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_pcpr_range}}",
      },
      {
        "name": "fullrank_sim_pcmef",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_sim_pcmef_weight}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_fullrank_sim_pcmef_range}}",
      },

      # duration 队列
      {
        "name": "duration_0_7s",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_duration_0_7s_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_duration_0_7s_range}}",
        "reverse_order": "{{fountain_splash_rerank_gen_seed_ensemble_duration_0_7s_reverse_order}}"
      },
      {
        "name": "duration_7_9s",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_duration_7_9s_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_duration_7_9s_range}}",
        "reverse_order": "{{fountain_splash_rerank_gen_seed_ensemble_duration_7_9s_reverse_order}}"
      },
      {
        "name": "duration_9_12s",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_duration_9_12s_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_duration_9_12s_range}}",
        "reverse_order": "{{fountain_splash_rerank_gen_seed_ensemble_duration_9_12s_reverse_order}}"
      },
      {
        "name": "duration_12_17s",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_duration_12_17s_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_duration_12_17s_range}}",
        "reverse_order": "{{fountain_splash_rerank_gen_seed_ensemble_duration_12_17s_reverse_order}}"
      },
      {
        "name": "duration_17_20s",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_duration_17_20s_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_duration_17_20s_range}}",
        "reverse_order": "{{fountain_splash_rerank_gen_seed_ensemble_duration_17_20s_reverse_order}}"
      },
      {
        "name": "duration_20_58s",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_duration_20_58s_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_duration_20_58s_range}}",
        "reverse_order": "{{fountain_splash_rerank_gen_seed_ensemble_duration_20_58s_reverse_order}}"
      },
      {
        "name": "duration_gt_58s",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_duration_gt_58s_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_duration_gt_58s_range}}",
        "reverse_order": "{{fountain_splash_rerank_gen_seed_ensemble_duration_gt_58s_reverse_order}}"
      },
      {
        "name": "duration_gt_120s",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_duration_gt_120s_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_duration_gt_120s_range}}",
        "reverse_order": "{{fountain_splash_rerank_gen_seed_ensemble_duration_gt_120s_reverse_order}}"
      },
      # 相关分队列
      {
        "name": "source_related_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_source_related_score_score}}", 
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_source_related_score_range}}",
        "reverse_order": "{{fountain_splash_rerank_gen_seed_ensemble_source_related_score_reverse_order}}"
      },
      # relate ltr 队列
      {
        "name": "splash_fullrank_ltr_fusion_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_splash_fullrank_ltr_fusion_score_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_splash_fullrank_ltr_fusion_score_range}}",
      },
      {
        "name": "splash_fullrank_ltr_act_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_splash_fullrank_ltr_act_score_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_splash_fullrank_ltr_act_score_range}}",
      },
      {
        "name": "splash_fullrank_ltr_act_v2_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_splash_fullrank_ltr_act_v2_score_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_splash_fullrank_ltr_act_v2_score_range}}",
      },
      {
        "name": "splash_fullrank_ltr_wtd_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_splash_fullrank_ltr_wtd_score_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_splash_fullrank_ltr_wtd_score_range}}",
      },
      {
        "name": "splash_fullrank_ltr_lvtr_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_splash_fullrank_ltr_lvtr_score_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_splash_fullrank_ltr_lvtr_score_range}}",
      },
      {
        "name": "splash_fullrank_ltr_svtr_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_splash_fullrank_ltr_svtr_score_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_splash_fullrank_ltr_svtr_score_range}}",
        "reverse_order": "{{fountain_splash_rerank_gen_seed_ensemble_splash_fullrank_ltr_svtr_score_reverse_order}}",
      },
      {
        "name": "splash_fullrank_ltr_like_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_splash_fullrank_ltr_like_score_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_splash_fullrank_ltr_like_score_range}}",
      },
      {
        "name": "splash_fullrank_ltr_follow_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_splash_fullrank_ltr_follow_score_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_splash_fullrank_ltr_follow_score_range}}",
      },
      {
        "name": "splash_fullrank_ltr_comment_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_splash_fullrank_ltr_comment_score_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_splash_fullrank_ltr_comment_score_range}}",
      },
      {
        "name": "splash_fullrank_ltr_next_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_splash_fullrank_ltr_next_score_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_splash_fullrank_ltr_next_score_range}}",
      },
      {
        "name": "splash_fullrank_ltr_relate_evtr_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_splash_fullrank_ltr_relate_evtr_score_score}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_splash_fullrank_ltr_relate_evtr_score_range}}",
      },
      # 审核队列
      {
        "name": "fountain_fullrank_bad_item_similary_score",
        "weight_base": "{{fountain_splash_rerank_gen_seed_ensemble_bad_item_similary_score_weight}}",
        "bias_range": "{{fountain_splash_rerank_gen_seed_ensemble_bad_item_similary_score_range}}",
        "reverse_order": "{{fountain_splash_rerank_gen_seed_ensemble_bad_item_similary_score_reverse_order}}"
      },
    ]

    return res

def gen_photo_features_for_idx_position(idx):
    return [
        "pId_idx{}".format(idx),
        "aId_idx{}".format(idx),
        "pPctr_idx{}".format(idx),
        "pPltr_idx{}".format(idx),
        "pPwtr_idx{}".format(idx),
        "pPftr_idx{}".format(idx),
        "pPhtr_idx{}".format(idx),
        "pPlvtr_idx{}".format(idx),
        "pPsvtr_idx{}".format(idx),
        "pPvtr_idx{}".format(idx),
        "pPptr_idx{}".format(idx),
        "pPcmtr_idx{}".format(idx),
        "pPcmef_idx{}".format(idx),
        "pPepstr_idx{}".format(idx),
        "pPfrScore1_idx{}".format(idx),
        "pPfrScore2_idx{}".format(idx),
        "pPwtd_idx{}".format(idx),
        "pMcPctr_idx{}".format(idx),
        "pMcPltr_idx{}".format(idx),
        "pMcPwtr_idx{}".format(idx),
        "pMcPlvtr_idx{}".format(idx),
        "pMcPsvtr_idx{}".format(idx),
        "pEmpCtr_idx{}".format(idx),
        "pEmpLtr_idx{}".format(idx),
        "pEmpWtr_idx{}".format(idx),
        "pEmpFtr_idx{}".format(idx),
        "pEmpPtr_idx{}".format(idx),
        "pEmpCmtr_idx{}".format(idx),
        "pEmpHtr_idx{}".format(idx),
        "pAuthorFansCount_idx{}".format(idx),
        "pUploadType_idx{}".format(idx),
        "pDurationMs_idx{}".format(idx),
        "pHotShow_idx{}".format(idx),
        "pHotClick_idx{}".format(idx),
        "pHotLike_idx{}".format(idx),
        "pHotFollow_idx{}".format(idx),
        "pHotHate_idx{}".format(idx),
        "pHotReport_idx{}".format(idx),
        "pHotLiving_idx{}".format(idx),
        "pHotExptag_idx{}".format(idx),
        "pUploadRate_idx{}".format(idx),
        "pCityId_idx{}".format(idx),
        "pProvinceId_idx{}".format(idx),
        "pContentLevel_idx{}".format(idx),
        "pAvgWatchtime_idx{}".format(idx),
        "pHetuTagLevel1Id_idx{}".format(idx),
        "pHetuTagLevel2Id_idx{}".format(idx),
        "pDnnClusterId_idx{}".format(idx),
        "pMmuImgClusterV1_idx{}".format(idx),
        "pMmuImgClusterV3_idx{}".format(idx),
        "pMusic_idx{}".format(idx),
        "pMmuContentId_idx{}".format(idx),
        "pMusicComboId_idx{}".format(idx),
        "pOcrCoverTextWordCount_idx{}".format(idx),
        "position_idx{}".format(idx),
    ]

def gen_photo_features_for_idx_position_v3(idx):
  return [
    "aId_idx{}".format(idx),
    "pAvgWatchtime_idx{}".format(idx),
    "pCityId_idx{}".format(idx),
    "pContentLevel_idx{}".format(idx),
    "pDurationMs_idx{}".format(idx),
    "pEmpCmtr_idx{}".format(idx),
    "pEmpCtr_idx{}".format(idx),
    "pEmpFtr_idx{}".format(idx),
    "pEmpHtr_idx{}".format(idx),
    "pEmpLtr_idx{}".format(idx),
    "pEmpPtr_idx{}".format(idx),
    "pEmpWtr_idx{}".format(idx),
    "pHetuTagLevel1Id_idx{}".format(idx),
    "pHetuTagLevel2Id_idx{}".format(idx),
    "pId_idx{}".format(idx),
    "pMcPctr_idx{}".format(idx),
    "pMcPltr_idx{}".format(idx),
    "pMcPlvtr_idx{}".format(idx),
    "pMcPsvtr_idx{}".format(idx),
    "pMcPwtr_idx{}".format(idx),
    "pMmuContentId_idx{}".format(idx),
    "pMmuImgClusterV1_idx{}".format(idx),
    "pMmuImgClusterV3_idx{}".format(idx),
    "pMusic_idx{}".format(idx),
    "pMusicComboId_idx{}".format(idx),
    "pOcrCoverTextWordCount_idx{}".format(idx),
    "pPcmef_idx{}".format(idx),
    "pPcmtr_idx{}".format(idx),
    "pPctr_idx{}".format(idx),
    "pPepstr_idx{}".format(idx),
    "pPftr_idx{}".format(idx),
    "pPhtr_idx{}".format(idx),
    "pPltr_idx{}".format(idx),
    "pPlvtr_idx{}".format(idx),
    "pPptr_idx{}".format(idx),
    "pProvinceId_idx{}".format(idx),
    "pPsvtr_idx{}".format(idx),
    "pPvtr_idx{}".format(idx),
    "pPwtd_idx{}".format(idx),
    "pPwtr_idx{}".format(idx),
    "pUploadType_idx{}".format(idx),
  ]

def gen_photo_short_stat_hetu_features(idx):
  prefix = ["pShortStatShow", "pShortStatClick"]
  hetu_prefix = ["Hetu1", "Hetu2", "Hetu3", "Hetu4", "Hetu5"]
  time_cnt_prefix = ["5m", "30m", "2h", "1d", "100n", "1000n"]
  hetu_short_prefix = [prefix[i] + hetu_prefix[j] + time_cnt_prefix[k] for i in range(len(prefix)) for j in range(len(hetu_prefix)) for k in range(len(time_cnt_prefix))]
  return [
    hetu_short_prefix[i] + "_idx{}".format(idx) for i in range(len(hetu_short_prefix))
  ]
def gen_photo_context_feature():
  return [
    "maxPctr_context",
    "maxPltr_context",
    "maxPwtr_context",
    "maxPftr_context",
    "maxPvtr_context",
    "maxPptr_context",
    "maxPcmtr_context",
    "maxPlivingtr_context",
    "avgPctr_context",
    "avgPltr_context",
    "avgPwtr_context",
    "avgPftr_context",
    "avgPvtr_context",
    "avgPptr_context",
    "avgPcmtr_context",
    "avgPlivingtr_context",
    "avg_duration_context",
    "hetu_level_one_count",
    "hetu_level_two_count",
    "0_9s_duration_photo_count",
    "9_15s_duration_photo_count",
    "15_20s_duration_photo_count",
    "20_58s_duration_photo_count",
    "gt_58s_duration_photo_count",
  ]

def gen_photo_features_for_all_position_new(n):
    return [attr for idx in range(n) for attr in gen_photo_features_for_idx_position(idx)] + [attr for idx in range(n) for attr in gen_photo_short_stat_hetu_features(idx)] + gen_photo_context_feature()

def gen_photo_features_for_all_position_v3(n):
  return [attr for idx in range(n) for attr in gen_photo_features_for_idx_position_v3(idx)]

def gen_list_predict_common_attrs():
    attrs = []
    for i in range(30):
        for tag in ["aid", "tag", "play"]:
            attr = "realshow_" + tag + "_" + str(i)
            attrs.append({"name" : attr, "as" : attr})
    return attrs

splash_rerank_expected_value_queues = [
  {
    "name": "fullrank_click_score",
    "weight": 0,
    "weight_attr": "fountain_splash_beamsearch_click_score_weight",
    "power_weight_attr": "fountain_splash_beamsearch_click_score_power_weight",
  },
  {
    "name": "fullrank_shortview_score",
    "weight": 0,
    "weight_attr": "fountain_splash_beamsearch_shortview_score_weight",
    "power_weight_attr": "fountain_splash_beamsearch_shortview_score_power_weight",
  },
  {
    "name": "fullrank_ltr_v4_fountain_finish_rate",
    "weight": 0,
    "weight_attr": "fountain_splash_beamsearch_finish_rate_weight",
    "power_weight_attr": "fountain_splash_beamsearch_finish_rate_power_weight",
  },
  {
    "name": "fullrank_watchtime_score",
    "weight": 0,
    "weight_attr": "fountain_splash_beamsearch_watchtime_weight",
    "power_weight_attr": "fountain_splash_beamsearch_watchtime_power_weight",
  },
  {
    "name": "fullrank_watchtime_ori_score",
    "weight": 0,
    "weight_attr": "fountain_splash_beamsearch_watchtime_ori_score_weight",
    "power_weight_attr": "fountain_splash_beamsearch_watchtime_ori_score_power_weight",
  },
  {
    "name": "fullrank_like_score",
    "weight": 0,
    "weight_attr": "fountain_splash_beamsearch_like_score_weight",
    "power_weight_attr": "fountain_splash_beamsearch_like_score_power_weight",
  },
  {
    "name": "fullrank_profile_score",
    "weight": 0,
    "weight_attr": "fountain_splash_beamsearch_profile_score_weight",
    "power_weight_attr": "fountain_splash_beamsearch_profile_score_power_weight",
  },
  {
    "name": "fullrank_comment_score",
    "weight": 0,
    "weight_attr": "fountain_splash_beamsearch_comment_score_weight",
    "power_weight_attr": "fountain_splash_beamsearch_comment_score_power_weight",
  },
  {
    "name": "fullrank_forward_score",
    "weight": 0,
    "weight_attr": "fountain_splash_beamsearch_forward_score_weight",
    "power_weight_attr": "fountain_splash_beamsearch_forward_score_power_weight",
  },
  {
    "name": "fullrank_follow_score",
    "weight": 0,
    "weight_attr": "fountain_splash_beamsearch_follow_score_weight",
    "power_weight_attr": "fountain_splash_beamsearch_follow_score_power_weight",
  },
  {
    "name": "fullrank_action_once_interact_score",
    "weight": 0,
    "weight_attr": "fountain_splash_beamsearch_action_once_interact_score_weight",
    "power_weight_attr": "fountain_splash_beamsearch_action_once_interact_score_power_weight",
  },
  {
    "name": "fullrank_neg_feedback_discount_score",
    "weight": 0,
    "weight_attr": "fountain_splash_beamsearch_neg_feedback_discount_score_weight",
    "power_weight_attr": "fountain_splash_beamsearch_neg_feedback_discount_score_power_weight",
  },
  {
    "name": "source_related_score",
    "weight": 0,
    "weight_attr": "fountain_splash_beamsearch_source_related_score_weight",
    "power_weight_attr": "fountain_splash_beamsearch_source_related_score_power_weight",
  },
  {
    "name": "fullrank_detail_new_pevtr_v2",
    "power_weight_attr": "fountain_splash_beamsearch_pevtr_v2_power_weight",
    "weight": 0,
    "weight_attr": "fountain_splash_beamsearch_pevtr_v2_weight",
    "raw_weight_attr": "fountain_splash_beamsearch_pevtr_v2_raw_weight_attr",
    "raw_pow_weight_attr": "fountain_splash_beamsearch_pevtr_v2_raw_pow_weight_attr",
  },
  {
    "name": "fullrank_sim_lstr",
    "power_weight_attr": "fountain_splash_beamsearch_sim_lstr_power_weight",
    "weight": 0,
    "weight_attr": "fountain_splash_beamsearch_sim_lstr_weight",
    "raw_weight_attr": "fountain_splash_beamsearch_sim_lstr_raw_weight_attr",
    "raw_pow_weight_attr": "fountain_splash_beamsearch_sim_lstr_raw_pow_weight_attr",
  },
  {
    "name": "fullrank_sim_evtr_playtime",
    "power_weight_attr": "fountain_splash_beamsearch_sim_evtr_playtime_power_weight",
    "weight": 0,
    "weight_attr": "fountain_splash_beamsearch_sim_evtr_playtime_weight",
    "raw_weight_attr": "fountain_splash_beamsearch_sim_evtr_playtime_raw_weight_attr",
    "raw_pow_weight_attr": "fountain_splash_beamsearch_sim_evtr_playtime_raw_pow_weight_attr",
  },
]

rerank_expected_value_queues = [
  {
    "name": "fullrank_sim_pevtr",
    "power_weight_attr": "fountain_expected_value_pevtr_power_weight",
    "weight": 0,
    "weight_attr": "fountain_expected_value_pevtr_weight",
    "raw_weight_attr": "fountain_rerank_expected_value_pevtr_raw_weight_attr",
    "raw_pow_weight_attr": "fountain_rerank_expected_value_pevtr_raw_pow_weight_attr",
  },
  {
    "name": "fullrank_detail_new_pevtr_v2",
    "power_weight_attr": "fountain_expected_value_pevtr_v2_power_weight",
    "weight": 0,
    "weight_attr": "fountain_expected_value_pevtr_v2_weight",
    "raw_weight_attr": "fountain_rerank_expected_value_pevtr_v2_raw_weight_attr",
    "raw_pow_weight_attr": "fountain_rerank_expected_value_pevtr_v2_raw_pow_weight_attr",
  },
  {
    "name": "fullrank_ltr_v4_fountain_finish_rate",
    "power_weight_attr": "fountain_expected_value_finish_rate_power_weight",
    "weight": 0,
    "weight_attr": "fountain_expected_value_finish_rate_weight",
    "raw_weight_attr": "fountain_rerank_expected_value_finish_rate_raw_weight_attr",
    "raw_pow_weight_attr": "fountain_rerank_expected_value_finish_rate_raw_pow_weight_attr",
  },
  {
    "name": "fullrank_sim_plvtr",
    "power_weight_attr": "fountain_expected_value_plvtr_power_weight",
    "weight": 0,
    "weight_attr": "fountain_expected_value_plvtr_weight",
    "raw_weight_attr": "fountain_rerank_expected_value_plvtr_raw_weight_attr",
    "raw_pow_weight_attr": "fountain_rerank_expected_value_plvtr_raw_pow_weight_attr",
  },
  {
    "name": "fullrank_sim_pfintr",
    "power_weight_attr": "fountain_expected_value_pfintr_power_weight",
    "weight": 0,
    "weight_attr": "fountain_expected_value_pfintr_weight",
    "raw_weight_attr": "fountain_rerank_expected_value_pfintr_raw_weight_attr",
    "raw_pow_weight_attr": "fountain_rerank_expected_value_pfintr_raw_pow_weight_attr",
  },
  {
    "name": "fullrank_trans_pvtr_score",
    "power_weight_attr": "fountain_expected_value_trans_pvtr_power_weight",
    "weight": 0,
    "weight_attr": "fountain_expected_value_trans_pvtr_weight",
    "raw_weight_attr": "fountain_rerank_expected_value_trans_pvtr_raw_weight_attr",
    "raw_pow_weight_attr": "fountain_rerank_expected_value_trans_pvtr_raw_pow_weight_attr",
  },
  {
    "name": "fullrank_sim_pvtr",
    "power_weight_attr": "fountain_expected_value_pvtr_power_weight",
    "weight": 0,
    "weight_attr": "fountain_expected_value_pvtr_weight",
    "raw_weight_attr": "fountain_rerank_expected_value_pvtr_raw_weight_attr",
    "raw_pow_weight_attr": "fountain_rerank_expected_value_pvtr_raw_pow_weight_attr",
  },
  {
    "name": "fullrank_sim_pcpr",
    "power_weight_attr": "fountain_expected_value_pcpr_power_weight",
    "weight": 0,
    "weight_attr": "fountain_expected_value_pcpr_weight",
    "raw_weight_attr": "fountain_rerank_expected_value_pcpr_raw_weight_attr",
    "raw_pow_weight_attr": "fountain_rerank_expected_value_pcpr_raw_pow_weight_attr",
  },
  {
    "name": "fullrank_sim_pltr",
    "power_weight_attr": "fountain_expected_value_pltr_power_weight",
    "weight": 0,
    "weight_attr": "fountain_expected_value_pltr_weight",
    "raw_weight_attr": "fountain_rerank_expected_value_pltr_raw_weight_attr",
    "raw_pow_weight_attr": "fountain_rerank_expected_value_pltr_raw_pow_weight_attr",
  },
  {
    "name": "fullrank_sim_pwtr",
    "power_weight_attr": "fountain_expected_value_pwtr_power_weight",
    "weight": 0,
    "weight_attr": "fountain_expected_value_pwtr_weight",
    "raw_weight_attr": "fountain_rerank_expected_value_pwtr_raw_weight_attr",
    "raw_pow_weight_attr": "fountain_rerank_expected_value_pwtr_raw_pow_weight_attr",
  },
  {
    "name": "fullrank_sim_pftr",
    "power_weight_attr": "fountain_expected_value_pftr_power_weight",
    "weight": 0,
    "weight_attr": "fountain_expected_value_pftr_weight",
    "raw_weight_attr": "fountain_rerank_expected_value_pftr_raw_weight_attr",
    "raw_pow_weight_attr": "fountain_rerank_expected_value_pftr_raw_pow_weight_attr",
  },
  {
    "name": "fullrank_sim_pcltr",
    "power_weight_attr": "fountain_expected_value_pcltr_power_weight",
    "weight": 0,
    "weight_attr": "fountain_expected_value_pcltr_weight",
    "raw_weight_attr": "fountain_rerank_expected_value_pcltr_raw_weight_attr",
    "raw_pow_weight_attr": "fountain_rerank_expected_value_pcltr_raw_pow_weight_attr",
  },
  {
    "name": "fullrank_sim_pcmtr",
    "power_weight_attr": "fountain_expected_value_pcmtr_power_weight",
    "weight": 0,
    "weight_attr": "fountain_expected_value_pcmtr_weight",
    "raw_weight_attr": "fountain_rerank_expected_value_pcmtr_raw_weight_attr",
    "raw_pow_weight_attr": "fountain_rerank_expected_value_pcmtr_raw_pow_weight_attr",
  },
  {
    "name": "fullrank_sim_pcmef",
    "power_weight_attr": "fountain_expected_value_pcmef_power_weight",
    "weight": 0,
    "weight_attr": "fountain_expected_value_pcmef_weight",
    "raw_weight_attr": "fountain_rerank_expected_value_pcmef_raw_weight_attr",
    "raw_pow_weight_attr": "fountain_rerank_expected_value_pcmef_raw_pow_weight_attr",
  },
  {
    "name": "fullrank_sim_pptr",
    "power_weight_attr": "fountain_expected_value_pptr_power_weight",
    "weight": 0,
    "weight_attr": "fountain_expected_value_pptr_weight",
    "raw_weight_attr": "fountain_rerank_expected_value_pptr_raw_weight_attr",
    "raw_pow_weight_attr": "fountain_rerank_expected_value_pptr_raw_pow_weight_attr",
  },
  {
    "name": "fullrank_sim_lstr",
    "power_weight_attr": "fountain_expected_value_lstr_power_weight",
    "weight": 0,
    "weight_attr": "fountain_expected_value_lstr_weight",
    "raw_weight_attr": "fountain_rerank_expected_value_lstr_raw_weight_attr",
    "raw_pow_weight_attr": "fountain_rerank_expected_value_lstr_raw_pow_weight_attr",
  },
  {
    "name": "fullrank_sim_pepstr",
    "power_weight_attr": "fountain_expected_value_pepstr_power_weight",
    "weight": 0,
    "weight_attr": "fountain_expected_value_pepstr_weight",
    "raw_weight_attr": "fountain_rerank_expected_value_pepstr_raw_weight_attr",
    "raw_pow_weight_attr": "fountain_rerank_expected_value_pepstr_raw_pow_weight_attr",
  },
  {
    "name": "fullrank_action_once_watchtime_score",
    "power_weight_attr": "fountain_expected_value_action_once_watchtime_power_weight",
    "weight": 0,
    "weight_attr": "fountain_expected_value_action_once_watchtime_weight",
    "raw_weight_attr": "fountain_rerank_expected_value_action_once_watchtime_raw_weight_attr",
    "raw_pow_weight_attr": "fountain_rerank_expected_value_action_once_watchtime_raw_pow_weight_attr",
  },
  {
    "name": "fullrank_action_once_interact_score",
    "power_weight_attr": "fountain_expected_value_action_once_interact_power_weight",
    "weight": 0,
    "weight_attr": "fountain_expected_value_action_once_interact_weight",
    "raw_weight_attr": "fountain_rerank_expected_value_action_once_interact_raw_weight_attr",
    "raw_pow_weight_attr": "fountain_rerank_expected_value_action_once_interact_raw_pow_weight_attr",
  },
]

rerank_features_v3 = [
  {"name": "author__id", "as": "aId"},
  {"name": "empirical_watchtime", "as": "pAvgWatchtime"},
  {"name": "featurePCityId", "as": "pCityId"},
  {"name": "content_safety_level_with_namespace__level_hot_online", "as": "pContentLevel"},
  {"name": "featurePDurationMs", "as": "pDurationMs"},
  {"name": "empirical_cmtr", "as": "pEmpCmtr"},
  {"name": "empirical_ctr", "as": "pEmpCtr"},
  {"name": "empirical_ftr", "as": "pEmpFtr"},
  {"name": "empirical_htr", "as": "pEmpHtr"},
  {"name": "empirical_ltr", "as": "pEmpLtr"},
  {"name": "empirical_ptr", "as": "pEmpPtr"},
  {"name": "empirical_wtr", "as": "pEmpWtr"},
  {"name": "featurePHetuTagLevel1", "as": "pHetuTagLevel1Id"},
  {"name": "featurePHetuTagLevel2", "as": "pHetuTagLevel2Id"},
  {"name": "photo_id", "as": "pId"},
  {"name": "cascade_pctr", "as": "pMcPctr"},
  {"name": "cascade_pltr", "as": "pMcPltr"},
  {"name": "cascade_plvtr", "as": "pMcPlvtr"},
  {"name": "cascade_psvtr", "as": "pMcPsvtr"},
  {"name": "cascade_pwtr", "as": "pMcPwtr"},
  {"name": "mmu_content_id", "as": "pMmuContentId"},
  {"name": "mmu_img_cluster_v1", "as": "pMmuImgClusterV1"},
  {"name": "featurePMmuImgClusterV3", "as": "pMmuImgClusterV3"},
  {"name": "featurePMusic", "as": "pMusic"},
  {"name": "featurePMusicComboId", "as": "pMusicComboId"},
  {"name": "featurePOcrCoverTextWordCount", "as": "pOcrCoverTextWordCount"},
  {"name": "fullrank_sim_pcmef", "as": "pPcmef"},
  {"name": "fullrank_sim_pcmtr", "as": "pPcmtr"},
  {"name": "fullrank_sim_pevtr", "as": "pPctr"},
  {"name": "fullrank_sim_pepstr", "as": "pPepstr"},
  {"name": "fullrank_sim_pftr", "as": "pPftr"},
  {"name": "fullrank_sim_phtr", "as": "pPhtr"},
  {"name": "fullrank_sim_pltr", "as": "pPltr"},
  {"name": "fullrank_sim_plvtr", "as": "pPlvtr"},
  {"name": "fullrank_sim_pptr", "as": "pPptr"},
  {"name": "featurePProvinceId", "as": "pProvinceId"},
  {"name": "fullrank_sim_out_pctr", "as": "pPsvtr"},
  {"name": "fullrank_sim_pvtr", "as": "pPvtr"},
  {"name": "fullrank_sim_pfintr", "as": "pPwtd"},
  {"name": "fullrank_sim_pwtr", "as": "pPwtr"},
  {"name": "featurePUploadType", "as": "pUploadType"},
]

rerank_gen_model_send_item_feas = [
  "cascade_pctr",
  "cascade_pltr",
  "cascade_pwtr",
  "cascade_plvtr",
  "cascade_psvtr",
  "cascade_ptr",
  "cascade_pcmtr",
  "cascade_pftr",
  "fullrank_detail_pctr",
  "fullrank_detail_pltr",
  "fullrank_detail_pwtr",
  "fullrank_detail_pftr",
  "fullrank_detail_plvtr",
  "fullrank_detail_pvtr",
  "fullrank_detail_psvr",
  "fullrank_detail_pcmtr",
  "fullrank_detail_pptr",
  "fullrank_detail_pwtd",
  "fullrank_sim_pcpr",
  "fullrank_sim_pcltr",
  "fullrank_sim_pepstr",
  "fullrank_act_wtd",
  "fullrank_sim_psvr",
  "fullrank_ltr_v4_fountain_next",
  "fountain_related_score_v2",
  "fullrank_ltr_score"
]

rerank_gen_model_send_common_feas = [
  { "name": "rerank_deep_ltr_trimmed_user_info", "as": "user_info_str" },
  { "name": "featureSourcePId", "as": "source_pid" },
  { "name": "sourcePidAuthorId", "as": "source_aid" },
  { "name": "sourcePidHetuLevelOneList", "as": "source_hetu_tag_level1_list" },
  { "name": "sourcePidHetuLevelTwoList", "as": "source_hetu_tag_level2_list" },
  { "name": "sourcePidDuration", "as": "source_duration_ms" },
  { "name": "sourcePidTagId", "as": "source_tag" },
]

rerank_gen_model_send_user_feas = [
  "active_days",
  "basic_info.age_segment",
  "location.city_id",
  "location.region_type",
  "client_id",
  "device_id",
  "gender",
  "infer_gender",
  "true_gender",
  "request_location.poi_type",
  "request_location.province_id",
  "request_location.city_id",
  "visit_mod",
  "user_profile.exp_stat.exp_click",
  "user_profile.exp_stat.exp_like",
  "user_profile.exp_stat.exp_follow",
  "user_profile.exp_stat.exp_realshow",
  "user_profile.exp_stat.exp_long_view",
  "user_profile.user_level",
  "realtime_click_list",
  "realtime_follow_list",
  "realtime_forward_list",
  "realtime_like_list",
  "user_profile_v1.click_list.author_id",
  "user_profile_v1.click_list.photo_id",
  "user_profile_v1.follow_list.author_id",
  "user_profile_v1.follow_list.photo_id",
  "user_profile_v1.like_list.author_id",
  "user_profile_v1.like_list.photo_id",
  "user_profile_v1.video_playing_stat.playing_time",
  "user_profile_v1.video_playing_stat.author_id",
  "user_profile_v1.video_playing_stat.photo_id",
  "user_profile_v1.video_playing_stat.client_timestamp",
  "fountain_reco_user_profile.click_list.author_id",
  "fountain_reco_user_profile.click_list.photo_id",
  "fountain_reco_user_profile.follow_list.author_id",
  "fountain_reco_user_profile.follow_list.photo_id",
  "fountain_reco_user_profile.like_list.author_id",
  "fountain_reco_user_profile.like_list.photo_id",
  "fountain_reco_user_profile.comment_list.author_id",
  "fountain_reco_user_profile.comment_list.photo_id",
  "fountain_reco_user_profile.video_play_stat.photo_id",
  "fountain_reco_user_profile.video_play_stat.author_id",
  "fountain_reco_user_profile.video_play_stat.video_duration",
  "fountain_reco_user_profile.video_play_stat.playing_time",
  "fountain_reco_user_profile.video_play_stat.client_timestamp",
]

rerank_flash_eval_model_send_item_feas = [
  "cascade_pctr",
  "cascade_pltr",
  "cascade_pwtr",
  "cascade_plvtr",
  "cascade_psvtr",
  "cascade_ptr",
  "cascade_pcmtr",
  "cascade_pftr",
  "fullrank_detail_pctr",
  "fullrank_detail_pltr",
  "fullrank_detail_pwtr",
  "fullrank_detail_pftr",
  "fullrank_detail_plvtr",
  "fullrank_detail_pvtr",
  "fullrank_detail_psvr",
  "fullrank_detail_pcmtr",
  "fullrank_detail_pptr",
  "fullrank_detail_pwtd",
  "fullrank_sim_pcpr",
  "fullrank_sim_pcltr",
  "fullrank_sim_pepstr",
  "fullrank_act_wtd",
  "fullrank_sim_psvr",
  "fullrank_ltr_v4_fountain_next",
  "fountain_related_score_v2",
  "fullrank_ltr_score"
]

rerank_flash_eval_model_send_common_feas = [
  { "name": "rerank_deep_ltr_trimmed_user_info", "as": "user_info_str" },
  { "name": "featureSourcePId", "as": "source_pid" },
  { "name": "sourcePidAuthorId", "as": "source_aid" },
  { "name": "sourcePidHetuLevelOneList", "as": "source_hetu_tag_level1_list" },
  { "name": "sourcePidHetuLevelTwoList", "as": "source_hetu_tag_level2_list" },
  { "name": "sourcePidDuration", "as": "source_duration_ms" },
  { "name": "sourcePidTagId", "as": "source_tag" },
  { "name": "featureSimilarUserList", "as": "similar_user_list" },
  { "name": "rerank_list_item_idx_flat_list", "as": "origin_rerank_list_item_idx_flat_list" },
  "page",
]

rerank_flash_eval_model_send_user_feas = [
  "active_days",
  "basic_info.age_segment",
  "location.city_id",
  "location.region_type",
  "client_id",
  "device_id",
  "gender",
  "infer_gender",
  "true_gender",
  "request_location.poi_type",
  "request_location.province_id",
  "request_location.city_id",
  "visit_mod",
  "user_profile.exp_stat.exp_click",
  "user_profile.exp_stat.exp_like",
  "user_profile.exp_stat.exp_follow",
  "user_profile.exp_stat.exp_realshow",
  "user_profile.exp_stat.exp_long_view",
  "user_profile_v1.click_list.author_id",
  "user_profile_v1.click_list.photo_id",
  "user_profile_v1.follow_list.author_id",
  "user_profile_v1.follow_list.photo_id",
  "user_profile_v1.like_list.author_id",
  "user_profile_v1.like_list.photo_id",
  "user_profile_v1.hate_list.photo_id",
  "user_profile_v1.video_playing_stat.playing_time",
  "user_profile_v1.video_playing_stat.author_id",
  "user_profile_v1.video_playing_stat.photo_id",
  "user_profile_v1.video_playing_stat.client_timestamp",
  "user_profile.user_level",
  "realtime_click_list",
  "realtime_follow_list",
  "realtime_forward_list",
  "realtime_like_list",
  "fountain_reco_user_profile.click_list.author_id",
  "fountain_reco_user_profile.click_list.photo_id",
  "fountain_reco_user_profile.comment_list.author_id",
  "fountain_reco_user_profile.comment_list.photo_id",
  "fountain_reco_user_profile.follow_list.author_id",
  "fountain_reco_user_profile.follow_list.photo_id",
  "fountain_reco_user_profile.like_list.author_id",
  "fountain_reco_user_profile.like_list.photo_id",
  "fountain_reco_user_profile.video_play_stat.photo_id",
  "fountain_reco_user_profile.video_play_stat.author_id",
  "fountain_reco_user_profile.video_play_stat.video_duration",
  "fountain_reco_user_profile.video_play_stat.playing_time",
  "fountain_reco_user_profile.video_play_stat.client_timestamp",
]

rerank_eval_model_send_common_feas_splash_v0 = {
  "featureUId": "uId",
  "featureDeviceId": "dId",
  "featureGender": "uGender",
  "featureAge": "uAge",
  "featureAgeSegment": "uAgeSeg",
  "featureProvinceId": "uProvinceId",
  "featureCityId": "uCityId",
  "featureClientId": "uClientId",
  "featureVisitMod": "uMod",
  "featureVisitNet": "uNetwork",
  "featureRealtimeClickList": "uRealtimeClickList",
  "featureRealtimeLikeList": "uRealtimeLikeList",
  "featureRealtimeFollowList": "uRealtimeFollowList",
  "featureRealtimeForwardList": "uRealtimeForwardList",
  "featureUserProfileV1LikeAidList": "uLikePhotoAuthorList",
  "featureUserProfileV1FollowAidList": "uFollowPhotoAuthorList",
  "featureSourcePId": "featureSourcePId",
  "sourcePidAuthorId": "sourcePidAuthorId",
  "sourcePidFirstLevelCategory": "sourcePidFirstLevelCategory",
  "sourcePidSecondLevelCategory": "sourcePidSecondLevelCategory",
  "sourcePidThirdLevelCategory": "sourcePidThirdLevelCategory",
  "sourcePidHetuLevelOneList": "SourcePidHetuTagLevel1",
  "sourcePidHetuLevelTwoList": "SourcePidHetuTagLevel2",
  "sourcePidHetuLevelThreeList": "SourcePidHetuTagLevel3",
  "sourcePidHetu0": "sourcePidHetu0",
  "sourcePidDnnCluster": "sourcePidDnnCluster",
  "sourcePidDuration": "sourcePidDurationMs"
}

rerank_eval_model_send_common_feas_splash_v1 = {
  "featureUId": "uId",
  "featureDeviceId": "dId",
  "featureGender": "uGender",
  "featureAge": "uAge",
  "featureAgeSegment": "uAgeSeg",
  "featureProvinceId": "uProvinceId",
  "featureCityId": "uCityId",
  "featureClientId": "uClientId",
  "featureVisitMod": "uMod",
  "featureVisitNet": "uNetwork",
  "featureFountainProfileClickPidList": "featureFountainProfileClikPidList",
  "featureUserProfileV1ClickPidList": "uClickPhotoList",
  "featureUserProfileV1FollowPidList": "uFollowPhotoList",
  "featureUserProfileV1LikePidList": "uLikePhotoList",
  "featureUserProfileV1CommentPidList": "uCommentPhotoList",
  "featureRealtimeClickList": "uRealtimeClickList",
  "featureRealtimeLikeList": "uRealtimeLikeList",
  "featureRealtimeFollowList": "uRealtimeFollowList",
  "featureRealtimeForwardList": "uRealtimeForwardList",
  "featureUserProfileV1LikeAidList": "uLikePhotoAuthorList",
  "featureUserProfileV1FollowAidList": "uFollowPhotoAuthorList",
  "featureSourcePId": "featureSourcePId",
  "sourcePidAuthorId": "sourcePidAuthorId",
  "sourcePidFirstLevelCategory": "sourcePidFirstLevelCategory",
  "sourcePidSecondLevelCategory": "sourcePidSecondLevelCategory",
  "sourcePidThirdLevelCategory": "sourcePidThirdLevelCategory",
  "sourcePidHetuLevelOneList": "SourcePidHetuTagLevel1",
  "sourcePidHetuLevelTwoList": "SourcePidHetuTagLevel2",
  "sourcePidHetuLevelThreeList": "SourcePidHetuTagLevel3",
  "sourcePidHetu0": "sourcePidHetu0",
  "sourcePidDnnCluster": "sourcePidDnnCluster",
  "sourcePidDuration": "sourcePidDurationMs",
  "basic_info_gender": "uBasicGender",
}