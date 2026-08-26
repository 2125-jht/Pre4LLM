#!/usr/bin/env python3
# coding=utf-8

def add_queue_attr(queues):
  for queue in queues:
    queue_name = queue["name"]
    if "rank_alpha" not in queue:
      queue["rank_alpha"] = "explore_vrs_" + queue_name + "_" + "rank_alpha"
    if "rank_beta" not in queue:
      queue["rank_beta"] = "explore_vrs_" + queue_name + "_" + "rank_beta"
    if "rank_smooth" not in queue:
      queue["rank_smooth"] = "explore_vrs_" + queue_name + "_" + "rank_smooth"
    if "value_beta" not in queue:
      queue["value_beta"] = "explore_vrs_" + queue_name + "_" + "value_beta"
    if "value_alpha" not in queue:
      queue["value_alpha"] = "explore_vrs_" + queue_name + "_" + "value_alpha"
    if "value_omega" not in queue:
      queue["value_omega"] = "explore_vrs_" + queue_name + "_" + "value_omega"
    if "reverse_order" not in queue:
      queue["reverse_order"] = "explore_vrs_" + queue_name + "_" + "reverse_order"
    if "linear_continuous_multiplication" not in queue:
      queue["linear_continuous_multiplication"] = "explore_vrs_" + queue_name + "_" + "linear_continuous_multiplication"
    if "expand_alpha_scope" not in queue:
      queue["expand_alpha_scope"] = "explore_vrs_" + queue_name + "_" + "expand_alpha_scope"
    if "es_vrs_alpha" not in queue:
      queue["es_vrs_alpha"] = "explore_vrs_" + queue_name + "_" + "es_vrs_alpha"
    if "es_vrs_beta" not in queue:
      queue["es_vrs_beta"] = "explore_vrs_" + queue_name + "_" + "es_vrs_beta"
    if "es_vrs_omega" not in queue:
      queue["es_vrs_omega"] = "explore_vrs_" + queue_name + "_" + "es_vrs_omega"
    if "default" not in queue:
      queue["default"] = 0.0
    if "score_type" not in queue:
      queue["score_type"] = "double"
  return queues

def value_and_rank_score_queues():
  queues = [
    {
      "name": "score_pctr",
      "save_es_queue_score_to_attr": "save_es_pctr_score_to_kafka"
    },
    {
      "name": "score_pltr"
    },
    {
      "name": "score_pwtr"
    },
    {
      "name": "score_pftr"
    },
    {
      "name": "score_pcmtr"
    },
    {
      "name": "score_pptr"
    },
    {
      "name": "score_pcmef"
    },
    {
      "name": "score_pdtr"
    },
    {
      "name": "score_pcltr"
    },
    {
      "name": "score_phtr"
    },
    {
      "name": "duration_ms",
      "score_type" : "integer"
    },
    {
      "name": "score_psvr"
    },
    {
      "name": "pevtr"
    },
    {
      "name": "fr_score1"
    },
    {
      "name": "fr_score2",
    },
    {
      "name": "score_pepstr"
    },
    {
      "name": "corr_fetr"
    },
    {
      "name": "corr_fountain_eff"
    },
    {
      "name": "awesome_wtd_score",
      "save_es_queue_score_to_attr": "save_es_awesome_wtd_score_to_kafka"
    },
    {
      "name": "score_consume_time_ltr"
    },
    {
      "name": "consume_time_pf2r_score"
    },
    {
      "name": "watch_time_fusion_score"
    },
    {
      "name": "corr_cpr"
    },
    {
      "name": "cpr",
    },
    {
      "name": "fountain_eff",
      "save_es_queue_score_to_attr": "save_es_fountain_eff_score_to_kafka"
    },
    {
      "name": "corr_pwtr"
    },
    {
      "name": "corr_pctr"
    },
    {
      "name": "fetr"
    },
    {
      "name": "corr_fr_score2_formula"
    },
    {
      "name": "gen_l2r_fusion_score"
    },
    {
      "name": "multitask_ltr_pfstr"
    },
    {
      "name": "multitask_ltr_pwtd"
    },
    {
      "name": "multitask_ltr_pcvtr"
    },
    {
      "name": "multitask_ltr_pctr"
    },
    {
      "name": "coordinated_watchtime_score"
    },
    {
      "name": "cascade_distill_play_7s",
    },
  ]

  queues = add_queue_attr(queues)

  return queues;
