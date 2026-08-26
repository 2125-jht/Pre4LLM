from rerank import CommonModule

class PicRerankModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  # 精排模型产出的队列
  def fr_queues(self):
    queues = [
      {
        "name": "pctr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pctr_score",
        "raw_score_b_attr": "explore_pic_fullrank_pctr_raw_score_b",
        "raw_score_k_attr": "explore_pic_fullrank_pctr_raw_score_k",
        "raw_score_p_attr": "explore_pic_fullrank_pctr_raw_score_p",
      },
      {
        "name": "pltr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pltr_score",
        "raw_score_b_attr": "explore_pic_fullrank_pltr_raw_score_b",
        "raw_score_k_attr": "explore_pic_fullrank_pltr_raw_score_k",
        "raw_score_p_attr": "explore_pic_fullrank_pltr_raw_score_p",
      },
      {
        "name": "pwtr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pwtr_score",
        "raw_score_b_attr": "explore_pic_fullrank_pwtr_raw_score_b",
        "raw_score_k_attr": "explore_pic_fullrank_pwtr_raw_score_k",
        "raw_score_p_attr": "explore_pic_fullrank_pwtr_raw_score_p",
      },
      {
        "name": "pftr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pftr_score",
      },
      {
        "name": "pcmtr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pcmtr_score",
      },
      {
        "name": "pptr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pptr_score",
      },
      {
        "name": "pcmef",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pcmef_score",
      },
      {
        "name": "pevtr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pevtr_score",
      },
      {
        "name": "fr_score1",
        "power_weight_attr": "explore_pic_power_weight_fullrank_fr_score1_score",
      },
      {
        "name": "fr_score2",
        "power_weight_attr": "explore_pic_power_weight_fullrank_fr_score2_score",
      },
      {
        "name": "pepstr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pepstr_score",
      },
      {
        "name": "pdtr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pdtr_score",
      },
      {
        "name": "pcltr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_pcltr_score",
        "raw_score_b_attr": "explore_pic_fullrank_pcltr_raw_score_b",
        "raw_score_k_attr": "explore_pic_fullrank_pcltr_raw_score_k",
        "raw_score_p_attr": "explore_pic_fullrank_pcltr_raw_score_p",
      },
      {
        "name": "phtr",
        "reverse_order": True,
        "power_weight_attr": "explore_pic_power_weight_htr_score",
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
        "name": "life_ltr_pctr",
        "power_weight_attr": "el_pic_rr_es_power_weight_life_ltr_pctr",
        "raw_score_b_attr": "el_pic_rr_es_life_b_ltr_pctr",
        "raw_score_k_attr": "el_pic_rr_es_life_k_ltr_pctr",
        "raw_score_p_attr": "el_pic_rr_es_life_p_ltr_pctr",
      },
    ]
    return queues

  # LTR队列和提权队列等非精排队列
  def ltr_and_manual_queues(self):
    queues = [
      {
        "name": "consume_time_ltr",
        "power_weight_attr": "explore_pic_power_weight_fullrank_l2r_score"
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
      },
      {
        "name": "pic_ltr_lvtr",
        "power_weight_attr": "explore_pic_power_weight_pic_lvtr",
      },
      {
        "name": "pic_ltr_fvtr",
        "power_weight_attr": "explore_pic_power_weight_pic_fvtr",
        "personalized_power_weight_attr": "pic_ltr_fvtr_personalized_weight",
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
        "default": 1.0,
      },
      {
        "name": "top_video_sim_score",
        "power_weight_attr": "explore_pic_power_weight_top_video_sim_score",
        "default": 1.0,
      },
      {
        "name": "fr_follow_author_pic_boost_score",
        "power_weight_attr": "explore_pic_power_weight_follow_author_pic_boost_score",
        "default": 1.0
      },
      {
        "name": "fr_target_hetu_pic_boost_score",
        "power_weight_attr": "explore_pic_power_weight_target_hetu_pic_boost_score",
        "default": 1.0
      },
      {
        "name": "fr_pic_ensemble_long_caption_score",
        "power_weight_attr": "explore_pic_power_weight_long_caption_score",
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
      },
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

  def quantile_ensemble_queues(self):
    queues = self.fr_queues()
    for queue in queues:
      queue["name"] += "_quantile"
      queue["power_weight_attr"] = "explore_pic_power_weight_" + queue["name"]
    queues.extend(self.ltr_and_manual_queues())
    return queues
  
  def process(self) -> None:
    self.flow \
      .if_("skip_explore_final_pic_rerank == 0", to_be_delete = "date=2024-05-29;committer=xubaoquan") \
        .switch_("explore_pic_rerank_rank_mode") \
        .case_("quantile_rank", to_be_delete = "date=2023-11-16;committer=xubaoquan") \
          .explore_memory_data_enrich(
            data_key = "{{explore_pxtr_quantile_map}}",
            data_type = "string_double_vector_map",
            save_data_ptr_to_attr = "explore_pxtr_quantile",
          ) \
          .explore_global_quantile_xtr_enricher(
            memory_data_map_ptr = "explore_pxtr_quantile",
            prefix = "explore_pxtr_quantile_",
            queues = self.quantile_trans_queues(),
            target_item = {
              "is_picture" : 1
            }
          ) \
          .explore_pic_rerank(
            save_score_to_attr = "fr_pic_ensemble_score",
            limit = "{{explore_pic_rerank_limit}}",
            picture_attr = "is_picture",
            queues = self.quantile_ensemble_queues(),
          ) \
        .default_() \
          .explore_pic_rerank(
            save_score_to_attr = "fr_pic_ensemble_score",
            limit = "{{explore_pic_rerank_limit}}",
            picture_attr = "is_picture",
            queues = self.ensemble_queues(),
          ) \
        .end_() \
        .if_("skip_explore_pic_rerank_variant == 0", to_be_delete = "date=2024-05-29;committer=xubaoquan") \
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
          ) \
        .end_() \
      .end_() \

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
          "hetu_level_one",
          "hetu_level_two",
          "hetu_level_five",
          "pic_ltr_fvtr_personalized_weight",
          "pctr_quantile",
          "pltr_quantile",
          "pwtr_quantile",
          "pftr_quantile",
          "pctr",
          "pltr",
          "pwtr",
          "pftr",
          "top_video_sim_score",
        ],
        for_debug_request_only = True
      )
