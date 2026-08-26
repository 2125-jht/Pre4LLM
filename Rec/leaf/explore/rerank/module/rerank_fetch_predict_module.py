from rerank import CommonModule
from rerank.module.rerank_features import *

class RerankFetchPredict(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
    .if_("enable_use_explore_rerank == 1") \
      .explore_rerank_attr(
          user_info_attr = "user_info_ptr"
      ) \
      .if_("enable_explore_la_rerank_es_score_adjust > 0") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "infer_uv_ctr", "as": "origin_value"},
            {"name": "explore_rerank_ensemble_sort_pure_fr_score2_weight", "as": "pctr_weight"},
            {"name": "explore_rerank_la_ensemble_sort_pfr_score2_weight_max", "as": "weight_max"},
            {"name": "explore_rerank_la_ensemble_sort_pfr_score2_weight_base", "as": "weight_base"}
          ],
          export_common_attr = [
            {"name": "new_pctr_weight", "as": "explore_rerank_ensemble_sort_pure_fr_score2_weight"}
          ],
          function_name = "AdjustFullRankPxtrWeight",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "infer_uv_ctr", "as": "origin_value"},
            {"name": "explore_rerank_ensemble_sort_ensemble_score_weight", "as": "pctr_weight"},
            {"name": "explore_rerank_la_ensemble_sort_fulles_weight_max", "as": "weight_max"},
            {"name": "explore_rerank_la_ensemble_sort_fulles_weight_base", "as": "weight_base"}
          ],
          export_common_attr = [
            {"name": "new_pctr_weight", "as": "explore_rerank_ensemble_sort_ensemble_score_weight"}
          ],
          function_name = "AdjustFullRankPxtrWeight",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_rerank_skip_add_label_aware_fea == 0") \
        .explore_common_user_feature_enricher(
          user_info_attr = "user_info_ptr",
          user_view_pids_attr = "uViewPidListV1",
          user_view_aids_attr = "uViewAidListV1",
          user_effective_view_label_attr = "uEffectiveViewLabelListV1",
          user_long_view_label_attr = "uLongViewLabelListV1",
          user_short_view_label_attr = "uShortViewLabelListV1",
          user_view_hetu1_attr = "uViewHetu1ListV1",
          user_view_hetu2_attr = "uViewHetu2ListV1"
        ) \
      .end_() \
      .enrich_attr_by_light_function(
        export_item_attr = gen_photo_features_for_all_position(10) + ["corr_pctr", "generated_diversity_lists"],
        function_name = "EmptyFunction",
        class_name = "ExploreLightFunctionSetV2",
        item_list_from_attr = "retrieval_list_keys",
      )

    self.get_eval_candidate_list_idx()
    self.flow.if_("enable_explore_rerank_flash_eval_model_predict == 1")
    self.rerank_flash_eval_model()
    self.flow.else_()
    self.rerank_eval_list_predict()
    self.flow.end_()

    # 图文 eval score 计算逻辑
    self.flow.if_("enable_explore_rerank_picture_eval_score == 1")
    self.calc_retrieval_list_picture_eval_score()
    self.flow.end_()
    
    # 多 rerank 分数合并逻辑
    self.flow.if_("enable_explore_rerank_eval_ensemble_score == 1")
    self.calc_rerank_eval_ensemble_score()
    self.flow.end_()

    self.flow \
      .pack_item_attr(
        item_source = {
          "reco_results": False,
          "common_attr": [
            "retrieval_list_keys",
          ]
        },
        mappings = [
          {
            "aggregator": "concat",
            "from_item_attr": "rerank_context",
            "to_common_attr": "rerank_list_score_list",
          },
        ],
      ) \
      .sort(
          score_from_attr = "rerank_context",
          item_list_from_attr = "retrieval_list_keys",
      ) \
      .list_wise_item_attr(
          seq_item_attr_name = "generated_diversity_lists",
          seq_score_attr_name = "rerank_context",
          item_list_from_attr = "retrieval_list_keys",
          ssd_div_score_attr = "ssd_div_score",
          pic_insert_flag_attr = "pic_insert_flag"
      ) \
      .sort(
          stable_sort = True,
          score_from_attr = "virtual_rerank_score",
      )

    self.flow.end_()

  def calc_retrieval_list_picture_eval_score(self):
      return self.flow \
        .calc_by_formula1(
          kconf_key = "formula.scenarioKey79.explore_rerank_picture_evaluator_score",
          import_item_attr = [
            "corr_pctr_psvr",
            "plvtr",
            "awesome_wtd",
            "pctr",
            "pltr",
            "pwtr",
            "pftr",
            "pcmtr",
            "pdtr",
            "pcltr",
            "pptr",
            "pcmef",
            "pepstr",
            "pevtr",
            "fr_score1",
            "fr_score2",
            "phtr",
            "fetr",
            "fountain_eff",
            "is_picture",
            "fr_pic_ensemble_score",
            "explore_fr_ensemble_score"
          ],
          export_formula_value = [
            {"name": "final_score", "as": "picture_evaluator_score"},
          ],
          abtest_biz_name = "KUAISHOU_APPS",
          perf_tag = "{{explore_rerank_picture_evaluator_score_f1_perf_tag}}",
          target_item = {
            "mix_mark" : [1, 2]
          },
        ) \
        .explore_picture_listwise_score_enricher(
          item_list_from_attr = "retrieval_list_keys",
          seq_item_attr_name = "generated_diversity_lists",
          first_pos_weight = "{{explore_rerank_picture_evaluator_first_pos_weight}}",
          last_pos_weight = "{{explore_rerank_picture_evaluator_last_pos_weight}}",
          enable_nolinear = "{{explore_rerank_picture_evaluator_enable_nolinear}}",
          pos_tail_factor = "{{explore_rerank_picture_evaluator_pos_tail_factor}}",
          output_attr = "picture_rerank_context",
        )

  # flash eval 架构下将融分步骤放到 infer 服务 , 输入候选集 + list index , 直接返回最终 list score
  def rerank_flash_eval_model(self):
    return self.flow \
      .explore_custom_trim_user_info(
        user_info_attr = "userInfo",
        save_trimed_user_info_to_attr = "rerank_deep_ltr_trimmed_user_info",
        trim_user_info = rerank_flash_eval_model_send_user_feas,
      ) \
      .delegate_enrich(
        name = "explore_rerank_flash_eval_model",
        kess_service = "{{explore_rerank_flash_eval_model_kess_service}}",
        recv_common_attrs = [
          "eval_list_scores"
        ],
        timeout_ms = 100,
        send_item_attrs = rerank_flash_eval_model_send_item_feas,
        send_common_attrs = rerank_flash_eval_model_send_common_feas,
        request_type = "{{explore_rerank_flash_eval_model_request_type}}",
        partition_size = "{{explore_rerank_flash_eval_model_partition_size}}",
        target_item = {
          "mix_mark" : [1, 2]
        }
      ) \
      .dispatch_common_attr(
        from_common_attr = "eval_list_scores",
        to_item_attr = "rerank_context",
        item_list_from_attr = "retrieval_list_keys",
      )

  def rerank_eval_list_predict(self):
    return self.flow \
      .common_predict(
        loss_function_name = ["l2r_pos" + str(i) for i in range(10)] + ["list_ltr0"],
        loss_default_value = 0.0,
        kess_service = "{{fr_rerank_kai_predict_service}}",
        service_group = "PRODUCTION",
        timeout_ms = 100,
        extra_common_attrs = user_features(),
        item_attrs = gen_photo_features_for_all_position(10),
        item_list_from_attr = "retrieval_list_keys",
      ) \
      .explore_listwise_score_enricher(
        item_attrs = ["l2r_pos" + str(i) for i in range(10)] + ["list_ltr0"],
        item_list_from_attr = "retrieval_list_keys",
        seq_item_attr_name = "generated_diversity_lists",
        topk_num = "{{fr_rerank_predict_topk_num}}",
        pxtr_attr = "corr_pctr",
        pxtr_weight = "{{fr_rerank_predict_pxtr_weight}}",
        loss_name = "{{fr_rerank_predict_service_loss_name}}",
        output_attr = "rerank_context",
      ) \
      .if_("enable_rerank_ensemble_sort == 1") \
        .list_ensemble_sort(
          item_list_from_attr = "retrieval_list_keys",
          output_attr = "rerank_context",
          seq_item_attr_name = "generated_diversity_lists",
          use_proportion = "{{fr_rerank_ensemble_use_proportion}}",
          use_pow_rank = "{{fr_rerank_ensemble_use_power_rank}}",
          fountain_enable_list_ensemble_sort = True,
          fountain_rerank_ensemble_list_weight = "{{fr_rerank_ensemble_list_weight}}",
          queues = [
            {
              "name": "explore_fr_ensemble_score",
              "weight_base": "{{explore_rerank_ensemble_sort_ensemble_score_weight}}"
            },
            {
              "name": "fr_score2",
              "weight_base": "{{explore_rerank_ensemble_sort_pure_fr_score2_weight}}"
            },
            {
              "name": "corr_pctr",
              "weight_base": "{{explore_rerank_ensemble_sort_corr_pctr_weight}}"
            },
            {
              "name": "awesome_wtd",
              "weight_base": "{{explore_rerank_ensemble_sort_awesome_wtd_weight}}"
            },
            {
              "name": "corr_pwtr",
              "weight_base": "{{explore_rerank_ensemble_sort_corr_pwtr_weight}}"
            },
            {
              "name": "pltr",
              "weight_base": "{{explore_rerank_ensemble_sort_pltr_weight}}"
            },
          ]
        ) \
      .end_()

  # 封装获取list_item_idx函数
  def get_eval_candidate_list_idx(self):
    return self.flow \
      .pack_item_attr(
        item_source = {
          "reco_results": False,
          "common_attr": [
            "retrieval_list_keys"
          ],
        },
        mappings = [
          {
            "from_item_attr": "generated_diversity_lists",
            "to_common_attr": "rerank_list_item_list_aggregate",
          },
        ],
      ) \
      .enrich_attr_by_light_function(
        item_list_from_attr = "rerank_list_item_list_aggregate",
        export_item_attr = [
          "rerank_list_enter_index",
        ],
        function_name = "EmptyFunction",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .pack_item_attr(
        item_source = {
          "reco_results": False,
          "common_attr": [
            "rerank_list_item_list_aggregate",
          ],
        },
        mappings = [
          {
            "from_item_attr": "rerank_list_enter_index",
            "to_common_attr": "rerank_list_item_idx_flat_list",
            "default_val": -1,
          },
        ],
      )

  def calc_rerank_eval_ensemble_score(self):
      """
      多 rerank 分数合并逻辑：
      step1 rerank 各业务分数归一化至 [0, 1]
      step2 各业务权重总和为 1 且限制图文业务权重 (picture_rerank_context_weight) 最大值为 0.3
      step3 计算融合权重 = 业务归一化分数 * 业务权重
      """
      return self.flow \
        .normalize_attr(
          item_list_from_attr = "retrieval_list_keys",
          input_attr = "picture_rerank_context",
          output_attr = "picture_rerank_context_norm",
          mode = "min_max_scale",
          default_val = 0.0,
          eps = 1e-15
        ) \
        .normalize_attr(
          item_list_from_attr = "retrieval_list_keys",
          input_attr = "rerank_context",
          output_attr = "rerank_context_norm",
          mode = "min_max_scale",
          default_val = 0.0,
          eps = 1e-15
        ) \
        .gen_common_attr_by_lua(
          # 限制图文 rerank 业务权重
          attr_map={
            "picture_rerank_context_weight": "math.min(explore_rerank_picture_evaluator_max_list_weight_limit, explore_rerank_picture_evaluator_weight)",
            "rerank_context_weight": "1 - math.min(explore_rerank_picture_evaluator_max_list_weight_limit, explore_rerank_picture_evaluator_weight)",
          }
        ) \
        .calc_weighted_sum(
          item_list_from_attr = "retrieval_list_keys",
          fomula_version = 0,
          channels = [
            {"name": "picture_rerank_context_norm", "weight": "{{picture_rerank_context_weight}}"},
            {"name": "rerank_context_norm", "weight": "{{rerank_context_weight}}"},
          ],
          output_item_attr = "rerank_context",
        )
  
  def calc_result_count_to_ab_metric(self):
      return self.flow \
        .cast_attr_type(
          attr_type_cast_configs=[
            {
              "to_type": "double",
              "from_item_attr": "prerank_final_index_photo",
              "to_item_attr": "prerank_final_index_double"
            },
            {
              "to_type": "double",
              "from_item_attr": "cascade_final_index",
              "to_item_attr": "cascade_final_index_double"
            },
            {
              "to_type": "double",
              "from_item_attr": "rank_final_index",
              "to_item_attr": "rank_final_index_double"
            },
            {
              "to_type": "double",
              "from_item_attr": "awesome_wtd_index",
              "to_item_attr": "awesome_wtd_index_double"
            },
            {
              "to_type": "double",
              "from_item_attr": "pctr_index",
              "to_item_attr": "pctr_index_double"
            },
            {
              "to_type": "double",
              "from_item_attr": "pltr_index",
              "to_item_attr": "pltr_index_double"
            },
            {
              "to_type": "double",
              "from_item_attr": "pwtr_index",
              "to_item_attr": "pwtr_index_double"
            },
            {
              "to_type": "double",
              "from_item_attr": "psvr_index",
              "to_item_attr": "psvr_index_double"
            },
          ]
        ) \
        .pack_item_attr(
          item_source = {
            "reco_results": True,
            "total_limit": 10,
          },
          mappings = [
            {
              "aggregator": "avg",
              "from_item_attr": "prerank_final_index_double",
              "to_common_attr": "rerank_top10_prerank_index_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "cascade_final_index_double",
              "to_common_attr": "rerank_top10_cascade_index_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "rank_final_index_double",
              "to_common_attr": "rerank_top10_rank_index_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "awesome_wtd_index_double",
              "to_common_attr": "rerank_top10_awesome_wtd_index_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "pctr_index_double",
              "to_common_attr": "rerank_top10_pctr_index_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "pltr_index_double",
              "to_common_attr": "rerank_top10_pltr_index_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "pwtr_index_double",
              "to_common_attr": "rerank_top10_pwtr_index_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "psvr_index_double",
              "to_common_attr": "rerank_top10_psvr_index_avg"
            },
          ],
          target_item = {"is_picture" : 0}
        ) \
        .count_reco_result(
          save_count_to = "rerank_top10_follow_author_count",
          target_item = {"is_follow_author": 1},
          range_end = 10,
        ) \
        .count_reco_result(
          save_count_to = "rerank_top10_all_page_valid_interest_count",
          target_item = {"is_all_page_valid_interest": 1},
          range_end = 10
        ) \
        .count_reco_result(
          save_count_to = "rerank_top10_new_interest_count",
          target_item = {"is_new_interest_explore": 1},
          range_end = 10
        ) \
        .count_reco_result(
          save_count_to = "rerank_top10_outer_field_interest_count",
          target_item = {"is_outer_field_interest": 1},
          range_end = 10
        ) \
        .count_reco_result(
          save_count_to = "rerank_top10_show_ration_level6_count",
          target_item = {"show_ration_level": 6},
          range_end = 10
        ) \
        .count_reco_result(
          save_count_to = "rerank_top10_explore_show_gt_show_ration_result_count",
          select_item = {
              "attr_name": "explore_stat__real_show_count",
              "compare_to": "{{show_ration_realshow_threshold}}",
              "select_if": ">"
          },
          range_end = 10
        ) \
        .count_reco_result(
          save_count_to = "rerank_top10_bias_interest_count",
          target_item = {"is_bias_interest_tagnex": 1},
          range_end = 10
        ) \
        .send_abtest_metrics(
          metrics = [
            "rerank_top10_bias_interest_count",
            "rerank_top10_follow_author_count",
            "rerank_top10_prerank_index_avg",
            "rerank_top10_cascade_index_avg",
            "rerank_top10_rank_index_avg",
            "rerank_top10_awesome_wtd_index_avg",
            "rerank_top10_pctr_index_avg",
            "rerank_top10_pltr_index_avg",
            "rerank_top10_pwtr_index_avg",
            "rerank_top10_psvr_index_avg",
            "rerank_top10_all_page_valid_interest_count",
            "rerank_top10_new_interest_count",
            "rerank_top10_outer_field_interest_count",
            "rerank_top10_show_ration_level6_count",
            "rerank_top10_explore_show_gt_show_ration_result_count"
          ],
          metric_name_prefix = "explore_reco_leaf_",
        )

  def post_process(self) -> None:
    self.flow.if_("_IS_ABTEST_METRICS_SAMPLING_REQUEST_ == 1 and _IS_ONLINE_SERVICE_ == 1 and _IS_NOT_BACKUP_ == 1")
    self.calc_result_count_to_ab_metric()
    self.flow.end_()
    self.flow \
      .if_("enable_use_explore_rerank == 1") \
        .log_debug_info(
          common_attrs = [
            "explore_rerank_ensemble_sort_ensemble_score_weight",
            "explore_rerank_ensemble_sort_pure_fr_score2_weight",
            "rerank_output_item_key_list_top10",
            "rerank_output_item_key_list",
            "rerank_list_item_idx_flat_list",
            "rerank_list_score_list"
          ],
          item_attrs = [
              "rerank_context",
              "virtual_rerank_score",
          ] + ["l2r_pos" + str(i) for i in range(10)] + ["list_ltr0"],
          item_list_from_attr = "retrieval_list_keys",
          for_debug_request_only = True
        ) \
        .log_debug_info(
          item_attrs = [
              "rerank_context",
          ],
          item_list_from_attr = "retrieval_list_keys",
          for_debug_request_only = True
        ) \
      .end_() \
      .if_("enable_explore_pic_cluster_counter == 1") \
        .explore_pic_cluster_counter_enricher(
          save_pic_cluster_distr_str_attr = "rerank_pic_cluster_distr_str",
          save_long_term_interest_cnt_attr = "rerank_pic_long_term_interest_count",
          save_short_term_interest_cnt_attr = "rerank_pic_short_term_interest_count",
          save_explore_interest_cnt_attr = "rerank_pic_explore_interest_count",
          save_unknown_interest_cnt_attr = "rerank_pic_unknown_interest_count",
          save_pic_cnt_attr = "rerank_pic_count",
          save_hetu_cnt_attr = "rerank_pic_hetu_count",
          long_term_interest_list_attr = "explore_pic_long_interest_list",
          short_term_interest_list_attr = "explore_pic_short_interest_list",
          explore_interest_list_attr = "explore_pic_explore_interest_list",
          hetu_list_attr = "hetu_tag_level_info__hetu_level_one",
          target_item = {"is_picture": 1},
          range_end = 10
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "uPicLongInterestClusterIdList", "as": "long_interest_cluster_list"},
            {"name": "uPicValidInterestClusterIdList", "as": "valid_interest_cluster_list"},
            {"name": "uSingleValidPicCluster7dList", "as": "pic_single_valid_interest_cluster_list"},
            {"name": "uDoubleOutsideValidPicCluster7dList", "as": "pic_double_valid_interest_cluster_list"},
            {"name": "pic_recent_search_cluster_id_632_list", "as": "recent_search_cluster_list"},
          ],
          import_item_attr = [
            "cluster_id_632"
          ],
          export_common_attr = [
            {"name": "cluster_count", "as": "rerank_pic_cluster_count"},
            {"name": "long_interest_count", "as": "rerank_pic_long_interest_count"},
            {"name": "valid_interest_count", "as": "rerank_pic_valid_interest_count"},
            {"name": "pic_single_valid_interest_count", "as": "rerank_pic_single_valid_interest_count"},
            {"name": "pic_double_valid_interest_count", "as": "rerank_pic_double_valid_interest_count"},
            {"name": "has_recent_search_interest_realshow", "as": "rerank_pic_has_recent_search_interest_realshow"},
          ],
          function_name = "CountPicInterestClusterDistribution",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {"is_picture": 1},
          range_end = 10
        ) \
      .end_()

