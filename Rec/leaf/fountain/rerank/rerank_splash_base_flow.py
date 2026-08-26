#!/usr/bin/env python3
# coding=utf-8

from dragonfly.common_leaf_dsl import LeafFlow
from dragonfly.ext.explore.explore_api_mixin import ExploreApiMixin
from dragonfly.ext.subdivision.subdivision_api_mixin import subdivisionApiMixin
from rerank.ab_params import gen_splash_rerank_abtest, abtest_is_or_not_splash_rerank
from rerank.rerank_base import gen_splash_seed_ensemble_queues, gen_variant_config2, gen_variant_config, splash_rerank_expected_value_queues
from rerank.rerank_base import gen_photo_features_for_all_position_new, gen_photo_features_for_all_position_v3, rerank_features_v3
from rerank.rerank_base import rerank_eval_model_send_common_feas_splash_v0, rerank_eval_model_send_common_feas_splash_v1
from rerank.rerank_base import gen_beamsearch_filter_queues
from dump_attr_to_kafka import dump_attr_to_kafka
from util import enrich_ab_param

# 对齐模型训练特征，注释掉线上环境获取不到的item attr
rerank_features = [
    # 调用精排 or 通过rankResult传入
    {"name": "fullrank_sim_pevtr",  "as" : "pPctr"},
    {"name": "fullrank_sim_pltr",  "as" : "pPltr"},
    {"name": "fullrank_sim_pwtr",  "as" : "pPwtr"},
    {"name": "fullrank_sim_pftr",  "as" : "pPftr"},
    {"name": "fullrank_sim_phtr",  "as" : "pPhtr"},
    {"name": "fullrank_sim_plvtr",  "as" : "pPlvtr"},
    {"name": "fullrank_sim_out_pctr",  "as" : "pPsvtr"},
    {"name": "fullrank_sim_pvtr",  "as" : "pPvtr"},
    {"name": "fullrank_sim_pptr",  "as" : "pPptr"},
    {"name": "fullrank_sim_pcmtr",  "as" : "pPcmtr"},
    #{"name": "fullrank_plivewtr",  "as" : "pPlivingwtr"},
    {"name": "fullrank_sim_pcmef",  "as" : "pPcmef"},
    {"name": "fullrank_sim_pepstr",  "as" : "pPepstr"},

    # 通过分布式索引
    {"name": "photo_id",  "as" : "pId"},
    {"name": "author__id",  "as" : "aId"},
    {"name": "author__fans_count",  "as" : "pAuthorFansCount"},
    {"name": "featurePUploadType",  "as" : "pUploadType"},
    {"name": "explore_stat__show_count",  "as" : "pHotShow"},
    {"name": "explore_stat__click_count",  "as" : "pHotClick"},
    {"name": "explore_stat__like_count",  "as" : "pHotLike"},
    {"name": "explore_stat__follow_count",  "as" : "pHotFollow"},
    {"name": "explore_stat__negative_count",  "as" : "pHotHate"},
    {"name": "explore_stat__report_detail__total_report_count",  "as" : "pHotReport"},
    #{"name": "click_upload_rate",  "as" : "pUploadRate"},
    {"name": "featurePCityId",  "as" : "pCityId"},
    {"name": "featurePProvinceId",  "as" : "pProvinceId"},
    {"name": "featurePDurationMs",  "as" : "pDurationMs"},
    {"name": "content_safety_level_with_namespace__level_hot_online",  "as" : "pContentLevel"},
    #{"name": "author.gender",  "as" : "pAuthorGender"},
    {"name": "featurePHetuTagLevel1",  "as" : "pHetuTagLevel1Id"},
    {"name": "featurePHetuTagLevel2",  "as" : "pHetuTagLevel2Id"},
    {"name": "featurePDnnClusterId",  "as" : "pDnnClusterId"},
    {"name": "mmu_img_cluster_v1",  "as" : "pMmuImgClusterV1"},
    {"name": "featurePMmuImgClusterV3",  "as" : "pMmuImgClusterV3"},
    {"name": "mmu_content_id",  "as" : "pMmuContentId"},
    {"name": "featurePMusic",  "as" : "pMusic"},
    {"name": "featurePMusicComboId",  "as" : "pMusicComboId"},
    {"name": "featurePOcrCoverTextWordCount",  "as" : "pOcrCoverTextWordCount"},

    # 通过rpc传入rankResult对齐recoPhotoInfo的口径
    {"name": "cascade_pctr",  "as" : "pMcPctr"}, # 粗排分从RPC传过来的RankResult里拿
    {"name": "cascade_pltr",  "as" : "pMcPltr"},
    {"name": "cascade_pwtr",  "as" : "pMcPwtr"},
    {"name": "cascade_plvtr",  "as" : "pMcPlvtr"},
    {"name": "cascade_psvtr",  "as" : "pMcPsvtr"},
    {"name": "empirical_ctr",  "as" : "pEmpCtr"},
    {"name": "empirical_ltr",  "as" : "pEmpLtr"},
    {"name": "empirical_wtr",  "as" : "pEmpWtr"},
    {"name": "empirical_ftr",  "as" : "pEmpFtr"},
    {"name": "empirical_ptr",  "as" : "pEmpPtr"},
    {"name": "empirical_cmtr",  "as" : "pEmpCmtr"},
    {"name": "empirical_htr",  "as" : "pEmpHtr"},
    {"name": "empirical_watchtime",  "as" : "pAvgWatchtime"},
    # 暂时没找到, 获取相对比较麻烦 先不传入
    # {"name": "living",  "as" : "pHotLiving"},
    # 这个需要问一下
    {"name": "reason",  "as" : "pHotExptag"},
]

class RerankSplashBaseFlow(LeafFlow, subdivisionApiMixin, ExploreApiMixin):
  def __init__(self):
    LeafFlow.__init__(self, "rerank_splash_base")
    self \
      .namespace_(ns = "rerank_splash_base", nest = True) \
      ._count_photo_type_distribution("leaf_show_splash") \
      ._rerank() \
      .enrich_attr_by_light_function(
        export_item_attr = ["fullrank_ensemble_score"],
        function_name = "EmptyFunction",
        class_name = "ExploreLightFunctionSetV2",
      ) \

    dump_attr_to_kafka(
      self,
      stage_name = "rerank",
      dump_item_attr_list = [
        "virtual_rerank_score",
        "fullrank_ensemble_score"
      ],
    )

    cpu_cost_debug_info = [
      "retrieval_splash_cpu_cost_ts",
      "filter_splash_cpu_cost_ts",
      "cascade_splash_cpu_cost_ts",
      "full_rank_splash_cpu_cost_ts",
      "post_process_splash_cpu_cost_ts",
      "rerank_splash_cpu_cost_ts",
    ]

    self \
      .count_reco_result(save_count_to = "current_reco_result_cnt") \
      .copy_attr(
        attrs = [
          {
            "from_common": "fountain_reco_leaf_retrieval_splash_ts",
            "to_common": "retrieval_splash_ts"
          },
          {
            "from_common": "fountain_reco_leaf_retrieval_splash_cpu_cost_ts",
            "to_common": "retrieval_splash_cpu_cost_ts"
          },
          {
            "from_common": "fountain_reco_leaf_filter_splash_ts",
            "to_common": "filter_splash_ts"
          },
          {
            "from_common": "fountain_reco_leaf_filter_splash_cpu_cost_ts",
            "to_common": "filter_splash_cpu_cost_ts"
          }
        ]
      ) \
      .copy_user_meta_info(
        save_flow_cpu_cost_to_attr = "rerank_splash_cpu_cost_ts",
      ) \
      .gen_common_attr_by_lua(
        attr_map = {
          "rank_splash_ts": "util.GetTimestamp() - rank_splash_begin_ts",
          "total_splash_ts": "util.GetTimestamp() - prepare_begin_ts",
          "request_splash_count": "1",
          "empty_result_count": "current_reco_result_cnt == 0"
        },
      ) \
      .explore_environment_type_enrich(
        type_map = {
          "fountainRecoLeaf": "prod",
          "fountainLeaf_2022Q1combo": "prod", # 扩索引实验 hold leaf
          "fountainRecoLeafGray": "gray",
          "fountainRecoLeafAbGray": "gray",
          "fountainRecoLeafV2": "gray",
          "fountainRecoLeafTest": "gray",
          "default": "other",
        },
        save_type_to_attr = "env_type",
      ) \
      .gen_common_attr_by_lua(
        attr_map = {
          "_IS_ONLINE_SERVICE_": "env_type == \"prod\" or env_type == \"gray\"",
        }
      ) \
      .send_abtest_metrics(
        skip = "{{return _IS_ONLINE_SERVICE_ == 0}}",
        metrics = [
          "retrieval_splash_ts",
          "filter_splash_ts",
          "cascade_splash_ts",
          "rank_splash_ts",
          "total_splash_ts",
          "retrieval_splash_cpu_cost_ts",
          "filter_splash_cpu_cost_ts",
          "cascade_splash_cpu_cost_ts",
          "full_rank_splash_cpu_cost_ts",
          "post_process_splash_cpu_cost_ts",
          "rerank_splash_cpu_cost_ts",
          "filter_finish_splash_single_picture_count",
          "filter_finish_splash_long_picture_count",
          "filter_finish_splash_cluster_picture_count",
          "leaf_show_splash_single_picture_count",
          "leaf_show_splash_long_picture_count",
          "leaf_show_splash_cluster_picture_count",
          "filter_finish_splash_item_num",
          "request_splash_count",
          "rank_splash_model_input_count",
          "empty_result_count",
          "filter_splash_related_score_filtered_success"
        ],
        metric_name_prefix = "fountain_reco_leaf_"
      ) \
      .log_debug_info(
        common_attrs = cpu_cost_debug_info,
        for_debug_request_only = True,
      ) \
      .namespace_()

  def _rerank(self):
    self \
    .get_abtest_params(
        biz_name = "RECO_RPC",
        ab_params = enrich_ab_param(gen_splash_rerank_abtest()),
        prioritized_suffix = "{{_ABTEST_SUFFIX_LIST_}}"
    ) \
    .get_abtest_params(
      biz_name = "RECO_RPC",
      ab_params =  abtest_is_or_not_splash_rerank,
      prioritized_suffix = "{{_ABTEST_SUFFIX_LIST_}}"
    ) \
    .get_abtest_params(
        biz_name = "MOBILE",
        prioritized_suffix = "{{_ABTEST_SUFFIX_LIST_}}",
        ab_params = [{
            "param_name" : "enable_fountain_ui",
            "param_type" : "bool",
            "default_value" : True,
            "attr_name" : "enable_fountain_ui",
        },
        {
            "param_name" : "enable_fountain_rerank_v4_dpp_add_new_server",
            "param_type" : "bool",
            "default_value" : True,
            "attr_name" : "enable_fountain_rerank_v4_dpp_add_new_server",
        },
        ]
    ) \
    .enrich_attr_by_lua(
        import_common_attr = [
            "enable_use_fountain_splash_rerank",
            "enable_fountain_rerank_v4_dpp_add_new_server"
        ],
        export_common_attr = [
            "enable_use_fountain_splash_rerank"
        ],
        function_for_common = "handle_common_attr_new_ab_splash",
        lua_script_file = "fountain/rerank/lua/all_lua.lua",
    ) \
    .log_debug_info(
        common_attrs = [
            "request_num",
            "enable_use_fountain_splash_rerank"
        ],
        for_debug_request_only = True,
    ) \
    .if_("enable_use_fountain_splash_rerank == 1 and (fountain_rerank_enable_use_fountain_ui == 1 or enable_fountain_ui == 1) and (only_rerank_request_num_is_2 == 0 or request_num == 10)") \
        .if_("skip_fountain_splash_rerank_limit == 0") \
            .if_("fountain_splash_rerank_enable_control_diversity_quota == 1") \
              .explore_control_hetu_count_arranger(
                duration_ms_attr = "duration_ms",
                enable_duration_control_diversity = "{{fountain_splash_rerank_enable_duration_control_diversity}}",
                duration_control_diversity_start = "{{fountain_splash_rerank_duration_control_diversity_start}}",
                keep_size = "{{fountain_splash_rerank_limit_size}}",
                duration_0_7s_max_size = "{{fountain_splash_rerank_control_duration_0_7s_max_size}}",
                duration_7_9s_max_size = "{{fountain_splash_rerank_control_duration_7_9s_max_size}}",
                duration_9_12s_max_size = "{{fountain_splash_rerank_control_duration_9_12s_max_size}}",
                duration_12_17s_max_size = "{{fountain_splash_rerank_control_duration_12_17s_max_size}}",
                duration_17_20s_max_size = "{{fountain_splash_rerank_control_duration_17_20s_max_size}}",
                duration_20_58s_max_size = "{{fountain_splash_rerank_control_duration_20_58s_max_size}}",
              ) \
            .else_() \
              .truncate(
                  size_limit = "{{fountain_splash_rerank_limit_size}}",
              ) \
            .end_if_() \
        .end_if_() \
        .if_("fountain_enable_rerank_full_link_sample_splash == 1") \
            .copy_item_meta_info(
                save_item_seq_to_attr = "item_seq"
            ) \
        .end_() \
        .log_debug_info(
            common_attrs = [
                "request_num",
                "only_rerank_request_num_is_2"
            ],
            for_debug_request_only = True,
        ) \
        .copy_item_meta_info(
            save_item_type_to_attr = "item_type",
            save_reason_to_attr = "reason",
        ) \
        .copy_attr(
            attrs=[{
                "from_common": "fountain_splash_rerank_duration_adjust_param",
                "to_common": "fountain_rerank_duration_adjust_param"
            },
            {
                "from_common": "fountain_splash_rerank_duration_adjust_level",
                "to_common": "fountain_rerank_duration_adjust_level"
            }]
        ) \
        .enrich_attr_by_lua(
            import_common_attr = [
                "fountain_rerank_duration_adjust_param",
                "fountain_rerank_duration_adjust_level",
                "score_factor_coffe"
            ],
            import_item_attr = [
                "fullrank_sim_out_pctr",
                "fullrank_sim_pevtr",
                "fullrank_sim_pltr",
                "fullrank_sim_pwtr",
                "fullrank_sim_pftr",
                "fullrank_sim_longview_score_no_bias",
                "fullrank_sim_psvr",
                "fullrank_sim_pptr",
                "fullrank_sim_pcmtr",
                "fullrank_sim_pcmef",
                "fullrank_sim_pwatchtime_no_bias",
                "fullrank_sim_pvtr",
                "fullrank_sim_pepstr",
                "fullrank_ltr_score",
                "fullrank_ensemble_score",
                "fullrank_detail_new_pevtr_v2",
                "fullrank_sim_lstr",
                "fullrank_sim_pcltr",
                "fullrank_sim_plvtr",
                "fullrank_sim_pfintr",
                "fullrank_ltr_v4_fountain_finish_rate",
                "fullrank_ltr_v4_fountain_next",
                "fountain_splash_slide",
                "fullrank_opportunity_cost_score",
                "fullrank_ada_xtr_score",
                "fullrank_trans_pvtr_score",
                "fullrank_act_ctr",
                "fullrank_cl_play_time",
                "topk_mgs_expected_score",
                "fullrank_sim_pcpr",
                "fullrank_act_wtd",
                "fullrank_pure_value_score",
                "duration_ms",
                "fullrank_ori_pswptr"
            ],
            export_item_attr = [
                "fullrank_click_score",
                "fullrank_like_score",
                "fullrank_follow_score",
                "fullrank_forward_score",
                "fullrank_profile_score",
                "fullrank_comment_score",
                "fullrank_longview_score",
                "fullrank_shortview_score",
                "fullrank_shortviewinorder_score",
                "fullrank_watchtime_score",
                "fullrank_watchtime_ori_score",
                "fullrank_l2r_score",
                "fullrank_pepstr_score",
                "fullrank_outctr_score",
                "fullrank_neg_feedback_discount_score",
                "fullrank_evtr_v2_score",
                "fullrank_lstr_score",
                "fullrank_collect_score",
                "fullrank_cmef_score",
                "fullrank_lvtr_ori_score",
                "fullrank_pfintr_score",
                "fullrank_finish_score",
                "fullrank_next_score",
                "fullrank_slide_score",
                "fullrank_opportunity_cost_score",
                "fullrank_ada_xtr_score",
                "fullrank_trans_pvtr_score",
                "fullrank_act_ctr_score",
                "fullrank_topk_mgs_expected_score",
                "fullrank_pcpr_score",
                "fullrank_act_wtd_score",
                "fullrank_pure_value_score",
                "fullrank_fusion_pctr_score",
                "fullrank_fusion_pcltr_score",
                "fullrank_cl_play_time_score",
                "fullrank_min_act_rank_reci_score",
                "fullrank_ori_pswptr_score"
            ],
            function_for_item = "full_rank_score_cal_splash",
            lua_script_file = "fountain/rerank/lua/all_lua.lua",
        ) \
        .enrich_attr_by_lua(
            import_item_attr = [
                "explore_stat__real_show_count",
                "explore_stat__click_count",
                "explore_stat__like_count",
                "explore_stat__follow_count",
                "explore_stat__forward_count",
                "explore_stat__profile_enter_count",
                "explore_stat__comment_count",
                "explore_stat__negative_count",
            ],
            export_item_attr = [
                "empirical_ctr",
                "empirical_ltr",
                "empirical_wtr",
                "empirical_ftr",
                "empirical_ptr",
                "empirical_cmtr",
                "empirical_htr",
            ],
            function_for_item = "splash_calculate",
            lua_script_file = "fountain/rerank/lua/all_lua.lua",
        ) \
        .enrich_attr_by_lua(
            import_item_attr = [
              "duration_ms"
            ],
            export_item_attr = [
              "duration_0_7s",
              "duration_7_9s",
              "duration_9_12s",
              "duration_12_17s",
              "duration_17_20s",
              "duration_20_58s",
              "duration_gt_58s",
              "duration_gt_120s",
            ],
            function_for_item = "splash_convert_photo_info_attr",
            lua_script_file = "fountain/rerank/lua/all_lua.lua",
        ) \
        .log_debug_info(
            item_attrs = [
                "item_type",
                "reason",
                "mmu_content_id",
                "mmu_img_cluster_v1",
                "author__fans_count",
                "fullrank_finish_score",
                "fullrank_next_score",
                "fullrank_slide_score",
                "fullrank_opportunity_cost_score",
                "fullrank_ada_xtr_score",
                "fullrank_trans_pvtr_score",
                "fullrank_topk_mgs_expected_score",
                "fullrank_pcpr_score",
                "fullrank_act_wtd_score",
                "fullrank_pure_value_score",
            ],
            item_num_limit = 10,
            target_item = {
                "item_type": [0, 1],
            },
            for_debug_request_only = True,
        ) \
        .if_("enable_rerank_splash_cal_user_group_emp_xtr == 1") \
          .calc_splash_rerank_user_custom_ensemble_weight() \
        .end_if_() \
        .if_("enable_rerank_splash_ten_user_group_adjust_wtachtime == 1") \
          .gen_common_attr_by_lua(
            attr_map = {
              "fountain_splash_rerank_gen_seed_ensemble_fullrank_watchtime_score": "fountain_splash_rerank_gen_seed_ensemble_fullrank_watchtime_score * user_group_emp_playtime"
            }
          ) \
        .end_() \
        .if_("enable_rerank_splash_ten_user_group_adjust_pfintr == 1") \
          .gen_common_attr_by_lua(
            attr_map = {
              "fountain_splash_rerank_gen_seed_ensemble_fullrank_pfintr_score": "fountain_splash_rerank_gen_seed_ensemble_fullrank_pfintr_score * user_group_emp_playtime"
            }
          ) \
        .end_() \
        .if_("enable_fountain_splash_rerank_gen_ensemble_list == 1") \
            .gen_ensemble_second_sequence(
                max_sequence_num = "{{fountain_splash_rerank_gen_seed_ensemble_max_sequence_num}}",
                origin_seq_range = "{{fountain_splash_rerank_gen_seed_ensemble_origin_seq_range}}",
                sequence_max_size = "{{fountain_splash_rerank_gen_seed_ensemble_seq_max_size}}",
                gen_final_seq_max_size = "{{fountain_splash_rerank_gen_final_seq_max_size}}",
                return_item_type = 2,
                target_item = { "item_type": [0, 1], },
                queues = gen_splash_seed_ensemble_queues(),
                proportion_temperature = "{{fountain_rerank_gen_seed_ensemble_proportion_temperature}}",
                use_proportion = "{{fountain_splash_rerank_gen_seed_ensemble_use_proportion}}",
                use_pow_rank = "{{fountain_splash_rerank_gen_seed_ensemble_use_pow_rank}}",
                discount_map = "{{fountain_splash_rerank_discount_map}}",
                enable_gen_ensemble_use_order_13 = "{{fountain_splash_rerank_enable_gen_ensemble_use_order_13}}",
                enable_gen_ensemble_use_order_23 = "{{fountain_splash_rerank_enable_gen_ensemble_use_order_23}}",
                hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
            ) \
        .end_() \
        .if_("enable_fountain_splash_model_mdp_gen_list == 1") \
            .fountain_calc_ensemble_score(
                save_score_to_attr = "rerank_expected_value_score_splash",
                user_power_calc_v2 = "{{fountain_rerank_expected_value_enable_power_calc_v2_splash}}",
                use_rank_with_absolute_score = "{{fountain_rerank_expected_value_use_rank_with_absolute_score_splash}}",
                queues = splash_rerank_expected_value_queues,
            ) \
            .explore_mdp_gen_list_enricher(
                candidate_size = "{{fountain_splash_mdp_gen_list_candidate_size}}",
                output_len = "{{fountain_splash_mdp_gen_list_output_len}}",
                item_value_attr_name = "rerank_expected_value_score_splash",
                item_next_attr_name = "fountain_splash_slide", # 首屏 next 预估
                output_attr = "retrieval_list_keys_9",
                pnext_alpha = "{{fountain_splash_mdp_gen_list_pnext_alpha}}",
                pnext_beta = "{{fountain_splash_mdp_gen_list_pnext_beta}}",
                beam_size = "{{fountain_splash_mdp_gen_list_beam_size}}",
                value_mean = "{{fountain_splash_mdp_gen_list_value_mean}}",
                seq_item_attr_name = "generated_variant_lists",
                return_item_type = 2,
            ) \
        .end_if_() \
        .split_string(
            input_common_attr = "fountain_splash_rerank_bs_search_num_ab_str",
            output_common_attr = "fountain_splash_rerank_bs_search_num_list",
            delimiters = ",",
            trim_spaces = True,
            skip_empty_tokens = True,
            parse_to_int = True,
        ) \
        .gen_beamsearch_sequence(
            list_wise_item_type = 2,
            reason = 5,
            target_item = {
                "item_type" : [0, 1],
            },
            queues = gen_beamsearch_filter_queues(),
            enable_fountain_rerank_use_beamsearch = "{{enable_fountain_splash_rerank_use_beamsearch}}",
            fountain_rerank_beamsearch_size = "{{fountain_rerank_splash_beamsearch_size}}",
            fountain_rerank_beamsearch_rate_type = "{{fountain_rerank_splash_beamsearch_rate_type}}",
            fountain_rerank_beamsearch_max_len = "{{fountain_rerank_splash_beamsearch_max_len}}",
            bs_fix_top1_method = "{{fountain_splash_rerank_bs_fix_top1_method}}",
            bs_search_num_attr = "fountain_splash_rerank_bs_search_num_list",
            rerank_expected_value_score_splash_attr = "rerank_expected_value_score_splash",
        ) \
        .retrieve_by_common_attrs(
            attrs = [{
                "name":"retrieval_list_keys_1",
                "reason":66701,
            }, {
                "name":"retrieval_list_keys_5",
                "reason":66705,
            }, {
                "name":"retrieval_list_keys_9",
                "reason":66709,
            }],
            reset_existing_item_attrs = False
        ) \
        .list_variant_diversity(
            page_size = "{{fountain_splash_rerank_gen_final_seq_max_size}}",
            use_type = 1,
            variant_config = gen_variant_config2(),
            save_decay_score_to_attr = "variant_score",
            enable_skip_diversity = "{{fountain_splash_rerank_enable_skip_diversity}}",
            enable_fix_variant_attr = "{{rerank_enable_fix_variant_attr}}",
        ) \
        .retrieve_by_common_attr(
            attr = "retrieval_list_keys_2",
            reason = 66702,
            reset_existing_item_attrs = False
        ) \
        .copy_item_meta_info(
            save_item_type_to_attr = "item_type",
            save_reason_to_attr = "reason",
        ) \
        .filter_by_attr(
            attr_name = "item_type",
            remove_if = "==",
            compare_to = 2,
            remove_if_attr_missing = False,
        ) \
        .list_variant_filter(
            seq_item_attr_name = "generated_diversity_lists",
            max_retrieval_num = "{{fountain_splash_rerank_max_retrieval_num}}",
            target_item = { "item_type": 3 },
            variant_config = gen_variant_config(),
        ) \
        .rerank_gen_model() \
        .explore_rerank_collect_single_score_candidate(
          candidates = [
            {
              "source": "retrieval_list_keys_7",
              "enable": "{{fountain_splash_rerank_enable_gen_ar_model}}",
              "score_attr": "rerank_gen_score_list",
              "list_size": "{{fountain_rerank_gen_model_beam_size}}",
              "save_candidate_to_attr": "rerank_gen_ar_candidate",
            },
          ],
        ) \
        .explore_rerank_gen_list_by_model_old(
            candidate_attrs = [
                "rerank_gen_ar_candidate",
            ],
            item_type = 3,
            seq_len = "{{fountain_splash_rerank_gen_final_seq_max_size}}",
            target_item = {
                "item_type" : [0, 1],
            },
        ) \
        .if_("enable_fountain_splash_rerank_gen_original_list == 1") \
            .gen_original_sequence(
                item_type = 3,
                reason = 4,
                sequence_length = "{{fountain_splash_rerank_gen_final_seq_max_size}}",
                output_sequence_attr_name = "generated_diversity_lists",
                target_item = {
                    "item_type" : [0, 1],
                },
            ) \
        .end_() \
        .retrieve_by_common_attrs(
            attrs = [{
                "name":"retrieval_list_keys_4",
                "reason":66704,
            }, {
                "name":"retrieval_list_keys_7",
                "reason":66707,
            }],
            reset_existing_item_attrs = False
        ) \
        .copy_item_meta_info(
            save_item_type_to_attr = "item_type",
            save_reason_to_attr = "reason",
        ) \
        .deduplicate() \
        ._gen_list_reason_metrics_before_eval() \
        .if_("fountain_splash_skip_rerank_predict == 0") \
          .switch_("fountain_splash_rerank_evaluator_model_version") \
            .case_(1) \
              .rerank_eval_model_predict(rerank_features_v3, gen_photo_features_for_all_position_v3(2), rerank_eval_model_send_common_feas_splash_v1) \
            .case_(2) \
              .rerank_flash_eval_model() \
            .default_() \
              .rerank_eval_model_predict(rerank_features, gen_photo_features_for_all_position_new(2), rerank_eval_model_send_common_feas_splash_v0) \
          .end_() \
        .end_() \
        .enrich_attr_by_light_function(
          export_item_attr = [
            "rerank_context",
          ],
          function_name = "EmptyFunction",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "item_type": 3,
          },
        ) \
        .sort(
            score_from_attr = "rerank_context",
            target_item = {
                "item_type": 3,
            },
        ) \
        .if_("fountain_splash_rerank_use_seed_seq == 1") \
            .list_wise_item_attr(
                seq_item_attr_name = "generated_variant_lists",
                seq_score_attr_name = "rerank_context",
                target_item = {
                    "item_type": 3,
                },
            ) \
        .else_() \
            .list_wise_item_attr(
                seq_item_attr_name = "generated_diversity_lists",
                seq_score_attr_name = "rerank_context",
                target_item = {
                    "item_type": 3,
                },
            ) \
        .end_() \
        .limit( # 评估后只需要 top1 list
            size = 1,
            target_item = {
                "item_type": 3,
            },
        ) \
        ._gen_list_reason_metrics_after_eval() \
        .sort(
            name = "fountain_splash_rr",
            traceback = True,
            stable_sort = True,
            score_from_attr = "virtual_rerank_score",
        ) \
        .log_debug_info(
            target_item = {
                "item_type": 1,
            },
            item_num_limit = 2,
            for_debug_request_only = True,
        ) \
        .filter_by_attr(
            attr_name = "item_type",
            remove_if = ">=",
            compare_to = 2,
            remove_if_attr_missing = False,
        ) \
        .if_("fountain_splash_enable_write_rerank_top_result_to_redis == 1") \
            .write_rerank_top_result_to_redis() \
        .end_() \
        .if_("fountain_enable_rerank_full_link_sample_splash == 1") \
            .rerank_full_link_sample_splash() \
        .end_() \
    .end_if_() \
    .perflog_reason_count(
        check_point = "rerank_finish",
        range_end = "{{request_num}}"
    )
    return self

  def rerank_eval_model_predict(self, item_attr_map, item_features_flat, common_features_map):
    # 抽出方便支持不同长度的 list
    self \
      .list_wise_seq_attr(
        upload_time_attr = "upload_time",
        duration_ms_attr = "duration_ms",
        hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
        hetu_level_two_attr = "hetu_tag_level_info__hetu_level_two",
        hetu_level_three_attr = "hetu_tag_level_info__hetu_level_three",
        hetu_level_four_attr = "hetu_tag_level_info__hetu_level_four",
        hetu_level_five_attr = "hetu_tag_level_info__hetu_level_five",
        item_attrs_transform_map = item_attr_map,
        seq_item_attr_name = "generated_diversity_lists",
        target_item = {
          "item_type": 3,
        },
        output_attrs = item_features_flat
      ) \
      .list_item_predict(
        kess_service = "{{fountain_splash_rerank_kess_service}}",
        service_group = "PRODUCTION",
        origin_seq_reason = "{{fountain_splash_rerank_origin_reason}}",
        timeout_ms = "{{fountain_splash_rerank_predict_timeout_ms}}",
        layer_name = "{{fountain_splash_rerank_predict_layer_name}}",
        loss_name = "{{fountain_splash_rerank_predict_loss_name}}",
        enable_use_kai = "{{fountain_splash_rerank_predict_use_kai}}",
        rerank_list_next_weight = "{{fountain_splash_rerank_list_next_weight_new}}",
        enable_use_list_score = "{{fountain_splash_rerank_enable_use_list_score}}",
        item_next_attr = "{{fountain_splash_rerank_item_next_score_attr}}",
        shard_num = 1,
        output_attr = "rerank_context",
        output_list_attr = "rerank_item_score",
        disable_common_attr_missing_warning = True,
        use_odd_score = "{{fountain_rerank_splash_use_odd2}}",
        enable_use_pos_context_score = "{{fountain_rerank_splash_enable_pos_context}}",
        last_coeff = "{{fountain_rerank_splash_last_coeff2}}",
        enable_use_pos_next_multiply = "{{fountain_rerank_splash_enable_pos_next2}}",
        target_item = {
          "item_type": 3,
        },
        attr_name_transform_map = common_features_map,
        item_attrs = item_features_flat,
      )
    return self

  def rerank_gen_model(self):
    self \
      .if_("fountain_splash_rerank_enable_gen_ar_model == 1") \
        .explore_custom_trim_user_info(
          user_info_attr = "userInfo",
          save_trimed_user_info_to_attr = "rerank_deep_ltr_trimmed_user_info",
          trim_user_info = [
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
            "fountain_reco_user_profile.follow_list.author_id",
            "fountain_reco_user_profile.follow_list.photo_id",
            "fountain_reco_user_profile.like_list.author_id",
            "fountain_reco_user_profile.like_list.photo_id",
            "fountain_reco_user_profile.video_play_stat.photo_id",
            "fountain_reco_user_profile.video_play_stat.author_id",
            "fountain_reco_user_profile.video_play_stat.video_duration",
            "fountain_reco_user_profile.video_play_stat.playing_time",
            "realtime_click_list",
            "realtime_follow_list",
            "realtime_forward_list",
            "realtime_like_list",
          ],
        ) \
        .delegate_enrich(
          target_item = { "item_type": [0, 1], },
          kess_service = "{{fountain_splash_rerank_gen_ar_model_kess_service}}",
          recv_item_attrs = [
            {"name": "rerank_gen_score", "as": "rerank_gen_score_list"}
          ],
          timeout_ms = 100,
          send_item_attrs = [
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
          ],
          send_common_attrs = [
            { "name": "rerank_deep_ltr_trimmed_user_info", "as": "user_info_str" },
            { "name": "featureSourcePId", "as": "source_pid" },
            { "name": "sourcePidAuthorId", "as": "source_aid" },
            { "name": "sourcePidHetuLevelOneList", "as": "source_hetu_tag_level1_list" },
            { "name": "sourcePidHetuLevelTwoList", "as": "source_hetu_tag_level2_list" },
            { "name": "sourcePidDuration", "as": "source_duration_ms" },
            { "name": "sourcePidTagId", "as": "source_tag" },
          ],
          request_type = "{{fountain_splash_rerank_gen_ar_model_request_type}}",
          partition_size = "{{fountain_splash_rerank_gen_ar_model_partition_size}}",
        ) \
      .end_()
    return self

  # flash eval 架构下将融分步骤放到 infer 服务 , 输入候选集 + list index , 直接返回最终 list score
  # list 长度 6 , 所有 list 通过 retrieval_list_keys 得到 . 推全后将 rerank_list_item_idx_flat_list 发送全链路样本流
  def rerank_flash_eval_model(self):
    return self \
      .pack_common_attr(
        input_common_attrs = ["retrieval_list_keys_2", "retrieval_list_keys_4", "retrieval_list_keys_7"],
        output_common_attr = "retrieval_list_keys",
        deduplicate = True
      ) \
      .enrich_attr_by_light_function(
        item_list_from_attr = "retrieval_list_keys",
        export_item_attr = [
          "generated_diversity_lists",
        ],
        function_name = "EmptyFunction",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .pack_item_attr(
        item_source = {
          "reco_results": False,
          "common_attr": [
            "retrieval_list_keys",
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
          "item_seq",
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
            "from_item_attr": "item_seq",
            "to_common_attr": "rerank_list_item_idx_flat_list",
          },
        ],
      ) \
      .explore_custom_trim_user_info(
        user_info_attr = "userInfo",
        save_trimed_user_info_to_attr = "rerank_deep_ltr_trimmed_user_info",
        trim_user_info = [
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
        ],
      ) \
      .delegate_enrich(
        kess_service = "{{fountain_splash_rerank_eval_mtl_model_kess_service}}",
        recv_common_attrs = [
          "eval_list_scores"
        ],
        timeout_ms = 50,
        send_item_attrs = [
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
        ],
        send_common_attrs = [
          { "name": "rerank_deep_ltr_trimmed_user_info", "as": "user_info_str" },
          { "name": "featureSourcePId", "as": "source_pid" },
          { "name": "sourcePidAuthorId", "as": "source_aid" },
          { "name": "sourcePidHetuLevelOneList", "as": "source_hetu_tag_level1_list" },
          { "name": "sourcePidHetuLevelTwoList", "as": "source_hetu_tag_level2_list" },
          { "name": "sourcePidDuration", "as": "source_duration_ms" },
          { "name": "sourcePidTagId", "as": "source_tag" },
          { "name": "featureSimilarUserList", "as": "similar_user_list" },
          "rerank_list_item_idx_flat_list",
          "page",
        ],
        request_type = "{{fountain_splash_rerank_eval_mtl_model_request_type}}",
        partition_size = "{{fountain_splash_rerank_eval_mtl_model_partition_size}}",
        target_item = {
          "item_type": [0, 1],
        },
      ) \
      .dispatch_common_attr(
        from_common_attr = "eval_list_scores",
        to_item_attr = "rerank_context",
        target_item = {
          "item_type": 3,
        },
      )

  def write_rerank_top_result_to_redis(self):
    self \
    .if_("fountain_splash_rerank_top_result_enable_reason_select == 1") \
      .split_string(
        input_common_attr = "fountain_splash_rerank_top_result_select_reasons",
        output_common_attr = "fountain_splash_rerank_top_result_select_reasons_list",
        delimiters = ",",
        trim_spaces = True,
        skip_empty_tokens = True,
        parse_to_int = True,
      ) \
    .else_() \
      .set_attr_value(
        common_attrs=[
          {
            "name": "fountain_splash_rerank_top_result_select_reasons_list",
            "type": "int_list",
            "value": []
          }
        ]
      ) \
    .end_() \
    .pack_item_attr(
      item_source = {
        "reco_results": True
      },
      mappings = [{
        "from_item_attr": "photo_id",
        "to_common_attr": "rerank_top_photo_id_list",
        "aggregator": "concat"
      }],
      range_start = "{{fountain_splash_rerank_top_photo_start_index}}",
      range_end = "{{fountain_splash_rerank_top_photo_end_index}}",
      select_item = {
        "join": "and",
        "filters": [
          {
            "enable": "{{fountain_splash_rerank_top_result_enable_reason_select}}",
            "attr_name": "reason",
            "select_if": "in",
            "compare_to": "{{fountain_splash_rerank_top_result_select_reasons_list}}",
            "select_if_attr_missing": False
          },
          {
            "enable": "{{fountain_splash_rerank_top_result_enable_relate_select}}",
            "attr_name": "source_related_score",
            "select_if": ">",
            "compare_to": "{{fountain_splash_rerank_top_result_relate_select_threshold}}",
            "select_if_attr_missing": False
          },
          {
            "enable": "{{fountain_splash_rerank_top_result_enable_ctr_select}}",
            "attr_name": "fullrank_empirical_ctr",
            "select_if": ">",
            "compare_to": "{{fountain_splash_rerank_top_result_ctr_select_threshold}}",
            "select_if_attr_missing": False
          },
          {
            "enable": "{{fountain_splash_rerank_top_result_enable_watchtime_select}}",
            "attr_name": "fullrank_empirical_watchtime",
            "select_if": ">",
            "compare_to": "{{fountain_splash_rerank_top_result_watchtime_select_threshold}}",
            "select_if_attr_missing": False
          }
        ]
      }
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "rerank_top_photo_id_list", "as": "universal_set_list"},
        {"name": "fountain_rerank_top_photo_id_retrieval_list", "as": "sub_set_list"}
      ],
      export_common_attr = [
        {"name": "difference_list", "as": "rerank_top_photo_id_list"}
      ],
      function_name = "GetDifferenceSet",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .pack_common_attr(
      input_common_attrs = [
        "rerank_top_photo_id_list",
        "fountain_rerank_top_photo_id_retrieval_list"
      ],
      output_common_attr = "rerank_top_photo_id_list",
      deduplicate = True,
      limit_num = "{{fountain_splash_rerank_top_photo_size}}",
    ) \
    .write_to_redis(
      kcc_cluster = "recoExploreNegPhoto",
      expire_second = "{{fountain_splash_rerank_top_photo_redis_expire_seconds}}",
      key_prefix = "{{fountain_splash_rerank_top_photo_key_prefix}}",
      key = "{{featureSourcePId}}",
      value = "{{rerank_top_photo_id_list}}"
    )
    return self

  def rerank_full_link_sample_splash(self):
    self \
    .get_kconf_params(
      kconf_configs=[
        {
          "kconf_key": "reco.offline.fountainSplashFulllinkDistillRankParam",
          "value_type": "json",
          "json_path": "rerank_full_link_distill_sample_begin",
          "export_common_attr": "fountain_rerank_full_link_distill_sample_begin"
        },
        {
          "kconf_key": "reco.offline.fountainSplashFulllinkDistillRankParam",
          "value_type": "json",
          "json_path": "rerank_full_link_distill_sample_end",
          "export_common_attr": "fountain_rerank_full_link_distill_sample_end"
        },
        {
          "kconf_key": "reco.offline.fountainSplashFulllinkDistillRankParam",
          "value_type": "json",
          "json_path": "rerank_full_link_distill_sample_num",
          "export_common_attr": "fountain_rerank_full_link_distill_sample_num"
        },
        {
          "kconf_key": "reco.offline.fountainSplashFulllinkDistillRankParam",
          "value_type": "json",
          "json_path": "rerank_full_link_distill_sample_ratio",
          "export_common_attr": "fountain_rerank_full_link_distill_sample_ratio"
        },
        {
          "kconf_key": "reco.hot.frCarmPidListSampleCount",
          "json_path": "fountain",
          "default_value": 0,
          "export_common_attr": "fountain_fr_carm_pid_sample_count"
        },
      ]
    ) \
    .if_("fountain_fr_carm_pid_sample_count > 0") \
      .explore_mc_distill_sample_enrich(  # 精排主模型 carm 特征回流
        candidate_list_attr = "cascade_output_item_id_list",
        sample_num = "{{fountain_fr_carm_pid_sample_count}}",
        save_sample_result_to = "fr_carm_pid_list",
      ) \
    .end_() \
    .explore_full_link_context_sample_reco_log_enricher(
      sample_config = [
        {
          "sample_begin": "fountain_rerank_full_link_distill_sample_begin",
          "sample_end": "fountain_rerank_full_link_distill_sample_end",
          "sample_num": "fountain_rerank_full_link_distill_sample_num",
          "label_name": "final_pos",
        },
      ],
      sample_ratio = "fountain_rerank_full_link_distill_sample_ratio",
      user_info_attr = "userInfoPb",
      load_attr = "fountain_full_link_reco_log_message",
      save_result_to = "fountain_full_link_reco_log_message_final",
      rank_index = "rank_index_after_es",
      final_index = "item_seq",
      cascade_pctr = "cascade_pctr",
      cascade_pltr = "cascade_pltr",
      cascade_pwtr = "cascade_pwtr",
      cascade_pftr = "cascade_pftr",
      cascade_pptr = "cascade_ptr",
      cascade_pcmtr = "cascade_pcmtr",
      cascade_plvtr = "cascade_plvtr",
      cascade_pvtr = "cascade_pwatch_time",
      pctr = "fullrank_sim_pevtr",
      pltr = "fullrank_sim_pltr",
      pwtr = "fullrank_sim_pwtr",
      pftr = "fullrank_sim_pftr",
      pptr = "fullrank_sim_pptr",
      pcmtr = "fullrank_sim_pcmtr",
      plvtr = "fullrank_sim_plvtr",
      pvtr = "fullrank_sim_pvtr",
      psvr = "fullrank_sim_psvr",
      pepstr = "fullrank_sim_pepstr",
      pcltr = "fullrank_sim_pcltr",
      pwtd = "fullrank_sim_pfintr",
      pcpr = "fullrank_sim_pcpr",
      fullrank_ltr_score = "fullrank_ltr_score",
      fullrank_act_wtd = "fullrank_act_wtd",
      fullrank_ltr_v4_fountain_next = "fullrank_ltr_v4_fountain_next",
      fountain_related_score_v2 = "fountain_related_score_v2",
      size_limit = 30,
      # 需要 set 一次 llsid
      enable_set_user_info = True,
      tab = 666,
    ) \
    .send_with_kafka(
      common_attr = "fountain_full_link_reco_log_message_final",
      topic_name = "full_link_samples",
    )

    return self
  
  def calc_splash_rerank_user_custom_ensemble_weight(self):
    self \
      .gen_common_attr_by_lua(
        attr_map = {
          "fountain_rerank_ensemble_power_weight_fullrank_ltr_emp": "fountain_rerank_ensemble_power_weight_fullrank_ltr_emp * user_group_emp_ltr",
          "fountain_rerank_ensemble_power_weight_fullrank_wtr_emp": "fountain_rerank_ensemble_power_weight_fullrank_wtr_emp * user_group_emp_wtr",
          "fountain_rerank_ensemble_power_weight_fullrank_ftr_emp": "fountain_rerank_ensemble_power_weight_fullrank_ftr_emp * user_group_emp_ftr",
          "fountain_rerank_ensemble_power_weight_fullrank_cmtr_emp": "fountain_rerank_ensemble_power_weight_fullrank_cmtr_emp * user_group_emp_cmtr",
          "fountain_rerank_ensemble_power_weight_fullrank_ptr_emp": "fountain_rerank_ensemble_power_weight_fullrank_ptr_emp * user_group_emp_ptr",
        },
      ) \
      .pack_item_attr(
        item_source = {
          "reco_results": True,
        },
        mappings = [
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_pevtr",
            "to_common_attr": "pevtr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_pltr",
            "to_common_attr": "pltr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_pwtr",
            "to_common_attr": "pwtr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_pftr",
            "to_common_attr": "pftr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_pcmtr",
            "to_common_attr": "pcmtr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_pptr",
            "to_common_attr": "pptr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_pepstr",
            "to_common_attr": "pepstr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_plvtr",
            "to_common_attr": "plvtr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "fullrank_sim_pfintr",
            "to_common_attr": "pfintr_avg"
          },
        ],
      ) \
      .enrich_attr_by_lua(
        import_common_attr = [
          "fountain_rerank_ensemble_power_weight_adjust_ratio_min",
          "fountain_rerank_ensemble_power_weight_adjust_ratio_max",
          "fountain_rerank_ensemble_power_weight_adjust_request_ratio",
          "fountain_rerank_ensemble_power_weight_fullrank_ltr_emp",
          "fountain_rerank_ensemble_power_weight_fullrank_wtr_emp",
          "fountain_rerank_ensemble_power_weight_fullrank_cmtr_emp",
          "fountain_rerank_ensemble_power_weight_fullrank_ptr_emp",
          "fountain_rerank_ensemble_power_weight_fullrank_ftr_emp",
          "fountain_rerank_ensemble_power_weight_fullrank_epstr_emp",
          "fountain_rerank_ensemble_power_weight_fullrank_evtr_emp",
          "fountain_rerank_ensemble_power_weight_fullrank_lvtr_emp",
          "fountain_rerank_ensemble_power_weight_fullrank_fintr_emp",
          "userExpLtr",
          "userExpWtr",
          "userExpCmtr",
          "userExpPtr",
          "userExpFtr",
          "userExpEptr",
          "user_emp_evtr",
          "user_emp_lvtr",
          "user_emp_watch_time",
          "pltr_avg",
          "pwtr_avg",
          "pftr_avg",
          "pcmtr_avg",
          "pptr_avg",
          "pepstr_avg",
          "pevtr_avg",
          "plvtr_avg",
          "pfintr_avg",
        ],
        export_common_attr = [
          "emp_ltr_factor",
          "emp_wtr_factor",
          "emp_cmtr_factor",
          "emp_ptr_factor",
          "emp_ftr_factor",
          "emp_epstr_factor",
          "emp_evtr_factor",
          "emp_lvtr_factor",
          "emp_watchtime_factor",
        ],
        function_for_common = "cal_user_emp_ada_weight_factor",
        lua_script_file = "fountain/rerank/lua/all_lua.lua",
      ) \
      .gen_common_attr_by_lua(
        attr_map = {
          "fountain_splash_rerank_gen_seed_ensemble_fullrank_like_score": "emp_ltr_factor * fountain_splash_rerank_gen_seed_ensemble_fullrank_like_score",
          "fountain_splash_rerank_gen_seed_ensemble_fullrank_follow_score": "emp_wtr_factor * fountain_splash_rerank_gen_seed_ensemble_fullrank_follow_score",
          "fountain_splash_rerank_gen_seed_ensemble_fullrank_comment_score": "emp_cmtr_factor * fountain_splash_rerank_gen_seed_ensemble_fullrank_comment_score",
          "fountain_splash_rerank_gen_seed_ensemble_fullrank_profile_score": "emp_ptr_factor * fountain_splash_rerank_gen_seed_ensemble_fullrank_profile_score",
          "fountain_splash_rerank_gen_seed_ensemble_fullrank_forward_score": "emp_ftr_factor * fountain_splash_rerank_gen_seed_ensemble_fullrank_forward_score",
          "fountain_splash_rerank_gen_seed_ensemble_fullrank_pepstr_score": "emp_epstr_factor * fountain_splash_rerank_gen_seed_ensemble_fullrank_pepstr_score",
          "fountain_splash_rerank_gen_seed_ensemble_fullrank_click_score": "emp_evtr_factor * fountain_splash_rerank_gen_seed_ensemble_fullrank_click_score",
          "fountain_splash_rerank_gen_seed_ensemble_fullrank_lvtr_ori_score": "emp_lvtr_factor * fountain_splash_rerank_gen_seed_ensemble_fullrank_lvtr_ori_score",
          "fountain_splash_rerank_gen_seed_ensemble_fullrank_pfintr_score": "emp_watchtime_factor * fountain_splash_rerank_gen_seed_ensemble_fullrank_pfintr_score",
        },
      ) \

    return self

  def _count_photo_type_distribution(self, stage):
    self \
      .count_reco_result(
        save_count_to = "%s_single_picture_count" % stage,
        target_item = {"picture_type": 1}
      ) \
      .count_reco_result(
        save_count_to = "%s_long_picture_count" % stage,
        target_item = {"picture_type": 2}
      ) \
      .count_reco_result(
        save_count_to = "%s_cluster_picture_count" % stage,
        target_item = {"picture_type": 3}
      ) \
    
    return self

  def _gen_list_reason_metrics_before_eval(self):
    # 统计评估前的 lsit count
    self \
      .if_("_IS_ONLINE_SERVICE_ == 1 and _IS_PERF_SAMPLING_REQUEST_ == 1") \
        .count_reco_result(
          save_count_to = "splash_rerank_gen_list_total_cnt",
          target_item = {
            "item_type": 3,
          },
        ) \
        .count_reco_result(
          save_count_to = "splash_rerank_gen_list_reason_66702_cnt",
          target_reason = 66702, # es
        ) \
        .count_reco_result(
          save_count_to = "splash_rerank_gen_list_reason_66704_cnt",
          target_reason = 66704, # origin
        ) \
        .count_reco_result(
          save_count_to = "splash_rerank_gen_list_reason_66707_cnt",
          target_reason = 66707, # model_ar
        ) \
      .end_()
    return self

  def _gen_list_reason_metrics_after_eval(self):
    # 统计评估后的 lsit count 并发送
    self \
      .if_("_IS_ONLINE_SERVICE_ == 1 and _IS_PERF_SAMPLING_REQUEST_ == 1") \
        .count_reco_result(
          save_count_to = "splash_rerank_eval_list_total_cnt",
          target_item = {
            "item_type": 3,
          },
        ) \
        .count_reco_result(
          save_count_to = "splash_rerank_eval_list_reason_66702_cnt",
          target_reason = 66702, # es
        ) \
        .count_reco_result(
          save_count_to = "splash_rerank_eval_list_reason_66704_cnt",
          target_reason = 66704, # origin
        ) \
        .count_reco_result(
          save_count_to = "splash_rerank_eval_list_reason_66707_cnt",
          target_reason = 66707, # model_ar
        ) \
        .send_abtest_metrics(
          metrics = [
            "splash_rerank_gen_list_total_cnt",
            "splash_rerank_gen_list_reason_66702_cnt",
            "splash_rerank_gen_list_reason_66704_cnt",
            "splash_rerank_gen_list_reason_66707_cnt",
            "splash_rerank_eval_list_total_cnt",
            "splash_rerank_eval_list_reason_66702_cnt",
            "splash_rerank_eval_list_reason_66704_cnt",
            "splash_rerank_eval_list_reason_66707_cnt",
          ],
          metric_name_prefix = "fountain_reco_leaf_"
        ) \
      .end_()

    return self
