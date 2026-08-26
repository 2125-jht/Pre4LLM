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
      .switch_("explore_rerank_predict_use_common_predict") \
        .case_(2, to_be_delete = "date=2023-11-16;committer=xuwei09") \
          .enrich_attr_by_light_function(
            export_item_attr = gen_photo_features_for_all_position(10) + ["corr_pctr", "generated_diversity_lists"],
            function_name = "EmptyFunction",
            class_name = "ExploreLightFunctionSetV2",
            item_list_from_attr = "retrieval_list_keys",
          ) \
          .delegate_enrich(
            kess_service = "{{explore_rerank_delegate_kai_kess_service}}",
            recv_item_attrs = ["l2r_pos" + str(i) for i in range(10)],
            timeout_ms = 100,
            send_item_attrs = gen_photo_features_for_all_position(10),
            send_common_attrs = user_features(),
            item_list_from_attr = "retrieval_list_keys",
            request_type = "default"
          ) \
          .explore_listwise_score_enricher(
            item_attrs = ["l2r_pos" + str(i) for i in range(10)],
            item_list_from_attr = "retrieval_list_keys",
            seq_item_attr_name = "generated_diversity_lists",
            topk_num = "{{fr_rerank_predict_topk_num}}",
            pxtr_attr = "corr_pctr",
            pxtr_weight = "{{fr_rerank_predict_pxtr_weight}}",
            loss_name = "{{fr_rerank_delegate_kai_service_loss_name}}",
            output_attr = "rerank_context",
          ) \
        .case_(1) \
          .enrich_attr_by_light_function(
            export_item_attr = gen_photo_features_for_all_position(10) + ["corr_pctr", "generated_diversity_lists"],
            function_name = "EmptyFunction",
            class_name = "ExploreLightFunctionSetV2",
            item_list_from_attr = "retrieval_list_keys",
          ) \
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
        .default_() \
          .explore_list_item_predict(
            OPT_REORDER = False,
            kess_service = "{{fr_rerank_predict_service}}",
            service_group = "PRODUCTION",
            origin_seq_reason = "{{fr_rerank_origin_reason}}",
            timeout_ms = "{{fr_rerank_predict_service_timeout_ms}}",
            layer_name = "{{fr_rerank_predict_tensoer_layer}}",
            shard_num = "{{fr_rerank_predict_shard_num}}",
            output_attr = "rerank_context",
            output_list_attr = "rerank_score_list",
            disable_common_attr_missing_warning = True,
            use_odd_score = "{{fr_rerank_predict_use_odd_score}}",
            extra_common_attrs = user_features(),
            item_attrs = gen_photo_features_for_all_position(10),
            item_list_from_attr = "retrieval_list_keys"
          ) \
      .end_() \
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
              "name": "fetr",
              "weight_base": "{{explore_rerank_ensemble_sort_fetr_weight}}"
            },
            {
              "name": "feff",
              "weight_base": "{{explore_rerank_ensemble_sort_feff_weight}}"
            },
          ]
        ) \
      .end_() \
      .sort(
          score_from_attr = "rerank_context",
          item_list_from_attr = "retrieval_list_keys",
      ) \
      .list_wise_item_attr(
          seq_item_attr_name = "generated_diversity_lists",
          seq_score_attr_name = "rerank_context",
          item_list_from_attr = "retrieval_list_keys",
      ) \
      .sort(
          stable_sort = True,
          score_from_attr = "virtual_rerank_score",
      ) \
    .end_()
    self.content_diversify()
    self.life_force_insert()
    self.content_control_perf()

  def content_diversify(self) -> None:
    self.flow \
      .if_("xlife_enable_target_content_control == 1") \
        .if_("enable_life_target_hetu_new == 1") \
          .get_kconf_params(
            kconf_configs = [
              {
                "kconf_key": "reco.eyeshot.LifeTabTargetHetuL2Json",
                "json_path": "{{life_target_hetu_version}}",
                "export_common_attr": "target_hetu_l2_list"
              },
              {
                "kconf_key": "reco.eyeshot.LifeTabGrayHetuL2Json",
                "json_path": "{{life_target_hetu_version}}",
                "export_common_attr": "gray_hetu_l2_list"
              },
              {
                "kconf_key": "reco.eyeshot.LifeTabTargetHetuL1Json",
                "json_path": "{{life_target_hetu_version}}",
                "export_common_attr": "target_hetu_l1_list"
              },
              {
                "kconf_key": "reco.eyeshot.LifeTabGrayHetuL1Json",
                "json_path": "{{life_target_hetu_version}}",
                "export_common_attr": "gray_hetu_l1_list"
              },
            ]
          ) \
          .enrich_attr_by_light_function(
            import_item_attr = [
              "photo_id",
              "hetu_tag_level_info__hetu_level_one",
              "hetu_tag_level_info__hetu_level_two"
            ],
            import_common_attr = [
              "target_hetu_l2_list",
              "gray_hetu_l2_list",
              "target_hetu_l1_list",
              "gray_hetu_l1_list"
            ],
            export_item_attr = [
              "gray_target", # 灰度 + 非生活打散，生活设为pid，灰度 + 非生活设为1
              "not_life_target" # 非生活打散，灰度 + 生活设置为pid，非生活设为1
            ],
            function_name = "ContentControlDiversifyTagV2",
            class_name = "ExploreLifeLightFunctionSet"
          ) \
        .else_() \
          .get_kconf_params(
            kconf_configs = [{
              "kconf_key": "reco.eyeshot.LifeTabTargetHetu",
              "value_type": "list_int64",
              "export_common_attr": "target_hetu_list",
              "default_value": []
            }]
          ) \
          .get_kconf_params(
            kconf_configs = [{
              "kconf_key": "reco.eyeshot.LifeTabNotTargetHetu",
              "value_type": "list_int64",
              "export_common_attr": "not_target_hetu_list",
              "default_value": []
            }]
          ) \
          .enrich_attr_by_light_function(
            import_item_attr = [
              "photo_id",
              "hetu_tag_level_info__hetu_level_one",
              "hetu_tag_level_info__hetu_level_two"
            ],
            import_common_attr = [
              "target_hetu_list",
              "not_target_hetu_list"
            ],
            export_item_attr = [
              "gray_target", # 灰度 + 非生活打散，生活设为pid，灰度 + 非生活设为1
              "not_life_target" # 非生活打散，灰度 + 生活设置为pid，非生活设为1
            ],
            function_name = "ContentControlDiversifyTag",
            class_name = "ExploreLifeLightFunctionSet"
          ) \
        .end_() \
        .diversify_by_rules(
          name = "xlife_gray_content_control",
          range_end = "{{xlife_content_control_limit_thres}}",
          max_satisfied_pick = 10,
          rules = [
            dict(attr_name= "gray_target",
                  enabled="{{enable_xlife_gray_control}}",
                  window_size="{{xlife_gray_control_window}}",
                  max_num="{{xlife_gray_control_max}}",
                  priority="{{xlife_gray_control_priority}}"),
            dict(attr_name= "not_life_target",
                  enabled="{{enable_xlife_target_control}}",
                  window_size="{{xlife_target_control_window}}",
                  max_num="{{xlife_target_control_max}}",
                  priority="{{xlife_target_control_priority}}"),
            dict(attr_name= "hetu_sim_cluster_id",
                  enabled="{{enable_xlife_rerank_hetu_cluster_diversity}}",
                  window_size="{{xlife_rerank_hetu_cluster_diversity_winsize}}",
                  max_num="{{xlife_rerank_hetu_cluster_diversity_maxnum}}",
                  priority="{{xlife_rerank_hetu_cluster_diversity_priority}}"),
            dict(attr_name= "hetu_sim_cluster_id862",
                  enabled="{{enable_xlife_rerank_hetu_cluster862_diversity}}",
                  window_size="{{xlife_rerank_hetu_cluster862_diversity_winsize}}",
                  max_num="{{xlife_rerank_hetu_cluster862_diversity_maxnum}}",
                  priority="{{xlife_rerank_hetu_cluster862_diversity_priority}}"),
            dict(attr_name= "is_good_looking",
                  enabled="{{enable_xlife_rerank_beauty_diversity}}",
                  window_size="{{xlife_rerank_beauty_diversity_winsize}}",
                  max_num="{{xlife_rerank_beauty_diversity_maxnum}}",
                  priority="{{xlife_rerank_beauty_diversity_priority}}"),
            dict(attr_name= "specified_hetu5_found", 
                  enabled="{{enable_xlife_rerank_hetu5_beauty_diversity}}",
                  window_size="{{xlife_rerank_hetu5_beauty_diversity_winsize}}",
                  max_num="{{xlife_rerank_hetu5_beauty_diversity_maxnum}}",
                  priority="{{xlife_rerank_hetu5_beauty_diversity_priority}}"),
            dict(attr_name= "author__id",
                  enabled="{{enable_xlife_rerank_author_diversity}}",
                  window_size="{{xlife_rerank_author_divstsity_winszie}}",
                  max_num="{{xlife_rerank_author_diversity_maxnum}}",
                  priority="{{xlife_rerank_author_diversity_priority}}"),
            dict(attr_name= "second_tag_quality_level1",
                 enabled="{{enable_xlife_rerank_second_tag_quality_level1_diversity}}",
                 window_size= "{{xlife_rerank_second_tag_quality_level1_winsize}}",
                 max_num="{{xlife_rerank_second_tag_quality_level1_maxnum}}",
                 priority="{{xlife_rerank_second_tag_quality_level1_priority}}"),
            dict(attr_name= "second_tag_quality_level2",
                 enabled="{{enable_xlife_rerank_second_tag_quality_level2_diversity}}",
                 window_size= "{{xlife_rerank_second_tag_quality_level2_winsize}}",
                 max_num="{{xlife_rerank_second_tag_quality_level2_maxnum}}",
                 priority="{{xlife_rerank_second_tag_quality_level2_priority}}"),
            dict(attr_name= "is_merchant_cart",
                 enabled="{{enable_xlife_rerank_merchant_cart_diversity}}",
                 window_size= "{{xlife_rerank_merchant_cart_winsize}}",
                 max_num="{{xlife_rerank_merchant_cart_maxnum}}",
                 priority="{{xlife_rerank_merchant_cart_priority}}"),
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
          ]
        ) \
        .set_attr_value(
          item_attrs = [{
            "name": "life_target",
            "type": "int",
            "value": 1
          }]
        ) \
        .copy_attr(
          attrs = [
            {
              "from_item": "photo_id",
              "to_item": "life_target"
            }
          ],
          target_item = {"gray_target": 1}
        ) \
        .diversify_by_rules(
          name = "xlife_hetu1_content_control",
          max_satisfied_pick="{{ultimate_variety_engineer_slot_num_shuanglie}}",
          range_end="{{ultimate_variety_gen_engineer_limit_thres}}",
          rules=[
            dict(attr_name= "hetu_tag_level_info__hetu_level_one",
                  enabled="{{enable_rerank_hetu1_diversify}}",
                  window_size="{{rerank_hetu1_diversify_winsize}}",
                  max_num="{{rerank_hetu1_diversify_max}}",
                  priority="{{ultimate_variety_shuanglie_priority1}}"),
          ],
          target_item = {"life_target": 1}
        ) \
        .if_("xlife_enable_gray_target_hetu_diver == 1") \
          .set_attr_value(
            item_attrs = [{
              "name": "gray_diver_target",
              "type": "int",
              "value": 1
            }]
          ) \
          .copy_attr(
            attrs = [{
              "from_item": "photo_id",
              "to_item": "gray_diver_target"
            }],
            target_item = { "not_life_target": 1 }
          ) \
          .copy_attr(
            attrs = [{
              "from_item": "photo_id",
              "to_item": "gray_diver_target"
            }],
            target_item = { "life_target": 1 }
          ) \
          .diversify_by_rules(
            name = "xlife_hetu1_gray_content_control",
            max_satisfied_pick="{{ultimate_variety_engineer_slot_num_shuanglie}}",
            range_end="{{ultimate_variety_gen_engineer_limit_thres}}",
            rules=[
              dict(attr_name= "hetu_tag_level_info__hetu_level_one",
                    enabled="{{enable_rerank_hetu1_diversify}}",
                    window_size="{{rerank_hetu1_diversify_winsize}}",
                    max_num="{{rerank_hetu1_diversify_max}}",
                    priority="{{ultimate_variety_shuanglie_priority1}}"),
            ],
            target_item = {"gray_diver_target": 1}
          ) \
        .end_() \
      .end_() \

  def life_force_insert(self) -> None:
    self.flow \
      .if_("enable_life_direct_tab_force_insert == 1") \
        .force_insert(
          reason = 2416,
          position = 0,
          limit = 1
        ) \
      .end_() \
      .if_("is_fresh_request == 1 and enable_life_active_interest_force_insert == 1 and (enable_life_active_interest_insert_judge ~= 1 or follow_author_insert_num > 0)") \
        .force_insert(
          reason = 2422,
          position = 1,
          limit = 1
        ) \
      .else_if_("enable_life_active_interest_force_insert_passby == 1 and page == 1 and uIsNotLifePassBy ~= 1") \
        .force_insert(
          reason = 2422,
          position = 1,
          limit = 1
        ) \
      .else_if_("enable_life_active_interest_force_insert_rerank == 1 and page == 1 and (life_active_interest_force_insert_limit_low_active ~= 1 or uIsLifeHighActive ~= 1)") \
        .set_attr_value(
          item_attrs=[
            {
              "name": "is_active_interest",
              "type": "int",
              "value": 1
            }
          ],
          target_item={
            "reason": 2422
          },
        ) \
        .enrich_attr_by_light_function(
          import_item_attr = [
            {"name": "score_pctr", "as": "score"},
            {"name": "is_active_interest", "as": "is_insert_item_attr"},
          ],
          import_common_attr = [
            {"name": "life_active_interest_force_insert_position", "as": "force_insert_position"},
          ],
          export_item_attr = [
            {"name": "promote_to_position", "as": "life_active_interest_promote_to_position"},
          ],
          function_name = "CalForceInsertPosition",
          class_name = "ExploreLifeLightFunctionSet"
        ) \
        .force_insert(
          position_from_attr = "life_active_interest_promote_to_position",
        ) \
      .end_() \
      .if_("enable_life_hotfire_yellow_force_insert == 1 and page == 1 and (life_hotfire_yellow_force_insert_limit_low_active ~= 1 or uIsLifeHighActive ~= 1)") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            {"name": "score_pctr", "as": "score"},
            {"name": "is_hotfire_yellow", "as": "is_insert_item_attr"},
          ],
          import_common_attr = [
            {"name": "life_hotfire_yellow_force_insert_position", "as": "force_insert_position"},
          ],
          export_item_attr = [
            {"name": "promote_to_position", "as": "life_hotfire_yellow_promote_to_position"},
          ],
          function_name = "CalForceInsertPosition",
          class_name = "ExploreLifeLightFunctionSet"
        ) \
        .force_insert(
          position_from_attr = "life_hotfire_yellow_promote_to_position",
        ) \
      .end_() \
      .if_("enable_life_search_latest_force_insert == 1 and (page == 1 or ((page or 100) <= 3 and util.Random() < life_search_latest_force_insert_prob))") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            {"name": "score_pctr", "as": "score"},
            {"name": "is_search_boost", "as": "is_insert_item_attr"},
          ],
          import_common_attr = [
            {"name": "life_search_latest_force_insert_position", "as": "force_insert_position"},
            {"name": "life_search_latest_force_insert_coeff", "as": "coeff"},
          ],
          export_item_attr = [
            {"name": "promote_to_position", "as": "life_search_latest_promote_to_position"},
          ],
          function_name = "CalForceInsertPosition",
          class_name = "ExploreLifeLightFunctionSet"
        ) \
        .force_insert(
          position_from_attr = "life_search_latest_promote_to_position",
        ) \
      .end_()

  def content_control_perf(self) -> None:
    self.flow \
    .get_abtest_params(
      biz_name = "RECO_RPC",
      ab_params = [{
        "param_name": "explore_xlife_perf_mark",
        "param_type": "string",
        "default_value": "all"
      }]
    ) \
    .copy_attr(
      attrs = [
        {
          "from_item": "photo_id",
          "to_item": "gray_target"
        }
      ],
      target_item= {"not_life_target" : 1}
    ) \
    .str_format(
      format_string = "rerank.life_quota.%s_avg",
      input_attrs = ["explore_xlife_perf_mark"],
      output_attr = "rerank_life_quota_check_point_avg",
    ) \
    .str_format(
      format_string = "rerank.life_quota.%s_cnt",
      input_attrs = ["explore_xlife_perf_mark"],
      output_attr = "rerank_life_quota_check_point_cnt",
    ) \
    .count_item_attr(
      counters = [{
        "check_attr_name": "not_life_target",
        "output_attr_name": "is_not_life",
        "check_values": [1],
      },{
        "check_attr_name": "life_target",
        "output_attr_name": "is_life",
        "check_values": [1],
      },{
        "check_attr_name": "gray_target",
        "output_attr_name": "is_gray",
        "check_values": [1],
      }]
    ) \
    .pack_item_attr(
      item_source = {
        "reco_results": True,
        "total_limit": "{{explore_xlife_rerank_quota_perf_num}}"
      },
      mappings = [{
        "aggregator": "sum",
        "from_item_attr": "is_not_life",
        "to_common_attr": "rerank_not_life_quota_num",
      },{
        "aggregator": "sum",
        "from_item_attr": "is_life",
        "to_common_attr": "rerank_life_quota_num",
      },{
        "aggregator": "sum",
        "from_item_attr": "is_gray",
        "to_common_attr": "rerank_gray_quota_num",
      },]
    ) \
    .perflog_attr_value(
      check_point = "{{rerank_life_quota_check_point_avg}}",
      aggregator = "avg",
      common_attrs = [
        "rerank_not_life_quota_num",
        "rerank_life_quota_num", 
        "rerank_gray_quota_num",
      ],
    ) \
    .perflog_attr_value(
      check_point = "{{rerank_life_quota_check_point_cnt}}",
      aggregator = "count",
      common_attrs = [
        "rerank_not_life_quota_num",
        "rerank_life_quota_num", 
        "rerank_gray_quota_num"
      ],
    ) \

  def post_process(self) -> None:
    self.flow \
      .if_("enable_use_explore_rerank == 1") \
        .log_debug_info(
          common_attrs = [
            "explore_rerank_ensemble_sort_ensemble_score_weight",
            "explore_rerank_ensemble_sort_pure_fr_score2_weight",
            "rerank_output_item_key_list_top10",
            "rerank_output_item_key_list"
          ],
          item_attrs = [
              "rerank_context",
              "rerank_score_list",
              "virtual_rerank_score",
          ] + ["l2r_pos" + str(i) for i in range(10)] + ["list_ltr0"],
          item_list_from_attr = "retrieval_list_keys",
          for_debug_request_only = True
        ) \
        .log_debug_info(
          item_attrs = [
              "rerank_context",
              "rerank_score_list"
          ],
          item_list_from_attr = "retrieval_list_keys",
          for_debug_request_only = True
        ) \
      .end_()