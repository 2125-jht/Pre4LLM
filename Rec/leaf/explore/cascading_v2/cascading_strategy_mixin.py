from dragonfly.ext.common_leaf_base_mixin import CommonLeafBaseMixin

class CascadingStrategyMixin(CommonLeafBaseMixin):
  """
  双列发现页外流粗排策略函数 Mixin 实现
  """

  # region prerank

  def prerank_low_cost_photo_discount(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_prerank_low_cost_photo_discount_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1,
        "is_low_cost_photo": 1
      }
    )
    return self

  def prerank_pic_search_boost(self, score_attr, flag_attr, target_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_prerank_pic_search_boost_coef", "as": "boost_discount_coeff"},
        {"name": "explore_prerank_pic_search_boost_topk", "as": "topk"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1,
        target_attr: 1,
      }
    )
    return self

  def prerank_pic_recent_search_boost(self, score_attr, flag_attr, target_attr):
    self \
    .if_("enable_explore_prerank_pic_recent_search_boost_sort_topk == 1") \
      .sort(
        score_from_attr = score_attr,
        target_item = {
          flag_attr: 1,
          target_attr: 1
        },
        partial_sort = True,
        partial_num = "{{explore_prerank_pic_recent_search_boost_topk}}"
      ) \
    .end_() \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_prerank_pic_recent_search_boost_coef", "as": "boost_discount_coeff"},
        {"name": "explore_prerank_pic_recent_search_boost_topk", "as": "topk"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1,
        target_attr: 1,
      }
    )
    return self

  def prerank_pic_double_valid_interest_cluster_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_prerank_pic_double_valid_interest_cluster_boost_coef", "as": "boost_discount_coeff"},
        {"name": "explore_prerank_pic_double_valid_interest_cluster_boost_thres", "as": "boost_discount_thres"},
        {"name": "explore_prerank_pic_double_valid_interest_cluster_boost_topk", "as": "topk"},
      ],
      import_item_attr = [
        {"name": "cascade_prerank_pctr", "as": "need_item_attr"},
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountWithThres",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1,
        "is_pic_double_valid_interest_cluster": 1
      }
    )
    return self

  def prerank_pic_recent_interest_cluster_boost(self, score_attr, flag_attr):
    self \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_pic_recent_interest_cluster_id_list", "as": "key_list"},
        {"name": "explore_pic_recent_interest_cluster_score_list", "as": "value_list"},
        {"name": "explore_prerank_pic_recent_interest_cluster_score_alpha", "as": "score_alpha"},
        {"name": "explore_prerank_pic_recent_interest_cluster_score_beta", "as": "score_beta"},
        {"name": "explore_prerank_pic_recent_interest_cluster_score_sigma", "as": "score_sigma"},
        {"name": "enable_explore_prerank_pic_recent_interest_cluster_score_range_limit", "as": "enable_score_range_limit"},
        {"name": "explore_prerank_pic_recent_interest_cluster_score_lower_bound", "as": "score_lower_bound"},
        {"name": "explore_prerank_pic_recent_interest_cluster_score_upper_bound", "as": "score_upper_bound"},
        {"name": "enable_explore_prerank_pic_recent_interest_cluster_boost_only_low_interest_user", "as": "enable_only_low_interest_user"},
        {"name": "enable_explore_prerank_pic_recent_interest_cluster_boost_low_interest_user_enhance", "as": "enable_low_interest_user_enhance"},
        {"name": "explore_prerank_pic_recent_interest_cluster_boost_low_interest_user_enhance_base_coef", "as": "enhance_base_coeff"},
        {"name": "explore_prerank_pic_recent_interest_cluster_boost_low_interest_user_enhance_weight", "as": "enhance_weight"},
        {"name": "explore_prerank_pic_recent_interest_cluster_boost_low_interest_thres", "as": "low_interest_thres"},
        {"name": "uDoubleOutsideValidPicCluster7dList", "as": "user_interest"},
        {"name": "enable_explore_prerank_pic_recent_interest_cluster_boost_only_not_interest_cluster", "as": "enable_only_not_interest_cluster"},
      ],
      import_item_attr = [
        "cluster_id_632",
      ],
      export_item_attr = [
        {"name": "recent_interest_score", "as": "explore_prerank_pic_recent_interest_score"}
      ],
      function_name = "CalcPicRecentInterestScore",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1
      }
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "explore_prerank_pic_recent_interest_score", "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1
      }
    )
    return self

  def prerank_select_photo_by_interest(self, score_attr_name, flag_attr_name):
    self \
      .explore_control_hetu_count_enricher(
        save_flag_to_attr = "prerank_diversity_select_flag",
        keep_size = "{{prerank_final_candidate_num}}", 
        enable_hetu_control_diversity = "{{prerank_enable_hetu_control_diversity}}",
        hetu_level_five_attr = "hetu_tag_level_info__hetu_level_five",
        hetu5_max_size = "{{prerank_control_hetu5_max_size}}",
        enable_minority_control_diversity = "{{prerank_enable_minority_control_diversity}}",
        is_minority_photo_attr = "is_minority_photo",
        minority_max_size = "{{prerank_control_minority_max_size}}",
        save_is_degraded_common_attr = "prerank_hetu_quota_control_is_degraded",
        target_item = {
          flag_attr_name: 1
        }
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          {"name": "prerank_diversity_select_flag", "as": "flag"},
        ],
        export_item_attr = [
          {"name": "score", "as": score_attr_name},
        ],
        function_name = "SetMinimumScoreByFlag",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          flag_attr_name: 1
        }
      )
    return self

  def prerank_pic_valid_interest_cluster_boost(self, score_attr, flag_attr):
    self.sort(
      score_from_attr = score_attr,
      target_item = {
        flag_attr: 1,
        "is_pic_valid_interest_cluster": 1
      },
      partial_sort = True,
      partial_num = "{{explore_prerank_pic_valid_interest_cluster_boost_topk}}"
    ) \
    .if_("enable_explore_prerank_pic_valid_interest_cluster_boost_low_interest_user_enhance == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_prerank_pic_valid_interest_cluster_boost_coef", "as": "base_coeff"},
          {"name": "explore_prerank_pic_valid_interest_cluster_boost_low_interest_user_enhance_base_coef", "as": "enhance_base_coeff"},
          {"name": "explore_prerank_pic_valid_interest_cluster_boost_low_interest_user_enhance_weight", "as": "enhance_weight"},
          {"name": "explore_prerank_pic_valid_interest_cluster_boost_low_interest_thres", "as": "low_interest_thres"},
          {"name": "pic_double_outside_valid_interest_num", "as": "user_interest_num"},
        ],
        export_common_attr = [
          {"name": "enhance_coeff", "as": "explore_prerank_pic_valid_interest_cluster_boost_low_interest_user_enhance_coef"}
        ],
        function_name = "CalcPicLowInterestUserBoost",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_prerank_pic_valid_interest_cluster_boost_low_interest_user_enhance_coef", "as": "boost_discount_coeff"},
          {"name": "explore_prerank_pic_valid_interest_cluster_boost_thres", "as": "boost_discount_thres"},
          {"name": "explore_prerank_pic_valid_interest_cluster_boost_topk", "as": "topk"},
        ],
        import_item_attr = [
          {"name": "cascade_prerank_pctr", "as": "need_item_attr"},
          {"name": score_attr, "as": "ensemble_score"},
        ],
        export_item_attr = [
          {"name": "ensemble_score", "as": score_attr}
        ],
        function_name = "BoostOrDiscountWithThres",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          flag_attr: 1,
          "is_pic_valid_interest_cluster": 1
        }
      ) \
    .else_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_prerank_pic_valid_interest_cluster_boost_coef", "as": "boost_discount_coeff"},
          {"name": "explore_prerank_pic_valid_interest_cluster_boost_thres", "as": "boost_discount_thres"},
          {"name": "explore_prerank_pic_valid_interest_cluster_boost_topk", "as": "topk"},
        ],
        import_item_attr = [
          {"name": "cascade_prerank_pctr", "as": "need_item_attr"},
          {"name": score_attr, "as": "ensemble_score"},
        ],
        export_item_attr = [
          {"name": "ensemble_score", "as": score_attr}
        ],
        function_name = "BoostOrDiscountWithThres",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          flag_attr: 1,
          "is_pic_valid_interest_cluster": 1
        }
      ) \
    .end_()
    return self

  def prerank_pic_long_interest_cluster_boost(self, score_attr, flag_attr):
    self.sort(
      score_from_attr = score_attr,
      target_item = {
        flag_attr: 1,
        "is_pic_long_interest_cluster": 1
      },
      partial_sort = True,
      partial_num = "{{explore_prerank_pic_long_interest_cluster_boost_topk}}"
    ) \
    .if_("enable_explore_prerank_pic_long_interest_cluster_boost_low_interest_user_enhance == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_prerank_pic_long_interest_cluster_boost_coef", "as": "base_coeff"},
          {"name": "explore_prerank_pic_long_interest_cluster_boost_low_interest_user_enhance_base_coef", "as": "enhance_base_coeff"},
          {"name": "explore_prerank_pic_long_interest_cluster_boost_low_interest_user_enhance_weight", "as": "enhance_weight"},
          {"name": "explore_prerank_pic_long_interest_cluster_boost_low_interest_thres", "as": "low_interest_thres"},
          {"name": "pic_double_outside_valid_interest_num", "as": "user_interest_num"},
        ],
        export_common_attr = [
          {"name": "enhance_coeff", "as": "explore_prerank_pic_long_interest_cluster_boost_low_interest_user_enhance_coef"}
        ],
        function_name = "CalcPicLowInterestUserBoost",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_prerank_pic_long_interest_cluster_boost_low_interest_user_enhance_coef", "as": "explore_prerank_pic_long_interest_cluster_boost_discount_coeff"},
          {"name": "explore_prerank_pic_long_interest_cluster_boost_thres", "as": "boost_discount_thres"},
          {"name": "explore_prerank_pic_long_interest_cluster_boost_topk", "as": "topk"},
        ],
        import_item_attr = [
          {"name": "cascade_prerank_pctr", "as": "need_item_attr"},
          {"name": score_attr, "as": "ensemble_score"},
        ],
        export_item_attr = [
          {"name": "ensemble_score", "as": score_attr}
        ],
        function_name = "BoostOrDiscountWithThres",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          flag_attr: 1,
          "is_pic_long_interest_cluster": 1
        }
      ) \
    .else_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_prerank_pic_long_interest_cluster_boost_coef", "as": "boost_discount_coeff"},
          {"name": "explore_prerank_pic_long_interest_cluster_boost_thres", "as": "boost_discount_thres"},
          {"name": "explore_prerank_pic_long_interest_cluster_boost_topk", "as": "topk"},
        ],
        import_item_attr = [
          {"name": "cascade_prerank_pctr", "as": "need_item_attr"},
          {"name": score_attr, "as": "ensemble_score"},
        ],
        export_item_attr = [
          {"name": "ensemble_score", "as": score_attr}
        ],
        function_name = "BoostOrDiscountWithThres",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          flag_attr: 1,
          "is_pic_long_interest_cluster": 1
        }
      ) \
    .end_()
    return self

  # endregion

  # region mc

  def mc_calc_pic_real_pctr(self, flag_attr):
    self.enrich_attr_by_light_function(  # 计算粗排真实 ctr
      import_item_attr = [
        {"name": "cascade_pctr", "as": "pctr"},
        {"name": "cascade_psvtr", "as": "psvr"},
      ],
      export_item_attr = [
        {"name": "real_pctr", "as": "cascade_real_pctr"},
      ],
      function_name = "CalcRealPctr",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1
      }
    ) \
    .if_("enable_explore_pic_mc_real_pctr_personalized_weight > 0") \
      .enrich_attr_by_light_function(  # 计算个性化权重
        import_common_attr = [
          {"name": "explore_pic_mc_real_pctr_weight_config_str", "as": "pxtr_attr_config_str"},
          {"name": "explore_pic_mc_real_pctr_weight_avg_top_num", "as": "avg_top_num"},
          {"name": "explore_pic_mc_real_pctr_weight_trans_params_str", "as": "trans_params_str"},
          {"name": "explore_pic_mc_real_pctr_weight_enable_trans", "as": "enable_trans"},
          {"name": "explore_pic_mc_real_pctr_weight_trans_alpha", "as": "trans_alpha"},
          {"name": "explore_pic_mc_real_pctr_weight_trans_bias", "as": "trans_bias"},
          {"name": "explore_pic_mc_real_pctr_weight_trans_pow", "as": "trans_pow"},
          {"name": "explore_pic_mc_real_pctr_weight_trans_min", "as": "trans_min"},
          {"name": "explore_pic_mc_real_pctr_weight_trans_max", "as": "trans_max"},
        ],
        export_common_attr = [
          {"name": "pxtr_topn_avg_score", "as": "explore_pic_mc_real_pctr_pow_weight"},
        ],
        import_item_attr = [
          "mc_ensemble_pctr",
          "cascade_pltr",
          "cascade_pwtr",
          "cascade_pcltr",
          "cascade_pcmtr",
          "cascade_pftr",
        ],
        function_name = "CalcPxtrStatScore",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          flag_attr: 1
        }
      ) \
    .end_()
    return self

  def mc_boost_pic_es_by_follow_author(self, score_attr, flag_attr):
    """
    关注作者 pic_s1_es boost
    Owner: chenqiaojun, caozhong
    Date: 2023-07-31
    :param score_attr:
    :param flag_attr:
    :return:
    """
    self.enrich_attr_by_light_function(
      import_common_attr=[
        {"name": "cascading_follow_author_pic_boost_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr=[
        {"name": "is_follow_author", "as": "need_item_attr"},
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr=[
        {"name": "ensemble_score", "as": score_attr}
      ],
      function_name="BoostOrDiscount",
      class_name="ExploreLightFunctionSetV2",
      target_item={
        flag_attr: 1
      },
    )
    return self

  def mc_boost_pic_es_by_caption(self, score_attr, flag_attr):
    """
    长文本图文 pic_s1_es boost
    Owner: chenqiaojun, caozhong
    Date: 2023-07-31
    :param score_attr:
    :param flag_attr:
    :return:
    """
    self.enrich_attr_by_light_function(
      import_common_attr=[
        {"name": "cascade_channel_caption_boost_coef", "as": "caption_boost_coef"},
        {"name": "cascade_channel_caption_boost_len_thresh", "as": "caption_boost_len_thresh"},
        {"name": "cascade_channel_caption_boost_len_max", "as": "caption_boost_len_max"},
        {"name": "cascade_channel_boost_only_xhs_photo", "as": "boost_only_xhs_photo"},
        {"name": "cascade_channel_boost_only_picture", "as": "boost_only_picture"},
      ],
      import_item_attr=[
        {"name": score_attr, "as": "score"},
        "caption_length",
        "is_xhs_type_photo",
        "is_picture",
      ],
      export_item_attr=[
        {"name": "score", "as": score_attr},
      ],
      function_name="BoostWithCaption",
      class_name="ExploreLightFunctionSetV2",
      target_item={
        flag_attr: 1
      }
    )
    return self

  def mc_pic_valid_interest_cluster_boost(self, score_attr, flag_attr):
    self.sort(
      score_from_attr = score_attr,
      target_item = {
        flag_attr: 1,
        "is_pic_valid_interest_cluster": 1
      },
      partial_sort = True,
      partial_num = "{{explore_mc_pic_valid_interest_cluster_boost_topk}}"
    ) \
    .if_("enable_explore_mc_pic_valid_interest_cluster_boost_low_interest_user_enhance == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_mc_pic_valid_interest_cluster_boost_coef", "as": "base_coeff"},
          {"name": "explore_mc_pic_valid_interest_cluster_boost_low_interest_user_enhance_base_coef", "as": "enhance_base_coeff"},
          {"name": "explore_mc_pic_valid_interest_cluster_boost_low_interest_user_enhance_weight", "as": "enhance_weight"},
          {"name": "explore_mc_pic_valid_interest_cluster_boost_low_interest_thres", "as": "low_interest_thres"},
          {"name": "pic_double_outside_valid_interest_num", "as": "user_interest_num"},
        ],
        export_common_attr = [
          {"name": "enhance_coeff", "as": "explore_mc_pic_valid_interest_cluster_boost_low_interest_user_enhance_coef"}
        ],
        function_name = "CalcPicLowInterestUserBoost",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_mc_pic_valid_interest_cluster_boost_low_interest_user_enhance_coef", "as": "boost_discount_coeff"},
          {"name": "explore_mc_pic_valid_interest_cluster_boost_thres", "as": "boost_discount_thres"},
          {"name": "explore_mc_pic_valid_interest_cluster_boost_topk", "as": "topk"},
        ],
        import_item_attr = [
          {"name": "mc_ensemble_pctr", "as": "need_item_attr"},
          {"name": score_attr, "as": "ensemble_score"},
        ],
        export_item_attr = [
          {"name": "ensemble_score", "as": score_attr}
        ],
        function_name = "BoostOrDiscountWithThres",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          flag_attr: 1,
          "is_pic_valid_interest_cluster": 1
        }
      ) \
    .else_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_mc_pic_valid_interest_cluster_boost_coef", "as": "boost_discount_coeff"},
          {"name": "explore_mc_pic_valid_interest_cluster_boost_thres", "as": "boost_discount_thres"},
          {"name": "explore_mc_pic_valid_interest_cluster_boost_topk", "as": "topk"},
        ],
        import_item_attr = [
          {"name": "mc_ensemble_pctr", "as": "need_item_attr"},
          {"name": score_attr, "as": "ensemble_score"},
        ],
        export_item_attr = [
          {"name": "ensemble_score", "as": score_attr}
        ],
        function_name = "BoostOrDiscountWithThres",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          flag_attr: 1,
          "is_pic_valid_interest_cluster": 1
        }
      ) \
    .end_()
    return self

  def mc_pic_long_interest_cluster_boost(self, score_attr, flag_attr):
    self.sort(
      score_from_attr = score_attr,
      target_item = {
        flag_attr: 1,
        "is_pic_long_interest_cluster": 1
      },
      partial_sort = True,
      partial_num = "{{explore_mc_pic_long_interest_cluster_boost_topk}}"
    ) \
    .if_("enable_explore_mc_pic_long_interest_cluster_boost_low_interest_user_enhance == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_mc_pic_long_interest_cluster_boost_coef", "as": "base_coeff"},
          {"name": "explore_mc_pic_long_interest_cluster_boost_low_interest_user_enhance_base_coef", "as": "enhance_base_coeff"},
          {"name": "explore_mc_pic_long_interest_cluster_boost_low_interest_user_enhance_weight", "as": "enhance_weight"},
          {"name": "explore_mc_pic_long_interest_cluster_boost_low_interest_thres", "as": "low_interest_thres"},
          {"name": "pic_double_outside_valid_interest_num", "as": "user_interest_num"},
        ],
        export_common_attr = [
          {"name": "enhance_coeff", "as": "explore_mc_pic_long_interest_cluster_boost_low_interest_user_enhance_coef"}
        ],
        function_name = "CalcPicLowInterestUserBoost",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_mc_pic_long_interest_cluster_boost_low_interest_user_enhance_coef", "as": "boost_discount_coeff"},
          {"name": "explore_mc_pic_long_interest_cluster_boost_thres", "as": "boost_discount_thres"},
          {"name": "explore_mc_pic_long_interest_cluster_boost_topk", "as": "topk"},
        ],
        import_item_attr = [
          {"name": "mc_ensemble_pctr", "as": "need_item_attr"},
          {"name": score_attr, "as": "ensemble_score"},
        ],
        export_item_attr = [
          {"name": "ensemble_score", "as": score_attr}
        ],
        function_name = "BoostOrDiscountWithThres",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          flag_attr: 1,
          "is_pic_long_interest_cluster": 1
        }
      ) \
    .else_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_mc_pic_long_interest_cluster_boost_coef", "as": "boost_discount_coeff"},
          {"name": "explore_mc_pic_long_interest_cluster_boost_thres", "as": "boost_discount_thres"},
          {"name": "explore_mc_pic_long_interest_cluster_boost_topk", "as": "topk"},
        ],
        import_item_attr = [
          {"name": "mc_ensemble_pctr", "as": "need_item_attr"},
          {"name": score_attr, "as": "ensemble_score"},
        ],
        export_item_attr = [
          {"name": "ensemble_score", "as": score_attr}
        ],
        function_name = "BoostOrDiscountWithThres",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          flag_attr: 1,
          "is_pic_long_interest_cluster": 1
        }
      ) \
    .end_()
    return self

  def mc_pic_search_boost(self, score_attr, flag_attr, target_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_pic_search_boost_coef", "as": "boost_discount_coeff"},
        {"name": "explore_mc_pic_search_boost_topk", "as": "topk"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1,
        target_attr: 1,
      }
    )
    return self

  def mc_pic_recent_search_boost(self, score_attr, flag_attr, target_attr):
    self \
    .if_("enable_explore_mc_pic_recent_search_boost_sort_topk == 1") \
      .sort(
        score_from_attr = score_attr,
        target_item = {
          flag_attr: 1,
          target_attr: 1
        },
        partial_sort = True,
        partial_num = "{{explore_mc_pic_recent_search_boost_topk}}"
      ) \
    .end_() \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_pic_recent_search_boost_coef", "as": "boost_discount_coeff"},
        {"name": "explore_mc_pic_recent_search_boost_topk", "as": "topk"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1,
        target_attr: 1,
      }
    )
    return self

  def mc_pic_double_valid_interest_cluster_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_pic_double_valid_interest_cluster_boost_coef", "as": "boost_discount_coeff"},
        {"name": "explore_mc_pic_double_valid_interest_cluster_boost_thres", "as": "boost_discount_thres"},
        {"name": "explore_mc_pic_double_valid_interest_cluster_boost_topk", "as": "topk"},
      ],
      import_item_attr = [
        {"name": "mc_ensemble_pctr", "as": "need_item_attr"},
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountWithThres",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1,
        "is_pic_double_valid_interest_cluster": 1
      }
    )
    return self
  
  def mc_pic_recent_interest_cluster_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_pic_recent_interest_cluster_id_list", "as": "key_list"},
        {"name": "explore_pic_recent_interest_cluster_score_list", "as": "value_list"},
        {"name": "explore_mc_pic_recent_interest_cluster_score_alpha", "as": "score_alpha"},
        {"name": "explore_mc_pic_recent_interest_cluster_score_beta", "as": "score_beta"},
        {"name": "explore_mc_pic_recent_interest_cluster_score_sigma", "as": "score_sigma"},
        {"name": "enable_explore_mc_pic_recent_interest_cluster_score_range_limit", "as": "enable_score_range_limit"},
        {"name": "explore_mc_pic_recent_interest_cluster_score_lower_bound", "as": "score_lower_bound"},
        {"name": "explore_mc_pic_recent_interest_cluster_score_upper_bound", "as": "score_upper_bound"},
        {"name": "enable_explore_mc_pic_recent_interest_cluster_boost_only_low_interest_user", "as": "enable_only_low_interest_user"},
        {"name": "enable_explore_mc_pic_recent_interest_cluster_boost_low_interest_user_enhance", "as": "enable_low_interest_user_enhance}"},
        {"name": "explore_mc_pic_recent_interest_cluster_boost_low_interest_user_enhance_base_coef", "as": "enhance_base_coeff"},
        {"name": "explore_mc_pic_recent_interest_cluster_boost_low_interest_user_enhance_weight", "as": "enhance_weight"},
        {"name": "explore_mc_pic_recent_interest_cluster_boost_low_interest_thres", "as": "low_interest_thres"},
        {"name": "uDoubleOutsideValidPicCluster7dList", "as": "user_interest"},
        {"name": "enable_explore_mc_pic_recent_interest_cluster_boost_only_not_interest_cluster", "as": "enable_only_not_interest_cluster"},
      ],
      import_item_attr = [
        "cluster_id_632",
      ],
      export_item_attr = [
        {"name": "recent_interest_score", "as": "explore_mc_pic_recent_interest_score"}
      ],
      function_name = "CalcPicRecentInterestScore",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1
      }
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "explore_mc_pic_recent_interest_score", "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1
      }
    )
    return self

  def mc_cascade_pctr_calibration(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_cascade_pctr_calibration_upload_time", "as": "pctr_calibration_upload_time"},
      ],
      import_item_attr = [
        "upload_time",
        {"name": "cascade_pctr", "as": "pctr"},
      ],
      export_item_attr = [
        {"name": "pctr", "as": "cascade_corr_pctr"},
      ],
      function_name = "PxtrCalibration",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def mc_impression_audit_adjust(self, score_attr, flag_attr):
    self \
      .transform_item_attr( # 观感审二级字段大于0才是已审核
        mappings = [{
          "check_attr_name": "audit_b_second_tag",
          "check_attr_type": "int",
          "output_attr_name": "is_impression_audit",
          "output_attr_type": "int",
          "output_default_value": 0,
          "rules": [{
            "check_range": {
              "lower_bound": 1
            },
            "output_value": 1
          }]
        }],
        target_item = {
          flag_attr: 1,
        },
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "impression_audit_adjust_coeff_map_str", "as": "adjust_coeff_map_str_attr"},
          {"name": "impression_audit_emp_xtr_adjust_flag", "as": "emp_xtr_adjust_flag"},
          {"name": "impression_audit_emp_ctr_avg", "as": "emp_ctr_avg"},
          {"name": "impression_audit_emp_watchtime_avg", "as": "emp_watchtime_avg"},
          {"name": "impression_audit_emp_xtr_coeff_a", "as": "emp_xtr_coeff_a"},
          {"name": "impression_audit_emp_xtr_coeff_b", "as": "emp_xtr_coeff_b"}
        ],
        import_item_attr = [
          {"name": "content_safety_level_with_namespace__level_hot_online", "as": "audit_level_attr"},
          {"name": score_attr, "as": "ensemble_score_attr"},
          "upload_time",
          {"name": "explore_stat__real_show_count", "as": "realshow_count"},
          {"name": "explore_stat__click_count", "as": "click_count"},
          {"name": "explore_stat__view_length_sum", "as": "watchtime_sum"}
        ],
        export_item_attr = [
          {"name": "ensemble_score_attr", "as": score_attr},
        ],
        function_name = "AuditAdjustScore",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          flag_attr: 1,
          "is_impression_audit": 1,
        },
      )
    return self

  def mc_marketing_compensation_discount(self, score_attr, flag_attr):
    self \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "mc_s2_marketing_compensation_discount_ctr_weight", "as": "ctr_weight"},
          {"name": "mc_s2_marketing_compensation_discount_watchtime_weight", "as": "watchtime_weight"},
          {"name": "mc_s2_marketing_compensation_discount_score_base", "as": "score_base"},
          {"name": "mc_s2_marketing_compensation_discount_score_base_ratio", "as": "score_base_ratio"},
          {"name": "mc_s2_marketing_compensation_discount_coef", "as": "old_coeff"},
        ],
        import_item_attr = [
          {"name": "mc_ensemble_pctr", "as": "ctr"},
          {"name": "mc_ensemble_pwtd_inverse", "as": "watchtime"},
        ],
        export_common_attr = [
          {"name": "coeff", "as": "mc_s2_marketing_compensation_discount_reward_coeff"}
        ],
        function_name = "CalcRewardCoeff",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "mc_s2_marketing_compensation_discount_scale_factor", "as": "scale_factor"},
          {"name": "mc_s2_marketing_compensation_discount_reward_coeff", "as": "reward_coeff"},
          {"name": "mc_s2_marketing_compensation_discount_coef", "as": "old_coeff"},
        ],
        export_common_attr = [
          {"name": "new_coeff", "as": "mc_s2_marketing_compensation_discount_coef"}
        ],
        function_name = "MarketingCompensationPhotoAdjust",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "mc_s2_marketing_compensation_discount_coef", "as": "boost_discount_coeff"},
        ],
        import_item_attr = [
          {"name": score_attr, "as": "score"},
        ],
        export_item_attr = [
          {"name": "score", "as": score_attr}
        ],
        function_name = "BoostOrDiscountV2",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          flag_attr: 1,
          "is_marketing_compensation_photo": 1
        },
      )
    return self

  def mc_replace_cascade_ctr_corr(self, flag_attr):
    """
    Module: RankingScoreModule
    功能: 真实ctr替换ctr队列
    Owner: xuwei09
    Date: 2024-05-09
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_replace_cascade_ctr_corr_alpha", "as": "eff_ctr_corr_alpha"},
        {"name": "explore_replace_cascade_ctr_corr_power", "as": "eff_ctr_corr_power"},
      ],
      import_item_attr = [
        {"name": "cascade_corr_pctr", "as": "pctr"},
        "hetu_tag_level_info_v2__hetu_level_one",
        {"name": "cascade_psvtr", "as": "psvr"},
        "is_picture"
      ],
      export_item_attr = [
        {"name": "pctr", "as": "cascade_corr_pctr_psvr"},
      ],
      function_name = "RealCtrReplaceCtr",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1,
      },
    )
    return self

  def mc_topk_new_photo_pctr(self, flag_attr):
    self.sort(
      score_from_attr = "mc_ensemble_pctr",
      target_item = {
        flag_attr: 1
      },
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "standard_explore_realshow_pid_list", "as": "explore_realshow_ids"},
        {"name": "uStandardExploreRealshowTimestampList", "as": "explore_realshow_timestamps"},
        {"name": "uStandardExploreRealshowHetuTag2List", "as": "explore_realshow_hetu2_list"},
        {"name": "explore_mc_s2_topk_new_photo_pctr_k", "as": "topk"},
        {"name": "explore_mc_s2_topk_new_photo_pctr_realshow_time_window", "as": "time_window"},
        {"name": "explore_mc_s2_topk_new_photo_pctr_realshow_photo_window", "as": "photo_window"}
      ],
      import_item_attr = [
        "hetu_tag_level_info_v2__hetu_level_two",
        {"name": "mc_ensemble_pctr", "as": "pctr"},
      ],
      export_item_attr = [
        {"name": "new_pctr", "as": "topk_new_photo_pctr"},
      ],
      function_name = "CalTopkNewPhotoPxtr",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1,
      },
    ) \
    .enrich_attr_by_light_function(
      item_list_from_attr = "standard_explore_realshow_pid_list",
      import_common_attr = [
        {"name": "uStandardExploreRealshowTimestampList", "as": "explore_realshow_timestamps"},
        {"name": "explore_mc_s2_topk_new_cid_photo_pctr_realshow_time_window", "as": "time_window"},
        {"name": "explore_mc_s2_topk_new_cid_photo_pctr_realshow_photo_window", "as": "photo_window"}
      ],
      import_item_attr = [
        "cluster_id_632",
      ],
      export_common_attr = [
        "realshow_window_cid_list",
      ],
      function_name = "CalRealshowWindowCidList",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        "realshow_window_cid_list",
        {"name": "explore_mc_s2_topk_new_cid_photo_pctr_k", "as": "topk"},
      ],
      import_item_attr = [
        "cluster_id_632",
        {"name": "mc_ensemble_pctr", "as": "pctr"},
      ],
      export_item_attr = [
        {"name": "new_pctr", "as": "topk_new_cid_photo_pctr"},
      ],
      function_name = "CalTopkNewCidPhotoPxtr",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1,
      },
    )
    return self

  def mc_interest_cid_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "uOldMmuClusterId300ListList",
        {"name": "user_develop_interest_cid_and_score_list", "as": "interest_and_score_list"},
        {"name": "mc_s2_interest_cluster_id_num_threshold", "as": "interest_cluster_id_num_threshold"},
        {"name": "mc_s2_identified_interest_cluster_id_num_threshold", "as": "identified_interest_cluster_id_num_threshold"},
        {"name": "mc_s2_unidentified_interest_cluster_id_num_threshold", "as": "unidentified_interest_cluster_id_num_threshold"},
        {"name": "mc_s2_interest_score_cids_ori_boost_alpha_coeff", "as": "interest_score_cids_ori_boost_alpha_coeff"},
        {"name": "mc_s2_interest_score_cids_ori_boost_coeff", "as": "interest_score_cids_ori_boost_coeff"},
        {"name": "mc_s2_identified_interest_boost_alpha_coeff", "as": "identified_interest_boost_alpha_coeff"},
        {"name": "mc_s2_identified_interest_boost_beta_coeff", "as": "identified_interest_boost_beta_coeff"},
        {"name": "mc_s2_identified_interest_boost_omega_coeff", "as": "identified_interest_boost_omega_coeff"},
        {"name": "mc_s2_unidentified_interest_boost_alpha_coeff", "as": "unidentified_interest_boost_alpha_coeff"},
        {"name": "mc_s2_unidentified_interest_boost_beta_coeff", "as": "unidentified_interest_boost_beta_coeff"},
        {"name": "mc_s2_unidentified_interest_boost_omega_coeff", "as": "unidentified_interest_boost_omega_coeff"},
        {"name": "mc_s2_develop_interest_score_lower_bound", "as": "develop_interest_score_lower_bound"},
        {"name": "mc_s2_develop_interest_score_upper_bound", "as": "develop_interest_score_upper_bound"},
        {"name": "mc_s2_identified_interest_score_lower_bound", "as": "identified_interest_score_lower_bound"},
        {"name": "mc_s2_unidentified_interest_score_lower_bound", "as": "unidentified_interest_score_lower_bound"},
        {"name": "enable_mc_s2_interest_score_boost", "as": "enable_interest_score_cids_boost"},
        {"name": "enable_mc_s2_identified_interest_score_boost", "as": "enable_identified_interest_score_cids_boost"},
        {"name": "enable_mc_s2_unidentified_interest_score_boost", "as": "enable_unidentified_interest_score_cids_boost"},
      ],
      import_item_attr = [
        {"name": "interest_cluster_id", "as": "hetu_sim_cluster_id862"},
      ],
      export_item_attr = [
        {"name": "interest_cids_coeff", "as": "cascade_interest_cids_coeff"},
      ],
      function_name = "CalInterestCidsCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1
      }
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "cascade_interest_cids_coeff", "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1
      }
    )
    return self

  def mc_short_uninterest_decay_discount(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "mc_s2_short_uninterest_decay_discount_coeff", "as": "boost_discount_coeff"},
        {"name": "mc_s2_short_uninterest_decay_discount_alpha", "as": "interest_decay_alpha"},
        {"name": "mc_s2_short_uninterest_decay_discount_beta", "as": "interest_decay_beta"},
      ],
      import_item_attr = [
        {"name": "short_uninterest_cid_num", "as": "decay_num"},
      ],
      export_item_attr = [
        {"name": "interest_decay_coeff", "as": "mc_s2_short_uninterest_decay_discount_coeff"}
      ],
      function_name = "CalInterestDecayCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1
      }
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "mc_s2_short_uninterest_decay_discount_coeff", "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1
      }
    )
    return self

  def mc_user_uninterest_cluster_862_adjust(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function( # 用户新兴趣调权
      import_common_attr = [
        {"name": "explore_mc_s2_user_uninterest_cluster_862_adjust_coeff_a", "as": "coeff_a"},
        {"name": "explore_mc_s2_user_uninterest_cluster_862_adjust_coeff_b", "as": "coeff_b"},
        {"name": "explore_mc_s2_user_uninterest_cluster_862_adjust_coeff_c", "as": "coeff_c"},
        {"name": "explore_mc_s2_user_uninterest_cluster_862_adjust_cluster_type", "as": "cluster_type"},
        {"name": "explore_mc_s2_user_uninterest_cluster_862_adjust_version", "as": "adjust_version"},
        {"name": "explore_mc_s2_user_uninterest_cluster_862_adjust_count", "as": "adjust_count"},
      ],
      import_item_attr = [
        "hetu_sim_cluster_id",
        {"name": "mc_s2_interest_id", "as": "cascade_cluster_id"},
        {"name": score_attr, "as": "ensemble_score_attr"},
      ],
      export_item_attr = [
        {"name": "ensemble_score_attr", "as": score_attr},
      ],
      function_name = "UserCluster862Adjust",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr : 1
      }
    )
    return self
  
  # endregion

  # region common

  def mc_pic_boost_coef_with_flag(self, coef_attr, score_attr, flag_attrs, boost_num_max_attr, boost_num_ratio_attr):
    target_map = {flag_attr: 1 for flag_attr in flag_attrs}
    
    self.count_reco_result( # 统计 item 数量
      save_count_to = "mc_pic_flag_target_item_count", 
      target_item = target_map
    ) \
    .gen_common_attr_by_lua( # 计算要 boost 多少个, 同时控制最大比例和个数
      attr_map={
        "mc_pic_flag_target_item_boost_num": f"math.min(math.ceil(mc_pic_flag_target_item_count * {boost_num_ratio_attr}), {boost_num_max_attr})",
      }
    ) \
    .enrich_attr_by_light_function( # 执行 boost
      import_common_attr=[
        {"name": coef_attr, "as": "boost_discount_coeff"},
        {"name": "mc_pic_flag_target_item_boost_num", "as": "topk"}
      ],
      import_item_attr=[{"name": score_attr, "as": "score"},],
      export_item_attr=[{"name": "score", "as": score_attr},],
      function_name="BoostOrDiscountV2",
      class_name="ExploreLightFunctionSetV2",
      target_item=target_map
    )
    return self

  def mc_cs_boost(self, score_attr, flag_attr, stage="prerank"):
    if stage=="prerank":
      kconf_key = "formula.scenarioKey10.explore_cold_combined_score_prerank"
    else:
      kconf_key = "formula.scenarioKey75.explore_cold_combined_score_mc_s2"
    
    self.calc_by_formula1(
      kconf_key = kconf_key,
      import_item_attr = [
        {"name": "explore_stat__real_show_count", "as": "current_impr", "default_val": 0},
        {"name": "explore_stat__click_count", "as": "current_click", "default_val": 0},
        {"name": "cold_item_quality_score", "as": "item_quality_score", "default_val": 0.0},
        {"name": "item_upload_second", "as": "created_second", "default_val": 0},
      ],
      export_formula_value = [
        {"name": "cold_combined_score", "as": "explore_cold_photo_score_%s"%stage}
      ],
      abtest_biz_name = "KUAISHOU_APPS",
      perf_tag = "explore_cold_combined_score_%s"%stage,
      target_item = {
        flag_attr: 1,
        "is_same_author_tail": 1
      }
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "explore_cold_photo_score_%s"%stage, "as": "boost_discount_coeff"},
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr},
      ],
      function_name = "BoostOrDiscountWithItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr: 1,
        "is_same_author_tail": 1
      }
    )
    return self

  # endregion
