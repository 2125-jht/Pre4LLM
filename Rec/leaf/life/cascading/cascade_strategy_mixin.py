#!/usr/bin/env python3
# coding=utf-8

from dragonfly.ext.common_leaf_base_mixin import CommonLeafBaseMixin

class ExploreCascadeStrategyMixin(CommonLeafBaseMixin):
  """
  双列发现页外流粗排策略函数 Mixin 实现
  """

  def boost_young_photo_by_vv(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "basic_info_age_segment_v2",
        {"name": "mc_young_vv_photo_threshold", "as": "young_vv_photo_threshold"},
        {"name": "mc_young_vv_pic_threshold", "as": "young_vv_pic_threshold"},
        {"name": "mc_city_vv_boost_threshold", "as": "city_vv_boost_threshold"},
        {"name": "mc_young_photo_18_23_prob_boost_threshold", "as": "young_photo_18_23_prob_boost_threshold"},
        {"name": "mc_young_photo_24_30_prob_boost_threshold", "as": "young_photo_24_30_prob_boost_threshold"},
        {"name": "mc_young_vv_boost_coeff", "as": "young_vv_boost_coeff"},
        {"name": "mc_city_vv_boost_coeff", "as": "city_vv_boost_coeff"},
        {"name": "mc_young_photo_18_23_prob_boost_coeff", "as": "young_photo_18_23_prob_boost_coeff"},
        {"name": "mc_young_photo_24_30_prob_boost_coeff", "as": "young_photo_24_30_prob_boost_coeff"},
        {"name": "mc_young_age_score_cliff_ratio", "as": "young_age_score_cliff_ratio"},
        {"name": "mc_age_0_12_score_cliff_ratio", "as": "age_0_12_score_cliff_ratio"},
        {"name": "mc_age_12_17_score_cliff_ratio", "as": "age_12_17_score_cliff_ratio"},
        {"name": "mc_age_18_23_score_cliff_ratio", "as": "age_18_23_score_cliff_ratio"},
        {"name": "mc_age_24_30_score_cliff_ratio", "as": "age_24_30_score_cliff_ratio"},
        {"name": "mc_age_31_40_score_cliff_ratio", "as": "age_31_40_score_cliff_ratio"},
        {"name": "mc_age_41_49_score_cliff_ratio", "as": "age_41_49_score_cliff_ratio"},
        {"name": "mc_age_greater_50_score_cliff_ratio", "as": "age_greater_50_score_cliff_ratio"},
        {"name": "mc_enable_personal_cliff_ratio", "as": "enable_personal_cliff_ratio"},
        {"name": "mc_enable_young_photo_boost_rate_threshold", "as": "enable_young_photo_boost_rate_threshold"},
        {"name": "mc_young_photo_boost_rate_threshold", "as": "young_photo_boost_rate_threshold"}
      ],
      import_item_attr = [
        "is_picture",
        "da_young_18_30_vv_rate",
        "da_1_2_city_vv_rate",
        "young_photo_18_23_prob",
        "young_photo_24_30_prob",
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        "young_age_boost_rate",
        {"name": "score", "as": score_attr},
      ],
      target_item = { flag_attr: 1 },
      function_name = "YoungAgeBoost",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def multiply_gate_score(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "hot_cascade_psvtr_gate_alpha", "as": "svtr_alpha"},
        {"name": "hot_cascade_psvtr_gate_beta", "as": "svtr_beta"},
        {"name": "hot_cascade_pctr_gate_alpha", "as": "ctr_alpha"},
        {"name": "hot_cascade_pctr_gate_beta", "as": "ctr_beta"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "es_score"},
        {"name": "mc_ensemble_psvtr", "as": "svtr_score"},
        {"name": "mc_ensemble_pctr", "as": "ctr_score"},
      ],
      export_item_attr = [
        {"name": "es_score", "as": score_attr},
      ],
      target_item = { flag_attr: 1 },
      function_name = "EsScoreMultiplyGate",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def boost_click_count(self, score_attr):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": score_attr, "as": "ensemble_score"},
        "explore_stat__click_count"
      ],
      import_common_attr = [
        "click_thred",
        "boost_click_count_alpha",
        "boost_click_count_beta",
        "boost_click_count_omega",
        "boost_click_val_max",
        "boost_click_val_min"
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      function_name = "BoostClickCount",
      class_name = "ExploreLightFunctionSetV2"
    )
    return self

  def mc_s2_photo_click_boost(self, score_attr_name):
    """
    Module: photo_queue.py
    功能: 根据 xhs 用户在当前item的点击调整分数
    Owner: libingchen
    Date: 2023-06-03
    :return:
    """
    self.enrich_attr_by_light_function( # (libingchen) xhs 原始 click rate boost v2 
      import_common_attr = [
        {"name": "explore_mc_whole_boost_click_count_alpha", "as": "whole_boost_click_count_alpha"},
        {"name": "explore_mc_whole_boost_click_count_beta", "as": "whole_boost_click_count_beta"},
        {"name": "explore_mc_whole_boost_click_count_omega", "as": "whole_boost_click_count_omega"},
        {"name": "explore_mc_outflow_boost_click_count_alpha", "as": "outflow_boost_click_count_alpha"},
        {"name": "explore_mc_outflow_boost_click_count_beta", "as": "outflow_boost_click_count_beta"},
        {"name": "explore_mc_outflow_boost_click_count_omega", "as": "outflow_boost_click_count_omega"},
      ],
      import_item_attr = [
        {"name": score_attr_name, "as": "input_score"},
        {"name": "xhs_install_find_click_value", "as": "whole_click"},
        {"name": "xhs_install_find_outflow_click_value", "as": "outflow_click"}
      ],
      export_item_attr = [
        {"name": "output_score", "as": score_attr_name},
      ],
      function_name = "ExploreBoostByClickRateV2",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def mc_s2_select_photo_by_interest(self, score_attr_name, flag_attr_name):
    self.sort(
       score_from_attr = score_attr_name,
       target_item = {
         flag_attr_name : 1
       }
    ) \
    .explore_control_hetu_count_enricher(
      user_hetu_stat_attr = "colossus_hetu_distribution_hetu_stat",
      hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
      hetu_level_two_attr = "hetu_tag_level_info__hetu_level_two",
      hetu_level_five_attr = "hetu_tag_level_info__hetu_level_five",
      duration_ms_attr = "duration_ms",
      save_flag_to_attr = "mc_s2_diversity_select_flag",
      enable_hetu_control_interest = "{{hot_cascade_enable_hetu_control_interest}}",
      enable_hetu_control_diversity = "{{hot_cascade_enable_hetu_control_diversity}}",
      enable_duration_control_diversity = "{{hot_cascade_enable_duration_control_diversity}}",
      hetu_control_interest_start = "{{hot_cascade_hetu_control_interest_start}}",
      hetu_control_diversity_start = "{{hot_cascade_hetu_control_diversity_start}}",
      duration_control_diversity_start = "{{hot_cascade_duration_control_diversity_start}}",
      keep_size = "{{mc_final_candidate_num}}",
      hetu1_max_size = "{{hot_cascade_control_hetu1_max_size}}",
      hetu2_max_size = "{{hot_cascade_control_hetu2_max_size}}",
      hetu5_max_size = "{{hot_cascade_control_hetu5_max_size}}",
      duration_0s_max_size = "{{hot_cascade_control_duration_0s_max_size}}",
      duration_0_7s_max_size = "{{hot_cascade_control_duration_0_7s_max_size}}",
      duration_7_9s_max_size = "{{hot_cascade_control_duration_7_9s_max_size}}",
      duration_9_12s_max_size = "{{hot_cascade_control_duration_9_12s_max_size}}",
      duration_12_17s_max_size = "{{hot_cascade_control_duration_12_17s_max_size}}",
      duration_17_20s_max_size = "{{hot_cascade_control_duration_17_20s_max_size}}",
      target_item = {
        flag_attr_name : 1
      }
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "mc_s2_diversity_select_flag", "as": "flag"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr_name},
      ],
      function_name = "SetMinimumScoreByFlag",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr_name : 1
      }
    )
    return self

  def boost_hot_content_retr(self, score_attr_name, flag_attr_name):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "mc_s2_hot_content_retr_boost_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": score_attr_name, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr_name},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        flag_attr_name : 1,
        "reason" : [10030, 10031, 10032]
      }
    )
    return self

  def refinement_boost_personified_author(self, score_attr, flag_attr):
    """
    Module: photo_queue
    功能: 细分用户和视频维度，精细化对人格化账号提权
    Owner: xubaoquan
    Date: 2023-07-12
    :return:
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "basic_info_age_segment_v2", "as": "basic_info_age_segment_v2"},
        {"name": "basic_info_gender_v2", "as": "basic_info_gender_v2"},
        {"name": "explore_personifed_author_boost_ptr", "as": "boost_map_ptr"},
        {"name": "refinement_boost_personified_author_redis_prefix", "as": "redis_prefix"},
        {"name": "cascade_refinement_boost_personified_author_power_weight", "as": "power_weight"},
      ],
      import_item_attr = [
        {"name": "author__gender", "as": "author__gender"},
        {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_level_one_list"},
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr},
      ],
      target_item = { 
        flag_attr: 1,
        "eyeshot_source" : 1
      },
      function_name = "UniverseRefinementBoost",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self


  def calc_pic_set_variety_score(self, flag_attr):
    """
    图文粗排多样性队列
    通过计算 item 与候选集中心 embedding 的距离，及该 item 的离散程度，作为多样性分
    Owner: caozhong
    Date: 2023-07-31
    :param flag_attr: 粗排分组 attr
    :return:
    """
    self.enrich_attr_by_light_function(
        import_common_attr=[
          {"name": "pic_cascade_variety_embedding_size", "as": "emb_size"},
          {"name": "pic_cascade_variety_score_alpha", "as": "alpha"},
          {"name": "pic_cascade_variety_score_beta", "as": "beta"},
          {"name": "pic_cascade_variety_score_ctr_alpha", "as": "ctr_alpha"},
          {"name": "pic_cascade_variety_score_ctr_beta", "as": "ctr_beta"},
        ],
        import_item_attr=[
          {"name": "pic_mmu_embedding", "as": "item_embedding"},
          {"name": "mc_ensemble_pctr", "as": "mc_pctr"}
        ],
        export_item_attr=[
          "pic_variety_score",
        ],
        function_name="CalcPicVarietyScore",
        class_name="ExploreLightFunctionSetV2",
        target_item={
          flag_attr: 1,
        },
      )
    return self

  def boost_pic_cascade_s1_es_by_follow_author(self, score_attr, flag_attr):
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
            {"name": "is_picture_follow_author", "as": "need_item_attr"},
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

  def boost_pic_cascade_s1_es_by_caption(self, score_attr, flag_attr):
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
          export_common_attr=[
            {"name": "boost_count", "as": "cascade_channel_caption_photo_boost_count"},
          ],
          function_name="BoostWithCaption",
          class_name="ExploreLightFunctionSetV2",
          target_item={
            flag_attr: 1
          }
        )
    return self

  def boost_pic_cascade_s1_es_by_target_hetu(self, score_attr, flag_attr):
    """
    特定河图类目 pic_s1_es boost
    Owner: chenqiaojun, caozhong
    Date: 2023-07-31
    :param score_attr:
    :param flag_attr:
    :return:
    """
    self.enrich_attr_by_light_function(
          import_common_attr=[
            {"name": "cascade_target_hetu_pic_mc_s1_boost_coeff", "as": "boost_discount_coeff"},
          ],
          import_item_attr=[
            {"name": "is_boost_hetu_pic", "as": "need_item_attr"},
            {"name": score_attr, "as": "ensemble_score"},
          ],
          export_item_attr=[
            {"name": "ensemble_score", "as": score_attr},
          ],
          function_name="BoostOrDiscount",
          class_name="ExploreLightFunctionSetV2",
          target_item={
            flag_attr: 1,
          }
        )
    return self

  def boost_pic_cascade_s1_es_by_hetu_ratio(self, score_attr, flag_attr):
    """
    根据河图占比调整 pic_s1_es
    Owner: chenqiaojun, caozhong
    Date: 2023-07-31
    :param score_attr:
    :param flag_attr:
    :return:
    """
    self.sort(
          score_from_attr=score_attr,
          target_item={
            flag_attr: 1
          }
        ) \
        .enrich_attr_by_light_function(
          import_common_attr=[
            {"name": "cascade_s1_hetu_decay_coeff", "as": "decay_coeff"},
            {"name": "cascade_s1_hetu_decay_keep_size_coeff", "as": "decay_keep_size_coeff"},
          ],
          import_item_attr=[
            {"name": score_attr, "as": "score"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_level_one_list"},
            {"name": "is_key_target_hetu_pic", "as": "is_target_hetu"},
          ],
          export_item_attr=[
            {"name": "score", "as": score_attr},
          ],
          function_name="HetuRatioDecay",
          class_name="ExploreLightFunctionSetV2",
          target_item={
            flag_attr: 1
          }
        )
    return self

  def boost_pic_cascade_s1_es_by_revisited(self, score_attr, flag_attr):
    """
    复访作品调整 pic_s1_es
    Owner: chenqiaojun, caozhong
    Date: 2023-07-31
    :param score_attr:
    :param flag_attr:
    :return:
    """
    self.enrich_attr_by_light_function(
          import_common_attr=[
            {"name": "cascade_revisited_item_boost_coef", "as": "boost_weight"}
          ],
          import_item_attr=[
            {"name": score_attr, "as": "ensemble_score"},
          ],
          export_item_attr=[
            {"name": "ensemble_score", "as": score_attr}
          ],
          function_name="EnsembleScoreBoost",
          class_name="ExploreLightFunctionSetV2",
          target_item={
            flag_attr: 1,
            "reason": 13071,
          },
        )
    return self

  def boost_pic_cascade_s1_es_by_hetu_distribution(self, score_attr, flag_attr):
    """
    根据候选集hetu分布 和 用户历史河图分布 调整 pic_s1_es
    Owner: gaodong, gengxiao
    Date: 2023-07-31
    :param score_attr:
    :param flag_attr:
    :return:
    """
    self.sort(
          score_from_attr=score_attr,
        ) \
        .explore_photo_distribution_adjust_enricher(
          colossus_total_count_attr="colossus_hetu_distribution_total_count",
          user_hetu_stat_attr="colossus_hetu_distribution_hetu_stat",
          colossus_total_count_threshold="{{cascading_s1_pic_hetu_distribution_colossus_total_count_threshold}}",
          max_count="{{cascading_s1_pic_hetu_distribution_max_count}}",
          global_fuse_corr="{{cascading_s1_pic_hetu_distribution_global_fuse_corr}}",
          hetu_level_one_attr="hetu_tag_level_info__hetu_level_one",
          candidate_hetu_adjust_coeff_map_attr="candidate_hetu_adjust_coeff_map"
        ) \
        .enrich_attr_by_light_function(
          import_common_attr=[
            {"name": "cascading_s1_pic_hetu_distribution_hetu_coef_alpha", "as": "hetu_coef_alpha"},
            {"name": "cascading_s1_pic_hetu_distribution_hetu_coef_beta", "as": "hetu_coef_beta"},
            {"name": "cascading_s1_pic_hetu_distribution_hetu_discount_threshold", "as": "hetu_discount_threshold"},
            {"name": "cascading_s1_pic_hetu_distribution_hetu_encourage_threshold", "as": "hetu_encourage_threshold"},
            "candidate_hetu_adjust_coeff_map",
          ],
          import_item_attr=[
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_level_one_list"},
            {"name": score_attr, "as": "es_score"},
          ],
          export_item_attr=[
            {"name": "es_score", "as": score_attr},
          ],
          function_name="AdjustScoreByHetuDistribution",
          class_name="ExploreLightFunctionSetV2",
        )
    return self
  
  def prerank_search_score_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_prerank_search_score_boost_coeff", "as": "boost_discount_coeff"},
        {"name": "explore_prerank_search_score_boost_thres", "as": "boost_discount_thres"},
      ],
      import_item_attr = [
        {"name": "search_score", "as": "need_item_attr"},
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      target_item = { flag_attr: 1 },
      function_name = "BoostOrDiscountWithThres",
      class_name = "ExploreLightFunctionSetV2",
    ) \

    return self

  def prerank_user_pos_hetu_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "life_prerank_user_pos_hetu_boost_coeff", "as": "boost_coeff"},
        {"name": "user_positive_hetu2_list", "as": "pos_hetu_list"}
      ],
      import_item_attr = [
        {"name": "hetu_tag_level_info__hetu_level_two", "as": "hetu_list"},
        {"name": score_attr, "as": "ensemble_score"}
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      target_item = { flag_attr: 1 },
      function_name = "UserPositiveHetuEsBoost",
      class_name = "ExploreLifeLightFunctionSet",
    ) \

    return self

  def mc_calc_search_score(self, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_search_score_threshold", "as": "ann_dist_threshold"}, 
      ],
      import_item_attr = [
        {"name": "q2i_ann_score", "as": "ann_dist_list"},  
      ],
      export_item_attr = [
          {"name": "ann_dist", "as": "search_score"},  
      ],
      target_item = { flag_attr: 1 },
      function_name = "AnnCalThresholdValueForDistList",
      class_name = "ExploreLightFunctionSetV2",
    ) \

    return self

  def mc_search_score_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "explore_mc_search_score_boost_coeff", "as": "boost_discount_coeff"},
        {"name": "explore_mc_search_score_boost_thres", "as": "boost_discount_thres"},
      ],
      import_item_attr = [
        {"name": "search_score", "as": "need_item_attr"},
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      target_item = { flag_attr: 1 },
      function_name = "BoostOrDiscountWithThres",
      class_name = "ExploreLightFunctionSetV2",
    ) \

    return self

  def mc_user_pos_hetu_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "life_mc_user_pos_hetu_boost_coeff", "as": "boost_coeff"},
        {"name": "user_positive_hetu2_list", "as": "pos_hetu_list"}
      ],
      import_item_attr = [
        {"name": "hetu_tag_level_info__hetu_level_two", "as": "hetu_list"},
        {"name": score_attr, "as": "ensemble_score"}
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      target_item = { flag_attr: 1 },
      function_name = "UserPositiveHetuEsBoost",
      class_name = "ExploreLifeLightFunctionSet",
    ) \

    return self

  def prerank_hotfire_yellow_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "life_prerank_hotfire_yellow_boost_coef", "as": "boost_discount_coeff"},
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
        flag_attr : 1,
        "is_hotfire_yellow" : 1
      }
    ) \

    return self

  def mc_hotfire_yellow_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "life_mc_hotfire_yellow_boost_coef", "as": "boost_discount_coeff"},
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
        flag_attr : 1,
        "is_hotfire_yellow" : 1
      }
    ) \

    return self

  def gen_is_marketing_compensation_photo(self):
    self.split_string(
      input_common_attr = "life_marketing_compensation_photo_tags_list_str",
      output_common_attr = "life_marketing_compensation_photo_tags_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "life_marketing_compensation_photo_tags_list", "as": "tags_list"},
        {"name": "life_marketing_compensation_high_value_author_ignore", "as": "high_value_author_ignore"},
        {"name": "life_marketing_compensation_open_reason_thres", "as": "open_reason_thres"},
        "high_value_black_author_map_ptr"
      ],
      import_item_attr = [
        "sirius_distribution_info__mark_cod",
        "author__id"
      ],
      export_item_attr = [
        "is_marketing_compensation_photo"
      ],
      function_name = "GenIsMarketingCompensationPhoto",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def mc_s2_marketing_compensation_discount(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "life_mc_s2_marketing_compensation_discount_coef", "as": "boost_discount_coeff"},
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
          flag_attr : 1,
          "is_marketing_compensation_photo" : 1
        }
    )
    return self

  def fountain_gen_is_marketing_compensation_photo(self):
    self.split_string(
      input_common_attr = "life_fountain_marketing_compensation_photo_tags_list_str",
      output_common_attr = "life_fountain_marketing_compensation_photo_tags_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "life_fountain_marketing_compensation_photo_tags_list", "as": "tags_list"},
        {"name": "life_fountain_marketing_compensation_high_value_author_ignore", "as": "high_value_author_ignore"},
        {"name": "life_fountain_marketing_compensation_open_reason_thres", "as": "open_reason_thres"},
        "high_value_black_author_map_ptr"
      ],
      import_item_attr = [
        "sirius_distribution_info__mark_cod",
        "author__id"
      ],
      export_item_attr = [
        "is_marketing_compensation_photo"
      ],
      function_name = "GenIsMarketingCompensationPhoto",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def mc_marketing_compensation_adjust(self):
    self \
      .if_("life_fountain_enable_mc_calc_marketing_compensation_coeff == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "life_fountain_mc_marketing_compensation_adjust_ctr_weight", "as": "ctr_weight"},
            {"name": "life_fountain_mc_marketing_compensation_adjust_watchtime_weight", "as": "watchtime_weight"},
            {"name": "life_fountain_mc_marketing_compensation_adjust_score_base", "as": "score_base"},
            {"name": "life_fountain_mc_marketing_compensation_adjust_adjust_version", "as": "adjust_version"},
            {"name": "life_fountain_mc_marketing_compensation_adjust_score_base_ratio", "as": "score_base_ratio"},
          ],
          import_item_attr = [
            {"name": "cascade_pctr", "as": "ctr"},
            {"name": "cascade_pwatch_time", "as": "watchtime"},
          ],
          export_item_attr = [
            {"name": "coeff", "as": "mc_marketing_compensation_coeff"},
          ],
          function_name = "CalcRewardCoeff",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {"is_marketing_compensation_photo": 1}
        ) \
      .end_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "life_fountain_mc_marketing_compensation_adjust_scale_factor", "as": "scale_factor"},
          {"name": "life_fountain_mc_marketing_compensation_adjust_base_coeff", "as": "base_coeff"},
        ],
        import_item_attr = [
          {"name": "mc_marketing_compensation_coeff", "as": "reward_coeff"},
        ],
        export_item_attr = [
          {"name": "new_coeff", "as": "mc_adjust_coeff_final"},
        ],
        function_name = "MarketingCompensationPhotoAdjust",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {"is_marketing_compensation_photo": 1}
      )
    return self

  def gen_is_olympic_photo(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "life_olympic_latest_photo_hour_limit", "as": "hour_limit"}
      ],
      import_item_attr = [
        "upload_time",
        "hetu_tag_level_info__hetu_tag"
      ],
      export_item_attr = [
        "is_olympic",
        "is_olympic_latest"
      ],
      function_name = "GenIsOlympicPhoto",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def prerank_search_topk_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "life_prerank_search_topk_boost_coef", "as": "boost_coeff"},
        {"name": "life_prerank_search_boost_topk", "as": "topk"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr},
      ],
      function_name = "BoostTopk",
      class_name = "ExploreLifeLightFunctionSet",
      target_item = {
        flag_attr : 1,
        "reason" : [2704]
      }
    )
    return self

  def mc_search_topk_boost(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "life_mc_search_topk_boost_coef", "as": "boost_coeff"},
        {"name": "life_mc_search_boost_topk", "as": "topk"},
      ],
      import_item_attr = [
        {"name": score_attr, "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": score_attr},
      ],
      function_name = "BoostTopk",
      class_name = "ExploreLifeLightFunctionSet",
      target_item = {
        flag_attr : 1,
        "reason" : [2704]
      }
    )
    return self

  def life_mc_s2_diversity_weight_adjust(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "recent_unclick_rate", "as": "alpha"},
        {"name": "life_mc_s2_diversity_weight_adjust_beta", "as": "beta"},
        {"name": "life_mc_s2_diversity_weight_adjust_gamma", "as": "gamma"},
        {"name": "life_mc_s2_diversity_weight_adjust_coeff_max", "as": "adjust_coeff_max"},
        {"name": "xlife_mc_ensemble_s2_diversity_fr_power_weight", "as": "input_weight"},
      ],
      export_common_attr = [
        {"name": "output_weight", "as": "xlife_mc_ensemble_s2_diversity_fr_power_weight"},
      ],
      function_name = "AdjustWeightByMultiply",
      class_name = "ExploreLifeLightFunctionSet",
    )
    return self

  def get_user_group_emp_xtr(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "basic_info_age_segment_v2",
        "basic_info_gender_v2",
      ],
      export_common_attr = [
        "emp_xtr_user_group_prefix_ltr",
        "emp_xtr_user_group_prefix_wtr",
        "emp_xtr_user_group_prefix_ftr",
        "emp_xtr_user_group_prefix_cmtr",
      ],
      function_name = "CalUserGroupEmpXtrPrefix",
      class_name = "ExploreLifeLightFunctionSet",
    ) \
    .get_kconf_params(
      kconf_configs = [
        {
          "kconf_key": "reco.eyeshot.lifeUserGroupAgeGenderEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_ltr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_ltr"
        },
        {
          "kconf_key": "reco.eyeshot.lifeUserGroupAgeGenderEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_wtr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_wtr"
        },
        {
          "kconf_key": "reco.eyeshot.lifeUserGroupAgeGenderEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_ftr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_ftr"
        },
        {
          "kconf_key": "reco.eyeshot.lifeUserGroupAgeGenderEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_cmtr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_cmtr"
        }
      ]
    )
    return self

  def gen_is_low_cost_photo(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "negative_aid_set_ptr", "as": "aid_set_ptr"}
      ],
      import_item_attr = [
        "author__id"
      ],
      export_item_attr = [
        {"name": "is_target_photo", "as": "is_low_cost_photo"}
      ],
      function_name = "AidInSet",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def prerank_low_cost_photo_discount(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "life_prerank_low_cost_photo_discount_coef", "as": "boost_discount_coeff"},
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
        flag_attr : 1,
        "is_low_cost_photo" : 1
      }
    )
    return self

  def mc_low_cost_photo_adjust(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "life_fountain_mc_low_cost_photo_discount_coeff", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "mc_adjust_coeff_final", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "mc_adjust_coeff_final"},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {"is_low_cost_photo": 1}
    )
    return self

  def gen_is_minority_photo(self):    
    self.split_string(
      input_common_attr = "life_minority_photo_tags_bits_list_str",
      output_common_attr = "life_minority_photo_tags_bits_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True,
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "life_minority_photo_tags_bits_list", "as": "minority_photo_bits_list"},
      ],
      import_item_attr = [
        "data_set_tags_bit",
      ],
      export_item_attr = [
        "is_minority_photo",
      ],
      function_name = "IsMinorityPhotoV2",
      class_name = "ExploreLifeLightFunctionSet",
    )
    return self

  def llm_negative_photo_adjust(self, score_attr, flag_attr):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "life_explore_mc_s2_llm_negative_photo_adjust_tag_coeff_map_str", "as": "tag_coeff_map_str"},
      ],
      import_item_attr = [
        "hetu_tag_level_info_v2__hetu_tag",
        "explore_stat__click_count",
        "explore_stat__report_count",
        "fountain_stats__real_show_count",
        "fountain_stats__report_count",
        {"name": score_attr, "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": score_attr}
      ],
      function_name = "LlmNegativePhotoAdjust",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def mc_llm_negative_photo_adjust(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "life_fountain_mc_llm_negative_photo_adjust_tag_coeff_map_str", "as": "tag_coeff_map_str"},
      ],
      import_item_attr = [
        "hetu_tag_level_info_v2__hetu_tag",
        "explore_stat__click_count",
        "explore_stat__report_count",
        "fountain_stats__real_show_count",
        "fountain_stats__report_count",
        {"name": "mc_adjust_coeff_final", "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "mc_adjust_coeff_final"}
      ],
      function_name = "LlmNegativePhotoAdjust",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
  
  def gen_tired_switch_behave_ids(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "colossus_photo_id_list",
        "colossus_play_time_list",
        "colossus_timestamp_list",
        "colossus_channel_list",
      ],
      export_common_attr = [
        "is_siwtched_from_danlie",
        "is_tired_of_danlie",
        "last_danlie_photo_id_list",
        "last_tired_photo_id_list",
      ],
      function_name = "GenTiredSwitchBehaveIds",
      class_name = "ExploreLifeLightFunctionSet",
    )
    return self 