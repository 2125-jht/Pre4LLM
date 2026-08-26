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
            "interact_cost",
            "awesome_wtd_score",
            "dtr",
            "pdbfrtr",
            "interact_fusion",
            "watch_time_fusion",
            "frctr_fusion",
            "frcltr_fusion",
            "rerank_pure_value_score",
            "corr_fetr",
            "corr_fountain_eff",
            "rerank_cpr_corr",
            "rerank_pevtr_corr",
            "pltr",
            "pftr",
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
            "fr_ctcvr_score",
            "fr_ctcvr_gmv_score",
            "fr_elive_ctcvr_score",
            "fr_elive_ctcvr_gmv_score",
            "produce_mtctr", 
            "produce_twhtr", 
            "produce_mfctr",
            "produce_mtcotr", 
            "produce_mtjtr", 
            "produce_mtm1",
            "produce_upload_sum_score",
            "produce_consuv_sum_score",
            "life_ltr_pctr",
            "psvr",
            "gen_l2r_fusion_score",
            "listwise_distill_score",
            "rank_distill_ctr",
            "rank_distill_ltr",
            "coordinated_watchtime_score",
            "duration_ms",
            "sensitive_related_score",
            ]
    prefix = "{{explore_rerank_gen_seed_ensemble_"
    for i, q in enumerate(queues) :
      t = {"name" : q }
      t.update({"weight_base" : prefix + q + '_weight}}'})
      t.update({"bias_range" : prefix + q + '_range}}'})
      t.update({"weight_lower_bound" : prefix + q + '_lower_bound}}'})
      t.update({"raw_pow_weight" : prefix + q + '_raw_pow_weight}}'})
      t.update({"raw_weight" : prefix + q + '_raw_weight}}'})
      t.update({"raw_weight_multiply" : prefix + q + '_raw_weight_multiply}}'})
      t.update({"raw_pow_weight_multiply" : prefix + q + '_raw_pow_weight_multiply}}'})
      t.update({"score_norm" : prefix + q + '_score_norm}}'})
      t.update({"que_discount_coef" : prefix + q + '_que_discount_coef}}'})
      t.update({"weight_addAndMul" : prefix + q + '_weight_addAndMul}}'})
      t.update({"raw_pow_weight_addAndMul" : prefix + q + '_raw_pow_weight_addAndMul}}'})
      t.update({"raw_weight_addAndMul" : prefix + q + '_raw_weight_addAndMul}}'})
      if (q == "cltr"):
        t.update({"variant_weight" : "{{rerank_cltr_score_variant_weight}}"})
        t.update({"avg_xtr" : "{{avg_cltr}}"})
        t.update({"dynamic_weight" : "{{rerank_cltr_score_dynamic_weight}}"})
        t.update({"min_ratio" : "{{rerank_cltr_score_min_ratio}}"})
        t.update({"max_ratio" : "{{rerank_cltr_score_max_ratio}}"})
        t.update({"user_xtr" : "{{realtime_cltr}}"})
      res.append(t)

    return res

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
          enabled = "{{enable_life_rerank_minority_photo_diversity}}",
          window_size = "{{life_rerank_minority_photo_diversity_winsize}}",
          max_num = "{{life_rerank_minority_photo_diversity_max_num}}",
          priority = "{{life_rerank_minority_photo_diversity_priority}}"),
    dict(attr_name= "is_grpr_pron_photo",
          enabled="{{rerank_variety_shuanglie_enable12}}",
          window_size= "{{rerank_variety_shuanglie_winsize12}}",
          max_num="{{rerank_variety_shuanglie_max12}}",
          priority="{{rerank_variety_shuanglie_priority12}}"),
    dict(attr_name= "author__id",
          enabled="{{rerank_variety_shuanglie_enable11}}",
          window_size= "{{rerank_variety_shuanglie_winsize11}}",
          max_num="{{rerank_variety_shuanglie_max11}}",
          priority="{{rerank_variety_shuanglie_priority11}}"),
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
          priority="{{rerank_variety_shuanglie_priority8}}"),
    dict(attr_name= "is_follow_author",
          enabled="{{rerank_variety_shuanglie_enable7}}",
          window_size="{{rerank_variety_shuanglie_winsize7}}",
          max_num="{{rerank_variety_shuanglie_max7}}",
          priority="{{rerank_variety_shuanglie_priority7}}"),
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
          priority="{{rerank_variety_shuanglie_priority2}}")
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
      "pxtr_attr": "awesome_wtd",
      "weight_attr": "explore_dpp_hetu_ensemble_fullrank_awesome_score",
    },
    {
      "pxtr_attr": "ctr",
      "weight_attr": "explore_dpp_hetu_ensemble_fullrank_click_score",
    },
    {
      "pxtr_attr": "fr_score1",
      "weight_attr": "explore_dpp_hetu_ensemble_fullrank_fr_score1_score",
    },
    {
      "pxtr_attr": "fr_score2",
      "weight_attr": "explore_dpp_hetu_ensemble_fullrank_fr_score2_score",
    },
    {
      "pxtr_attr": "interact_fusion",
      "weight_attr": "explore_dpp_hetu_ensemble_fullrank_interact_fusion_score",
    },
    {
      "pxtr_attr": "interact_cost",
      "weight_attr": "explore_dpp_hetu_ensemble_fullrank_interact_cost_score",
    },

  ]
  return queues

class RerankGenList(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def enrich_item_attr(self, target_item = {}):
    return self.flow \
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
          lua_script_file = "life/rerank/lua/module/rerank_gen_list__multi_lua.lua",
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
          lua_script_file = "life/rerank/lua/module/rerank_gen_list__multi_lua.lua",
          range_end = "{{dpp_diversity_candidate_size}}",
          target_item = target_item,
      ) \
      .if_("enable_get_pic_coff == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
              "rerank_single_pic_coff",
              "rerank_nosingle_pic_coff"
          ],
          import_item_attr = [
              "duration_ms",
              "upload_type",
              "picture_type",
              "photo_picture_count"
          ],
          export_item_attr = [
              "rerank_pic_coff_attr_transfer"
          ],
          function_name = "RerankPicCoff",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
              "is_picture" : 1
          }
        ) \
      .end_() \
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
      .if_("enable_get_duration_debias == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
              "duration_debias_cluster_num",
          ],          
          import_item_attr = [
              "duration_ms",
          ],
          export_item_attr = [
              "corr_pctr_debias_name",
              "corr_pwtr_debias_name",
              "pltr_debias_name",
              "pftr_debias_name",
              "plvtr_debias_name",
              "pcltr_debias_name",
              "pcmtr_debias_name",
          ],
          function_name = "GetXtrDurationDebiasBucketName",
          class_name = "ExploreLightFunctionSetV2",
          range_end = "{{dpp_diversity_candidate_size}}",
          target_item = target_item,
        ) \
        .get_kconf_params(
          kconf_configs = [
              {
                  "kconf_key": "reco.offline.exploreDurationDebiasConfig",
                  "json_path": "{{corr_pctr_debias_name}}",
                  "default_value": 0.0,
                  "export_item_attr": "pctr_duration_debias_kconf"
              },
              {
                  "kconf_key": "reco.offline.exploreDurationDebiasConfig",
                  "json_path": "{{corr_pwtr_debias_name}}",
                  "default_value": 0.0,
                  "export_item_attr": "pwtr_duration_debias_kconf"
              },
              {
                  "kconf_key": "reco.offline.exploreDurationDebiasConfig",
                  "json_path": "{{pltr_debias_name}}",
                  "default_value": 0.0,
                  "export_item_attr": "pltr_duration_debias_kconf"
              },
              {
                  "kconf_key": "reco.offline.exploreDurationDebiasConfig",
                  "json_path": "{{pftr_debias_name}}",
                  "default_value": 0.0,
                  "export_item_attr": "pftr_duration_debias_kconf"
              },
              {
                  "kconf_key": "reco.offline.exploreDurationDebiasConfig",
                  "json_path": "{{plvtr_debias_name}}",
                  "default_value": 0.0,
                  "export_item_attr": "plvtr_duration_debias_kconf"
              },
              {
                  "kconf_key": "reco.offline.exploreDurationDebiasConfig",
                  "json_path": "{{pcltr_debias_name}}",
                  "default_value": 0.0,
                  "export_item_attr": "pcltr_duration_debias_kconf"
              },
              {
                  "kconf_key": "reco.offline.exploreDurationDebiasConfig",
                  "json_path": "{{pcmtr_debias_name}}",
                  "default_value": 0.0,
                  "export_item_attr": "pcmtr_duration_debias_kconf"
              }
          ]
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
              "duration_debias_weight_pctr",
              "duration_debias_weight_pwtr",
              "duration_debias_weight_pltr",
              "duration_debias_weight_pftr",
              "duration_debias_weight_plvtr",
              "duration_debias_weight_pcltr",
              "duration_debias_weight_pcmtr",
          ],
          import_item_attr = [
              "pctr_duration_debias_kconf",
              "pwtr_duration_debias_kconf",
              "pltr_duration_debias_kconf",
              "pftr_duration_debias_kconf",
              "plvtr_duration_debias_kconf",
              "pcltr_duration_debias_kconf",
              "pcmtr_duration_debias_kconf",
          ],
          export_item_attr = [
              "pctr_duration_debias_coffe",
              "pwtr_duration_debias_coffe",
              "pltr_duration_debias_coffe",
              "pftr_duration_debias_coffe",
              "plvtr_duration_debias_coffe",
              "pcltr_duration_debias_coffe",
              "pcmtr_duration_debias_coffe",
          ],
          function_name = "GetXtrDurationTransferCoffe",
          class_name = "ExploreLightFunctionSetV2",
          range_end = "{{dpp_diversity_candidate_size}}",
          target_item = target_item,
        ) \
      .end_() \
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
              "explore_fullrank_pure_value_score",
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
              "rerank_pure_value_score",
              "rerank_cpr_corr",
              "rerank_pevtr_corr",
              "min_act_rank",
              "gen_l2r_score_corr",
          ],
          function_for_item = "full_rank_score_cal",
          lua_script_file = "life/rerank/lua/module/rerank_gen_list__multi_lua.lua",
          range_end = "{{dpp_diversity_candidate_size}}",
          target_item = target_item,
      ) \
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
        lua_script_file = "life/rerank/lua/module/rerank_gen_list__multi_lua.lua",
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
      .enrich_attr_by_light_function(
        import_item_attr = [
          "hetu_tag_level_info__hetu_level_five"
        ],
        import_common_attr = [
          "specified_hetu5_str"
        ],
        export_item_attr = [
          "specified_hetu5_found"
        ],
        function_name = "DiversitySpecifiedHetu5",
        class_name = "ExploreLightFunctionSetV2",
        range_end = "{{dpp_diversity_candidate_size}}",
        target_item = target_item,
      ) \
      .if_("rerank_variety_shuanglie_enable10 == 1") \
        .copy_attr(
          attrs=[{
            "from_item": "is_picture",
            "to_item": "video_variant_attr"
          }],
          target_item = {
            "is_picture": 0
          }
        ) \
      .end_() \


  def enrich_common_attr(self):
    return self.flow \
      .enrich_with_protobuf(
          from_extra_var = "user_info_ptr",
          attrs = [
            dict(name="is_shuffle", path="feature_collection.is_shuffle")
          ]
      ) \
      .if_("explore_open_low_active_first_view_diversity == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "rerank_variety_shuanglie_enable8",
            "refreshTimes"
          ],
          export_common_attr = [
            "rerank_variety_shuanglie_enable8"
          ],
          function_name = "OnlyLowActiveFirstViewEnableRule",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("explore_rerank_sort_weight_adjust == 1") \
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
      .if_("rerank_enable_jarvis_param > 0 and fullrank_enable_jarvis_param > 0") \
        .gen_common_attr_by_lua(
          attr_map = {
            "explore_rerank_gen_seed_ensemble_ctr_weight" : "rerank_ctr_adjust_ratio_attr * explore_rerank_gen_seed_ensemble_ctr_weight",
            "explore_rerank_gen_seed_ensemble_ltr_weight" : "rerank_ltr_adjust_ratio_attr * explore_rerank_gen_seed_ensemble_ltr_weight",
            "explore_rerank_gen_seed_ensemble_wtr_weight" : "rerank_wtr_adjust_ratio_attr * explore_rerank_gen_seed_ensemble_wtr_weight",
            "explore_rerank_gen_seed_ensemble_ftr_weight" : "rerank_ftr_adjust_ratio_attr * explore_rerank_gen_seed_ensemble_ftr_weight",
            "explore_rerank_gen_seed_ensemble_cltr_weight" : "rerank_cltr_adjust_ratio_attr * explore_rerank_gen_seed_ensemble_cltr_weight",
            "explore_rerank_gen_seed_ensemble_ptr_weight" : "rerank_ptr_adjust_ratio_attr * explore_rerank_gen_seed_ensemble_ptr_weight",
            "explore_rerank_gen_seed_ensemble_cmtr_weight" : "rerank_cmtr_adjust_ratio_attr * explore_rerank_gen_seed_ensemble_cmtr_weight",
            "explore_rerank_gen_seed_ensemble_cmef_weight" : "rerank_cmef_adjust_ratio_attr * explore_rerank_gen_seed_ensemble_cmef_weight",
            "explore_rerank_gen_seed_ensemble_epstr_weight" : "rerank_epstr_adjust_ratio_attr * explore_rerank_gen_seed_ensemble_epstr_weight",
            "explore_rerank_gen_seed_ensemble_fr_score1_corr_weight" : "rerank_fr_score1_adjust_ratio_attr * explore_rerank_gen_seed_ensemble_fr_score1_corr_weight",
            "explore_rerank_gen_seed_ensemble_fr_score2_corr_weight" : "rerank_fr_score2_adjust_ratio_attr * explore_rerank_gen_seed_ensemble_fr_score2_corr_weight",
            "explore_rerank_gen_seed_ensemble_fetr_corr_weight" : "rerank_fetr_adjust_ratio_attr * explore_rerank_gen_seed_ensemble_fetr_corr_weight",
            "explore_rerank_gen_seed_ensemble_feff_weight" : "rerank_fountain_eff_adjust_ratio_attr * explore_rerank_gen_seed_ensemble_feff_weight",
            "explore_rerank_gen_seed_ensemble_duration_gt_58s_corr_weight" : "rerank_duration_adjust_ratio_attr * explore_rerank_gen_seed_ensemble_duration_gt_58s_corr_weight",
            "explore_rerank_gen_seed_ensemble_l2r_score_weight" : "rerank_l2r_score_adjust_ratio_attr * explore_rerank_gen_seed_ensemble_l2r_score_weight",
            "explore_rerank_gen_seed_ensemble_ensemble_score_weight" : "rerank_ensemble_score_adjust_ratio_attr * explore_rerank_gen_seed_ensemble_ensemble_score_weight",
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
      .if_("rerank_enable_cover_history > 0") \
        .gen_realtime_browse_set(
          enable_fountain_browse = False,
          enable_hot_browse = True,
          output_common_attr = "hotCoverHistory",
          realtime_hot_bs_size = "{{hot_cover_history_size}}",
          profile_time_threshold = "{{hot_cover_profile_time_threshold}}",
          user_info_ptr_attr = "user_info_ptr",
          enable_fix_real_show_list = "{{explore_enable_fix_real_show_list}}"
        ) \
      .end_() \

  def sequence_generator(self, target_item = {}):
    return self.flow \
      .split_string(
        input_common_attr="explore_rerank_pic_fixed_slot_config",
        output_common_attr="pic_fixed_slot_conf_list",
        delimiters=";",
      ) \
      .if_("enable_fresh_request_dynamic_config == 1") \
        .split_string(
          input_common_attr="fresh_request_pic_fixed_slot_config",
          output_common_attr="fresh_fixed_slot_conf_list",
          delimiters=";",
        ) \
      .end_() \
      .if_("enable_grpr_supervise_pron_attr == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "grpr_supervise_pron_report_limit", "as": "report_limit"},
            {"name": "grpr_supervise_pron_report_break_point", "as": "report_break_point"},
            {"name": "grpr_supervise_pron_low_report_rate_threshold_1", "as": "low_report_rate_threshold_1"},
            {"name": "grpr_supervise_pron_low_report_rate_threshold_2", "as": "low_report_rate_threshold_2"},
            {"name": "grpr_supervise_pron_report_rate_threshold", "as": "report_rate_threshold"},
          ],
          import_item_attr = [
            "click_count",
            {"name": "report_count", "as": "total_report_count_bkup"},
            {"name": "report_detail__total_report_count", "as": "total_report_count"},
            {"name": "report_detail__low_report_count", "as": "low_report_count"},
          ],
          export_item_attr = [
            "is_grpr_pron_photo"
          ],
          function_name = "IsGrprPronPhoto",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
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
        sequence_num_multiple = "{{explore_rerank_dpp_sequence_num_multiple}}",
        diversity_list_size = "{{dpp_diversity_list_size}}",
        rank_theta = "{{dpp_diversity_rank_theta}}",
        dm_epsilon = "{{dpp_diversity_dm_epsilon}}",
        enable_dpp = "{{enable_dpp_diversity_new}}",
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
        hetu_emsemble_attr = rerank_hetu_ensemble_queues(),
        enable_dpp_diversity_div_que = "{{enable_dpp_diversity_div_que}}",
        user_info_ptr_attr = "user_info_ptr",
        diversity_history_size = "{{dpp_diversity_history_size}}",
        diversity_queue_name = "diversity",
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
        prev_items_from_attr = "hotCoverHistory",
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
        user_hetu_stat_attr = "colossus_hetu_distribution_hetu_stat"
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

    self.flow.if_("explore_enable_sim_matrix_combo == 1")

    self.flow \
      .get_remote_embedding_lite(
        kess_service = "{{explore_emb_kess_name_for_dpp_emb}}",
        shard_num = 4,
        id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
        output_attr_name = "explore_dpp_emb",
        size = 64,
        client_side_shard = True
      ) \
    
    self.flow.else_()

    self.flow.if_("explore_enable_sim_matrix_combo_new == 1")

    self.flow \
      .get_remote_embedding_lite(
        kess_service = "{{explore_emb_kess_name_for_dpp_emb}}",
        shard_num = 8,
        id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
        output_attr_name = "explore_dpp_emb",
        size = 128,
        client_side_shard = True
      ) \
    
    self.flow.end_()

    self.flow.end_()

    self.flow.if_("explore_enable_gen_distill_listwise_model == 1")

    self.flow \
      .explore_rerank_attr(
        user_info_attr = "user_info_ptr"
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          {"name": "hetu_tag_level_info__hetu_level_one", "as": "extract_hetu_tag_list"},
        ],
        export_item_attr = [
          {"name": "first_hetu_tag", "as": "pHetuTagLevel1RerankList"}
        ],
        function_name = "ExtractFirstHetuTag",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          {"name": "hetu_tag_level_info__hetu_level_two", "as": "extract_hetu_tag_list"},
        ],
        export_item_attr = [
          {"name": "first_hetu_tag", "as": "pHetuTagLevel2RerankList"}
        ],
        function_name = "ExtractFirstHetuTag",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          {"name": "hetu_tag_level_info__hetu_level_three", "as": "extract_hetu_tag_list"},
        ],
        export_item_attr = [
          {"name": "first_hetu_tag", "as": "pHetuTagLevel3RerankList"}
        ],
        function_name = "ExtractFirstHetuTag",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .delegate_enrich(
        kess_service = "{{explore_rerank_listwise_distill_kai_kess_service}}",
        recv_item_attrs = [
          {"name": "rerank_distill", "as": "listwise_distill_score"},
        ],
        timeout_ms = 100,
        send_item_attrs = rerank_features_gen,
        send_common_attrs = user_features_full_link(),
        range_end = 60,
        request_type = "default"
      ) \

    self.flow.end_()

    self.flow.if_("explore_enable_mdp_gen_list == 1")

    self.flow \
      .explore_mdp_gen_list_enricher(
        candidate_size = "{{explore_mdp_gen_list_candidate_size}}",
        output_len = "{{explore_mdp_gen_list_output_len}}",
        item_value_attr_name = "explore_fullrank_pure_value_score",
        item_next_attr_name = "corr_pctr",
        output_attr = "retrieval_list_keys_7",
        pnext_alpha = "{{explore_mdp_gen_list_pnext_alpha}}",
        pnext_beta = "{{explore_mdp_gen_list_pnext_beta}}",
        beam_size = "{{explore_mdp_gen_list_beam_size}}",
        value_mean = "{{explore_mdp_gen_list_value_mean}}",
        seq_item_attr_name = "generated_diversity_lists",
      ) \
    
    self.flow.end_()

    self.flow.if_("explore_enable_dpp_gen_list == 1")

    self.flow.if_("skip_dpp_pic_mix_generator == 0")

     # 图文混排 generator 
    self.flow \
      .if_("explore_enable_calc_pic_quota == 1 and explore_enable_insert_pic_after_rerank_dpp == 0 and explore_enable_calc_pic_quota_fixed_load == 0") \
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
          target_item={ "is_picture": 1 }
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
        .if_("explore_pic_quota_enable_recent_realshow_decay == 1") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "expl_pic_recent_realshow_time_gap_min", "as": "recent_time_gap_min"},
              {"name": "uStandardRealShowPicAllIdList", "as": "pic_realshow_pids"},
              {"name": "uStandardClickPicAllIdList", "as": "pic_click_pids"},
              "user_info_ptr",
            ],
            export_common_attr = [
              "pic_recent_realshow_not_click_cnt",
            ],
            function_name = "ProccessPicActionList",
            class_name = "ExploreLightFunctionSetV2",
          ) \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "expl_pic_recent_realshow_not_click_max", "as": "realshow_not_click_max"},
              {"name": "expl_pic_recent_realshow_not_click_min", "as": "realshow_not_click_min"},
              {"name": "expl_pic_recent_realshow_ctr_base", "as": "ctr_base"},
              {"name": "pic_recent_realshow_not_click_cnt", "as": "realshow_not_click_cnt"},
              "pic_da_user_pref_ptr",
              "basic_info_age_segment_v2",
              "uIsPicDeep",
            ],
            export_common_attr = [
              "user_pic_recent_ctr_score",
            ],
            function_name = "PicCtrByRealshow",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .end_() \
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
          target_item = { "is_picture": 1 },
        ) \
      .end_() \
      .if_("explore_enable_insert_pic_after_rerank_dpp == 1 and explore_enable_calc_pic_quota_fixed_load == 0") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "expl_pic_insert_after_dpp_score_method", "as": "score_method"},
            {"name": "expl_pic_insert_after_dpp_video_top_num", "as": "top_num"},
          ],
          import_item_attr = [
            "corr_pctr",
            "pltr",
            "pwtr",
            "pcltr",
          ],
          export_common_attr = [
            {"name": "viceo_topk_avg", "as": "fr_viceo_topk_avg"},
          ],
          function_name = "CalcVideoTopkScoreAvg",
          class_name = "ExploreLightFunctionSetV2",
          target_item = { "is_picture": 0 },
        ) \
        .if_("expl_pic_insert_after_dpp_consider_pic_type == 1") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "pic_fr_rel_score_pct_map", "as": "pic_score_pct_map"},
              {"name": "expl_pic_insert_after_dpp_pct_data_key_single_pic", "as": "pct_data_key"},
              {"name": "expl_pic_insert_after_dpp_thd_pct_milli_single_pic", "as": "thd_pct_milli"},
              {"name": "expl_pic_insert_after_dpp_range_end_single_pic", "as": "range_end"},
              {"name": "expl_pic_insert_after_dpp_score_method", "as": "score_method"},
              {"name": "fr_viceo_topk_avg", "as": "viceo_topk_avg"},
              {"name": "expl_pic_insert_after_dpp_viceo_topk_pow", "as": "viceo_topk_pow"},
              {"name": "expl_pic_insert_after_dpp_reserve_order", "as": "reserve_order"},
              {"name": "expl_single_pic_insert_after_dpp_min_quota", "as": "min_quota"},
              {"name": "expl_single_pic_insert_after_dpp_max_quota", "as": "max_quota"},
            ],
            import_item_attr = [
              "photo_id",
              "corr_pctr",
              "pltr",
              "pwtr",
              "pcltr",
            ],
            export_common_attr = [
              {"name": "pic_pids", "as": "pic_list_to_insert_after_rerank_dpp_single_pic"},
            ],
            function_name = "CalcPicQuotaAndPidsLife",
            class_name = "ExploreLifeLightFunctionSet",
            target_item = { "picture_type": 1 },
          ) \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "pic_fr_rel_score_pct_map", "as": "pic_score_pct_map"},
              {"name": "expl_pic_insert_after_dpp_pct_data_key_long_set_pic", "as": "pct_data_key"},
              {"name": "expl_pic_insert_after_dpp_thd_pct_milli_long_set_pic", "as": "thd_pct_milli"},
              {"name": "expl_pic_insert_after_dpp_range_end_long_set_pic", "as": "range_end"},
              {"name": "expl_pic_insert_after_dpp_score_method", "as": "score_method"},
              {"name": "fr_viceo_topk_avg", "as": "viceo_topk_avg"},
              {"name": "expl_pic_insert_after_dpp_viceo_topk_pow", "as": "viceo_topk_pow"},
              {"name": "expl_pic_insert_after_dpp_reserve_order", "as": "reserve_order"},
              {"name": "expl_long_set_insert_after_dpp_min_quota", "as": "min_quota"},
              {"name": "expl_long_set_insert_after_dpp_max_quota", "as": "max_quota"},
            ],
            import_item_attr = [
              "photo_id",
              "corr_pctr",
              "pltr",
              "pwtr",
              "pcltr",
            ],
            export_common_attr = [
              {"name": "pic_pids", "as": "pic_list_to_insert_after_rerank_dpp_long_set_pic"},
            ],
            function_name = "CalcPicQuotaAndPidsLife",
            class_name = "ExploreLifeLightFunctionSet",
            target_item = { "picture_type": [2, 3] },
          ) \
          .pack_common_attr(
            input_common_attrs = [
              "pic_list_to_insert_after_rerank_dpp_single_pic",
              "pic_list_to_insert_after_rerank_dpp_long_set_pic"
            ],
            output_common_attr = "pic_list_to_insert_after_rerank_dpp",
            limit_num = "{{expl_pic_insert_after_dpp_max_pic_num}}",
            deduplicate = True,
          ) \
        .else_() \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "pic_fr_rel_score_pct_map", "as": "pic_score_pct_map"},
              {"name": "expl_pic_insert_after_dpp_pct_data_key", "as": "pct_data_key"},
              {"name": "expl_pic_insert_after_dpp_thd_pct_milli", "as": "thd_pct_milli"},
              {"name": "expl_pic_insert_after_dpp_range_end", "as": "range_end"},
              {"name": "expl_pic_insert_after_dpp_score_method", "as": "score_method"},
              {"name": "fr_viceo_topk_avg", "as": "viceo_topk_avg"},
              {"name": "expl_pic_insert_after_dpp_viceo_topk_pow", "as": "viceo_topk_pow"},
              {"name": "expl_pic_insert_after_dpp_reserve_order", "as": "reserve_order"},
              {"name": "expl_pic_insert_after_dpp_min_quota", "as": "min_quota"},
            ],
            import_item_attr = [
              "photo_id",
              "corr_pctr",
              "pltr",
              "pwtr",
              "pcltr",
            ],
            export_common_attr = [
              {"name": "pic_pids", "as": "pic_list_to_insert_after_rerank_dpp_all_pic"},
            ],
            function_name = "CalcPicQuotaAndPidsLife",
            class_name = "ExploreLifeLightFunctionSet",
            target_item = { "is_picture": 1 },
          ) \
          .pack_common_attr(
            input_common_attrs = [
              "pic_list_to_insert_after_rerank_dpp_all_pic"
            ],
            output_common_attr = "pic_list_to_insert_after_rerank_dpp",
            limit_num = "{{expl_pic_insert_after_dpp_max_pic_num}}",
            deduplicate = True,
          ) \
        .end_() \
        .set_attr_value(
          common_attrs = [{"name": "dpp_mix_rerank_pic_queue_size", "type": "int", "value": 0}],
        ) \
      .end_() \
      .if_("explore_enable_calc_pic_quota_fixed_load == 1") \
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
        .if_("explore_pic_quota_enable_recent_realshow_decay == 1") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "expl_pic_recent_realshow_time_gap_min", "as": "recent_time_gap_min"},
              {"name": "uStandardRealShowPicAllIdList", "as": "pic_realshow_pids"},
              {"name": "uStandardClickPicAllIdList", "as": "pic_click_pids"},
              "user_info_ptr",
            ],
            export_common_attr = [
              "pic_recent_realshow_not_click_cnt",
            ],
            function_name = "ProccessPicActionList",
            class_name = "ExploreLightFunctionSetV2",
          ) \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "expl_pic_recent_realshow_not_click_max", "as": "realshow_not_click_max"},
              {"name": "expl_pic_recent_realshow_not_click_min", "as": "realshow_not_click_min"},
              {"name": "expl_pic_recent_realshow_ctr_base", "as": "ctr_base"},
              {"name": "pic_recent_realshow_not_click_cnt", "as": "realshow_not_click_cnt"},
              "pic_da_user_pref_ptr",
              "basic_info_age_segment_v2",
              "uIsPicDeep",
            ],
            export_common_attr = [
              "user_pic_recent_ctr_score",
            ],
            function_name = "PicCtrByRealshow",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .end_() \
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
      .end_() \
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
                {"name": "expl_pic_insert_after_dpp_min_quota", "as": "min_quota"},
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
              function_name="CalcPicQuotaAndPidsLife",
              class_name="ExploreLifeLightFunctionSet",
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
                {"name": "expl_pic_insert_after_dpp_min_quota", "as": "min_quota"},
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
              function_name="CalcPicQuotaAndPidsLife",
              class_name="ExploreLifeLightFunctionSet",
              target_item={"is_picture": 1},
            ) \
        .end_() \
      .end_() \
      .if_("enable_life_low_active_adjust_rank_quota == 1 and uIsLifeHighActive ~= 1") \
        .gen_common_attr_by_lua(
          attr_map = {
            "dpp_mix_rerank_video_queue_size": "life_low_active_rank_quota",
          },
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
      ) \
      .if_("dpp_mix_rerank_single_pic_as_video == 1") \
        .copy_item_meta_info(
          save_item_seq_to_attr = "pic_rank",
          target_item = {"picture_type": 1}
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "dpp_mix_rerank_single_pic_max_num", "as": "single_pic_max_num"},
            {"name": "dpp_mix_rerank_single_pic_max_pic_rank", "as": "max_pic_rank"},
            {"name": "dpp_mix_rerank_single_pic_pctr_thd", "as": "pctr_thd"},
          ],
          import_item_attr = [
            "is_picture",
            "picture_type",
            "corr_pctr",
            "pic_rank",
          ],
          export_item_attr = [
            "mix_mark",
          ],
          function_name = "MarkMixItemSinglePic",  # 单独标记单图
          class_name = "ExploreLightFunctionSetV2",
          range_end = "{{dpp_mix_rerank_single_pic_candidate_size}}",
        ) \
      .end_()

    self.enrich_item_attr( # 抽取dpp需要的 item attr
        target_item = {
          "mix_mark" : [1, 2]
        }
      )
    self.flow \
      .if_("skip_dpp_pic_mix_reward == 0")  \
        .explore_calc_ensemble_score(
          save_score_to_attr = "mix_reward",
          user_power_calc = 1,
          queues = generate_mix_queues(fr_pxtrs()) + generate_mix_queues(fr_fountain_pxtrs()),
          target_item = {
            "mix_mark" : [1, 2]
          }
        ) \
      .end_() \
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
      ) \
      .if_("skip_mix_ensemble_discount_picture == 0") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "mix_ensemble_discount_picture_coef", "as": "boost_discount_coeff"},
          ],
          import_item_attr = [
            {"name": "is_picture", "as": "need_item_attr"},
            {"name": "mix_ensemble_score", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "mix_ensemble_score"},
          ],
          function_name = "BoostOrDiscount",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "mix_mark" : [1, 2]
          }
        ) \
      .end_() \
      .if_("enable_rerank_duration_boost == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "rerank_duration_boost_coef_0_7s", "as": "duration_boost_coef_0_7s"},
            {"name": "rerank_duration_boost_coef_7_12s", "as": "duration_boost_coef_7_12s"},
            {"name": "rerank_duration_boost_coef_12_20s", "as": "duration_boost_coef_12_20s"},
            {"name": "rerank_duration_boost_coef_20_58s", "as": "duration_boost_coef_20_58s"},
            {"name": "rerank_duration_boost_coef_58_120s", "as": "duration_boost_coef_58_120s"},
            {"name": "rerank_duration_boost_coef_120_s", "as": "duration_boost_coef_120_s"},
            {"name": "rerank_boost_with_type_watchtime", "as": "boost_with_type_watchtime"},
          ],
          import_item_attr = [
            "duration_ms",
            "empirical_watch_time",
            {"name": "fr_score2", "as": "pred_watch_time"},
            {"name": "mix_ensemble_score", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "mix_ensemble_score"},
          ],
          function_name = "BoostOrDiscountWithDuration",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "mix_mark" : [1, 2]
          }
        ) \
      .end_()
    self.sequence_generator(
        target_item = {
          "mix_mark" : [1, 2]
        }
      )
      
    self.flow.else_()

    # 基线 generator
    self.enrich_item_attr( # 抽取dpp需要的 item attr
        target_item = {}
      )
    self.sequence_generator(
        target_item = {}
      )

    self.flow.end_()

    self.flow.end_()

    self.flow \
      .pack_common_attr(
        input_common_attrs = ["retrieval_list_keys_6", "retrieval_list_keys_7"],
        output_common_attr = 'retrieval_list_keys',
        deduplicate = True
      ) \
      .switch_("enable_use_new_rerank_features") \
        .case_(0) \
          .if_("enable_skip_old_list_fea == 0") \
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
        .case_(1) \
          .if_("enable_skip_old_list_fea == 0") \
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
          .if_("enable_skip_old_list_fea == 0") \
            .list_wise_seq_attr(
              item_attrs_transform_map = rerank_features_new_v2,
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
      .end_()
