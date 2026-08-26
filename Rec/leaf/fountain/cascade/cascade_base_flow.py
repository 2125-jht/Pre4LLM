#!/usr/bin/env python3
# coding=utf-8

from dragonfly.common_leaf_dsl import LeafFlow
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.subdivision.subdivision_api_mixin import subdivisionApiMixin
from dragonfly.ext.explore.explore_api_mixin import ExploreApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from cascade.cascade_utils import cascade_ltr_common_feature, cascade_fc_feature, cascade_fc_sim3_feature, cascade_slide_features, cascade_full_link_distill_user_features
from dump_attr_to_kafka import dump_attr_to_kafka
from cascade.cascade_fast_queues import mc_pxtr_fractile_score_queues

class CascadeBaseFlow(LeafFlow, MioApiMixin, subdivisionApiMixin,ExploreApiMixin, GsuApiMixin):
  def _enrich_cascade_score_fast(self):
    """
    非首屏粗排模型预估值填充和计算
    ------
    - 内部流粗排模型预估
    - 外部流粗排模型预估
    - 粗排 l2r 模型预估
    """
    self \
    ._cascade_model_for_all() \
    ._cascade_fc_model() \
    ._cascade_model_for_fast() \
    ._cascade_model_for_all_exp() \
    ._get_user_feature() \
    .switch_("enable_get_user_group_bucket") \
      .case_(1) \
        .get_user_group_emp_xtr() \
      .case_(2) \
        .get_user_new_group_emp_xtr() \
      .case_(3) \
        .get_user_ten_group_emp_xtr() \
      .case_(4) \
        .get_user_isweekend_timeslot_group_emp_xtr() \
    .end_() \
    ._cascade_score_enrich_lua() \
    ._cascade_produce_solve_score_fast() \
    ._cascade_merchant_solve_score_fast() \
    ._cascade_ftr_debias() \
    ._cascade_life_stage_cid_ipw_debias() \
    ._cascade_age_gender_prof_cid_ipw_debias() \
    ._cascade_age_gender_north_cid_ipw_debias() \
    ._cascade_age_gender_cid_ipw_debias() \
    ._cascade_interact_playtime_adjust() \
    ._cascade_touch_high_follow_adjust() \
    ._cascade_action_once_score() \
    ._cascade_pc_combine_score() \
    .if_('enable_fountain_fc_wtd_inverse == 1')\
      ._cascade_fc_wtd_inverse()\
    .end_()\
    .if_('enable_fountain_calc_cascade_emp_report_rate_score == 1')\
      .cascade_cal_emp_report_rate_score()\
    .end_() \
    .if_('enable_fountain_cal_rise_follow_boost_score_mc_s1 == 1')\
      ._fountain_cal_rise_follow_boost_score_mc_s1()\
    .end_() \
    .if_('enable_fountain_cal_rise_follow_boost_light_score_mc_s1 == 1')\
      ._fountain_cal_rise_follow_boost_light_score_mc_s1()\
    .end_() \
    .if_("enable_fountain_mc_get_xtr_fractile_score == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      ._cascade_get_xtr_fractile_score() \
    .end_() \
    ._cascade_pre_filter() \
    ._cascade_pre_filter_v2()
    return self

  def _enrich_cascade_score_splash(self):
    """
    粗排模型预估值填充和计算，首屏
    ------
    - 内部流粗排模型预估
    - 外部流粗排模型预估
    - 粗排 l2r 模型预估
    """
    self \
    ._cascade_model_for_all() \
    ._cascade_fc_model() \
    ._cascade_model_for_all_exp() \
    ._get_user_feature_splash() \
    .switch_("enable_get_user_group_bucket") \
      .case_(1) \
        .get_user_group_emp_xtr() \
      .case_(2) \
        .get_user_new_group_emp_xtr() \
      .case_(3) \
        .get_user_ten_group_emp_xtr() \
      .case_(4) \
        .get_user_isweekend_timeslot_group_emp_xtr() \
    .end_() \
    .if_('enable_fountain_fc_wtd_inverse == 1')\
      ._cascade_fc_wtd_inverse()\
    .end_()\
    .if_('enable_fountain_calc_cascade_emp_report_rate_score == 1')\
      .cascade_cal_emp_report_rate_score()\
    .end_() \
    ._cascade_score_enrich_lua() \
    ._cascade_ftr_debias() \
    ._cascade_action_once_score() \
    ._cascade_pre_filter() \
    ._cascade_pre_filter_v2()
    return self

  def _cascade_model_for_all(self):
    """
    模型组模型&&l2r,首屏非首屏同时请求
    """
    self \
    .get_item_attr_by_predict_fetcher_v2(
      # 内部流粗排模型预估
      skip = "{{fountain_new_arch_cascade_skip_predict}}",
      kess_service = "{{fountain_cascade_new_arch_predict_kess_service}}",
      service_group = "PRODUCTION",
      timeout_ms = 300,
      user_info_attr = "userInfo",
      output_prefix = "cascade_",
      tower_request_type = "{{fountain_cascade_new_arch_tower_request_type}}",
      pxtr = ["pctr", "pltr", "pwtr", "pftr", "plvtr", "psvtr", "ptr", "pwatch_time", "pepstr", "pcestr", "pcmtr", "pwtd", "pcltr", "phtr", "pcotr"],
    ) \
    .if_("enable_fountain_cascade_comment_model_predict == 1") \
      .cascade_comment_model_predict() \
    .end_()
    return self

  def _cascade_model_for_fast(self):
    """
    只在非首屏请求模型, 仅对电商用户请求粗精排服务
    """
    self \
      .if_("merchant_buyer_type ~= nil and merchant_buyer_type >= fountain_merchant_need_request_buyer_type") \
        .if_("enable_fountain_mc_merchant_vedio_predict == 1") \
          .delegate_enrich( # 挂车短视频粗排PXTR预估
            kess_service = "{{fountain_mc_merchant_photo_double_tower_service}}",
            send_common_attrs = [
              { "name": "kuibaUserAttrStr", "as": "user_info_str" },
            ],
            request_type = "{{fountain_mc_merchant_photo_double_tower_request_type}}",
            timeout_ms = 100,
            infer_output_type = 2,
            recv_item_attrs = [{
              "name": pred,
              "as": "merchant_tower_" + pred
            } for pred in ["ctr", "cvr", "gmv"]],
            target_item = {"is_merchant_cart": 1}
          ) \
        .end_() \
        .if_("enable_fountain_mc_merchant_living_predict == 1") \
          .delegate_enrich( # live头像短视频粗排PXTR预估
            kess_service = "{{fountain_mc_merchant_living_double_tower_service}}",
            send_common_attrs = [
              { "name": "kuibaUserAttrStr", "as": "user_info_str" },
            ],
            request_type = "{{fountain_mc_merchant_living_double_tower_request_type}}",
            timeout_ms = 100,
            infer_output_type = 2,
            recv_item_attrs = [{
              "name": pred,
              "as": "merchant_elive_tower_" + pred
            } for pred in ["ctr", "cvr", "gmv"]],
            target_item={"is_merchant_living": 1}
          ) \
        .end_() \
      .end_() \
      .switch_("fountain_mc_distill_pointwise_model_predict_switch") \
        .case_(2) \
          ._cascade_distill_pointwise_uni_pred() \
        .case_(3) \
          ._cascade_distill_pointwise_reward_ltr_uni_pred() \
        .default_() \
          .if_("fountain_skip_mc_distill_model_predict == 0") \
            .delegate_enrich(
              kess_service = "{{fountain_mc_distill_model_service}}",
              recv_item_attrs = [{"name": "fast_rank", "as": "cascade_distill_fast_rank"}],
              timeout_ms = 100,
              send_item_attrs = ["item_id"],
              send_common_attrs = cascade_ltr_common_feature,
              request_type = "default",
            ) \
          .end_if_() \
      .end_() \
      .if_("enable_fountain_cascade_produce_predict_all == 1") \
        .if_("enable_fountain_cascade_produce_photo_predict == 1 and fountain_cascade_need_produce_model_flag > 0") \
          .delegate_enrich( # enable_fountain_cascade_produce_predict_all 做小流量反转使用
            kess_service = "{{fountain_cascading_produce_photo_double_tower_service}}",
            send_common_attrs = [
              { "name": "kuibaUserAttrStr", "as": "user_info_str" },
            ],
            recv_item_attrs = [{
              "name": pred,
              "as": "fountain_produce_cascade_" + pred
            } for pred in ["mtctr", "twhtr", "mtcotr", "mtjtr", "kym", "csti", "sjctr"]],
            timeout_ms = 100,
            request_type = "{{fountain_cascading_produce_photo_double_tower_request_type}}",
            partition_size = "{{fountain_cascading_produce_photo_double_tower_partition_size}}",
            use_item_id_in_attr = "item_id",
            use_packed_item_attr = True,
            infer_output_type = 2
          ) \
        .end_() \
        .if_("enable_fountain_cascade_produce_photo_predict == 2 and fountain_produce_consume_deep_user == 0 and fountain_produce_user_type > fountain_cascade_produce_user_switch") \
          .delegate_enrich( # 因为模型infer用到的info不同，逻辑额外起一套，切换后删除老逻辑
            kess_service = "{{fountain_cascading_produce_photo_double_tower_new_service}}",
            send_common_attrs = [
              { "name": "userInfo", "as": "user_info_str" },
            ],
            recv_item_attrs = [{
              "name": pred,
              "as": "fountain_produce_cascade_" + pred
            } for pred in ["mtctr", "twhtr", "mtcotr", "mtjtr", "kym", "csti", "sjctr"]],
            timeout_ms = 100,
            request_type = "{{fountain_cascading_produce_photo_double_tower_new_request_type}}",
            partition_size = "{{fountain_cascading_produce_photo_double_tower_new_partition_size}}",
            use_item_id_in_attr = "item_id",
            use_packed_item_attr = True,
            infer_output_type = 2
          ) \
        .end_() \
      .end_() \
      .if_("fountain_cascade_enable_slide_comment_model_predict == 1") \
        ._cascade_slide_comment_model_predict() \
      .end_() \
      .if_("enable_fountain_cascade_batch_similar_model_predict == 1") \
        .cascade_batch_similar_model_predict() \
      .end_()
    return self

  def _cascade_distill_pointwise_uni_pred(self):
    self \
      .explore_custom_trim_user_info(
        user_info_attr = "userInfo",
        save_trimed_user_info_to_attr = "mc_distill_trimmed_user_info",
        trim_user_info = [
          "device_id",
          "basic_info.age_segment",
          "gender",
          "infer_gender",
          "true_gender",
          "location.city_id",
          "visit_mod",
          "realtime_click_list",
          "realtime_follow_list",
          "realtime_like_list",
          "follow_count",
          "upload_count",
          "request_location.province_id",
          "request_location.city_id",
        ],
      ) \
      .delegate_enrich(
        kess_service = "{{fountain_mc_distill_model_service}}",
        recv_item_attrs = [
          {"name": "distill_fr", "as": "cascade_distill_fast_rank"},
          {"name": "distill_rerank", "as": "cascade_distill_rerank"},
          {"name": "distill_show", "as": "cascade_distill_show"},
        ],
        timeout_ms = 100,
        send_common_attrs = [
          { "name": "mc_distill_trimmed_user_info", "as": "user_info_str" },
        ],
        request_type = "default",
      )
    return self

  def _cascade_distill_pointwise_reward_ltr_uni_pred(self):
    self \
      .explore_custom_trim_user_info(
        user_info_attr = "userInfo",
        save_trimed_user_info_to_attr = "mc_distill_trimmed_user_info",
        trim_user_info = [
          "device_id",
          "basic_info.age_segment",
          "gender",
          "infer_gender",
          "true_gender",
          "location.city_id",
          "visit_mod",
          "follow_count",
          "upload_count",
          "request_location.province_id",
          "request_location.city_id",
          "user_profile_v1.video_playing_stat.playing_time",
          "user_profile_v1.video_playing_stat.author_id",
          "user_profile_v1.video_playing_stat.photo_id",
          "user_profile_v1.follow_list.author_id",
          "user_profile_v1.follow_list.photo_id",
          "user_profile_v1.like_list.author_id",
          "user_profile_v1.like_list.photo_id",
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
          "fountain_reco_user_profile.video_play_stat.playing_time"
        ],
      ) \
      .delegate_enrich(
        name = "fountain_mc_distill_model",
        kess_service = "{{fountain_mc_distill_model_service}}",
        recv_item_attrs = [
          {"name": "distill_fr", "as": "cascade_distill_fast_rank"},
          {"name": "distill_show", "as": "cascade_distill_show"},
          {"name": "fl_realshow_reward", "as": "cascade_fl_realshow_reward"},
          {"name": "distill_reward", "as": "cascade_distill_reward"},
        ],
        timeout_ms = 100,
        send_common_attrs = [
          { "name": "mc_distill_trimmed_user_info", "as": "user_info_str" },
        ],
        request_type = "default",
      )
    return self

  def _cascade_fc_model(self):
    self \
    .if_("enable_fc_fountain_interface == 1") \
      .if_("enable_fc_feature_kconf == 1") \
        .extract_with_ks_sign_feature(
          extractor_kconf_path = "reco.hot.fountainLeafMcFeature",
          caller_model = "{{fountain_cascade_fc_predict_service}}",
          feature_list = cascade_fc_sim3_feature,
          update_ks_sign_feature_type = 1,
          update_interval_sec = 600,
          user_info_attr = "userInfoPb",
          common_slots_output = "user_feature_slots",
          common_parameters_output = "user_feature_signs",
        ) \
      .end_if_()\
      .if_("enable_fountain_fc_extract_photo_signs == 1") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            "photo_id", "author__id", "tag", "duration_ms", "upload_time"
          ],
          export_item_attr = [{"name":"context_slots", "as":"fountain_fc_car_slots"},
                              {"name":"context_signs", "as":"fountain_fc_car_signs"}],
          function_name = "GenCARSigns",
          class_name = "ExploreLightFunctionSetV2",
        )\
        .delegate_enrich(
          name = "fountain_mc_fc_predict",
          kess_service = "{{fountain_cascade_fc_predict_service}}",
          request_type = "{{fountain_cascade_fc_request_type}}",
          timeout_ms = 100,
          send_common_attrs = ["user_feature_slots", "user_feature_signs"],
          send_item_attrs = ["fountain_fc_car_slots", "fountain_fc_car_signs"],   
          recv_item_attrs = [
            {"name":"fc_pctr_value", "as":"cascade_fc_pctr"},
            {"name":"fc_plvr_value", "as":"cascade_fc_plvtr"},
            {"name":"fc_psvr_value", "as":"cascade_fc_psvtr"},
            {"name":"fc_pvtr_value", "as":"cascade_fc_pvtr"},
            {"name":"fc_pvtr2_value", "as":"cascade_fc_pvtr2"},
            {"name":"fc_pwtd2_value", "as": "cascade_fc_pwtd2"},
            {"name":"fc_ltr_value", "as": "cascade_fc_pltr"},
            {"name":"fc_wtr_value", "as": "cascade_fc_pwtr"},
            {"name":"fc_ftr_value", "as": "cascade_fc_pftr"},
            {"name":"fc_cmtr_value", "as": "cascade_fc_pcmtr"}
          ],
          use_item_id_in_attr = "item_id",
          use_packed_item_attr = True,
        ) \
      .end_if_()\
    .end_if_()
    return self

  def _cascade_model_for_all_exp(self):
    """
    首屏非首屏都会请求的模型,实验中，是否首屏请求通过开关控制
    推全后,根据是否有首屏放入对应的processor中
    """
    self \
    .enrich_attr_by_lua(
      import_common_attr = [
        "common_request_type",
      ],
      export_common_attr = [
        "fountain_casade_is_fast",
      ],
      function_for_common = "cascade_control_model",
      lua_script_file = "fountain/cascade/lua/cascade_control.lua",
    ) \
    .gen_common_attr_by_lua(
        attr_map = {
          "featureFountainIsFirstPage": "1 - morePage",
        },
    ) \
    .switch_("fountain_cascade_tower_model_predict_switch") \
      .case_(1) \
        .explore_custom_trim_user_info(
          user_info_attr = "userInfo",
          save_trimed_user_info_to_attr = "mc_tower_model_trimmed_user_info",
          trim_user_info = [
            "device_id",
            "gender",
            "infer_gender",
            "true_gender",
            "basic_info.age_segment",
            "location.city_id",
            "visit_mod",
            "visit_net",
            "user_profile.user_level",
            "feature_collection.explore_low_active_level",
            "upload_count",
            "follow_count",
            "fans_count",
            "user_profile_v1.real_show_list.photo_id",
            "user_profile_v1.real_show_list.time_ms",
            "user_profile_v1.real_show_list.page_type",
            "user_profile_v1.real_show_list.label.click",
            "user_profile_v1.real_show_list.label.like",
            "user_profile_v1.real_show_list.label.follow",
            "user_profile.exp_stat.exp_click",
            "user_profile.exp_stat.exp_like",
            "user_profile.exp_stat.exp_follow",
            "user_profile.exp_stat.exp_realshow",
            "user_profile.exp_stat.exp_long_view",
            "request_location.poi_type",
            "request_location.province_id",
            "request_location.city_id",
            "location.region_type",
            "user_interest_profile.hetu_level_one_long_term_id",
            "user_interest_profile.hetu_level_one_long_term_score",
            "user_interest_profile.hetu_level_two_long_term_id",
            "user_interest_profile.hetu_level_two_long_term_score",
            "user_interest_profile.hetu_level_three_long_term_id",
            "user_interest_profile.hetu_level_three_long_term_score",
            "user_profile_v1.click_list.author_id",
            "user_profile_v1.click_list.photo_id",
            "user_profile_v1.click_list.hetu_tag_level_info.hetu_level_one",
            "user_profile_v1.follow_list.author_id",
            "user_profile_v1.follow_list.photo_id",
            "user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_one",
            "user_profile_v1.like_list.author_id",
            "user_profile_v1.like_list.photo_id",
            "user_profile_v1.like_list.hetu_tag_level_info.hetu_level_one",
            "user_profile_v1.video_playing_stat.playing_time",
            "user_profile_v1.video_playing_stat.author_id",
            "user_profile_v1.video_playing_stat.photo_id",
            "fountain_reco_user_profile.follow_list.author_id",
            "fountain_reco_user_profile.follow_list.photo_id",
            "fountain_reco_user_profile.like_list.author_id",
            "fountain_reco_user_profile.like_list.photo_id",
            "fountain_reco_user_profile.video_play_stat.photo_id",
            "fountain_reco_user_profile.video_play_stat.author_id",
            "fountain_reco_user_profile.video_play_stat.video_duration",
            "fountain_reco_user_profile.video_play_stat.playing_time",
          ],
        ) \
        .delegate_enrich(
          name = "fountain_mc_integrated_tower_predict",
          kess_service = "{{fountain_mc_integrated_tower_predict_kess_service}}",
          recv_item_attrs = [
            {"name":"wtd", "as":"cascade_wtd_kai"},
            {"name":"act", "as":"cascade_act_kai"},
            {"name":"click_comment_button", "as":"cascade_click_comment_button"},
            {"name":"slide", "as":"cascade_slide_kai"},
            {"name":"finish_rate", "as":"cascade_ftr_kai"},
            {"name":"prob_view", "as":"cascade_prob_view"},
          ],
          timeout_ms = 100,
          send_common_attrs = [
            { "name": "mc_tower_model_trimmed_user_info", "as": "user_info_str" },
            { "name": "featureSourcePId", "as": "source_pid" },
            { "name": "sourcePidDuration", "as": "source_duration_ms" },
            { "name": "sourcePidTagId", "as": "source_tag" },
            { "name": "sourcePidAuthorId", "as": "source_aid" },
            { "name": "sourcePidHetuLevelOneList", "as": "source_hetu_tag_level1_list" },
            { "name": "sourcePidHetuLevelTwoList", "as": "source_hetu_tag_level2_list" },
          ],
          request_type = "default",
        ) \
      .default_() \
        .if_("skip_fountain_cascade_wtd_act_kai_predict == 0 and skip_fountain_cascade_wtd_kai_predict == 1") \
          .if_("fountain_cascade_wtd_act_kai_predict_all == 1 or fountain_casade_is_fast == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
            .delegate_enrich(
              kess_service = "{{fountain_cascade_wtd_act_predict_kai_kess_service}}",
              recv_item_attrs = [
                {"name":"wtd", "as":"cascade_wtd_kai"},
                {"name":"act", "as":"cascade_act_kai"},
                {"name":"wtd_percent_10", "as":"cascade_wtd_10p"},
                {"name":"wtd_percent_20", "as":"cascade_wtd_20p"},
                {"name":"wtd_percent_30", "as":"cascade_wtd_30p"},
                {"name":"wtd_percent_40", "as":"cascade_wtd_40p"},
                {"name":"wtd_percent_50", "as":"cascade_wtd_50p"},
                {"name":"wtd_percent_60", "as":"cascade_wtd_60p"},
                {"name":"wtd_percent_70", "as":"cascade_wtd_70p"},
                {"name":"wtd_percent_80", "as":"cascade_wtd_80p"},
                {"name":"wtd_percent_90", "as":"cascade_wtd_90p"},
                ],
              timeout_ms = 100,
              send_item_attrs = ["item_id"],
              send_common_attrs = cascade_ltr_common_feature,
              request_type = "default",
            ) \
          .end_if_() \
        .end_if_() \
    .end_() \

    return self

  def _cascade_pre_filter(self):
    """
    粗排模型负向过滤
    ------
    - phtr
    - psvtr
    """
    self \
    .filter_by_attr(
      attr_name = "cascade_phtr",
      remove_if = ">",
      compare_to = "{{fountain_cascade_phtr_filter_limit}}",
      remove_if_attr_missing = False,
      skip = "{{skip_fountain_cascade_phtr_filter}}")
    return self

  def _cascade_pre_filter_v2(self):
    """
    粗排模型负向过滤阈值调整
    ------
    - phtr
    """
    self \
    .if_("skip_fountain_cascade_phtr_filter_v2 == 0") \
      .enrich_with_protobuf(
        from_extra_var = "userInfoPb",
        attrs = [
          dict(name="hate_ts_list", path="fountain_reco_user_profile.hate_list.time_ms"),
        ]
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "hate_ts_list",
          {"name": "ft_base_phtr_filter_threshold", "as": "base_phtr_filter_threshold"},
          {"name": "ft_min_phtr_filter_threshold", "as": "min_phtr_filter_threshold"},
          {"name": "ft_recent_minute_for_high_freq_hate", "as": "recent_minute_for_high_freq_hate"},
          {"name": "ft_phtr_thrshold_temperature", "as": "phtr_thrshold_temperature"},
          {"name": "ft_phtr_thrshold_smooth", "as": "phtr_thrshold_smooth"},
          {"name": "ft_mc_htr_filter_ltr_threshold", "as": "mc_htr_filter_ltr_threshold"},
          {"name": "ft_mc_htr_filter_wtr_threshold", "as": "mc_htr_filter_wtr_threshold"},
        ],
        import_item_attr = [
          "cascade_phtr",
          {"name": "cascade_pltr", "as": "mc_ensemble_pltr"},
          {"name": "cascade_pwtr", "as": "mc_ensemble_pwtr"},
         ],
        export_item_attr = [
          "mc_need_htr_filter",
        ],
        function_name = "NeedHtrFilter",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .filter_by_attr(
        attr_name = "mc_need_htr_filter",
        remove_if = "==",
        compare_to = 1,
      ) \
    .end_()

    return self

  def _get_wtd_table(self):
    self \
    .get_kconf_params(
      skip = "{{skip_fountain_cascade_get_wtd_table}}",
      kconf_configs = [
      {
        "kconf_key": "{{cascade_wtd_score_table_kconf}}",
        "value_type": "json",
        "json_path": "durationSeg",
        "export_common_attr": "cascade_wtd_table_seg"
      },
      {
        "kconf_key": "{{cascade_wtd_score_table_kconf}}",
        "value_type": "json",
        "json_path": "duration_id_0",
        "export_common_attr": "cascade_wtd_table_0"
      },
      {
        "kconf_key": "{{cascade_wtd_score_table_kconf}}",
        "value_type": "json",
        "json_path": "duration_id_1",
        "export_common_attr": "cascade_wtd_table_1"
      },
      {
        "kconf_key": "{{cascade_wtd_score_table_kconf}}",
        "value_type": "json",
        "json_path": "duration_id_2",
        "export_common_attr": "cascade_wtd_table_2"
      },
      {
        "kconf_key": "{{cascade_wtd_score_table_kconf}}",
        "value_type": "json",
        "json_path": "duration_id_3",
        "export_common_attr": "cascade_wtd_table_3"
      },
      {
        "kconf_key": "{{cascade_wtd_score_table_kconf}}",
        "value_type": "json",
        "json_path": "duration_id_4",
        "export_common_attr": "cascade_wtd_table_4"
      },
      {
        "kconf_key": "{{cascade_wtd_score_table_kconf}}",
        "value_type": "json",
        "json_path": "duration_id_5",
        "export_common_attr": "cascade_wtd_table_5"
      },
      {
        "kconf_key": "{{cascade_wtd_score_table_kconf}}",
        "value_type": "json",
        "json_path": "duration_id_6",
        "export_common_attr": "cascade_wtd_table_6"
      },
      {
        "kconf_key": "{{cascade_wtd_score_table_kconf}}",
        "value_type": "json",
        "json_path": "duration_id_7",
        "export_common_attr": "cascade_wtd_table_7"
      },
      {
        "kconf_key": "{{cascade_wtd_score_table_kconf}}",
        "value_type": "json",
        "json_path": "duration_id_8",
        "export_common_attr": "cascade_wtd_table_8"
      },
      {
        "kconf_key": "{{cascade_wtd_score_table_kconf}}",
        "value_type": "json",
        "json_path": "duration_id_9",
        "export_common_attr": "cascade_wtd_table_9"
      },
      ]
    )
    return self


  def get_wtd_mix_score(self):
    self \
      .get_wtd_score() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "cascade_wtd_table_seg",
          "cascade_wtd_table_0",
          "cascade_wtd_table_1",
          "cascade_wtd_table_2",
          "cascade_wtd_table_3",
          "cascade_wtd_table_4",
          "cascade_wtd_table_5",
          "cascade_wtd_table_6",
          "cascade_wtd_table_7",
          "cascade_wtd_table_8",
          "cascade_wtd_table_9",
          "cascade_wtd_duration_mix_threshold",
          "cascade_wtd_duration_mix_max_value"
        ],
        import_item_attr = [
          "cascade_wtd_kai",
          "duration_ms",
          "cascade_wtd_10p",
          "cascade_wtd_20p",
          "cascade_wtd_30p",
          "cascade_wtd_40p",
          "cascade_wtd_50p",
          "cascade_wtd_60p",
          "cascade_wtd_70p",
          "cascade_wtd_80p",
          "cascade_wtd_90p",
        ],
        export_item_attr = [
          "cascade_wtd_percent",
          "cascade_wtd_duration_mix",
        ],
        function_name = "GetMcWtdMix",
        class_name = "ExploreLightFunctionSetV2",
      )
    return self

  def get_wtd_score(self):
    self \
      .if_("fountain_mc_enable_calc_distill_fr_wtd_score == 1", to_be_delete = "date=2024-05-29;committer=gengxiao03") \
        .enrich_attr_by_light_function(  # duration 和 playtime 阈值的配置, 在 kconf 中配置, 需要保证有序
          import_common_attr = [
            "cascade_wtd_table_seg",
            "cascade_wtd_table_0",
            "cascade_wtd_table_1",
            "cascade_wtd_table_2",
            "cascade_wtd_table_3",
            "cascade_wtd_table_4",
            "cascade_wtd_table_5",
            "cascade_wtd_table_6",
            "cascade_wtd_table_7",
            "cascade_wtd_table_8",
            "cascade_wtd_table_9",
            {"name" : "fountain_mc_wtd_fintr_fintr_low_bound", "as" : "fintr_low_bound"},
            {"name" : "fountain_mc_wtd_fintr_fintr_upper_bound", "as" : "fintr_upper_bound"},
            {"name" : "fountain_mc_wtd_fintr_fintr_power", "as" : "fintr_power"},
            {"name" : "fountain_mc_calc_distill_fr_wtd_score_enable_linear_transform", "as" : "enable_linear_transform"}
          ],
          import_item_attr = [
            "duration_ms",
            "cascade_wtd_kai"
          ],
          export_item_attr = [
            "cascade_wtd_kai_mix",
            "cascade_wtd_fintr"
          ],
          function_name = "GetMcDistillFrWtdScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .else_() \
        .enrich_attr_by_light_function(  # duration 和 playtime 阈值的配置, 在 kconf 中配置, 需要保证有序
          import_common_attr = [
            "cascade_wtd_table_seg",
            "cascade_wtd_table_0",
            "cascade_wtd_table_1",
            "cascade_wtd_table_2",
            "cascade_wtd_table_3",
            "cascade_wtd_table_4",
            "cascade_wtd_table_5",
            "cascade_wtd_table_6",
            "cascade_wtd_table_7",
            "cascade_wtd_table_8",
            "cascade_wtd_table_9",
            {"name" : "fountain_mc_wtd_fintr_fintr_low_bound", "as" : "fintr_low_bound"},
            {"name" : "fountain_mc_wtd_fintr_fintr_upper_bound", "as" : "fintr_upper_bound"},
            {"name" : "fountain_mc_wtd_fintr_fintr_power", "as" : "fintr_power"}
          ],
          import_item_attr = [
            "duration_ms",
            "cascade_wtd_kai"
          ],
          export_item_attr = [
            "cascade_wtd_kai_mix",
            "cascade_wtd_fintr"
          ],
          function_name = "GetMcWtdScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \

    return self

  def _cascade_score_enrich_lua(self):
    """
    模型分计算lua
    """
    self \
    .if_("enable_fountain_mc_prerank_score_calc==1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .enrich_attr_by_lua(
        import_common_attr=[
          "fountain_prerank_weights",
        ],
        export_common_attr=[
          "fountain_prerank_pxtr_weights",
        ],
        function_for_common="parse_prerank_weights",
        lua_script_file="fountain/cascade/lua/calc_pxtr.lua"
      ) \
      .enrich_attr_by_lua(
        import_common_attr=[
          "fountain_prerank_pxtr_weights",
        ],
        import_item_attr=[
          "duration_ms",
          "cascade_pctr",
          "cascade_pltr",
          "cascade_pwtr",
          "cascade_pftr",
          "cascade_plvtr",
          "cascade_psvtr",
          "cascade_ptr",
          "cascade_pwatch_time",
          "cascade_pepstr",
          "cascade_pcestr",
          "cascade_pcmtr",
          "cascade_pwtd",
          "cascade_pcltr",
          "cascade_phtr",
          "cascade_pcotr",
        ],
        export_item_attr=["prerank_score"],
        function_for_item="cal_prerank_score",
        lua_script_file="fountain/cascade/lua/calc_pxtr.lua"
      ) \
    .end_() \
    .enrich_attr_by_lua(
      import_common_attr = [
        "fountain_cascade_lvtr_sigmoid_bias_double",
      ],
      export_common_attr = [
        "fountain_cascade_lvtr_sigmoid_bias",
      ],
      function_for_common = "cascade_lvtr_sigmoid_bias_fix_common",
      lua_script_file = "fountain/cascade/lua/calc_pxtr.lua",
    ) \
    .if_("enable_fc_fountain_interface == 1") \
      .if_("enable_fc_replace_fountain_interface == 1") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            "cascade_fc_pctr", "cascade_fc_plvtr", "cascade_fc_psvtr", "cascade_fc_pvtr",
          ],
          export_item_attr = ["cascade_pctr", "cascade_plvtr", "cascade_psvtr", "cascade_pwatch_time"],
          function_name = "ReplaceMcPxtr",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_fc_replace_fountain_interface_interact == 1") \
        .copy_attr(
          attrs=[
            {
              "from_item": "cascade_fc_pltr",
              "to_item": "cascade_pltr"
            },
            {
              "from_item": "cascade_fc_pwtr",
              "to_item": "cascade_pwtr"
            },
            {
              "from_item": "cascade_fc_pftr",
              "to_item": "cascade_pftr"
            },
          ]
        ) \
      .end_() \
      .if_("enable_fc_replace_fountain_interface_comment == 1") \
        .copy_attr(
          attrs=[
            {
              "from_item": "cascade_fc_pcmtr",
              "to_item": "cascade_pcmtr"
            },
          ]
        ) \
      .end_() \
    .end_() \
    .transform_item_attr(
      mappings = [{
        "check_attr_name": "upload_type",
        "check_attr_type": "int",
        "output_attr_name": "picture_variant_attr",
        "output_attr_type": "int",
        "rules": [{
          "check_values": [ 10, 11],
          "output_value": 1
        }]
      }]) \
    .if_("enable_fountain_cascade_pslide_multiply_prob_view == 1", to_be_delete = "date=2024-05-29;committer=gengxiao03") \
      .enrich_attr_by_light_function(
        import_item_attr = [
          {"name": "cascade_prob_view", "as": "boost_discount_coeff"},
          {"name": "cascade_slide_kai", "as": "score"},
        ],
        export_item_attr = [
          {"name": "score", "as": "cascade_slide_kai"},
        ],
        function_name = "BoostOrDiscountByItemCoeff",
        class_name = "ExploreLightFunctionSetV2"
      ) \
    .end_if_() \
    .if_("enable_enrich_cascade_score_in_base == 1") \
      .fountain_enrich_cascade_score(
        pwatch_time_attr = "cascade_pwatch_time",
        pptr_attr = "cascade_ptr",
        pepstr_attr = "cascade_pepstr",
        pcestr_attr = "cascade_pcestr",
        pcmtr_attr = "cascade_pcmtr",
        pwtd_attr = "cascade_pwtd",
        pslide_attr = "cascade_slide_kai",
        svtr_coeff = "{{fountain_cascade_svtr_coeff}}",
        svtr_power = "{{fountain_cascade_svtr_power}}",
        short_play_discount_value = "{{fountain_cascade_short_play_discount_value}}",
        lvtr_use_predict_watch_time = "{{fountain_cascade_ensemble_lvtr_use_predict_watch_time}}",
        mid_photo_boost_coeff = "{{fountain_cascade_mid_photo_boost_coeff}}",
      ) \
    .end_() \
    .copy_user_meta_info(
      save_request_type_to_attr="common_request_type",
    ) \
    ._get_wtd_table() \
    .if_("skip_cascade_wtd_mix_score_calc == 0 and fountain_casade_is_fast == 1") \
      .get_wtd_mix_score() \
    .end_() \
    .enrich_attr_by_light_function(
      skip = "{{fountain_skip_calc_cascade_questionnaire_score}}",
      import_common_attr = [
        "fountain_questionnaire_score_min_total_count",
        "fountain_questionnaire_score_pos_threshold",
        "fountain_questionnaire_score_neg_threshold",
        "fountain_questionnaire_score_unsure_threshold",
        "fountain_questionnaire_score_use_global",
        {"name": "ft_questionnaire_score_enable_topk_or_audit_valid", "as": "questionnaire_score_enable_topk_or_audit_valid"},
        {"name": "ft_questionnaire_score_topk_level_threshold", "as": "questionnaire_score_topk_level_threshold"},
        {"name": "ft_questionnaire_score_audit_level_threshold", "as": "questionnaire_score_audit_level_threshold"},
      ],
      import_item_attr = [
        "questionnaire_info__positive_count",
        "questionnaire_info__negative_count",
        "questionnaire_info__unsure_count",
        "explore_questionnaire_info__negative_count",
        "explore_questionnaire_info__positive_count",
        "explore_questionnaire_info__unsure_count",
        "topk_audit_level",
        "audit_hot_high_tag_level",
      ],
      export_item_attr = [
        "questionnaire_score"
      ],
      function_name = "CalcQuestionnaireScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_lua(
      skip = "{{fountain_skip_calc_cascade_phtr_discount_score}}",
      import_common_attr = [
        "fountain_cascade_phtr_discount_score_pow_weight",
        "fountain_cascade_phtr_discount_score_weight",
      ],
      import_item_attr = [
        "cascade_score",
        "cascade_phtr",
      ],
      export_item_attr = [
        "cascade_score"
      ],
      function_for_item = "calc_cascade_phtr_discount_score",
      lua_script_file = "fountain/cascade/lua/calc_pxtr.lua",
    )
    return self

  def _cascade_merchant_solve_score_fast(self):
    self.enrich_attr_by_light_function(
      skip = "{{return enable_fountain_mc_merchant_vedio_predict == 0}}",
      import_item_attr = [
        { "name": "merchant_tower_ctr", "as": "ctr_input" },
        { "name": "merchant_tower_cvr", "as": "cvr_input" },
        { "name": "merchant_tower_gmv", "as": "gmv_input" }
      ],
      export_item_attr = [
        { "name": "ctcvr_out", "as": "merchant_mc_photo_ctcvr_score" },
      ],
      function_name = "CalMcMerchantFountainCtcvr",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      skip = "{{return enable_fountain_mc_merchant_living_predict == 0}}",
      import_item_attr = [
        { "name": "merchant_elive_tower_ctr", "as": "ctr_input" },
        { "name": "merchant_elive_tower_cvr", "as": "cvr_input" },
        { "name": "merchant_elive_tower_gmv", "as": "gmv_input" }
      ],
      export_item_attr = [
        { "name": "ctcvr_out", "as": "merchant_mc_living_ctcvr_score" },
        { "name": "ctcvr_gmv_out", "as": "merchant_mc_living_ctcvr_gmv_score" },
      ],
      function_name = "CalMcMerchantFountainCtcvr",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def _cascade_produce_solve_score_fast(self):
    self \
      .if_("enable_fountain_cascade_produce_photo_predict == 2 and fountain_produce_consume_deep_user == 0 and fountain_produce_user_type > fountain_cascade_produce_user_switch") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            { "name": "fountain_produce_cascade_mtctr", "as": "produce_cascade_mtctr" },
            { "name": "fountain_produce_cascade_twhtr", "as": "produce_cascade_twhtr" },
            { "name": "fountain_produce_cascade_mtcotr", "as": "produce_cascade_mtcotr" },
            { "name": "fountain_produce_cascade_mtjtr", "as": "produce_cascade_mtjtr" },
            { "name": "fountain_produce_cascade_kym", "as": "produce_cascade_kym" },
            { "name": "fountain_produce_cascade_csti", "as": "produce_cascade_csti" },
            { "name": "fountain_produce_cascade_sjctr", "as": "produce_cascade_sjctr" },
            { "name": "cascade_pctr", "as": "produce_cascade_pctr" },
            { "name": "cascade_pwatch_time", "as": "produce_cascade_pwatch_time" },
            { "name": "cascade_longview_score", "as": "produce_cascade_longview" },
            { "name": "csm_to_crt_new_upload_photo_cnt", "as": "produce_uploads_cnt" }
          ],
          import_common_attr = [
            { "name": "fountain_produce_user_type", "as": "produce_user_type" },
            { "name": "fountain_cascade_new_user_mtctr_wgt", "as": "new_user_mtctr_wgt"},
            { "name": "fountain_cascade_new_user_twhtr_wgt", "as": "new_user_twhtr_wgt"},
            { "name": "fountain_cascade_new_user_mtcotr_wgt", "as": "new_user_mtcotr_wgt"},
            { "name": "fountain_cascade_new_user_mtjtr_wgt", "as": "new_user_mtjtr_wgt"},
            { "name": "fountain_cascade_new_user_kym_wgt", "as": "new_user_kym_wgt"},
            { "name": "fountain_cascade_new_user_csti_wgt", "as": "new_user_csti_wgt"},
            { "name": "fountain_cascade_new_user_sjctr_wgt", "as": "new_user_sjctr_wgt"},
            { "name": "fountain_cascade_month_user_mtctr_wgt", "as": "month_user_mtctr_wgt"},
            { "name": "fountain_cascade_month_user_twhtr_wgt", "as": "month_user_twhtr_wgt"},
            { "name": "fountain_cascade_month_user_mtcotr_wgt", "as": "month_user_mtcotr_wgt"},
            { "name": "fountain_cascade_month_user_mtjtr_wgt", "as": "month_user_mtjtr_wgt"},
            { "name": "fountain_cascade_month_user_kym_wgt", "as": "month_user_kym_wgt"},
            { "name": "fountain_cascade_month_user_csti_wgt", "as": "month_user_csti_wgt"},
            { "name": "fountain_cascade_month_user_sjctr_wgt", "as": "month_user_sjctr_wgt"},
            { "name": "fountain_cascade_weeks_user_mtctr_wgt", "as": "weeks_user_mtctr_wgt"},
            { "name": "fountain_cascade_weeks_user_twhtr_wgt", "as": "weeks_user_twhtr_wgt"},
            { "name": "fountain_cascade_weeks_user_mtcotr_wgt", "as": "weeks_user_mtcotr_wgt"},
            { "name": "fountain_cascade_weeks_user_mtjtr_wgt", "as": "weeks_user_mtjtr_wgt"},
            { "name": "fountain_cascade_weeks_user_kym_wgt", "as": "weeks_user_kym_wgt"},
            { "name": "fountain_cascade_weeks_user_csti_wgt", "as": "weeks_user_csti_wgt"},
            { "name": "fountain_cascade_weeks_user_sjctr_wgt", "as": "weeks_user_sjctr_wgt"},
            { "name": "fountain_cascade_week_user_mtctr_wgt", "as": "week_user_mtctr_wgt"},
            { "name": "fountain_cascade_week_user_twhtr_wgt", "as": "week_user_twhtr_wgt"},
            { "name": "fountain_cascade_week_user_mtcotr_wgt", "as": "week_user_mtcotr_wgt"},
            { "name": "fountain_cascade_week_user_mtjtr_wgt", "as": "week_user_mtjtr_wgt"},
            { "name": "fountain_cascade_week_user_kym_wgt", "as": "week_user_kym_wgt"},
            { "name": "fountain_cascade_week_user_csti_wgt", "as": "week_user_csti_wgt"},
            { "name": "fountain_cascade_week_user_sjctr_wgt", "as": "week_user_sjctr_wgt"},
            { "name": "fountain_cascade_mtcotr_wgt", "as": "cascade_mtcotr_wgt"},
            { "name": "fountain_cascade_sjctr_wgt", "as": "cascade_sjctr_wgt"},
            { "name": "fountain_cascade_pctr_wgt", "as": "cascade_pctr_wgt"},
            { "name": "fountain_cascade_pwatch_time_wgt", "as": "cascade_pwatch_time_wgt"},
            { "name": "fountain_cascade_longview_wgt", "as": "cascade_longview_wgt"},
            { "name": "fountain_cascade_new_user_qua_score", "as": "new_user_qua_score"},
            { "name": "fountain_cascade_month_user_qua_score", "as": "month_user_qua_score"},
            { "name": "fountain_cascade_weeks_user_qua_score", "as": "weeks_user_qua_score"},
            { "name": "fountain_cascade_week_user_qua_score", "as": "week_user_qua_score"},
            { "name": "fountain_cascade_new_user_produce_thr_l1", "as": "new_user_produce_thr_l1"},
            { "name": "fountain_cascade_month_user_produce_thr_l1", "as": "month_user_produce_thr_l1"},
            { "name": "fountain_cascade_weeks_user_produce_thr_l1", "as": "weeks_user_produce_thr_l1"},
            { "name": "fountain_cascade_week_user_produce_thr_l1", "as": "week_user_produce_thr_l1"}
          ],
          export_item_attr = [
            { "name": "produce_cascade_new_user_score", "as": "fountain_produce_cascade_new_user_score" },
            { "name": "produce_cascade_month_user_score", "as": "fountain_produce_cascade_month_user_score" },
            { "name": "produce_cascade_weeks_user_score", "as": "fountain_produce_cascade_weeks_user_score" },
            { "name": "produce_cascade_week_user_score", "as": "fountain_produce_cascade_week_user_score" },
            { "name": "produce_cascade_consume_score", "as": "fountain_produce_cascade_consume_score" },
            { "name": "is_produce_item_l1", "as": "fountain_produce_mc_is_produce_item_l1" },
            { "name": "is_produce_uploads_item", "as": "fountain_produce_mc_is_produce_uploads_item" },
          ],
          function_name = "CalProduceCascadeScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()
    return self

  def _cascade_ftr_debias(self):
    """
    ftr 线上纠偏: 可采用 ipw 纠偏和结合 duration 的纠偏
    """
    self \
    .if_("fountain_cascade_ftr_slide_kai_predict_all == 1 or fountain_casade_is_fast == 1") \
      .if_("fountain_cascade_fintr_transfer_new == 1") \
        .explore_memory_data_enrich(
          data_key = "{{fountain_cascade_pfintr_debias_map}}",
          data_type = "string_double_vector_map",
          save_data_ptr_to_attr = "fintr_debias_map_ptr",
        ) \
        .explore_trans_fintr_enricher(
          enable_transfer_sigmoid = "{{fountain_cascade_enable_transfer_fintr_sigmoid}}",
          get_fintr_quantile_mode = "{{fountain_cascade_get_fintr_quantile_mode}}",
          fintr_debias_map_attr = "fintr_debias_map_ptr",
          fintr_redis_key_prefix = "{{fountain_cascade_fintr_redis_key_prefix}}",
          fintr_short_photo_cluster_dist = "{{fountain_cascade_fintr_photo_cluster_dist}}",
          fintr_long_photo_threshold = "{{fountain_cascade_fintr_long_photo_threshold}}",
          fintr_long_photo_cluster_dist = "{{fountain_cascade_fintr_long_photo_cluster_dist}}",
          max_dura_limit = "{{fountain_cascade_fintr_max_dura_limit}}",
          max_fintr_limit = "{{fountain_cascade_fintr_max_cluster_limit}}",
          fintr_dist_reciprocal = "{{fountain_cascade_fintr_dist_reciprocal}}",
          enable_map_fintr_positive = "{{fountain_cascade_enable_map_fintr_positive}}",
          enable_multi_duration = "{{fountain_cascade_fintr_enable_multi_duration}}",
          fintr_duration_max_value = "{{fountain_cascade_fintr_duration_max_value}}",
          fintr_duration_power_weight = "{{fountain_cascade_fintr_duration_power_weight}}",
          fintr_duration_offset = "{{fountain_cascade_fintr_duration_offset}}",
          fintr_duration_value_upper_bound = "{{fountain_cascade_fintr_duration_value_upper_bound}}",
          duration_ms_attr = "duration_ms",
          fintr_attr = "cascade_ftr_kai",
          save_fintr_duration_to_attr = "cascade_ftr_kai_duration",
          save_fintr_quantile_to_attr = "cascade_ipw_opt_ftr",
          enable_multiply_prob_view = "{{enable_fountain_cascade_pfintr_multiply_prob_view}}",
          prob_view_attr = "cascade_prob_view",
        ) \
      .end_if_() \
    .end_if_()
    return self

  def _enrich_is_picture(self):
    """
    判断是否是图片，影响后续分桶策略
    """
    self \
    .transform_item_attr(  # 判断是否是图片
      mappings = [{
        "check_attr_name": "upload_type",
        "check_attr_type": "int",
        "output_attr_name": "is_picture",
        "output_attr_type": "int",
        "rules": [{
          "check_values": [7, 10, 11, 70],
          "output_value": 1
        }]
      }]) \
    .transform_item_attr(  # 判断是否是图片
      mappings = [{
        "check_attr_name": "duration_ms",
        "check_attr_type": "int",
        "output_attr_name": "is_picture",
        "output_attr_type": "int",
        "rules": [{
          "check_range": {
            "upper_bound": 101, # 不包含
          },
          "output_value": 1
        }]
      }])

    return self

  def _timestamp_begin(self, name: str):
    return self \
      .gen_common_attr_by_lua(
        attr_map = {
          name + "_begin_ts": "util.GetTimestamp()",
        },
      )

  def _timestamp_end(self, name: str):
    return self \
      .gen_common_attr_by_lua(
        attr_map = {
          name + "_ts": "util.GetTimestamp() - " + name + "_begin_ts",
        },
      )

  def _count_stage_cpu_cost(self, name: str):
    return self \
      .copy_user_meta_info(
        save_flow_cpu_cost_to_attr = name + "_cpu_cost_ts",
      )

  def _get_emp_xtr(self):
    self \
    .explore_user_emp_xtr_enricher(
      colossus_resp_attr = "colossus_resp_v2",
      save_user_click_count = "user_colossus_click_count",
      save_user_emp_ltr = "user_emp_ltr",
      save_user_emp_wtr = "user_emp_wtr",
      save_user_emp_ftr = "user_emp_ftr",
      save_user_emp_htr = "user_emp_htr",
      save_user_emp_cmtr = "user_emp_cmtr",
      save_user_emp_eptr = "user_emp_eptr",
      save_user_emp_svtr = "user_emp_svtr",
      save_user_emp_evtr = "user_emp_evtr",
      save_user_emp_lvtr = "user_emp_lvtr",
      save_user_emp_fintr = "user_emp_fintr",
      save_user_emp_watch_time = "user_emp_watch_time",
      save_user_emp_finish_rate = "user_emp_finish_rate",
      save_user_emp_watch_time_long_video = "user_emp_watch_time_long_video",
      save_user_emp_finish_rate_long_video = "user_emp_finish_rate_long_video",
      save_user_emp_actiononce_ratio = "user_emp_actiononce",
      use_fountain_count_threshold = "{{use_fountain_count_threshold}}"
    ) \
    .perflog_attr_value(
      check_point="fountain.fast.emp_xtr",
      common_attrs=["user_colossus_click_count",
        "user_emp_ltr","user_emp_wtr",
        "user_emp_ftr","user_emp_htr",
        "user_emp_cmtr","user_emp_eptr",
        "user_emp_watch_time", "user_emp_finish_rate",
        "user_emp_watch_time_long_video",
        "user_emp_finish_rate_long_video",
        ]
    ) \
    .log_debug_info(
      common_attrs = [
        "user_emp_watch_time",
        "user_emp_finish_rate",
        "user_emp_watch_time_long_video",
        "user_emp_finish_rate_long_video",
      ],
      for_debug_request_only = True,
    )

    return self

  def _interactive_emp_xtr_change(self):
    self \
    .enrich_attr_by_lua(
      import_common_attr = [
        "user_emp_ltr",
        "user_emp_wtr",
        "user_emp_cmtr",
        "user_emp_ftr",
        "user_emp_eptr"
      ],
      export_common_attr = [
        "userExpLtr",
        "userExpWtr",
        "userExpCmtr",
        "userExpFtr",
        "userExpEptr"
      ],
      function_for_common = "emp_xtr_change",
      lua_script_file = "fountain/cascade/lua/cascade_control.lua"
    ) \

    return self

  def _cascade_pc_combine_score(self):
    self \
    .enrich_attr_by_lua(
      skip = "{{fountain_skip_calc_pc_combine_score}}",
      import_item_attr = [
        "cascade_pcotr",
        "cascade_pctr",
        "cascade_plvtr",
        "cascade_pwatch_time",
        "duration_ms",
      ],
      export_item_attr = [
        "pc_duration",
        "pc_evtr",
        "pc_lvtr",
        "pc_vtr",
      ],
      function_for_item = "calc_pc_combine_score",
      lua_script_file = "fountain/cascade/lua/calc_pxtr.lua",
    ) \
    .log_debug_info(
      for_debug_request_only=True,
      item_attrs=["pc_duration", "pc_evtr", "pc_lvtr", "pc_lvtr"],
    )
    return self
  
  def _cascade_fc_wtd_inverse(self):
    self \
      .get_kconf_params(
        kconf_configs = [
        {
          "kconf_key": "{{cascade_fc_wtd_table_seg_kconf}}",
          "value_type": "json",
          "json_path": "duration_seg",
          "export_common_attr": "cascade_fc_wtd_table_seg"
        },        {
          "kconf_key": "{{cascade_fc_wtd_table_kconf}}",
          "value_type": "json",
          "json_path": "0",
          "export_common_attr": "cascade_fc_wtd_table_0"
        },
        {
          "kconf_key": "{{cascade_fc_wtd_table_kconf}}",
          "value_type": "json",
          "json_path": "9",
          "export_common_attr": "cascade_fc_wtd_table_1"
        },
        {
          "kconf_key": "{{cascade_fc_wtd_table_kconf}}",
          "value_type": "json",
          "json_path": "13",
          "export_common_attr": "cascade_fc_wtd_table_2"
        },
        {
          "kconf_key": "{{cascade_fc_wtd_table_kconf}}",
          "value_type": "json",
          "json_path": "20",
          "export_common_attr": "cascade_fc_wtd_table_3"
        },
        {
          "kconf_key": "{{cascade_fc_wtd_table_kconf}}",
          "value_type": "json",
          "json_path": "38",
          "export_common_attr": "cascade_fc_wtd_table_4"
        },
        {
          "kconf_key": "{{cascade_fc_wtd_table_kconf}}",
          "value_type": "json",
          "json_path": "71",
          "export_common_attr": "cascade_fc_wtd_table_5"
        },
        {
          "kconf_key": "{{cascade_fc_wtd_table_kconf}}",
          "value_type": "json",
          "json_path": "118",
          "export_common_attr": "cascade_fc_wtd_table_6"
        },
        {
          "kconf_key": "{{cascade_fc_wtd_table_kconf}}",
          "value_type": "json",
          "json_path": "195",
          "export_common_attr": "cascade_fc_wtd_table_7"
        },
        {
          "kconf_key": "{{cascade_fc_wtd_table_kconf}}",
          "value_type": "json",
          "json_path": "inf",
          "export_common_attr": "cascade_fc_wtd_table_8"
        },
        ]
      )\
      .enrich_attr_by_light_function(
        import_common_attr=[
          "cascade_fc_wtd_table_seg",
          "cascade_fc_wtd_table_0",
          "cascade_fc_wtd_table_1",
          "cascade_fc_wtd_table_2",
          "cascade_fc_wtd_table_3",
          "cascade_fc_wtd_table_4",
          "cascade_fc_wtd_table_5",
          "cascade_fc_wtd_table_6",
          "cascade_fc_wtd_table_7",
          "cascade_fc_wtd_table_8"
        ],
        import_item_attr=[
          "cascade_fc_pwtd2",
          "duration_ms"
        ],
        export_item_attr=[
          "cascade_fc_pwtd2_inverse"
        ],
        function_name="CalcFountainCascadefcWtdInverse",
        class_name="ExploreLightFunctionSetV2",
      )
    return self

  def _cascade_action_once_score(self):
    self \
    .enrich_attr_by_lua(
      skip = "{{fountain_skip_calc_cascade_action_once_score}}",
      import_common_attr = [
        "cascade_action_once_watchtime_score_pctr_weight",
        "cascade_action_once_watchtime_score_plvtr_weight",
        "cascade_action_once_watchtime_score_pfintr_quantile_weight",
        "cascade_action_once_watchtime_score_pslide_weight",
        "cascade_action_once_watchtime_score_pwatch_time_weight",
        "cascade_action_once_watchtime_score_pwtd_weight",
        "cascade_action_once_watchtime_score_pwtd_kai_weight",
        "cascade_action_once_watchtime_score_pftr_duration_weight",
        "cascade_action_once_interact_score_phtr_weight",
      ],
      import_item_attr = [
        "cascade_pctr",
        "cascade_plvtr",
        "cascade_ipw_opt_ftr",
        "cascade_slide_kai",
        "cascade_pwatch_time",
        "cascade_pwtd",
        "cascade_wtd_kai",
        "cascade_ftr_kai_duration",
        "cascade_pltr",
        "cascade_pwtr",
        "cascade_pftr",
        "cascade_pcmtr",
        "cascade_pcestr",
        "cascade_ptr",
        "cascade_pepstr",
        "cascade_pcltr",
        "cascade_phtr"
      ],
      export_item_attr = [
        "cascade_action_once_interact_score",
        "cascade_action_once_watchtime_score",
      ],
      function_for_item = "cal_action_once_score",
      lua_script_file = "fountain/cascade/lua/calc_pxtr.lua",
    )
    return self

  def _cascade_get_xtr_fractile_score(self):
    self \
    .explore_absolute_xtr_score_que_enricher(
      explore_absolute_xtr_boost_threshold = "{{fountain_mc_pxtr_fractile_boost_threshold}}",
      explore_absolute_xtr_boost_weight = "{{fountain_mc_pxtr_fractile_boost_weight}}",
      enable_explore_absolute_xtr_cliff = "{{enable_fountain_mc_pxtr_fractile_cliff}}",
      pxtr_fractile_kconf_path = "reco.offline.FountainMcPxtrFractile",
      absolute_xtr_score_que_attr = "cascade_pxtr_fractile_score",
      queues = mc_pxtr_fractile_score_queues
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_mc_pxtr_fractile_ensemble_fintr_fractile_wgt", "as": "fintr_fractile_weight"},
        {"name": "fountain_mc_pxtr_fractile_ensemble_wtd_fractile_wgt", "as": "wtd_weight"},
      ],
      import_item_attr = [
        {"name": "cascade_pxtr_fractile_score", "as": "fractile_score"},
        {"name": "cascade_ipw_opt_ftr", "as": "fintr_fractile"},
        {"name": "cascade_pwtd", "as": "wtd"},
      ],
      export_item_attr = [
        {"name": "fractile_score", "as": "cascade_pxtr_fractile_score"},
      ],
      function_name = "AddFractileWeightedScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .if_("enable_get_xtr_fractile_triple_score == 1 and enable_fountain_mc_calc_triplem_time_score == 0", to_be_delete = "date=2024-05-29;committer=gengxiao03") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "enable_fountain_mc_fractile_triple_score_min_max_trans", "as": "enable_min_max_trans"},
          {"name": "fountain_cascade_triplem_evtr_weight", "as": "triplem_evtr_weight"},
          {"name": "fountain_cascade_triplem_fintr_fractile_weight", "as": "triplem_lvtr_weight"},
          {"name": "fountain_cascade_triplem_slide_weight", "as": "triplem_enable_evtr_v2_weight"},
          {"name": "fountain_cascade_triplem_pwatch_time_weight", "as": "triplem_vtr_weight"},
          {"name": "fountain_cascade_triplem_pwtd_reverse_weight", "as": "triplem_fintr_weight"},
          {"name": "fountain_cascade_triplem_pwtd_weight", "as": "triplem_cpr_weight"}
        ],
        import_item_attr = [
          {"name": "cascade_pctr_fractile_score", "as": "evtr"},
          {"name": "cascade_ipw_opt_ftr", "as": "lvtr"},
          {"name": "cascade_slide_kai_fractile_score", "as": "evtr_v2"},
          {"name": "cascade_pwatch_time_fractile_score", "as": "vtr"},
          {"name": "cascade_pwtd_fractile_score", "as": "fintr"},
          {"name": "cascade_pwtd", "as": "cpr"},
        ],
        export_item_attr = [
          {"name": "triplem_time_score", "as": "cascade_triplem_time_score"}
        ],
        function_name = "CalTriplemScore",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_if_()

    return self

  def enrich_cascade_features_by_lua(self):
    """
    精排模型特征填充
    """
    self\
    .enrich_attr_by_lua(
      skip = "{{fountain_skip_cascade_lua_feature_trans}}",
      import_common_attr = [
        "currentTimeMs",
        "cascade_wtd_table_seg",
      ],
      import_item_attr = [
        "upload_time",
        "explore_stat__click_count",
        "explore_stat__like_count",
        "explore_stat__follow_count",
        "explore_stat__forward_count",
        "explore_stat__long_play_count",
        "explore_stat__real_show_count",
        "explore_stat__short_play_count",
        "explore_stat__view_length_sum",
        "author__exp_stat__exp_click",
        "author__exp_stat__exp_like",
        "author__exp_stat__exp_follow",
        "author__exp_stat__exp_long_view",
        "author__exp_stat__exp_realshow",
        "author__exp_stat__exp_forward",
        "author__exp_stat__exp_short_view",
        "author__exp_stat__exp_watch_time",
        "duration_ms",
      ],
      export_item_attr = [
        "featurePUploadTimeDiff",
        "featurePHotClickCount",
        "featurePHotLikeCount",
        "featurePHotFollowCount",
        "featurePHotLongViewCount",
        "featurePHotCtr",
        "featurePHotLtr",
        "featurePHotWtr",
        "featurePHotFtr",
        "featurePHotLvtr",
        "featurePHotSvtr",
        "featurePHotAvgWatchTime",
        "featurePAClickCount",
        "featurePALikeCount",
        "featurePAFollowCount",
        "featurePALongViewCount",
        "featurePACtr",
        "featurePALtr",
        "featurePAWtr",
        "featurePAFtr",
        "featurePALvtr",
        "featurePASvtr",
        "featurePAAvgWatchTime",
        "fountainWtdV3DurationId"
      ],
      function_for_item = "cascade_feature_trans",
      lua_script_file = "fountain/cascade/lua/cascade_feature_trans.lua",
    )
    return self

  def _dump_attr_to_kafka(self, stage_name : str, dump_item_attr_list : list, dump_common_attr_list : list = []):
    """
    dump item attr to kafka
    """
    dump_attr_to_kafka(self, stage_name, dump_item_attr_list, dump_common_attr_list)
    return self

  def _get_user_feature(self):
    self \
    .enrich_with_protobuf(
      from_extra_var = "userInfoPb",
      attrs = [
        dict(name="age_segment", path="basic_info.age_segment"),
        dict(name="basic_info_age_segment_v2", path="basic_info.age_segment_v2"),
        dict(name="basic_info_gender_v2", path="basic_info.gender_v2"),
        dict(name="basic_info_gender", path="basic_info.gender", skip_unset_field=True),
      ],
    )
    return self

  def _get_user_feature_splash(self):
    self \
    .enrich_with_protobuf(
      from_extra_var = "userInfoPb",
      attrs = [
        dict(name="age_segment", path="basic_info.age_segment"),
        dict(name="basic_info_age_segment_v2", path="basic_info.age_segment_v2"),
        dict(name="basic_info_gender_v2", path="basic_info.gender_v2"),
      ],
    )
    return self

  def get_user_group_emp_xtr(self):
    self \
    .enrich_attr_by_light_function(
      import_common_attr = [
        "basic_info_age_segment_v2",
        "basic_info_gender_v2",
      ],
      export_common_attr = [
        "emp_xtr_user_group_prefix_ltr",
        "emp_xtr_user_group_prefix_wtr",
        "emp_xtr_user_group_prefix_ftr",
        "emp_xtr_user_group_prefix_cmtr",
        "emp_xtr_user_group_prefix_ptr",
      ],
      function_name = "CalUserGroupBucket",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .get_kconf_params(
      kconf_configs = [
        {
          "kconf_key": "reco.author.userGroupAgeGenderEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_ltr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_ltr"
        },
        {
          "kconf_key": "reco.author.userGroupAgeGenderEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_wtr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_wtr"
        },
        {
          "kconf_key": "reco.author.userGroupAgeGenderEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_ftr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_ftr"
        },
        {
          "kconf_key": "reco.author.userGroupAgeGenderEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_cmtr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_cmtr"
        },
        {
          "kconf_key": "reco.author.userGroupAgeGenderEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_ptr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_ptr"
        }
      ]
    )
    return self
  
  def get_user_new_group_emp_xtr(self):
    self \
    .set_attr_value(
      no_overwrite = True,
      common_attrs = [
        {
          "name": "fountain_gender_data_type",
          "type": "int",
          "value": 1,
        }
      ]
    ) \
    .enrich_attr_by_light_function(         
      import_common_attr = [
        "basic_info_age_segment_v2",
        "uGender",
        {"name": "fountain_gender_data_type", "as": "is_gender_data_type_list"},
      ],
      export_common_attr = [
        "emp_xtr_user_group_prefix_ltr",
        "emp_xtr_user_group_prefix_wtr",
        "emp_xtr_user_group_prefix_ftr",
        "emp_xtr_user_group_prefix_cmtr",
        "emp_xtr_user_group_prefix_ptr",
      ],
      function_name = "CalNewUserGroupBucket",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .get_kconf_params(
      kconf_configs = [
        {
          "kconf_key": "{{fountain_user_group_interaction_ratio_kconf}}",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_ltr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_ltr"
        },
        {
          "kconf_key": "{{fountain_user_group_interaction_ratio_kconf}}",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_wtr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_wtr"
        },
        {
          "kconf_key": "{{fountain_user_group_interaction_ratio_kconf}}",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_ftr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_ftr"
        },
        {
          "kconf_key": "{{fountain_user_group_interaction_ratio_kconf}}",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_cmtr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_cmtr"
        },
        {
          "kconf_key": "{{fountain_user_group_interaction_ratio_kconf}}",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_ptr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_ptr"
        }
      ]
    )
    return self
  
  def get_user_ten_group_emp_xtr(self):
    self \
    .enrich_attr_by_light_function(         
      import_common_attr = [
        "uMultiDimensionGroupKV",
        "uMultiDimensionGroupDetailKV",
        "get_user_ten_group_emp_xtr_type"
      ],
      export_common_attr = [
        "emp_xtr_user_group_prefix_ltr",
        "emp_xtr_user_group_prefix_wtr",
        "emp_xtr_user_group_prefix_ftr",
        "emp_xtr_user_group_prefix_cmtr",
        "emp_xtr_user_group_prefix_ptr",
        "emp_xtr_user_group_prefix_play_time",
      ],
      function_name = "CalTenGroupUserGroupBucket",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .get_kconf_params(
      kconf_configs = [
        {
          "kconf_key": "reco.author.userTenGroupAgeGenderEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_ltr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_ltr"
        },
        {
          "kconf_key": "reco.author.userTenGroupAgeGenderEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_wtr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_wtr"
        },
        {
          "kconf_key": "reco.author.userTenGroupAgeGenderEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_ftr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_ftr"
        },
        {
          "kconf_key": "reco.author.userTenGroupAgeGenderEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_cmtr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_cmtr"
        },
        {
          "kconf_key": "reco.author.userTenGroupAgeGenderEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_ptr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_ptr"
        },
        {
          "kconf_key": "reco.author.userTenGroupAgeGenderEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_play_time}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_playtime"
        },
      ]
    )
    return self

  def get_user_isweekend_timeslot_group_emp_xtr(self):
    self \
    .enrich_attr_by_light_function(         
      import_common_attr = [
        "is_work_day",
        "request_hour",
        "basic_info_age_segment_v2",
        "user_gender",
      ],
      export_common_attr = [
        "emp_xtr_user_group_prefix_ltr",
        "emp_xtr_user_group_prefix_wtr",
        "emp_xtr_user_group_prefix_ftr",
        "emp_xtr_user_group_prefix_cmtr",
        "emp_xtr_user_group_prefix_ptr",
        "emp_xtr_user_group_prefix_play_time",
      ],
      function_name = "CalWeekendTimeUserGroupBucket",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .get_kconf_params(
      kconf_configs = [
        {
          "kconf_key": "reco.author.userGroupAgeGenderWeekendTimeslotEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_ltr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_ltr"
        },
        {
          "kconf_key": "reco.author.userGroupAgeGenderWeekendTimeslotEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_wtr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_wtr"
        },
        {
          "kconf_key": "reco.author.userGroupAgeGenderWeekendTimeslotEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_ftr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_ftr"
        },
        {
          "kconf_key": "reco.author.userGroupAgeGenderWeekendTimeslotEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_cmtr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_cmtr"
        },
        {
          "kconf_key": "reco.author.userGroupAgeGenderWeekendTimeslotEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_ptr}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_ptr"
        },
        {
          "kconf_key": "reco.author.userGroupAgeGenderWeekendTimeslotEmpXtr",
          "value_type": "double",
          "json_path": "{{emp_xtr_user_group_prefix_play_time}}",
          "default_value": 1.0,
          "export_common_attr": "user_group_emp_playtime"
        },
      ]
    )
    return self
  
  def _cal_user_group_emp_xtr_in_cascade(self):
    self \
    .gen_common_attr_by_lua(
      attr_map={
        "fountain_cascade_ensemble_power_weight_cascade_like_emp": "fountain_cascade_ensemble_power_weight_cascade_like_emp * user_group_emp_ltr",
        "fountain_cascade_ensemble_power_weight_cascade_follow_emp": "fountain_cascade_ensemble_power_weight_cascade_follow_emp * user_group_emp_wtr",
        "fountain_cascade_ensemble_power_weight_cascade_forward_emp": "fountain_cascade_ensemble_power_weight_cascade_forward_emp * user_group_emp_ftr",
        "fountain_cascade_ensemble_power_weight_cascade_comment_emp": "fountain_cascade_ensemble_power_weight_cascade_comment_emp * user_group_emp_cmtr",
      }
    )
    return self
        
  def _mc_high_value_pic_boost(self):
    self \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_mc_hv_pic_boost_coef", "as": "boost_discount_coeff"},
        ],
        import_item_attr = [
          {"name": "cascade_ensemble_score", "as": "score"},
        ],
        export_item_attr = [
          {"name": "score", "as": "cascade_ensemble_score"},
        ],
        function_name = "BoostOrDiscountV2",
        class_name = "ExploreLightFunctionSetV2",
        target_item = { "high_value_pic_flag": 1 },
      )
    return self

  def _enrich_personilize_issue_score(self):
    self \
      .explore_custom_trim_user_info(
        user_info_attr = "userInfo",
        save_trimed_user_info_to_attr = "personilize_issue_trimmed_user_info",
        trim_user_info = [
          "id",
          "device_id",
          "request_location.city_id",
          "request_location.province_id",
          "gender",
          "infer_year",
          "basic_info.age_segment",
          "location.city_id",
          "client_id",
          "visit_mod",
          "user_profile.user_level",
          "active_days",
          "user_profile.exp_stat.exp_click",
          "user_profile.exp_stat.exp_like",
          "user_profile.exp_stat.exp_follow",
          "user_profile.exp_stat.exp_realshow",
          "user_profile.exp_stat.exp_long_view",
        ],
      ) \
      .delegate_enrich(
        kess_service = "{{fountain_personilize_issue_kess_service}}",
        recv_item_attrs = [
          {"name": "hot_content", "as": "hot_content_feedback_score"},
          {"name": "authority_content", "as": "authority_content_feedback_score"},
          {"name": "personified_author", "as": "personified_author_feedback_score"},
        ],
        timeout_ms = 100,
        send_item_attrs = [],
        send_common_attrs = [
          { "name": "personilize_issue_trimmed_user_info", "as": "user_info_str" },
        ],
        request_type = "{{fountain_personilize_issue_request_type}}",
        partition_size = "{{fountain_personilize_issue_partition_size}}",
      )
    return self

  def _mc_s2_operation_target_photo_boost(self):
    contents = [
      {
        "name" : "is_hot_content",
        "coefficient_attr" : "fountain_mc_s2_operation_hot_content_boost_coef",
        "feedback_score_attr" : "hot_content_feedback_score",
      },
      {
        "name" : "is_authority_content",
        "coefficient_attr" : "fountain_mc_s2_operation_authority_content_boost_coef",
        "feedback_score_attr" : "authority_content_feedback_score",
      },
      {
        "name" : "is_personified_author",
        "coefficient_attr" : "fountain_mc_s2_operation_personified_author_boost_coef",
        "feedback_score_attr" : "personified_author_feedback_score",
      },
    ]

    self \
      .explore_user_feedback_issue_enrich(
        contents = contents,
        score_attr = "cascade_ensemble_score"
      )
    return self

  def _mc_hot_content_retr_boost(self):
    self \
      .if_("enable_hot_content_thompson_sampling_corr_calculate == 1", to_be_delete = "date=2024-05-29;committer=guohao") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_mc_hot_content_retr_boost_coef", "as": "value"},
            {"name": "hot_content_corr", "as": "weight"},
          ],
          export_common_attr = [
            {"name": "new_value", "as": "fountain_mc_hot_content_retr_boost_coef"},
          ],
          function_name = "CalExploreDoubleMultiDouble",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_mc_hot_content_retr_boost_coef", "as": "boost_discount_coeff"},
        ],
        import_item_attr = [
          {"name": "cascade_ensemble_score", "as": "score"},
        ],
        export_item_attr = [
          {"name": "score", "as": "cascade_ensemble_score"},
        ],
        function_name = "BoostOrDiscountV2",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          "reason" : [341, 416]
        }
      )
    return self

  def _calc_true_living(self):
    self.if_("enable_fountain_calc_true_living == 1") \
      .explore_memory_data_enrich(
        data_key = "merchant_live_authors_set_v2",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "merchant_live_authors_set__memory_data_v2",
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          {"name": "author__id", "as": "author__id"},
          { "name": "live_photo_info__is_living", "as": "is_living" },
        ],
        import_common_attr = [
          "merchant_live_authors_set__memory_data_v2",
        ],
        export_item_attr = [
          "is_true_living",
        ],
        function_name = "GetIsTrueLiving",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .end_()
    return self

  def _audit_adjust_score(self):
    self \
      .if_("fountain_mc_enable_audit_cold_review_level_adjust == 1") \
        .split_string(
          input_common_attr = "fountain_mc_audit_cold_review_level_black_tag_set_str",
          output_common_attr = "fountain_mc_audit_cold_review_level_black_tag_set",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True,
        ) \
        .transform_item_attr(
          mappings = [{
            "check_attr_name": "audit_cold_review_level",
            "check_attr_type": "int",
            "output_attr_name": "is_audit_cold_review_level_discount",
            "output_attr_type": "int",
            "rules": [{
              "check_values": ["{{fountain_mc_audit_cold_review_level_black_tag_set}}"],
              "output_value": 1
            }]
          }]
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_mc_audit_cold_review_level_black_coeff", "as": "boost_discount_coeff"}
          ],
          import_item_attr = [
            {"name": "is_audit_cold_review_level_discount", "as": "need_item_attr"},
            {"name": "cascade_ensemble_score", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "cascade_ensemble_score"},
          ],
          function_name = "BoostOrDiscount",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("fountain_mc_enable_impression_audit_adjust == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
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
          }]
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_mc_impression_audit_adjust_coeff_map_str", "as": "adjust_coeff_map_str_attr"}
          ],
          import_item_attr = [
            {"name": "content_safety_level_with_namespace__level_hot_online", "as": "audit_level_attr"},
            {"name": "cascade_ensemble_score", "as": "ensemble_score_attr"},
            "upload_time"
          ],
          export_item_attr = [
            {"name": "ensemble_score_attr", "as": "cascade_ensemble_score"},
          ],
          function_name = "AuditAdjustScore",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {
            "is_impression_audit": 1,
          },
        ) \
      .end_() \
      .if_("fountain_mc_enable_high_hot_audit_adjust == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_mc_high_hot_audit_adjust_coeff_map_str", "as": "adjust_coeff_map_str_attr"}
          ],
          import_item_attr = [
            {"name": "audit_hot_high_tag_level", "as": "audit_level_attr"},
            {"name": "cascade_ensemble_score", "as": "ensemble_score_attr"},
            "upload_time"
          ],
          export_item_attr = [
            {"name": "ensemble_score_attr", "as": "cascade_ensemble_score"},
          ],
          function_name = "AuditAdjustScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("fountain_mc_enable_topk_audit_adjust == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_mc_topk_audit_adjust_coeff_map_str", "as": "adjust_coeff_map_str_attr"}
          ],
          import_item_attr = [
            {"name": "topk_audit_level", "as": "audit_level_attr"},
            {"name": "cascade_ensemble_score", "as": "ensemble_score_attr"},
            "upload_time"
          ],
          export_item_attr = [
            {"name": "ensemble_score_attr", "as": "cascade_ensemble_score"},
          ],
          function_name = "AuditAdjustScore",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()
    return self

  def _fountain_mc_follow_people_boost(self):
    self \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_mc_valid_follow_boost_discount_coeff", "as": "valid_follow_boost_discount_coeff"},
          {"name": "fountain_mc_low_follow_boost_discount_coeff", "as": "low_follow_boost_discount_coeff"},
          {"name": "fountain_mc_media_follow_boost_discount_coeff", "as": "media_follow_boost_discount_coeff"},
          {"name": "fountain_mc_high_follow_boost_discount_coeff", "as": "high_follow_boost_discount_coeff"},
          {"name": "fountain_mc_no_follow_boost_discount_coeff", "as": "no_follow_boost_discount_coeff"},
          {"name": "user_follow_type", "as": "user_follow_type"},
          {"name": "user_fountain_follow_aid_list", "as": "follow_aid_list"},
        ],
        import_item_attr = [
          {"name": "cascade_ensemble_score", "as": "score"},
          {"name": "author__id", "as": "aid"}
        ],
        export_item_attr = [
          {"name": "score", "as": "cascade_ensemble_score"},
        ],
        function_name = "FollowPeopleBoostOrDiscount",
        class_name = "ExploreLightFunctionSetV2",
      )
    return self

  def _refinement_boost_personified_author(self):
    """
    Module: cascade_v12_flow
    功能: 细分用户和视频维度，精细化对人格化账号提权
    Owner: xubaoquan
    Date: 2023-07-19
    :return:
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "basic_info_age_segment_v2", "as": "basic_info_age_segment_v2"},
        {"name": "basic_info_gender_v2", "as": "basic_info_gender_v2"},
        {"name": "explore_personifed_author_boost_ptr", "as": "boost_map_ptr"},
        {"name": "refinement_boost_personified_author_redis_prefix", "as": "redis_prefix"},
        {"name": "fountain_cascade_refinement_boost_personified_author_power_weight", "as": "power_weight"},
      ],
      import_item_attr = [
        {"name": "author__gender", "as": "author__gender"},
        {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_level_one_list"},
        {"name": "cascade_ensemble_score", "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "cascade_ensemble_score"},
      ],
      target_item = { 
        "eyeshot_source" : 1
      },
      function_name = "UniverseRefinementBoost",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def _get_mmu_embedding(self):
    self \
      .explore_embedding_candidates_attr_enricher(
        trans_type = "fountain_candidates",
        enable_fix_low_hit_rate = True,
        enable_not_click = False,
        enable_play_stat = True,
        enable_hate = False,
        enable_explore_not_click = False,
        enable_source_photo = True,
        source_pid_attr = "featureSourcePId",
        session_history_max_size = "{{fountain_mc_mgs_diversity_max_size}}",
        user_info_ptr_attr = "userInfoPb",
        export_common_attr = "topk_mgs_embedding_source_pids",
        check_point = "cascade",
      ) \
      .get_remote_embedding_lite(
        kess_service = "grpc_MMUHetuSimContentEmbedding",
        shard_num = 4,
        id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
        input_attr_name = "topk_mgs_embedding_source_pids",
        output_attr_name = "topk_mgs_embeddings",
        query_source_type = "common_attr",
        size = 64,
        client_side_shard = True
      )
    return self

  def _calc_topk_mgs_expected_score(self):
    self \
      .explore_get_embedding_map_enricher(
        embedding_list_attr = "topk_mgs_embeddings",
        source_pids_list_attr = "topk_mgs_embedding_source_pids",
        dim_size = 64,
        export_common_attr = "topk_mgs_pid_embedding_map",
      ) \
      .explore_diversity_update_enricher(
        user_info_ptr_attr = "userInfoPb",
        pid_embedding_common_attr = "topk_mgs_pid_embedding_map",
        export_item_attr = "topk_mgs_expected_score",
        history_feed_back_version = 3,
        dim_size = 64,
        expected_score_cand_size = "{{fountain_mc_topk_mgs_expected_score_cand_num}}",
        max_interval_second = "{{fountain_mc_topk_mgs_expected_score_max_interval_second}}",
        min_duration_threshold = "{{fountain_mc_topk_mgs_expected_score_min_duration_threshold}}",
        dpp_diversity_mgs_topk = "{{fountain_mc_topk_mgs_expected_score_topk_num}}",
        max_playtime_threshold = "{{fountain_mc_topk_mgs_expected_score_max_playtime_threshold}}",
        min_playtime_threshold = "{{fountain_mc_topk_mgs_expected_score_min_playtime_threshold}}",
        enable_use_weight = "{{fountain_mc_topk_mgs_expected_score_enable_use_weight}}",
        weight_version = "{{fountain_mc_topk_mgs_expected_score_weight_version}}",
        ratio_scale = "{{fountain_mc_topk_mgs_expected_score_ratio_scale}}",
        ratio_pow_weight = "{{fountain_mc_topk_mgs_expected_score_ratio_pow_weight}}",
      )
    return self

  def _calc_negative_feedback_similary_score(self):
    """
    计算候选item和历史负反馈(短播 曝光未点击等)的相似分
    """
    self \
      .explore_embedding_candidates_attr_enricher(
        trans_type = "fountain_candidates",
        enable_fix_low_hit_rate = "{{fountain_mc_enable_fix_mmu_embedding_low_hit_rate}}",
        user_info_ptr_attr = "userInfoPb",
        export_common_attr = "embedding_source_pids",
        check_point = "cascade",
      ) \
      .get_remote_embedding_lite(
        kess_service = "{{fountain_mc_emb_kess_name_for_neg_feedback_sim_score}}",
        shard_num = 4,
        id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
        input_attr_name = "embedding_source_pids",
        output_attr_name = "mmu_embeddings",
        query_source_type = "common_attr",
        size = 64,
        client_side_shard = True
      ) \
      .explore_custom_embedding_score_enricher(
        check_point_ = "cascade",
        enable_fountain_version = True,
        enable_fix_low_hit_rate = "{{fountain_mc_enable_fix_mmu_embedding_low_hit_rate}}",
        user_info_ptr_attr = "userInfoPb",
        embedding_list_attr = "mmu_embeddings",
        source_pids_list_attr = "embedding_source_pids",
        calc_type = "action_bucket_dot",
        not_click_limit_hour = "{{fountain_mc_neg_feedback_sim_score_not_click_hour_limit}}",
        play_stat_limit_hour = "{{fountain_mc_neg_feedback_sim_score_play_stat_hour_limit}}",
        extra_not_click_limit_hour = "{{fountain_mc_neg_feedback_sim_score_extra_not_click_hour_limit}}",
        short_view_threshold = "{{fountain_mc_neg_feedback_sim_score_short_view_threshold}}",
        not_click_weight = "{{fountain_mc_neg_feedback_sim_score_not_click_weight}}",
        short_view_weight = "{{fountain_mc_neg_feedback_sim_score_short_view_weight}}",
        extra_not_click_weight = "{{fountain_mc_neg_feedback_sim_score_extra_not_click_weight}}",
        export_item_attr = "hate_similary_score",
        dim_size = 64
      )
    return self

  
  def _mc_s1_collection_type_boost(self):
    """
    粗排 s1 对合集内容 boost
    """
    self \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_mc_s1_collection_type_boost_coef", "as": "boost_discount_coeff"},
        ],
        import_item_attr = [
          {"name": "cascade_variant_sort_adjust_score", "as": "score"},
        ],
        export_item_attr = [
          {"name": "score", "as": "cascade_variant_sort_adjust_score"},
        ],
        function_name = "BoostOrDiscountV2",
        class_name = "ExploreLightFunctionSetV2",
        target_item = { 
          "is_collection" : 1,
        }
      )
    return self
  
  def _mc_s2_collection_type_boost(self):
    """
    粗排 s2 对合集内容 boost
    """
    self \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_mc_s2_collection_type_boost_coef", "as": "boost_discount_coeff"},
        ],
        import_item_attr = [
          {"name": "cascade_ensemble_score", "as": "score"},
        ],
        export_item_attr = [
          {"name": "score", "as": "cascade_ensemble_score"},
        ],
        function_name = "BoostOrDiscountV2",
        class_name = "ExploreLightFunctionSetV2",
        target_item = { 
          "is_collection" : 1,
        }
      )
    return self

  def _get_behaviour_hetu_diversity_boost_coeff(self):
    """
    行为期望 & 类目多样性
    """
    self \
      .pack_item_attr(
        item_source = {
          "reco_results": True,
        },
        mappings = [
          {
            "aggregator": "avg",
            "from_item_attr": "cascade_pctr",
            "to_common_attr": "mc_pctr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "cascade_plvtr",
            "to_common_attr": "mc_plvtr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "cascade_slide_kai",
            "to_common_attr": "mc_pslide_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "cascade_pwatch_time",
            "to_common_attr": "mc_pwatch_time_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "cascade_pwtd",
            "to_common_attr": "mc_pwtd_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "cascade_pltr",
            "to_common_attr": "mc_pltr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "cascade_pwtr",
            "to_common_attr": "mc_pwtr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "cascade_pepstr",
            "to_common_attr": "mc_pepstr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "cascade_pcmtr",
            "to_common_attr": "mc_pcmtr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "cascade_pcltr",
            "to_common_attr": "mc_pcltr_avg"
          },
          {
            "aggregator": "avg",
            "from_item_attr": "cascade_pftr",
            "to_common_attr": "mc_pftr_avg"
          },
        ]
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "colossus_hetu_distribution_hetu_stat",
          {"name": "fountain_mc_behaviour_hetu_diversity_hetu_coef_beta", "as": "hetu_coef_beta"},
          {"name": "fountain_mc_behaviour_hetu_diversity_enable_unknown_hetu_adjust", "as": "enable_unknown_hetu_adjust"},
          {"name": "mc_pctr_avg", "as": "pctr_avg"},
          {"name": "mc_plvtr_avg", "as": "plvtr_avg"},
          {"name": "mc_pslide_avg", "as": "pslide_avg"},
          {"name": "mc_pwatch_time_avg", "as": "pwatch_time_avg"},
          {"name": "mc_pwtd_avg", "as": "pwtd_avg"},
          {"name": "mc_pltr_avg", "as": "pltr_avg"},
          {"name": "mc_pwtr_avg", "as": "pwtr_avg"},
          {"name": "mc_pepstr_avg", "as": "pepstr_avg"},
          {"name": "mc_pcmtr_avg", "as": "pcmtr_avg"},
          {"name": "mc_pcltr_avg", "as": "pcltr_avg"},
          {"name": "mc_pftr_avg", "as": "pftr_avg"},
          {"name": "fountain_mc_behaviour_hetu_diversity_pctr_alpha", "as": "pctr_alpha"},
          {"name": "fountain_mc_behaviour_hetu_diversity_plvtr_alpha", "as": "plvtr_alpha"},
          {"name": "fountain_mc_behaviour_hetu_diversity_pslide_alpha", "as": "pslide_alpha"},
          {"name": "fountain_mc_behaviour_hetu_diversity_pwatch_time_alpha", "as": "pwatch_time_alpha"},
          {"name": "fountain_mc_behaviour_hetu_diversity_pwtd_alpha", "as": "pwtd_alpha"},
          {"name": "fountain_mc_behaviour_hetu_diversity_pltr_alpha", "as": "pltr_alpha"},
          {"name": "fountain_mc_behaviour_hetu_diversity_pwtr_alpha", "as": "pwtr_alpha"},
          {"name": "fountain_mc_behaviour_hetu_diversity_pepstr_alpha", "as": "pepstr_alpha"},
          {"name": "fountain_mc_behaviour_hetu_diversity_pcmtr_alpha", "as": "pcmtr_alpha"},
          {"name": "fountain_mc_behaviour_hetu_diversity_pcltr_alpha", "as": "pcltr_alpha"},
          {"name": "fountain_mc_behaviour_hetu_diversity_pftr_alpha", "as": "pftr_alpha"},
          {"name": "fountain_mc_behaviour_hetu_diversity_pctr_beta", "as": "pctr_beta"},
          {"name": "fountain_mc_behaviour_hetu_diversity_plvtr_beta", "as": "plvtr_beta"},
          {"name": "fountain_mc_behaviour_hetu_diversity_pslide_beta", "as": "pslide_beta"},
          {"name": "fountain_mc_behaviour_hetu_diversity_pwatch_time_beta", "as": "pwatch_time_beta"},
          {"name": "fountain_mc_behaviour_hetu_diversity_pwtd_beta", "as": "pwtd_beta"},
          {"name": "fountain_mc_behaviour_hetu_diversity_pltr_beta", "as": "pltr_beta"},
          {"name": "fountain_mc_behaviour_hetu_diversity_pwtr_beta", "as": "pwtr_beta"},
          {"name": "fountain_mc_behaviour_hetu_diversity_pepstr_beta", "as": "pepstr_beta"},
          {"name": "fountain_mc_behaviour_hetu_diversity_pcmtr_beta", "as": "pcmtr_beta"},
          {"name": "fountain_mc_behaviour_hetu_diversity_pcltr_beta", "as": "pcltr_beta"},
          {"name": "fountain_mc_behaviour_hetu_diversity_pftr_beta", "as": "pftr_beta"},
        ],
        import_item_attr = [
          {"name": "cascade_pctr", "as": "pctr"},
          {"name": "cascade_plvtr", "as": "plvtr"},
          {"name": "cascade_slide_kai", "as": "pslide"},
          {"name": "cascade_pwatch_time", "as": "pwatch_time"},
          {"name": "cascade_pwtd", "as": "pwtd"},
          {"name": "cascade_pltr", "as": "pltr"},
          {"name": "cascade_pwtr", "as": "pwtr"},
          {"name": "cascade_pepstr", "as": "pepstr"},
          {"name": "cascade_pcmtr", "as": "pcmtr"},
          {"name": "cascade_pcltr", "as": "pcltr"},
          {"name": "cascade_pftr", "as": "pftr"},
          {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_level_one_list"},
        ],
        export_item_attr = [
          {"name": "behaviour_hetu_diversity_boost_coeff", "as": "mc_behaviour_hetu_diversity_boost_coeff"}
        ],
        function_name = "GetBehaviourHetuDiversityBoostCoeff",
        class_name = "ExploreLightFunctionSetV2",
      )
    return self


  def _mc_s2_behaviour_hetu_diversity_boost(self):
    """
    行为期望 & 类目多样性, S2 boost
    """
    self \
      .enrich_attr_by_light_function(
        import_item_attr = [
          {"name": "mc_behaviour_hetu_diversity_boost_coeff", "as": "boost_discount_coeff"},
          {"name": "cascade_ensemble_score", "as": "score"},
        ],
        export_item_attr = [
          {"name": "score", "as": "cascade_ensemble_score"},
        ],
        function_name = "BoostOrDiscountByItemCoeff",
        class_name = "ExploreLightFunctionSetV2"
      )
    return self

  def _mc_calc_adjust_coeff(self):
    """
    粗排多样性调权系数计算
    """
    self \
      .if_("enable_fountain_mc_calc_candidate_diversity_coeff == 1", to_be_delete = "date=2024-05-29;committer=liuhao07") \
        .split_string(
          input_common_attr = "fountain_mc_age_group_candidate_diversity_coeff_str",
          output_common_attr = "fountain_mc_age_group_candidate_diversity_coeff",
          delimiters = ",",
          parse_to_double = True,
          trim_spaces = True,
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_mc_candidate_diversity_coeff_a", "as": "coeff_a"},
            {"name": "fountain_mc_candidate_diversity_coeff_b", "as": "coeff_b"},
            {"name": "fountain_mc_candidate_diversity_coeff_c", "as": "coeff_c"},
            {"name": "fountain_mc_candidate_diversity_use_no_tag", "as": "use_no_tag"},
            {"name": "fountain_mc_age_group_candidate_diversity_coeff", "as": "age_group_candidate_diversity_coeff"},
            "basic_info_age_segment_v2",
          ],
          import_item_attr = [
            {"name": "hetu_tag_level_info_v2__hetu_level_two", "as": "hetu_level_two"},
          ],
          export_item_attr = [
            {"name": "diversity_coeff", "as": "mc_candidate_diversity_coeff"}
          ],
          function_name = "CalcCandidateDiversityCoeff",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_() \
      .if_("enable_fountain_mc_calc_behaviour_hetu_diversity == 1", to_be_delete = "date=2024-05-29;committer=gengxiao03") \
        ._get_behaviour_hetu_diversity_boost_coeff() \
      .end_() \
      .if_("enable_fountain_mc_calc_hetu_coeff == 1", to_be_delete = "date=2024-05-29;committer=liuhao07") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_mc_hetu_coeff_map_str", "as": "hetu_coeff_map"},
          ],
          import_item_attr = [
            {"name": "hetu_tag_level_info_v2__hetu_level_one", "as": "hetu_level_one"},
          ],
          export_item_attr = [
            {"name": "hetu_coeff", "as": "mc_hetu_coeff"}
          ],
          function_name = "CalcHetuCoeff",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_() \
      .if_("enable_fountain_mc_calc_duration_coeff == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_mc_duration_coeff_map_str", "as": "duration_coeff_map"},
          ],
          import_item_attr = [
            "picture_type",
            "duration_ms"
          ],
          export_item_attr = [
            {"name": "duration_coeff", "as": "mc_duration_coeff"}
          ],
          function_name = "CalcDurationCoeff",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_() \
      .if_("enable_fountain_mc_calc_explore_coeff == 1", to_be_delete = "date=2024-05-29;committer=liuhao07") \
        .split_string(
          input_common_attr = "fountain_mc_age_group_explore_coeff_str",
          output_common_attr = "fountain_mc_age_group_explore_coeff",
          delimiters=",",
          parse_to_double = True,
          trim_spaces = True,
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "basic_info_age_segment_v2",
            {"name": "fountain_mc_age_group_explore_coeff", "as": "age_group_explore_coeff"},
            "explore_hetu_list"
          ],
          import_item_attr = [
            {"name": "hetu_tag_level_info__hetu_level_two", "as": "hetu_level_two"},
          ],
          export_item_attr = [
            {"name": "explore_coeff", "as": "mc_explore_coeff"}
          ],
          function_name = "CalcExploreCoeff",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_() \
      .if_("enable_fountain_mc_s1_calc_pos_neg_ratio_boost_coeff == 1", to_be_delete = "date=2024-05-29;committer=gengxiao03") \
        .calc_mc_pos_neg_ratio_boost_coeff( # 只要S1计算系数 S1的booster自动做boost
        ) \
      .end_() \
      .if_("enable_fountain_mc_s1_calc_watch_time_boost_coeff == 1", to_be_delete = "date=2024-05-29;committer=gengxiao03") \
        .calc_mc_watch_time_boost_coeff( # 只要S1计算系数 S1的booster自动做boost
        ) \
      .end_() \
      .enrich_attr_by_light_function(
        import_item_attr = [
          {"name": "mc_candidate_diversity_coeff", "as": "candidate_diversity_coeff"},
          {"name": "mc_behaviour_hetu_diversity_boost_coeff", "as": "behaviour_hetu_diversity_coeff"},
          {"name": "mc_hetu_coeff", "as": "hetu_coeff"},
          {"name": "mc_duration_coeff", "as": "duration_coeff"},
          {"name": "mc_explore_coeff", "as": "explore_coeff"},
          {"name": "fountain_mc_pos_neg_ratio_boost_coeff", "as": "pos_neg_ratio_coeff"},
          {"name": "mc_watch_time_boost_coeff", "as": "watch_time_coeff"},
        ],
        export_item_attr = [
          "mc_adjust_coeff_final"
        ],
        function_name = "CalcMcAdjustCoeffFinal",
        class_name = "ExploreLightFunctionSetV2"
      ) \
      .if_("enable_fountain_mc_bid_coeff_boost == 1") \
        .calc_mc_bid_boost_coeff() \
      .end_() \
      .if_("enable_fountain_mc_calc_high_share_boost_coeff == 1 and bid_follow_num ~= 0 and (user_msg_cnt_ssm_today + user_msg_cnt_gsm_today <= 0)") \
        .calc_mc_high_share_boost_coeff() \
      .end_() \
      .if_("enable_fountain_mc_marketing_compensation_adjust == 1") \
        .mc_marketing_compensation_adjust() \
      .end_() \
      .if_("enable_fountain_bad_comment_pids_memory_data == 1") \
        .explore_memory_data_enrich(
          data_key = "bad_comment_pids",
          data_type = "uint64_set",
          save_data_ptr_to_attr = "bad_comment_pids_ptr",
        ) \
      .end_() \
      .if_("enable_fountain_mc_bad_comment_pids_adjust == 1") \
        .mc_bad_comment_pids_score_adjust() \
      .end_() \
      .if_("enable_fountain_mc_sideinfo_retargeting_score_adjust == 1") \
        .mc_sideinfo_retargeting_score_adjust() \
      .end_() \
      .if_("enable_mc_marketing_compensation_photo_personal_adjust == 1") \
        .mc_marketing_compensation_photo_personal_adjust() \
      .end_() \
      .if_("enable_fountain_mc_low_cost_photo_adjust == 1") \
        .mc_low_cost_photo_adjust() \
      .end_() \
      .if_("enable_fountain_mc_high_photo_count_author_adjust == 1") \
        .mc_high_photo_count_author_adjust() \
      .end_() \
      .if_("enable_fountain_mc_llm_negative_photo_adjust == 1") \
        .mc_llm_negative_photo_adjust() \
      .end_() \
      .if_("enable_fountain_mc_llm_negative_photo_personal_adjust == 1") \
        .mc_llm_negative_photo_personal_adjust() \
      .end_() \
      .if_("enable_fountain_mc_sexy_induce_adjust == 1") \
        .mc_sexy_induce_adjust() \
      .end_() \
      .if_("enable_fountain_mc_sexy_induce_personal_adjust == 1") \
        .mc_sexy_induce_personal_adjust() \
      .end_() \
      .if_("enable_fountain_mc_living_photo_adjust == 1") \
        .mc_living_photo_adjust_by_paying_type() \
      .end_() \
      .if_("enable_fountain_mc_career_interest_tagnex_tgi_adjust == 1") \
        .mc_interest_tagnex_tgi_adjust("career") \
      .end_() \
      .if_("enable_fountain_mc_group_interest_tagnex_tgi_adjust == 1") \
        .mc_interest_tagnex_tgi_adjust("group") \
      .end_()
    return self

  def calc_mc_pos_neg_ratio_boost_coeff(self):
    self \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_mc_like_hate_ratio_boost_alpha", "as": "like_hate_ratio_alpha"},
        {"name": "fountain_mc_like_hate_ratio_boost_weight", "as": "like_hate_ratio_weight"},
        {"name": "fountain_mc_long_short_view_ratio_boost_alpha", "as": "long_short_view_ratio_alpha"},
        {"name": "fountain_mc_long_short_view_ratio_boost_weight", "as": "long_short_view_ratio_weight"},
      ],
      import_item_attr = [
        {"name": "cascade_pltr", "as": "pltr_attr"},
        {"name": "cascade_phtr", "as": "phtr_attr"},
        {"name": "cascade_plvtr", "as": "plvtr_attr"},
        {"name": "cascade_psvtr", "as": "psvtr_attr"},
      ],
      export_item_attr = [
        {"name": "boost_coeff", "as": "fountain_mc_pos_neg_ratio_boost_coeff"},
      ],
      function_name = "CalcPosNegRatioBoostCoeff",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def calc_mc_watch_time_boost_coeff(self):
    self \
    .switch_("fountain_mc_watch_time_boost_mode") \
      .case_(1, to_be_delete = "date=2024-05-29;committer=gengxiao03") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_mc_watch_time_boost_alpha", "as": "alpha"},
            {"name": "fountain_mc_watch_time_boost_upper_bound", "as": "upper_bound"},
          ],
          import_item_attr = [
            {"name": "cascade_ftr_kai_duration", "as": "pwatch_time_attr"},
          ],
          export_item_attr = [
            {"name": "boost_coeff", "as": "mc_watch_time_boost_coeff"},
          ],
          function_name = "CalcWatchTimeBoostCoeff",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .case_(2, to_be_delete = "date=2024-05-29;committer=gengxiao03") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_mc_watch_time_boost_alpha", "as": "alpha"},
            {"name": "fountain_mc_watch_time_boost_upper_bound", "as": "upper_bound"},
          ],
          import_item_attr = [
            {"name": "cascade_wtd_percent", "as": "pwatch_time_attr"},
          ],
          export_item_attr = [
            {"name": "boost_coeff", "as": "mc_watch_time_boost_coeff"},
          ],
          function_name = "CalcWatchTimeBoostCoeff",
          class_name = "ExploreLightFunctionSetV2",
        ) \
    .end_()
    return self

  def calc_mc_bid_boost_coeff(self):
    self \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_mc_bid_coeff_boost_weight", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "mc_adjust_coeff_final", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "mc_adjust_coeff_final"},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {"is_bid_follow_author": 1}
    )
    return self
  
  def calc_mc_high_share_boost_coeff(self):
    self \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_mc_high_share_boost_weight", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "mc_adjust_coeff_final", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "mc_adjust_coeff_final"},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {"is_high_share_photo": 1}
    )
    return self

  def gen_is_marketing_compensation_photo(self):
    self \
    .split_string(
      input_common_attr = "fountain_marketing_compensation_photo_tags_list_str",
      output_common_attr = "fountain_marketing_compensation_photo_tags_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True
    ) \
    .explore_memory_data_enrich(
      data_key = "high_value_black_author_map",
      data_type = "uint64_double_vector_map",
      save_data_ptr_to_attr = "high_value_black_author_map_ptr",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_marketing_compensation_photo_tags_list", "as": "tags_list"},
        {"name": "fountain_marketing_compensation_high_value_author_ignore", "as": "high_value_author_ignore"},
        {"name": "fountain_marketing_compensation_open_reason_thres", "as": "open_reason_thres"},
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
    .if_("enable_fountain_mc_calc_marketing_compensation_coeff == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_mc_marketing_compensation_adjust_ctr_weight", "as": "ctr_weight"},
          {"name": "fountain_mc_marketing_compensation_adjust_watchtime_weight", "as": "watchtime_weight"},
          {"name": "fountain_mc_marketing_compensation_adjust_score_base", "as": "score_base"},
          {"name": "fountain_mc_marketing_compensation_adjust_adjust_version", "as": "adjust_version"},
          {"name": "fountain_mc_marketing_compensation_adjust_score_base_ratio", "as": "score_base_ratio"},
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
        {"name": "fountain_mc_marketing_compensation_adjust_scale_factor", "as": "scale_factor"},
        {"name": "fountain_mc_marketing_compensation_adjust_base_coeff", "as": "base_coeff"},
        {"name": "fountain_mc_marketing_compensation_adjust_bug_fix", "as": "bug_fix"},
      ],
      import_item_attr = [
        {"name": "mc_adjust_coeff_final", "as": "old_coeff"},
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

  def gen_is_low_cost_photo(self):
    self \
    .enrich_attr_by_light_function(
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
  
  def gen_is_minority_photo(self):
    self \
    .split_string(
      input_common_attr = "fountain_fast_minority_photo_tags_bits_list_str",
      output_common_attr = "fountain_fast_minority_photo_tags_bits_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True,
    ) \
    .split_string(
      input_common_attr = "fountain_minority_photo_manjiao_markcode_tags_str",
      output_common_attr = "fountain_minority_photo_manjiao_markcode_tags",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True,
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_fast_minority_photo_tags_bits_list", "as": "minority_photo_bits_list"},
        {"name": "fountain_minority_photo_manjiao_markcode_tags", "as": "manjiao_markcode_tags"}
      ],
      import_item_attr = [
        "data_set_tags_bit",
        "manjiao_markcode" # 慢脚专项，详情咨询 @liuhao07 @yangliu03 https://docs.corp.kuaishou.com/d/home/fcACmJ5D5riM4G-Av4SEwRJz2
      ],
      export_item_attr = [
        "is_minority_photo",
      ],
      function_name = "IsMinorityPhotoV2",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def mc_low_cost_photo_adjust(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_mc_low_cost_photo_discount_coeff", "as": "boost_discount_coeff"},
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

  def mc_high_photo_count_author_adjust(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "high_upload_photo_author_map_ptr",
        {"name": "fountain_mc_high_photo_count_author_photo_coeff", "as": "boost_discount_coeff"},
        {"name": "fountain_mc_high_photo_count_author_pos_neg_ratio_coeff", "as": "pos_neg_ratio_coeff"},
      ],
      import_item_attr = [
        "author__id",
        {"name": "mc_adjust_coeff_final", "as": "ensemble_score"},
      ],
      export_item_attr = [
        {"name": "ensemble_score", "as": "mc_adjust_coeff_final"}
      ],
      function_name = "HighPhotoCountAuthorPhotoAdjustV2",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def cascade_comment_model_predict(self):
    self.explore_custom_trim_user_info(
      user_info_attr = "userInfo",
      save_trimed_user_info_to_attr = "cascade_comment_model_trimmed_user_info",
      trim_user_info = [
        "id",
        "active_days",
        "basic_info.age_segment",
        "device_id",
        "gender",
        "request_location.province_id",
        "request_location.city_id",
        "user_profile.exp_stat.exp_click",
        "user_profile.exp_stat.exp_like",
        "user_profile.exp_stat.exp_follow",
        "user_profile.exp_stat.exp_realshow",
        "user_profile.exp_stat.exp_long_view",
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
        "user_profile_v1.click_list.author_id",
        "user_profile_v1.click_list.photo_id",
        "user_profile_v1.click_list.page_type",
        "user_profile_v1.follow_list.author_id",
        "user_profile_v1.follow_list.photo_id",
        "user_profile_v1.like_list.author_id",
        "user_profile_v1.like_list.photo_id",
        "user_profile_v1.video_playing_stat.playing_time",
        "user_profile_v1.video_playing_stat.author_id",
        "user_profile_v1.video_playing_stat.photo_id",
        "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_one",
        "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_two",
        "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_three",
        "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_level_five",
        "user_profile_v1.real_show_list.hetu_tag_level_info.hetu_tag",
        "user_profile_v1.click_list.hetu_tag_level_info.hetu_level_one",
        "user_profile_v1.click_list.hetu_tag_level_info.hetu_level_two",
        "user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_one",
        "user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_two",
        "user_profile_v1.like_list.hetu_tag_level_info.hetu_level_one",
        "user_profile_v1.like_list.hetu_tag_level_info.hetu_level_two",
        "user_profile_v1.video_playing_stat.video_duration",
        "user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_one",
        "user_profile_v1.video_playing_stat.hetu_tag_level_info.hetu_level_two",
        "realtime_follow_list",
        "realtime_like_list",
        "upload_count",
        "infer_year",
        "follow_count",
        "fans_count",
        "visit_net",
        "location.city_level",
        "is_douyin",
        "user_profile_v1.real_show_list.photo_id",
        "user_profile_v1.real_show_list.author_id",
        "user_profile_v1.real_show_list.time_ms",
        "user_profile_v1.real_show_list.page_type",
        "user_profile_v1.real_show_list.label.click",
        "user_profile_v1.real_show_list.label.like",
        "user_profile_v1.real_show_list.label.follow",
        "user_profile_v1.real_show_list.label.hate",
        "feature_collection.explore_low_active_level",
        "user_interest_profile.hetu_level_one_long_term_id",
        "user_interest_profile.hetu_level_one_long_term_score",
        "user_interest_profile.hetu_level_two_long_term_id",
        "user_interest_profile.hetu_level_two_long_term_score",
        "user_interest_profile.hetu_level_three_long_term_id",
        "user_interest_profile.hetu_level_three_long_term_score",
      ],
    ) \
    .delegate_enrich(
      name = "fountain_cascade_comment_model",
      kess_service = "{{fountain_cascade_comment_model_service}}",
      send_common_attrs = [
        {"name": "cascade_comment_model_trimmed_user_info", "as": "user_info_str"},
      ],
      recv_item_attrs = [
        {"name": "watch_comment", "as": "cascading_watch_comment_score"},
        {"name": "comment_like", "as": "cascading_comment_like_score"},
        {"name": "comment_time", "as": "cascading_comment_time_score"},
        {"name": "is_valid_play", "as": "cascading_valid_play_score"},
      ],
      timeout_ms = 100,
    )
    return self

  def cascade_cal_emp_report_rate_score(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_vv_thres_for_emp_report_score", "as": "vv_thres"},
      ],
      import_item_attr = [
        "explore_stat__click_count",
        "explore_stat__report_count",
        "fountain_stats__real_show_count",
        "fountain_stats__report_count",
      ],
      export_item_attr = [
        "emp_report_score"
      ],
      function_name = "CalcEmpReportScore",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def mc_llm_negative_photo_adjust(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_mc_llm_negative_photo_adjust_tag_coeff_map_str", "as": "tag_coeff_map_str"},
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

  def mc_llm_negative_photo_personal_adjust(self):
    self.calc_by_formula1(
      kconf_key = "formula.scenarioKey34.McFountainLlmNeagtivePhotoDeboost",
      import_item_attr = [
        "hetu_info_for_llm_negative",
        "explore_stat__click_count",
        "explore_stat__report_count",
        "fountain_stats__real_show_count",
        "fountain_stats__report_count",
      ],
      import_common_attr = [
        "uToleranceScoreKV"
      ],
      export_formula_value = [
        {"name": "final_score", "as": "final_llm_personal_score"}
      ],
      abtest_biz_name = "KUAISHOU_APPS"
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "mc_adjust_coeff_final", "as": "score"},
        {"name": "final_llm_personal_score", "as": "boost_discount_coeff"},
      ],
      export_item_attr = [
        {"name": "score", "as": "mc_adjust_coeff_final"},
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2"
    )
    return self

  def fountain_cal_update_xtr_score_mc_s1(self):
    self.split_string(
      input_common_attr = "fountain_update_fix_xtr_weight_mc_s1_str",
      output_common_attr = "fountain_update_fix_xtr_weight_mc_s1_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "fountain_update_fix_xtr_power_mc_s1_str",
      output_common_attr = "fountain_update_fix_xtr_power_mc_s1_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "fountain_update_fix_xtr_buttom_mc_s1_str",
      output_common_attr = "fountain_update_fix_xtr_buttom_mc_s1_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "fountain_update_fix_xtr_upper_mc_s1_str",
      output_common_attr = "fountain_update_fix_xtr_upper_mc_s1_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .set_attr_value( 
      no_overwrite=True,
      common_attrs=[
        {
          "name": "fountain_mc_s1_update_xtr_name_list",
          "type": "string_list",
          "value": self.update_fix_xtr_name()
        }
      ]
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_update_fix_xtr_weight_mc_s1_list", "as": "update_fix_xtr_weight_list"},
        {"name": "fountain_update_fix_xtr_power_mc_s1_list", "as": "update_fix_xtr_power_list"},
        {"name": "fountain_update_fix_xtr_buttom_mc_s1_list", "as": "update_fix_xtr_buttom_list"},
        {"name": "fountain_update_fix_xtr_upper_mc_s1_list", "as": "update_fix_xtr_upper_list"},
        {"name": "fountain_update_window_width_mc_s1", "as": "window_width"},
        {"name": "fountain_mc_ensemble_s1_window_duration_ratio", "as": "window_duration_ratio"},
        {"name": "fountain_mc_s1_update_xtr_name_list", "as": "fix_xtr_list"},
      ],
      import_item_attr = [
        "upload_time",
        "duration_ms",
        "cascade_pctr",
        "cascade_psvtr",
        "cascade_plvtr",
        "cascade_pcotr",
        "cascade_pltr",
        "cascade_pwtr",
        "cascade_pcmtr",
        "cascade_pepstr",
        "cascade_pcestr",
        "cascade_pcltr",
        "cascade_pwtd",
        "cascade_pwatch_time"
      ],
      export_item_attr = [
        {"name": "update_bar_score", "as": "cascade_update_xtr_fix_mc_s1_score"}
      ],
      function_name = "FixWindowXtr",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  # 双列和关注页合作涨关粗排非首屏队列
  def _fountain_cal_rise_follow_boost_score_mc_s1(self):
    self.calc_by_formula1(
      kconf_key = "formula.scenarioKey24.CascadeFountainRiseFollowBoost",
      import_item_attr = [
        "duration_ms",
        "cascade_pltr",
        "cascade_pwtr",
        "cascade_pwtd",
        "cascade_longview_score",
        "cascade_pwatch_time",
        "cascade_pctr"
      ],
      export_formula_value = [
        "cascade_rise_follow_boost_score"
      ],
      abtest_biz_name = "KUAISHOU_APPS"
    )
    return self
  
  def _fountain_cal_rise_follow_boost_light_score_mc_s1(self):
    self.split_string(
      input_common_attr = "fountain_boost_follow_xtr_weight_mc_s1_str",
      output_common_attr = "fountain_boost_follow_xtr_weight_mc_s1_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "fountain_boost_follow_xtr_alpha_mc_s1_str",
      output_common_attr = "fountain_boost_follow_xtr_alpha_mc_s1_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .split_string(
      input_common_attr = "fountain_boost_follow_xtr_beta_mc_s1_str",
      output_common_attr = "fountain_boost_follow_xtr_beta_mc_s1_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .set_attr_value(
      no_overwrite=True,
      common_attrs=[
        {
          "name": "fountain_mc_s1_boost_follow_xtr_name_list",
          "type": "string_list",
          "value": self.boost_follow_xtr_name()
        }
      ]
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_cal_rise_pwtr_reshape_alpha", "as": "reshape_alpha"},
        {"name": "fountain_cal_rise_pwtr_reshape_max_value", "as": "reshape_max_value"},
      ],
      import_item_attr = [
        "cascade_pwtr",
      ],
      export_item_attr = [
        "cascade_reshape_pwtr",
      ],
      function_name = "CalSigmoidReshapeScore",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_boost_follow_xtr_weight_mc_s1_list", "as": "boost_follow_xtr_weight_list"},
        {"name": "fountain_boost_follow_xtr_alpha_mc_s1_list", "as": "boost_follow_xtr_alpha_list"},
        {"name": "fountain_boost_follow_xtr_beta_mc_s1_list", "as": "boost_follow_xtr_beta_list"},
        {"name": "fountain_mc_s1_boost_follow_xtr_name_list", "as": "boost_follow_xtr_list"},
      ],
      import_item_attr = [
        "cascade_reshape_pwtr",
        "cascade_pwtr",
        "cascade_pwtd",
        "cascade_longview_score",
        "cascade_pwatch_time",
        "cascade_pctr"
      ],
      export_item_attr = [
        {"name": "cascade_follow_score", "as": "cascade_rise_follow_boost_score"}
      ],
      function_name = "CalNewFollowBoostScore",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def boost_follow_xtr_name(self):
    update_fix_xtrs = [
      "cascade_reshape_pwtr",
      "cascade_pwtr",
      "cascade_pwtd",
      "cascade_longview_score",
      "cascade_pwatch_time",
      "cascade_pctr"
    ]
    return update_fix_xtrs

  def update_fix_xtr_name(self):
    update_fix_xtrs = [
      "cascade_pctr",
      "cascade_psvtr",
      "cascade_plvtr",
      "cascade_pcotr",
      "cascade_pltr",
      "cascade_pwtr",
      "cascade_pcmtr",
      "cascade_pepstr",
      "cascade_pcestr",
      "cascade_pcltr",
      "cascade_pwtd",
      "cascade_pwatch_time"
    ]
    return update_fix_xtrs
  
  def fountain_cal_hetu_retargeting_score(self):
    self \
    .set_attr_default_value(
      item_attrs=[
        {
          "name": "hetu_retargeting_score",
          "type": "double",
          "value": 1.0
        }
      ]
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "hetu_tag_level_info__hetu_level_two", "as": "extract_hetu_tag_list"},
      ],
      export_item_attr = [
        {"name": "first_hetu_tag", "as": "hetu_level_two_top1"},
      ],
      function_name = "ExtractFirstHetuTag",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .explore_interest_hetu_retargeting_enricher(
      colossus_v2_attr = "colossus_resp_v2",
      tag_name_attr = "hetu_level_two_top1",
      interest_hetu_enable_explore_page = "{{fountain_interest_hetu_enable_explore_page}}",
      interest_hetu_enable_fountain_page = "{{fountain_interest_hetu_enable_fountain_page}}",
      interest_hetu_enable_buttom_page = "{{fountain_interest_hetu_enable_buttom_page}}",
      interest_hetu_enable_other_page = "{{fountain_interest_hetu_enable_other_page}}",
      interest_hetu_stat_day_upper = "{{fountain_interest_hetu_stat_day_upper}}",
      interest_hetu_stat_day_lower = "{{fountain_interest_hetu_stat_day_lower}}",
      vv_num_upper_coeff = "{{fountain_vv_num_upper_coeff}}",
      valid_vv_rate_upper_coeff = "{{fountain_valid_vv_rate_upper_coeff}}",
      vv_num_lower_coeff ="{{fountain_vv_num_lower_coeff}}",
      valid_vv_rate_lower_coeff = "{{fountain_valid_vv_rate_lower_coeff}}",
      interest_hetu_alpha_coeff = "{{fountain_interest_hetu_alpha_coeff}}",
      interest_hetu_beta_coeff = "{{fountain_interest_hetu_beta_coeff}}",
      retarget_lower_vv_rate = "{{fountain_retarget_lower_vv_rate}}",
      retarget_upper_vv_rate = "{{fountain_retarget_upper_vv_rate}}",
      click_num_limit = "{{fountain_click_num_limit}}",
      output_attr = "hetu_retargeting_score"
    )
    return self
  
  def fountain_cal_sidinfo_retargeting_score(self):
    self \
    .enrich_attr_by_light_function(
      item_list_from_attr = "fountain_retatget_history_vv_list",
      import_common_attr = [
        {"name": "fountain_retatget_mode", "as": "retarget_mode"},
        {"name": "fountain_retatget_tagnet_upper", "as": "tagnet_upper"},
        {"name": "fountain_retatget_tagnet_lower", "as": "tagnet_lower"},
        {"name": "fountain_retatget_cal_vv_rate_size_lower", "as": "cal_vv_rate_size_lower"},
      ],
      import_item_attr = [
        "author__id",
        "hetu_tag_level_info__hetu_level_two",
        "hetu_sim_cluster_id",
        "user_hash_tag_id",
        "hetu_tag_level_info__hetu_tag"
      ],
      export_common_attr = [
        {"name": "show_sidinfo_list", "as": "fountain_retarget_show_sidinfo_list"},
        {"name": "vv_rate_sidinfo_list", "as": "fountain_retarget_vv_rate_sidinfo_list"},
      ],
      function_name = "CalHistoryVvRate",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      item_list_from_attr = "fountain_retarget_interest_colossus_trigger_list",
      import_common_attr = [
        {"name": "fountain_retarget_interest_colossus_trigger_weight_list", "as": "weight_list"},
        {"name": "fountain_retatget_mode", "as": "retarget_mode"},
        {"name": "fountain_retatget_index_power", "as": "retatget_index_power"},
        {"name": "fountain_retatget_tagnet_upper", "as": "tagnet_upper"},
        {"name": "fountain_retatget_tagnet_lower", "as": "tagnet_lower"},
        {"name": "fountain_retarget_show_sidinfo_list", "as": "show_list"},
        {"name": "fountain_retarget_vv_rate_sidinfo_list", "as": "vv_rate_list"},
        {"name": "fountain_retatget_vv_rate_limit", "as": "vv_rate_limit"},
        {"name": "fountain_retatget_vv_rate_avg", "as": "vv_rate_avg"},
        {"name": "fountain_retatget_vv_ratio_alpha", "as": "vv_ratio_alpha"},
        {"name": "fountain_retatget_vv_ratio_beta", "as": "vv_ratio_beta"},
      ],
      import_item_attr = [
        "author__id",
        "photo_id",
        "hetu_tag_level_info__hetu_level_two",
        "hetu_sim_cluster_id",
        "user_hash_tag_id",
        "hetu_tag_level_info__hetu_tag"
      ],
      export_common_attr = [
        {"name": "retarget_id_list", "as": "fountain_retarget_id_list"},
        {"name": "retarget_score_list", "as": "fountain_retarget_score_list"},
        {"name": "retarget_vv_rate_ratio_list", "as": "fountain_retarget_discount_score_list"},
      ],
      function_name = "GetReTargetScoreList",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_retarget_id_list", "as": "key_list"},
        {"name": "fountain_retarget_score_list", "as": "value_list"},
        {"name": "fountain_retarget_discount_score_list", "as": "discount_score_list"},
        {"name": "fountain_retatget_mode", "as": "retarget_mode"},
        {"name": "fountain_retatget_tagnet_upper", "as": "tagnet_upper"},
        {"name": "fountain_retatget_tagnet_lower", "as": "tagnet_lower"},
        {"name": "fountain_retatget_score_alpha", "as": "score_alpha"},
        {"name": "fountain_retatget_score_beta", "as": "score_beta"},
        {"name": "fountain_retatget_score_sigma", "as": "score_sigma"},
        {"name": "enable_fountain_retarget_score_range_limit", "as": "enbale_score_range_limit"},
        {"name": "fountain_retarget_score_lower_bound", "as": "score_lower_bound"},
        {"name": "fountain_retarget_score_upper_bound", "as": "score_upper_bound"},
      ],
      import_item_attr = [
        "author__id",
        "hetu_tag_level_info__hetu_level_two",
        "hetu_sim_cluster_id",
        "user_hash_tag_id",
        "hetu_tag_level_info__hetu_tag"
      ],
      export_item_attr = [
        "sidinfo_retargeting_score",
        "sidinfo_retargeting_discount_score",
      ],
      function_name = "CalcReTargetScore",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
  
  def gen_is_sexy_induce_photo(self):
    self \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "sexy_induce_photo_set_ptr", "as": "aid_set_ptr"}
      ],
      import_item_attr = [
        "author__id"
      ],
      export_item_attr = [
        {"name": "is_target_photo", "as": "is_sexy_induce_photo"}
      ],
      function_name = "AidInSet",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
  
  def mc_sexy_induce_adjust(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_mc_s2_sexy_induce_deboost_coeff", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "mc_adjust_coeff_final", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "mc_adjust_coeff_final"},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {"is_sexy_induce_photo": 1}
    )
    return self

  def mc_living_photo_adjust_by_paying_type(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "living_certain_aid_list", "as": "attr_list"},
      ],
      import_item_attr = [
        {"name": "author__id", "as": "attr"},
      ],
      export_item_attr = [
        {"name": "is_in_set", "as": "is_certain_ua"},
      ],
      function_name = "AttrIsInSet",
      class_name = "ExploreLightFunctionSetV2"
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "is_live_big_g_user", "as": "is_live_big_g_user"},
        {"name": "uUserKuaishouLivePayTag", "as": "user_live_paying_type"},
        {"name": "fountain_mc_living_photo_boost_coef_str", "as": "paying_user_boost_coef_str"},
        {"name": "fountain_mc_living_photo_boost_coef_big_g", "as": "boost_coef_big_g"},
      ],
      export_common_attr = [
        {"name": "living_boost_coef", "as": "mc_fountain_living_photo_coef"}
      ],
      function_name = "LivingCalcBoostCoef",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "mc_fountain_living_photo_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "mc_adjust_coeff_final", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "mc_adjust_coeff_final"},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {
        "is_true_living" : 1, "is_certain_ua" : 1
      }
    )
    return self
  
  def cal_fountain_photo_cluster_id_632(self):
    self.get_kconf_params(
      kconf_configs = [{
        "kconf_key": "reco.interestExplore.remapClusterId632",
        "value_type": "list_int64",
        "default_value": [],
        "export_common_attr": "remap_cluster_id_632_list"
      }]
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        "remap_cluster_id_632_list",
      ],
      import_item_attr = [
        "hetu_sim_cluster_id",
      ], 
      export_item_attr = [
        "cluster_id_632",
      ],
      function_name = "CalcClusterId632",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    
    return self
  
  def cal_is_in_set(self, input_set_name, default_value, item_flag, output_name):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": input_set_name, "as": "attr_list"},
        {"name": default_value, "as": "default_value"},
      ],
      import_item_attr = [
        {"name": item_flag, "as": "attr"}
      ],
      export_item_attr = [
        {"name": "is_in_set", "as": output_name}
      ],
      function_name = "AttrIsInSet",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
  
  def cal_fountain_import_explore_valid_interest_score(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        "uOldMmuClusterId300ListList",
        {"name": "uExploreShortValidInterestAndScoreList", "as": "user_valid_interest_cid_and_score_list"},
        {"name": "fountain_explore_cid_valid_num_threshold", "as": "valid_interest_cluster_id_num_threshold"},
        {"name": "fountain_explore_cid_valid_user_boost_alpha_coeff", "as": "valid_interest_user_boost_alpha_coeff"},
        {"name": "fountain_explore_cid_valid_user_boost_beta_coeff", "as": "valid_interest_user_boost_beta_coeff"},
        {"name": "fountain_explore_cid_valid_user_boost_omega_coeff", "as": "valid_interest_user_boost_omega_coeff"},
        {"name": "fountain_explore_cid_valid_score_lower_bound", "as": "develop_valid_interest_score_lower_bound"},
        {"name": "enable_fountain_add_explore_cid_valid_score_cids_boost", "as": "enable_valid_interest_score_cids_boost"},
      ],
      import_item_attr = [
        {"name": "cluster_id_632", "as": "hetu_sim_cluster_id862"},
      ],
      export_item_attr = [
        {"name": "valid_interest_cids_coeff", "as": "explore_valid_interest_score"},
      ],
      function_name = "CalValidInterestCidsCoeff",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
  
  def cal_fountain_import_gamora_interest_score(self):
    self.cal_is_in_set(
      input_set_name = "uExploreGamoraInterestList", default_value = "fountain_cluster_id_632_default_value",
      item_flag = "cluster_id_632", output_name = "gamora_interest_score"
    )
    return self

  def cal_is_import_explore_interest_user(self):
    self.set_attr_value(
      no_overwrite = True,
      common_attrs = [
        {
          "name": "fountain_final_view_limit",
          "type": "double",
          "value": 1.0
        }
      ]
    ) \
    .if_("enable_fountain_limit_explore_rate == 1") \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "colossus_channel_list",
          {"name": "fountain_min_explore_view_cnt", "as": "explore_min_explore_view_cnt"},
          {"name": "fountain_min_fountain_view_cnt", "as": "explore_min_fountain_view_cnt"},
          {"name": "fountain_ef_weight_alpha", "as": "explore_ef_weight_alpha"},
          {"name": "fountain_ef_weight_beta", "as": "explore_ef_weight_beta"},
          {"name": "fountain_ef_weight_min", "as": "explore_ef_weight_min"},
          {"name": "fountain_ef_weight_max", "as": "explore_ef_weight_max"}
        ],
        export_common_attr = [
          "explore_fountain_view_weight"
        ],
        function_name = "CalcExploreFountainViewWeight",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .gen_common_attr_by_lua(
        attr_map = {
          "fountain_final_view_limit": "fountain_lower_fountain_view_weight_limit - explore_fountain_view_weight",
        }
      ) \
    .end_() \
    .if_("fountain_final_view_limit >= 0.0") \
      .set_attr_value(
        no_overwrite = True,
        common_attrs = [
          {
            "name": "is_import_explore_interest_user",
            "type": "int",
            "value": 1
          }
        ]
      ) \
    .end_()
    return self

  def extract_hetu_info_tag_for_llm(self):
    self.split_string(
      input_common_attr = "fountain_tag_llm_negative_set_str",
      output_common_attr = "fountain_tag_llm_negative_set_str_list",
      delimiters = ",",
      trim_spaces = True,
      skip_empty_tokens = True,
      parse_to_int = True
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        "hetu_tag_level_info_v2__hetu_tag"
      ],
      import_common_attr = [
        {"name": "fountain_tag_llm_negative_set_str_list", "as": "tag_llm_negative_set_list"},
      ],
      export_item_attr = [
        {"name": "hetu_target_info_tag", "as": "hetu_info_for_llm_negative"}
      ],
      function_name = "ExtractHetuInfoTag",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def mc_sexy_induce_personal_adjust(self):
    self.calc_by_formula1(
      kconf_key = "formula.scenarioKey85.FountainSexyInduceDeboost",
      import_item_attr = [
      ],
      import_common_attr = [
        "uSexyInterestScore",
      ],
      export_formula_value = [
        {"name": "final_score", "as": "final_sexy_deboost_score","to_common": True}
      ],
      abtest_biz_name = "KUAISHOU_APPS"
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "final_sexy_deboost_score", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "mc_adjust_coeff_final", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "mc_adjust_coeff_final"},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {"is_sexy_induce_photo": 1}
    )
    return self

  def pack_fountain_mc_cascade(self):
    self.pack_item_attr(
      item_source = {
        "reco_results": True
      },
      mappings = [{
        "from_item_attr": "photo_id",
        "to_common_attr": "embedding_source_pids",
        "aggregator": "concat"
      }],
      target_item = {"is_marketing_compensation_photo": 1}
    )
    return self

  def pack_fountain_positive_trigger(self):
    self.explore_colossus_v2_trigger_enrich(
      colossus_resp_attr = "colossus_resp_v2",
      output_colossus_trigger_attr = "colossus_user_info_fountain_positive_photo_id_to_ecology_list",
      enable_default_select_triggers = "{{enable_fountain_default_select_triggers_to_ecology}}",
      enable_different_signals_triggers = "{{enable_fountain_positive_triggers_to_ecology}}",
      different_signals_triggers_select_num = "{{fountain_positive_triggers_select_num_to_ecology}}",
      different_signals_triggers_min_play_time = "{{fountain_positive_triggers_to_ecology_min_play_time}}",
      different_signals_triggers_play_time_ratio = "{{fountain_positive_triggers_to_ecology_play_time_ratio}}",
      different_signals_triggers_min_days_ago = "{{fountain_positive_triggers_to_ecology_min_days_ago}}",
      different_signals_triggers_max_days_ago = "{{fountain_positive_triggers_to_ecology_max_days_ago}}",
      enable_different_signals_triggers_action_explore_list = "{{enable_fountain_positive_triggers_to_ecology_action_explore_list}}",
      enable_different_signals_triggers_action_completion_list = "{{enable_fountain_positive_triggers_to_ecology_action_completion_list}}",
      enable_different_signals_triggers_action_interact_list = "{{enable_fountain_positive_triggers_to_ecology_action_interact_list}}",
      enable_different_signals_triggers_action_timestamp_order = "{{enable_fountain_positive_triggers_to_ecology_timestamp_order}}",
      enable_not_select_bottom_selection_page = "{{enable_fountain_positive_triggers_to_ecology_not_select_bottom_selection_page}}",
      enable_only_select_explore_colossus_list = "{{enable_fountain_positive_triggers_to_ecology_only_select_explore_colossus_list}}",
      enable_only_select_high_interest_tab = "{{enable_fountain_positive_triggers_to_ecology_only_select_high_interest_tab}}",
      enable_select_high_interest_and_profile_tab = "{{enable_fountain_positive_triggers_to_ecology_select_high_interest_and_profile_tab}}",
      enable_only_select_fountain_colossus_list =  "{{enable_fountain_positive_triggers_to_ecology_select_only_select_fountain_colossus_list}}",
      enable_only_unselect_explore_colossus_list =  "{{enable_fountain_positive_triggers_to_ecology_only_unselect_explore_colossus_list}}",
      enable_only_unselect_fountain_colossus_list =  "{{enable_fountain_positive_triggers_to_ecology_only_unselect_fountain_colossus_list}}",
      enable_get_longview_trigger = "{{enable_fountain_positive_get_longview_trigger_ecology}}",
    ) \
    .gen_common_attr_by_lua(
      attr_map={
        "colossus_user_info_fountain_positive_size": "#(colossus_user_info_fountain_positive_photo_id_to_ecology_list or {})",
      }
    ) \
    .pack_common_attr(
      input_common_attrs = [
        "embedding_source_pids",
        "colossus_user_info_fountain_positive_photo_id_to_ecology_list"
      ],
      output_common_attr = "embedding_source_pids",
      deduplicate = True
    )
    return self

  def get_fountain_trigger_embbedding(self):
    self.get_remote_embedding_lite(
      kess_service = "{{fountain_positive_triggers_to_ecology_score_kess}}",
      shard_num = 4,
      id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
      input_attr_name = "embedding_source_pids",
      output_attr_name = "mmu_embeddings",
      query_source_type = "common_attr",
      size = 64,
      client_side_shard = True
    )
    return self

  def cal_fountain_positive_triggers_to_ecology_score(self):
    self.explore_custom_embedding_score_enricher(
      enable_fix_low_hit_rate = "{{enable_fountain_marketing_fix_mmu_embedding_low_hit_rate}}",
      user_info_ptr_attr = "userInfoPb",
      embedding_list_attr = "mmu_embeddings",
      source_pids_list_attr = "embedding_source_pids",
      target_pids_list_attr = "colossus_user_info_fountain_positive_photo_id_to_ecology_list",
      calc_type = "list_similarity",
      dim_size = 64,
      export_item_attr = "fountain_ecology_positive_score",
      target_item = {"is_marketing_compensation_photo": 1}
    )
    return self
  
  def mc_bad_comment_pids_score_adjust(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_mc_bad_comment_pids_discount_coef", "as": "boost_discount_coeff"},
        {"name": "bad_comment_pids_ptr", "as": "boost_set"},
      ],
      import_item_attr = [
        {"name": "mc_adjust_coeff_final", "as": "boost_score"},
      ],
      export_item_attr = [
        {"name": "boost_score", "as": "mc_adjust_coeff_final"}
      ],
      function_name = "PidBoost",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def mc_sideinfo_retargeting_score_adjust(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "sidinfo_retargeting_score", "as": "boost_discount_coeff"},
        {"name": "mc_adjust_coeff_final", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "mc_adjust_coeff_final"}
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def mc_marketing_compensation_photo_personal_adjust(self):
    self.calc_by_formula1(
      kconf_key = "formula.scenarioKey42.McFountainMarketingPhotoDeboost",
      import_item_attr = [
        "fountain_ecology_positive_score"
      ],
      import_common_attr = [
        "colossus_user_info_fountain_positive_size"
      ],
      export_formula_value = [
        {"name": "final_score", "as": "final_mc_marketing_compensation_photo_score"}
      ],
      abtest_biz_name = "KUAISHOU_APPS",
      target_item = {"is_marketing_compensation_photo": 1}
    ) \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "final_mc_marketing_compensation_photo_score", "as": "boost_discount_coeff"},
        {"name": "mc_adjust_coeff_final", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "mc_adjust_coeff_final"},
      ],
      function_name = "BoostOrDiscountWithItemCoeff",
      class_name = "ExploreLightFunctionSetV2",
      target_item = {"is_marketing_compensation_photo": 1}
    )
    return self

  def _disable_forward_social_queue(self):
    self \
    .if_("fountain_cascade_disable_forward_social_queue_condition == 1 and (bid_follow_num == 0 or (u_inside_share_active_degree_detail_code > 3))") \
      .gen_common_attr_by_lua(
        attr_map={
          "fountain_variant_cluster_sort_weight_cascade_forward_score_social": "0.0",
        }
      ) \
    .end_() \
    .if_("fountain_cascade_disable_forward_social_queue_condition == 2 and (bid_follow_num == 0 or (u_share_num_30d == 0 and u_message_active_degree ~= 5 and u_message_active_degree ~= 6))") \
      .gen_common_attr_by_lua(
        attr_map={
          "fountain_variant_cluster_sort_weight_cascade_forward_score_social": "0.0",
        }
      ) \
    .end_() \
    .if_("fountain_cascade_disable_forward_social_queue_condition == 3 and (bid_follow_num == 0 or (user_msg_cnt_ssm_today + user_msg_cnt_gsm_today > 0))") \
      .gen_common_attr_by_lua(
        attr_map={
          "fountain_variant_cluster_sort_weight_cascade_forward_score_social": "0.0",
        }
      ) \
    .end_()
    return self

  def _disable_forward_dur_social_queue(self):
    self \
    .if_("fountain_cascade_disable_forward_dur_social_queue_condition == 1 and (bid_follow_num == 0 or (u_inside_share_active_degree_detail_code > 3))") \
      .gen_common_attr_by_lua(
        attr_map={
          "fountain_variant_cluster_sort_weight_cascade_forward_dur_score_social": "0.0",
        }
      ) \
    .end_() \
    .if_("fountain_cascade_disable_forward_dur_social_queue_condition == 2 and (bid_follow_num == 0 or (u_share_num_30d == 0 and u_message_active_degree ~= 5 and u_message_active_degree ~= 6))") \
      .gen_common_attr_by_lua(
        attr_map={
          "fountain_variant_cluster_sort_weight_cascade_forward_dur_score_social": "0.0",
        }
      ) \
    .end_() \
    .if_("fountain_cascade_disable_forward_dur_social_queue_condition == 3 and (bid_follow_num == 0 or (user_msg_cnt_ssm_today + user_msg_cnt_gsm_today > 0))") \
      .gen_common_attr_by_lua(
        attr_map = {
          "fountain_variant_cluster_sort_weight_cascade_forward_dur_score_social" : "0.0"
        }
      ) \
    .end_() \
    .split_string(
      input_common_attr = "fountain_cascade_pftr_dur_percentile_str",
      output_common_attr = "fountain_cascade_pftr_dur_percentile_list",
      delimiters = ",",
      parse_to_double = True
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_cascade_pftr_dur_percentile_list", "as": "percentile_list"},
        {"name": "fountain_cascade_pftr_dur_gama", "as": "gama"},
        {"name": "fountain_cascade_pftr_dur_threshold", "as": "threshold"}
      ],
      import_item_attr = [
        {"name": "duration_ms", "as": "duration"},
        {"name": "cascade_pftr", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "cascade_pftr_dur_social"},
      ],
      function_name = "CalculateCascadePftrDurScore",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
  
  # 调用单列评论小模型，模型负责人@白恩洋
  def _cascade_slide_comment_model_predict(self):
    self.pack_item_attr(
      item_source = {
        "reco_results": True,
      },
      mappings = [
        {
          "from_item_attr": "photo_id",
          "to_common_attr": "slide_comment_model_pids",
          "aggregator": "concat",
        },
      ]
    ) \
    .pack_item_attr(
      item_source = {
        "reco_results": True,
      },
      mappings = [
        {
          "from_item_attr": "author__id",
          "to_common_attr": "slide_comment_model_aids",
          "aggregator": "concat",
        },
      ]
    ) \
    .delegate_enrich(
      kess_service = "{{fountain_cascade_slide_comment_model_service}}",
      partition_size = "{{fountain_cascade_slide_comment_model_request_num}}",
      send_common_attrs = [
        {"name": "userInfo", "as": "user_info_str"},
        {"name": "fountain_cascade_slide_comment_model_request_tab_id", "as": "tab_id"},
        {"name": "slide_comment_model_pids", "as": "photo_ids"},
        {"name": "slide_comment_model_aids", "as": "author_ids"}
      ],
      recv_item_attrs = [
        {"name": "comment_stay_time", "as": "slide_comment_stay_time"},
        {"name": "effective_read_comment", "as": "slide_effective_read_comment"},
        {"name": "comment_consume_depth", "as": "slide_comment_consume_depth"}
      ],
      request_type = "{{fountain_cascade_slide_comment_model_request_type}}",
      timeout_ms = 50,
      infer_output_type = 2,
    )
    return self

  def _cascade_touch_high_follow_adjust(self):
    """
    关注和双列合作涨关注摸高实验
    """
    self \
    .if_("enable_fountain_cascade_touch_high_follow_adjust == 1") \
      .gen_common_attr_by_lua(
        attr_map={
          "fountain_variant_cluster_sort_weight_cascade_follow_score": "fountain_variant_cluster_sort_weight_cascade_follow_score * fountain_variant_cluster_sort_weight_cascade_follow_score_rise_follow_adjust_coeff"
        }
      ) \
    .end_if_()
    return self

  def _cascade_interact_playtime_adjust(self):
    """
    低互动人群提权时长队列
    """
    self \
    .if_("enable_fountain_cascade_playtime_adjust == 1 and user_is_low_interact == 1") \
      .gen_common_attr_by_lua(
        attr_map={
          "fountain_fast_cascade_variant_cluster_sort_pwatch_time_raw_weight": "fountain_fast_cascade_variant_cluster_sort_pwatch_time_raw_weight * fountain_cascade_variant_cluster_sort_pwatch_time_raw_weight_adjust_coeff",
          "fountain_fast_cascade_variant_cluster_sort_longview_raw_weight": "fountain_fast_cascade_variant_cluster_sort_longview_raw_weight * fountain_cascade_variant_cluster_sort_longview_raw_weight_adjust_coeff",
          "fountain_fast_cascade_variant_cluster_sort_shortview_raw_weight": "fountain_fast_cascade_variant_cluster_sort_shortview_raw_weight * fountain_cascade_variant_cluster_sort_shortview_raw_weight_adjust_coeff",
          "fountain_fast_cascade_variant_cluster_sort_click_raw_weight": "fountain_fast_cascade_variant_cluster_sort_click_raw_weight * fountain_cascade_variant_cluster_sort_click_raw_weight_adjust_coeff",
          "fountain_variant_cluster_sort_weight_cascade_pwatch_time": "fountain_variant_cluster_sort_weight_cascade_pwatch_time * fountain_variant_cluster_sort_weight_cascade_pwatch_time_adjust_coeff",
          "fountain_variant_cluster_sort_weight_cascade_longview_score": "fountain_variant_cluster_sort_weight_cascade_longview_score * fountain_variant_cluster_sort_weight_cascade_longview_score_adjust_coeff",
          "fountain_variant_cluster_sort_weight_cascade_shortview_score": "fountain_variant_cluster_sort_weight_cascade_shortview_score * fountain_variant_cluster_sort_weight_cascade_shortview_score_adjust_coeff",
          "fountain_variant_cluster_sort_weight_cascade_click_score": "fountain_variant_cluster_sort_weight_cascade_click_score * fountain_variant_cluster_sort_weight_cascade_click_score_adjust_coeff"
        }
      ) \
    .end_if_()
    return self
  
  def cascade_batch_similar_model_predict(self):
    self.set_attr_default_value(
      item_attrs = [
        {
          "name": "explore_fountain_cascade_batch_similar_pc12h",
          "type": "int",
          "value": 6000
        }
      ],
    ) \
    .delegate_enrich(
      kess_service = "{{fountain_cascade_dynamic_i2i_kess_service}}",
      recv_item_attrs=[
        {"name": "batch_similar_score_attr", "as": "batch_similar_score"},
      ],
      timeout_ms = 100,
      send_item_attrs=[
        {"name": "explore_fountain_cascade_batch_similar_pc12h", "as": "vv_cnt_g"},
        {"name": "explore_fountain_cascade_batch_similar_pc12h", "as": "vv_cnt_n"},
      ],
      send_common_attrs = [
        { "name": "fountain_retarget_interest_colossus_trigger_list", "as": "pid_list"},
      ],
      request_type="{{fountain_cascade_i2i_dynamic_request_type}}",
      partition_size = "{{fountain_cascade_i2i_dynamic_partition_size}}",
      use_packed_item_attr = True,
      infer_output_type = 2
    )
    return self

  def fountain_cal_dynamic_i2i_score_mc_s1(self):
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "batch_similar_score", "as": "batch_score"},
        {"name": "sidinfo_retargeting_discount_score", "as": "retargeting_score"},
      ],
      import_common_attr = [
        {"name": "fountain_cascade_dynamic_i2i_score_switch", "as": "cal_method"},
        {"name": "fountain_cascade_dynamic_i2i_score_threshold", "as": "score_threshold"},
      ],
      export_item_attr = [
        {"name": "max_score", "as": "i2i_dynamic_score"},
      ],
      function_name = "CalI2IDynamicScore",
      class_name = "ExploreLightFunctionSetV2"
    )
    return self

  def fountain_cal_user_interest_tagnex_tgi(self, interest_type, case1_attr, case2_attr):
    version_attr = f"fountain_user_{interest_type}_interest_tagnex_tgi_version"
    prefix_attr = f"fountain_user_{interest_type}_interest_tagnex_tgi_prefix"
    list_attr = f"fountain_user_{interest_type}_interest_tagnex_tgi_list"
    coeff_attr = f"fountain_user_{interest_type}_interest_tagnex_tgi_coeff"
    bias_attr = f"fountain_user_{interest_type}_interest_tagnex_tgi_bias"
    score_attr = f"fountain_user_{interest_type}_interest_tagnex_tgi_score"
    attr_min = f"fountain_user_{interest_type}_interest_tagnex_circle_attr_min"
    attr_max = f"fountain_user_{interest_type}_interest_tagnex_circle_attr_max"
    use_single_match_item_attr = f"fountain_user_{interest_type}_interest_use_single_match_item"
    self.switch_(version_attr) \
      .case_(1) \
        .str_format(
          format_string = "%s_%d",
          input_attrs = [prefix_attr, case1_attr],
          output_attr = f"user_{interest_type}_interest_tagnex_tgi_key",
        ) \
      .case_(2) \
        .str_format(
          format_string = "%s_%d",
          input_attrs = [prefix_attr, case2_attr],
          output_attr = f"user_{interest_type}_interest_tagnex_tgi_key",
        ) \
    .end_() \
    .get_kconf_params(
      kconf_configs = [{
        "kconf_key": "reco.offline.fountainuserGroupandCareerInterestTagnexTgiStat",
        "json_path": "{{" + f"user_{interest_type}_interest_tagnex_tgi_key" + "}}",
        "value_type": "list_int64",
        "default_value": [],
        "export_common_attr": list_attr
      }]
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": list_attr, "as": "match_list"},
        {"name": coeff_attr, "as": "coeff"},
        {"name": bias_attr, "as": "bias"},
        {"name": attr_min, "as": "attr_min"},
        {"name": attr_max, "as": "attr_max"},
        {"name": use_single_match_item_attr, "as": "use_single_match_item"},
      ],
      import_item_attr = [
        {"name" : "hetu_tag_level_info__hetu_tag", "as" : "hetu_tag"},
        {"name" : "cluster_id_632", "as" : "cluster_id_632"}
      ],
      export_item_attr = [
        {"name": "match_score", "as": score_attr}
      ],
      function_name = "CalMatchScore",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def mc_interest_tagnex_tgi_adjust(self, interest_type):
    score_attr = f"fountain_user_{interest_type}_interest_tagnex_tgi_score"
    self.enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "mc_adjust_coeff_final", "as": "score"},
        {"name": score_attr, "as": "boost_discount_coeff"},
      ],
      export_item_attr = [
        {"name": "score", "as": "mc_adjust_coeff_final"},
      ],
      function_name = "BoostOrDiscountByItemCoeff",
      class_name = "ExploreLightFunctionSetV2"
    )
    return self

  def gen_is_reason_top_photo(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_reason_top_photo_white_list", "as": "reason_white_list"},
        {"name": "fountain_reason_top_photo_top_k", "as": "top_k"},
      ],
      export_item_attr = [
        {"name": "is_reason_top_photo", "as": "is_top_reason_topk_boost_photo"},
      ],
      function_name = "CalIsReasonTopPhoto",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
  
  def _cascade_life_stage_cid_ipw_debias(self):
    """
    人生阶段 x cid ipw纠偏
    """
    self \
    .if_("enable_fountain_cascade_life_stage_cid_ipw_debias == 1") \
      .explore_memory_data_enrich(
        data_key = "{{fountain_cascade_life_stage_cid_ipw_map}}",
        data_type = "string_double_map",
        save_data_ptr_to_attr = "life_stage_cid_ipw_map_ptr",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "uStudentLabelV1KV",
          "uBirthLabelV1KV",
          "uMarriageLabelV1KV",
          "life_stage_cid_ipw_map_ptr",
          {"name": "fountain_cascade_life_stage_cid_ipw_redis_prefix", "as": "prefix"},
          {"name": "fountain_cascade_life_stage_cid_ipw_debias_upper_bound", "as": "upper_bound"},
          {"name": "fountain_cascade_life_stage_cid_ipw_debias_alpha", "as": "alpha"},
          {"name": "fountain_cascade_life_stage_cid_ipw_debias_beta", "as": "beta"},
        ],
        import_item_attr = [
          "cluster_id_632",
          {"name": "cascade_plvtr", "as": "plvtr"},
        ],
        export_item_attr = [
          {"name": "debias_score", "as": "cascade_life_stage_cid_ipw_debias_plvtr"},
        ],
        function_name = "CalLifeStageCidIpwDebias",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_if_()
    return self

  def gen_is_reason_top_photo_modify_day0(self):
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fountain_reason_top_photo_white_list", "as": "reason_white_list"},
        {"name": "fountain_reason_top_photo_top_k", "as": "top_k"},
        {"name": "fountain_reason_top_photo_prob_day_1_2", "as": "prob_day_1_2"},
        {"name": "fountain_reason_top_photo_prob_day_2_7", "as": "prob_day_2_7"},
        {"name": "fountain_reason_top_photo_prob_day_8_30", "as": "prob_day_8_30"},
      ],
      import_item_attr = [
        "upload_time",
      ],
      export_item_attr = [
        {"name": "is_reason_top_photo", "as": "is_top_reason_topk_boost_photo"},
      ],
      function_name = "CalIsReasonTopPhotoModifyDay0",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def _cascade_age_gender_prof_cid_ipw_debias(self):
    """
    年龄 x 性别 x 职业一级 x cid ipw纠偏
    """
    self \
    .if_("enable_fountain_cascade_age_gender_prof_cid_ipw_debias == 1") \
      .explore_memory_data_enrich(
        data_key = "{{fountain_cascade_age_gender_prof_cid_ipw_map}}",
        data_type = "string_double_map",
        save_data_ptr_to_attr = "age_gender_prof_cid_ipw_map_ptr",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "basic_info_age_segment_v2",
          "basic_info_gender_v2",
          "uJobIdLv1KV",
          "age_gender_prof_cid_ipw_map_ptr",
          {"name": "fountain_cascade_age_gender_prof_cid_ipw_redis_prefix", "as": "prefix"},
          {"name": "fountain_cascade_age_gender_prof_cid_ipw_debias_upper_bound", "as": "upper_bound"},
          {"name": "fountain_cascade_age_gender_prof_cid_ipw_debias_alpha", "as": "alpha"},
          {"name": "fountain_cascade_age_gender_prof_cid_ipw_debias_beta", "as": "beta"},
        ],
        import_item_attr = [
          "cluster_id_632",
          {"name": "cascade_plvtr", "as": "plvtr"},
        ],
        export_item_attr = [
          {"name": "debias_score", "as": "cascade_age_gender_prof_cid_ipw_debias_plvtr"},
        ],
        function_name = "CalAgeGenderProfCidIpwDebias",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_if_()
    return self

  def _cascade_age_gender_north_cid_ipw_debias(self):
    """
    年龄 x 性别 x 南北方 x cid ipw纠偏
    """
    self \
    .if_("enable_fountain_cascade_age_gender_north_cid_ipw_debias == 1") \
      .explore_memory_data_enrich(
        data_key = "{{fountain_cascade_age_gender_north_cid_ipw_map}}",
        data_type = "string_double_map",
        save_data_ptr_to_attr = "age_gender_north_cid_ipw_map_ptr",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "basic_info_age_segment_v2",
          "basic_info_gender_v2",
          "uIsNorthKV",
          "age_gender_north_cid_ipw_map_ptr",
          {"name": "fountain_cascade_age_gender_north_cid_ipw_redis_prefix", "as": "prefix"},
          {"name": "fountain_cascade_age_gender_north_cid_ipw_debias_upper_bound", "as": "upper_bound"},
          {"name": "fountain_cascade_age_gender_north_cid_ipw_debias_alpha", "as": "alpha"},
          {"name": "fountain_cascade_age_gender_north_cid_ipw_debias_beta", "as": "beta"},
        ],
        import_item_attr = [
          "cluster_id_632",
          {"name": "cascade_plvtr", "as": "plvtr"},
        ],
        export_item_attr = [
          {"name": "debias_score", "as": "cascade_age_gender_north_cid_ipw_debias_plvtr"},
        ],
        function_name = "CalAgeGenderNorthCidIpwDebias",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_if_()
    return self

  def _cascade_age_gender_cid_ipw_debias(self):
    """
    年龄 x 性别 x 南北方 x cid ipw纠偏
    """
    self \
    .if_("enable_fountain_cascade_age_gender_cid_ipw_debias == 1") \
      .explore_memory_data_enrich(
        data_key = "{{fountain_cascade_age_gender_cid_ipw_map}}",
        data_type = "string_double_map",
        save_data_ptr_to_attr = "age_gender_cid_ipw_map_ptr",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "basic_info_age_segment_v2",
          "basic_info_gender_v2",
          "age_gender_cid_ipw_map_ptr",
          {"name": "fountain_cascade_age_gender_cid_ipw_redis_prefix", "as": "prefix"},
          {"name": "fountain_cascade_age_gender_cid_ipw_debias_upper_bound", "as": "upper_bound"},
          {"name": "fountain_cascade_age_gender_cid_ipw_debias_alpha", "as": "alpha"},
          {"name": "fountain_cascade_age_gender_cid_ipw_debias_beta", "as": "beta"},
        ],
        import_item_attr = [
          "cluster_id_632",
          {"name": "cascade_plvtr", "as": "plvtr"},
        ],
        export_item_attr = [
          {"name": "debias_score", "as": "cascade_age_gender_cid_ipw_debias_plvtr"},
        ],
        function_name = "CalAgeGenderCidIpwDebias",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_if_()
    return self