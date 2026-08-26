from cascading import CommonModule
from cascading.module.fountain_cascading_utils import cascade_ltr_common_feature, cascade_fc_feature, cascade_fc_sim3_feature, cascade_slide_features, cascade_full_link_distill_user_features

class FountainFastCascadingPredictModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
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
    .if_("enable_fc_fountain_interface == 1") \
      .if_("enable_new_fc_feature == 1") \
        .extract_with_ks_sign_feature(
          feature_list = cascade_fc_sim3_feature,
          update_ks_sign_feature_type = 1,
          update_interval_sec = 600,
          user_info_attr = "user_info_ptr",
          common_slots_output = "user_feature_slots",
          common_parameters_output = "user_feature_signs",
        ) \
      .else_() \
        .if_("enable_fc_feature_kconf == 1") \
          .extract_with_ks_sign_feature(
            extractor_kconf_path = "reco.hot.fountainLeafMcFeature",
            caller_model = "{{fountain_cascade_fc_predict_service}}",
            feature_list = cascade_fc_sim3_feature,
            update_ks_sign_feature_type = 1,
            update_interval_sec = 600,
            user_info_attr = "user_info_ptr",
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
        .end_if_()\
      .end_() \
      .if_("enable_fountain_fc_extract_photo_signs == 1") \
        .delegate_enrich(
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
          ],
          use_item_id_in_attr = "item_id",
          use_packed_item_attr = True,
        ) \
      .else_()\
        .delegate_enrich(
          # 内流全连接粗排模型预估
          kess_service = "{{fountain_cascade_fc_predict_service}}",
          request_type = "{{fountain_cascade_fc_request_type}}",
          timeout_ms = 100,
          send_common_attrs = ["user_feature_slots", "user_feature_signs"],
          # send_item_attrs = ["item_slots", "item_parameters"], #note
          recv_item_attrs = [
            {"name":"fc_pctr_value", "as":"cascade_fc_pctr"},
            {"name":"fc_plvr_value", "as":"cascade_fc_plvtr"},
            {"name":"fc_psvr_value", "as":"cascade_fc_psvtr"},
            {"name":"fc_pvtr_value", "as":"cascade_fc_pvtr"},
          ],
          use_item_id_in_attr = "item_id",
          use_packed_item_attr = True,
        ) \
      .end_()\
    .end_if_() \
    .filter_by_attr(
      attr_name = "cascade_phtr",
      remove_if = ">",
      compare_to = "{{fountain_cascade_phtr_filter_limit}}",
      remove_if_attr_missing = False,
      skip = "{{skip_fountain_cascade_phtr_filter}}") \
    .if_("enable_fc_after_prerank_fast == 1") \
      .extract_with_ks_sign_feature(
        extractor_kconf_path = "reco.hot.fountainLeafMcFeature",
        caller_model = "{{fountain_cascade_fc_predict_service}}",
        feature_list = cascade_fc_sim3_feature,
        update_ks_sign_feature_type = 1,
        update_interval_sec = 600,
        user_info_attr = "user_info_ptr",
        common_slots_output = "user_feature_slots",
        common_parameters_output = "user_feature_signs",
      ) \
      .delegate_enrich(
        kess_service = "{{fountain_cascade_fc_predict_service}}",
        request_type = "{{fountain_cascade_fc_request_type}}",
        timeout_ms = 100,
        send_common_attrs = ["user_feature_slots", "user_feature_signs"],
        recv_item_attrs = [
          {"name":"fc_pctr_value", "as":"cascade_fc_pctr"},
          {"name":"fc_plvr_value", "as":"cascade_fc_plvtr"},
          {"name":"fc_psvr_value", "as":"cascade_fc_psvtr"},
          {"name":"fc_pvtr_value", "as":"cascade_fc_pvtr"},
        ],
        use_item_id_in_attr = "item_id",
        use_packed_item_attr = True,
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          "cascade_fc_pctr", "cascade_fc_plvtr", "cascade_fc_psvtr", "cascade_fc_pvtr",
        ],
        export_item_attr = ["cascade_pctr", "cascade_plvtr", "cascade_psvtr", "cascade_pwatch_time"],
        function_name = "ReplaceMcPxtr",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .delegate_enrich(
          kess_service = "{{fountain_mc_integrated_tower_predict_kess_service}}",
          recv_item_attrs = [
            {"name":"wtd", "as":"cascade_wtd_kai"},
            {"name":"act", "as":"cascade_act_kai"},
            {"name":"click_comment_button", "as":"cascade_click_comment_button"},
            {"name":"slide", "as":"cascade_slide_kai"},
            {"name":"finish_rate", "as":"cascade_ftr_kai"}
            ],
          timeout_ms = 100,
          send_item_attrs = ["item_id"],
          send_common_attrs = cascade_slide_features,
          request_type = "default",
        ) \
      .fountain_enrich_cascade_score( #enrich cascade_score by fc_pxtr
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
      .log_debug_info(
        for_debug_request_only=True,
        item_attrs=[
          "cascade_fc_psvtr",
          "cascade_psvtr"
          ],
        common_attrs=["enable_fc_after_prerank_fast"]
      ) \
    .end_() \
    .if_("enable_fountain_life_mc_distill_model == 1") \
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
          "upload_count"
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
      ) \
    .end_()
      