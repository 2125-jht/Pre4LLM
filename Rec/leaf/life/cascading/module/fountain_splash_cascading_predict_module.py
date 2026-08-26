from cascading import CommonModule
from cascading.module.fountain_cascading_utils import cascade_ltr_common_feature, cascade_fc_sim3_feature, cascade_slide_features


class FountainSplashCascadingPredictModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .copy_item_meta_info(
        save_item_id_to_attr="item_id"  # 在photoInfoModule里已产出
      ) \
      .transform_item_attr(
        mappings=[{
          "check_attr_name": "upload_type",
          "check_attr_type": "int",
          "output_attr_name": "picture_variant_attr",
          "output_attr_type": "int",
          "rules": [{
            "check_values": [10, 11],
            "output_value": 1
          }]
        }]
      ) \
      .transform_item_attr(
        mappings=[
          {
            "check_attr_name": "author__id",
            "check_attr_type": "int",
            "output_attr_name": "is_photo_author_followed",
            "output_attr_type": "int",
            # 检查规则
            "rules": [{
              # 当 author__id 在 followAuthors 内时
              "check_values": ["{{followAuthors}}"],
              "output_value": 1,
            }]
          },
        ]
      ) \
      .set_attr_value(
        item_attrs=[
          {
            "name": "is_follow_author",  # 关注作者建议使用此字段
            "type": "int",
            "value": 1
          }
        ],
        target_item={
          "is_photo_author_followed": 1
        },
      ) \
      .enrich_attr_by_lua(
        import_common_attr=[
          "common_request_type"
        ],
        export_common_attr=[
          "fountain_casade_is_fast",
        ],
        function_for_common="cascade_control_model",
        lua_script_file="./life/cascading/lua/module/fountain_splash_cascading_predict__control.lua",
      ) \
      .if_("skip_fountain_cascade_wtd_kai_predict == 0 ") \
        .if_("fountain_cascade_wtd_kai_predict_all == 1 or fountain_casade_is_fast == 1") \
          .delegate_enrich(
            kess_service="{{fountain_cascade_wtd_predict_kai_kess_service}}",
            recv_item_attrs=[
              {"name": "wtd", "as": "cascade_wtd_kai"},
            ],
            timeout_ms=200,
            send_item_attrs=["item_id"],
            send_common_attrs=cascade_ltr_common_feature,
            request_type="default",
          ) \
        .end_if_() \
      .end_if_() \
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
      .delegate_enrich(
        # 内流粗排预估新接口
        skip = "{{skip_fountain_cascade_new_interface_predict}}",
        kess_service = "{{fountain_cascade_new_arch_predict_kess_service}}",
        request_type = "{{fountain_cascade_new_arch_tower_request_type}}",
        timeout_ms = 100,
        send_common_attrs =  [
          { "name": "userInfo", "as": "user_info_str" }
        ],
        recv_item_attrs = [
          { "name": "ctr", "as": "cascade_pctr" },
          { "name": "lvr", "as": "cascade_plvtr" },
          { "name": "svr", "as": "cascade_psvtr" },
          { "name": "ptr", "as": "cascade_ptr" },
          { "name": "ltr", "as": "cascade_pltr" },
          { "name": "wtr", "as": "cascade_pwtr" },
          { "name": "ftr", "as": "cascade_pftr" },
          { "name": "vtr", "as": "cascade_pwatch_time" },
          { "name": "wtd", "as": "cascade_pwtd"},
          { "name": "cmtr", "as": "cascade_pcmtr" },
          { "name": "ces", "as": "cascade_pcestr" },
          { "name": "eps", "as": "cascade_pepstr" },
          { "name": "cltr", "as": "cascade_pcltr" },
          { "name": "htr", "as": "cascade_phtr"},
          { "name": "cotr", "as": "cascade_pcotr"},
          { "name": "swptr", "as": "cascade_pswptr"},
          { "name": "swp_after", "as": "cascade_pswptr_after"},
          { "name": "lstr", "as": "cascade_plstr"},
          { "name": "lsst", "as": "cascade_plsst"},
          { "name": "pair_evtr", "as": "cascade_pair_evtr"},
          { "name": "pair_lvtr", "as": "cascade_pair_lvtr"},
        ],
        for_predict = True,
        use_packed_item_attr = True,
        infer_output_type = 2,
        use_item_id_in_attr = "item_id",
      ) \
      .if_("fountain_cascade_ftr_slide_kai_predict_all == 1 or fountain_casade_is_fast == 1") \
        .delegate_enrich(
          kess_service="{{fountain_cascade_ftr_slide_predict_kai_kess_service}}",
          recv_item_attrs=[
            {"name": "slide", "as": "cascade_slide_kai"},
            {"name": "ftr", "as": "cascade_ftr_kai"}
          ],
          timeout_ms=100,
          send_item_attrs=["item_id"],
          send_common_attrs=cascade_slide_features,
          request_type="default",
        ) \
      .end_if_() \
      .fountain_enrich_cascade_score(
        pwatch_time_attr="cascade_pwatch_time",
        pptr_attr="cascade_ptr",
        pepstr_attr="cascade_pepstr",
        pcestr_attr="cascade_pcestr",
        pcmtr_attr="cascade_pcmtr",
        pwtd_attr="cascade_pwtd",
        pslide_attr="cascade_slide_kai",
        svtr_coeff="{{fountain_cascade_svtr_coeff}}",
        svtr_power="{{fountain_cascade_svtr_power}}",
        short_play_discount_value="{{fountain_cascade_short_play_discount_value}}",
        lvtr_use_predict_watch_time="{{fountain_cascade_ensemble_lvtr_use_predict_watch_time}}",
        mid_photo_boost_coeff="{{fountain_cascade_mid_photo_boost_coeff}}",
      ) \
      .if_("fountain_skip_calc_cascade_ftr_duration == 0") \
        .enrich_attr_by_lua(  # 结合 duration 的纠偏
          import_common_attr=[
            "cascade_ftr_kai_duration_max",
            "cascade_ftr_kai_duration_weight",
            "cascade_ftr_kai_duration_min",
            "cascade_ftr_kelly_avg_duration",
            "cascade_ftr_kai_enable_transfer_1",
            "cascade_ftr_kai_enable_transfer_2",
            "cascade_ftr_kai_duration_ftr_power",
            "cascade_ftr_kai_duration_ftr_offset",
          ],
          import_item_attr=[
            "duration_ms",
            "cascade_ftr_kai"
          ],
          export_item_attr=[
            "cascade_ftr_kai_duration",
            "cascade_ftr_kai_kelly"
          ],
          function_for_item="cascade_ftr_duration",
          lua_script_file="./life/cascading/lua/module/fountain_splash_cascading_predict__calc_pxtr.lua",
        ) \
      .end_if_() \
      .if_("cascade_ftr_kai_use_ipw_weight_new == 1") \
        .enrich_attr_by_lua(
          import_common_attr=[
            "enable_opt_cascade_ftr_ipw_bucket",
            "ftr_redis_key_opt_prefix"
          ],
          import_item_attr=[
            "duration_ms",
            "cascade_ftr_kai",
          ],
          export_item_attr=[
            "cascade_ftr_kai_redis_key"
          ],
          function_for_item="cascade_ftr_redis_key",
          lua_script_file="./life/cascading/lua/module/fountain_splash_cascading_predict__calc_pxtr.lua",
        ) \
        .get_item_attr_from_redis(
          cluster_name="recoNewUserPhotos",
          timeout_ms=50,
          cache_bits=16,
          redis_key_from="cascade_ftr_kai_redis_key",
          save_value_to="cascade_ftr_kai_ipw_value",
        ) \
        .enrich_attr_by_lua(
          import_common_attr=[
            "cascade_ftr_ipw_debias_v1",
            "cascade_ftr_ipw_debias_v2",
            "cascade_ftr_ipw_debias_v3",
            "cascade_ftr_kai_ipw_value_default",
            "cascade_ftr_ipw_debias_ftr_alpha",
            "cascade_ftr_ipw_debias_ftr_factor",
            "cascade_ftr_ipw_debias_ftr_beta",
            "cascade_ftr_ipw_debias_pct_beta"
          ],
          import_item_attr=[
            "cascade_ftr_kai",
            "cascade_ftr_kai_ipw_value"
          ],
          export_item_attr=[
            "cascade_ipw_opt_ftr",
          ],
          function_for_item="cascade_ftr_ipw_debias",
          lua_script_file="./life/cascading/lua/module/fountain_splash_cascading_predict__calc_pxtr.lua",
        ) \
      .end_if_() \
      .if_("enable_fountain_fc_extract_photo_signs == 1") \
        .extract_with_ks_sign_feature(
          extractor_kconf_path="reco.hot.fountainLeafMcFeature",
          caller_model="{{fountain_cascade_fc_predict_service}}",
          feature_list=cascade_fc_sim3_feature,
          update_ks_sign_feature_type=1,
          update_interval_sec=600,
          user_info_attr="user_info_ptr",
          common_slots_output="user_feature_slots",
          common_parameters_output="user_feature_signs",
        ) \
        .enrich_attr_by_light_function(
          import_item_attr=[
            "photo_id", "author__id", "tag", "duration_ms", "upload_time",
          ],
          export_item_attr=[{"name": "context_slots", "as": "fountain_fc_car_slots"},
                            {"name": "context_signs", "as": "fountain_fc_car_signs"}],
          function_name="GenCARSigns",
          class_name="ExploreLightFunctionSetV2",
        ) \
        .delegate_enrich(
          kess_service="{{fountain_cascade_fc_predict_service}}",
          request_type="{{fountain_cascade_fc_request_type}}",
          timeout_ms=100,
          send_common_attrs=["user_feature_slots", "user_feature_signs"],
          send_item_attrs=["fountain_fc_car_slots", "fountain_fc_car_signs"],
          recv_item_attrs=[
            {"name": "fc_pvtr2_value", "as": "cascade_fc_pvtr2"},
          ],
          use_item_id_in_attr="item_id",
          use_packed_item_attr=True,
        ) \
      .end_() \
      .if_("skip_fountain_cascade_wtd_act_kai_predict == 0 and skip_fountain_cascade_wtd_kai_predict == 1") \
        .if_("fountain_cascade_wtd_act_kai_predict_all == 1 or fountain_casade_is_fast == 1") \
          .delegate_enrich(
            kess_service="{{fountain_cascade_wtd_act_predict_kai_kess_service}}",
            recv_item_attrs=[
              {"name": "wtd", "as": "cascade_wtd_kai"},
              {"name": "act", "as": "cascade_act_kai"},
              {"name": "wtd_percent_10", "as": "cascade_wtd_10p"},
              {"name": "wtd_percent_20", "as": "cascade_wtd_20p"},
              {"name": "wtd_percent_30", "as": "cascade_wtd_30p"},
              {"name": "wtd_percent_40", "as": "cascade_wtd_40p"},
              {"name": "wtd_percent_50", "as": "cascade_wtd_50p"},
              {"name": "wtd_percent_60", "as": "cascade_wtd_60p"},
              {"name": "wtd_percent_70", "as": "cascade_wtd_70p"},
              {"name": "wtd_percent_80", "as": "cascade_wtd_80p"},
              {"name": "wtd_percent_90", "as": "cascade_wtd_90p"},
            ],
            timeout_ms=100,
            send_item_attrs=["item_id", ],
            send_common_attrs=cascade_ltr_common_feature,
            request_type="default",
          ) \
        .end_if_() \
      .end_if_() \
      .get_kconf_params(
        skip="{{skip_fountain_cascade_get_wtd_table}}",
        kconf_configs=[
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
      ) \
      .enrich_attr_by_light_function(
        import_common_attr=[
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
          {"name": "fountain_mc_wtd_fintr_fintr_low_bound", "as": "fintr_low_bound"},
          {"name": "fountain_mc_wtd_fintr_fintr_upper_bound", "as": "fintr_upper_bound"},
          {"name": "fountain_mc_wtd_fintr_fintr_power", "as": "fintr_power"},
        ],
        import_item_attr=[
          "duration_ms",
          "cascade_wtd_kai"
        ],
        export_item_attr=[
          "cascade_wtd_kai_mix",
        ],
        function_name="GetMcWtdScore",
        class_name="ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_lua(
        skip="{{fountain_skip_calc_cascade_action_once_score}}",
        import_common_attr=[
          "cascade_action_once_watchtime_score_pctr_weight",
          "cascade_action_once_watchtime_score_plvtr_weight",
          "cascade_action_once_watchtime_score_pfintr_quantile_weight",
          "cascade_action_once_watchtime_score_pslide_weight",
          "cascade_action_once_watchtime_score_pwatch_time_weight",
          "cascade_action_once_watchtime_score_pwtd_weight",
          "cascade_action_once_watchtime_score_pwtd_kai_weight",
          "cascade_action_once_watchtime_score_pftr_duration_weight",
          "cascade_action_once_interact_score_pltr_weight",
          "cascade_action_once_interact_score_pwtr_weight",
          "cascade_action_once_interact_score_pftr_weight",
          "cascade_action_once_interact_score_pcmtr_weight",
          "cascade_action_once_interact_score_pcmef_weight",
          "cascade_action_once_interact_score_pptr_weight",
          "cascade_action_once_interact_score_pepstr_weight",
          "cascade_action_once_interact_score_pcltr_weight",
          "cascade_action_once_interact_score_phtr_weight",
        ],
        import_item_attr=[
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
        export_item_attr=[
          "cascade_action_once_interact_score",
          "cascade_action_once_watchtime_score",
        ],
        function_for_item="cal_action_once_score",
        lua_script_file="./life/cascading/lua/module/fountain_splash_cascading_predict__calc_pxtr.lua",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr=[
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
        import_item_attr=[
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
        export_item_attr=[
          "cascade_wtd_percent",
          "cascade_wtd_duration_mix",
        ],
        function_name="GetMcWtdMix",
        class_name="ExploreLightFunctionSetV2",
      ) \
      .explore_custom_embedding_score_enricher(
        check_point_="cascade",
        enable_fountain_version=True,
        enable_fix_low_hit_rate="{{fountain_mc_enable_fix_mmu_embedding_low_hit_rate}}",
        user_info_ptr_attr="user_info_ptr",
        embedding_list_attr="mmu_embeddings",
        source_pids_list_attr="embedding_source_pids",
        calc_type="action_bucket_dot",
        not_click_limit_hour="{{fountain_mc_neg_feedback_sim_score_not_click_hour_limit}}",
        play_stat_limit_hour="{{fountain_mc_neg_feedback_sim_score_play_stat_hour_limit}}",
        extra_not_click_limit_hour="{{fountain_mc_neg_feedback_sim_score_extra_not_click_hour_limit}}",
        short_view_threshold="{{fountain_mc_neg_feedback_sim_score_short_view_threshold}}",
        not_click_weight="{{fountain_mc_neg_feedback_sim_score_not_click_weight}}",
        short_view_weight="{{fountain_mc_neg_feedback_sim_score_short_view_weight}}",
        extra_not_click_weight="{{fountain_mc_neg_feedback_sim_score_extra_not_click_weight}}",
        export_item_attr="hate_similary_score",
        dim_size=64
      ) \
      .if_("enable_life_fountain_splash_gen_minority_photo == 1") \
        .gen_is_minority_photo() \
      .end_() \
      .log_debug_info(
        item_attrs = [
          "cascade_ftr_kai_kelly"
        ]
      )

    return self
