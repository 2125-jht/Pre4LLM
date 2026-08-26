from dragonfly.common_leaf_dsl import LeafFlow
from dragonfly.ext.explore.explore_api_mixin import ExploreApiMixin

class PreparingFlow(LeafFlow, ExploreApiMixin):
  def __init__(self, name: str):
    super().__init__(name)

    self \
      .namespace_(ns = name, nest = True) \
      .if_("userInfo == nil") \
        .return_(1, "Empty user info str") \
      .end_() \
      .gen_common_attr_by_lua(
        attr_map = {
          "prepare_begin_ts": "util.GetTimestamp()",
        },
      ) \
      .copy_user_meta_info(
        save_need_traceback_to_attr = "is_traceback_request",
      ) \
      .explore_environment_type_enrich(
        type_map = {
          "exploreRecoLeaf": "prod",
          "exploreRecoLeafExpand": "prod",
          "exploreRecoLeafGray": "gray",
          "exploreRecoLeafAbGray": "gray",
          "exploreRecoLeafDryrun": "dryrun",
          "default": "other",
        },
        save_type_to_attr = "env_type",
      ) \
      .get_kconf_params(
        kconf_configs = [{
          "kconf_key": "reco.explore.leafPerfSamplingRate",
          "default_value": 0.01,
          "export_common_attr": "perf_sampling_rate",
        }],
      ) \
      .if_("is_traceback_request == 1") \
        .get_kconf_params(
          kconf_configs = [
            {
              "kconf_key": "reco.explore.leafAbtestMetricsSamplingRateForTraceback",
              "default_value": 1.0,
              "export_common_attr": "abtest_matrics_sampling_rate",
            },
          ],
        ) \
      .else_() \
        .get_kconf_params(
          kconf_configs = [
            {
              "kconf_key": "reco.explore.leafAbtestMetricsSamplingRate",
              "default_value": 0.1,
              "export_common_attr": "abtest_matrics_sampling_rate",
            },
          ],
        ) \
      .end_() \
      .get_kconf_params(
        kconf_configs = [
          {
            "kconf_key": "reco.index.enablePhotoDisLikeFilter",
            "default_value": False,
            "export_common_attr": "explore_enable_high_hate_report_filter"
          },
          {
            "kconf_key": "reco.index.photoDislikeHateThr",
            "json_path": "filter_recent_real_show_thres_attr",
            "default_value": 0,
            "export_common_attr": "explore_filter_recent_real_show_thres_attr"
          },
          {
            "kconf_key": "reco.index.photoDislikeHateThr",
            "json_path": "filter_recent_hate_ratio_thres_attr",
            "default_value": 0.0,
            "export_common_attr": "explore_filter_recent_hate_ratio_thres_attr"
          },
          {
            "kconf_key": "reco.index.photoDislikeHateThr",
            "json_path": "filter_recent_report_ratio_thres_attr",
            "default_value": 0.0,
            "export_common_attr": "explore_filter_recent_report_ratio_thres_attr"
          },
          {
            "kconf_key": "reco.index.photoDislikeHateThr",
            "json_path": "filter_recent_hate_count_thres_attr",
            "default_value": 0,
            "export_common_attr": "explore_filter_recent_hate_count_thres_attr"
          },
          {
            "kconf_key": "reco.index.photoDislikeHateThr",
            "json_path": "filter_recent_report_count_thres_attr",
            "default_value": 0,
            "export_common_attr": "explore_filter_recent_report_count_thres_attr"
          },
          {
            "kconf_key": "reco.index.photoDislikeHateThr",
            "json_path": "filter_real_show_thres_attr",
            "default_value": 0,
            "export_common_attr": "explore_filter_real_show_thres_attr"
          },
          {
            "kconf_key": "reco.index.photoDislikeHateThr",
            "json_path": "filter_hate_ratio_thres_attr",
            "default_value": 0.0,
            "export_common_attr": "explore_filter_hate_ratio_thres_attr"
          },
          {
            "kconf_key": "reco.index.photoDislikeHateThr",
            "json_path": "filter_report_ratio_thres_attr",
            "default_value": 0.0,
            "export_common_attr": "explore_filter_report_ratio_thres_attr"
          },
          {
            "kconf_key": "reco.index.photoDislikeHateThr",
            "json_path": "filter_hate_count_thres_attr",
            "default_value": 0,
            "export_common_attr": "explore_filter_hate_count_thres_attr"
          },
          {
            "kconf_key": "reco.index.photoDislikeHateThr",
            "json_path": "filter_report_count_thres_attr",
            "default_value": 0,
            "export_common_attr": "explore_filter_report_count_thres_attr"
          }
        ]
      ) \
      .gen_common_attr_by_lua(
        attr_map = {
          "_IS_NOT_BACKUP_": "_REQ_TYPE_ ~= \"backup\"",
          "_IS_ONLINE_SERVICE_": "env_type == \"prod\" or env_type == \"gray\"",
          "_ABTEST_SUFFIX_LIST_": "({[\"la\"] = {\"_la\"}, [\"life\"] = {\"_life\"}, [\"tnu\"] = {\"_tnu\", \"_tnu_plus\", \"_la\"}, [\"la_legacy\"] = {\"_lal\"}, [\"first_screen\"] = {\"_fs\"}, [\"tnu_plus\"] = {\"_tnu_plus\", \"_la\"}, [\"low_mid_active\"] = {\"_lma\"}})[_REQ_TYPE_]",
          "_ABTEST_REPORT_HIT_": "({[\"la\"] = 1, [\"life\"] = 1, [\"tnu\"] = 1, [\"tnu_plus\"] = 1})[_REQ_TYPE_]",
          "_IS_PERF_SAMPLING_REQUEST_": "util.Random() < perf_sampling_rate and 1 or 0",
          "_IS_ABTEST_METRICS_SAMPLING_REQUEST_": "util.Random() < abtest_matrics_sampling_rate"
        }
      ) \
      .copy_attr(
        attrs = [
          {
            "from_common": "_IS_ABTEST_METRICS_SAMPLING_REQUEST_",
            "to_common": "enable_explore_pic_cluster_counter",
          },
        ],
      ) \
      .enrich_attr_by_lua(
        import_common_attr = [
          "_ABTEST_SUFFIX_LIST_",
        ],
        export_common_attr = [
          "_ABTEST_SUFFIX_LIST_",
        ],
        function_for_common = "calculate",
        lua_script = """
          function calculate()
            if _ABTEST_SUFFIX_LIST_ then
              table.insert(_ABTEST_SUFFIX_LIST_, 1, "_dryrun")
            else
              _ABTEST_SUFFIX_LIST_ = {"_dryrun"}
            end
            return _ABTEST_SUFFIX_LIST_
          end
        """,
        skip = "{{return env_type ~= \"dryrun\"}}",
      ) \
      .parse_protobuf_from_string(
        input_attr = "userInfo",
        output_attr = "user_info_ptr",
        class_name = "ks.reco.UserInfo",
      ) \
      .if_("user_info_ptr == nil", no_delete = True) \
        .return_(1, "Error user info str") \
      .end_() \
      .enrich_with_protobuf(
        from_extra_var = "user_info_ptr",
        attrs = [
          dict(name = "uIsBigG", path = "feature_collection.features", sample_attr_name = "uIsBigG", skip_unset_field = True),
          dict(name = "user_live_paying_type", path = "user_class.user_live_paying_type", skip_unset_field = True),
        ],
      ) \
      .gen_common_attr_by_lua(
        attr_map = {
          "is_live_big_g_user": "uIsBigG or 0",
          "is_live_paying_user": "user_live_paying_type ~= nil and user_live_paying_type >= 0 and user_live_paying_type <= 10 and 1 or 0",
          "is_live_high_paying_user": "user_live_paying_type ~= nil and user_live_paying_type >= 3 and user_live_paying_type <= 6 and 1 or 0",
        },
      ) \
      .gen_common_attr_by_lua(
        attr_map = {
          "_ABTEST_USER_TAG_NAMES_": "{\"uIsBigG\", \"uIsLivePayingUser\", \"uIsLiveHighPayingUser\"}",
          "_ABTEST_USER_TAG_VALUES_": "{tostring(is_live_big_g_user), tostring(is_live_paying_user), tostring(is_live_high_paying_user)}",
        },
      ) \
      .get_abtest_params(
        biz_name = "RECO_RPC",
        ab_params = [
          ("increase_quota_after_peak_time_window", ""),
          ("explore_retrieval_limit_num", 5000, "_ABTEST_RETRIEVAL_LIMIT_NUM_"),
          ("enable_explore_dump_attrs_to_kafka", 0, "enable_dump_attrs_to_kafka"),
          ("explore_personal_quota_register_exptag", ""),
          ("explore_personal_quota_upper_limit", 1.2),
          ("explore_personal_quota_lower_limit", 0.8),
          ("explore_personal_quota_realshow_threshold", 200),
          ("explore_personal_quota_cal_ratio_mode", 0),
          ("explore_personal_quota_power", 1.0),
          ("hetu_memory_rank_final_result_num", 2000),
          ("comirec_elegant_final_result_num_la", 1500, "comirec_elegant_final_result_num"),
          ("enable_explore_cascading_v2_flow", False),
        ]
      ) \
      .get_kconf_params(
        kconf_configs = 
        [
          {
            "kconf_key": "reco.index.fallbackRetrievalParams",
            "json_path": "enable_topk_fallback_retrieval",
            "export_common_attr": "enable_retr_filter_downgrade",
            "default_value": 0
          },
          {
            "kconf_key": "reco.explore.auditTopkPassRetrNum",
            "export_common_attr": "audit_topk_pass_retr_final_result_num",
            "default_value": 6000
          }
        ]
      ) \
      .set_attr_value(
        no_overwrite = True,
        common_attrs = [
          {
            "name": "tab",
            "type": "int",
            "value": 0
          }
        ]
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "increase_quota_after_peak_time_window",
        ],
        export_common_attr = [
          {"name":"increase_quota_after_peak", "as":"increase_quota_status"}
        ],
        function_name = "GetIncreaseQuotaStatus",
        class_name = "ExploreLightFunctionSetV2",
        skip = "{{return #increase_quota_after_peak_time_window == 0}}"
      ) \
      .explore_retr_personal_quota_enrich(
        register_exptag = "{{explore_personal_quota_register_exptag}}",
        cluster_name = "recoExploreUserStat",
        time_out = 20,
        upper_limit = "{{explore_personal_quota_upper_limit}}",
        lower_limit = "{{explore_personal_quota_lower_limit}}",
        save_data_ptr_to_attr = "reason_ratio_map_attr",
        realshow_threshold = "{{explore_personal_quota_realshow_threshold}}",
        cal_ratio_mode = "{{explore_personal_quota_cal_ratio_mode}}",
        power = "{{explore_personal_quota_power}}",
        skip = "{{return #explore_personal_quota_register_exptag == 0}}"
      ) \
      .namespace_()
