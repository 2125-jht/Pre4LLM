from dragonfly.common_leaf_dsl import LeafFlow
from dragonfly.ext.explore.explore_api_mixin import ExploreApiMixin

class PreparingFlow(LeafFlow, ExploreApiMixin):
  def __init__(self, name: str):
    super().__init__(name)

    self \
      .namespace_(ns = name, nest = True) \
      .if_("userInfo == nil", to_be_delete = "date=2024-05-29;committer=denghong") \
        .return_(1, "Empty user info str") \
      .end_() \
      .gen_common_attr_by_lua(
        attr_map = {
          "prepare_begin_ts": "util.GetTimestamp()",
        },
      ) \
      .explore_environment_type_enrich(
        type_map = {
          "fountainRecoLeaf": "prod",
          "fountainRecoLeafGray": "gray",
          "fountainRecoLeafAbGray": "gray",
          "fountainRecoLeafTest": "prod",
          "fountainRecoLeafDryrun": "dryrun",
          "fountainRecoLeafV2": "gray",
          "default": "other",
        },
        save_type_to_attr = "env_type",
      ) \
      .get_kconf_params(
        kconf_configs = [
          {
            "kconf_key": "reco.fountain.leafPerfSamplingRate",
            "default_value": 0.01,
            "export_common_attr": "perf_sampling_rate",
          },
          {
            "kconf_key": "reco.fountain.leafRankS2SamplingRate",
            "default_value": 0.0,
            "export_common_attr": "rank_s2_sampling_rate",
          },
          {
            "kconf_key": "reco.index.enablePhotoDisLikeFilter",
            "default_value": False,
            "export_common_attr": "fountain_enable_high_hate_report_filter"
          },
          {
            "kconf_key": "reco.index.photoDislikeHateThr",
            "json_path": "filter_recent_real_show_thres_attr",
            "default_value": 0,
            "export_common_attr": "fountain_filter_recent_real_show_thres_attr"
          },
          {
            "kconf_key": "reco.index.photoDislikeHateThr",
            "json_path": "filter_recent_hate_ratio_thres_attr",
            "default_value": 0.0,
            "export_common_attr": "fountain_filter_recent_hate_ratio_thres_attr"
          },
          {
            "kconf_key": "reco.index.photoDislikeHateThr",
            "json_path": "filter_recent_report_ratio_thres_attr",
            "default_value": 0.0,
            "export_common_attr": "fountain_filter_recent_report_ratio_thres_attr"
          },
          {
            "kconf_key": "reco.index.photoDislikeHateThr",
            "json_path": "filter_recent_hate_count_thres_attr",
            "default_value": 0,
            "export_common_attr": "fountain_filter_recent_hate_count_thres_attr"
          },
          {
            "kconf_key": "reco.index.photoDislikeHateThr",
            "json_path": "filter_recent_report_count_thres_attr",
            "default_value": 0,
            "export_common_attr": "fountain_filter_recent_report_count_thres_attr"
          },
          {
            "kconf_key": "reco.index.photoDislikeHateThr",
            "json_path": "filter_real_show_thres_attr",
            "default_value": 0,
            "export_common_attr": "fountain_filter_real_show_thres_attr"
          },
          {
            "kconf_key": "reco.index.photoDislikeHateThr",
            "json_path": "filter_hate_ratio_thres_attr",
            "default_value": 0.0,
            "export_common_attr": "fountain_filter_hate_ratio_thres_attr"
          },
          {
            "kconf_key": "reco.index.photoDislikeHateThr",
            "json_path": "filter_report_ratio_thres_attr",
            "default_value": 0.0,
            "export_common_attr": "fountain_filter_report_ratio_thres_attr"
          },
          {
            "kconf_key": "reco.index.photoDislikeHateThr",
            "json_path": "filter_hate_count_thres_attr",
            "default_value": 0,
            "export_common_attr": "fountain_filter_hate_count_thres_attr"
          },
          {
            "kconf_key": "reco.index.photoDislikeHateThr",
            "json_path": "filter_report_count_thres_attr",
            "default_value": 0,
            "export_common_attr": "fountain_filter_report_count_thres_attr"
          },
          {
            "kconf_key": "reco.index.fallbackRetrievalParams",
            "json_path": "enable_topk_fallback_retrieval",
            "export_common_attr": "enable_retr_filter_downgrade",
            "default_value": 0
          }
        ]
      ) \
      .copy_user_meta_info(
        save_request_type_to_attr = "common_request_type",
        save_request_num_to_attr = "request_num"
      ) \
      .gen_common_attr_by_lua(
        attr_map = {
          "_IS_ONLINE_SERVICE_": "env_type == \"prod\" or env_type == \"gray\"",
          "_ABTEST_SUFFIX_LIST_": "{({[\"fountain_fast_v1_life\"] = \"_life\", [\"fountain_splash_life\"] = \"_life\", [\"fountain_fast_pic_inside\"] = \"_picin\", [\"fountain_splash_pic_inside\"] = \"_picin\", [\"fountain_fast_life_pic_inside\"] = \"_lifepicin\", [\"fountain_splash_life_pic_inside\"] = \"_lifepicin\", [\"fountain_splash_vane\"] = \"_vane\"})[_REQ_TYPE_]}",
          "_ABTEST_REPORT_HIT_": "({[\"fountain_fast_v1_life\"] = 1, [\"fountain_splash_life\"] = 1, [\"fountain_fast_pic_inside\"] = 1, [\"fountain_splash_pic_inside\"] = 1, [\"fountain_fast_life_pic_inside\"] = 1, [\"fountain_splash_life_pic_inside\"] = 1})[_REQ_TYPE_]",
          "_RANDOM_": "util.Random()",
        }
      ) \
      .gen_common_attr_by_lua(
        attr_map = {
          "_IS_PERF_SAMPLING_REQUEST_": "_RANDOM_ < perf_sampling_rate and 1 or 0",
          "_SEND_RANK_S2_STAGE_SAMPLE_": "_RANDOM_ < rank_s2_sampling_rate and 1 or 0",
        }
      ) \
      .set_attr_value(
        no_overwrite = True,
        common_attrs = [
          {
            "name": "tab",
            "type": "int",
            "value": 666
          }
        ]
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
      .namespace_()
