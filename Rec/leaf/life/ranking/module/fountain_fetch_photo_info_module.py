from ranking import CommonModule
from ranking.fountain_ranking_features import fullrank_common_attrs, fullrank_splash_attrs, fullrank_fast_attrs, fullrank_common_copy_attrs

class FountainFetchPhotoInfoModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("request_type == 'fountain_splash_life' or request_type == 'fountain_splash_life_pic_inside'") \
        .get_item_attr_by_distributed_flat_index(
          photo_store_kconf_key = "reco.distributedIndex.hotPhotoInfoCommonIndex",
          perf_log = "fullrank",
          photo_store_request_data_set_tags_attr = "fountain_request_data_set_tags",
          use_dynamic_photo_store = True,
          item_id_attr = "item_id",
          attrs = fullrank_common_attrs + fullrank_splash_attrs,
        ) \
      .else_() \
        .get_item_attr_by_distributed_flat_index(
          photo_store_kconf_key = "reco.distributedIndex.hotPhotoInfoCommonIndex",
          perf_log = "fullrank",
          photo_store_request_data_set_tags_attr = "fountain_request_data_set_tags",
          use_dynamic_photo_store = True,
          item_id_attr = "item_id",
          attrs = fullrank_common_attrs + fullrank_fast_attrs,
        ) \
      .end_if_() \
      .copy_attr(
        attrs = fullrank_common_copy_attrs,
      ) \
      .set_attr_value(
        no_overwrite=True,
        item_attrs=[
          {
            "name": "hetu_level_one_v2",
            "type": "int_list",
            "value": []
          }
        ]
      ) \
      .enrich_attr_by_lua(
        import_common_attr = [
          "currentTimeMs"
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
          "hetu_tag_level_info__hetu_level_one",
          "hetu_tag_level_info_v2__hetu_level_one",
          "hetu_tag_level_info_v2__hetu_level_two",
          "hetu_tag_level_info_v2__hetu_level_three",
          "hetu_level_one_v2",
          "duration_ms"
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
          "featurePHetu0",
          "hetu_level_one_tag_index",
          "hetu_level_two_tag_index",
          "hetu_level_three_tag_index",
          "hetu_level_one_v2_index",
          "fountainDurationPercent",
          "fullrank_empirical_ctr",
          "fullrank_empirical_ltr",
          "fullrank_empirical_wtr",
          "fullrank_empirical_ftr",
          "fullrank_empirical_ptr",
          "fullrank_empirical_cmtr",
          "fullrank_empirical_htr",
          "fullrank_empirical_watchtime"
        ],
        function_for_item = "fullrank_feature_trans",
        lua_script_file = "life/ranking/lua/module/fountain_ranking_score__fullrank_feature_trans.lua",
      )

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_attrs = [
          "hetu_level_one_v2",
          "hetu_level_one_tag_index",
          "hetu_level_two_tag_index",
          "hetu_level_three_tag_index",
          "hetu_level_one_v2_index",
          "fullrank_empirical_ctr",
          "fullrank_empirical_ltr",
          "fullrank_empirical_wtr",
          "fullrank_empirical_ftr",
          "fullrank_empirical_ptr",
          "fullrank_empirical_cmtr",
          "fullrank_empirical_htr",
          "fullrank_empirical_watchtime"
        ],
        for_debug_request_only = True,
        item_num_limit = 10,
      )
