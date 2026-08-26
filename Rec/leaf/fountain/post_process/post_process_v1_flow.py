#!/usr/bin/env python3
# coding=utf-8

from dragonfly.common_leaf_dsl import LeafFlow
from dragonfly.ext.subdivision.subdivision_api_mixin import subdivisionApiMixin
from dragonfly.ext.explore.explore_api_mixin import ExploreApiMixin
from post_process.ab_params import post_process_common_params, post_process_fast_params, post_process_fast_params_ab_hit
from util import enrich_ab_param

class PostProcessV1Flow(LeafFlow, subdivisionApiMixin, ExploreApiMixin):
  def __init__(self):
    LeafFlow.__init__(self, "post_process_v1")
    self \
      .namespace_(ns = "post_process_v1", nest = True) \
      ._process() \
      .namespace_()

  def _process(self):
    self \
    .get_abtest_params(
      biz_name = "RECO_RPC",
      ab_params = enrich_ab_param(post_process_common_params + post_process_fast_params),
      prioritized_suffix = "{{_ABTEST_SUFFIX_LIST_}}"
    ) \
    .get_abtest_params(
      biz_name = "RECO_RPC",
      ab_params = post_process_fast_params_ab_hit,
      prioritized_suffix = "{{_ABTEST_SUFFIX_LIST_}}"
    ) \
    .variant(
      variant_config = {
        "default_decay_window_size": 20,
        # 设置为 2 表示不允许重复，只能出现 1 次
        "default_decay_occurrent_times": 2,
        # 打散衰减系数
        "default_decay_rate": 0.2,
        # 按 author_id 打散
        "author__id": {
          "decay_window_size": 20,
          "decay_occurrent_times": 2,
          "decay_rate": 0.1,
        },
        "picture_variant_attr": {
          "decay_window_size": 7,
          "decay_occurrent_times": 2,
          "decay_rate": 0.1,
        },
        "author__category_detail__fourth_level_id": {
          "decay_window_size": 6,
          "decay_occurrent_times": 3,
          "decay_rate": 0.1,
        },
      },
      skip="{{skip_fountain_variant}}",
    ) \
    .transform_item_attr(
      mappings = [{
        "check_attr_name": "duration_ms",
        "check_attr_type": "int",
        "output_attr_name": "short_duration_variant_attr",
        "output_attr_type": "int",
        "rules": [{
          "check_range": {
            "upper_bound": "{{short_duration_photo_upper_bound}}"
          },
          "output_value": 1
        }],
      }]) \
    .enrich_attr_by_lua(
      import_item_attr=["item_seq_neg_discount"],
      export_item_attr=["normalized_item_seq_neg_discount"],
      function_for_item="calculate",
      lua_script_file = "fountain/post_process/lua/gen_norm_item_seq_neg_discount.lua") \
    .enrich_attr_by_lua(
      import_common_attr = [
        "currentTimeMs",
        "fountain_variant_upload_days_threshold",
        "skip_variant_v9_gr_pr",
      ],
      import_item_attr = [
        "similar_event_id",
        "hetu_tag_v2",
        "author__is_gr_account",
        "author__is_pr_account",
        "photo_id",
        "upload_time",
        "hetu_tag_level_info__hetu_level_two",
        "hetu_tag_level_info__hetu_level_three",
      ],
      export_item_attr = [
        "is_olympic_photo",
        "hetu_tag_v2_move_time",
        "similar_event_variant_attr",
        "pr_gr_account_photo_id",
        "is_overdue_photo",
        "is_pgc_hetu_level_two",
        "is_pgc_hetu_level_three",
      ],
      function_for_item = "gen_variant_attr",
      lua_script_file = "fountain/post_process/lua/gen_variant_attr.lua",
    ) \
    .fountain_mmr_variant(
      skip = "{{skip_fountain_variant_mmr}}",
      user_info_attr = "userInfo",
      mmr_need_diversity_num = "{{fountain_variant_mmr_need_diversity_num}}",
      mmr_cal_candidate_num = "{{fountain_variant_mmr_cal_candidate_num}}",
      item_ensemble_score_attr = "{{fountain_variant_mmr_item_ensemble_score_attr}}",
      mmr_lambda = "{{fountain_variant_mmr_lambda}}",
      mmr_gamma = "{{fountain_variant_mmr_gamma}}",
      mmr_no_max_weight_use = "{{fountain_mmr_no_max_weight_use}}",
      enable_mmr_his_feedback_diversity = "{{fountain_enable_mmr_his_feedback_diversity}}",
      mmr_no_his_max_weight_use = "{{fountain_mmr_no_his_max_weight_use}}",
      mmr_diversity_dim_weight = "{{fountain_variant_mmr_diversity_dim_weight}}",
      mmr_history_diversity_dim_weight = "{{fountain_variant_mmr_history_diversity_dim_weight}}",
      mmr_cal_noclick_num = "{{fountain_variant_mmr_cal_noclick_num}}",
      mmr_real_show_expired_gap = "{{fountain_variant_mmr_real_show_expired_gap}}",
      mmr_page_num_limit = "{{fountain_variant_mmr_page_num_limit}}",
      mmr_time_weight_param = "{{fountain_variant_mmr_time_weight_param}}",
      mmr_time_interval_param = "{{fountain_variant_mmr_time_interval_param}}",
      enable_mmr_photo_embedding = "{{fountain_enable_mmr_photo_embedding}}",
      item_attrs = [
        "hetu_level_one", "hetu_level_two", "hetu_level_five", "author__id",
        "photo_dnn_cluster_id", "mmu_img_cluster_v3", "tag",
        "author__category_detail__first_level_id",
        "author__category_detail__second_level_id",
        "author__category_detail__third_level_id",
        "GE_cluster_id", "mmu_text_lda_topic", "mmu_text_cluster", "upload_type",
        "is_overdue_photo", "is_photo_author_followed",
      ],
      item_list_attrs = [
        "hetu_tag_level_info__hetu_face_id", "online_lda_topic__ids", "hetu_tag_level_info__hetu_tag",
      ],
      item_count_attrs = [
        "explore_stat__show_count",
      ],
      his_item_attrs = [
        "hetu_level_one", "hetu_level_two", "author__id",
        "photo_dnn_cluster_id", "mmu_img_cluster_v3", "tag", "hetu_level_five",
      ],
    ) \
    .copy_item_meta_info(
      save_item_seq_to_attr="item_seq_current",
      skip="{{fountain_skip_save_item_seq_current}}",
    ) \
    .enrich_attr_by_lua(
      skip="{{fountain_skip_save_item_seq_current}}",
      import_item_attr=["item_seq_current"],
      export_item_attr=["normalized_item_seq_current"],
      function_for_item="calculate",
      lua_script_file="fountain/post_process/lua/gen_norm_item_seq.lua"
    ) \
    .fountain_variant_multi_page(
      skip = "{{skip_fountain_variant_multi_page}}",
      window_limit_size = "{{variant_multi_page_window_size}}",
      candidate_limit_size = "{{variant_multi_page_candidate_size}}",
      item_score_from_attr = "{{fountain_variant_multi_page_score_attr}}",
      variant_config = {
        "default_decay_window_size": 20,
        # 设置为 2 表示不允许重复，只能出现 1 次
        "default_decay_occurrent_times": 2,
        # 打散衰减系数
        "default_decay_rate": 0.2,
        # 按照河图一级标签打散
        "hetu_level_one_v2": {
          "decay_occurrent_times": "{{variant_multi_page_hetu_one_times}}",
          "decay_rate":  "{{variant_multi_page_hetu_one_decay_rate}}",
        },
        # 按照河图二级标签打散
        "hetu_level_two_v2": {
          "decay_occurrent_times": "{{variant_multi_page_hetu_two_times}}",
          "decay_rate":  "{{variant_multi_page_hetu_two_decay_rate}}",
        },
        "hetu_level_three_v2": {
          "decay_occurrent_times": "{{variant_multi_page_hetu_three_times}}",
          "decay_rate":  "{{variant_multi_page_hetu_three_decay_rate}}",
        },
        "hetu_face_id_v2": {
          "decay_occurrent_times": "{{variant_multi_page_hetu_face_id_times}}",
          "decay_rate":  "{{variant_multi_page_hetu_face_id_decay_rate}}",
        },
      },
      user_info_attr = "userInfo",
    ) \
    .variant(
      skip = "{{skip_variant_v8_author}}",
      variant_config = {
        "default_decay_window_size": 20,
        "default_decay_occurrent_times": 2,
        "default_decay_rate": 0.2,
        "author__category_detail__second_level_id": {
          "decay_window_size": 7,
          "decay_occurrent_times": 3,
          "decay_rate": 0.1,
          "enabled": "{{enabled_fountain_variant_author_level_two}}"
        },
      }
    ) \
    .variant(
      skip = "{{skip_variant_v9_hetu}}",
      variant_config = {
        "default_decay_window_size": 5,
        "default_decay_occurrent_times": 2,
        "default_decay_rate": 0.2,
        "hetu_level_two":{
          "decay_window_size":5,
          "decay_occurrent_times":2,
          "decay_rate":0.1,
          "enabled": "{{enabled_fountain_variant_hetu_level_two}}"
        },
        "hetu_level_one": {
          "decay_window_size":5,
          "decay_occurrent_times":3,
          "decay_rate":0.1,
          "enabled": "{{enabled_fountain_variant_hetu_level_one}}"
        },
        "is_photo_author_followed": {
          "decay_window_size": "{{fountain_variant_follow_author_id_win_size}}",
          "decay_occurrent_times": 2,
          "decay_rate": 0.1,
          "enabled": "{{enabled_fountain_variant_follow_author_id}}"
        },
        "is_overdue_photo": {
          "decay_occurrent_times": "{{fountain_variant_is_overdue_photo_decay_times}}",
          "decay_rate": 0.1,
          "decay_window_size": "{{fountain_variant_is_overdue_photo_win_size}}",
          "enabled": "{{enable_fountain_variant_is_overdue_photo}}"
        },
        # 奥运图片打散
        "is_olympic_photo": {
          "decay_window_size": "{{fountain_variant_olympic_photo_win_size}}",
          "decay_occurrent_times": 2,
          "decay_rate": 0.1,
          "enabled": "{{enabled_fountain_variant_olympic_photo}}"
        },
        # 新河图打散
        "hetu_level_one_v2":{
          "decay_window_size": "{{fountain_variant_hetu_level_one_win_size_new}}",
          "decay_occurrent_times": "{{fountain_variant_hetu_level_one_decay_times_new}}",
          "decay_rate":0.1,
          "any_of" : True,
          "enabled": "{{enabled_fountain_variant_hetu_level_one_new}}"
        },
        "hetu_level_two_v2": {
          "decay_window_size": "{{fountain_variant_hetu_level_two_win_size_new}}",
          "decay_occurrent_times": "{{fountain_variant_hetu_level_two_decay_times_new}}",
          "decay_rate":0.1,
          "any_of" : True,
          "enabled": "{{enabled_fountain_variant_hetu_level_two_new}}"
        },
        "hetu_level_five_v2": {
          "decay_window_size": "{{fountain_variant_hetu_level_five_win_size_new}}",
          "decay_occurrent_times": "{{fountain_variant_hetu_level_five_decay_times_new}}",
          "decay_rate":0.1,
          "any_of" : True,
          "enabled": "{{enabled_fountain_variant_hetu_level_five_new}}"
        },
        # 河图 tag 打散
        "hetu_tag_v2_move_time": {
          "decay_window_size": "{{fountain_variant_hetu_tag_win_size}}",
          "decay_occurrent_times": "{{fountain_variant_hetu_tag_decay_times}}",
          "decay_rate": 0.1,
          "enabled": "{{enabled_fountain_variant_hetu_tag}}"
        },
        # 河图 face_id 打散
        "hetu_face_id_v2": {
          "decay_window_size": "{{fountain_variant_hetu_face_id_win_size}}",
          "decay_occurrent_times": "{{fountain_variant_hetu_face_id_decay_times}}",
          "decay_rate": 0.1,
          "any_of" : True,
          "enabled": "{{enabled_fountain_variant_hetu_face_id}}"
        },
        # 河图 cluster_id 打散
        "hetu_tag_level_info_v2__hetu_cluster_id": {
          "decay_window_size": "{{fountain_variant_hetu_cluster_id_win_size}}",
          "decay_occurrent_times": "{{fountain_variant_hetu_cluster_id_decay_times}}",
          "decay_rate": 0.1,
          "enabled": "{{enabled_fountain_variant_hetu_cluster_id}}"
        },
        # 媒体号打散
        "similar_event_variant_attr": {
          "decay_window_size": "{{fountain_variant_similar_event_id_win_size}}",
          "decay_occurrent_times": "{{fountain_variant_similar_event_id_decay_times}}",
          "decay_rate": 0.1,
          "enabled": "{{enabled_fountain_variant_similar_event_id}}"
        },
        # 时长 <7s 打散
        "short_duration_variant_attr": {
          "decay_window_size": "{{fountain_variant_short_duration_decay_window_size}}",
          "decay_occurrent_times": "{{fountain_variant_short_duration_decay_occurrent_times}}",
          "decay_rate": 0.1,
          "enabled": "{{enable_fountain_variant_short_duration}}"
        },
        # pgc高发的hetu_level_two类目视频打散
        "is_pgc_hetu_level_two": {
          "decay_window_size": 6,
          "decay_occurrent_times": "{{fountain_variant_is_pgc_hetu_level_two_occurrent_times}}",
          "decay_rate": 0.1,
          "enabled": "{{enabled_fountain_variant_is_pgc_hetu_level_two}}"
        },
        # pgc高发的hetu_level_three类目视频打散
        "is_pgc_hetu_level_three": {
          "decay_window_size": 6,
          "decay_occurrent_times": "{{fountain_variant_is_pgc_hetu_level_three_occurrent_times}}",
          "decay_rate": 0.1,
          "enabled": "{{enabled_fountain_variant_is_pgc_hetu_level_three}}"
        },
      }
    ) \
    .enrich_attr_by_lua(
      skip="{{fountain_skip_gen_hetu_tag_type}}",
      import_item_attr=["hetu_tag_v2", "picture_variant_attr"],
      export_item_attr=["hetu_tag_v2_theme", "hetu_tag_v2_style", "hetu_tag_v2_normal", "picture_variant_attr_adjust"],
      function_for_item="calculate",
      lua_script_file="fountain/post_process/lua/gen_hetu_tag_type.lua"
    )\
    .if_("fountain_skip_variant_merge_v2 == 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .gen_realtime_browse_set(
        enable_fountain_browse = "{{fountain_skip_enable_fountain_browse}}",
        enable_hot_browse = "{{fountain_skip_enable_hot_browse}}",
        output_common_attr = "fountainRecentPlayList",
        realtime_hot_bs_size = "{{realtime_fountain_browse_set_size}}",
        realtime_fountain_bs_size = "{{realtime_hot_browse_set_size}}",
        profile_time_threshold = "{{fountain_profile_time_threshold}}",
        user_info_ptr_attr = "userInfoPb",
      ) \
      .get_item_attr_by_distributed_flat_index(
        photo_store_kconf_key = "reco.distributedIndex.hotPhotoInfoCommonIndex",
        perf_log = "post_process",
        photo_store_request_data_set_tags_attr = "fountain_request_data_set_tags",
        use_dynamic_photo_store = True,
        attrs = [
          "hetu_tag_level_info_v2__hetu_level_one",
          "hetu_tag_level_info_v2__hetu_level_two",
          "hetu_tag_level_info_v2__hetu_level_three",
          "hetu_tag_level_info_v2__hetu_level_four",
          "hetu_tag_level_info_v2__hetu_level_five",
          "hetu_tag_level_info_v2__hetu_tag",
          "hetu_tag_level_info_v2__hetu_face_id",
        ],
        additional_item_source = {
          "reco_results": False,
          "common_attr": ["fountainRecentPlayList"]
        }) \
      .enrich_attr_by_lua(
        import_item_attr = [
          "hetu_tag_level_info_v2__hetu_level_one",
          "hetu_tag_level_info_v2__hetu_level_two",
          "hetu_tag_level_info_v2__hetu_level_three",
          "hetu_tag_level_info_v2__hetu_level_five",
          "hetu_tag_level_info_v2__hetu_tag",
          "hetu_tag_level_info_v2__hetu_face_id",
        ],
        export_item_attr = [
          "hetu_level_one_v2",
          "hetu_level_two_v2",
          "hetu_level_three_v2",
          "hetu_level_five_v2",
          "hetu_tag_v2",
          "hetu_face_id_v2",
          "limit_hetu_table"
        ],
        item_list_from_attr = "fountainRecentPlayList",
        function_for_item = "calculate",
        lua_script_file = "fountain/full_rank/lua/gen_hetu_tag.lua",
      ) \
      .log_debug_info(
        item_list_from_attr = "fountainRecentPlayList",
        item_attrs = [
          "hetu_tag_level_info_v2__hetu_level_one",
          "hetu_tag_level_info_v2__hetu_level_two",
          "hetu_tag_level_info_v2__hetu_level_three",
          "hetu_tag_level_info_v2__hetu_level_five",
          "hetu_tag_level_info_v2__hetu_tag",
          "hetu_tag_level_info_v2__hetu_face_id",
          "hetu_face_id_v2",
          "hetu_tag_level_info_v2__hetu_level_four",
          "hetu_level_two_v2",
          "hetu_level_five_v2",
          "hetu_level_one_v2",
          "hetu_level_three_v2",
          "hetu_tag_v2",
          "limit_hetu_table",
        ],
        item_num_limit = 5,
        for_debug_request_only = True,
      ) \
      .diversify_by_rules(
        max_satisfied_pick="{{fountain_variant_merge_max_satisfied_v2}}",
        prev_items_from_attr = "fountainRecentPlayList",
        consider_prev_items = True,
        rules=[
          # 新河图tag打散
          dict(attr_name="hetu_tag_v2",
              enabled="{{enabled_fountain_variant_hetu_tag_v2}}",
              window_size="{{fountain_variant_hetu_tag_v2_win_size}}",
              max_num="{{fountain_variant_hetu_tag_v2_decay_times}}",
              priority=6),
          # 河图face id打散
          dict(attr_name="hetu_face_id_v2",
              enabled="{{enabled_fountain_variant_hetu_face_id_v2}}",
              window_size="{{fountain_variant_hetu_face_id_win_size}}",
              max_num="{{fountain_variant_hetu_face_id_decay_times}}",
              priority=6),
          # 河图 cluster_id 打散
          dict(attr_name="hetu_tag_level_info_v2__hetu_cluster_id",
              enabled="{{enabled_fountain_variant_hetu_cluster_id_v2}}",
              window_size="{{fountain_variant_hetu_cluster_id_win_size}}",
              max_num="{{fountain_variant_hetu_cluster_id_decay_times}}",
              priority=6),
          # 新河图一级打散
          dict(attr_name="hetu_level_one_v2",
              enabled="{{enabled_fountain_variant_hetu_level_one_new_v2}}",
              window_size="{{fountain_variant_hetu_level_one_win_size_new}}",
              max_num="{{fountain_variant_hetu_level_one_decay_times_new}}",
              priority=2),
          # 新河图二级打散
          dict(attr_name="hetu_level_two_v2",
              enabled="{{enabled_fountain_variant_hetu_level_two_new_v2}}",
              window_size= "{{fountain_variant_hetu_level_two_win_size_new}}",
              max_num="{{fountain_variant_hetu_level_two_decay_times_new}}",
              priority=3),
          # 新河图三级打散
          dict(attr_name="hetu_level_three_v2",
              enabled="{{enabled_fountain_variant_hetu_level_three_new_v2}}",
              window_size= "{{fountain_variant_hetu_level_three_win_size_new}}",
              max_num="{{fountain_variant_hetu_level_three_decay_times_new}}",
              priority=3),
          # 新河图五级打散
          dict(attr_name="hetu_level_five_v2",
              enabled="{{enabled_fountain_variant_hetu_level_five_new_v2}}",
              window_size= "{{fountain_variant_hetu_face_level_five_win_size_new}}",
              max_num="{{fountain_variant_hetu_face_level_five_decay_times_new}}",
              priority="{{fountain_variant_hetu_face_level_five_priority_new}}")
        ]
      ) \
    .end_if_() \
    .diversify_by_rules(
      skip="{{fountain_skip_variant_merge}}",
      max_satisfied_pick="{{fountain_variant_merge_max_satisfied}}",
      range_end="{{fountain_variant_merge_limit}}",
      rules=[
        dict(attr_name="picture_variant_attr_adjust",
             enabled="{{enabled_fountain_variant_pic}}",
             window_size= "{{fountain_variant_pic_win_size}}",
             max_num="{{fountain_variant_hetu_pic_decay_times}}",
             priority="{{fountain_variant_pic_priority}}"),
        dict(attr_name="hetu_level_five_v2",
             enabled="{{enabled_fountain_variant_hetu_level_five_new}}",
             window_size= "{{fountain_variant_hetu_face_level_five_win_size_new}}",
             max_num="{{fountain_variant_hetu_face_level_five_decay_times_new}}",
             priority="{{fountain_variant_hetu_face_level_five_priority_new}}"),
        dict(attr_name="hetu_tag_v2_theme",
             enabled="{{enabled_fountain_variant_hetu_tag_v2_theme}}",
             window_size= "{{fountain_variant_hetu_tag_v2_theme_win_size}}",
             max_num="{{fountain_variant_hetu_tag_v2_theme_decay_times}}",
             priority="{{fountain_variant_hetu_tag_v2_theme_priority}}"),
        dict(attr_name="hetu_tag_v2_style",
             enabled="{{enabled_fountain_variant_hetu_tag_v2_style}}",
             window_size= "{{fountain_variant_hetu_tag_v2_style_win_size}}",
             max_num="{{fountain_variant_hetu_tag_v2_style_decay_times}}",
             priority="{{fountain_variant_hetu_tag_v2_style_priority}}"),
        dict(attr_name="hetu_tag_v2_normal",
             enabled="{{enabled_fountain_variant_hetu_tag_v2_normal}}",
             window_size= "{{fountain_variant_hetu_tag_v2_normal_win_size}}",
             max_num="{{fountain_variant_hetu_tag_v2_normal_decay_times}}",
             priority="{{fountain_variant_hetu_tag_v2_normal_priority}}"),
        # 按 author_id 打散
        dict(attr_name= "author__id", enabled="{{enable_fountain_variant_author_id}}",
             window_size="{{fountain_variant_author_id_decay_window_size}}",
             window_type="slide",
             max_num= "{{fountain_variant_author_id_decay_occurrent_times}}",
             priority=9),
        # 按作者四级类目打散
        dict(attr_name="author__category_detail__fourth_level_id",
             enabled="{{enable_fountain_variant_author_fourth_level}}",
             window_size="{{fountain_variant_author_fourth_level_decay_window_size}}",
             max_num="{{fountain_variant_author_fourth_level_decay_occurrent_times}}",
             priority=8),
        # 按作者二级类目打散
        dict(attr_name="author__category_detail__second_level_id",
             enabled="{{enable_fountain_variant_author_second_level}}",
             window_size= "{{fountain_variant_author_second_level_decay_window_size}}",
             max_num="{{fountain_variant_author_second_level_decay_occurrent_times}}",
             priority=7),
        # 关注作者打散
        dict(attr_name="is_photo_author_followed",
             enabled="{{enabled_fountain_variant_follow_author_id}}",
             window_size= "{{fountain_variant_follow_author_id_win_size}}",
             max_num="{{fountain_variant_follow_author_id_decay_occurrent_times}}",
             priority=8),
        # 过期视频打散
        dict(attr_name="is_overdue_photo",
             enabled="{{enable_fountain_variant_is_overdue_photo}}",
             window_size="{{fountain_variant_is_overdue_photo_win_size}}",
             max_num="{{fountain_variant_is_overdue_photo_decay_times}}",
             priority=6),
        # 河图face id打散
        dict(attr_name="hetu_face_id_v2",
             enabled="{{enabled_fountain_variant_hetu_face_id}}",
             window_size="{{fountain_variant_hetu_face_id_win_size}}",
             max_num="{{fountain_variant_hetu_face_id_decay_times}}",
             priority=6),
        # 河图 cluster_id 打散
        dict(attr_name="hetu_tag_level_info_v2__hetu_cluster_id",
             enabled="{{enabled_fountain_variant_hetu_cluster_id}}",
             window_size="{{fountain_variant_hetu_cluster_id_win_size}}",
             max_num="{{fountain_variant_hetu_cluster_id_decay_times}}",
             priority=6),
        # 媒体号打散
        dict(attr_name="similar_event_variant_attr",
             enabled="{{enabled_fountain_variant_similar_event_id}}",
             window_size="{{fountain_variant_similar_event_id_win_size}}",
             max_num= "{{fountain_variant_similar_event_id_decay_times}}",
             priority=5),
        # 时长 <7s 打散
        dict(attr_name="short_duration_variant_attr",
             enabled="{{enable_fountain_variant_short_duration}}",
             window_size="{{fountain_variant_short_duration_decay_window_size}}",
             max_num="{{fountain_variant_short_duration_decay_occurrent_times}}",
             priority=3),
        # pgc高发的hetu_level_two类目视频打散
        dict(attr_name="is_pgc_hetu_level_two",
             enabled="{{enabled_fountain_variant_is_pgc_hetu_level_two}}",
             window_size="{{fountain_variant_is_pgc_hetu_level_two_window_size}}",
             window_tupe="slide",
             max_num= "{{fountain_variant_is_pgc_hetu_level_two_occurrent_times}}",
             priority=4,),
        # pgc高发的hetu_level_three类目视频打散
        dict(attr_name="is_pgc_hetu_level_three",
             enabled="{{enabled_fountain_variant_is_pgc_hetu_level_three}}",
             window_size="{{fountain_variant_is_pgc_hetu_level_three_window_size}}",
             window_type="slide",
             max_num="{{fountain_variant_is_pgc_hetu_level_three_occurrent_times}}",
             priority=5),
        # 新河图一级打散
        dict(attr_name="hetu_level_one_v2",
             enabled="{{enabled_fountain_variant_hetu_level_one_new}}",
             window_size="{{fountain_variant_hetu_level_one_win_size_new}}",
             max_num="{{fountain_variant_hetu_level_one_decay_times_new}}",
             priority="{{fountain_variant_hetu_level_one_v2_priority}}"),
        # 新河图二级打散
        dict(attr_name="hetu_level_two_v2",
             enabled="{{enabled_fountain_variant_hetu_level_two_new}}",
             window_size= "{{fountain_variant_hetu_level_two_win_size_new}}",
             max_num="{{fountain_variant_hetu_level_two_decay_times_new}}",
             priority="{{fountain_variant_hetu_level_two_v2_priority}}"),
        # v1河图一级打散
        dict(attr_name="hetu_level_one",
             enabled="{{enabled_fountain_variant_hetu_level_one_v1}}",
             window_size= "{{fountain_variant_hetu_level_one_v1_win_size}}",
             max_num="{{fountain_variant_hetu_level_one_v1_decay_times}}",
             priority="{{fountain_variant_hetu_level_one_v1_priority}}"),
        # v1河图二级打散
        dict(attr_name="hetu_level_two",
             enabled="{{enabled_fountain_variant_hetu_level_two_v1}}",
             window_size= "{{fountain_variant_hetu_level_two_v1_win_size}}",
             max_num="{{fountain_variant_hetu_level_two_v1_decay_times}}",
             priority="{{fountain_variant_hetu_level_two_v1_priority}}"),
        # v1河图五级打散
        dict(attr_name="hetu_level_five",
             enabled="{{enabled_fountain_variant_hetu_level_five_v1}}",
             window_size= "{{fountain_variant_hetu_level_five_v1_win_size}}",
             max_num="{{fountain_variant_hetu_level_five_v1_decay_times}}",
             priority="{{fountain_variant_hetu_level_five_v1_priority}}")
      ]
    ) \
    .perflog_attr_value(
      name = "fountain_fr_diversify",
      traceback = True,
      check_point = "audit_result_record",
      item_attrs = [
        "explore_operation_c_review_level",
        "audit_b_second_tag",
        "audit_hot_high_tag_level",
        "content_safety_level_with_namespace__level_hot_online",
        "topk_audit_level",
        "audit_hot_high_subdivision_level"
      ],
      aggregator = "count") \
    .perflog_attr_value(
      check_point = "variant_result_record",
      item_attrs = [
        "hetu_level_one",
        "hetu_level_two",
        "pr_gr_account_photo_id",
        "author__category_detail__second_level_id"
      ],
      range_end = 6) \
    .copy_item_meta_info(
      save_item_seq_to_attr="item_seq_current",
    ) \
    .enrich_attr_by_lua(
      skip="{{fountain_skip_last_page_hetu_boost}}",
      import_common_attr=[
        "last_page_hetu_one_total",
        "last_page_hetu_two_total",
        "last_page_hetu_three_total",
        "last_page_hetu_tag_total",
        "fountian_last_page_hetu_tag_boost",
        "fountian_last_page_hetu_three_boost",
        "fountian_last_page_hetu_two_boost",
        "fountian_last_page_hetu_one_boost"
      ],
      import_item_attr=[
        "item_seq_current",
        "hetu_level_one_v2",
        "hetu_level_two_v2",
        "hetu_level_three_v2",
        "hetu_level_five_v2",
        "hetu_tag_v2",
        "hetu_face_id_v2",],
      export_item_attr=["post_score_after_adjust"],
      function_for_item="post_score_fast_adjust",
      lua_script_file="fountain/post_process/lua/post_processor_adjust.lua",
    ) \
    .sort(
      skip="{{fountain_skip_last_page_hetu_boost}}",
      score_from_attr="post_score_after_adjust",
    ) \
    .perflog_attr_value(
      check_point="post_top100_",
      item_attrs=["duration_perf_id"],
      aggregator="count",
      range_end = 100,
    ) \
    .log_debug_info(
      common_attrs = [
        "featureUserIsFountainSplash",
        "featureUserIsFountainRequest",
        "uId",
        "deviceId",
        "similar_user_colossus_hetu_list",
        "cascade_adaptive_long_video_cut_ratio_config",
        "featureSourcePId",
        "similar_user_list_size",
        "explore_history_hetu_list_size",
        "explore_hetu_list_all_size",
        "fountain_fullrank_variant_cluster_checkpoint",
        "fountain_ensemble_power_weight_fullrank_lstr_score",
        "fountain_ensemble_power_weight_fullrank_pptr_score",
        "fountain_skip_pic_and_selfdup_id_deduplicate_in_pagesize",
        "fountain_skip_sim_remove_dup_id_deduplicate_in_pagesize",
        "fountain_ltr_v3_duration_weight",
        "fountain_ltr_v3_ctr_weight",
        "sample_user_features",
        "sample_item_features",
        "filtered_content_ids_item_key",
        "fullrank_click_score_adaptive_factor",
        "fullrank_sim_like_score_adaptive_factor",
        "fullrank_sim_follow_score_adaptive_factor",
        "fullrank_sim_pcmtr_adaptive_factor",
        "fullrank_sim_pptr_adaptive_factor",
        "fullrank_sim_phtr_adaptive_factor",
        "fullrank_sim_pvtr_adaptive_factor",
        "fullrank_sim_psvr_adaptive_factor",
        "fullrank_sim_plvtr_adaptive_factor",
        "fullrank_sim_out_pctr_adaptive_factor",
        "impression_audit_white_tags",
        "high_hot_audit_white_tags",
        "last_page_hetu_one_total",
        "last_page_hetu_two_total",
        "last_page_hetu_three_total",
        "last_page_hetu_tag_total",
        "user_comment_count",
        "user_like_count",
        "user_show_count",
        "fountain_comment_ltr_model_kconf_key",
        "fountian_last_page_hetu_tag_boost",
        "fountian_last_page_hetu_three_boost",
        "fountian_last_page_hetu_two_boost",
        "fountian_last_page_hetu_one_boost",
        "skip_fullrank_ensemble_score_v8",
        "fountain_relation_interaction_retr_redis_key",
        "fountain_skip_ltr_predict_v3",
        "enable_find_v4_fountain_ltr_v3_predict",
        "fountain_skip_cal_enrich_comment_ltr_item_attr_from_redis",
        "fountain_skip_cal_enrich_comment_ltr_common_attr_from_redis",
        "fountain_skip_cal_comment_ltr",
        "uid_show_day_7",
        "fountain_comment_ltr_model_kconf_key",
        "fountain_skip_cal_enrich_comment_ltr_item_attr_from_redis_mobile",
        "fountain_skip_cal_enrich_comment_ltr_common_attr_from_redis_mobile",
        "fountain_skip_cal_comment_ltr_mobile",
        "fountain_comment_ltr_model_kconf_key_mobile",
        "fountain_ensemble_power_weight_fullrank_ltr_v4_next",
        "fountain_fullrank_next_score_debias_pow_weight",
        "fountain_enable_mmr_photo_embedding",
        "avgFinishRateList",
        "userBrowseSetHetuLevel3",
        "userBrowseSetHetuLevel1",
        "userBrowseSetHetuLevel2",
        "fountain_recent_browse_set_attr_name",
        "fountain_skip_enable_fountain_browse",
        "fountain_skip_enable_hot_browse",
        "fountain_enable_wtp_v2_vtr_as_wtd",
        "fountain_fullrank_sim_pwtd_finish_rate_power",
        "fountain_ltr_v3_duration_weight",
        "fountain_ltr_v3_ctr_weight",
        "enable_fountain_pwatch_time_sigmoid_bias",
        "enable_fountain_pwatch_time_adjust",
        "fountain_vtr_max_value_1",
        "enable_personal_time_coeff",
        "featureUserAvgWatchTime",
        "fountain_personal_sigmoid_bias",
        "fountain_ensemble_score_v2_sim_pwtd_no_bias_weight",
        "fountain_ensemble_power_weight_fullrank_cl_score",
        "enable_fountain_pwatchtime_origin",
        "fountain_ensemble_power_weight_fullrank_pwtd_watchtime_score",
        "fountain_ltr_score_wtd_weight",
        "userExpLtr",
        "userExpWtr",
        "fountain_fast_ensemble_weight_cascade_like_score",
        "fountain_fast_ensemble_weight_cascade_follow_score",
        "fountain_fullrank_bucket_dura_for_pfr",
        "fountain_fullrank_bucket_fr_for_pfr",
        "fountain_fullrank_ensemble_emp_score_fullrank_pvtr_score",
        "hetu_level_one_v2_index_cascade_list",
        "hetu_level_one_v2_index_cascade_list_size_after",
        "hetu_level_one_v2_index_cascade_list_size_before",
        "sim_one_tags",
        "sim_two_tags",
        "sim_three_tags",
        "short_interest",
        "action_interest",
        "long_interest",
        "fountain_fullrank_finish_duration_factor_max_value_v2",
        "skip_fountain_finish_rate_adjust_v2",
        "fountain_fullrank_enable_cdf_fr_smooth",
        "fountain_fullrank_fr_smooth_para",
        "skip_fountain_finish_rate_adjust_v3",
        "fountain_fullrank_not_svr_pow_weight_for_pfr",
        "fullrank_emb_kess_name_for_hate_similary_score",
        "fountain_offline_collect_retrieval_max_keys",
        "fountain_offline_collect_retrieval_max_keys_splash",
        "fountain_cascade_lvtr_sigmoid_bias",
        "fountain_cascade_enable_lvtr_sigmoid_bias_fix",
        "fountain_fullrank_lt_emb_kess_name_for_fr",
        "fountain_fullrank_lt_emb_request_slot_id",
        "fountain_fullrank_use_xtr_raw_score_plvtr_para",
        "fountain_fullrank_use_xtr_raw_score_pvtr_para",
        "fountain_fullrank_use_xtr_raw_score_psvr_para",
        "fountain_fullrank_use_xtr_raw_score_pfintr_para",
        "featureFountainIsFirstPage",
        "common_request_type",
        "cascade_linear_score_weights",
        "fountain_hot_high_photo_skip_filter_types",
        "fountain_mc_emb_kess_name_for_neg_feedback_sim_score"
      ],
      item_attrs = [
        "photo_id",
        "author__id",
        "upload_type",
        "duration_ms",
        "fullrank_sim_pvtr",
        "fullrank_sim_pwtd_playtime",
        "fullrank_ensemble_score",
        "fullrank_ensemble_score_after_adjust",
        "fullrank_sim_pcltr",
        "cascade_pctr",
        "cascade_pltr",
        "cascade_pwtr",
        "cascade_pftr",
        "cascade_plvtr",
        "cascade_psvtr",
        "cascade_pcmtr",
        "cascade_ptr",
        "cascade_pcestr",
        "cascade_pepstr",
        "cascade_pwatch_time",
        "cascade_ensemble_score",
        "cascade_score",
        "questionnaire_info__exposure_count",
        "questionnaire_info__positive_count",
        "questionnaire_info__negative_count",
        "questionnaire_info__unsure_count",
        "explore_questionnaire_info__exposure_count",
        "explore_questionnaire_info__negative_count",
        "explore_questionnaire_info__positive_count",
        "explore_questionnaire_info__unsure_count",
        "questionnaire_score",
        "dup_cluster_id",
        "sim_remove_dup_id",
        "pic_and_selfdup_id",
        "dup_cluster_id_adjust",
        "sim_remove_dup_id_adjust",
        "pic_and_selfdup_id_adjust",
        "normalized_item_seq_current",
        "cascade_cluster_id",
        "cascade_cluster_type",
        "fullrank_sim_lstr",
        "fullrank_sim_pptr",
        "fullrank_sim_pfintr",
        "fullrank_sim_pcpr",
        "fullrank_sim_pwtd_v2_playtime",
        "fullrank_detail_pcmtr",
        "fullrank_detail_pcmef",
        "fullrank_detail_pptr",
        "fullrank_detail_pepstr",
        "fullrank_detail_phtr",
        "fullrank_final_lstr",
        "item_seq_neg_discount",
        "normalized_item_seq_neg_discount",
        "dup_cluster_id_duplicate_count",
        "sim_remove_dup_id_duplicate_count",
        "pic_and_selfdup_id_duplicate_count",
        "is_merchant",
        "cascade_distill_fast_rank",
        "comment_ltr",
        "photo_avg_like_with_show",
        "photo_max_like_with_show",
        "photo_like_cnt",
        "photo_only_at_cnt",
        "photo_emoji_cnt",
        "photo_kmoji_cnt",
        "photo_qiuziyuan_cnt",
        "photo_qiuhudong_cnt",
        "photo_zhuixing_cnt",
        "photo_aicheng_cnt",
        "photo_zanshang_cnt",
        "photo_feiwenben_cnt",
        "item_seq_current",
        "hetu_level_one_v2",
        "hetu_level_one_v2_index",
        "hetu_level_two_v2",
        "hetu_level_three_v2",
        "hetu_level_five_v2",
        "hetu_tag_v2",
        "hetu_face_id_v2",
        "hetu_tag_v2_theme",
        "hetu_tag_v2_style",
        "picture_variant_attr",
        "picture_variant_attr_adjust",
        "post_score_after_adjust",
        "fullrank_ltr_v4_fountain_finish_rate",
        "fullrank_ltr_v4_fountain_next",
        "fullrank_cl_score",
        "cascade_pctr_debias",
        "cascade_pltr_debias",
        "cascade_pwtr_debias",
        "cascade_longview_score_debias",
        "cascade_psvtr_debias",
        "cascade_pwatch_time_debias",
        "xgb_ltr",
        "hetu_tag_level_info__hetu_level_one",
        "hetu_tag_level_info__hetu_face_id",
        "hetu_tag_level_info__hetu_tag",
        "hetu_tag_level_info_v2__hetu_level_one",
        "hetu_tag_level_info_v2__hetu_level_two",
        "hetu_tag_level_info_v2__hetu_level_three",
        "hetu_level_one_tag_index",
        "hetu_level_two_tag_index",
        "hetu_level_three_tag_index",
        "fullrank_sim_click_score_debias",
        "fullrank_sim_like_score_debias",
        "fullrank_sim_longview_score_no_bias_debias",
        "fullrank_sim_pwatchtime_no_bias_debias",
        "fullrank_ori_pswptr",
        "source_related_score",
        "hetu_level_one_index",
        "hetu_level_one_v2_index_cascade",
        "duration_s",
        "cascade_discount_ratio",
        "cascade_ensemble_score",
        "emp_htr",
        "long_term_interest_ee_score",
        "cascade_ftr_kai",
        "cascade_ftr_kai_duration",
        "cascade_ipw_opt_ftr",
        "cascade_slide_kai",
        "duration_perf_id",
        "fullrank_sim_pwtd_playtime",
        "fountainDurationPercent"
      ],
      item_num_limit = 10,
      for_debug_request_only = True,
    ) \
    .perflog_reason_count(
      check_point = "post_process_finish",
    ) \
    .copy_user_meta_info(
      save_flow_cpu_cost_to_attr = "post_process_fast_cpu_cost_ts",
    ) \

    return self
