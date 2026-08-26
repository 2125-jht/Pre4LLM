from dragonfly.common_leaf_dsl import LeafFlow
from dragonfly.ext.explore.explore_api_mixin import ExploreApiMixin
from dragonfly.ext.explore_life.explore_life_api_mixin import ExploreLifeApiMixin


class FountainPostprocFlow(LeafFlow, ExploreApiMixin, ExploreLifeApiMixin):
  def __init__(self, name: str, return_common_attrs: list, return_item_attrs: list, traceback_item_attrs: list):
    super().__init__(name)
    self \
      .namespace_(ns=name, nest=True) \
      .switch_("request_type") \
        .case_("fountain_splash_life") \
          .variant_splash() \
        .case_("fountain_splash_life_pic_inside") \
          .variant_splash() \
        .case_("fountain_fast_v1_life") \
          .control_diversify() \
        .case_("fountain_fast_life_pic_inside") \
          .variant_fast() \
        .default_() \
          .do_nothing() \
      .end_() \
      .get_abtest_params(
        biz_name="RECO_RPC",
        ab_params=[
          ("skip_explore_leaf_sleep", 1, "skip_explore_leaf_sleep"),
          ("explore_leaf_sleep_ms", 0, "explore_leaf_sleep_ms")
        ],
      ) \
      .sleep(
        sleep_ms="{{explore_leaf_sleep_ms}}",
        skip="{{skip_explore_leaf_sleep}}"
      ) \
      .gen_common_attr_by_lua(
        attr_map={
          "explore_reco_leaf_total_ts": "util.GetTimestamp() - prepare_begin_ts",
          "explore_reco_leaf_request_count": "1",
        },
      ) \
      .send_abtest_metrics(
        skip="{{return _IS_ONLINE_SERVICE_ == 0}}",
        metrics=[
          "explore_reco_leaf_retrieval_ts",
          "explore_reco_leaf_filter_ts",
          "explore_reco_leaf_cascading_ts",
          "explore_reco_leaf_ranking_ts",
          "explore_reco_leaf_rerank_ts",
          "explore_reco_leaf_total_ts",
          "explore_reco_leaf_request_count",
          "explore_reco_leaf_rank_model_input_count",
          "explore_reco_leaf_cascade_model_pic_input_result_count",
          "explore_reco_leaf_cascade_model_input_result_count",
        ],
        metric_name_prefix="",
      ) \
      .log_debug_info(
        common_attrs=[
          'enable_merchant_photo_calc_type',
          'high_quality_tags',
          'colossus_resp_old',
          'gamora_hetu_adjust_history_list', 'opt_card_dis_like_list', 'opt_card_like_list', 'page_index', 'user_follow_type', 'user_gender',
          'explore_vv_3d', 'explore_zero_play_days_15d', 'find_visit_days_30d', 'infer_uv_ctr',
          'uGamoraUploadDayNum30d', 'uNebulaUploadDayNum30d', 'uSexyInterestScore', 'uStandardRealShowPicAllIdList',
          'sim_user_list', 'explore_mc_hourly_xtr_debias_map_ptr', 'prerank_duration_debias_bucket',
          'xhs_hetu_memorydata_set', 'merchant_live_authors_set__memory_data', 'explore_personifed_author_boost_ptr',
          'pic_xtr_fractile_score_attr_from_redis_ptr', 'explore_pic_xtr_cluster_emp_map_ptr', 'explore_pic_fr_pxtr_pcts_ptr',
          'fountain_ensemble_weight_cascade_longterm_score', 'fountain_mc_enable_high_hot_audit_adjust', 'fountain_mc_enable_impression_audit_adjust',
          'fountain_mc_impression_audit_adjust_coeff_map_str',
          "explore_reco_leaf_total_ts", "fountain_mc_variant_weight_action_day_num",
          "explore_reco_leaf_retrieval_cpu_cost_ts",
          "explore_reco_leaf_filter_cpu_cost_ts",
          "explore_reco_leaf_cascading_cpu_cost_ts",
          "explore_reco_leaf_ranking_cpu_cost_ts",
          "explore_reco_leaf_rerank_cpu_cost_ts",
          "explore_reco_leaf_total_cpu_cost_ts",
          "uIsLifeHighActive",
          "uIsNotLifePassBy",
          "uLifeLongTermAuthorList",
          "uLifeLongTermAuthorListV2",
          "uLifePreferAuthor",
          "device_active_degree",
          "search_click_list",
          "search_click_list_timestamps",
          "search_play_list",
          "search_play_list_timestamps",
          "search_play_list_play_duration",
          "search_play_list_video_duration",
          "uHetuCategoryInterestlv1IdList",
          "uHetuCategoryInterestlv1ScoreList",
          "uHetuCategoryInterestlv2IdList",
          "uHetuCategoryInterestlv2ScoreList"
        ],
        item_attrs=[
          'audit_hot_cover_level', 'author__gender', 'author__hetu_author_tag__hetu_level_one', 'author__hetu_author_tag__hetu_level_three',
          'author__hetu_author_tag__hetu_level_two', 'caption_length', 'explore_stat__long_play_count', 'explore_stat__profile_enter_count', 'explore_stat__short_play_count',
          'hetu_tag_level_info__hetu_level_four', 'is_big_v_white_author_photo', 'is_support_author', 'live_photo_info__is_living', 'location__city_id', 'location__poi',
          'location__province_id', 'merchant_photo_cart_relation', 'mmu_img_cluster_v3', 'music', 'ocr_cover_text_word_count', 'shuffle_policy', 'tag',
          'author', 'author_age_info', 'author_high_score_v2', 'chn', 'click_count', 'click_upload_rate', 'color', 'comment_count', 'content_safety_level_with_namespace',
          'explore_stat', 'explore_stat__external_download', 'filter', 'forward_count', 'hetu_tag_level_info', 'hetu_tag_level_info_v2', 'hetu_tag_level_info_v2__hetu_level_three',
          'hot_trend_generalized_info', 'infer_gender', 'infer_gender_iter1', 'infer_year', 'infer_year_iter1', 'lda_topic', 'like_count', 'location', 'location__community_type',
          'magic_face_id', 'mmu_cluster_music_id', 'mmu_content_id', 'mmu_face_age', 'mmu_face_gender', 'mmu_img_cluster_v1', 'mmu_img_cluster_v4', 'mmu_photo_low_quality_model',
          'mod', 'music_info__music_combo_id', 'nearby_feeling', 'photo_dnn_cluster_id', 'photo_high_end_status_bits', 'report_count', 'show_level_a', 'show_level_b',
          'view_length_sum', 'colossus_ann_rank_score', 'comirec_rank_score', 'pdn_rank_score', 'retr_rank', 'is_explore_photo',
          'cascade_pdtr', 'cascade_plvtr2', 'merchant_author_in_living', 'reason', "cascade_final_index", "is_high_quality_explore_photo", "is_hotfire_yellow",
          'fountain_stats__comment_count', 'fountain_stats__follow_count', 'fountain_stats__forward_count', 'nebula_stats__comment_count', 'nebula_stats__follow_count',
          'nebula_stats__forward_count', 'thanos_stats__comment_count', 'thanos_stats__follow_count', 'thanos_stats__forward_count', 'author_grade_key'
        ],
        for_debug_request_only=True
      ) \
      .namespace_()

    # .log_debug_info(  # 为支持多 request type ，由上游指定 return attrs ，这里是对依赖检测的一个 tricky
    #   common_attrs = return_common_attrs,
    #   item_attrs = return_item_attrs + traceback_item_attrs,
    #   for_debug_request_only = True,
    # ) \
    # .namespace_()

  def control_diversify(self):
    self \
    .get_abtest_params(
      biz_name="RECO_RPC",
      ab_params = [
        ("xlife_content_control_limit_thres", 200),
        ("enable_xlife_gray_control", False),
        ("xlife_fountain_gray_control_window", 5),
        ("xlife_fountain_gray_control_max", 2),
        ("xlife_fountain_gray_control_priority", 2),
        ("enable_xlife_target_control", False),
        ("xlife_fountain_target_control_window", 5),
        ("xlife_fountain_target_control_max", 1),
        ("xlife_fountain_target_control_priority", 3),
        ("enable_rerank_hetu1_diversify", False),
        ("xlife_fountain_rerank_hetu1_diversify_winsize", 4),
        ("xlife_fountain_rerank_hetu1_diversify_max", 1),
        ("enable_xlife_fountain_hetu_cluster_diversity", False),
        ("xlife_fountain_hetu_cluster_diversity_winsize", 4),
        ("xlife_fountain_hetu_cluster_diversity_maxnum", 1),
        ("enable_life_target_hetu_new", False),
        ("life_target_hetu_version", "v1"),
        ("enable_life_fountain_minority_photo_diversity", False),
        ("life_fountain_minority_photo_diversity_winsize", 6),
        ("life_fountain_minority_photo_diversity_max_num", 1),
        ("life_fountain_minority_photo_diversity_priority", 9)
      ]
    ) \
    .if_("enable_life_target_hetu_new == 1") \
      .get_kconf_params(
        kconf_configs = [
          {
            "kconf_key": "reco.eyeshot.LifeTabTargetHetuL2Json",
            "json_path": "{{life_target_hetu_version}}",
            "export_common_attr": "target_hetu_l2_list"
          },
          {
            "kconf_key": "reco.eyeshot.LifeTabGrayHetuL2Json",
            "json_path": "{{life_target_hetu_version}}",
            "export_common_attr": "gray_hetu_l2_list"
          },
          {
            "kconf_key": "reco.eyeshot.LifeTabTargetHetuL1Json",
            "json_path": "{{life_target_hetu_version}}",
            "export_common_attr": "target_hetu_l1_list"
          },
          {
            "kconf_key": "reco.eyeshot.LifeTabGrayHetuL1Json",
            "json_path": "{{life_target_hetu_version}}",
            "export_common_attr": "gray_hetu_l1_list"
          },
        ]
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          "photo_id",
          "hetu_tag_level_info__hetu_level_one",
          "hetu_tag_level_info__hetu_level_two"
        ],
        import_common_attr = [
          "target_hetu_l2_list",
          "gray_hetu_l2_list",
          "target_hetu_l1_list",
          "gray_hetu_l1_list"
        ],
        export_item_attr = [
          "gray_target", # 灰度 + 非生活打散，生活设为pid，灰度 + 非生活设为1
          "not_life_target" # 非生活打散，灰度 + 生活设置为pid，非生活设为1
        ],
        function_name = "ContentControlDiversifyTagV2",
        class_name = "ExploreLifeLightFunctionSet"
      ) \
    .else_() \
      .get_kconf_params(
        kconf_configs = [{
          "kconf_key": "reco.eyeshot.LifeTabTargetHetu",
          "value_type": "list_int64",
          "export_common_attr": "target_hetu_list",
          "default_value": []
        }]
      ) \
      .get_kconf_params(
        kconf_configs = [{
          "kconf_key": "reco.eyeshot.LifeTabNotTargetHetu",
          "value_type": "list_int64",
          "export_common_attr": "not_target_hetu_list",
          "default_value": []
        }]
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          "photo_id",
          "hetu_tag_level_info__hetu_level_one",
          "hetu_tag_level_info__hetu_level_two"
        ],
        import_common_attr = [
          "target_hetu_list",
          "not_target_hetu_list"
        ],
        export_item_attr = [
          "gray_target", # 灰度 + 非生活打散，生活设为pid
          "not_life_target" # 非生活打散，灰度 + 生活设置为pid
        ],
        function_name = "ContentControlDiversifyTag",
        class_name = "ExploreLifeLightFunctionSet"
      ) \
    .end_() \
    .diversify_by_rules(
      name = "xlife_fountain_gray_content_control",
      range_end = "{{xlife_content_control_limit_thres}}",
      max_satisfied_pick = 10,
      rules = [
        dict(attr_name= "gray_target",
              enabled="{{enable_xlife_gray_control}}",
              window_size="{{xlife_fountain_gray_control_window}}",
              max_num="{{xlife_fountain_gray_control_max}}",
              priority="{{xlife_fountain_gray_control_priority}}"),
        dict(attr_name= "not_life_target",
              enabled="{{enable_xlife_target_control}}",
              window_size="{{xlife_fountain_target_control_window}}",
              max_num="{{xlife_fountain_target_control_max}}",
              priority="{{xlife_fountain_target_control_priority}}")
      ]
    ) \
    .set_attr_value(
      item_attrs = [{
        "name": "life_target",
        "type": "int",
        "value": 1
      }]
    ) \
    .copy_attr(
      attrs = [{
        "from_item": "photo_id",
        "to_item": "life_target"
      }],
      target_item = {"gray_target": 1}
    ) \
    .diversify_by_rules(
      name = "xlife_fountain_hetu1_content_control",
      max_satisfied_pick=10,
      range_end="{{xlife_content_control_limit_thres}}",
      rules=[
        dict(attr_name= "hetu_tag_level_info__hetu_level_one",
              enabled="{{enable_rerank_hetu1_diversify}}",
              window_size="{{xlife_fountain_rerank_hetu1_diversify_winsize}}",
              max_num="{{xlife_fountain_rerank_hetu1_diversify_max}}",
              priority = 2),
        dict(attr_name= "author__id",
              priority = 6,
              window_size = 6,
              max_num = 1),
        dict(attr_name= "hetu_sim_cluster_id",
              enabled="{{enable_xlife_fountain_hetu_cluster_diversity}}",
              window_size="{{xlife_fountain_hetu_cluster_diversity_winsize}}",
              max_num="{{xlife_fountain_hetu_cluster_diversity_maxnum}}",
              priority=3),
        dict(attr_name = "is_minority_photo",
              enabled = "{{enable_life_fountain_minority_photo_diversity}}",
              window_size = "{{life_fountain_minority_photo_diversity_winsize}}",
              max_num = "{{life_fountain_minority_photo_diversity_max_num}}",
              priority = "{{life_fountain_minority_photo_diversity_priority}}")
      ],
      target_item = {"life_target": 1}
    )
    return self

  def variant_fast(self):
    self.diversify_by_rules(
      max_satisfied_pick=6,
      top_priority=8,
      rules=[
        dict(
          attr_name="author__id",
          priority=6,
          window_size=6,
          max_num=1,
        ),
        # 一级类目：一屏最多2个相同
        dict(
          attr_name="hetu_tag_level_info__hetu_level_one",
          priority=5,
          window_size=6,
          max_num=2,
        ),
        # 二级类目：一屏最多1个相同
        dict(
          attr_name="hetu_tag_level_info__hetu_level_two",
          priority=4,
          window_size=6,
          max_num=1,
        ),
        # 在结果集的前 10 个结果中强插 2 个直播短视频， 注意该情况下非直播短视频的 "is_living" 为缺省状态
        # dict(
        #   priority=2,
        #   window_type="top",
        #   window_size=10,
        #   min_num=2,
        #   attr_name="is_living"
        # ),
        # 以上规则配置中，第一条规则的优先级高于第二条规则 (6 > 2), 规则引擎会优先保证第一条规则被满足
      ]
    )
    return self

  def variant_splash(self):
    self.diversify_by_rules(
      max_satisfied_pick=6,
      top_priority=8,
      rules=[
        dict(
          attr_name="author__id",
          priority=6,
          window_size=3,
          max_num=1,
        ),
        # 在结果集的前 10 个结果中强插 2 个直播短视频， 注意该情况下非直播短视频的 "is_living" 为缺省状态
        # dict(
        #   priority=2,
        #   window_type="top",
        #   window_size=10,
        #   min_num=2,
        #   attr_name="is_living"
        # ),
        # 以上规则配置中，第一条规则的优先级高于第二条规则 (6 > 2), 规则引擎会优先保证第一条规则被满足
      ]
    )
    return self

