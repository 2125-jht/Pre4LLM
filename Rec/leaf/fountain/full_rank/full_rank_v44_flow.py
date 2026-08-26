#!/usr/bin/env python3
# coding=utf-8

from dragonfly.common_leaf_dsl import LeafFlow
from dragonfly.ext.subdivision.subdivision_api_mixin import subdivisionApiMixin
from full_rank.full_rank_base_flow import FullRankBaseFlow, item_features, user_features, fullrank_common_attrs, fullrank_fast_attrs, fullrank_common_copy_attrs
from full_rank.ab_params import fullrank_common_params, fullrank_fast_params, fullrank_common_param_abhit, fullrank_fast_param_abhit
from full_rank.fullrank_base_features import *
from full_rank.fullrank_base_queues import fountain_variant_cluster_sort_queue, fountain_rerank_pre_filter_queues
from util import enrich_ab_param


class FullRankV44Flow(FullRankBaseFlow, subdivisionApiMixin):
  def __init__(self):
    LeafFlow.__init__(self, "full_rank_v44")
    self \
      .namespace_(ns = "full_rank_v44", nest = True) \
      ._timestamp_begin("rank_fast") \
      ._rank() \
      .if_("fountain_fullrank_enable_adjust_post_processing == 1") \
        ._fountain_fullrank_adjust_post_processing() \
      .else_() \
        ._diversity() \
        ._truncate() \
      .end_if_() \
      ._rank_stage2_count_distribution() \
      .namespace_()

  def fullrank_rrwtd_predict_fast(self):
    """
    仅在非首屏生效的rerank wtd模型
    """
    self \
      .if_("enable_fountain_deep_rrwtd_predict == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .delegate_enrich(
          kess_service = "{{fountain_fullrank_deep_rrwtd_kess_service}}",
          recv_item_attrs = [
            {"name": "point_pos", "as": "fullrank_rrwtd_score"},
            {"name": "point_next", "as": "fullrank_rrnext_score"},
          ],
          timeout_ms = 100,
          send_item_attrs = rr_photo_features,
          send_common_attrs = rr_user_feature,
          request_type = "{{fountain_deep_rrwtd_request_type}}",
          partition_size = "{{fountain_deep_rrwtd_partition_size}}",
        ) \
      .end_if_()
    return self

  def enrich_merchant_rank_model_attr(self):
    self \
    .if_("merchant_buyer_type ~= nil and merchant_buyer_type >= fountain_rank_merchant_need_request_buyer_type") \
      .if_("enable_fountain_rank_merchant_vedio_predict == 1") \
        .delegate_enrich(
          kess_service = "{{fountain_rank_merchant_vedio_service}}",
          send_common_attrs = [
              { "name": "kuibaUserAttrStr", "as": "user_info_str" },
          ],
          recv_item_attrs = [
            {"name": "pcart_ctr", "as": "merchant_pcart_ctr"},
            {"name": "cvr", "as": "merchant_cart_cvr"},
            {"name": "gmv_fen", "as": "merchant_cart_gmv_fen"}
          ],
          timeout_ms = 100,
          request_type = "{{fountain_rank_merchant_vedio_request_type}}",
          partition_size = "{{fountain_rank_merchant_vedio_partition_size}}",
          target_item={"is_merchant_cart": 1}
        ) \
      .end_() \
      .if_("enable_fountain_rank_merchant_living_predict == 1") \
        .get_merchant_living_item_attr_by_distributed_index(
          photo_store_kconf_key = "reco.distributedIndex.exploreMerchantLivingPhotoStoreConfig",
          use_dynamic_photo_store = True,
          photo_store_rpc_req_cache_rate = 0,
          attrs = ["s_eshop_first_live_id"],
          item_id_attr = "merchant_author_in_living",
        ) \
        .delegate_enrich(
          kess_service = "{{fountain_rank_merchant_living_service}}",
          send_common_attrs = [
            { "name": "kuibaUserAttrStr", "as": "user_info_str" },
          ],
          send_item_attrs = [
            {"name": "s_eshop_first_live_id", "as": "leaf_living_pId"}
          ],
          recv_item_attrs = [
            {"name": "elive_ctr", "as": "merchant_elive_ctr"},
            {"name": "elive_cvr", "as": "merchant_elive_cvr"},
            {"name": "elive_price", "as": "merchant_elive_price"}
          ],
          timeout_ms = 100,
          request_type = "{{fountain_rank_merchant_living_request_type}}",
          partition_size = "{{fountain_rank_merchant_living_partition_size}}",
          target_item={"is_merchant_living": 1}
        ) \
      .end_() \
    .end_()
    return self

  def enrich_living_rank_model_attr(self):
    self \
    .if_("enable_fountain_rank_living_predict == 1") \
    .enrich_attr_by_light_function(
      import_item_attr = [
        {"name": "hetu_tag_level_info_v2__hetu_level_one", "as": "extract_hetu_tag_list"},
      ],
      export_item_attr = [
        {"name": "first_hetu_tag_id", "as": "hetu_v2_level_one_top1_tag"}
      ],
      function_name = "ExtractFirstHetuV2Tag",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .pack_item_attr(
      item_source = {
        "reco_results": True,
      },
      mappings = [
        {
          "from_item_attr": "photo_id",
          "to_common_attr": "living_ranking_carm_pids",
          "default_val": 0,
        },
        {
          "from_item_attr": "author__id",
          "to_common_attr": "living_ranking_carm_aids",
          "default_val": 0,
        },
        {
          "from_item_attr": "live_photo_info__live_id",
          "to_common_attr": "living_ranking_carm_liveids",
          "default_val": 0,
        },
        {
          "from_item_attr": "duration_s",
          "to_common_attr": "living_ranking_carm_durations",
          "default_val": 0,
        },
        {
          "from_item_attr": "hetu_v2_level_one_top1_tag",
          "to_common_attr": "living_ranking_carm_hetu_tags",
          "default_val": 0,
        },
        {
          "from_item_attr": "hetu_tag_level_info_v2__hetu_cluster_id",
          "to_common_attr": "living_ranking_carm_clusters",
          "default_val": 0,
        },
      ],
    ) \
    .set_attr_default_value(
      item_attrs = [
        {
          "name" : "lp_fc_inside_wtr_list",
          "type" : "double_list",
          "value" : [0.0]
        },
        {
          "name" : "lp_fc_living_ctr_v1_list",
          "type" : "double_list",
          "value" : [0.0]
        },
        {
          "name" : "plive_ctr_list",
          "type" : "double_list",
          "value" : [0.0]
        },
        {
          "name" : "plive_wtr_list",
          "type" : "double_list",
          "value" : [0.0]
        },
        {
          "name" : "plive_ltr_list",
          "type" : "double_list",
          "value" : [0.0]
        },
        {
          "name" : "plive_watch5s_list",
          "type" : "double_list",
          "value" : [0.0]
        },
        {
          "name" : "plive_watch20s_list",
          "type" : "double_list",
          "value" : [0.0]
        },
      ]
    ) \
    .pack_item_attr_to_item_attr(
      from_item_attrs = ["fullrank_sim_pevtr"],
      to_item_attr="fullrank_sim_pevtr_list",
      default_val=0.0
    ) \
    .pack_item_attr_to_item_attr(
      from_item_attrs = ["fullrank_sim_pltr"],
      to_item_attr="fullrank_sim_pltr_list",
      default_val=0.0
    ) \
    .pack_item_attr_to_item_attr(
      from_item_attrs = ["fullrank_sim_pwtr"],
      to_item_attr="fullrank_sim_pwtr_list",
      default_val=0.0
    ) \
    .pack_item_attr_to_item_attr(
      from_item_attrs = ["fullrank_sim_psvr"],
      to_item_attr="fullrank_sim_psvr_list",
      default_val=0.0
    ) \
    .pack_item_attr_to_item_attr(
      from_item_attrs = ["fullrank_sim_pftr"],
      to_item_attr="fullrank_sim_pftr_list",
      default_val=0.0
    ) \
    .pack_item_attr_to_item_attr(
      from_item_attrs = ["fullrank_sim_pptr"],
      to_item_attr="fullrank_sim_pptr_list",
      default_val=0.0
    ) \
    .pack_item_attr_to_item_attr(
      from_item_attrs = ["fullrank_sim_plvtr"],
      to_item_attr="fullrank_sim_plvtr_list",
      default_val=0.0
    ) \
    .cast_attr_type(
      attr_type_cast_configs = [
        {
          "to_type": "double",
          "from_item_attr": "is_photo_author_followed",
          "to_item_attr": "is_follow_author_double"
        }
      ]
    ) \
    .set_attr_value(
      common_attrs = [
        {
          "name": "living_revenue_channel",
          "type": "string",
          "value": "jingxuan",
        },
      ],
    ) \
    .dispatch_common_attr(
      from_common_attr = "living_revenue_channel",
      to_item_attr = "living_revenue_channel",
    ) \
    .delegate_enrich(
      kess_service = "{{fountain_living_revenue_predict_service}}",
      timeout_ms = "{{fountain_living_revenue_predict_timeout_ms}}",
      request_type = "{{fountain_living_revenue_predict_request_type}}",
      send_common_attrs = [
        { "name": "living_ranking_carm_pids", "as": "carm_pids" },
        { "name": "living_ranking_carm_aids", "as": "carm_aids" },
        { "name": "living_ranking_carm_liveids", "as": "carm_liveids" },
        { "name": "living_ranking_carm_durations", "as": "carm_durations" },
        { "name": "living_ranking_carm_hetu_tags", "as": "carm_hetu_tags" },
        { "name": "living_ranking_carm_clusters", "as": "carm_clusters" },
      ],

      send_item_attrs = [
      { "name": "living_revenue_channel", "as": "pChannel" },
      { "name": "reason", "as": "pReason" },
      { "name": "live_photo_info__live_id", "as": "leaf_living_pId" },
      { "name": "is_follow_author_double", "as": "is_follow_author" },
      { "name": "fullrank_sim_pevtr_list", "as": "sl_mc_pctr" },
      { "name": "fullrank_sim_pltr_list", "as": "sl_mc_pltr" },
      { "name": "fullrank_sim_pwtr_list", "as": "sl_mc_pwtr" },
      { "name": "fullrank_sim_psvr_list", "as": "sl_mc_psvr" },
      { "name": "fullrank_sim_pftr_list", "as": "sl_mc_pftr" },
      { "name": "fullrank_sim_pptr_list", "as": "sl_mc_pptr" },
      { "name": "fullrank_sim_plvtr_list", "as": "sl_mc_plvtr" },
      { "name": "lp_fc_inside_wtr_list", "as": "lp_fc_inside_wtr" },
      { "name": "lp_fc_living_ctr_v1_list", "as": "pwtl2" },
      { "name": "plive_ctr_list", "as": "cascade_plive_ctr" },
      { "name": "plive_wtr_list", "as": "cascade_plive_wtr" },
      { "name": "plive_ltr_list", "as": "cascade_plive_ltr" },
      { "name": "plive_watch5s_list", "as": "cascade_plive_svr" },
      { "name": "plive_watch20s_list", "as": "cascade_plive_lvr" },
      ],
      use_sample_list_attr_flag = True,
      sample_list_ptr_attr = "kuiba_user_attr",
      recv_item_attrs = [
        { "name": "living_ctr", "as": "fr_living_ctr" },
        { "name": "living_lwtr", "as": "fr_living_lwtr" },
        { "name": "living_gtr", "as": "fr_living_gtr" },
        { "name": "living_lvtr", "as": "fr_living_lvtr" },
        { "name": "living_ltr", "as": "fr_living_ltr" },
      ],
      partition_size = "{{fountain_living_revenue_predict_partition_size}}",
      target_item={"is_true_living": 1}
    ) \
    .end_()
    return self

  def calc_merchant_rank_model_attr_score(self):
    self \
    .enrich_attr_by_light_function(
      skip = "{{return enable_fountain_rank_merchant_vedio_predict == 0}}",
      import_item_attr = [
        { "name": "merchant_pcart_ctr", "as": "ctr_input" },
        { "name": "merchant_cart_cvr", "as": "cvr_input" },
        { "name": "merchant_cart_gmv_fen", "as": "gmv_input" }
      ],
      export_item_attr = [
        { "name": "ctcvr_gmv_out", "as": "merchant_fr_photo_gmv_score" },
      ],
      function_name = "CalFrMerchantFountainCartCtcvr",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      skip = "{{return enable_fountain_rank_merchant_living_predict == 0}}",
      import_item_attr = [
        { "name": "merchant_elive_ctr", "as": "ctr_input" },
        { "name": "merchant_elive_cvr", "as": "cvr_input" },
        { "name": "merchant_elive_price", "as": "gmv_input" }
      ],
      export_item_attr = [
        { "name": "ctcvr_out", "as": "merchant_fr_living_ctcvr_score"},
        { "name": "ctcvr_gmv_out", "as": "merchant_fr_living_ctcvr_gmv_score"},
      ],
      function_name = "CalFrMerchantFountainlivingCtcvr",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def calc_living_rank_model_attr_score(self):
    self \
    .if_("is_live_big_g_user == 1") \
      .copy_attr(
        attrs=[
          {
            "from_common": "fountain_living_mix_score_ctr_weight_big_g",
            "to_common": "fountain_living_mix_score_ctr_weight"
          },
          {
            "from_common": "fountain_living_mix_score_lwtr_weight_big_g",
            "to_common": "fountain_living_mix_score_lwtr_weight"
          },
          {
            "from_common": "fountain_living_mix_score_gtr_weight_big_g",
            "to_common": "fountain_living_mix_score_gtr_weight"
          },
          {
            "from_common": "fountain_living_mix_score_lvtr_weight_big_g",
            "to_common": "fountain_living_mix_score_lvtr_weight"
          },
          {
            "from_common": "fountain_living_mix_score_ltr_weight_big_g",
            "to_common": "fountain_living_mix_score_ltr_weight"
          },
          {
            "from_common": "fountain_living_mix_score_ther_big_g",
            "to_common": "fountain_living_mix_score_ther"
          },
        ]
      ) \
    .else_if_("is_live_high_paying_user == 1") \
      .copy_attr(
        attrs=[
          {
            "from_common": "fountain_living_mix_score_ctr_weight_high_paying_user",
            "to_common": "fountain_living_mix_score_ctr_weight"
          },
          {
            "from_common": "fountain_living_mix_score_lwtr_weight_high_paying_user",
            "to_common": "fountain_living_mix_score_lwtr_weight"
          },
          {
            "from_common": "fountain_living_mix_score_gtr_weight_high_paying_user",
            "to_common": "fountain_living_mix_score_gtr_weight"
          },
          {
            "from_common": "fountain_living_mix_score_lvtr_weight_high_paying_user",
            "to_common": "fountain_living_mix_score_lvtr_weight"
          },
          {
            "from_common": "fountain_living_mix_score_ltr_weight_high_paying_user",
            "to_common": "fountain_living_mix_score_ltr_weight"
          },
          {
            "from_common": "fountain_living_mix_score_ther_high_paying_user",
            "to_common": "fountain_living_mix_score_ther"
          },
        ]
      ) \
    .else_if_("is_live_paying_user == 1") \
      .copy_attr(
        attrs=[
          {
            "from_common": "fountain_living_mix_score_ctr_weight_paying_user",
            "to_common": "fountain_living_mix_score_ctr_weight"
          },
          {
            "from_common": "fountain_living_mix_score_lwtr_weight_paying_user",
            "to_common": "fountain_living_mix_score_lwtr_weight"
          },
          {
            "from_common": "fountain_living_mix_score_gtr_weight_paying_user",
            "to_common": "fountain_living_mix_score_gtr_weight"
          },
          {
            "from_common": "fountain_living_mix_score_lvtr_weight_paying_user",
            "to_common": "fountain_living_mix_score_lvtr_weight"
          },
          {
            "from_common": "fountain_living_mix_score_ltr_weight_paying_user",
            "to_common": "fountain_living_mix_score_ltr_weight"
          },
          {
            "from_common": "fountain_living_mix_score_ther_paying_user",
            "to_common": "fountain_living_mix_score_ther"
          },
        ]
      ) \
    .else_() \
      .copy_attr(
        attrs=[
          {
            "from_common": "fountain_living_mix_score_ctr_weight",
            "to_common": "fountain_living_mix_score_ctr_weight"
          },
          {
            "from_common": "fountain_living_mix_score_lwtr_weight",
            "to_common": "fountain_living_mix_score_lwtr_weight"
          },
          {
            "from_common": "fountain_living_mix_score_gtr_weight",
            "to_common": "fountain_living_mix_score_gtr_weight"
          },
          {
            "from_common": "fountain_living_mix_score_lvtr_weight",
            "to_common": "fountain_living_mix_score_lvtr_weight"
          },
          {
            "from_common": "fountain_living_mix_score_ltr_weight",
            "to_common": "fountain_living_mix_score_ltr_weight"
          },
          {
            "from_common": "fountain_living_mix_score_ther",
            "to_common": "fountain_living_mix_score_ther"
          },
        ]
      ) \
    .end_() \
    .enrich_attr_by_light_function(
      import_common_attr = [
        "fountain_living_mix_score_ctr_weight",
        "fountain_living_mix_score_lwtr_weight",
        "fountain_living_mix_score_gtr_weight",
        "fountain_living_mix_score_lvtr_weight",
        "fountain_living_mix_score_ltr_weight",
        "fountain_living_mix_score_ther",
        "fountain_living_certain_aid_boost_coeff",
        "fountain_enable_living_certain_aid_boost",
        "living_certain_aid_list"
      ],
      import_item_attr = [
        "fr_living_ctr",
        "fr_living_lwtr",
        "fr_living_gtr",
        "fr_living_lvtr",
        "fr_living_ltr",
        "author__id"
      ],
      export_item_attr = [
        "fr_living_mix_score",
      ],
      function_name = "CalFrLivingFountainMixscore",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self

  def enrich_produce_rank_model_attr(self):
    self \
    .if_("enable_fountain_rank_produce_all_switch == 1 and enable_fountain_rank_produce_vedio_predict >= 1") \
      .enrich_attr_by_light_function( # 增加全局反转开关enable_fountain_rank_produce_all_switch
        import_common_attr = [
          { "name": "enable_fountain_rank_produce_need_divuser", "as": "enable_explore_rank_produce_need_divuser" },
          { "name": "fountain_ranking_produce_real_show_photo_recent_hours", "as": "explore_ranking_produce_real_show_photo_recent_hours" },
          { "name": "fountain_ranking_produce_his_zhongcao_threholds", "as": "explore_ranking_produce_his_zhongcao_threholds" },
          { "name": "fountain_ranking_produce_his_magic_face_threholds", "as": "explore_ranking_produce_his_magic_face_threholds" },
          { "name": "userInfoPb", "as": "user_info_ptr" },
          { "name": "enable_fountain_rank_produce_need_divuser_v2", "as": "rank_produce_need_divuser_v2" },
          { "name": "uGamoraUploadDayNum30d", "as": "gamora_upload_day_num_30d" },
          { "name": "uNebulaUploadDayNum30d", "as": "nebula_upload_day_num_30d" }
        ],
        export_common_attr = [
          { "name": "ranking_need_produce_flag", "as": "fountain_ranking_need_produce_model_flag" }
        ],
        function_name = "JudgeNeedProduceModel",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .if_("(enable_fountain_rank_produce_vedio_predict == 1 and fountain_ranking_need_produce_model_flag > 0) or (enable_fountain_rank_produce_vedio_predict == 2 and fountain_produce_user_type > fountain_rank_produce_user_switch)") \
        .delegate_enrich(
          kess_service = "{{fountain_rank_produce_video_service}}",
          send_common_attrs = [
            { "name": "userInfo", "as": "user_info_str" }
          ],
          send_item_attrs = [
            { "name": "live_photo_info__is_living", "as": "living" },
            { "name": "reco_photo_info", "as": "reco_photo_info_str" }
          ],
          recv_item_attrs = [
            { "name": "mtctr", "as": "fountain_produce_rank_mtctr" },
            { "name": "twhtr", "as": "fountain_produce_rank_twhtr" },
            { "name": "mfctr", "as": "fountain_produce_rank_mfctr" },
            { "name": "mtcotr", "as": "fountain_produce_rank_mtcotr" },
            { "name": "mtjtr", "as": "fountain_produce_rank_mtjtr" },
            { "name": "mtm1", "as": "fountain_produce_rank_mtm1" },
            { "name": "uploadw", "as": "fountain_produce_rank_uploadw" },
            { "name": "uploads", "as": "fountain_produce_rank_uploads" },
            { "name": "consuv", "as": "fountain_produce_rank_consuv" },
            { "name": "consuv_v2", "as": "fountain_produce_rank_consuv_v2"},
            { "name": "consuv_public", "as": "fountain_produce_rank_consuv_public" },
          ],
          timeout_ms = "{{fountain_rank_produce_video_predict_timeout_ms}}",
          request_type = "{{fountain_rank_produce_video_request_type}}",
          partition_size = "{{fountain_rank_produce_video_partition_size}}",
        ) \
      .end_() \
    .end_()
    return self

  def calc_produce_rank_model_attr_score(self):
    self \
    .if_("enable_fountain_rank_produce_vedio_predict == 1 and fountain_ranking_need_produce_model_flag > 0") \
      .enrich_attr_by_light_function(
        import_item_attr = [
          { "name": "fountain_produce_rank_uploadw", "as": "uploadw_input" },
          { "name": "fountain_produce_rank_uploads", "as": "uploads_input" },
          { "name": "fountain_produce_rank_consuv", "as": "consuv_input" },
          { "name": "fountain_produce_rank_consuv_v2", "as": "consuv_v2_input"},
          { "name": "fountain_produce_rank_consuv_public", "as": "consuv_public_input" },
        ],
        export_item_attr = [
          { "name": "produce_upload_sum_out", "as": "fountain_produce_rank_upload_sum_score" },
          { "name": "produce_consuv_sum_out", "as": "fountain_produce_rank_consuv_sum_score" },
        ],
        function_name = "CalProduceSocre",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
    .if_("enable_fountain_rank_produce_vedio_predict == 2 and fountain_produce_user_type > fountain_rank_produce_user_switch") \
      .enrich_attr_by_light_function(
        import_item_attr = [
          { "name": "fountain_produce_rank_mtctr", "as": "produce_rank_mtctr" },
          { "name": "fountain_produce_rank_twhtr", "as": "produce_rank_twhtr" },
          { "name": "fountain_produce_rank_mfctr", "as": "produce_rank_mfctr" },
          { "name": "fountain_produce_rank_mtcotr", "as": "produce_rank_mtcotr" },
          { "name": "fountain_produce_rank_mtjtr", "as": "produce_rank_mtjtr" },
          { "name": "fountain_produce_rank_mtm1", "as": "produce_rank_mtm1" },
          { "name": "fountain_produce_rank_uploadw", "as": "produce_rank_uploadw" },
          { "name": "fountain_produce_rank_uploads", "as": "produce_rank_uploads" },
          { "name": "fountain_produce_rank_consuv", "as": "produce_rank_consuv" },
          { "name": "fountain_produce_rank_consuv_v2", "as": "produce_rank_consuv_v2" },
          { "name": "fountain_produce_rank_consuv_public", "as": "produce_rank_consuv_public" },
        ],
        import_common_attr = [
          { "name": "fountain_produce_user_type", "as": "produce_user_type" },
          { "name": "fountain_rank_new_user_mtctr_wgt", "as": "new_user_mtctr_wgt" },
          { "name": "fountain_rank_new_user_twhtr_wgt", "as": "new_user_twhtr_wgt" },
          { "name": "fountain_rank_new_user_mfctr_wgt", "as": "new_user_mfctr_wgt" },
          { "name": "fountain_rank_new_user_mtcotr_wgt", "as": "new_user_mtcotr_wgt" },
          { "name": "fountain_rank_new_user_mtjtr_wgt", "as": "new_user_mtjtr_wgt" },
          { "name": "fountain_rank_new_user_mtm1_wgt", "as": "new_user_mtm1_wgt" },
          { "name": "fountain_rank_new_user_uploadw_wgt", "as": "new_user_uploadw_wgt" },
          { "name": "fountain_rank_new_user_uploads_wgt", "as": "new_user_uploads_wgt" },
          { "name": "fountain_rank_new_user_consuv_wgt", "as": "new_user_consuv_wgt" },
          { "name": "fountain_rank_new_user_consuv_v2_wgt", "as": "new_user_consuv_v2_wgt" },
          { "name": "fountain_rank_new_user_consuv_public_wgt", "as": "new_user_consuv_public_wgt" },
          { "name": "fountain_rank_month_user_mtctr_wgt", "as": "month_user_mtctr_wgt" },
          { "name": "fountain_rank_month_user_twhtr_wgt", "as": "month_user_twhtr_wgt" },
          { "name": "fountain_rank_month_user_mfctr_wgt", "as": "month_user_mfctr_wgt" },
          { "name": "fountain_rank_month_user_mtcotr_wgt", "as": "month_user_mtcotr_wgt" },
          { "name": "fountain_rank_month_user_mtjtr_wgt", "as": "month_user_mtjtr_wgt" },
          { "name": "fountain_rank_month_user_mtm1_wgt", "as": "month_user_mtm1_wgt" },
          { "name": "fountain_rank_month_user_uploadw_wgt", "as": "month_user_uploadw_wgt" },
          { "name": "fountain_rank_month_user_uploads_wgt", "as": "month_user_uploads_wgt" },
          { "name": "fountain_rank_month_user_consuv_wgt", "as": "month_user_consuv_wgt" },
          { "name": "fountain_rank_month_user_consuv_v2_wgt", "as": "month_user_consuv_v2_wgt" },
          { "name": "fountain_rank_month_user_consuv_public_wgt", "as": "month_user_consuv_public_wgt" },
          { "name": "fountain_rank_weeks_user_mtctr_wgt", "as": "weeks_user_mtctr_wgt" },
          { "name": "fountain_rank_weeks_user_twhtr_wgt", "as": "weeks_user_twhtr_wgt" },
          { "name": "fountain_rank_weeks_user_mfctr_wgt", "as": "weeks_user_mfctr_wgt" },
          { "name": "fountain_rank_weeks_user_mtcotr_wgt", "as": "weeks_user_mtcotr_wgt" },
          { "name": "fountain_rank_weeks_user_mtjtr_wgt", "as": "weeks_user_mtjtr_wgt" },
          { "name": "fountain_rank_weeks_user_mtm1_wgt", "as": "weeks_user_mtm1_wgt" },
          { "name": "fountain_rank_weeks_user_uploadw_wgt", "as": "weeks_user_uploadw_wgt" },
          { "name": "fountain_rank_weeks_user_uploads_wgt", "as": "weeks_user_uploads_wgt" },
          { "name": "fountain_rank_weeks_user_consuv_wgt", "as": "weeks_user_consuv_wgt" },
          { "name": "fountain_rank_weeks_user_consuv_v2_wgt", "as": "weeks_user_consuv_v2_wgt" },
          { "name": "fountain_rank_weeks_user_consuv_public_wgt", "as": "weeks_user_consuv_public_wgt" },
          { "name": "fountain_rank_week_user_mtctr_wgt", "as": "week_user_mtctr_wgt" },
          { "name": "fountain_rank_week_user_twhtr_wgt", "as": "week_user_twhtr_wgt" },
          { "name": "fountain_rank_week_user_mfctr_wgt", "as": "week_user_mfctr_wgt" },
          { "name": "fountain_rank_week_user_mtcotr_wgt", "as": "week_user_mtcotr_wgt" },
          { "name": "fountain_rank_week_user_mtjtr_wgt", "as": "week_user_mtjtr_wgt" },
          { "name": "fountain_rank_week_user_mtm1_wgt", "as": "week_user_mtm1_wgt" },
          { "name": "fountain_rank_week_user_uploadw_wgt", "as": "week_user_uploadw_wgt" },
          { "name": "fountain_rank_week_user_uploads_wgt", "as": "week_user_uploads_wgt" },
          { "name": "fountain_rank_week_user_consuv_wgt", "as": "week_user_consuv_wgt" },
          { "name": "fountain_rank_week_user_consuv_v2_wgt", "as": "week_user_consuv_v2_wgt" },
          { "name": "fountain_rank_week_user_consuv_public_wgt", "as": "week_user_consuv_public_wgt" },
          { "name": "fountain_new_user_clk_qua_score", "as": "new_user_clk_qua_score" },
          { "name": "fountain_month_user_clk_qua_score", "as": "month_user_clk_qua_score" },
          { "name": "fountain_weeks_user_clk_qua_score", "as": "weeks_user_clk_qua_score" },
          { "name": "fountain_week_user_clk_qua_score", "as": "week_user_clk_qua_score" },
          { "name": "fountain_new_user_ups_qua_score", "as": "new_user_ups_qua_score" },
          { "name": "fountain_month_user_ups_qua_score", "as": "month_user_ups_qua_score" },
          { "name": "fountain_weeks_user_ups_qua_score", "as": "weeks_user_ups_qua_score" },
          { "name": "fountain_week_user_ups_qua_score", "as": "week_user_ups_qua_score" },
        ],
        export_item_attr = [
          { "name": "produce_rank_new_user_clk_score", "as": "fountain_produce_rank_new_user_clk_score" },
          { "name": "produce_rank_month_user_clk_score", "as": "fountain_produce_rank_month_user_clk_score" },
          { "name": "produce_rank_weeks_user_clk_score", "as": "fountain_produce_rank_weeks_user_clk_score" },
          { "name": "produce_rank_week_user_clk_score", "as": "fountain_produce_rank_week_user_clk_score" },
          { "name": "produce_rank_new_user_ups_score", "as": "fountain_produce_rank_new_user_ups_score" },
          { "name": "produce_rank_month_user_ups_score", "as": "fountain_produce_rank_month_user_ups_score" },
          { "name": "produce_rank_weeks_user_ups_score", "as": "fountain_produce_rank_weeks_user_ups_score" },
          { "name": "produce_rank_week_user_ups_score", "as": "fountain_produce_rank_week_user_ups_score" },
        ],
        function_name = "CalProduceRankScore",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_()
    return self

  def enrich_plc_rank_model_attr_fast(self):
    self \
    .if_("enable_fountain_plc_rank_model_predict == 1") \
      .explore_custom_trim_user_info(
        user_info_attr = "userInfo",
        save_trimed_user_info_to_attr = "fullrank_plc_trimmed_user_info",
        trim_user_info = [
          "id",
          "device_id",
          "follow_count",
          "like_count",
          "upload_count",
          "basic_info.age_segment",
          "basic_info.gender",
          "gender",
          "true_year",
          "true_gender",
          "infer_gender",
          "app_version"
          "active_days",
          "location.city_id",
          "location.region_type",
          "client_id",
          "fans_count",
          "request_location.poi_type",
          "request_location.province_id",
          "request_location.city_id",
          "visit_mod",
          "user_profile_v1.click_list.author_id",
          "user_profile_v1.click_list.photo_id",
          "user_profile_v1.follow_list.author_id",
          "user_profile_v1.follow_list.photo_id",
          "user_profile_v1.like_list.author_id",
          "user_profile_v1.like_list.photo_id",
          "user_profile_v1.video_playing_stat.playing_time",
          "user_profile_v1.video_playing_stat.author_id",
          "user_profile_v1.video_playing_stat.photo_id",
        ],
      ) \
      .enrich_attr_by_light_function(
        export_item_attr = [
          "is_plc_item",
        ],
        import_common_attr = [
          "fountain_plc_business_type_predict_str",
        ],
        import_item_attr = [
          "plc_business_type",
        ],
        function_name = "IsPlcItemAttr",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .delegate_enrich(
        kess_service = "{{fountain_plc_rank_model_service}}",
        send_common_attrs = [
          { "name": "fullrank_plc_trimmed_user_info", "as": "user_info_str" }
        ],
        send_item_attrs = [
          { "name": "live_photo_info__is_living", "as": "living" },
          { "name": "fullrank_sim_pevtr", "as": "pctr"},
          { "name": "fullrank_sim_pltr", "as": "pltr"},
          { "name": "fullrank_sim_pwtr", "as": "pwtr"},
          { "name": "fullrank_sim_pftr", "as": "pftr"},
          { "name": "fullrank_sim_plvtr", "as": "plvtr"},
          { "name": "fullrank_sim_pvtr", "as": "pvtr"},
          { "name": "fullrank_sim_pptr", "as": "pptr"},
          { "name": "fullrank_sim_pcmtr", "as": "pcmtr"},
          "cascade_pctr",
          "cascade_pltr",
          "cascade_pwtr",
          "cascade_pftr",
          "cascade_plvtr",
          {"name": "cascade_psvtr", "as": "cascade_psvr"},
          "reason",
        ],
        recv_item_attrs = [
          {"name": "plc_click_predict", "as": "fountain_plc_ctr"},
        ],
        target_item = {"is_plc_item": 1},
        timeout_ms = 100,
        request_type = "{{fountain_plc_rank_model_request_type}}",
        partition_size = "{{fountain_plc_rank_model_partition_size}}",
      ) \
    .end_()
    return self

  def related_score_weight_adjust_only_fast(self):
    self \
    .gen_common_attr_by_lua(
      attr_map = {
        "fountain_fullrank_source_related_score_weight": "fountain_fullrank_source_related_score_weight * fullrank_user_intn_rate",
      }
    ) \
    .if_("enable_fullrank_source_related_score_weight_adjust_by_page == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "fountain_fullrank_source_related_score_weight": "fountain_fullrank_source_related_score_weight * fountain_related_weight_page_decay",
        }
      ) \
    .end_()
    return self

  def fr_fast_weight_duration_longview_adjust_watchtime_queues(self):
    self \
    .if_("enable_fountain_fr_duration_longview_adjust_pfintr == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "fountain_ensemble_power_weight_fullrank_pfintr_score": "fountain_ensemble_power_weight_fullrank_pfintr_score * fountain_duration_longview_adjust_weight",
        }
      ) \
    .end_() \
    .if_("enable_fountain_fr_duration_longview_adjust_pfintr_raw_weight == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "fountain_fullrank_ensemble_pfintr_raw_weight_attr": "fountain_fullrank_ensemble_pfintr_raw_weight_attr * fountain_duration_longview_adjust_weight",
        }
      ) \
    .end_() \
    .if_("enable_fountain_fr_duration_longview_adjust_pvtr == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "fountain_ensemble_power_weight_fullrank_pvtr_score": "fountain_ensemble_power_weight_fullrank_pvtr_score * fountain_duration_longview_adjust_weight",
        }
      ) \
    .end_() \
    .if_("enable_fountain_fr_duration_longview_adjust_plvtr == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "fountain_ensemble_weight_fullrank_sim_plvtr": "fountain_ensemble_weight_fullrank_sim_plvtr * fountain_duration_longview_adjust_weight",
        }
      ) \
    .end_() \
    .if_("enable_fountain_fr_duration_longview_adjust_pfr == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "fountain_ensemble_power_weight_fullrank_dura_cdf_pfr": "fountain_ensemble_power_weight_fullrank_dura_cdf_pfr * fountain_duration_longview_adjust_weight",
        }
      ) \
    .end_()
    return self
  
  def _his_cur_playtime_trend_adjust(self):
    self \
    .enrich_attr_by_light_function(
      import_common_attr = [
        "colossus_play_time_list",
        {"name": "fountain_playtime_trend_adjust_his_num", "as": "playtime_trend_adjust_his_num"},
        {"name": "fountain_playtime_trend_adjust_recent_num", "as": "playtime_trend_adjust_recent_num"},
        {"name": "fountain_playtime_trend_adjust_weight_alpha", "as": "playtime_trend_adjust_weight_alpha"},
        {"name": "fountain_playtime_trend_adjust_weight_beta", "as": "playtime_trend_adjust_weight_beta"},
        {"name": "fountain_playtime_trend_adjust_weight_max", "as": "playtime_trend_adjust_weight_max"},
        {"name": "fountain_playtime_trend_adjust_weight_min", "as": "playtime_trend_adjust_weight_min"},
      ],
      export_common_attr = ["fountain_playtime_trend_adjust_weight"],
      function_name = "HisCurPlaytimeTrendAdjust",
      class_name = "ExploreLightFunctionSetV2",
    )
    return self
  
  def fr_fast_weight_playtime_trend_adjust_watchtime_queues(self):
    self \
    .if_("enable_fountain_fr_playtime_trend_adjust_pfintr == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "fountain_ensemble_power_weight_fullrank_pfintr_score": "fountain_ensemble_power_weight_fullrank_pfintr_score * fountain_playtime_trend_adjust_weight",
        }
      ) \
    .end_() \
    .if_("enable_fountain_fr_playtime_trend_adjust_pvtr == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "fountain_ensemble_power_weight_fullrank_pvtr_score": "fountain_ensemble_power_weight_fullrank_pvtr_score * fountain_playtime_trend_adjust_weight",
        }
      ) \
    .end_() \
    .if_("enable_fountain_fr_playtime_trend_adjust_plvtr == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "fountain_ensemble_weight_fullrank_sim_plvtr": "fountain_ensemble_weight_fullrank_sim_plvtr * fountain_playtime_trend_adjust_weight",
        }
      ) \
    .end_() \
    .if_("enable_fountain_fr_playtime_trend_adjust_pfr == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "fountain_ensemble_power_weight_fullrank_dura_cdf_pfr": "fountain_ensemble_power_weight_fullrank_dura_cdf_pfr * fountain_playtime_trend_adjust_weight",
        }
      ) \
    .end_()
    return self
  
  def fr_fast_weight_duration_longview_adjust_interaction_queues(self):
    self \
    .if_("enable_fountain_fr_duration_longview_adjust_like_score == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "fountain_ensemble_power_weight_fullrank_like_score": "fountain_ensemble_power_weight_fullrank_like_score / fountain_duration_longview_adjust_weight",
        }
      ) \
    .end_() \
    .if_("enable_fountain_fr_duration_longview_adjust_follow_score == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "fountain_ensemble_power_weight_fullrank_follow_score": "fountain_ensemble_power_weight_fullrank_follow_score / fountain_duration_longview_adjust_weight",
        }
      ) \
    .end_() \
    .if_("enable_fountain_fr_duration_longview_adjust_forward_score == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "fountain_ensemble_weight_forward_score": "fountain_ensemble_weight_forward_score / fountain_duration_longview_adjust_weight",
        }
      ) \
    .end_() \
    .if_("enable_fountain_fr_duration_longview_adjust_comment_score == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "fountain_ensemble_power_weight_fullrank_pcmtr_score": "fountain_ensemble_power_weight_fullrank_pcmtr_score / fountain_duration_longview_adjust_weight",
        }
      ) \
    .end_() \
    .if_("enable_fountain_fr_duration_longview_adjust_collect_score == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "fountain_ensemble_power_weight_fullrank_pcltr_score": "fountain_ensemble_power_weight_fullrank_pcltr_score / fountain_duration_longview_adjust_weight",
        }
      ) \
    .end_()
    return self

  def _rank(self):
    self \
    .get_abtest_params(
      biz_name = "RECO_RPC",
      ab_params = enrich_ab_param(fullrank_common_params + fullrank_fast_params),
      prioritized_suffix = "{{_ABTEST_SUFFIX_LIST_}}"
    ) \
    .get_abtest_params(
      biz_name = "RECO_RPC",
      ab_params = fullrank_common_param_abhit + fullrank_fast_param_abhit,
      prioritized_suffix = "{{_ABTEST_SUFFIX_LIST_}}"
    ) \
    .if_("enable_replace_colossus_v2_from_fr == 1") \
      .copy_attr(
        attrs = [
          {
            "from_common": "colossus_v2_resp",
            "to_common": "colossus_resp_v2",
          },
        ],
      ) \
    .end_() \
    .if_("fountain_enable_rank_write_rank_neg_result_to_redis == 1") \
      .pack_item_attr(
        item_source = {
          "reco_results": True
        },
        mappings = [{
          "from_item_attr": "photo_id",
          "to_common_attr": "rank_candidate_photo_id_list",
        }]
      ) \
    .end_() \
    .set_attr_value(
      common_attrs = [
        {
          "name" : "featureUserIsFountainSplash",
          "type" : "int",
          "value" : 0
        },
        {
          "name" : "featureUserIsFountainRequest",
          "type" : "int",
          "value" : 1
        }
      ]
    ) \
    .disable_forward_social_queue() \
    .fetch_similar_user_list() \
    .fetch_duration_group_id() \
    .get_item_attr_by_distributed_flat_index(
      skip = "{{fountain_skip_get_fullrank_attrs_distributed}}",
      photo_store_kconf_key = "reco.distributedIndex.hotPhotoInfoCommonIndex",
      perf_log = "fullrank",
      photo_store_request_data_set_tags_attr = "fountain_request_data_set_tags",
      use_dynamic_photo_store = True,
      item_id_attr = "item_id",
      attrs = fullrank_common_attrs + fullrank_fast_attrs,
    ) \
    .copy_attr(
      attrs = fullrank_common_copy_attrs,
    ) \
    .if_("enable_fountain_transform_photo_proinc_type == 1") \
      .item_attr_operation(
        item_attr_a = "photo_proinc_type",
        common_attr_b = 8,
        operator = "&",
        output_attr = "userfulness_author_tag"
      ) \
      .cast_attr_type(
        attr_type_cast_configs = [
          {
            "to_type": "double",
            "from_item_attr": "userfulness_author_tag",
            "to_item_attr": "userfulness_author_score"
          }
        ]
      ) \
    .end_if_() \
    .if_("enable_fountain_transform_photo_proinc_type_to_authority_tag == 1") \
      .item_attr_operation(
        item_attr_a = "photo_proinc_type",
        common_attr_b = 16,
        operator = "&",
        output_attr = "authority_author_tag"
      ) \
      .cast_attr_type(
        attr_type_cast_configs = [
          {
            "to_type": "double",
            "from_item_attr": "authority_author_tag",
            "to_item_attr": "authority_author_score"
          }
        ]
      ) \
    .end_if_() \
    .if_("enable_fountain_transform_photo_proinc_type_to_expertise_tag == 1") \
      .item_attr_operation(
        item_attr_a = "photo_proinc_type",
        common_attr_b = 64,
        operator = "&",
        output_attr = "expertise_author_tag"
      ) \
      .cast_attr_type(
        attr_type_cast_configs = [
          {
            "to_type": "double",
            "from_item_attr": "expertise_author_tag",
            "to_item_attr": "expertise_author_score"
          }
        ]
      ) \
    .end_if_() \
    .split_string(
      input_common_attr = "exclude_hetu_level_one_tag_list_str",
      output_common_attr = "exclude_hetu_level_one_tag_list",
      delimiters = ",",
      parse_to_int = True
    ) \
    .if_("enable_fountain_gen_photo_original_submission_tag == 1") \
      .explore_memory_data_enrich(
        data_key = "kuaishou_official_accounts",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "kuaishou_official_account_set_ptr"
      ) \
      .enrich_attr_by_light_function(
        import_item_attr = [
          "sirius_distribution_info__mark_cod",
          "hetu_tag_level_info__hetu_level_one",
          "author__id"
        ],
        import_common_attr = [
          {"name": "enable_fountain_original_submission_combine_exclusion", "as": "enable_exclusion"},
          "kuaishou_official_account_set_ptr",
          "exclude_hetu_level_one_tag_list"
        ],
        export_item_attr = [
          "original_submission_author_tag",
        ],
        function_name = "GenPhotoOriginalSubmissionTag",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_if_() \
    .if_("enable_fountain_gen_photo_personalization_author_tag == 1") \
      .explore_memory_data_enrich(
        data_key = "kuaishou_official_accounts",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "kuaishou_official_account_set_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "personalization_authors",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "personalization_author_set_ptr"
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "personalization_author_set_ptr",
          {"name": "enable_fountain_personalization_combine_exclusion", "as": "enable_exclusion"},
          "kuaishou_official_account_set_ptr",
          "exclude_hetu_level_one_tag_list"
        ],
        import_item_attr = [
          "author__id",
          "hetu_tag_level_info__hetu_level_one",
        ],
        export_item_attr = [
          "personalization_author_tag",
        ],
        function_name = "GenPhotoPersonalizationTag",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_() \
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
      function_for_item = "calculate",
      lua_script_file = "fountain/full_rank/lua/gen_hetu_tag.lua",
      skip = "{{fountain_skip_trans_hetu_tag_item_attr}}") \
    .if_("enable_fountain_related_score_calc_v3_fast == 1") \
      .explore_transform_hetu_tag(
        output_attrs = ["hetu_level_one_v2", "hetu_level_two_v2",  "hetu_level_three_v2", "hetu_level_four_v2", "hetu_tag_v2", "hetu_face_id_v2"],
        hetu_tag_attrs = ["hetu_tag_level_info_v2__hetu_level_one", "hetu_tag_level_info_v2__hetu_level_two", "hetu_tag_level_info_v2__hetu_level_three", "hetu_tag_level_info_v2__hetu_level_four", "hetu_tag_level_info_v2__hetu_tag", "hetu_tag_level_info_v2__hetu_face_id"]
      ) \
      .fountain_calc_related_score_v2(
        enable_cal_photo_sim_by_intersect = True,
        diversity_dim_weight = "{{source_related_dim_weight_v3_fast}}",
        save_score_to_attr = "source_related_score",
        int_source_attrs = [
          "source_hetu_sim_cluster_id", "source_hetu_cluster_id_v2",
          "sourcePidMmuImgClusterV3", "sourcePidMmuTextCluster", 
          "sourcePidAuthorId", "sourcePidFirstLevelCategory",
          "sourcePidSecondLevelCategory", "sourcePidThirdLevelCategory",
          "sourcePidTagId", "sourcePidUploadType",
        ],
        int_list_source_attrs = [
          "source_hetu_level_one_v2", "source_hetu_level_two_v2",
          "source_hetu_level_three_v2", "source_hetu_level_four_v2",
          "source_hetu_tag_v2", "source_hetu_face_id_v2", "source_user_hash_tag_id",
        ],
        int_item_attrs = [
          "hetu_sim_cluster_id", "hetu_tag_level_info_v2__hetu_cluster_id",
          "mmu_img_cluster_v3", "mmu_text_cluster",
          "author__id", "author__category_detail__first_level_id",
          "author__category_detail__second_level_id", "author__category_detail__third_level_id",
          "tag", "upload_type",
        ],
        int_list_item_attrs = [
          "hetu_level_one_v2", "hetu_level_two_v2",
          "hetu_level_three_v2", "hetu_level_four_v2",
          "hetu_tag_v2", "hetu_face_id_v2", "user_hash_tag_id",
        ],
      ) \
    .end_() \
    .set_attr_default_value(
      item_attrs = [
        {
          "name": "fountain_splash_slide",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "splash_fullrank_ltr_fusion_score",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "splash_fullrank_ltr_act_score",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "splash_fullrank_ltr_act_v2_score",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "splash_fullrank_ltr_wtd_score",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "splash_fullrank_ltr_lvtr_score",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "splash_fullrank_ltr_svtr_score",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "splash_fullrank_ltr_like_score",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "splash_fullrank_ltr_follow_score",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "splash_fullrank_ltr_comment_score",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "splash_fullrank_ltr_next_score",
          "type": "double",
          "value": 0.0
        },
        {
          "name": "splash_fullrank_ltr_relate_evtr_score",
          "type": "double",
          "value": 0.0
        },
      ]
    ) \
    .calculate_comment_ltr()\
    .calc_hate_list_similary_score() \
    .if_("fountain_skip_long_term_interest_ee==0") \
      .calc_long_term_interest_ee_score(
        user_info_pb_name = "userInfoPb",
        hetu_attrs = "hetu_tag_level_info__hetu_level_one;hetu_tag_level_info__hetu_level_two;hetu_tag_level_info__hetu_level_three;hetu_tag_level_info__hetu_level_four;hetu_tag_level_info__hetu_level_five;hetu_tag_level_info__hetu_face_id;hetu_tag_level_info__hetu_tag",
        enable_click_history = "{{fountain_fullrank_enable_click_history}}",
        enable_like_history = "{{fountain_fullrank_enable_like_history}}",
        enable_follow_history = "{{fountain_fullrank_enable_follow_history}}",
        enable_long_view_history = "{{fountain_fullrank_enable_long_view_history}}",
        long_view_threshold = "{{fountain_fullrank_long_view_threshold}}",
        export_item_attr = "long_term_interest_ee_score",
        enable_division_way = "{{fountain_fullrank_enable_division_way}}",
        photo_hetu_tag_level_info_type = "{{foutnain_fullrank_photo_hetu_tag_level_info_type}}",
        boost_threshold = "{{fountain_fullrank_long_term_interest_ee_boost_threshold}}",
      ) \
      .get_kconf_params(
          kconf_configs = [{
            "kconf_key": "reco.fountain.highValueHetuList",
            "value_type": "list_int64",
            "defult_value": [134, 120, 114, 189, 220, 316, 179, 199, 325, 161, 208, 203],
            "export_common_attr": "high_value_hetu_list"
          }]
        ) \
      .enrich_attr_by_lua(
        import_common_attr = [
          "fountain_fullrank_high_value_hetu_debias_coef",
          "fountain_fullrank_enable_only_longterm_debias",
          "high_value_hetu_list",
        ],
        import_item_attr = [
          "long_term_interest_ee_score",
          "hetu_tag_level_info_v2__hetu_level_one",
          ],
        export_item_attr = [
          "long_term_interest_ee_score",
        ],
        function_for_item = "calc_fullrank_high_value_hetu_debias",
        lua_script_file = "fountain/cascade/lua/high_value_hetu_debias.lua",
      ) \
      .log_debug_info(
        common_attrs = [
          "page",
          "high_value_hetu_list",
          "fountain_fullrank_high_value_hetu_debias_coef",
          ],
        item_attrs = [
          "photo_id",
          "long_term_interest_ee_score",
          "hetu_tag_level_info_v2__hetu_level_one",
          "hetu_level_one_v2",
          "limit_hetu_table"
        ],
        item_num_limit = 10,
        for_debug_request_only = True,
      ) \
    .end_if_() \
    .if_("fountain_enable_get_pxtr_index == 1") \
      .get_cascade_index() \
    .end_() \
    .enrich_fullrank_features_by_lua() \
    .enrich_fullrank_score_attr_fast() \
    .fullrank_reco_pxtr_predict() \
    .request_feature_server() \
    .enrich_merchant_rank_model_attr() \
    .enrich_living_rank_model_attr() \
    .enrich_produce_rank_model_attr() \
    .enrich_plc_rank_model_attr_fast() \
    .fullrank_cl_ltr_predict_pre() \
    .fetch_sim_gsu_feature() \
    .fullrank_cl_ltr_predict_post() \
    .if_("fountain_enable_get_ranking_pxtr_index == 1") \
      .get_ranking_index() \
    .end_() \
    .if_("fountain_fast_rank_enable_ensemble_filter == 1 and fountain_rank_ensemble_filter_opt == 1") \
      .rank_stage1_interact_playtime_adjust() \
      .ensemble_filter("fountain_fr_stage1") \
      .rank_stage1_count_distribution() \
    .end_if_() \
    .trans_sim_pxtr_names() \
    .replace_sim_pxtr_by_reco_model() \
    .if_("fountain_enable_pxtr_calibration == 1") \
      .fountain_fullrank_pxtr_calibration() \
    .end_() \
    .fullrank_ltr_predict_fast() \
    .fullrank_rrwtd_predict_fast() \
    .enrich_fullrank_score_attr() \
    .calculate_xgb_ltr() \
    .if_("fountain_skip_user_ada_xtr_score==0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .cal_user_ada_xtr_score() \
    .cal_user_rl_xtr_score() \
    .end_if_() \
    .if_("fountain_enable_calc_topk_mgs_expected_score == 1") \
      .calc_topk_mgs_expected_score() \
    .end_if_() \
    .if_("fountain_enable_calc_interact_similarity_score == 1", to_be_delete = "date=2024-05-29;committer=lijinyu") \
      .calc_interact_similarity_score() \
    .end_if_() \
    .cal_opportunity_cost_score() \
    .cal_action_once_score() \
    .cal_cascade_linear_score() \
    .cal_value_multiply_score() \
    .if_("enable_fountain_fullrank_iput_score == 1") \
      .calc_fountain_fullrank_iput_score() \
    .end_() \
    .if_("fountain_enable_fullrank_get_hetu_behavior_score == 1") \
     .fullrank_get_hetu_behavior_score() \
    .end_() \
    .cal_satisfy_score() \
    .cal_fit_ptime_score() \
    .if_("fountain_enable_rank_triplem_time_queue == 1") \
      .cal_triplem_time_score() \
    .end_if_() \
    .if_("fountain_enable_rank_triplem_interaction_queue == 1") \
      .cal_triplem_interaction_score() \
    .end_if_() \
    .calc_debias_mix_score() \
    .calc_debias_score() \
    .calculate_debias_pxtr() \
    .fullrank_life_stage_cid_ipw_debias() \
    .fullrank_age_gender_prof_cid_ipw_debias() \
    .fullrank_age_gender_north_cid_ipw_debias() \
    .fullrank_age_gender_cid_ipw_debias() \
    .if_("enable_fountain_cal_rise_follow_boost_score_full_rank == 1") \
      ._fountain_cal_rise_follow_boost_score_full_rank() \
    .end_if_() \
    .cal_distill_fusion_score() \
    .if_("fountain_fast_rank_enable_ensemble_filter == 1 and fountain_rank_ensemble_filter_opt == 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .fullrank_ensemble_pre_filter() \
    .end_if_() \
    .calc_fr_cdf_mapping() \
    .enrich_attr_by_lua(
      import_common_attr = [
        "fountain_fullrank_fr_duration_factor_offset",
        "skip_fountain_finish_rate_adjust",
        "fountain_fullrank_finish_duration_factor_max_value",
        "fountain_fullrank_finish_duration_factor_pow_weight",
        "fountain_ensemble_power_weight_fullrank_ltr_v4_next",
        "page",
        "fountain_fullrank_next_score_debias_pow_weight",
        "skip_fountain_fullrank_ltr_v4_next_splash",
        "fountain_fullrank_enable_cdf_fr_smooth",
        "fountain_fullrank_cdf_fr_smooth_alpha",
        "fountain_fullrank_cdf_fr_smooth_beta",
        "skip_fountain_finish_rate_adjust_v3",
        "fountain_fullrank_not_svr_pow_weight_for_pfr",
        "fountain_skip_fr_pred_only_fast_v1",
       ],
      import_item_attr = [
        "fullrank_ltr_v4_fountain_finish_rate",
        "duration_ms",
        "fullrank_sim_psvr",
        "long_term_interest_ee_score",
        "fullrank_dura_cdf_pfr",
       ],
      export_common_attr = [
        "fountain_ensemble_power_weight_fullrank_ltr_v4_next",
      ],
      export_item_attr = [
        "fullrank_ltr_v4_fountain_finish_rate",
        "long_term_interest_ee_score",
        "fullrank_dura_cdf_pfr",
      ],
      function_for_item = "unify_fullrank_pxtr",
      function_for_common = "unify_fullrank_common_attr",
      lua_script_file = "fountain/full_rank/lua/unify_fullrank_pxtr.lua",
    ) \
    .if_("skip_fullrank_variant_cluster_sort == 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .if_("enable_fountain_timeout_optmize == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .if_("fountain_fullrank_variant_use_new_cluster == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .fountain_variant_cluster_sort_v2(
          size_limit = "{{fullrank_variant_cluster_sort_limit_size}}",
          global_cut_ratio = "{{fountain_fullrank_variant_cluster_global_cut_ratio}}",
          min_survival = "{{fountain_fullrank_variant_cluster_min_survival}}",
          check_point =  "{{fountain_fullrank_variant_cluster_checkpoint}}",
          cluster_sort_list_attr_name = "limit_hetu_table",
          cluster_config = "{{fountain_fullrank_variant_cluster_sort_config}}",
          enable_user_profile_config = "{{fountain_fullrank_variant_enable_profile_config}}",
          cut_ratio_decay = "{{fountain_fullrank_variant_cut_ratio_decay}}",
          cut_ratio_boost = "{{fountain_fullrank_variant_cut_ratio_boost}}",
          user_profile_hetu_count_threshold = "{{fountain_fullrank_variant_profile_hetu_threshold}}",
          neg_time_limit_second = "{{fountain_fullrank_variant_profile_neg_time}}",
          neg_play_sec_limit = "{{fountain_fullrank_variant_profile_neg_play_time}}",
          pos_time_limit_second = "{{fountain_fullrank_variant_profile_pos_time}}",
          pos_play_sec_limit = "{{fountain_fullrank_variant_profile_pos_play_time}}",
          enable_proportional = "{{fountain_fullrank_variant_enable_proportion}}",
          user_info_attr = "userInfo",
          save_score_to_attr = "fullrank_variant_cluster_score",
          use_power_calc = "{{fountain_fullrank_variant_use_power_calc}}",
          use_reciprocal = "{{fountain_fullrank_variant_use_reciprocal}}",
          use_reciprocal_v2 = "{{fountain_fullrank_variant_use_reciprocal_v2}}",
          queues = fountain_variant_cluster_sort_queue
        ) \
        .else_() \
        .fountain_variant_cluster_sort_v2(
          size_limit = "{{fullrank_variant_cluster_sort_limit_size}}",
          global_cut_ratio = "{{fountain_fullrank_variant_cluster_global_cut_ratio}}",
          min_survival = "{{fountain_fullrank_variant_cluster_min_survival}}",
          check_point =  "{{fountain_fullrank_variant_cluster_checkpoint}}",
          cluster_sort_list_attr_name = "hetu_tag_level_info__hetu_level_one",
          cluster_config = "{{fountain_fullrank_variant_cluster_sort_config}}",
          enable_user_profile_config = "{{fountain_fullrank_variant_enable_profile_config}}",
          cut_ratio_decay = "{{fountain_fullrank_variant_cut_ratio_decay}}",
          cut_ratio_boost = "{{fountain_fullrank_variant_cut_ratio_boost}}",
          user_profile_hetu_count_threshold = "{{fountain_fullrank_variant_profile_hetu_threshold}}",
          neg_time_limit_second = "{{fountain_fullrank_variant_profile_neg_time}}",
          neg_play_sec_limit = "{{fountain_fullrank_variant_profile_neg_play_time}}",
          pos_time_limit_second = "{{fountain_fullrank_variant_profile_pos_time}}",
          pos_play_sec_limit = "{{fountain_fullrank_variant_profile_pos_play_time}}",
          enable_proportional = "{{fountain_fullrank_variant_enable_proportion}}",
          user_info_attr = "userInfo",
          save_score_to_attr = "fullrank_variant_cluster_score",
          use_power_calc = "{{fountain_fullrank_variant_use_power_calc}}",
          use_reciprocal = "{{fountain_fullrank_variant_use_reciprocal}}",
          use_reciprocal_v2 = "{{fountain_fullrank_variant_use_reciprocal_v2}}",
          queues = fountain_variant_cluster_sort_queue
        ) \
        .end_() \
      .else_() \
        .fountain_variant_cluster_sort(
          size_limit = "{{fullrank_variant_cluster_sort_limit_size}}",
          global_cut_ratio = "{{fountain_fullrank_variant_cluster_global_cut_ratio}}",
          min_survival = "{{fountain_fullrank_variant_cluster_min_survival}}",
          check_point =  "{{fountain_fullrank_variant_cluster_checkpoint}}",
          cluster_sort_list_attr_name = "hetu_tag_level_info__hetu_level_one",
          cluster_config = "{{fountain_fullrank_variant_cluster_sort_config}}",
          enable_user_profile_config = "{{fountain_fullrank_variant_enable_profile_config}}",
          cut_ratio_decay = "{{fountain_fullrank_variant_cut_ratio_decay}}",
          cut_ratio_boost = "{{fountain_fullrank_variant_cut_ratio_boost}}",
          user_profile_hetu_count_threshold = "{{fountain_fullrank_variant_profile_hetu_threshold}}",
          neg_time_limit_second = "{{fountain_fullrank_variant_profile_neg_time}}",
          neg_play_sec_limit = "{{fountain_fullrank_variant_profile_neg_play_time}}",
          pos_time_limit_second = "{{fountain_fullrank_variant_profile_pos_time}}",
          pos_play_sec_limit = "{{fountain_fullrank_variant_profile_pos_play_time}}",
          enable_proportional = "{{fountain_fullrank_variant_enable_proportion}}",
          user_info_attr = "userInfo",
          save_score_to_attr = "fullrank_variant_cluster_score",
          use_power_calc = "{{fountain_fullrank_variant_use_power_calc}}",
          use_reciprocal = "{{fountain_fullrank_variant_use_reciprocal}}",
          use_reciprocal_v2 = "{{fountain_fullrank_variant_use_reciprocal_v2}}",
          queues = fountain_variant_cluster_sort_queue
        ) \
      .end_if_() \
    .end_if_() \
    .perflog_reason_count(
      check_point = "post_fullrank_cluster_sort",
    ) \
    .explore_sphinx_param() \
    .pack_item_attr(
      skip = "{{fountain_fullrank_skip_calc_pxtr_avg}}",
      item_source = {
        "reco_results": True,
      },
      mappings = [
      {
        "aggregator": "avg",
        "from_item_attr": "fullrank_sim_click_score",
        "to_common_attr": "pctr_avg"
      },
      {
        "aggregator": "avg",
        "from_item_attr": "fullrank_sim_psvr",
        "to_common_attr": "psvr_avg"
      },
      {
        "aggregator": "avg",
        "from_item_attr": "fullrank_sim_pltr",
        "to_common_attr": "pltr_avg"
      },
      {
        "aggregator": "avg",
        "from_item_attr": "fullrank_sim_pwtr",
        "to_common_attr": "pwtr_avg"
      },
      {
        "aggregator": "avg",
        "from_item_attr": "fullrank_sim_pftr",
        "to_common_attr": "pftr_avg"
      },
      {
        "aggregator": "avg",
        "from_item_attr": "fullrank_sim_pcmtr",
        "to_common_attr": "pcmtr_avg"
      },
      {
        "aggregator": "avg",
        "from_item_attr": "fullrank_sim_pptr",
        "to_common_attr": "pptr_avg"
      },
      {
        "aggregator": "avg",
        "from_item_attr": "fullrank_sim_pevtr",
        "to_common_attr": "pevtr_avg"
      },
      {
        "aggregator": "avg",
        "from_item_attr": "fullrank_detail_new_pevtr_v2",
        "to_common_attr": "pevtr_v2_avg"
      },
      {
        "aggregator": "avg",
        "from_item_attr": "fullrank_sim_plvtr",
        "to_common_attr": "plvtr_avg"
      },
      {
        "aggregator": "avg",
        "from_item_attr": "fullrank_sim_pfintr",
        "to_common_attr": "pfintr_avg"
      },
      {
        "aggregator": "avg",
        "from_item_attr": "fullrank_sim_pwatchtime_no_bias",
        "to_common_attr": "pwatchtime_avg"
      },
      ]
      ) \
    .if_("enable_use_request_based_pxtr_ada_weight == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
      .cal_request_pxtr_ada_weight() \
    .end_if_() \
    .if_("enable_cal_user_group_emp_xtr_in_rank == 1") \
      .cal_user_group_emp_xtr_all() \
    .end_if_() \
    .if_("enable_cal_user_group_dynamic_weight_in_rank == 1", to_be_delete = "date=2024-05-29;committer=xuwei09") \
      .cal_user_group_dynamic_weight() \
    .end_if_() \
    .if_("enable_cal_request_adaptive_score == 1") \
      .cal_request_adaptive_score() \
    .end_if_() \
    .if_("fountain_rank_enable_get_pxtr_boost_coef == 1") \
      .request_xtr_adap_weight() \
    .end_if_() \
    .if_("fountain_rank_enable_request_pxtr_judge == 1") \
      .cal_request_rank_weight_adjust() \
    .end_if_() \
    .calc_merchant_rank_model_attr_score() \
    .if_("enable_fountain_calc_living_rank_model_attr_score == 1") \
      .calc_living_rank_model_attr_score() \
    .end_if_() \
    .calc_produce_rank_model_attr_score() \
    .calc_plc_rank_model_attr_score() \
    .if_("enable_fountain_fr_htr_weight_adjust_by_user_htr == 1 and fountain_recent_hate_count > fountain_fr_koc_htr_count_threshold") \
      ._fullrank_htr_weight_adjust_by_uv_htr() \
    .end_if_() \
    .if_("enable_fountain_share_pull_ftr_weight_adjust_coef == 1") \
      .share_pull_ftr_weight_adjust_coef() \
    .end_() \
    .if_("enable_fountain_cal_share_pull_ftr_full_rank == 1") \
      .cal_share_pull_ftr_full_rank() \
    .end_() \
    .pack_item_attr(
      skip = "{{fountain_fullrank_skip_calc_user_intn_rate}}",
      item_source = {
        "reco_results": True,
      },
      mappings = [
        {
          "aggregator": "copy",
          "from_item_attr": "fullrank_ltr_intn_rate",
          "to_common_attr": "fullrank_user_intn_rate"
        },
      ]
    ) \
    .if_("enable_fountain_cal_related_weight_page_decay == 1") \
      .gen_common_attr_by_lua(
        attr_map = {
          "fountain_related_weight_page_decay": "math.exp(fountain_related_weight_page_decay_base * (fountain_related_weight_page_decay_from - page))",
        }
      ) \
    .end_() \
    .if_("enable_fountain_related_score_weight_adjust_only_fast == 1") \
      .related_score_weight_adjust_only_fast() \
    .end_() \
    .fr_fast_weight_duration_longview_adjust_watchtime_queues() \
    .fr_fast_weight_duration_longview_adjust_interaction_queues() \
    .if_("enable_fountain_fast_get_playtime_trend_adjust == 1") \
      ._his_cur_playtime_trend_adjust() \
    .end_() \
    .fr_fast_weight_playtime_trend_adjust_watchtime_queues() \
    .calc_fullrank_ensemble_score() \
    .enrich_attr_by_lua(
      import_item_attr = [
        "hetu_tag_level_info__hetu_level_one",
        "hetu_tag_level_info__hetu_level_two",
        "hetu_tag_level_info__hetu_level_five",
      ],
      export_item_attr = [
        "hetu_level_one",
        "hetu_level_two",
        "hetu_level_five",
      ],
      function_for_item = "calculate",
      lua_script_file = "fountain/full_rank/lua/trans_item_attr.lua",
    ) \
    .if_("enable_fountain_fullrank_es_boost == 1") \
      .if_("skip_fountain_fullrank_rrr_discount == 0") \
        .rrr_discount() \
      .end_if_() \
      .fr_ensemble_score_multiply_gate() \
      .fr_discount_single_pic() \
      .if_("skip_fullrank_negative_feedback_discount == 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .fountain_negative_feedback_discount(
          user_info_attr = "userInfo",
          save_score_to_attr = "fullrank_discount_ratio",
          discount_score = "{{fullrank_negative_feedback_discount_score}}",
          min_neg_feedback = "{{fullrank_negative_feedback_min_count}}",
          time_limit_second = "{{fullrank_negative_feedback_time_limit_sec}}"
        ) \
      .end_if_() \
      .if_("skip_fullrank_negative_feedback_discount_v2 == 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .fountain_negative_feedback_discount_v2(
          user_info_attr = "userInfoPb",
          save_score_to_attr = "fullrank_discount_ratio",
          enable_fountain_user_profile = "{{fountain_nfd_v2_enable_fountain_profile}}",
          enable_hot_user_profile = "{{fountain_nfd_v2_enable_hot_profile}}",
          enable_not_click_list = "{{fountain_nfd_v2_enable_not_click_list}}",
          enable_play_stat_list = "{{fountain_nfd_v2_enable_play_stat_list}}",
          enable_hate_list = "{{fountain_nfd_v2_enable_hate_list}}",
          discount_score = "{{fountain_nfd_v2_discount_score}}",
          neg_feedback_threshold = "{{fountain_nfd_v2_neg_feedback_threshold}}",
          period_decay_factor = "{{fountain_nfd_v2_period_decay_factor}}",
          no_click_factor = "{{fountain_nfd_v2_not_click_discount_factor}}",
          video_play_stat_factor = "{{fountain_nfd_v2_play_stat_discount_factor}}",
          hate_list_factor = "{{fountain_nfd_v2_hate_list_discount_factor}}",
          play_time_thresold_0 = "{{fountain_nfd_v2_play_time_thresold_0}}",
          play_time_thresold_1 = "{{fountain_nfd_v2_play_time_thresold_1}}",
          time_limit_second = "{{fountain_nfd_v2_time_limit_second}}",
          attr_keys = ["hetu_level_one", "hetu_level_two", "photo_dnn_cluster_id", "mmu_img_cluster_v3", "tag"],
        ) \
        .log_debug_info(
          item_attrs = [
            "fullrank_discount_ratio",
            "fullrank_ensemble_score",
          ],
          item_num_limit = 10,
          for_debug_request_only = True,
        ) \
      .end_if_() \
      .if_("enable_fountain_fullrank_score_adjust_fast == 1") \
        .enrich_attr_by_lua(
          import_common_attr = [
            "skip_fullrank_negative_feedback_discount",
            "skip_fullrank_negative_feedback_discount_v2",
            "enable_community_discount",
            "community_discount_ratio",
            "long_duration_boost",
            "long_duration_boost_min_plvtr",
            "fullrank_enable_questionnaire_boost",
            "fullrank_questionnaire_boost_ratio",
            "fullrank_questionnaire_boost_threshold",
          ],
          import_item_attr = [
            "fullrank_ensemble_score",
            "fullrank_discount_ratio",
            "explore_operation_c_review_level",
            "duration_ms",
            "fullrank_sim_plvtr",
            "questionnaire_score",
          ],
          export_item_attr = [
            "fullrank_ensemble_score_after_adjust",
          ],
          function_for_item = "fullrank_score_adjust_fast",
          lua_script_file = "fountain/full_rank/lua/fullrank_score_adjust.lua",
        ) \
      .end_() \
      .if_("enable_fountain_fullrank_heat_boost == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .enrich_attr_by_light_function( # (wangyalong03) 视频热度冷却
          import_common_attr = [
            {"name": "fountain_fullrank_heat_boost_decay_coeff", "as": "decay_coeff"},
            {"name": "fountain_fullrank_heat_boost_init_heat", "as": "init_heat"},
            {"name": "fountain_fullrank_heat_boost_min_heat", "as": "min_heat"},
          ],
          import_item_attr = [
            {"name": "upload_time", "as": "upload_time_ms"},
          ],
          export_item_attr = [
            {"name": "output_heat", "as": "fountain_fullrank_heat_boost_output_heat"},
          ],
          function_name = "LawOfCooling",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .enrich_attr_by_light_function(
          import_item_attr = [
            {"name": "fountain_fullrank_heat_boost_output_heat", "as": "boost_discount_coeff"},
            {"name": "fullrank_ensemble_score_after_adjust", "as": "score"},
          ],
          export_item_attr = [
            {"name": "score", "as": "fullrank_ensemble_score_after_adjust"},
          ],
          function_name = "BoostOrDiscountWithItemCoeff",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_fountain_fullrank_top_follow_boost == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .enrich_attr_by_light_function( # (wangyalong03) top关注作者boost
          import_common_attr = [
            {"name": "top_follow_author_list", "as": "attr_list"},
          ],
          import_item_attr = [
            {"name": "author__id", "as": "attr"},
          ],
          export_item_attr = [
            {"name": "is_in_set", "as": "is_top_follow_author"},
          ],
          function_name = "AttrIsInSet",
          class_name = "ExploreLightFunctionSetV2"
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_fullrank_top_follow_boost_weight", "as": "boost_weight"},
            {"name": "fountain_fullrank_top_follow_weaken_weight", "as": "weaken_weight"},
          ],
          import_item_attr = [
            {"name": "fullrank_ensemble_score_after_adjust", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "fullrank_ensemble_score_after_adjust"},
          ],
          function_name = "EnsembleScoreBoost",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {"is_top_follow_author": 1}
        ) \
      .end_() \
      .if_("enable_fountain_hetu_level1_discount == 1", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .fullrank_hetu_level1_discount() \
      .end_() \
      .if_("enable_fountain_ranking_personified_author_boost == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_ranking_personified_author_boost_coef", "as": "personified_author_coeff"},
            {"name": "fountain_ranking_blacklist_author_boost_coef", "as": "blacklist_author_coeff"},
            {"name": "fountain_ranking_merchant_cart_boost_coef", "as": "merchant_cart_coeff"},
            {"name": "fountain_personified_author_fans_thre_max", "as": "author_fans_thre_max"},
            {"name": "fountain_personified_author_fans_thre_min", "as": "author_fans_thre_min"},
          ],
          import_item_attr = [
            {"name": "author__fans_count", "as": "author_fans_count"},
            {"name": "eyeshot_source", "as": "eyeshot_source"},
            {"name": "merchant_photo_cart_relation", "as": "cart_relation"},
            {"name": "merchant_item_info__item_id_list", "as": "cart_itemlist"},
            {"name": "live_photo_info__is_living", "as": "is_living"},
            {"name": "fullrank_ensemble_score_after_adjust", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "fullrank_ensemble_score_after_adjust"},
          ],
          function_name = "PersonifiedAuthorBoost",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_fountain_fr_hv_pic_boost == 1") \
        .fr_high_value_pic_boost() \
      .end_() \
      .if_("enable_fountain_fr_interact_similarity_score_boost == 1", to_be_delete = "date=2024-05-29;committer=lijinyu") \
        .fr_interact_similarity_score_boost() \
      .end_() \
      .if_("enable_fountain_fr_top_sv_hetu_discount == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_fr_top_sv_hetu_discount_coeff", "as": "discount_coeff"},
            {"name": "fountain_enable_top_sv_hetu2", "as": "enable_top_sv_hetu2"},
            {"name": "fountain_top_sv_hetu_count", "as": "top_sv_hetu_count"},
            {"name": "fountain_hetu_psvtr_mix_coeff", "as": "hetu_psvtr_mix_coeff"},
            {"name": "fountain_enable_dynamic_coeff", "as": "enable_dynamic_coeff"},
            {"name": "fountain_top_sv_stat_hetu_score_lower_bound", "as": "top_sv_stat_hetu_score_lower_bound"},
            "colossus_hetu_emp_svtr_stat"
          ],
          import_item_attr = [
            {"name": "fullrank_ensemble_score_after_adjust", "as": "es_score"},
            {"name": "fullrank_sim_psvr", "as": "psvtr"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "hetu_level_one_list"},
            {"name": "hetu_tag_level_info__hetu_level_two", "as": "hetu_level_two_list"},
          ],
          export_item_attr = [
            {"name": "es_score", "as": "fullrank_ensemble_score_after_adjust"},
          ],
          function_name = "DiscountTopSvHetus",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_fountain_fr_produce_uploads_boost == 1", to_be_delete = "date=2024-05-29;committer=liuyipeng03") \
        ._fr_produce_uploads_boost() \
      .end_() \
      .if_("enable_fountain_fr_merchant_photo_boost == 1") \
        ._fr_merchant_photo_boost_by_buyer_type() \
      .end_() \
      .if_("enable_fountain_fr_living_photo_boost == 1") \
        ._fr_living_photo_boost_by_paying_type() \
      .end_() \
      .if_("enable_fountain_fr_merchant_live_boost == 1", to_be_delete = "date=2024-05-29;committer=zhanglinjiang") \
        ._fr_merchant_live_boost_by_buyer_type() \
      .end_() \
      .if_("enable_fountain_rank_refinement_boost_personified_author == 1", to_be_delete = "date=2024-05-29;committer=xubaoquan") \
        ._refinement_boost_personified_author() \
      .end_() \
      .if_("fountain_rank_enable_high_photo_count_author_adjust == 1") \
        .high_photo_count_author_adjust() \
      .end_() \
      .if_("fountain_rank_enable_high_photo_count_author_adjust_v2 == 1") \
        .high_photo_count_author_adjust_v2() \
      .end_() \
      .if_("fountain_rank_enable_collection_type_boost == 1", to_be_delete = "date=2024-05-29;committer=wangyalong03") \
        .fr_s2_collection_type_boost() \
      .end_() \
      .audit_adjust_score() \
      .if_("enable_fountain_fr_s2_pos_neg_ratio_boost == 1", to_be_delete = "date=2024-05-29;committer=gengxiao03") \
        .fr_pos_neg_ratio_boost() \
      .end_() \
      .if_("enable_fountain_fr_s2_watch_time_boost == 1") \
        .fr_watch_time_boost() \
      .end_() \
      .if_("enable_fountain_fullrank_bid_follow_boost == 1") \
        .bid_follow_boost() \
      .end_() \
      .if_("enable_fountain_fullrank_high_share_boost == 1") \
        .high_share_boost() \
      .end_() \
      .if_("enable_fountain_rank_marketing_compensation_adjust == 1") \
        .rank_marketing_compensation_adjust() \
      .end_() \
      .if_("enable_fountain_sideinfo_retargeting_score_adjust == 1") \
        .fr_sideinfo_retargeting_score_adjust() \
      .end_() \
      .if_("enable_fr_marketing_compensation_photo_personal_adjust == 1") \
        .fr_marketing_compensation_photo_personal_adjust() \
      .end_() \
      .if_("enable_fountain_rank_protogenetic_advertise_adjust == 1") \
        .rank_protogenetic_advertise_adjust() \
      .end_() \
      .if_("fountain_rank_enable_llm_negative_photo_adjust == 1") \
        .llm_negative_photo_adjust() \
      .end_() \
      .if_("fountain_rank_enable_llm_negative_photo_personal_adjust == 1") \
        .fr_llm_negative_photo_personal_adjust() \
      .end_() \
      .if_("enable_fountain_fr_user_intrest_adjust == 1") \
        .user_intrest_adjust() \
      .end_() \
      .if_("enable_fountain_fr_boost_useful_author == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_useful_author_boost_coeff", "as": "boost_discount_coeff"},
            {"name": "fountain_useful_author_boost_thres", "as": "boost_discount_thres"},
          ],
          import_item_attr = [
            {"name": "userfulness_author_score", "as": "need_item_attr"},
            {"name": "fullrank_ensemble_score_after_adjust", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "fullrank_ensemble_score_after_adjust"},
          ],
          function_name = "BoostOrDiscountWithThres",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_fountain_fr_boost_relate_intn == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fullrank_user_intn_rate", "as": "user_intn_rate"},
          ],
          import_item_attr = [
            {"name": "fountain_related_score_v2", "as": "related_thresh_score"},
            {"name": "fullrank_ensemble_score_after_adjust", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "fullrank_ensemble_score_after_adjust"},
          ],
          function_name = "BoostOrDiscountWithUserIntn",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_fountain_fr_boost_authority_author == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_fr_authority_author_boost_coeff", "as": "boost_discount_coeff"},
            {"name": "fountain_fr_authority_author_boost_thres", "as": "boost_discount_thres"},
          ],
          import_item_attr = [
            {"name": "authority_author_score", "as": "need_item_attr"},
            {"name": "fullrank_ensemble_score_after_adjust", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "fullrank_ensemble_score_after_adjust"},
          ],
          function_name = "BoostOrDiscountWithThres",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_fountain_fr_boost_expertise_author == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_fr_expertise_author_boost_coeff", "as": "boost_discount_coeff"},
            {"name": "fountain_fr_expertise_author_boost_thres", "as": "boost_discount_thres"},
          ],
          import_item_attr = [
            {"name": "expertise_author_score", "as": "need_item_attr"},
            {"name": "fullrank_ensemble_score_after_adjust", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "fullrank_ensemble_score_after_adjust"},
          ],
          function_name = "BoostOrDiscountWithThres",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_fountain_fr_boost_original_submission_author == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_fr_original_submission_author_boost_coeff", "as": "boost_discount_coeff"}
          ],
          import_item_attr = [
            {"name": "original_submission_author_tag", "as": "need_item_attr"},
            {"name": "fullrank_ensemble_score_after_adjust", "as": "ensemble_score"},
          ],
          export_item_attr = [
            {"name": "ensemble_score", "as": "fullrank_ensemble_score_after_adjust"},
          ],
          function_name = "BoostOrDiscount",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .if_("enable_fountain_fr_boost_personalization_author == 1") \
        .if_("enable_fountain_fr_personalization_author_individual_boost == 0") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "fountain_fr_personalization_author_boost_coeff", "as": "boost_discount_coeff"}
            ],
            import_item_attr = [
              {"name": "personalization_author_tag", "as": "need_item_attr"},
              {"name": "fullrank_ensemble_score_after_adjust", "as": "ensemble_score"},
            ],
            export_item_attr = [
              {"name": "ensemble_score", "as": "fullrank_ensemble_score_after_adjust"},
            ],
            function_name = "BoostOrDiscount",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .else_() \
          .split_string(
            input_common_attr = "fountain_fr_personalization_author_individual_boost_pcmef_attr_str",
            output_common_attr = "fountain_fr_personalization_author_individual_boost_pcmef_attr_list",
            delimiters = ",",
            parse_to_double = True
          ) \
          .split_string(
            input_common_attr = "fountain_fr_personalization_author_individual_boost_pwtr_attr_str",
            output_common_attr = "fountain_fr_personalization_author_individual_boost_pwtr_attr_list",
            delimiters = ",",
            parse_to_double = True
          ) \
          .split_string(
            input_common_attr = "fountain_fr_personalization_author_individual_boost_plsst_attr_str",
            output_common_attr = "fountain_fr_personalization_author_individual_boost_plsst_attr_list",
            delimiters = ",",
            parse_to_double = True
          ) \
          .split_string(
            input_common_attr = "fountain_fr_personalization_author_individual_boost_pvtr_attr_str",
            output_common_attr = "fountain_fr_personalization_author_individual_boost_pvtr_attr_list",
            delimiters = ",",
            parse_to_double = True
          ) \
          .split_string(
            input_common_attr = "fountain_fr_personalization_author_individual_boost_pswpst_attr_str",
            output_common_attr = "fountain_fr_personalization_author_individual_boost_pswpst_attr_list",
            delimiters = ",",
            parse_to_double = True
          ) \
          .split_string(
            input_common_attr = "fountain_fr_personalization_author_individual_boost_cid_valid_interest_score_attr_str",
            output_common_attr = "fountain_fr_personalization_author_individual_boost_cid_valid_interest_score_attr_list",
            delimiters = ",",
            parse_to_double = True
          ) \
          .split_string(
            input_common_attr = "fountain_fr_personalization_author_individual_boost_pltr_attr_str",
            output_common_attr = "fountain_fr_personalization_author_individual_boost_pltr_attr_list",
            delimiters = ",",
            parse_to_double = True
          ) \
          .split_string(
            input_common_attr = "fountain_fr_personalization_author_individual_boost_pcltr_attr_str",
            output_common_attr = "fountain_fr_personalization_author_individual_boost_pcltr_attr_list",
            delimiters = ",",
            parse_to_double = True
          ) \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "fountain_fr_personalization_author_boost_skip_full_and_high_users", "as": "skip_full_and_high_users"},
              {"name": "fountain_fr_personalization_author_boost_max_score", "as": "max_score"},
              {"name": "fountain_fr_personalization_author_individual_boost_pcmef_attr_list", "as": "pcmef_attr_list"},
              {"name": "fountain_fr_personalization_author_individual_boost_pwtr_attr_list", "as": "pwtr_attr_list"},
              {"name": "fountain_fr_personalization_author_individual_boost_plsst_attr_list", "as": "plsst_attr_list"},
              {"name": "fountain_fr_personalization_author_individual_boost_pvtr_attr_list", "as": "pvtr_attr_list"},
              {"name": "fountain_fr_personalization_author_individual_boost_pswpst_attr_list", "as": "pswpst_attr_list"},
              {"name": "fountain_fr_personalization_author_individual_boost_cid_valid_interest_score_attr_list", "as": "pcvis_attr_list"},
              {"name": "fountain_fr_personalization_author_individual_boost_pltr_attr_list", "as": "pltr_attr_list"},
              {"name": "fountain_fr_personalization_author_individual_boost_pcltr_attr_list", "as": "pcltr_attr_list"},
              {"name": "featureActiveDays", "as": "active_days"},
            ],
            import_item_attr = [
              {"name": "personalization_author_tag", "as": "need_item_attr"},
              {"name": "fullrank_ensemble_score_after_adjust", "as": "ensemble_score"},
              {"name": "fullrank_sim_pcmef", "as": "pcmef"},
              {"name": "fullrank_sim_pwtr", "as": "pwtr"},
              {"name": "fullrank_sim_lsst", "as": "plsst"},
              {"name": "fullrank_sim_pvtr", "as": "pvtr"},
              {"name": "fullrank_sim_swpst", "as": "pswpst"},
              {"name": "fullrank_sim_pltr", "as": "pltr"},
              {"name": "fullrank_sim_pcltr", "as": "pcltr"},
            ],
            export_item_attr = [
              {"name": "ensemble_score", "as": "fullrank_ensemble_score_after_adjust"},
            ],
            function_name = "HighQualityAuthorIndividualBoost",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .end_() \
      .end_() \
      .if_("enable_fountain_rank_consecutive_nonclick_tag_exit == 1") \
        .fr_consecutive_nonclick_tag_exit() \
      .end_() \
      .if_("enable_fountain_fr_boost_eco_living_good_author == 1") \
        .fr_eco_living_good_author_boost() \
      .end_() \
      .if_("enable_fountain_fr_group_interest_tagnex_tgi_adjust == 1") \
        .fr_interest_tagnex_tgi_adjust("group") \
      .end_() \
      .if_("enable_fountain_fr_career_interest_tagnex_tgi_adjust == 1") \
        .fr_interest_tagnex_tgi_adjust("career") \
      .end_() \
    .else_() \
      .copy_attr(
        attrs = [
          {
            "from_item": "fullrank_ensemble_score",
            "to_item": "fullrank_ensemble_score_after_adjust"
          }
        ]
      ) \
    .end_if_() \
    .sort(
      skip = "{{skip_fullrank_ensemble_score_adjust}}",
      score_from_attr = "fullrank_ensemble_score_after_adjust",
    ) \
    .copy_item_meta_info(
      save_item_seq_to_attr = "rank_final_index",
    ) \
    ._calc_result_count_to_ab_metric() \
    .if_("fountain_enable_rank_write_rank_neg_result_to_redis == 1") \
      .write_rank_neg_result_to_redis() \
    .end_() \
    .if_("fountain_enable_rank_write_rank_pos_result_to_redis == 1") \
      .write_rank_pos_result_to_redis() \
    .end_() \
    .if_("fountain_rank_enable_dedup_on_same_author == 1") \
      .deduplicate(
        on_item_attr = "author__id",
      ) \
    .end_if_() \
    .perflog_attr_value(
      check_point="rank_top100_",
      item_attrs=["duration_perf_id"],
      aggregator="count",
      range_end = 100,
    ) \
    .copy_item_meta_info(
      skip = "{{skip_fullrank_save_item_seq_neg_discount}}",
      save_item_seq_to_attr = "item_seq_neg_discount"
    ) \
    .perflog_reason_count(
      check_point = "fullrank_finish",
    ) \
    .if_("fountain_enable_cascade_distill_full_link_sample == 1") \
      .rank_full_link_sample_log() \
    .end_if_() \
    .if_("fountain_enable_cascade_distill_sample == 1") \
      .get_kconf_params(
        kconf_configs=[
        {
          "kconf_key": "reco.fountain.sampleUserFeatures",
          "value_type": "list_string",
          "default_value": [],
          "export_common_attr": "sample_user_features"
        },
        {
          "kconf_key": "reco.fountain.sampleItemFeatures",
          "value_type": "list_string",
          "default_value": [],
          "export_common_attr": "sample_item_features"
        },
        {
          "kconf_key": "reco.offline.fountainMcDistillRankParam",
          "value_type": "json",
          "json_path": "fullrank_seg1_sample_begin",
          "export_common_attr": "fountain_fullrank_seg1_sample_begin"
        },
        {
          "kconf_key": "reco.offline.fountainMcDistillRankParam",
          "value_type": "json",
          "json_path": "fullrank_seg1_sample_end",
          "export_common_attr": "fountain_fullrank_seg1_sample_end"
        },
        {
          "kconf_key": "reco.offline.fountainMcDistillRankParam",
          "value_type": "json",
          "json_path": "fullrank_seg1_sample_num",
          "export_common_attr": "fountain_fullrank_seg1_sample_num"
        },
        {
          "kconf_key": "reco.offline.fountainMcDistillRankParam",
          "value_type": "json",
          "json_path": "fullrank_seg2_sample_begin",
          "export_common_attr": "fountain_fullrank_seg2_sample_begin"
        },
        {
          "kconf_key": "reco.offline.fountainMcDistillRankParam",
          "value_type": "json",
          "json_path": "fullrank_seg2_sample_end",
          "export_common_attr": "fountain_fullrank_seg2_sample_end"
        },
        {
          "kconf_key": "reco.offline.fountainMcDistillRankParam",
          "value_type": "json",
          "json_path": "fullrank_seg2_sample_num",
          "export_common_attr": "fountain_fullrank_seg2_sample_num"
        },
        {
          "kconf_key": "reco.offline.fountainMcDistillRankParam",
          "value_type": "json",
          "json_path": "fullrank_seg3_sample_begin",
          "export_common_attr": "fountain_fullrank_seg3_sample_begin"
        },
        {
          "kconf_key": "reco.offline.fountainMcDistillRankParam",
          "value_type": "json",
          "json_path": "fullrank_seg3_sample_end",
          "export_common_attr": "fountain_fullrank_seg3_sample_end"
        },
        {
          "kconf_key": "reco.offline.fountainMcDistillRankParam",
          "value_type": "json",
          "json_path": "fullrank_seg3_sample_num",
          "export_common_attr": "fountain_fullrank_seg3_sample_num"
        },
      ]) \
      .fountain_enrich_sample_package(
        item_attrs = item_features,
        item_attrs_from_kconf = "sample_item_features",
        common_attrs = user_features,
        common_attrs_from_kconf = "sample_user_features",
        sample_config = [
          {
            "sample_begin": "fountain_fullrank_seg1_sample_begin",
            "sample_end": "fountain_fullrank_seg1_sample_end",
            "sample_num": "fountain_fullrank_seg1_sample_num",
            "label_name": "fullrank_rank_seg1",
          },
          {
            "sample_begin": "fountain_fullrank_seg2_sample_begin",
            "sample_end": "fountain_fullrank_seg2_sample_end",
            "sample_num": "fountain_fullrank_seg2_sample_num",
            "label_name": "fullrank_rank_seg2",
          },
          {
            "sample_begin": "fountain_fullrank_seg3_sample_begin",
            "sample_end": "fountain_fullrank_seg3_sample_end",
            "sample_num": "fountain_fullrank_seg3_sample_num",
            "label_name": "fullrank_rank_seg3",
          }
        ],
        load_attr = "cascadeSamplePackage",
        output_attr = "cascadeSamplePackage",
        check_point = "fullrank_positive",
        size_limit = "{{fountain_fullrank_seg1_sample_size_limit}}",
      ) \
      .log_debug_info(
        common_attrs = [
          "fountain_fullrank_seg1_sample_begin",
          "fountain_fullrank_seg1_sample_end",
          "fountain_fullrank_seg1_sample_num",
          "fountain_fullrank_seg2_sample_begin",
          "fountain_fullrank_seg2_sample_end",
          "fountain_fullrank_seg2_sample_num",
          "fountain_fullrank_seg3_sample_begin",
          "fountain_fullrank_seg3_sample_end",
          "fountain_fullrank_seg3_sample_num",
        ],
        item_num_limit = 10,
        for_debug_request_only = True,
      ) \
    .end_if_() \
    .copy_user_meta_info(
      save_flow_cpu_cost_to_attr = "full_rank_fast_cpu_cost_ts",
    ) 

    return self

  def _diversity(self):
    self \
    .if_("fountain_enable_fr_diversity == 1") \
      .diversify_by_rules(
        max_satisfied_pick="{{fountain_fr_diversity_max_satisfied_pick}}",
        rules=[
          dict(attr_name = "hetu_tag_level_info__hetu_level_one",
               enabled = "{{fountain_enable_hetu1_diversity}}",
               window_size = "{{fountain_fr_diversity_winsize_hetu1}}",
               max_num = "{{fountain_fr_diversity_max_hetu1}}",
               priority = "{{fountain_fr_diversity_priority_hetu1}}"),
          dict(attr_name = "hetu_tag_level_info__hetu_level_two",
               enabled = "{{fountain_enable_hetu2_diversity}}",
               window_size = "{{fountain_fr_diversity_winsize_hetu2}}",
               max_num = "{{fountain_fr_diversity_max_hetu2}}",
               priority = "{{fountain_fr_diversity_priority_hetu2}}"),
          dict(attr_name = "hetu_tag_level_info__hetu_level_three",
               enabled = "{{fountain_enable_hetu3_diversity}}",
               window_size = "{{fountain_fr_diversity_winsize_hetu3}}",
               max_num = "{{fountain_fr_diversity_max_hetu3}}",
               priority = "{{fountain_fr_diversity_priority_hetu3}}"),
          dict(attr_name = "hetu_tag_level_info__hetu_level_five",
               enabled = "{{fountain_enable_hetu5_diversity}}",
               window_size = "{{fountain_fr_diversity_winsize_hetu5}}",
               max_num = "{{fountain_fr_diversity_max_hetu5}}",
               priority = "{{fountain_fr_diversity_priority_hetu5}}"),
          dict(attr_name = "cluster_id_632",
               enabled = "{{fountain_enable_cid632_diversity}}",
               window_size = "{{fountain_fr_diversity_winsize_cid632}}",
               max_num = "{{fountain_fr_diversity_max_cid632}}",
               priority = "{{fountain_fr_diversity_priority_cid632}}"),
          dict(attr_name = "duration_0_7s",
               enabled = "{{fountain_fr_diversity_enable_duration_0_7s}}",
               window_size = "{{fountain_fr_diversity_winsize_duration_0_7s}}",
               max_num = "{{fountain_fr_diversity_max_duration_0_7s}}",
               priority = "{{fountain_fr_diversity_priority_duration_0_7s}}"),
          dict(attr_name = "duration_7_9s",
               enabled = "{{fountain_fr_diversity_enable_duration_7_9s}}",
               window_size = "{{fountain_fr_diversity_winsize_duration_7_9s}}",
               max_num = "{{fountain_fr_diversity_max_duration_7_9s}}",
               priority = "{{fountain_fr_diversity_priority_duration_7_9s}}"),
          dict(attr_name = "duration_9_12s",
               enabled = "{{fountain_fr_diversity_enable_duration_9_12s}}",
               window_size = "{{fountain_fr_diversity_winsize_duration_9_12s}}",
               max_num = "{{fountain_fr_diversity_max_duration_9_12s}}",
               priority = "{{fountain_fr_diversity_priority_duration_9_12s}}"),
          dict(attr_name = "duration_12_17s",
               enabled = "{{fountain_fr_diversity_enable_duration_12_17s}}",
               window_size = "{{fountain_fr_diversity_winsize_duration_12_17s}}",
               max_num = "{{fountain_fr_diversity_max_duration_12_17s}}",
               priority = "{{fountain_fr_diversity_priority_duration_12_17s}}"),
          dict(attr_name = "duration_17_20s",
               enabled = "{{fountain_fr_diversity_enable_duration_17_20s}}",
               window_size = "{{fountain_fr_diversity_winsize_duration_17_20s}}",
               max_num = "{{fountain_fr_diversity_max_duration_17_20s}}",
               priority = "{{fountain_fr_diversity_priority_duration_17_20s}}"),
          dict(attr_name = "duration_gt_120s",
               enabled = "{{fountain_fr_diversity_enable_duration_gt_120s}}",
               window_size = "{{fountain_fr_diversity_winsize_duration_gt_120s}}",
               max_num = "{{fountain_fr_diversity_max_duration_gt_120s}}",
               priority = "{{fountain_fr_diversity_priority_duration_gt_120s}}"),
        ],
      ) \
    .end_if_() \

    return self

  def rank_stage1_count_distribution(self):
    """
    精排 stage1 之后统计视频分布
    """
    self \
    .count_reco_result(
      save_count_to = "fountain_fr_stage1_hot_content_count",
      target_item = {"is_hot_content": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_fr_stage1_authority_content_count",
      target_item = {"is_authority_content": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_fr_stage1_personified_author_count",
      target_item = {"is_personified_author": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_fr_stage1_duration_0_7s_count",
      target_item = {"duration_0_7s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_fr_stage1_duration_7_9s_count",
      target_item = {"duration_7_9s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_fr_stage1_duration_9_12s_count",
      target_item = {"duration_9_12s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_fr_stage1_duration_12_17s_count",
      target_item = {"duration_12_17s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_fr_stage1_duration_17_20s_count",
      target_item = {"duration_17_20s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_fr_stage1_duration_20_58s_count",
      target_item = {"duration_20_58s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_fr_stage1_duration_58_120s_count",
      target_item = {"duration_58_120s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_fr_stage1_duration_gt_120s_count",
      target_item = {"duration_gt_120s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_fr_stage1_collection_count",
      target_item = {"is_collection": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_fr_stage1_count"
    ) \

    return self
  def rank_full_link_sample_log(self):
    self \
    .copy_item_meta_info(
      save_item_seq_to_attr = "rank_index_after_es"
    ) \
    .get_kconf_params(
      kconf_configs=[
        {
          "kconf_key": "reco.offline.fountainFulllinkDistillRankParam",
          "value_type": "json",
          "json_path": "rank_s2_full_link_distill_sample_begin",
          "export_common_attr": "fountain_rank_s2_full_link_distill_sample_begin"
        },
        {
          "kconf_key": "reco.offline.fountainFulllinkDistillRankParam",
          "value_type": "json",
          "json_path": "rank_s2_full_link_distill_sample_end",
          "export_common_attr": "fountain_rank_s2_full_link_distill_sample_end"
        },
        {
          "kconf_key": "reco.offline.fountainFulllinkDistillRankParam",
          "value_type": "json",
          "json_path": "rank_s2_full_link_distill_sample_num",
          "export_common_attr": "fountain_rank_s2_full_link_distill_sample_num"
        },
        {
          "kconf_key": "reco.offline.fountainFulllinkDistillRankParam",
          "value_type": "json",
          "json_path": "rank_s2_full_link_distill_sample_ratio",
          "export_common_attr": "fountain_rank_s2_full_link_distill_sample_ratio"
        },
      ]
    ) \
    .explore_full_link_context_sample_reco_log_enricher(
      sample_config = [
        {
          "sample_begin": "fountain_rank_s2_full_link_distill_sample_begin",
          "sample_end": "fountain_rank_s2_full_link_distill_sample_end",
          "sample_num": "fountain_rank_s2_full_link_distill_sample_num",
          "label_name": "rank_neg",
        },
      ],
      sample_ratio = "fountain_rank_s2_full_link_distill_sample_ratio",
      user_info_attr = "userInfoPb",
      load_attr = "fountain_full_link_reco_log_message",
      output_attr = "fountain_full_link_reco_log_message",
      rank_index = "rank_index_after_es",
      cascade_pctr = "cascade_pctr",
      cascade_pltr = "cascade_pltr",
      cascade_pwtr = "cascade_pwtr",
      cascade_pftr = "cascade_pftr",
      cascade_pptr = "cascade_ptr",
      cascade_pcmtr = "cascade_pcmtr",
      cascade_plvtr = "cascade_plvtr",
      cascade_pvtr = "cascade_pwatch_time",
      pctr = "fullrank_sim_pevtr",
      pltr = "fullrank_sim_pltr",
      pwtr = "fullrank_sim_pwtr",
      pftr = "fullrank_sim_pftr",
      pptr = "fullrank_sim_pptr",
      pcmtr = "fullrank_sim_pcmtr",
      plvtr = "fullrank_sim_plvtr",
      pvtr = "fullrank_sim_pvtr",
    ) \

    return self

  def _fr_produce_uploads_boost(self):
    """
    功能: 【内流-精排-生产】根据有带产能力item进行boost
    Owner: liuyipeng
    Date: 2023-11-30
    :return:
    """
    self \
      .if_("enable_fountain_rank_produce_all_switch == 1 and fountain_produce_user_type > fountain_fullrank_upload_boost_user_switch", to_be_delete = "date=2024-05-29;committer=liuyipeng03") \
        .enrich_attr_by_light_function( # enable_fountain_rank_produce_all_switch 全局反转开关
          import_common_attr = [
            {"name": "fountain_fullrank_produce_uploads_boost_coef", "as": "boost_discount_coeff"},
          ],
          import_item_attr = [
            {"name": "fullrank_ensemble_score_after_adjust", "as": "score"},
          ],
          export_item_attr = [
            {"name": "score", "as": "fullrank_ensemble_score_after_adjust"},
          ],
          function_name = "BoostOrDiscountV2",
          class_name = "ExploreLightFunctionSetV2",
          target_item = { "fountain_produce_mc_is_produce_uploads_item": 1 },
        ) \
      .end_()
    return self

  def _fr_merchant_photo_boost_by_buyer_type(self):
    """
    功能: 【内流-精排-挂车短视频】根据买家分层调整对电商视频调权，新买家降权，老买家提权，整体控电商load
    Owner: zhanglinjiang
    Date: 2023-07-07
    :return:
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "merchant_buyer_type", "as": "buyer_type"},
        {"name": "fountain_fr_merchant_photo_boost_coef", "as": "buyer_boost_coef"},
      ],
      export_common_attr = [
        {"name": "merchant_boost_coef", "as": "fr_fountain_merchant_photo_coef"}
      ],
      function_name = "MerchantCalcBoostCoef",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_fountain_merchant_photo_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "fullrank_ensemble_score_after_adjust", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "fullrank_ensemble_score_after_adjust"},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = { "is_merchant_cart": 1 },
    )

    return self

  def _fr_living_photo_boost_by_paying_type(self):
    """
    功能: 【内流-精排-直播短视频】根据用户付费分层调整对直播短视频调权
    Owner: chenliangliang03
    Date: 2024-03-19
    :return:
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "is_live_big_g_user", "as": "is_live_big_g_user"},
        {"name": "uUserKuaishouLivePayTag", "as": "user_live_paying_type"},
        {"name": "fountain_fr_living_photo_boost_coef_str", "as": "paying_user_boost_coef_str"},
        {"name": "fountain_fr_living_photo_boost_coef_big_g", "as": "boost_coef_big_g"},
      ],
      export_common_attr = [
        {"name": "living_boost_coef", "as": "fr_fountain_living_photo_coef"}
      ],
      function_name = "LivingCalcBoostCoef",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_fountain_living_photo_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "fullrank_ensemble_score_after_adjust", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "fullrank_ensemble_score_after_adjust"},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = { "is_true_living": 1 },
    )

    return self

  def _fr_merchant_live_boost_by_buyer_type(self):
    """
    功能: 【内流-精排-live头像】根据买家分层调整对电商视频调权，新买家降权，老买家提权，整体控电商load
    Owner: zhanglinjiang
    Date: 2023-07-07
    :return:
    """
    self.enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "merchant_buyer_type", "as": "buyer_type"},
        {"name": "fountain_fr_merchant_live_boost_coef", "as": "buyer_boost_coef"},
      ],
      export_common_attr = [
        {"name": "merchant_boost_coef", "as": "fr_fountain_merchant_live_coef"}
      ],
      function_name = "MerchantCalcBoostCoef",
      class_name = "ExploreLightFunctionSetV2",
    ) \
    .enrich_attr_by_light_function(
      import_common_attr = [
        {"name": "fr_fountain_merchant_live_coef", "as": "boost_discount_coeff"},
      ],
      import_item_attr = [
        {"name": "fullrank_ensemble_score_after_adjust", "as": "score"},
      ],
      export_item_attr = [
        {"name": "score", "as": "fullrank_ensemble_score_after_adjust"},
      ],
      function_name = "BoostOrDiscountV2",
      class_name = "ExploreLightFunctionSetV2",
      target_item = { "is_merchant_living": 1 },
    )

    return self

  def _calc_result_count_to_ab_metric(self):
    return self \
      .cast_attr_type(
        attr_type_cast_configs=[
          {
            "to_type": "double",
            "from_item_attr": "mc_final_rank_index",
            "to_item_attr": "mc_final_rank_index_double"
          }
        ]
      ) \
      .pack_item_attr(
        item_source = {
          "reco_results": True,
          "total_limit": 60,
        },
        mappings = [
          {
            "aggregator": "avg",
            "from_item_attr": "mc_final_rank_index_double",
            "to_common_attr": "fountain_rank_top60_mc_index_avg"
          },
        ]
      )

  def _truncate_param_adjust(self):
    self \
      .if_("fountain_enable_user_need_break_cocoon_fr_s2 == 1 and user_need_break_cocoon_flag == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_fullrank_control_hetu1_max_size", "as": "value"},
            {"name": "fountain_user_need_break_cocoon_fr_s2_control_hetu1_coef", "as": "weight"}
          ],
          export_common_attr = [
            {"name": "new_value", "as": "fountain_fullrank_control_hetu1_max_size"}
          ],
          function_name = "CalExploreIntMultiDouble",
          class_name = "ExploreLightFunctionSetV2"
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_fullrank_control_hetu2_max_size", "as": "value"},
            {"name": "fountain_user_need_break_cocoon_fr_s2_control_hetu2_coef", "as": "weight"}
          ],
          export_common_attr = [
            {"name": "new_value", "as": "fountain_fullrank_control_hetu2_max_size"}
          ],
          function_name = "CalExploreIntMultiDouble",
          class_name = "ExploreLightFunctionSetV2"
        ) \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "fountain_fullrank_control_hetu5_max_size", "as": "value"},
            {"name": "fountain_user_need_break_cocoon_fr_s2_control_hetu5_coef", "as": "weight"}
          ],
          export_common_attr = [
            {"name": "new_value", "as": "fountain_fullrank_control_hetu5_max_size"}
          ],
          function_name = "CalExploreIntMultiDouble",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_()
    return self

  def _truncate(self):
    self._truncate_param_adjust()
    return self \
      ._control_hetu_duration_count_arranger() \
      ._fountain_ensemble_pre_filter()

  def _fountain_fullrank_adjust_post_processing(self):
    return self \
      .truncate(size_limit = "{{fountain_fullrank_truncate_size_for_pre_filter}}") \
      ._fountain_ensemble_pre_filter() \
      ._diversity() \
      .truncate(size_limit = "{{fountain_rerank_diversity_output_size}}")
  
  def _fountain_ensemble_pre_filter(self):
    return self \
      .if_("fountain_enable_rerank_pre_filter == 1") \
        .fountain_ensemble_pre_filter(
          guarantee_num = "{{fountain_rerank_pre_filter_guarantee_num}}",
          keep_photo_size = "{{fountain_rerank_pre_filter_keep_photo_size}}",
          queues = fountain_rerank_pre_filter_queues,
        )\
      .end_()

  def _control_hetu_duration_count_arranger(self):
    return self \
      .explore_control_hetu_count_arranger(
        hetu_level_one_attr = "hetu_tag_level_info__hetu_level_one",
        hetu_level_two_attr = "hetu_tag_level_info__hetu_level_two",
        hetu_level_five_attr = "hetu_tag_level_info__hetu_level_five",
        duration_ms_attr = "duration_ms",
        enable_hetu_control_diversity = "{{fountain_fullrank_enable_hetu_control_diversity}}",
        enable_duration_control_diversity = "{{fountain_fullrank_enable_duration_control_diversity}}",
        keep_size = "{{fountain_rerank_limit_size_produce_v4}}",
        hetu1_max_size = "{{fountain_fullrank_control_hetu1_max_size}}",
        hetu2_max_size = "{{fountain_fullrank_control_hetu2_max_size}}",
        hetu5_max_size = "{{fountain_fullrank_control_hetu5_max_size}}",
        duration_0_7s_max_size = "{{fountain_fullrank_control_duration_0_7s_max_size}}",
        duration_7_9s_max_size = "{{fountain_fullrank_control_duration_7_9s_max_size}}",
        duration_9_12s_max_size = "{{fountain_fullrank_control_duration_9_12s_max_size}}",
        duration_12_17s_max_size = "{{fountain_fullrank_control_duration_12_17s_max_size}}",
        duration_17_20s_max_size = "{{fountain_fullrank_control_duration_17_20s_max_size}}",
        duration_300_400s_max_size = "{{fountain_fullrank_control_duration_300_400s_max_size}}",
        duration_400s_inf_max_size = "{{fountain_fullrank_control_duration_400s_inf_max_size}}",
        enable_hetu_control_diversity_none_hetu = "{{fountain_fullrank_enable_hetu_control_diversity_none_hetu}}",
        none_hetu1_max_size = "{{fountain_fullrank_control_none_hetu1_max_size}}",
        none_hetu2_max_size = "{{fountain_fullrank_control_none_hetu2_max_size}}",
        none_hetu5_max_size = "{{fountain_fullrank_control_none_hetu5_max_size}}",
      )
      
  def _rank_stage2_count_distribution(self):
    """
    精排 stage2 之后统计视频分布
    """
    self \
    .count_reco_result(
      save_count_to = "fountain_fr_stage2_hot_content_count",
      target_item = {"is_hot_content": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_fr_stage2_authority_content_count",
      target_item = {"is_authority_content": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_fr_stage2_personified_author_count",
      target_item = {"is_personified_author": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_fr_stage2_duration_0_7s_count",
      target_item = {"duration_0_7s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_fr_stage2_duration_7_9s_count",
      target_item = {"duration_7_9s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_fr_stage2_duration_9_12s_count",
      target_item = {"duration_9_12s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_fr_stage2_duration_12_17s_count",
      target_item = {"duration_12_17s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_fr_stage2_duration_17_20s_count",
      target_item = {"duration_17_20s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_fr_stage2_duration_20_58s_count",
      target_item = {"duration_20_58s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_fr_stage2_duration_58_120s_count",
      target_item = {"duration_58_120s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_fr_stage2_duration_gt_120s_count",
      target_item = {"duration_gt_120s": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_fr_stage2_collection_count",
      target_item = {"is_collection": 1}
    ) \
    .count_reco_result(
      save_count_to = "fountain_fr_stage2_count"
    )
    return self
  
  def fullrank_life_stage_cid_ipw_debias(self):
    """
    人生阶段 x cid ipw纠偏
    """
    self \
    .if_("enable_fountain_fullrank_s2_life_stage_cid_ipw_debias == 1") \
      .explore_memory_data_enrich(
        data_key = "{{fountain_fullrank_s2_life_stage_cid_ipw_map}}",
        data_type = "string_double_map",
        save_data_ptr_to_attr = "life_stage_cid_ipw_map_ptr",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "uStudentLabelV1KV",
          "uBirthLabelV1KV",
          "uMarriageLabelV1KV",
          "life_stage_cid_ipw_map_ptr",
          {"name": "fountain_fullrank_s2_life_stage_cid_ipw_redis_prefix", "as": "prefix"},
          {"name": "fountain_fullrank_s2_life_stage_cid_ipw_debias_upper_bound", "as": "upper_bound"},
          {"name": "fountain_fullrank_s2_life_stage_cid_ipw_debias_alpha", "as": "alpha"},
          {"name": "fountain_fullrank_s2_life_stage_cid_ipw_debias_beta", "as": "beta"},
        ],
        import_item_attr = [
          "cluster_id_632",
          {"name": "fullrank_sim_plvtr", "as": "plvtr"},
        ],
        export_item_attr = [
          {"name": "debias_score", "as": "fullrank_life_stage_cid_ipw_debias_plvtr"},
        ],
        function_name = "CalLifeStageCidIpwDebias",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_if_()
    return self
  
  # 双列和关注页合作涨关精排非首屏队列
  def _fountain_cal_rise_follow_boost_score_full_rank(self):
    self.calc_by_formula1(
      kconf_key = "formula.scenarioKey85.FullRankFastFountainRiseFollowBoost",
      import_item_attr = [
        "fullrank_sim_like_score",
        "fullrank_sim_follow_score",
        "fullrank_sim_plvtr",
        "fullrank_sim_click_score",
        "fullrank_sim_pwatchtime_no_bias",
        "fountain_splash_slide",
        "fullrank_sim_pvtr_multi_pwtr",
        "fullrank_sim_longview_score_no_bias_debias"
      ],
      export_formula_value = [
        "full_rank_rise_follow_boost_score"
      ],
      abtest_biz_name = "KUAISHOU_APPS"
    )
    return self

  def fullrank_age_gender_prof_cid_ipw_debias(self):
    """
    年龄 x 性别 x 职业一级 x cid ipw纠偏
    """
    self \
    .if_("enable_fountain_fullrank_s2_age_gender_prof_cid_ipw_debias == 1") \
      .explore_memory_data_enrich(
        data_key = "{{fountain_fullrank_s2_age_gender_prof_cid_ipw_map}}",
        data_type = "string_double_map",
        save_data_ptr_to_attr = "age_gender_prof_cid_ipw_map_ptr",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "basic_info_age_segment_v2",
          "basic_info_gender_v2",
          "uJobIdLv1KV",
          "age_gender_prof_cid_ipw_map_ptr",
          {"name": "fountain_fullrank_s2_age_gender_prof_cid_ipw_redis_prefix", "as": "prefix"},
          {"name": "fountain_fullrank_s2_age_gender_prof_cid_ipw_debias_upper_bound", "as": "upper_bound"},
          {"name": "fountain_fullrank_s2_age_gender_prof_cid_ipw_debias_alpha", "as": "alpha"},
          {"name": "fountain_fullrank_s2_age_gender_prof_cid_ipw_debias_beta", "as": "beta"},
        ],
        import_item_attr = [
          "cluster_id_632",
          {"name": "fullrank_sim_plvtr", "as": "plvtr"},
        ],
        export_item_attr = [
          {"name": "debias_score", "as": "fullrank_age_gender_prof_cid_ipw_debias_plvtr"},
        ],
        function_name = "CalAgeGenderProfCidIpwDebias",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_if_()
    return self
  
  def fullrank_age_gender_north_cid_ipw_debias(self):
    """
    年龄 x 性别 x 南北方 x cid ipw纠偏
    """
    self \
    .if_("enable_fountain_fullrank_s2_age_gender_north_cid_ipw_debias == 1") \
      .explore_memory_data_enrich(
        data_key = "{{fountain_fullrank_s2_age_gender_north_cid_ipw_map}}",
        data_type = "string_double_map",
        save_data_ptr_to_attr = "age_gender_north_cid_ipw_map_ptr",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "basic_info_age_segment_v2",
          "basic_info_gender_v2",
          "uIsNorthKV",
          "age_gender_north_cid_ipw_map_ptr",
          {"name": "fountain_fullrank_s2_age_gender_north_cid_ipw_redis_prefix", "as": "prefix"},
          {"name": "fountain_fullrank_s2_age_gender_north_cid_ipw_debias_upper_bound", "as": "upper_bound"},
          {"name": "fountain_fullrank_s2_age_gender_north_cid_ipw_debias_alpha", "as": "alpha"},
          {"name": "fountain_fullrank_s2_age_gender_north_cid_ipw_debias_beta", "as": "beta"},
        ],
        import_item_attr = [
          "cluster_id_632",
          {"name": "fullrank_sim_plvtr", "as": "plvtr"},
        ],
        export_item_attr = [
          {"name": "debias_score", "as": "fullrank_age_gender_north_cid_ipw_debias_plvtr"},
        ],
        function_name = "CalAgeGenderNorthCidIpwDebias",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_if_()
    return self

  def fullrank_age_gender_cid_ipw_debias(self):
    """
    年龄 x 性别 x cid ipw纠偏
    """
    self \
    .if_("enable_fountain_fullrank_s2_age_gender_cid_ipw_debias == 1") \
      .explore_memory_data_enrich(
        data_key = "{{fountain_fullrank_s2_age_gender_cid_ipw_map}}",
        data_type = "string_double_map",
        save_data_ptr_to_attr = "age_gender_cid_ipw_map_ptr",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "basic_info_age_segment_v2",
          "basic_info_gender_v2",
          "age_gender_cid_ipw_map_ptr",
          {"name": "fountain_fullrank_s2_age_gender_cid_ipw_redis_prefix", "as": "prefix"},
          {"name": "fountain_fullrank_s2_age_gender_cid_ipw_debias_upper_bound", "as": "upper_bound"},
          {"name": "fountain_fullrank_s2_age_gender_cid_ipw_debias_alpha", "as": "alpha"},
          {"name": "fountain_fullrank_s2_age_gender_cid_ipw_debias_beta", "as": "beta"},
        ],
        import_item_attr = [
          "cluster_id_632",
          {"name": "fullrank_sim_plvtr", "as": "plvtr"},
        ],
        export_item_attr = [
          {"name": "debias_score", "as": "fullrank_age_gender_cid_ipw_debias_plvtr"},
        ],
        function_name = "CalAgeGenderCidIpwDebias",
        class_name = "ExploreLightFunctionSetV2",
      ) \
    .end_if_()
    return self