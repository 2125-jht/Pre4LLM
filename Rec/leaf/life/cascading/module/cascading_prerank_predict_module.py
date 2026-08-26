from cascading import CommonModule

class CascadingPrerankPredictModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.__append_explore_prerank()
    self.__append_pic_prerank()
    self.__append_fetch_remote_comment_cnt_data()
    self.__append_fetch_pic_xtr_quantile_rank_data()
    self.__append_pic_quota()

  def __append_explore_prerank(self) -> None:
    self.flow \
      .switch_("explore_prerank_model") \
      .case_(1) \
        .explore_common_user_feature_enricher(
          user_info_attr = "user_info_ptr",
          user_uid_attr = "uId",
          user_did_attr = "dId",
          user_province_attr =  "uProvinceId",
          user_city_attr = "uCityId",
          user_gender_attr = "uGender",
          user_infer_gender_attr = "uInferGender", 
          user_ture_gender_attr = "uTrueGender",
          user_basic_gender_attr = "uBasicGender",
          user_basic_age_attr = "uBasicAge",
          user_infer_year_attr = "uInferYear",
          user_true_year_attr = "uTrueYear",
          user_follow_cnt_attr = "uFollowCount",
          user_upload_cnt_attr = "uUploadCount",
          user_visit_mod_attr = "uVisitMod",
          user_click_pids_attr = "uClickPids",
          user_like_pids_attr = "uLikePids",
          user_follow_pids_attr = "uFollowAids"
        ) \
        .delegate_enrich(
          kess_service = "{{cascade_prerank_v2_tower_service}}",
          recv_item_attrs = [
            { "name": "ctr", "as": "cascade_prerank_pctr" },
            { "name": "ltr", "as": "cascade_prerank_pltr" }
          ],
          timeout_ms = 100,
          send_common_attrs = [
            "uId","dId","uProvinceId","uCityId","uGender","uInferGender","uTrueGender","uBasicGender","uBasicAge",
            "uInferYear", "uTrueYear", "uFollowCount", "uUploadCount", "uVisitMod", "uClickPids", "uLikePids", "uFollowAids" 
          ],
          request_type = "predict_by_la_perank"
        ) \
      .case_(2) \
        .explore_common_user_feature_enricher(
          user_info_attr = "user_info_ptr",
          user_uid_attr = "uId",
          user_did_attr = "dId",
          user_gender_attr = "uGender",
          user_infer_year_attr = "uInferYear",
          user_true_year_attr = "uTrueYear",
          user_basic_age_attr = "uBasicAge",
          user_follow_cnt_attr = "uFollowCount",
          user_upload_cnt_attr = "uUploadCount",
          user_visit_mod_attr = "uVisitMod",
          user_click_pids_attr = "uClickPids",
          user_like_pids_attr = "uLikePids",
          user_follow_pids_attr = "uFollowAids"
        ) \
        .delegate_enrich(
          kess_service = "{{cascade_prerank_tower_service_v1}}",
          recv_item_attrs = [
            {"name": "distill_fr", "as": "cascade_prerank_pctr"},
            {"name": "distill_rerank", "as": "cascade_prerank_pltr"},
            {"name": "distill_show", "as": "cascade_prerank_prstr"}
          ],
          timeout_ms = 100,
          send_common_attrs = [
            "uId", "dId", "uGender", "uInferYear", "uTrueYear", "uBasicAge", "uFollowCount", "uUploadCount",
            "uVisitMod", "uClickPids", "uLikePids", "uFollowAids"
          ],
          request_type = "predict_by_mc_prerank_v0"
        ) \
      .default_() \
        .explore_prerank_trim_userinfo(
          user_info_ptr_attr = "user_info_ptr",
          output_user_info_attr = "prerank_trim_user_info",
        ) \
        .get_item_attr_by_predict_fetcher_v2(
          kess_service = "{{cascade_prerank_tower_service}}",
          service_group = "PRODUCTION",
          timeout_ms = 100,
          user_info_attr = "prerank_trim_user_info",
          try_parse_user_info = False,
          output_prefix = "cascade_prerank_",
          tower_request_type = "{{cascade_prerank_tower_request_type}}",
          pxtr = ["pctr", "pltr"],
        ) \
      .end_() \
      .if_("explore_prerank_use_mc_tower == 1") \
        .delegate_enrich(
          kess_service = "{{mc_new_arch_tower_service}}",
          timeout_ms = 100,
          request_type = "{{mc_new_arch_tower_request_type}}",
          send_common_attrs = [
            { "name": "userInfo", "as": "user_info_str" },
          ],
          recv_item_attrs = [
            { "name": "ctr", "as": "prerank_mc_pctr" },
            { "name": "ltr", "as": "prerank_mc_pltr" },
            { "name": "wtr", "as": "prerank_mc_pwtr" },
            { "name": "ftr", "as": "prerank_mc_pftr" },
            { "name": "lvr", "as": "prerank_mc_plvtr" },
            { "name": "lvtr2", "as": "prerank_mc_plvtr2" },
            { "name": "svr", "as": "prerank_mc_psvtr" },
            { "name": "ptr", "as": "prerank_mc_ptr" },
            { "name": "vtr", "as": "prerank_mc_pwatch_time" },
            { "name": "eps", "as": "prerank_mc_pepstr" },
            # { "name": "ces", "as": "prerank_mc_pcestr" },
            { "name": "cmtr", "as": "prerank_mc_pcmtr" },
            { "name": "live", "as": "prerank_mc_plivingtr" },
            { "name": "cltr", "as": "prerank_mc_pcltr" },
            # { "name": "down", "as": "prerank_mc_pdtr" },
            { "name": "htr", "as": "prerank_mc_phtr"},
            { "name": "eftr", "as": "prerank_mc_peftr"},
            { "name": "efctr", "as": "prerank_mc_pefctr"},
            { "name": "cptr", "as": "prerank_mc_pcptr"},
            { "name": "wtd", "as": "prerank_mc_pwtd"},
            # picture
            # { "name": "pic_wtdPlaytime", "as": "prerank_mc_pic_wtd"},
            # { "name": "pic_lvtr", "as": "prerank_mc_pic_lvtr"},
            # { "name": "pic_cpr", "as": "prerank_mc_pic_cpr"},
          ],
          for_predict = True,
          use_packed_item_attr = True,
          infer_output_type = 2,
        ) \
      .end_()


  def __append_pic_prerank(self) -> None:
    self.flow \
      .if_("explore_pic_prerank_tower_model_infer_v2_skip == 0") \
      .explore_common_user_feature_enricher(
        user_info_attr = "user_info_ptr",
        user_uid_attr = "uId",
        user_did_attr = "dId",
        user_pic_click_pids_attr = "uClickPids",
        user_pic_like_pids_attr = "uLikePids",
        user_pic_follow_pids_attr = "uFollowListpid",
        user_gender_attr = "uGender",
        user_ture_gender_attr = "uTrueGender",
        user_infer_gender_attr = "uInferGender",
        user_basic_age_attr = "uAgeSeg",
        user_ori_province_attr = "uProvinceId",
        user_ori_city_attr = "uCityId",
        user_app_norm_name_attr = "uAppListV1",
        user_app_cate1_orig_attr = "ucat1ListV1",
        user_pic_play_list_attr = "uPlayPics",
      ) \
      .delegate_enrich(
        target_item={"is_picture": 1},  # 只预估图片
        kess_service="{{explore_pic_tower_model_infer_service}}",
        timeout_ms=100,
        send_common_attrs=[
          "uId", "dId", "uClickPids", "uLikePids", "uFollowListpid", "uGender", "uInferGender",
          "uTrueGender", "uAgeSeg", "uProvinceId", "uCityId", "uAppListV1", "ucat1ListV1", "uPlayPics",
          {"name": "pic_like_list",         "as": "featPicColLkPids"},
          {"name": "pic_follow_list",       "as": "featPicColFlPids"},
          {"name": "pic_comment_list",      "as": "featPicColCmtPids"},
          {"name": "pic_comment_aid_list",  "as": "featPicColCmtAids"},
          {"name": "pic_hetu_l1_cnt",       "as": "featPicColHetuList"},
          {"name": "pic_play_list",         "as": "featPicColPlayPids"},
        ],
        send_item_attrs=[
          "photo_id"
        ],
        recv_item_attrs=[
          {"name": "hot_click",       "as": "pic_hot_click"},
          {"name": "hot_long_view",   "as": "pic_hot_long_view"},
          {"name": "hot_finish_view", "as": "pic_hot_finish_view"},
          {"name": "hot_pos_wtd",     "as": "pic_hot_pos_wtd"},
          {"name": "hot_action",      "as": "pic_hot_action"},
          {"name": "hot_collect",     "as": "pic_hot_collect"},
          {"name": "hot_long_photo_scroll", "as": "pic_hot_scroll"},
        ],
        request_type="explorePicTowerInfer",
        for_predict=True,
        partition_size="{{pic_tower_model_partition_size}}"
      ) \
      .end_() \
      .if_("explore_pic_prerank_pic_longterm_infer__enable == 1") \
        .delegate_enrich(
          target_item={"is_picture": 1},
          kess_service="{{explore_pic_prerank_pic_longterm_infer__service}}",
          timeout_ms=100,
          send_common_attrs=[
            {"name": "userInfo", "as": "user_info_str"},
          ],
          send_item_attrs=[
            "photo_id"
          ],
          recv_item_attrs=[
            {"name": "longterm_click", "as": "pic_longterm_click"},
            {"name": "longterm_collect", "as": "pic_longterm_collect"},
            {"name": "longterm_revisit", "as": "pic_longterm_revisit"},
          ],
          request_type="explorePicTowerInfer",
          for_predict=True,
          partition_size="{{pic_tower_model_partition_size}}"
        ) \
      .end_()

  def __append_pic_quota(self) -> None:
    self.flow \
      .if_("explore_pic_quota_model_infer_skip == 0") \
      .explore_common_user_feature_enricher(
        user_info_attr = "user_info_ptr",
        user_uid_attr = "uId",
        user_did_attr = "dId",
        user_click_pids_attr = "uClickPids",
        user_like_pids_attr = "uLikePids",
        user_follow_pids_attr = "uFollowAids",
        user_gender_attr = "uGender",
        user_ture_gender_attr = "uTrueGender",
        user_infer_gender_attr = "uInferGender",
        user_basic_age_attr = "uAgeSeg",
        user_ori_province_attr = "uProvinceId",
        user_ori_city_attr = "uCityId",
        user_app_norm_name_attr = "uAppListV1",
        user_app_cate1_orig_attr = "ucat1ListV1",
        user_pic_play_list_attr = "uPlayPics",
      ) \
      .delegate_enrich(
        kess_service="{{explore_pic_quota_model_infer_service}}",
        timeout_ms=100,
        send_common_attrs=[
          "uId", "dId", "uClickPids", "uLikePids", "uFollowAids", "uGender", "uInferGender",
          "uTrueGender", "uAgeSeg", "uProvinceId", "uCityId", "uAppListV1", "ucat1ListV1", "uPlayPics",
          {"name": "pic_like_list",         "as": "featPicColLkPids"},
          {"name": "pic_follow_list",       "as": "featPicColFlPids"},
          {"name": "pic_comment_list",      "as": "featPicColCmtPids"},
          {"name": "pic_comment_aid_list",  "as": "featPicColCmtAids"},
          {"name": "pic_hetu_l1_cnt",       "as": "featPicColHetuList"},
          {"name": "pic_play_list",         "as": "featPicColPlayPids"},
        ],
        recv_common_attrs=[
          {"name": "pic_ratio", "as": "dynamic_pic_quota"},
        ],
        request_type="explorePicTowerInfer",
        for_predict=True,
      ) \
      .end_() 

  def __append_fetch_remote_comment_cnt_data(self) -> None:
    self.flow \
      .if_("remote_comment_cnt_data__enable == 1") \
        .get_remote_embedding(
          kess_service = "{{remote_comment_cnt_data__kess_name}}",
          shard_num = 2,
          id_converter=dict(type_name="mioEmbeddingIdConverter"),
          query_source_type = "item_id",
          client_side_shard = True,
          save_to_common_attr=False,
          slot = 50,
          output_attr_name = "god_comment_cnt",
          timeout_ms = 20,
          is_raw_data=True,
          raw_data_type='uint64',
          is_raw_data_list=False,
        ) \
        .get_remote_embedding(
          kess_service = "{{remote_comment_cnt_data__kess_name}}",
          shard_num = 2,
          id_converter=dict(type_name="mioEmbeddingIdConverter"),
          query_source_type = "item_id",
          client_side_shard = True,
          save_to_common_attr=False,
          slot = 51,
          output_attr_name = "hot_comment_cnt",
          timeout_ms = 20,
          is_raw_data=True,
          raw_data_type='uint64',
          is_raw_data_list=False,
        ) \
      .end_()

  def __append_fetch_pic_xtr_quantile_rank_data(self) -> None:
    self.flow \
      .if_("pic_xtr_quantile_rank__rawdata__enable == 1") \
        .get_remote_embedding_lite_v2(
          kess_service = "{{pic_xtr_quantile_rank__rawdata__kess_name}}",
          shard_num = 2,
          id_converter=dict(type_name="mioEmbeddingIdConverter"),
          query_source_type = "item_id",
          input_attr_name = "photo_id",
          client_side_shard = True,
          slot = 111,
          output_attr_name = "explore_rank_xtr_list",
          timeout_ms = 20,
          is_raw_data=True,
          raw_data_type='uint64',
          size = 10,
          target_item = {"is_picture": 1},
        ) \
      .end_()



  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = ["remote_comment_cnt_data__enable"],
        item_attrs = ["god_comment_cnt", "hot_comment_cnt"],
        for_debug_request_only = True,
      )
