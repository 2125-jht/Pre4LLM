from cascading import CommonModule
from cascading.cascade_util import hot_fc_features, hot_sim_fc_features, hot_combo_sim3_features

class CascadingPredictModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .count_reco_result(
        save_count_to = "explore_reco_leaf_cascade_model_pic_input_result_count",
        target_item = {"is_picture": 1},
      ) \
      .count_reco_result(
        save_count_to = "explore_reco_leaf_cascade_model_input_result_count",
      ) \
      .delegate_enrich(
        skip = "{{enable_cascade_downgrade}}",
        kess_service = "{{mc_new_arch_tower_service}}",
        timeout_ms = 100,
        request_type = "{{mc_new_arch_tower_request_type}}",
        send_common_attrs = [
          { "name": "userInfo", "as": "user_info_str" },
        ],
        recv_item_attrs = [
          { "name": "ctr", "as": "cascade_pctr" },
          { "name": "ltr", "as": "cascade_pltr" },
          { "name": "wtr", "as": "cascade_pwtr" },
          { "name": "ftr", "as": "cascade_pftr" },
          { "name": "lvr", "as": "cascade_plvtr" },
          { "name": "lvtr2", "as": "cascade_plvtr2" },
          { "name": "svr", "as": "cascade_psvtr" },
          { "name": "ptr", "as": "cascade_ptr" },
          { "name": "vtr", "as": "cascade_pwatch_time" },
          { "name": "eps", "as": "cascade_pepstr" },
          { "name": "ces", "as": "cascade_pcestr" },
          { "name": "cmtr", "as": "cascade_pcmtr" },
          { "name": "live", "as": "cascade_plivingtr" },
          { "name": "cltr", "as": "cascade_pcltr" },
          { "name": "down", "as": "cascade_pdtr" },
          { "name": "htr", "as": "cascade_phtr"},
          { "name": "eftr", "as": "cascade_peftr"},
          { "name": "efctr", "as": "cascade_pefctr"},
          { "name": "cptr", "as": "cascade_pcptr"},
          { "name": "wtd", "as": "cascade_pwtd"},
          # picture
          { "name": "pic_wtdPlaytime", "as": "cascade_pic_wtd"},
          { "name": "pic_lvtr", "as": "cascade_pic_lvtr"},
          { "name": "pic_cpr", "as": "cascade_pic_cpr"},
        ],
        for_predict = True,
        use_packed_item_attr = True,
        infer_output_type = 2,
      ) \
      .explore_life_embedding_candidates_attr_enricher(
        trans_type = "embedding_candidates",
        user_info_ptr_attr = "user_info_ptr",
        export_common_attr = "embedding_source_pids",
        enable_use_explore = "{{xlife_embedding_candidates_enable_use_explore}}",
        check_point = "cascade"
      ) \
      .get_remote_embedding_lite(
        skip = "{{return enable_fetch_mc_embedding_score == 0}}",
        kess_service = "{{mc_embedding_service_name}}",
        shard_num = 4,
        id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
        input_attr_name = "embedding_source_pids",
        output_attr_name = "mmu_embeddings",
        query_source_type = "common_attr",
        size = 64,
        client_side_shard = True
      )

    self.__append_fc_predict()
    self. __append_pic_predict()
    self. __append_life_predict()

    self.flow \
      .copy_item_meta_info(
        save_item_key_to_attr = "item_key",
      ) \
      .pack_item_attr(  # 保存进入粗排的结果集，单独发送样本流
        item_source = {
          "reco_results": True
        },
        mappings = [{
          "aggregator": "concat",
          "from_item_attr": "item_key",
          "to_common_attr": "cascade_input_item_key_list"
        }],
      )

  def __append_fc_predict(self) -> None:
    self.flow \
      .if_("enable_mc_fc_predict == 1") \
        .if_("enable_hot_fc_extract_photo_signs == 1") \
          .enrich_attr_by_light_function(
            import_item_attr = [
              "photo_id", "author__id", "tag", "duration_ms", "upload_time"
            ],
            export_item_attr = [{"name":"context_slots", "as":"hot_fc_car_slots"},
                                {"name":"context_signs", "as":"hot_fc_car_signs"}],
            function_name = "GenCARSigns",
            class_name = "ExploreLightFunctionSetV2",
          )\
        .end_if_()\
        .if_("enable_hot_fc_exp == 1") \
          .extract_with_ks_sign_feature( #修改为car模型配置
            extractor_kconf_path = "reco.hot.fountainLeafMcFeature",
            caller_model = "{{mc_fc_predict_service}}",
            user_info_attr = "user_info_ptr",
            common_slots_output = "fc_sign_common_slots",
            common_parameters_output = "fc_sign_common_parameters",
          ) \
          .delegate_enrich(
            kess_service = "{{mc_fc_predict_service}}",
            request_type = "{{mc_fc_request_type}}",
            timeout_ms = 100,
            send_common_attrs = [
              { "name": "fc_sign_common_slots", "as": "user_feature_slots" },
              { "name": "fc_sign_common_parameters", "as": "user_feature_signs" },
            ],
            send_item_attrs = ["hot_fc_car_slots", "hot_fc_car_signs"], 
            recv_item_attrs = [
              {'name':'fc_pctr_value', 'as': 'cascade_fc_pctr'},
              {'name':'fc_plvr_value', 'as': 'cascade_fc_plvtr'},
              {'name':'fc_psvr_value', 'as': 'cascade_fc_psvtr'},
              {'name':'fc_pvtr_value', 'as': 'cascade_fc_pvtr'},
            ],
            use_packed_item_attr = True,
          ) \
        .else_() \
          .extract_with_ks_sign_feature(
            feature_list = hot_sim_fc_features,
            user_info_attr = "user_info_ptr",
            common_slots_output = "fc_sign_common_slots",
            common_parameters_output = "fc_sign_common_parameters",
          ) \
          .delegate_enrich(
            kess_service = "{{mc_fc_predict_service}}",
            request_type = "{{mc_fc_request_type}}",
            timeout_ms = 100,
            send_common_attrs = [
              { "name": "fc_sign_common_slots", "as": "common_slots" },
              { "name": "fc_sign_common_parameters", "as": "common_parameters" },
            ],
            recv_common_attrs = [
              "mc_pxtr_label",
              "mc_pxtr_value",
            ],
          ) \
        .end_() \
      .end_()

  def __append_pic_predict(self) -> None:
    self.flow \
      .if_("skip_cascade_enrich_pic_play_cnt == 0 or explore_pic_tower_model_infer_v3_skip == 0") \
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
            user_pic_play_count = "user_pic_play_count",
            user_photo_play_count = "user_photo_play_count",
            user_pic_eff_play_count = "user_pic_eff_play_count",
            user_photo_eff_play_count = "user_photo_eff_play_count",
            effect_playtime_thresh_s = "user_effect_playtime_thresh_s",
        ) \
      .end_() \
      .if_("explore_pic_tower_model_infer_v3_skip == 0") \
        .delegate_enrich(
          target_item={"is_picture": 1},
          kess_service="{{explore_pic_tower_model_infer_service}}",
          timeout_ms=100,
          send_common_attrs=[
            "uId", "dId", "uClickPids", "uLikePids", "uFollowListpid", "uGender", "uInferGender",
            "uTrueGender", "uAgeSeg", "uProvinceId", "uCityId", "uAppListV1", "ucat1ListV1", "uPlayPics",
            {"name": "pic_like_list", "as": "featPicColLkPids"},
            {"name": "pic_follow_list", "as": "featPicColFlPids"},
            {"name": "pic_comment_list", "as": "featPicColCmtPids"},
            {"name": "pic_comment_aid_list", "as": "featPicColCmtAids"},
            {"name": "pic_hetu_l1_cnt", "as": "featPicColHetuList"},
            {"name": "pic_play_list", "as": "featPicColPlayPids"},
          ],
          recv_item_attrs=[
            {"name": "hot_click", "as": "pic_hot_click"},
            {"name": "hot_long_view", "as": "pic_hot_long_view"},
            {"name": "hot_finish_view", "as": "pic_hot_finish_view"},
            {"name": "hot_pos_wtd", "as": "pic_hot_pos_wtd"},
            {"name": "hot_action", "as": "pic_hot_action"},
            {"name": "hot_collect", "as": "pic_hot_collect"},
            {"name": "hot_enter_comment", "as": "pic_hot_enter_comment"},
            {"name": "hot_comment_effctive_stop", "as": "pic_hot_comment_effctive_stop"},
            {"name": "hot_long_photo_scroll", "as": "pic_hot_scroll"},
          ],
          request_type="explorePicTowerInfer",
          for_predict=True,
          partition_size="{{pic_tower_model_partition_size}}"
        ) \
      .end_()


  def explore_life_mc_user_feture(self):
    features = [ "uId", "dId", "uClickPids", "uLikePids", "uFollowListpid", "uFollowAids", "uGender", "uInferGender",
      "uTrueGender", "uAgeSeg", "uProvinceId", "uCityId", "uAppList", "ucat1List", "uPlayPics", "uCityLevel", "uRiskLevel",
      "uFollowCount", "uFansCount", "uUploadCount", "uTrueNewUser", "uLogin", "uVisitMod", "uNetwork", "cHourOfDay", "cDayOfWeek"]
    for key in ["uHotShow", "uHotClick", "uHotLike", "uHotFollow", "uHotHate"]:
      for suffix in ["5m", "1d", "1h", "100n", "1000n"]:
        features.append(key + suffix)
    return features
  
  def __append_life_predict(self) -> None:
    self.flow \
      .if_("enable_explore_life_prerank_tower_model_infer == 1") \
        .copy_attr(
          attrs = [{
            "from_item": "prerank_life_ctr",
            "to_item": "cascase_life_ctr",
          }],
        ) \
      .else_() \
        .if_("enable_explore_life_mc_tower_model_infer == 1") \
          .explore_common_user_feature_enricher(
            user_info_attr = "user_info_ptr",
            user_uid_attr = "uId",
            user_did_attr = "dId",
            user_click_pids_attr = "uClickPids",
            user_like_pids_attr = "uLikePids",
            global_follow_pids_attr = "uFollowListpid",
            user_follow_pids_attr = "uFollowAids",
            user_gender_attr = "uGender",
            user_ture_gender_attr = "uTrueGender",
            user_infer_gender_attr = "uInferGender",
            user_basic_age_attr = "uAgeSeg",
            user_ori_province_attr = "uProvinceId",
            user_ori_city_attr = "uCityId",
            user_app_norm_name_attr = "uAppList",
            user_app_cate1_orig_attr = "ucat1List",
            user_pic_play_list_attr = "uPlayPics",
            user_city_level_attr = "uCityLevel",
            user_risk_level_attr = "uRiskLevel",
            user_follow_cnt_attr = "uFollowCount",
            user_fans_cnt_attr = "uFansCount",
            user_upload_cnt_attr = "uUploadCount",
            user_true_new_attr = "uTrueNewUser",
            user_login_attr = "uLogin",
            user_visit_mod_attr =  "uVisitMod",
            user_visit_net_attr = "uNetwork",
            context_hour_of_day_attr = "cHourOfDay",
            context_day_of_week_attr = "cDayOfWeek",
            user_count_action_attr = "cnt_",
          ) \
          .delegate_enrich(
            kess_service = "{{explore_life_mc_tower_model_infer_service}}",
            timeout_ms = 100,
            send_common_attrs = self.explore_life_mc_user_feture(),
            send_item_attrs = [
              "photo_id"
            ],
            recv_item_attrs = [
              {"name": "life_click", "as": "cascase_life_ctr"},
            ],
            request_type = "{{explore_life_mc_request_type}}",
            for_predict = True,
          ) \
        .end_() \
      .end_() \

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = ["user_effect_playtime_thresh_s"],
        for_debug_request_only = True,
        target_item = { "is_picture" : 1 }
      ) \
      .log_debug_info(
        common_attrs = ["embedding_source_pids"],
        for_debug_request_only = True
      )
