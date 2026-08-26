from retrieval.retrieval_module import RetrievalModule


class ExploreStRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("use_cached_user_emb == 1") \
        .get_remote_embedding_lite(
          kess_service = "{{embedding_service_name}}",
          timeout_ms = 10,
          query_source_type = "user_id",
          output_attr_name = "user_embedding_concat",
          id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
          slot = 999,
          size = self.emb_size,
        ) \
      .end_() \
      .if_("user_embedding_concat == nil or #user_embedding_concat <= 0")

    for _attr_name in self.pack_common_attr_from_list:
      self.flow.pack_common_attr(
        input_common_attrs = [_attr_name],
        output_common_attr = _attr_name + "_packed",
        limit_num = "{{colossus_hisotry_num}}",
      )
  
    self.flow \
        .if_("enable_extract_common_user_feature == 1") \
          .explore_common_user_feature_enricher(
            user_info_attr = "user_info_ptr",
            user_uid_attr = "uId",
            user_did_attr = "dId",
            user_province_attr =  "uProvinceId",
            user_city_attr = "uCityId",
            user_click_pids_attr = "uClickPids",
            user_like_pids_attr = "uLikePids",
            user_follow_pids_attr = "uFollowAids",
            user_gender_attr = "uGender",
            user_infer_gender_attr = "uInferGender",
            user_ture_gender_attr = "uTrueGender",
            user_basic_age_attr = "uAgeSeg",
            user_app_package_attr = "uAppList",
            user_pic_play_list_attr = "uPlayPics",  
            user_pic_follow_pids_attr = "uFollowListpid",
            user_pic_play_aid_list_attr = "featPicRecentPlayAidList",
            user_pic_play_tag_list_attr = "featPicRecentPlayTagList",
            user_pic_play_ts_list_attr = "featPicRecentPlayTsList",
            user_pic_play_time_list_attr = "featPicRecentPlayTimeList",
            user_video_play_list_attr = "featRecentPlayPidList",
            user_video_play_aid_list_attr = "featRecentPlayAidList",
            user_video_play_tag_list_attr = "featRecentPlayTagList",
            user_video_play_ts_list_attr = "featRecentPlayTsList",
            user_video_play_time_list_attr = "featRecentPlayTimeList",
          ) \
        .end_() \
        .if_("enable_infer_user_emb == 1") \
          .delegate_enrich(
            kess_service = "{{predict_service_name}}",
            recv_common_attrs = [
              {"name": "user_top_layer", "as": "user_embedding_concat"},
            ],
            timeout_ms = 40,
            send_common_attrs = [
              {"name": "_USER_ID_",   "as": "uId"},
              {"name": "_DEVICE_ID_", "as": "dId"},
              "uProvinceId",
              "uCityId",
              "uClickPids",
              "uLikePids",
              "uFollowAids",
              "uGender",
              "uInferGender",
              "uTrueGender",
              "uAgeSeg",
              "uAppList",
              "uPlayPics",
              "uPlayPics",
              "uFollowListpid",
              "featPicRecentPlayAidList",
              "featPicRecentPlayTagList",
              "featPicRecentPlayTsList",
              "featPicRecentPlayTimeList",
              "featRecentPlayPidList",
              "featRecentPlayAidList",
              "featRecentPlayTagList",
              "featRecentPlayTsList",
              "featRecentPlayTimeList",
            ] + self.send_common_attrs,
            request_type = "default"
          ) \
        .end_() \
      .end_() \
      .if_(f"user_embedding_concat == nil or #user_embedding_concat ~= {self.emb_size}") \
        .return_() \
      .end_() \
      .gen_common_attr_by_lua(
        attr_map = {
          "uid_concat": "{_USER_ID_, _USER_ID_+1, _USER_ID_+2 ,_USER_ID_+3}",
        }
      ) \
      .retrieve_by_ann_embedding(
        reason = self.reason,
        kess_service = "{{ann_service_name}}",
        space = "ip",
        timeout_ms = 50,
        items_from_attr=["uid_concat"],
        embeddings_from_attr=["user_embedding_concat"],
        bound_type={
          "top_k": "{{retrieve_num}}"
        },
        algo_type={
          "scann": {},
        },
        src_bucket = "photo_src",
        dest_bucket = "{{ann_dest_bucket}}",
        save_distance_to_attr = "ann_dist_list",
      ) \
      .if_("ann_dist_threshold ~= nil and ann_dist_threshold > 0.0") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "ann_dist_threshold",
          ],
          import_item_attr = [
            "ann_dist_list",
          ],
          export_item_attr = [
            "ann_dist",
          ],
          function_name = "AnnCalThresholdValueForDistList",
          class_name = "ExploreLightFunctionSetV2",
          target_reason = self.reason,
        ) \
        .filter_by_attr(
          attr_name = "ann_dist",
          remove_if = "<",
          compare_to = "{{ann_dist_threshold}}"
        ) \
      .end_()

  @property
  def send_common_attrs(self) -> list:
    assert "send_common_attrs" in self.config
    return self.config.get("send_common_attrs")

  @property
  def pack_common_attr_from_list(self) -> list:
    assert "pack_common_attr_from_list" in self.config
    return self.config.get("pack_common_attr_from_list")

  @property
  def emb_size(self) -> int:
    assert "emb_size" in self.config
    return self.config.get("emb_size")