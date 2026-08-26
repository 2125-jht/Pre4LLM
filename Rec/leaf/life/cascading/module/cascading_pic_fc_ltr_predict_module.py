from cascading import CommonModule


class CascadingPicFcLtrPredictModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
    self.photo_features = [
      # photo & author basic features
      {"name": "author__id", "as": "aId"},
      {"name": "upload_type", "as": "pUploadType"},
      {"name": "location__province_id", "as": "pProvinceId"},
      {"name": "author__fans_count", "as": "aFansCount"},
      {"name": "photo_picture_count", "as": "pPictureCount"},
      {"name": "caption_length", "as": "pCaptionLength"},
      {"name": "location__city_id", "as": "pCityId"},
      {"name": "location__province_id", "as": "pProvinceId"},
      {"name": "duration_ms", "as": "pDurationMs"},
      {"name": "upload_type", "as": "pUploadType"},
      {"name": "author__gender", "as": "pAuthorGender"},
      {"name": "author_age_info__age_segment", "as": "pAuthorAgeSeg"},
      "is_picture",
      "upload_time",
      # mmu & hetu features
      {"name": "hetu_tag_level_info__hetu_level_one", "as": "pHetuTagLevel1"},
      {"name": "hetu_tag_level_info__hetu_level_two", "as": "pHetuTagLevel2"},
      {"name": "hetu_tag_level_info__hetu_level_three", "as": "pHetuTagLevel3"},
      {"name": "hetu_tag_level_info__hetu_level_five", "as": "pHetuTagLevel5"},
      {"name": "hetu_tag_level_info__hetu_tag", "as": "pHetuTagLevelTag"},
      {"name": "hetu_tag_level_info__hetu_face_id", "as": "pHetuTagFaceId"},
      {"name": "hetu_tag_level_info__hetu_cluster_id", "as": "pHetuClusterId"},
      {"name": "mmu_img_cluster_v3", "as": "pMmuImgClusterV3"},
      {"name": "music", "as": "pMusic"},
      # empirical xtrs
      {"name": "empirical_ctr", "as": "pEmpCtr"},
      {"name": "empirical_ltr", "as": "pEmpLtr"},
      {"name": "empirical_wtr", "as": "pEmpWtr"},
      {"name": "empirical_ftr", "as": "pEmpFtr"},
      {"name": "empirical_ptr", "as": "pEmpPtr"},
      {"name": "empirical_cmtr", "as": "pEmpCmtr"},
      {"name": "empirical_htr", "as": "pEmpHtr"},
      {"name": "empirical_watch_time", "as": "pEmpWatchTime"},
      # 计数特征
      {"name": "explore_stat__show_count", "as": "pHotShow"},
      {"name": "explore_stat__click_count", "as": "pHotClick"},
      {"name": "explore_stat__like_count", "as": "pHotLike"},
      {"name": "explore_stat__follow_count", "as": "pHotFollow"},
      {"name": "explore_stat__negative_count", "as": "pHotHate"},
      {"name": "explore_stat__report_detail__total_report_count", "as": "pHotReport"},
    ]

  def process(self) -> None:
    self.flow \
        .if_("enable_pic_cascade_fc_ltr_predict == 1") \
          .delegate_enrich(
            kess_service="{{pic_cascade_fc_ltr_service}}",
            recv_item_attrs=[
              {"name": "pic_cascade_click", "as": "pic_cascade_fc_pctr"},
              {"name": "pic_cascade_interact", "as": "pic_cascade_fc_interact_score"},
              {"name": "pic_cascade_ltr", "as": "pic_cascade_fc_ltr"},
              {"name": "pic_cascade_wtr", "as": "pic_cascade_fc_wtr"},
              {"name": "pic_cascade_cmtr", "as": "pic_cascade_fc_cmtr"},
              {"name": "pic_cascade_d2q", "as": "pic_cascade_fc_d2q"},
            ],
            timeout_ms=100,
            send_item_attrs=self.photo_features,
            send_common_attrs=[
              {"name": "userInfo", "as": "user"},
            ],
            request_type="pic_cascade_fc_predict",
            partition_size="{{pic_cascade_fc_predict_part_size}}",
            target_item={
              "is_picture": 1
            }
          ) \
        .end_() \
        .if_("enable_pic_cascade_variety == 1") \
          .get_remote_embedding_lite(
            kess_service="{{explore_mc_pic_s1_pic_variety_emb_server}}",
            shard_num=4,
            id_converter={"type_name": "kuibaEmbeddingIdConverter"},
            input_attr_name="photo_id",
            output_attr_name="pic_mmu_embedding",
            query_source_type="item_key",
            size=64,
            client_side_shard=True,
            partition_size="{{explore_mc_pic_s1_pic_variety_emb_part_size}}",
            target_item={
              "is_picture": 1,
            },
          ) \
        .end_()
