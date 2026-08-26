from retrieval.retrieval_module import RetrievalModule

class SimRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("#(source_hetu_level_one or {}) >= 1 and (source_hetu_level_one or {0})[1] > 0") \
        .explore_common_user_feature_enricher(
          user_info_attr = "userInfoPb",
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
          user_video_play_list_attr = "featRecentPlayPidList",
          user_video_play_aid_list_attr = "featRecentPlayAidList",
          user_video_play_tag_list_attr = "featRecentPlayTagList",
          user_video_play_ts_list_attr = "featRecentPlayTsList",
          user_video_play_time_list_attr = "featRecentPlayTimeList",
        ) \
        .gen_common_attr_by_lua(
          attr_map={
            "source_hetu_level_one_tag": "(source_hetu_level_one or {0})[1]",
            "selected_photo_sim_hetu_list": "{(source_hetu_level_one or {0})[1]}",
        }) \
        .explore_pic_colossus_stat(
          colossus_attr_name = "colossus_resp_v2",
          user_info_ptr_attr = "userInfoPb",
          # output attrs
          save_photo_hetu_l1_cnt = "photo_hetu_l1_cnt",
          save_photo_like_list = "photo_like_list",
          save_photo_follow_list = "photo_follow_list",
          save_photo_comment_list = "photo_comment_list",
          # sim retr input attrs
          enable_fountain_splash_sim_hetu = True,
          fountain_splash_sim_hetu1 = "{{source_hetu_level_one_tag}}",
        ) \
        .delegate_retrieve(
          kess_service = "{{fountain_splash_sim_u2i_retr_kess_name}}",
          timeout_ms = 100,
          shard_num = 1,
          reason = self.reason,
          request_type = "splash",
          request_num = "{{fountain_splash_sim_u2i_retr_total_num}}",
          send_common_attrs=[
            "uId", "dId", "uClickPids", "uLikePids", "uFollowAids", "uFollowListpid", "uGender", "uInferGender",
            "uTrueGender", "uAgeSeg", "uProvinceId", "uCityId", "uAppList", "uPlayPics",
            "featRecentPlayPidList",
            "featRecentPlayAidList", "featRecentPlayTagList",
            "featRecentPlayTsList", "featRecentPlayTimeList",
            {"name": "photo_hetu_l1_cnt", "as": "featPicColHetuListV2"},
            {"name": "photo_like_list", "as": "featPicColLkPids"},
            {"name": "photo_follow_list", "as": "featPicColFlPids"},
            {"name": "photo_comment_list", "as": "featPicColCmtPids"},
            "selected_photo_sim_hetu_list",
            {"name": "pHetuL1ActListPids__photo_Top1_splash", "as": "pHetuL1ActListPids__photo_Top1"},
            {"name": "pHetuL1ActListAids__photo_Top1_splash", "as": "pHetuL1ActListAids__photo_Top1"},
            {"name": "pHetuL1ActListTs__photo_Top1_splash", "as": "pHetuL1ActListTs__photo_Top1"},
            {"name": "pHetuL1ActListPtime__photo_Top1_splash", "as": "pHetuL1ActListPtime__photo_Top1"},
            {"name": "pHetuL1ActListAction__photo_Top1_splash", "as": "pHetuL1ActListAction__photo_Top1"},
            # source photo attr
            {"name": "featureSourcePId",        "as": "uSourcePId"},
            {"name": "SourcePhotoAuthorId",     "as": "uSourcePAId"},
            {"name": "source_hetu_level_one",   "as": "uSourcePHetuL1"},
            {"name": "source_hetu_level_two",   "as": "uSourcePHetuL2"},
            {"name": "source_hetu_level_three", "as": "uSourcePHetuL3"},
            {"name": "source_hetu_face_ids",    "as": "uSourcePHetuIP"},
            {"name": "source_hetu_level_five",  "as": "uSourcePHetuL5"},
          ],
        ) \
      .end_()
