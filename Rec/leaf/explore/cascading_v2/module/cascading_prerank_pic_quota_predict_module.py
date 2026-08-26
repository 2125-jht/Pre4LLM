from cascading_v2 import CommonModule

class CascadingPrerankPicQuotaPredictModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
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
        is_xhs_user_attr = "uIsXhsUser",
        user_true_new_attr = "uTrueNewUser",
        user_visit_channel_attr = "uVisitChannel",
      ) \
      .delegate_enrich(
        kess_service = "{{explore_pic_quota_model_infer_service}}",
        timeout_ms = 20,
        send_common_attrs = [
          "uId", "dId", "uClickPids", "uLikePids", "uFollowAids", "uGender", "uInferGender",
          "uTrueGender", "uAgeSeg", "uProvinceId", "uCityId", "uAppListV1", "ucat1ListV1", "uPlayPics",
          {"name": "pic_like_list",         "as": "featPicColLkPids"},
          {"name": "pic_follow_list",       "as": "featPicColFlPids"},
          {"name": "pic_comment_list",      "as": "featPicColCmtPids"},
          {"name": "pic_comment_aid_list",  "as": "featPicColCmtAids"},
          {"name": "pic_hetu_l1_cnt",       "as": "featPicColHetuList"},
          {"name": "pic_play_list",         "as": "featPicColPlayPids"},
          {"name": "featureVisitMod",         "as": "uVisitMod"},
          {"name": "location_city_level_v2",  "as": "uCityLevel"},
          {"name": "pic_stat_pic_play_cnt", "as": "colossus_pic_cnt"},
          {"name": "pic_stat_video_play_cnt", "as": "colossus_video_cnt"},
          "uStandardLikePicAllIdList","uStandardFollowPicAllIdList","uStandardLongviewPicAllIdList", 
          "uStandardCollectPicAllIdList","uIsXhsUser",
          "uTrueNewUser","uVisitChannel","short_term_pic_cnt","short_term_video_cnt",
        ],
        recv_common_attrs=[
          {"name": "pic_ratio", "as": "dynamic_pic_quota"},
        ],
      )
