from retrieval.retrieval_module import RetrievalModule

def get_features():
    features = ["selected_sim_hetu_list"]
    for prefix in ["pHetuL1ActList", "pHetuL1ActPicList"]:
        for fea_type in ["Pids", "Aids", "Ts", "Ptime", "Action"]:
            for suffix in range(4):
                features.append(prefix + fea_type + "__Top" + str(suffix+1))
    return features


class PicSsmU2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.retrieve_from_ssm_u2i()

  def retrieve_from_ssm_u2i(self):
    return self.flow \
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
        user_pic_play_aid_list_attr = "featPicRecentPlayAidList",
        user_pic_play_tag_list_attr = "featPicRecentPlayTagList",
        user_pic_play_ts_list_attr = "featPicRecentPlayTsList",
        user_pic_play_time_list_attr = "featPicRecentPlayTimeList",
        user_pic_follow_pids_attr = "uFollowListpid",
      ) \
      .explore_pic_colossus_stat(
        colossus_attr_name = "colossus_resp_v2",
        user_info_ptr_attr = "user_info_ptr",
        # output attrs
        save_pic_play_cnt = "pic_play_count",
        save_pic_play_list = "featPicColPlayPids",
        save_pic_like_list = "featPicColLkPids",
        save_pic_follow_list = "featPicColFlPids",
        save_pic_comment_list = "featPicColCmtPids",
        save_pic_comment_aid_list = "featPicColCmtAids",
        save_pic_hetu_l1_cnt = "featPicColHetuList",
        save_pic_hetu_l1_cnt_v2 = "featPicColHetuListV2",
        # input attrs
        enable_user_sim_hetu_attr = "{{explore_pic_ssm_u2i_enable_sim_hetu}}",
        pic_sim_hetu_count = "{{explore_pic_ssm_u2i_pic_sim_hetu_count}}",
        explore_with_photo_thompson_sampling = "{{explore_with_photo_thompson_sampling}}",
        explore_photo_only_key_hetu = "{{explore_photo_only_key_hetu}}",
        explore_with_pic_thompson_sampling = "{{explore_with_pic_thompson_sampling}}",
        explore_pic_only_key_hetu = "{{explore_pic_only_key_hetu}}",
      ) \
      .delegate_retrieve(
        kess_service = "{{explore_pic_ssm_u2i_retr_kess_name}}",
        timeout_ms = 70,
        shard_num = 1,
        reason = self.reason,
        request_type = "pic_ssm_u2i_retr",
        request_num = "{{explore_pic_ssm_u2i_retr_request_num}}",
        send_common_attrs=[
          "uId", "dId", "uClickPids", "uLikePids", "uFollowAids", "uFollowListpid", "uGender", "uInferGender",
          "uTrueGender", "uAgeSeg", "uProvinceId", "uCityId", "uAppList", "uPlayPics",
          {"name": "uPlayPics", "as": "featPicRecentPlayPidList"}, "featPicRecentPlayAidList", "featPicRecentPlayTagList",
          "featPicRecentPlayTsList", "featPicRecentPlayTimeList",
          "featPicColLkPids", "featPicColFlPids", "featPicColCmtPids", "featPicColCmtAids",
          "featPicColHetuList", "featPicColHetuListV2", "featPicColPlayPids", "pic_play_count",
        ] + get_features(),
      )
