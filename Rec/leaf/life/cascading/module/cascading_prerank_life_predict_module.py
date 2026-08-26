from cascading import CommonModule
from cascading.module.cascading_features import *

class CascadingPrerankLifePredictModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def explore_life_prerank_user_feture(self):
    features = [ "uId", "dId", "uClickPids", "uLikePids", "uFollowListpid", "uFollowAids", "uGender", "uInferGender",
      "uTrueGender", "uAgeSeg", "uProvinceId", "uCityId", "uAppList", "ucat1List", "uPlayPics", "uCityLevel", "uRiskLevel",
      "uFollowCount", "uFansCount", "uUploadCount", "uTrueNewUser", "uLogin", "uVisitMod", "uNetwork", "cHourOfDay", "cDayOfWeek"]
    for key in ["uHotShow", "uHotClick", "uHotLike", "uHotFollow", "uHotHate"]:
      for suffix in ["5m", "1d", "1h", "100n", "1000n"]:
        features.append(key + suffix)
    return features

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_life_prerank_tower_model_infer == 1") \
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
          kess_service = "{{explore_life_prerank_tower_model_infer_service}}",
          timeout_ms = 100,
          send_common_attrs = self.explore_life_prerank_user_feture(),
          recv_item_attrs = [
            {"name": "life_click", "as": "prerank_life_ctr"},
          ],
          request_type = "{{explore_life_prerank_request_type}}",
          for_predict = True,
        ) \
      .end_()
