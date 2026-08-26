from ranking import CommonModule

class UserLongTermEngagementModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def user_feture(self):
      features = [
        "uId",
        "dId",
        "uFollowCount",
        "uFansCount",
        "uUploadCount",
        "uCityId",
        "uProvinceId",
        "uGender",
        "uInferYear",
        "uTrueYear",
        "uBasicAge",
        "uNetwork",
        "uIsDouyin",
        "uCityLevelNew",
        "uHourOfDay",
        "uDayOfWeek",
        "uIsLowActiveUser",
        "uIsHighPotentialUser",
        "uIsLa",
        "uAgeGenderCity",

        "uClickPidsGlobal",
        "uClickAidsGlobal",
        "uClickHetu1Global",
        "uClickHetu2Global",
        "uClickChannelsGlobal",
        "uClickPidsGlobal_LEN",
        "uLikePidsGlobal",
        "uLikeAidsGlobal",
        "uFollowPidsGlobal",
        "uFollowAidsGlobal",
        "uCommentPidsGlobal",
        "uCommentAidsGlobal",
        "uHatePidsGlobal",
        "uHateAidsGlobal",
        "uClickPidsHot",
        "uClickAidsHot",
        "uClickHetu1Hot",
        "uClickHetu2Hot",
        "uClickPidsHot_LEN",
        "uLikePidsHot",
        "uLikeAidsHot",
        "uFollowPidsHot",
        "uFollowAidsHot",
        "uCommentPidsHot",
        "uCommentAidsHot",
        "uHatePidsHot",
        "uHateAidsHot",
        "uClickPidsFountain",
        "uClickAidsFountain",
        "uLikePidsFountain",
        "uLikeAidsFountain",
        "uFollowPidsFountain",
        "uFollowAidsFountain",
        "uCommentPidsFountain",
        "uCommentAidsFountain",
        "uHatePidsFountain",
        "uHateAidsFountain",
        "uShortViewPidsHot",
        "uShortViewAidsHot",
        "uLongViewPidsHot",
        "uLongViewAidsHot",
        "uShortViewPidsFountain",
        "uShortViewAidsFountain",
        "uLongViewPidsFountain",
        "uLongViewAidsFountain",

        "uLongTermHetuLevel1topN",
        "uLongTermHetuLevel2topN",
        "uLongTermHetuLevel3topN"
      ]

      for key in ["uHotShow", "uHotClick", "uHotLike", "uHotFollow", "uHotHate"]:
          for suffix in ["5m", "1h", "1d"]:
              features.append(key + suffix + "Hot")
      
      return features

    def photo_features(self):
        features = [
            {"name": "photo_id", "as": "pId"},
            {"name": "author__id", "as": "aId"},
            {"name": "author__fans_count", "as": "aFansCount"},
            {"name": "duration_ms", "as": "pDurationMs"},
            {"name": "upload_type", "as": "pUploadType"},
            {"name": "reason", "as": "pReason"},
            {"name": "location__city_id", "as": "pCityId"},
            {"name": "location__province_id", "as": "pProvinceId"},
            {"name": "hetu_tag_level_info__hetu_level_one", "as": "pHetuTagLevel1"},
            {"name": "hetu_tag_level_info__hetu_level_two", "as": "pHetuTagLevel2"},
            {"name": "hetu_tag_level_info__hetu_level_three", "as": "pHetuTagLevel3"},
            {"name": "hetu_tag_level_info__hetu_level_five", "as": "pHetuTagLevel5"},
            {"name": "hetu_tag_level_info__hetu_tag", "as": "pHetuTagLevelTag"},
            {"name": "hetu_tag_level_info__hetu_face_id", "as": "pHetuTagFaceId"},
            {"name": "hetu_tag_level_info__hetu_cluster_id", "as": "pHetuClusterId"},
            {"name": "user_hash_tag_id", "as": "pHashTagIdList"},
            {"name": "author__upload_count", "as": "aUploadCount"},
            {"name": "music", "as": "pMusic"},
            {"name": "audit_hot_high_tag_level", "as": "pAuditHotHighTagLevel"},
            {"name": "content_safety_level_with_namespace__level_hot_online", "as": "pLevelHotOnline"},
            {"name": "cascade_pctr", "as": "pMcPctr"},
            {"name": "cascade_pltr", "as": "pMcPltr"},
            {"name": "cascade_pwtr", "as": "pMcPwtr"},
            {"name": "cascade_plvtr", "as": "pMcPlvtr"},
            {"name": "cascade_psvtr", "as": "pMcPsvtr"},          
            {"name": "explore_stat__show_count", "as": "pHotShow"},
            {"name": "explore_stat__click_count", "as": "pHotClick"},
            {"name": "explore_stat__like_count", "as": "pHotLike"},
            {"name": "explore_stat__follow_count", "as": "pHotFollow"},
            {"name": "explore_stat__negative_count", "as": "pHotHate"},
            {"name": "is_picture", "as": "pIsPicture"},
            {"name": "is_follow_author", "as": "aIsFollowAuthor"},
            "pAgeHour",
            "pChannel",
            "pEmpCtrHot",
            "pEmpLtrHot",
            "pEmpWtrHot",
            "pEmpCmtrHot",
            "pEmpRealShowCountHot",
            "pEmpClickCountHot",
            "pEmpWatchTimeHot",
        ]

        for key in ["Hetu1", "Hetu2", "Hetu5", "HetuTag"]:
            for suffix in ["100n", "1000n"]:
                features.append("pShortStatShow" + key + suffix)
                features.append("pShortStatClick" + key + suffix)
                features.append("pShortStatClickRate" + key + suffix)

        return features

    def process(self) -> None:
      """leave empty function by AutoDelete"""

    def post_process(self) -> None:
      self.flow \
        .log_debug_info(
          common_attrs = self.user_feture(),
            for_debug_request_only = True
        )