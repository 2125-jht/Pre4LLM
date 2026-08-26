from retrieval import CommonModule

class KuibaUserModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .parse_protobuf_from_string(
        input_attr = "kuibaUserAttrStr",
        output_attr = "kuiba_user_attr",
        class_name = "kuiba::PredictItem",
      ) \
      .extract_kuiba_sample_attr(
        output_attrs = [
          "uBuyerEffectiveType",
          "uGamoraUploadDayNum30d",
          "uNebulaUploadDayNum30d",
          "uLastUploadDays",
          "uInsidePlayDuration7dKV",
          "uShareidCntKV",
          "uOpenShareidCntKV",
          "uOpenDeviceCntKV",
          "uPullNumKV",
          "uShareBringNewDeviceNumKV",
          "uUserKuaishouLivePayTag",
          "uAttributionPerShareKV",
          "uIsNicePicCsm",
          "uNebulaTopPgtrAidList",
          "uGamoraTopPgtrAidList",
          "uCertainAidListKV",
          "uGiftAllUAKV",
          "uLiveLongViewRedirectKV",
          "uMessageActiveDegreeCode",
          "uToUserShareSendNum30dKV",
          "uMultiDimensionGroupKV",
          "uMultiDimensionGroupDetailKV",
          "uSexyInterestScore",
          "uToleranceScoreKV",
          "uLLMHetuKV",
          "uInsideShareActiveDegreeDetailCode",
          "uOldMmuClusterId300ListList",
          "uLikeActiveScore",
          "uCommentActiveScore",
          "uShareActiveScore",
          "uFollowActiveScore",
          "uCollectActiveScore",
          "uExploreShortValidInterestAndScoreList",
          "uExploreGamoraInterestList",
          "uJobIdLv1KV",
          "uJobIdLv2KV",
          "uStudentLabelV1KV",
          "uBirthLabelV1KV",
          "uMarriageLabelV1KV",
          "uIsNorthKV"
        ],
        predict_item = "kuiba_user_attr",
        is_common_attr = True
      ) \
      .split_string(
        input_common_attr = "uGiftAllUAKV",
        output_common_attr = "user_gift_all_ua_list",
        delimiters = ",",
        parse_to_int = True,
      ) \
      .split_string(
        input_common_attr = "uCertainAidListKV",
        output_common_attr = "user_certain_aid_list",
        delimiters = ",",
        parse_to_int = True,
      ) \
      .split_string(
        input_common_attr = "uLiveLongViewRedirectKV",
        output_common_attr = "user_live_lv_aid_list",
        delimiters = ",",
        parse_to_int = True,
      ) \
      .pack_common_attr(
        input_common_attrs = [
          "uNebulaTopPgtrAidList",
          "uGamoraTopPgtrAidList",
          "user_certain_aid_list",
          "user_gift_all_ua_list",
          "user_live_lv_aid_list"],
        output_common_attr = "living_certain_aid_list",
        deduplicate = True
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "uBuyerEffectiveType"
        ],
        export_common_attr = [
          "merchant_buyer_type"
        ],
        function_name = "MerchantCalcBuyerType",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          "uLastUploadDays",
          "uInsidePlayDuration7dKV",
          "fountain_inside_play_duration_switch",
          "fountain_inside_play_duration_threhold",
          "enable_produce_need_new_produce_user",
          "enable_produce_need_14d_produce_user",
          "enable_produce_need_7d_produce_user",
          "enable_produce_need_1d_produce_user"
        ],
        export_common_attr = [
          {"name": "produce_user_type", "as": "fountain_produce_user_type"},
          {"name": "produce_consume_deep_user", "as": "fountain_produce_consume_deep_user"}
        ],
        function_name = "JudgeProduceUserType",
        class_name = "ExploreLightFunctionSetV2",
      )