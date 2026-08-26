from retrieval import CommonModule

class SplashKuibaUserModule(CommonModule):
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
          "uShareidCntKV",
          "uOpenShareidCntKV",
          "uOpenDeviceCntKV",
          "uPullNumKV",
          "uShareBringNewDeviceNumKV",
          "uAttributionPerShareKV",
          "uNebulaTopPgtrAidList",
          "uGamoraTopPgtrAidList",
          "uCertainAidListKV",
          "uGiftAllUAKV",
          "uUserKuaishouLivePayTag",
          "uLiveLongViewRedirectKV",
          "uMessageActiveDegreeCode",
          "uToUserShareSendNum30dKV",
          "uHighTimeDauRateKV",
          "uInsideShareActiveDegreeDetailCode"
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
          "uHighTimeDauRateKV"
        ],
        export_common_attr = [
          "active_days_gt_5min_rate"
        ],
        function_name = "ParseUserHighTimeInfo",
        class_name = "ExploreLightFunctionSetV2",
      ) 
