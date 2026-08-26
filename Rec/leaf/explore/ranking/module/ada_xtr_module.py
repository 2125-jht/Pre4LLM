from ranking import CommonModule

class AdaXtrModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def user_feature(self):
      features = [
        "uId",
        "dId",
        "uFollowCount",
        "uFansCount",
        "uUploadCount",
        "uUploadRate",
        "uRiskLevel",
        "uClientId",
        "uVisitMod",
        "uNetwork",
        "uClickPids",
        "uLikePids",
        "uFollowAids",
        "uCityId",
        "uProvinceId",
        "uGender",
        "uInferGender",
        "uTrueGender",
        "uBasicGender",
        "uInferYear",
        "uTrueYear",
        "uBasicAge",
        "uAppList",
      ]

      for i in range(30):
          for suffix in ["", "aid_", "tag_", "play_"]:
              features.append("longview_" + suffix + str(i))
      
      for i in range(30):
          for suffix in ["", "aid_", "tag_", "play_"]:
              features.append("shortview_" + suffix + str(i))
      
      for key in ["uHotShow", "uHotClick", "uHotLike", "uHotFollow", "uHotHate", "uHotCollect", "uHotForward"]:
          for suffix in ["1m", "5m", "30m", "1h", "1d", "100n", "1000n"]:
              features.append(key + suffix)
      
      return features

    def process(self) -> None:
      """leave empty function by AutoDelete"""

    def post_process(self) -> None:
      self.flow \
        .log_debug_info(
          common_attrs = [
            "user_ada_weight_tensor", "longview_", "shortview_", "cnt_"
          ] + self.user_feature(),
          for_debug_request_only = True
        )
