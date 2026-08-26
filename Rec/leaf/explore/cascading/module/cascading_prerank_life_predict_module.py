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
    """leave empty function by AutoDelete"""
