from cascading import CommonModule
from cascading.module.cascading_features import *

class CascadingPrerankLtrPredictModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def user_feature(self):
    features = [
      "uId",
      "dId",
      "uHourOfDay",
      "uDayOfWeek",
      "uExpCtr",
      "uExpLtr",
      "uExpWtr",
      "uExpFtr",
      "uExpLvtr",
      "uExpSvtr",
      "uExpHtr",
      "uExpClickCount",
      "uExpLikeCount",
      "uExpFollowCount",
      "uExpForwardCount",
      "uExpHateCount",
      "uFansCount",
      "uClickPidsHot",
      "uClickAidsHot",
      "uLikePidsHot",
      "uLikeAidsHot",
      "uFollowPidsHot",
      "uFollowAidsHot",
      "uCommentPidsHot",
      "uCommentAidsHot",
      "uCollectPidsHot",
      "uCollectAidsHot",
      "uForwardPidsHot",
      "uForwardAidsHot",
      "uPlayViewPidsGlobal",
      "uPlayViewAidsGlobal",
      "uPlayViewDurationGlobal",
      "uPlayViewPlaytimeGlobal",
      "uPlayViewHetu2Global",
      "uPlayViewChannelGlobal",
      "uUploadCount",
      "uCityLevelNew",
      "uCityId",
      "uProvinceId",
      "uGender",
      "uTrueYear",
      "uBasicAge",
      "uNetwork",
      "uIsLowActiveUser",
    ]

    return features
          
  def process(self) -> None:
    """leave empty function by AutoDelete"""
        
