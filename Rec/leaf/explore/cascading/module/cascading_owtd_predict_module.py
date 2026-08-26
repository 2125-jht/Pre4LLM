from cascading import CommonModule

class CascadingOwtdPredictModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def consume_time_user_feture(self):
    features = [
      "uId",
      "dId",
      "uLikePids",
      "uFollowAids",
      "uClickPidsV1",
      "uFollowCount",
      "uFansCount",
      "uUploadCount",
      "uUploadRate",
      "uCityId",
      "uProvinceId",
      "uGender",
      "uTrueGender",
      "uInferYear",
      "uTrueYear",
      "uBasicAge",
      "uNetwork",
      "uExpCtr",
      "uExpLtr",
      "uExpWtr",
      "uExpFtr",
      "uExpLvtr",
      "uExpSvtr",
      "uAvgWatchTime",
    ]

    for key in ["uHotShow", "uHotClick", "uHotLike", "uHotFollow", "uHotHate"]:
        for suffix in ["5m", "1h", "1d"]:
            features.append(key + suffix)
    for key in ["1", "2", "3"]:
        features.append("uLongTermHetuLevel" + key + "topN")
    
    features.extend([
      "uClickPidsV1Hetu1",
      "uClickPidsV1Hetu2",
      "uLikePidsV1Hetu1",
      "uLikePidsV1Hetu2",
      "ufollowAidsV1Hetu1",
      "ufollowAidsV1Hetu2",
      "uRealShowNoActionPids",
      "uRealShowNoActionAids",
      "uRealshowNoActionHetu1",
      "uRealshowNoActionHetu2",
      "uRealshowNoActionHetu3",
      "uRealshowNoActionHetuTag",
      "uViewPidListV1",
      "uViewAidListV1",
      "uEffectiveViewLabelListV1",
      "uLongViewLabelListV1",
      "uShortViewLabelListV1",
      "uShortViewPidListV1",
      "uShortViewAidListV1",
      "uEffectiveViewPidListV1",
      "uEffectiveViewAidListV1",
      "uLongViewPidListV1",
      "uLongViewAidListV1",
      "uFinishViewPidListV1",
      "uFinishViewAidListV1",
      "uFinishViewHetu1ListV1",
      "uFinishViewHetu2ListV1",
      "uNonFinishViewPidListV1",
      "uNonFinishViewAidListV1",
      "uViewHetu1ListV1",
      "uViewHetu2ListV1",

      {"name": "user_emp_ltr", "as": "uColossusEmpLtr"},
      {"name": "user_emp_wtr", "as": "uColossusEmpWtr"},
      {"name": "user_emp_ftr", "as": "uColossusEmpFtr"},
      {"name": "user_emp_cmtr", "as": "uColossusEmpCmtr"},
      {"name": "user_emp_eptr", "as": "uColossusEmpPtr"},
      {"name": "user_emp_svtr", "as": "uColossusEmpSvtr"},
      {"name": "user_emp_evtr", "as": "uColossusEmpEvtr"},
      {"name": "user_emp_lvtr", "as": "uColossusEmpLvtr"},
      {"name": "user_emp_fintr", "as": "uColossusEmpFintr"},
      {"name": "user_emp_watch_time", "as": "uColossusAvgWatchTime"},
      {"name": "user_emp_finish_rate", "as": "uColossusAvgFinishRate"},
    ])
    
    return features
  

  def process(self) -> None:
    """leave empty function by AutoDelete"""

      
      
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_attrs = [
            "duration_ms",
            "owtd_label1",
            "owtd_label2",
            "owtd_label3",
            "cascade_ordinal_prob",
            "cascade_ordinal_wtd"
        ],
        common_attrs = [
          "explore_mc_ordinal_duration_list",
          "explore_mc_ordinal_playtime_dist_list"
        ],
        for_debug_request_only = True,
        item_num_limit = 500
      )

