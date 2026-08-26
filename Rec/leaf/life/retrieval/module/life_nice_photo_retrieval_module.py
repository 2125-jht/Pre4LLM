from retrieval.retrieval_module import RetrievalModule

class LifeNicePhotoRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self):
    self.flow \
      .if_("uIsTnuCrowdUser == 1") \
        .get_kconf_params(
          kconf_configs = [
            {
              "kconf_key": "reco.eyeshot.LifeTabTargetHetuL2Json",
              "json_path": "{{life_target_hetu_version}}",
              "export_common_attr": "life_nice_photo_retr_hetu_l2_list"
            }
          ]
        ) \
        .retrieve_by_redis(
          reason = self.reason,
          retrieve_num = "{{life_nice_photo_retr_num}}",
          retrieve_num_per_key = "{{life_nice_photo_retr_num_per_key}}",
          cluster_name = "recoEyeshotClickHistory",
          timeout_ms = 50,
          key_from_attr = "life_nice_photo_retr_hetu_l2_list", 
          key_prefix = "{{life_nice_photo_retr_key_prefix}}",
          item_separator = ","
        ) \
        .deduplicate() \
        .filter_by_browse_set() \
        .shuffle() \
        .limit(
          size = "{{life_nice_photo_retr_num_final}}"
        ) \
      .end_()
