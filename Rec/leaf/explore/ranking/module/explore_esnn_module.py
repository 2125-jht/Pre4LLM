from ranking import CommonModule
from ranking.module.fetch_user_colossus_info_module import photo_colossus_features

class ExploreESNNModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("enable_explore_fr_esnn_kai2 == 1") \
        .explore_custom_trim_user_info(
          user_info_attr = "userInfo",
          save_trimed_user_info_to_attr = "deep_esnn_trimmed_user_info",
          trim_user_info = [
            "active_days",
            "basic_info.age_segment",
            "location.city_id",
            "gender",
            "infer_gender",
            "true_gender",
            "upload_count",
            "infer_year",
            "follow_count",
            "fans_count",
            "location.city_level",
            "request_location.province_id"
          ]
        ) \
        .delegate_enrich(
          kess_service = "{{explore_fr_esnn_kess_service}}",
          recv_item_attrs=[
            {"name": "es_score", "as": "esnn_model_score"},
          ],
          timeout_ms = 100,
          send_item_attrs = [
            {"name": "pltr", "as": "pPltr"},
            {"name": "pwtr", "as": "pPwtr"},
            {"name": "pftr", "as": "pPftr"},
            {"name": "psvr", "as": "pPsvtr"},
            {"name": "plvtr", "as": "pPlvtr"},
            {"name": "pcmtr", "as": "pPcmtr"},
            {"name": "pptr", "as": "pPptr"},
            {"name": "pcmef", "as": "pPcmef"},
            {"name": "phtr", "as": "pPhtr"},
            {"name": "pctr", "as": "pPctr"},
            {"name": "pvtr", "as": "pPvtr"},
            {"name": "pepstr", "as": "pPepstr"},
            {"name": "awesome_wtd", "as": "pPwtd"},
            {"name": "pcltr", "as": "pPcltr"},
            {"name": "cpr", "as": "pPcpr"},
            {"name": "fetr", "as": "pPfetr"},
            {"name": "pdtr", "as": "pPdtr"},
            {"name": "fr_score1", "as": "pPfrScore1"},
            {"name": "fr_score2", "as": "pPfrScore2"},
          ],
          send_common_attrs = [
            { "name": "deep_esnn_trimmed_user_info", "as": "user_info_str"},
          ],
          partition_size = "{{explore_fr_esnn_partition_size}}",
        ) \
      .end_if_()
