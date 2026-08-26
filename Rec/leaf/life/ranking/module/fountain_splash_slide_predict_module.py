from ranking import CommonModule
from ranking.fountain_ranking_features import user_features_v2, photo_features, photo_pxtr_features

class FountainSplashSlidePredictModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .delegate_enrich(
        skip = "{{skip_fountain_splash_slide_predict}}",
        kess_service = "{{fountain_splash_slide_predict_kess_service}}",
        recv_item_attrs = [{"name":"slide", "as":"fountain_splash_slide"}],
        timeout_ms = 150,
        send_item_attrs = [feature["name"] for feature in photo_features if feature["name"] not in photo_pxtr_features],
        send_common_attrs = user_features_v2,
        request_type = "kai_predict",
        partition_size = "{{fountain_splash_slide_predict_partition_size}}",
      )

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_attrs = [
          "fountain_splash_slide"
        ],
        for_debug_request_only = True
      )

