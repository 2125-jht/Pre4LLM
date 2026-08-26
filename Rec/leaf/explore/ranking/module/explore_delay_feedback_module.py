from ranking import CommonModule
from ranking.module.fetch_user_colossus_info_module import photo_colossus_features

class ExploreDelayFeedbackModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)
      
    def delay_feedback_model_user_feature(self):
      features = [
        {"name": "uId", "as": "user_id"},   
        {"name": "uGender", "as": "user_gender"},
        "user_age_segment",
        "uExploreActiveDays"
      ]

      return features

    def delay_feedback_model_item_feature(self):
      features = [
        "photo_id",
        {"name": "author__id", "as": "author_id"},
        {"name": "author__fans_count", "as": "author_fans_count"}, 

        {"name": "hetu_tag_level_info__hetu_cluster_id", "as": "hetu_cluster_id"},
        {"name": "hetu_tag_level_info__hetu_level_two", "as": "hetu_level_two_tag"},


        {"name": "pPltr", "as": "pltr"},
        {"name": "pPwtr", "as": "pwtr"},
        {"name": "pPlvtr", "as": "plvtr"},
        {"name": "pctr", "as": "pctr"},
        {"name": "pPcmtr", "as": "pcmtr"},
        {"name": "pPcmef", "as": "pcmef"},
        {"name": "pPptr", "as": "pptr"},
        {"name": "pPsvtr", "as": "psvtr"},

        "pctr_index",
        "pltr_index",
        "pwtr_index",
        "pvtr_index",
        "plvtr_index",

        {"name": "empirical_ctr", "as": "emp_ctr"},
        {"name": "empirical_ltr", "as": "emp_ltr"},
        {"name": "empirical_wtr", "as": "emp_wtr"}
      ]

      return features
    
    def process(self) -> None:
        self.flow \
          .if_("enable_explore_fr_delay_feedback_model == 1") \
            .delegate_enrich(
              kess_service = "{{explore_fr_delay_feedback_model}}",
              recv_item_attrs = [
                {"name": "ct_oa_orv", "as": "click_and_future_revisit_value"},
                {"name": "oa_orv", "as": "future_revisit_value"},
                {"name": "ct_oar", "as": "click_and_future_open_app_rate"},
                {"name": "oar", "as": "future_open_app_rate"},
              ],
              timeout_ms = 80,
              send_item_attrs = self.delay_feedback_model_item_feature(),
              send_common_attrs = self.delay_feedback_model_user_feature(),
              request_type = "default",
              partition_size = "{{explore_fr_delay_feedback_model_predict_partition_size}}",
            ) \
          .end_if_()
