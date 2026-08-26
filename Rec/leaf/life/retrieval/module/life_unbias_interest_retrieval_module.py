from retrieval.retrieval_module import RetrievalModule

class LifeUnbiasInterestRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self):
    self.flow \
      .if_("life_unbias_interest_retr_limit_low_active ~= 1 or uIsLifeHighActive ~= 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "life_unbias_interest_cluster_prefix", "as": "key_prefix"},
            "basic_info_age_segment_v2",
            "basic_info_gender_v2",
          ],
          export_common_attr = [
            "user_age_gender_key"
          ],
          function_name = "GetUserAgeGenderKey",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .get_kconf_params(
          kconf_configs = [{
            "kconf_key": "reco.eyeshot.life_unbias_interest_hetu2_map",
            "json_path": "{{user_age_gender_key}}",
            "default_value": "",
            "export_common_attr": "life_unbias_interest_str"
          }]
        ) \
        .split_string(
          input_common_attr = "life_unbias_interest_str",
          output_common_attr = "life_unbias_interest_list",
          delimiters = ",",
          skip_empty_tokens = True,
          trim_spaces = True,
          parse_to_int = True
        ) \
        .if_("enable_life_unbias_interest_adjust_low_active == 1 and uIsLifeHighActive ~= 1") \
          .gen_common_attr_by_lua(
            attr_map={
              "life_unbias_interest_start_hetu_count": "life_unbias_interest_start_hetu_count_low_active",
              "life_unbias_interest_final_hetu_count": "life_unbias_interest_final_hetu_count_low_active",
            }
          ) \
        .end_() \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "user_info_ptr",
            {"name": "life_unbias_interest_list", "as": "unbias_interest_list"},
            {"name": "life_unbias_interest_time_gap_min", "as": "time_gap_min"},
            {"name": "life_unbias_interest_short_play_thresh", "as": "short_play_thresh"},
            {"name": "life_unbias_interest_short_play_rate_thresh", "as": "short_play_rate_thresh"},
            {"name": "life_unbias_interest_start_hetu_count", "as": "start_hetu_count"},
            {"name": "life_unbias_interest_final_hetu_count", "as": "final_hetu_count"},
            {"name": "life_unbias_interest_filter_hate_hetu", "as": "filter_hate_hetu"},
          ],
          export_common_attr = [
            {"name": "final_unbias_interest_list", "as": "life_unbias_interest_list"},
          ],
          function_name = "ShuffleUnbiasInterestList",
          class_name = "ExploreLifeLightFunctionSet"
        ) \
        .if_("enable_life_unbias_interest_retr_redis_v2 == 1") \
          .retrieve_by_redis(
            reason = self.reason,
            retrieve_num = "{{life_unbias_interest_retr_num}}",
            retrieve_num_per_key = "{{life_unbias_interest_retr_num_per_key}}",
            cluster_name = "recoEyeshotClickHistory",
            timeout_ms = 50,
            key_from_attr = "life_unbias_interest_list", 
            key_prefix = "life_unbias_v2_",
            item_separator = ",",
            save_result_to_common_attr = "life_unbias_interest_pid_list"
          ) \
        .else_() \
          .retrieve_by_redis(
            reason = self.reason,
            retrieve_num = "{{life_unbias_interest_retr_num}}",
            retrieve_num_per_key = "{{life_unbias_interest_retr_num_per_key}}",
            cluster_name = "recoEyeshotClickHistory",
            timeout_ms = 50,
            key_from_attr = "life_unbias_interest_list", 
            key_prefix = "life_unbias_",
            item_separator = ",",
            save_result_to_common_attr = "life_unbias_interest_pid_list"
          ) \
        .end_() \
        .shuffle_list_attr(
          common_attr = "life_unbias_interest_pid_list"
        ) \
        .pack_common_attr(
          input_common_attrs = [
            "life_unbias_interest_pid_list",
          ],
          output_common_attr = "life_unbias_interest_pid_list",
          limit_num = "{{life_unbias_interest_retr_num_final}}",
          deduplicate = True
        ) \
        .retrieve_by_common_attr(
          attr = "life_unbias_interest_pid_list",
          reason = self.reason
        ) \
        .filter_by_common_attr(
          common_attr = [
            "browse_screen__pid_list"
          ]
        ) \
      .end_()
