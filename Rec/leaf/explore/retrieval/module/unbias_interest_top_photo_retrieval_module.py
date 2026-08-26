from retrieval.retrieval_module import RetrievalModule

class UnbiasInterestTopPhotoRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "explore_unbias_interest_top_photo_retr_cid_prefix", "as": "key_prefix"},
          "basic_info_age_segment_v2",
          "basic_info_gender_v2",
        ],
        export_common_attr = [
          "user_age_gender_key",
        ],
        function_name = "GetUserAgeGenderKey",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .get_kconf_params(
        kconf_configs = [{
          "kconf_key": "{{reco_kconf_key_param}}",
          "json_path": "{{user_age_gender_key}}",
          "default_value": "",
          "export_common_attr": "unbias_interest_cids_str"
        }]
      ) \
      .split_string(
        input_common_attr = "unbias_interest_cids_str",
        output_common_attr = "unbias_interest_cids",
        delimiters = ",",
        skip_empty_tokens = True,
        trim_spaces = True,
      ) \
      .if_("enable_only_inactive_user_crows == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "unbias_interest_cids", "as": "key_list"},
            {"name": "explore_unbias_interest_top_photo_retr_num_cid", "as": "max_key_cnt"},
            {"name": "explore_unbias_interest_top_photo_retr_num_per_cid", "as": "max_photo_cnt"},
            {"name": "unbias_pid_5w_set_ptr", "as": "photo_pool_map_ptr"},
            {"name": "explore_unbias_interest_top_photo_retr_cid_decay_rate", "as": "cid_decay_rate"},
            {"name": "explore_unbias_interest_top_photo_retr_cid_random_range", "as": "cid_random_range"},
          ],
          export_common_attr = [
            {"name": "selected_cids", "as": "unbias_interest_selected_cids"},
            {"name": "final_photo_pool", "as": "unbias_interest_top_photo_id_list"},
          ],
          function_name = "SelectPhotoPool",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .else_() \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "unbias_interest_cids", "as": "key_list"},
            {"name": "explore_unbias_interest_top_photo_retr_num_cid", "as": "max_key_cnt"},
            {"name": "explore_unbias_interest_top_photo_retr_num_per_cid", "as": "max_photo_cnt"},
            {"name": "unbias_interest_top_photo_map_ptr", "as": "photo_pool_map_ptr"},
            {"name": "explore_unbias_interest_top_photo_retr_cid_decay_rate", "as": "cid_decay_rate"},
            {"name": "explore_unbias_interest_top_photo_retr_cid_random_range", "as": "cid_random_range"},
          ],
          export_common_attr = [
            {"name": "selected_cids", "as": "unbias_interest_selected_cids"},
            {"name": "final_photo_pool", "as": "unbias_interest_top_photo_id_list"},
          ],
          function_name = "SelectPhotoPool",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_() \
      .shuffle_list_attr(
        common_attr = "unbias_interest_top_photo_id_list"
      ) \
      .retrieve_by_common_attr(
        attr = "unbias_interest_top_photo_id_list",
        reason = self.reason
      ) \
      .filter_by_common_attr(
        common_attr = [
          "browse_screen__pid_list"
        ]
      ) \
      .filter_by_browse_set() \
      .limit(
        size = "{{cand_num}}"
      )
