from retrieval.retrieval_module import RetrievalModule

class SplashFocalLifeAnnRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .split_string(
          input_common_attr = "fountain_splash_focal_life_community_hetu_l1_list_str",
          output_common_attr = "fountain_splash_focal_life_community_hetu_l1_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True,
        ) \
      .set_attr_value(
        common_attrs = [
          {
            "name" : "fountain_splash_focal_life_default_value_if_attr_list_is_empty",
            "type" : "int",
            "value" : 1
          }
        ]
      ) \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "fountain_splash_focal_life_community_hetu_l1_list", "as": "attr_list"},
          {"name": "fountain_splash_focal_life_default_value_if_attr_list_is_empty", "as": "default_value_if_attr_list_is_empty"},
          {"name": "source_hetu_level_one", "as": "attrs"},
        ],
        export_common_attr = [
          {"name": "is_in_set", "as": "is_community_hetu_l1_check_ok"},
        ],
        function_name = "CommonIntListAttrIsInSet",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .log_debug_info(
        common_attrs = [
          "is_community_hetu_l1_check_ok"
        ],
        for_debug_request_only = True
      ) \
      .if_("(is_community_hetu_l1_check_ok or 0) == 1") \
        .retrieve_by_ann_embedding(
          kess_service = "{{fountain_splash_focal_life_ann_service}}",
          timeout_ms = 50,
          reason = self.reason,
          items_from_attr = ["featureSourcePId"],
          bound_type = {
            "total_limit": "{{fountain_splash_focal_life_retr_num}}",
          },
          algo_type = {
            "scann": {},
          },
          src_data_type = "photo_src",
          src_bucket = "photo_src",
          dest_bucket = "{{fountain_splash_focal_life_dest_bucket}}",
          dest_bucket_item_type = 1,
        ) \
      .end_()