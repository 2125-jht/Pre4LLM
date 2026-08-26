from retrieval import CommonModule

class ColossusParseModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
        .if_("enable_fountain_colossus_parse == 1") \
          .gsu_common_colossus_resp_retriever(
            colossus_resp_attr = "colossus_resp_v2",
            colossus_service_name = "grpc_colossusSimV2",
            item_key_field = "photo_id",
            item_time_field = "timestamp",
            item_fields = dict(
              photo_id = "colossus_photo_id_list",
              play_time = "colossus_play_time_list",
              label = "colossus_label_list",
              timestamp = "colossus_timestamp_list",
              author_id = "colossus_author_id_list",
              channel = "colossus_channel_list",
              duration = "colossus_duration_list",
              tag = "colossus_tag_list",
            ),
            to_common_attr = True,
            max_item_num = "{{fountain_colossus_parse_max_len}}",
          ) \
        .end_()      

    if self.calc_fountain_possible:
      self.flow \
        .if_("enable_fountain_possible_calc == 1") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              "colossus_resp_v2",
              { "name": "fountain_possible_history_item_threshold", "as": "history_item_threshold" },
              { "name": "fountain_possible_explore_vv_threshold", "as": "explore_vv_threshold" },
              { "name": "fountain_possible_fountain_vv_threshold", "as": "fountain_vv_threshold" },
            ],
            export_common_attr = [
              "is_fountain_possible",
            ],
            function_name = "IsFountainPossible",
            class_name = "ExploreLightFunctionSetV2",
          ) \
          .perflog_attr_value(
            check_point = "fountain_first_refine",
            common_attrs = [
              "is_fountain_possible",
            ],
          ) \
        .end_()

  @property
  def calc_fountain_possible(self) -> str:
    return self.config.get("calc_fountain_possible", False)