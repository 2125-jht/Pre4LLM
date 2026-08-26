from retrieval import CommonModule

class ExploreNearLineUserUpdateModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .if_("cache_nearline_update_info == 1") \
        .str_format(
          format_string = "%s%d",
          input_attrs = ["nearline_prefix", "_USER_ID_"],
          output_attr = "nearline_redis_key",
        ) \
        .get_common_attr_from_redis(
          cluster_name = "recoUserExptagXtr",
          redis_params = [{
            "redis_key": "{{nearline_redis_key}}",
            "output_attr_name": "explore_nearline_last_update_timestamp_str"
          }]
        ) \
        .gen_common_attr_by_lua(
          attr_map = {
            "explore_nearline_last_update_timestamp" : "tonumber(explore_nearline_last_update_timestamp_str)",
          },
        ) \
        .if_("explore_nearline_last_update_timestamp ~= nil and explore_nearline_last_update_timestamp > 0") \
          .gen_common_attr_by_lua(
            attr_map = {
              "explore_nearline_update_second_str" : "tostring(math.floor(util.GetTimestamp() / 1000000))",
              "explore_nearline_update_gap_second": "math.floor(util.GetTimestamp() / 1000000) - explore_nearline_last_update_timestamp",
            },
          ) \
          .if_("explore_nearline_update_gap_second > nearline_max_gap_second") \
            .set_attr_value(
              common_attrs = [{
                "name": "explore_nearline_user_update_flag",
                "type": "int",
                "value": 1
              }]
            ) \
            .write_to_redis(
              kcc_cluster = "recoUserExptagXtr",
              timeout_ms = 10,
              key = "{{nearline_redis_key}}",
              value = "{{explore_nearline_update_second_str}}",
              expire_second = 600
            ) \
          .end_() \
        .else_() \
          .gen_common_attr_by_lua(
            attr_map = {
              "explore_nearline_update_second_str" : "tostring(math.floor(util.GetTimestamp() / 1000000))"
            },
          ) \
          .write_to_redis(
            kcc_cluster = "recoUserExptagXtr",
            timeout_ms = 10,
            key = "{{nearline_redis_key}}",
            value = "{{explore_nearline_update_second_str}}",
            expire_second = 600
          ) \
        .end_() \
      .end_()
      