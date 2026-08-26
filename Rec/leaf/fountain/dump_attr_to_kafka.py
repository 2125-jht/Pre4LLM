def dump_attr_to_kafka(flow, stage_name : str, dump_item_attr_list : list, dump_common_attr_list : list = [], range_end : str = None):
  """
  在该阶段的关键位置, 将全部 item 的重要 item attr 落盘
  """
  flow \
    .if_("enable_dump_attrs_to_kafka == 1")\
      .set_attr_value(
        common_attrs = [
          {
            "name": "dump_stage_name",
            "type": "string",
            "value": stage_name,
          },
        ],
      ) \
      .dump_context(
        common_attrs = [
          "_USER_ID_",
          "_DEVICE_ID_",
          "_REQ_ID_",
          "_REQ_TYPE_",
          "_REQ_TIME_",
          "dump_stage_name"
        ] + dump_common_attr_list,
        include_item_results = True,
        item_attrs = dump_item_attr_list,
        dump_to_attr = "dump_context_str",
        range_end = range_end,
      ) \
      .send_with_kafka(
        common_attr = "dump_context_str",
        topic_name = "reco_explore_leaf_dump_log",
      ) \
    .end_()

  return flow