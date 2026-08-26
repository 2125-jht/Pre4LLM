from retrieval.retrieval_module import RetrievalModule

class PicCrowdRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .get_common_attr_from_redis(
        cluster_name = "explorePicCrowd",
        redis_params = [
          {
            "key_prefix": "{{explore_pic_crowd_retr_uid_prefix}}",
            "redis_key": "{{_USER_ID_}}",
            "redis_value_type": "string",
            "output_attr_type": "string",
            "output_attr_name": "user_crowd_string",
          }
        ],
        is_async = True
      ) \
      .split_string(
        input_common_attr = "user_crowd_string",
        output_common_attr = "user_crowd_list",
        delimiters=",",
      ) \
      .retrieve_by_redis(
        reason = self.reason,
        cluster_name = "explorePicCrowd",
        timeout_ms = 10,
        retrieve_num = "{{explore_pic_crowd_retr_num}}",
        key_from_attr = "user_crowd_list",
        key_prefix = "{{explore_pic_crowd_retr_crowd_prefix}}",
        retrieve_num_per_key = "{{explore_pic_crowd_retr_num_per_key}}",
        item_separator = ",",
      ) \
      .deduplicate() \
      .filter_by_browse_set() \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"]
      ) \
      .shuffle() \
      .limit(size = "{{explore_pic_crowd_retr_num_final}}") \
      .set_attr_value(
        item_attrs = [
          {
            "name": "operation_pic", #  运营定向分发结果
            "type": "int",
            "value": 1
          }
        ]
      )