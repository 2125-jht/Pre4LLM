from retrieval.retrieval_module import RetrievalModule

class PopRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .gen_common_attr_by_lua(
        attr_map = {
          "cluster_redis_key": "cluster_redis_prefix .. _USER_ID_"
        }
      ) \
      .get_common_attr_from_redis(
        cluster_name = "recoColossusTriggers",
        timeout_ms = 100,
        redis_params = [
          {
            "redis_key": "{{cluster_redis_key}}",
            "output_attr_name": "user_cluster"
          }
        ]
      ) \
      .if_("user_cluster ~= nil") \
        .retrieve_by_redis(
          reason = self.reason,
          retrieve_num = 5000,
          cluster_name = "recoColossusTriggers",
          timeout_ms = 100,
          key_from_attr = "user_cluster",
          key_prefix = "{{photos_redis_prefix}}",
          item_separator = ",",
        ) \
        .filter_by_common_attr(
          common_attr = ["browse_screen__pid_list"]
        ) \
        .shuffle() \
        .limit("{{result_num}}") \
      .end_()
