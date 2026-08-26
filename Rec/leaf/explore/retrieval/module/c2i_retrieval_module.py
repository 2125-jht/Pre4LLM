from retrieval.retrieval_module import RetrievalModule

class C2IRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .get_common_attr_from_redis(
        cluster_name = "recoEyeshotFollow",
        redis_params = [
          {
            "redis_key": "{{featureUId}}",
            "output_attr_name": "user_cid_string",
            "key_prefix": "cuid"
          }
        ],
        timeout_ms = 25
      ) \
      .split_string(
        input_common_attr = "user_cid_string",
        output_common_attr = "user_cid_list",
        delimiters = ",",
        parse_to_int = True
      ) \
      .retrieve_by_remote_index(
        kess_service = "{{cluster_remote_index_service_name}}",
        timeout_ms = 25,
        reason = self.reason, 
        querys = [
          {
            "query": "cid:{{user_cid_list}}",
            "search_num": "{{cid_index_search_num}}", 
            "max_attr_num": "{{retrieve_num}}"
          }
        ]
      )
