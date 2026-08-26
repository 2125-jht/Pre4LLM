from common.retrieval import RetrievalModule

class LLMHetuRetrievalModule(RetrievalModule):
  def __init__(self, name: str, config_dir: str) -> None:
    super().__init__(name, config_dir)
  
  def process(self) -> None:
    self.flow \
      .split_string(
        input_common_attr = "uLLMHetuKV",
        output_common_attr = "hetu_list",
        delimiters=",",
      ) \
      .retrieve_by_remote_index(
        kess_service = "{{index_service_name}}",
        timeout_ms = 60,
        reason = self.reason,
        querys = [
          { 
          "query": "hetu_tag_v2:{{hetu_list}}",
          "search_num": "{{index_search_num}}"
          },
        ],
        default_total_request_num = "{{service_request_num}}",
        default_random_search = 1,
      )
