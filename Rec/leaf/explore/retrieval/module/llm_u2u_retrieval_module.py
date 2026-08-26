from retrieval.retrieval_module import RetrievalModule

class LlmU2URetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .delegate_retrieve(
        kess_service = "{{u2u_service_name}}",
        request_type = "default",
        timeout_ms = 60,
        recv_common_attrs = ["llm_sim_user_list"],
        send_common_attrs = [{"name": "similar_user_num", "as": "llm_u2u_sim_top_k"}],
      ) \
      .retrieve_by_remote_index(
        kess_service = "{{index_service_name}}",
        timeout_ms = 30,
        reason = self.reason,
        querys = [
          {
            "query": "usim:{{llm_sim_user_list}}",
            "search_num": 30,
          },
        ],
      ) \
      .limit("{{service_request_num}}")
