from retrieval.retrieval_module import RetrievalModule

class SplashVarA2aRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_ann_embedding(
        kess_service = "{{fountain_splash_var_a2a_retr_ann_service}}",
        timeout_ms = 50,
        reason = 1,
        items_from_attr = ["sourcePidAuthorId"],
        bound_type = {
          "top_k": "{{fountain_splash_var_a2a_sim_author_num}}",
        },
        algo_type = {
          "scann": {}
        },
        space = "cosine",
        src_data_type = "author",
        src_bucket = "author",
        dest_bucket = "author_bucket",
        save_result_to_common_attr = "sim_aids"
      ) \
      .retrieve_by_remote_index(
        kess_service = "{{fountain_splash_var_a2a_retr_index_service}}",
        timeout_ms = 50,
        reason = self.reason,
        querys = [
          {
            "query": "{{fountain_splash_var_a2a_ordering}}:{{sim_aids}}",
            "search_num": "{{fountain_splash_var_a2a_num_per_author}}",
            "random_search": 0
          }
        ]
      ) \
      .deduplicate() \
      .limit("{{fountain_splash_var_a2a_cand_num}}")
