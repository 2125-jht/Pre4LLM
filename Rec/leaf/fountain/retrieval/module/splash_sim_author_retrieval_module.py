from retrieval.retrieval_module import RetrievalModule

class SimAuthorRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_redis(
        cluster_name = "recoUserPreference",
        reason = self.reason,
        retrieve_num = "{{fountain_sim_author_retr_splash_max_author_num}}",
        timeout_ms = 15,
        key_from_attr = "SourcePhotoAuthorId",
        key_prefix = "{{fountain_sim_author_retr_splash_key_prefix}}",
        save_result_to_common_attr = "sim_author_retr_author_list",
        item_regex = "(\d+):[0-9]{1,}[.]{0,1}[0-9]*",
        skip = "{{skip_fountain_sim_author_retr_splash}}") \
      .retrieve_by_remote_index(
        kess_service = "{{fountain_sim_author_retr_splash_index_service}}",
        timeout_ms = 50,
        reason = self.reason,
        querys = [
          {
            "query": "authorId2PhotoIdOrderByUploadTime:{{sim_author_retr_author_list}}",
            "random_search": 0,
            "search_num": "{{fountain_sim_author_retr_splash_photo_num_per_author}}"
          }
        ],
        skip = "{{skip_fountain_sim_author_retr_splash}}")
