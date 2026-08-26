from retrieval import RetrievalModule

class FocalU2aRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .get_remote_embedding(
        kess_service = "{{embedding_service}}",
        timeout_ms = 10,
        id_converter = {
          "type_name": "mioEmbeddingIdConverter"
        },
        slot = 100,
        save_to_common_attr = True,
        output_embedding_list_attr = "user_embedding_list",
        query_source_type = "user_id",
        is_raw_data = False,
        is_raw_data_list = False
      ) \
      .retrieve_by_ann_embedding(
        reason = 1,
        kess_service = "{{ann_service}}",
        space = "ip",
        timeout_ms = 50,
        items_from_attr = ["_USER_ID_"],
        embeddings_from_attr = ["user_embedding_list"],
        bound_type = {
          "total_limit": "{{author_num}}"
        },
        algo_type = {
          "scann": {}
        },
        src_data_type = "user",
        src_bucket = "user",
        dest_bucket = "author",
        save_result_to_common_attr = "author_list"
      ) \
      .retrieve_by_remote_index(
        kess_service = "{{index_service}}",
        timeout_ms = 50,
        reason = self.reason,
        querys = [
          {
            "query": "authorId2PhotoIdOrderByExploreStatScore:{{author_list}}",
            "search_num": "{{num_per_author}}",
            "random_search": 0
          }
        ]
      ) \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"]
      ) \
      .limit("{{result_num}}")
