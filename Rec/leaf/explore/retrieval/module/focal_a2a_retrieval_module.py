from retrieval import RetrievalModule

class FocalA2aRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .copy_attr(
        attrs=[{
          "from_common": "profile_v1_interaction_trigger_aids",
          "to_common": "interact_aids"
        }]
      ) \
      .shuffle_list_attr(
        common_attr= "interact_aids"
      ) \
      .truncate(
        size_limit = "{{interact_num}}",
        item_list_from_attr = "interact_aids"
      ) \
      .copy_attr(
        attrs=[{
          "from_common": "colossus_user_info__trigger_author_list",
          "to_common": "colossus_aids"
        }]
      ) \
      .shuffle_list_attr(
        common_attr= "colossus_aids"
      ) \
      .truncate(
        size_limit = "{{colossus_num}}",
        item_list_from_attr = "colossus_aids"
      ) \
      .copy_attr(
        attrs=[{
          "from_common": "profile_v1_click_trigger_aids",
          "to_common": "realtime_aids"
        }]
      ) \
      .truncate(
        size_limit = "{{realtime_num}}",
        item_list_from_attr = "realtime_aids"
      ) \
      .pack_common_attr(
        input_common_attrs = ["realtime_aids", "interact_aids", "colossus_aids"],
        output_common_attr = "trigger_aids",
        deduplicate = True
      ) \
      .retrieve_by_ann_embedding(
        kess_service = "{{ann_service}}",
        timeout_ms = 50,
        reason = 1,
        items_from_attr = ["trigger_aids"],
        bound_type = {
          "top_k": "{{sim_author_num}}",
        },
        algo_type = {
          "scann": {}
        },
        space = "cosine",
        src_data_type = "author",
        src_bucket = "author",
        dest_bucket = "author",
        save_result_to_common_attr = "sim_aids"
      ) \
      .deduplicate(
        item_list_from_attr = "sim_aids",
      ) \
      .retrieve_by_remote_index(
        kess_service = "{{index_service}}",
        timeout_ms = 50,
        reason = self.reason,
        querys = [
          {
            "query": "{{ordering}}:{{sim_aids}}",
            "search_num": "{{num_per_author}}",
            "random_search": 0
          }
        ]
      ) \
      .deduplicate() \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"]
      ) \
      .limit("{{result_num}}")
