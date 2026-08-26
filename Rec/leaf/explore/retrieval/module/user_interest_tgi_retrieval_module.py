from retrieval.retrieval_module import RetrievalModule

class UserInterestTgiRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
      super().__init__(name)
  
  def process(self):
    self.flow \
      .if_("enable_divide_active_degree == 0 or find_user_active_degree == 1") \
        .retrieve_by_remote_colossusdb_index(
          client_kconf = "colossus.inverted_index_kconf_client.explore_lbs_retr_index_client",
          reason = self.reason,
          querys = [
            {
              "query": "texNexLevel3:{{" + self.input_tgi_list + "}}",
              "search_num": "{{search_num}}",
              "max_attr_num": "{{retrieve_num}}"
            }
          ]
        ) \
        .deduplicate() \
        .filter_by_common_attr(
          common_attr = ["browse_screen__pid_list"]
        ) \
        .limit(
          size = "{{request_num}}"
        ) \
      .end_()

  @property
  def input_tgi_list(self) -> str:
    return self.config.get("input_tgi_list")