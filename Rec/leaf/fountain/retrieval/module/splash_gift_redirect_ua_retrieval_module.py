from retrieval.retrieval_module import RetrievalModule

class SplashGiftRedirectUARetrievalModule(RetrievalModule):
  def __init__(self, name=str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .retrieve_by_remote_index( # 这里切换远程倒排，使用默认参数
        kess_service = "grpc_recoHotOrderedIndexServer",
        timeout_ms = 80,
        reason = self.reason,
        querys = [ # 这里 search_num 是必须参数，先写死成 10，需要再调
          {
            "query": "authorId2PhotoIdOrderByUploadTime:{{living_certain_aid_list}}",
            "search_num" : 10,
          }
        ],
        save_score_to_attr = "item_score",
      ) \
      .deduplicate()\
      .filter_by_browse_set() \
      .sort(score_from_attr="item_score") \
      .limit("{{gift_redirect_ua_final_result_num_explore_leaf_splash}}") 