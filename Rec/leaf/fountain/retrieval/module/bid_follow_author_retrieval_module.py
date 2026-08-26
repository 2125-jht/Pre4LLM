from retrieval.retrieval_module import RetrievalModule

class BidFollowAuthorRetrievalModule(RetrievalModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    # 双关作者召回
    def process(self) -> None:
      self.flow \
        .retrieve_by_remote_index(
          # 关注作者
          kess_service = "{{fountain_bid_follow_author_retrieval_query_server}}",
          timeout_ms = 100,
          reason = self.reason,
          reset_item_type = 1,
          common_query = "",
          querys = [{
            "query": "{{fountain_bid_follow_author_retrieval_query}}:{{friendAids}}",
            "search_num": "{{fountain_bid_follow_author_search_num}}"
          }],
          default_search_num = 100,
          default_total_request_num = "{{fountain_bid_follow_author_search_total_num}}",
          # save_score_to_attr = "bid_follow_photo_score",
        ) \
        .shuffle()
