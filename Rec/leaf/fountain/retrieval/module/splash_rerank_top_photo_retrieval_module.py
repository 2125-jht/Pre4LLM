from retrieval import RetrievalModule

class SplashRerankTopPhotoRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .retrieve_by_common_attr(
        attr = "fountain_rerank_top_photo_id_retrieval_list",
        reason = self.reason
      ) \
      .filter_by_common_attr(
        common_attr = [
          "browse_screen__pid_list"
        ]
      ) \
      .limit(
        size = "{{cand_num}}"
      )