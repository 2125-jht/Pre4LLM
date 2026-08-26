from retrieval.retrieval_module import RetrievalModule

class AnnU2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_ann_embedding(
        kess_service = "{{kess_service}}",
        space = self.space,
        timeout_ms = 100,
        reason = self.reason,
        items_from_attr = ["_USER_ID_"],
        attr_single_limit = 50,
        bound_type = {
          "total_limit": "{{total_limit}}",
        },
        algo_type = {
          "scann": {}
        },
        src_bucket = self.src_bucket,
        dest_bucket = self.dest_bucket,
        dest_bucket_item_type = 1,
        skip = "{{retrieval_skip}}"
      )

  @property
  def space(self) -> str:
    return self.config.get("space", "cosine")

  @property
  def src_bucket(self) -> str:
    return self.config.get("src_bucket", "user")
  
  @property
  def dest_bucket(self) -> str:
    return self.config.get("dest_bucket", "photo")