from retrieval.retrieval_module import RetrievalModule

class AnnI2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("retrieval_skip == 0", to_be_delete = "date=2024-05-29;committer=huzengyi") \
        .if_("use_effect_view_trigger == 0", to_be_delete = "date=2024-05-29;committer=liyunhao") \
          .copy_attr(
            attrs=[{
              "from_common": "featureFountainProfileEffViewPidList",
              "to_common": "triggers"
            }]) \
        .else_() \
          .copy_attr(
            attrs=[{
              "from_common": "featureFountainProfileLongViewPidListSub",
              "to_common": "triggers"
            }]) \
        .end_() \
        .retrieve_by_ann_embedding(
          kess_service = "{{kess_service}}",
          space = self.space,
          timeout_ms = 50,
          reason = self.reason,
          shard_num = 1,
          items_from_attr = ["triggers"],
          bound_type = {
            "top_k": "{{retr_topk_num}}",
          },
          algo_type = {
            "scann": {},
          },
          src_bucket = self.src_bucket,
          dest_bucket = self.dest_bucket,
          dest_bucket_item_type = 0,
        ) \
      .end_()

  @property
  def space(self) -> str:
    return self.config.get("space", "cosine")

  @property
  def src_bucket(self) -> str:
    return self.config.get("src_bucket", "photo")
  
  @property
  def dest_bucket(self) -> str:
    return self.config.get("dest_bucket", "photo")