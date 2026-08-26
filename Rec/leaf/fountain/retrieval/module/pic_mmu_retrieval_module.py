from retrieval.retrieval_module import RetrievalModule

class PicMMURetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .switch_("fountain_pic_mmu_retr_mode") \
        .case_(1, to_be_delete = "date=2024-05-29;committer=denghong") \
          .delegate_retrieve(
            kess_service="{{fountain_pic_mmu_retr_service}}",
            timeout_ms=50,
            reason=self.reason,
            request_type="default",
            request_num="{{fountain_pic_mmu_retr_num}}",
            send_common_attrs=[
                {"name": "userInfo", "as": "user"},
              ],
            send_common_attrs_in_request=False
          ) \
        .default_() \
          .retrieve_by_ann_embedding(
            reason=self.reason,
            timeout_ms=50,
            space="cosine",
            kess_service="{{fountain_pic_mmu_retr_service}}",
            items_from_attr=["featureFountainProfileLongViewPidListSub", "colossusRetrievalTrigger"],
            attr_single_limit=30,
            bound_type={
              "total_limit": "{{fountain_pic_mmu_retr_num}}",
            },
            algo_type={
              "scann": {},
            },
            src_bucket="photo",
            dest_bucket="{{fountain_pic_mmu_retr_dest}}",
            dest_bucket_item_type=0,
          ) \
      .end_()