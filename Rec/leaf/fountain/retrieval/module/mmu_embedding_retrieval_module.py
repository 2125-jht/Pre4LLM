from retrieval.retrieval_module import RetrievalModule

class MmuEmbeddingRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_mmu_embedding_gas(
        kess_service = "grpc_mmu_visionSearchGas",
        timeout_ms = 100,
        reason = self.reason,
        item_type = 1,
        source_photo_attr = "commonRetrievalPhotos",
        total_limit = 400,
        skip = "{{skip_fountain_mmu_subdividion_embedding_retrieval}}") # 这里skip参数通过后缀区分请求类型，不能省略
