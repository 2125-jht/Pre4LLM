from retrieval.retrieval_module import RetrievalModule

class SplashRelevanceMMUCategoryRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_relevance_mmu_category_content_tag(
        redis_server = "recoRelevancePhotoCategoryTag",
        key_source = "featureSourcePId",
        reason = self.reason,
        version = "{{reco_relevance_photo_mmu_content_tag_retrieval_version}}",
        total_limit = "{{fountain_photo_mmu_content_tag_retrieval_total_limit}}",
        skip = "{{fountain_retrieval_skip_relevance_photo_mmu_content_tag}}")