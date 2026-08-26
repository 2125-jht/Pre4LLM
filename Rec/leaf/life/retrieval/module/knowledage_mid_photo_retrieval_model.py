from retrieval import RetrievalModule

class KnowledgeMidPhotoRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  @classmethod
  def is_retrieval(cls) -> bool:
    return True
    
  def process(self) -> None:
    self.flow \
      .explore_memory_data_enrich(
        data_key = "{{memory_data_key}}",
        data_type = "string_uint64_vector_map",
        save_data_ptr_to_attr = "memory_data_map_ptr",
      ) \
      .explore_retrieve_by_knowledge_mid_photo(
        reason = self.reason,
        user_info_ptr_attr = "user_info_ptr",
        memory_data_ptr_attr = "memory_data_map_ptr",
        play_time_threshold = "{{play_time_threshold}}",
        duration_time_threshold = "{{duration_time_threshold}}",
        trigger_num = "{{trigger_num}}",
        trigger_key_prefix = "{{trigger_key_prefix}}",
      )
