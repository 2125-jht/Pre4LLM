from common.retrieval import RetrievalModule

class AuditTopKPassRetrievalModule(RetrievalModule):
  def __init__(self, name: str, config_dir: str) -> None:
    super().__init__(name, config_dir)

  def process(self):
    self.flow \
      .if_("enable_retr_filter_downgrade == 1") \
        .fake_retrieve(
          num = 1000,
          save_result_to_common_attr = "candidate_trigger_list"
        ) \
        .shuffle_list_attr(
          common_attr = "candidate_trigger_list"
        ) \
        .pack_common_attr(
          input_common_attrs = ["candidate_trigger_list"],
          output_common_attr = "trigger_list",
          limit_num = "{{trigger_num}}",
        ) \
        .fetch_kgnn_neighbors(
          id_from_common_attr = "trigger_list",
          save_neighbors_to = 'retr_result',
          kess_service = "grpc_kgnn_retrieval_index_for_system_safe-I2I",
          relation_name = "I2I",
          sample_num = 100,
          shard_num = 2,
          timeout_ms = 50,
          sample_type = "random",
          padding_type = "no_padding",
        ) \
        .shuffle_list_attr(
          common_attr = "retr_result"
        ) \
        .retrieve_by_common_attr(
          attr = "retr_result", 
          reason = self.reason
        ) \
        .limit(
          size = "{{retrieve_num}}"
        ) \
      .end_()