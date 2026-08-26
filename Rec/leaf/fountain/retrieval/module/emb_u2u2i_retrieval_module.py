from retrieval.retrieval_module import RetrievalModule

class EmbU2u2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .retrieve_by_ann_embedding(
        kess_service = "{{fountain_reco_emb_u2u2i_v2_stage1_kess_name}}",
        timeout_ms = 50,
        reason = 1, # 级联召回第一步统一reason统一为1
        space = "ip",
        items_from_attr = ["_USER_ID_"],
        bound_type = {
          "total_limit": "{{fountain_reco_emb_u2u2i_v2_retr_stage1_total_limit}}",
        },
        algo_type = {
          "faiss": {}
        },
        src_bucket = "{{fountain_reco_emb_u2u2i_v2_retr_stage1_src_bucket}}",
        dest_bucket = "{{fountain_reco_emb_u2u2i_v2_retr_stage1_dest_bucket}}",
        dest_bucket_item_type = 1,
        save_result_to_common_attr = "fountain_reco_emb_u2u_v2_users",
        skip = "{{skip_fountain_reco_emb_stage1_u2u2i_v2_retr}}"
      ) \
      .retrieve_by_redis(
        cluster_name = "recoOfflineGlobalNegativeSample",
        retrieve_num = "{{fountain_reco_emb_u2u2i_v2_retr_stage2_total_limit}}",
        timeout_ms = 50,
        key_from_attr = "fountain_reco_emb_u2u_v2_users",
        key_prefix = "fountain_data_mining_u2i_retr_",
        retrieve_num_per_key = "{{fountain_reco_emb_u2u2i_v2_retr_stage2_per_key}}",
        item_separator = ",",
        reason = self.reason,
        skip="{{skip_fountain_reco_emb_u2u2i_v2_retr}}",
      )
