from retrieval.retrieval_module import RetrievalModule

class EyeshotCertAuthorRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
      self.flow \
        .get_remote_embedding_lite(
          kess_service="grpc_exploreLookAlikeUserEmbServer",
          shard_num=1,
          timeout_ms=20,
          id_converter={
            "type_name": "kuibaEmbeddingIdConverter"
          },
          size=128,
          output_attr_name="la_user_embedding_list",
          query_source_type="user_id",
          client_side_shard=True
        ) \
        .retrieve_by_ann_embedding(
          reason=self.reason,
          timeout_ms=150,
          space="cosine",
          kess_service="{{fountain_eyeshot_cert_author_retr_service}}",
          items_from_attr=["_USER_ID_"],
          embeddings_from_attr=["la_user_embedding_list"],
          bound_type={
            "total_limit": "{{fountain_eyeshot_cert_author_retrieval_num}}",
          },
          algo_type={
            "scann": {},
          },
          src_bucket="photo",
          dest_bucket="{{fountain_eyeshot_cert_author_retrieval_dest}}",
          dest_bucket_item_type=0,
        ) \
      