from retrieval.retrieval_module import RetrievalModule


class ExploreStRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
      super().__init__(name)

  def process(self) -> None:
    self.flow \
      .switch_("retr_mode") \
        .case_(1, to_be_delete = "date=2024-05-29;committer=caozhong") \
          .get_remote_embedding_lite(
            kess_service="{{embedding_service_name}}",
            timeout_ms=10,
            query_source_type="user_id",
            output_attr_name="user_embedding_concat",
            id_converter={"type_name": "mioEmbeddingIdConverter"},
            slot=4016,
            size=self.emb_size,
          ) \
          .gen_common_attr_by_lua(
            attr_map={
                "uid_concat": "{_USER_ID_, _USER_ID_+1, _USER_ID_+2 ,_USER_ID_+3}",
            }
          ) \
          .retrieve_by_ann_embedding(
            reason=self.reason,
            kess_service="{{ann_service_name}}",
            space="ip",
            timeout_ms=50,
            items_from_attr=["uid_concat"],
            embeddings_from_attr=["user_embedding_concat"],
            bound_type={
                "total_limit": "{{retrieve_num}}",
            },
            algo_type={
                "scann": {},
            },
            src_bucket="user",
            dest_bucket="{{ann_dest_bucket}}",
          ) \
        .case_(2, to_be_delete = "date=2024-05-29;committer=caozhong") \
          .copy_user_meta_info(
            save_request_type_to_attr="request_type",
          ) \
          .delegate_retrieve(
            reason=self.reason,
            kess_service="{{ann_service_name}}",
            timeout_ms=50,
            request_type="{{request_type}}",
            request_num="{{retrieve_num}}",
            send_common_attrs_in_request=False,
            send_common_attrs=self.send_common_attrs,
            reset_item_type=0
          ) \
        .default_() \
          .retrieve_by_ann_embedding(
            reason=self.reason,
            kess_service="{{ann_service_name}}",
            space="ip",
            timeout_ms=50,
            items_from_attr=["_USER_ID_"],
            bound_type={
                "total_limit": "{{retrieve_num}}",
            },
            algo_type={
                "scann": {},
            },
            src_data_type="user",
            src_bucket="user",
            dest_bucket="{{ann_dest_bucket}}"
          ) \
     .end_()

  @property
  def send_common_attrs(self) -> list:
    assert "send_common_attrs" in self.config
    return self.config.get("send_common_attrs")

  @property
  def emb_size(self) -> int:
    assert "emb_size" in self.config
    return self.config.get("emb_size")
