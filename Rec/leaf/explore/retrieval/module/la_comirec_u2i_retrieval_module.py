from retrieval.retrieval_module import RetrievalModule

#错误的实例，代码误直接copy使用，通用行不高
#TODO: 下周五2023.1.13修改升级一下使用retrieval leaf做更高通用行
class LaComirecU2iRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("_USER_ID_ <= 0") \
        .return_() \
      .end_() \
      .enrich_attr_by_light_function(
        import_common_attr = [
          {"name": "_USER_ID_", "as": "user_id"},
          "interest_num",
          "shfit_num"
        ],
        export_common_attr = [
          "encoded_uid_lists",
          "uidTab"
        ],
        function_name = "MultiInterestShfitUidList",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .get_remote_embedding_lite(
        kess_service="{{embedding_service}}",
        timeout_ms=10,
        query_source_type="common_attr",
        input_attr_name="uidTab",
        output_attr_name="user_embedding_list",
        id_converter={"type_name": "kuibaEmbeddingIdConverter"},
        slot=self.slot,
        size=1024,
      ) \
      .retrieve_by_ann_embedding(
        kess_service = "{{ann_service}}",
        space = self.space,
        timeout_ms = 50,
        reason = self.reason,
        items_from_attr = ["encoded_uid_lists"],
        embeddings_from_attr = ["user_embedding_list"],
        bound_type = {
          "total_limit": "{{retrieve_num}}"
        },
        algo_type = {
          "scann": {},
        },
        src_bucket = "{{src_bucket}}",
        dest_bucket = "{{dest_bucket}}",
      ) \
      .filter_by_common_attr(
        common_attr = ["browse_screen__pid_list"]
      )

  @property
  def slot(self) -> int:
    assert "slot" in self.config
    return self.config["slot"]

  @property
  def space(self) -> str:
    assert "space" in self.config
    return self.config["space"]