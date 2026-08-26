from retrieval.retrieval_module import RetrievalModule

class FountainLongViewLikeI2IRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .switch_("request_type") \
        .case_("fountain_fast_v1_life") \
          .enrich_attr_by_lua(
            import_common_attr = [
              "featureUserHateList",
              "featureFountainProfileLongViewPidList"
            ],
            export_common_attr = [
              "featureFountainProfileLongViewPidListSub"
            ],
            function_for_common = "filter_trigger_list",
            lua_script_file = "life/retrieval/lua/module/cl_i2i_retr__trigger_generator.lua"
          ) \
          .retrieve_by_ann_embedding(
            kess_service = "{{fountain_long_view_like_i2i_retr_kess_service}}",
            space = "cosine",
            timeout_ms = 50,
            reason = self.reason,
            shard_num = 1,
            items_from_attr = ["featureFountainProfileLongViewPidListSub"],
            bound_type = {
              "top_k": "{{fountain_long_view_like_i2i_retr_topk_num}}",
            },
            algo_type = {
              "scann": {},
            },
            src_bucket = "photo",
            dest_bucket = "photo",
            dest_bucket_item_type = 0,
          ) \
      .end_()