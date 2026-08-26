from common.retrieval import RetrievalModule

class RetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name, "fountain/retrieval/config/module")

  def _sample_global_triggers(self, output_trigger_name: str):
    ## 把每个 i2i 召回里抽取 global trigger 的方法抽出来放到这里，在召回模块里直接调用这个方法进行抽取
    return \
      self.flow \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "global_normal_trigger_list",
            "global_high_value_trigger_list",
            "global_normal_trigger_weight_list",
            "global_high_value_trigger_weight_list",
            "sample_trigger_num",
            "start_idx"
          ],
          export_common_attr = [
            {"name": "final_trigger_list", "as": output_trigger_name},
          ],
          function_name = "SampleTriggers",
          class_name = "ExploreLightFunctionSetV2"
        )