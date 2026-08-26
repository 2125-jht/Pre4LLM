from ranking import CommonModule

class PxtrIndexModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def get_index(self, input_item_attr, output_item_attr):
    self.flow \
    .sort(
      score_from_attr = input_item_attr
    ) \
    .enrich_attr_by_light_function(
      export_item_attr = [
        {"name": "item_attr_index", "as": output_item_attr}
      ],
      function_name = "SaveItemSeqAddOne",
      class_name = "ExploreLightFunctionSetV2",
    )

    return self

  def process(self) -> None:
    self.flow \
    .if_(self.config.get("ab_enable", "") + " == 1")
    for input_item_attr, output_item_attr in self.config.get("index_dict", {}).items():
      self.get_index(input_item_attr, output_item_attr)
    self.flow \
    .end_()