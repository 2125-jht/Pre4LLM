from cascading_v2 import CommonModule

class CascadingShortWindowIndexModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("fetch_short_window_photo_info == 1") \
        .get_item_attr_by_distributed_common_index(
          photo_store_kconf_key = "reco.distributedIndex.hotShortWindowInfoCommonIndex",
          use_dynamic_photo_store = True,
          attrs = [
            "rc3h",
            "pc3h",
          ]
        ) \
      .end_()
