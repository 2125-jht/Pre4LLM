from cascading import CommonModule

class CascadingCalcXhsInstallAttrModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_request_explore_xhs_install_item == 1 and is_la_correct_user == 1") \
        .get_item_attr_by_remote_index(
          kess_service = "{{explore_xhs_hive_index_kess_service}}",
          timeout_ms = 100,
          partition_size = 1000,
          attrs = [
            "xhs_install_find_click_value",
            "xhs_install_find_outflow_click_value",
          ],
        ) \
      .end_()
