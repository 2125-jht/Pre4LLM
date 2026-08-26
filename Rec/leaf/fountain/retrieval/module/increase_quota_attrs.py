from retrieval import CommonModule

class IncreaseQuotaAttrsModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .enrich_attr_by_lua(
        import_common_attr = [
          "fountain_increase_quota_time_window"
        ],
        export_common_attr = ["increase_quota_status", "increase_quota_current_index", "increase_quota_window_len"],
        function_for_common = "increase_quota_status",
        lua_script_file = "fountain/retrieval/lua/module/increase_quota_attrs__increase_quota.lua")