from retrieval import CommonModule

class PageSizeDedupModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    if self.enable_pagesize_dedup:
      self.flow \
        .enrich_attr_by_lua(
          import_item_attr = [
            "item_id",
            "dup_cluster_id",
            "sim_remove_dup_id",
            "pic_and_selfdup_id",
          ],
          export_item_attr = [
            "dup_cluster_id_adjust",
            "sim_remove_dup_id_adjust",
            "pic_and_selfdup_id_adjust",
          ],
          function_for_item = "content_id_adjust_func",
          lua_script_file = "fountain/retrieval/lua/module/pagesize_dedup__content_id_adjust.lua",) \
        .deduplicate(
          skip = "{{fountain_skip_dup_cluster_id_deduplicate_in_pagesize}}",
          on_item_attr = "dup_cluster_id_adjust",
          save_dup_count_to = "dup_cluster_id_duplicate_count",
        ) \
        .deduplicate(
          skip = "{{fountain_skip_sim_remove_dup_id_deduplicate_in_pagesize}}",
          on_item_attr = "sim_remove_dup_id_adjust",
          save_dup_count_to = "sim_remove_dup_id_duplicate_count",
        ) \
        .deduplicate(
          skip = "{{fountain_skip_pic_and_selfdup_id_deduplicate_in_pagesize}}",
          on_item_attr = "pic_and_selfdup_id_adjust",
          save_dup_count_to = "pic_and_selfdup_id_duplicate_count",
        )
  
  @property
  def enable_pagesize_dedup(self) -> bool:
    return self.config.get("enable_pagesize_dedup", False)