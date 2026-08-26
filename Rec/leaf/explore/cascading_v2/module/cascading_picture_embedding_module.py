from cascading_v2 import CommonModule

class CascadingPictureEmbeddingModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_mc_calc_pic_diversity == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "user_info_ptr",
            {"name": "uStandardRealShowPicAllIdList", "as": "realshow_pic_ids"},
            {"name": "explore_pic_mc_diversity_history_num", "as": "history_num"},
            {"name": "explore_pic_mc_diversity_time_gap_min", "as": "time_gap_min"},
            {"name": "explore_pic_mc_diversity_enable_action", "as": "enable_action"},
            {"name": "explore_pic_mc_calc_negative_history_pic_ids_by_ddp", "as": "calc_by_ddp"},
            {"name": "uStandardExploreRealshowPhotoIdList", "as": "explore_realshow_ids"},
            {"name": "uStandardExploreRealshowTimestampList", "as": "explore_realshow_timestamps"},
            {"name": "uStandardExploreRealshowLabelList", "as": "explore_realshow_labels"},
          ],
          export_common_attr = [
            "history_pic_ids",
            "embedding_source_pic_ids",
          ],
          function_name = "GetEmbeddingSourcePicIds",
          class_name = "ExploreLightFunctionSetV2",
          target_item = {"is_picture": 1}
        ) \
        .get_remote_embedding_lite(
          kess_service = "{{explore_pic_mc_diversity_embedding_service_name}}",
          shard_num = 4,
          id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
          input_attr_name = "embedding_source_pic_ids",
          output_attr_name = "pic_embeddings",
          query_source_type = "common_attr",
          size = 64,
          client_side_shard = True
        ) \
      .end_()
