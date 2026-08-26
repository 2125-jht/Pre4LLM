from retrieval import CommonModule

class RetrievalPerfModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .if_("_IS_PERF_SAMPLING_REQUEST_ == 1") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            "upload_type",
            "duration_ms",
          ],
          export_item_attr = [
            "is_picture",
          ],
          function_name = "IsPicture",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .enrich_attr_by_light_function(
          import_common_attr=[
            "support_author_memory_data"
          ],
          import_item_attr=[
            {"name":"author__id", "as": "author_id"}
          ],
          export_item_attr=[
            "is_support_author_picture"
          ],
          function_name="IsSupportAuthorPic",
          class_name="ExploreLightFunctionSetV2",
          target_item={ "is_picture": 1 }
        ) \
      .end_()