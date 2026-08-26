from cascading import CommonModule

class CascadingHighQualityMerchantPhotoTypeAttrModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_high_quality_merchant_photo_calc_type == 1") \
      .enrich_attr_by_light_function(
        import_item_attr = [
          {"name": "content_safety_level_with_namespace__level_hot_online", "as": "impress_audit"},
          "empirical_ctr",
          "empirical_ltr",
          "empirical_wtr",
          "empirical_ftr",
          "empirical_cmtr",
          "empirical_watch_time"
        ],
        import_common_attr = [
          {"name": "high_quality_merchant_ctr_threshold", "as": "ctr_threshold"},
          {"name": "high_quality_merchant_ltr_threshold", "as": "ltr_threshold"}, 
          {"name": "high_quality_merchant_wtr_threshold", "as": "wtr_threshold"},
          {"name": "high_quality_merchant_ftr_threshold", "as": "ftr_threshold"},
          {"name": "high_quality_merchant_cmtr_threshold", "as": "cmtr_threshold"},
          {"name": "high_quality_merchant_wtd_threshold", "as": "wtd_threshold"}
        ],
        export_item_attr = [
          "is_high_quality_merchant_cart"
        ],
        function_name = "IsMerchantHighQuality",
        class_name = "ExploreLightFunctionSetV2",
        target_item = {
          "is_merchant_cart" : 1
        }
      ) \
    .end_()
