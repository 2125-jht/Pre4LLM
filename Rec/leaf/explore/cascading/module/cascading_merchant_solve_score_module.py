from cascading import CommonModule

class CascadingMerchantSolveScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)


  def process(self) -> None:
    self.flow \
      .if_("enable_explore_mc_merchant_vedio_predict == 1") \
      .enrich_attr_by_light_function(
        import_item_attr = [
          { "name": "merchant_tower_ctr", "as": "ctr_input" },
          { "name": "merchant_tower_cvr", "as": "cvr_input" },
          { "name": "merchant_tower_gmv", "as": "gmv_input" },
          { "name": "cascade_pctr", "as": "cascade_pctr" }
        ],
        export_item_attr = [
          { "name": "ctcvr_out", "as": "mc_ctcvr_score" },
          { "name": "ctcvr_gmv_out", "as": "mc_ctcvr_gmv_score" },
        ],
        function_name = "CalMcMerchantCtcvr",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .end_() \
      .if_("enable_explore_mc_merchant_living_predict == 1") \
      .enrich_attr_by_light_function(
        import_item_attr = [
          { "name": "merchant_elive_tower_ctr", "as": "ctr_input" },
          { "name": "merchant_elive_tower_cvr", "as": "cvr_input" },
          { "name": "merchant_elive_tower_gmv", "as": "gmv_input" },
          { "name": "cascade_pctr", "as": "cascade_pctr" }
        ],
        export_item_attr = [
          { "name": "ctcvr_out", "as": "mc_elive_ctcvr_score" },
          { "name": "ctcvr_gmv_out", "as": "mc_elive_ctcvr_gmv_score" },
        ],
        function_name = "CalMcMerchantCtcvr",
        class_name = "ExploreLightFunctionSetV2",
      ) \
      .end_()

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_attrs = [
          "is_merchant_cart", 
          "is_merchant_living", 
          "merchant_tower_ctr", 
          "merchant_tower_cvr", 
          "merchant_tower_gmv", 
          "mc_ctcvr_score", 
          "mc_ctcvr_gmv_score",
          "merchant_elive_tower_ctr", 
          "merchant_elive_tower_cvr", 
          "merchant_elive_tower_gmv", 
          "mc_elive_ctcvr_score", 
          "mc_elive_ctcvr_gmv_score"
        ],
        common_attrs = [
          "enable_explore_mc_merchant_vedio_predict",
          "enable_explore_mc_merchant_living_predict",
        ],
      )

