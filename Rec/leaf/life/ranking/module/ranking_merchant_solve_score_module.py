from ranking import CommonModule

class RankingMerchantSolveScoreModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def process(self) -> None:
        self.flow \
        .enrich_attr_by_light_function(
          skip = "{{return enable_explore_rank_merchant_vedio_predict == 0}}",
          import_item_attr = [
            { "name": "merchant_pcart_ctr", "as": "ctr_input" },
            { "name": "merchant_cart_cvr", "as": "cvr_input" },
            { "name": "merchant_cart_gmv_fen", "as": "gmv_input" },
            { "name": "pctr", "as": "rank_pctr_input" },
          ],
          export_item_attr = [
            { "name": "ctcvr_out", "as": "fr_ctcvr_score" },
            { "name": "ctcvr_gmv_out", "as": "fr_ctcvr_gmv_score" },
          ],
          function_name = "CalFrMerchantCartCtcvr",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .enrich_attr_by_light_function(
          skip = "{{return enable_explore_rank_merchant_living_predict == 0}}",
          import_item_attr = [
            { "name": "merchant_elive_ctr", "as": "ctr_input" },
            { "name": "merchant_elive_cvr", "as": "cvr_input" },
            { "name": "merchant_elive_price", "as": "gmv_input" },
            { "name": "pctr", "as": "rank_pctr_input" },
          ],
          export_item_attr = [
            { "name": "ctcvr_out", "as": "fr_elive_ctcvr_score" },
            { "name": "ctcvr_gmv_out", "as": "fr_elive_ctcvr_gmv_score" },
          ],
          function_name = "CalFrMerchantlivingCtcvr",
          class_name = "ExploreLightFunctionSetV2",
        ) \

    def post_process(self) -> None:
      self.flow \
        .log_debug_info(
          item_attrs = [
            "is_merchant_cart", 
            "is_merchant_living",
            "merchant_author_in_living",
            "s_eshop_first_live_id",
            "merchant_pcart_ctr", 
            "merchant_cart_cvr", 
            "merchant_cart_gmv_fen", 
            "fr_ctcvr_score", 
            "fr_ctcvr_gmv_score",
            "merchant_elive_ctr", 
            "merchant_elive_cvr", 
            "merchant_elive_price", 
            "fr_elive_ctcvr_score", 
            "fr_elive_ctcvr_gmv_score"
          ],
          common_attrs = [
            "enable_explore_rank_merchant_vedio_predict",
            "enable_explore_rank_merchant_living_predict",
          ],
        )
