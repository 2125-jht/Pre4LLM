from cascading import CommonModule

class CascadingMerchantPhotoTypeAttrModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    MERCHANT_GOODS_PHOTO_STORE_EXPORT_ATTRS = [
      "price_info"
    ]

    self.flow \
      .if_("disable_merchant_explore_all_photo_optimize == 0 and enable_merchant_photo_calc_type == 1") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            {"name": "author__id", "as": "author__id"},
            {"name": "merchant_item_info__item_id_list", "as": "cart_itemlist"},
            {"name": "merchant_photo_cart_relation", "as": "cart_relation"},
          ],
          import_common_attr = [
            "merchant_live_authors_set__memory_data",
          ],
          export_item_attr = [
            "is_merchant_cart",
            "is_merchant_living",
            "merchant_author_in_living",
            "merchant_item_id",
            "sign_merchant_item_id"
          ],
          function_name = "MerchantGetAuthorInLiving",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .get_merchant_item_attr_by_distributed_index(
          photo_store_kconf_key = "reco.distributedIndex.exploreMerchantPhotoStoreConfig",
          use_dynamic_photo_store = True,
          photo_store_rpc_req_cache_rate = 0,
          attrs = MERCHANT_GOODS_PHOTO_STORE_EXPORT_ATTRS,
          perf_log = "merchant_photo_rank",
          item_id_attr = "sign_merchant_item_id",
          target_item = {"is_merchant_cart": 1},
        ) \
      .end_()

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_attrs = [
          "merchant_item_info__item_id_list", 
          "merchant_photo_cart_relation",
          "author__id",
          "merchant_author_in_living",
          "is_merchant_cart",
          "is_merchant_living",
          "merchant_item_id"
        ],
        common_attrs = [
          "merchant_live_authors_set__memory_data",
          "uBuyerEffectiveType",
          "merchant_buyer_type"
        ],
      )