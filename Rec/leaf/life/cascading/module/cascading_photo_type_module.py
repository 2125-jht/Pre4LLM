from cascading import CommonModule

class CascadingPhotoTypeModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("enable_explore_calc_photo_type == 1") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            "eyeshot_source"
          ],
          export_item_attr = [
            "is_personified_author",
            "is_blacklist_author",
            "is_hot_content"
          ],
          function_name = "CalcPhotoType",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_() \
      .if_("enable_sim_hetu_cluster_id_trans == 1") \
        .explore_trans_sim_cluster_id_enricher(
          cluster_id_input_attr = "hetu_sim_cluster_id",
          cluster_id862_attr = "hetu_sim_cluster_id862" # mmu hetu sim cluster id 862版
        ) \
      .end_() \
      .if_("life_enable_adjust_marketing_compensation_photo == 1") \
        .gen_is_marketing_compensation_photo() \
      .end_() \
      .if_("life_enable_gen_is_olympic_photo == 1") \
        .gen_is_olympic_photo() \
      .end_() \
      .if_("enable_life_gen_is_low_cost_photo == 1") \
        .gen_is_low_cost_photo() \
      .end_() \
      .if_("enable_life_gen_minority_photo == 1") \
        .gen_is_minority_photo() \
      .end_() \
      .if_("enable_get_danlie_tired_score == 1") \
        .gen_tired_switch_behave_ids() \
        .get_remote_embedding_lite(
              kess_service = "grpc_hotMcEmbed",
              shard_num = 8,
              id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
              input_attr_name = "photo_id",
              output_attr_name = "mmu_hetu_embedding",
              query_source_type = "item_attr",
              size = 128,
              client_side_shard=True
                )\
        .get_remote_embedding_lite(
              kess_service = "grpc_hotMcEmbed",
              shard_num = 8,
              id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
              item_list_from_attr="colossus_photo_id_list",
              output_attr_name = "neg_mmu_hetu_embedding",
              size = 128,
              client_side_shard=True
            ) \
        .filter_by_attr(
                item_list_from_attr="colossus_photo_id_list",
                attr_name="neg_mmu_hetu_embedding",
                remove_if_attr_missing=True,
            ) \
        .oversea_calc_negative_feedback(
                item_embedding_attr="mmu_hetu_embedding",
                neg_item_embedding_attr="neg_mmu_hetu_embedding",
                export_item_attr="raw_similarity_score",
                negative_items_from_attr="colossus_photo_id_list",
                allow_empty_embedding = True,
            ) \
        .set_attr_value(
          no_overwrite=True,
          item_attrs=[
            {
              "name": "life_danlie_depress_pow_func_constant",
              "type": "double",
              "value": 2.0
            },
          ]) \
        .item_attr_operation(
              item_attr_a="life_danlie_depress_pow_func_constant" ,
              item_attr_b="raw_similarity_score",
              operator="pow",
              output_attr="origin_danlie_depress_score",
          ) \
        .normalize_attr(
            input_attr="origin_danlie_depress_score",  # 这里改了
            output_attr="norm_danlie_depress_score",
            mode="min_max_scale",
            default_val=0.0,
        ) \
        .switch_("enable_life_danlie_depress_score") \
        .case_(1) \
          .copy_attr( attrs=[{ "from_item": "norm_danlie_depress_score",
            "to_item": "danlie_depress_score"}]) \
        .case_(2) \
          .if_("is_siwtched_from_danlie == 1") \
            .copy_attr( attrs=[{ "from_item": "norm_danlie_depress_score",
              "to_item": "danlie_depress_score"}]) \
          .end_() \
        .case_(3) \
          .if_("is_tired_of_danlie == 1 and is_siwtched_from_danlie == 1") \
            .copy_attr( attrs=[{ "from_item": "norm_danlie_depress_score",
              "to_item": "danlie_depress_score"}]) \
          .end_() \
        .default_() \
          .do_nothing() \
        .end_() \
        .log_debug_info(
          common_attrs=["last_danlie_photo_id_list",
                        "is_tired_of_danlie",
            "is_siwtched_from_danlie",
            ],
          item_attrs = [
            "origin_danlie_depress_score",
            "norm_danlie_depress_score"
          ],
          for_debug_request_only = True,
          respect_sample_loggging = True,
        ) \
        .end_() \
        
