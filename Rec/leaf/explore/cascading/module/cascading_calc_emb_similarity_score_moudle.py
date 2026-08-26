from cascading import CommonModule

class CascadingCalcEmbSimilarityScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("explore_enable_pack_unaudit_item == 1") \
        .pack_item_attr(
          item_source = {
            "reco_results": True
          },
          mappings = [{
            "from_item_attr": "photo_id",
            "to_common_attr": "embedding_unaudit_pids",
            "aggregator": "concat"
          }],
           select_item = {
            "join": "or",
            "filters": [{
              "attr_name": "audit_hot_cover_level",
              "compare_to": 0,
              "select_if": "<=",
              "select_if_attr_missing": True
            },
            {
              "attr_name": "audit_b_second_tag",
              "compare_to": 0,
              "select_if": "<=",
              "select_if_attr_missing": True
            },
            {
              "attr_name": "audit_hot_high_tag_level",
              "compare_to": 0, 
              "select_if": "<=",
              "select_if_attr_missing": True
            }]
          }
        ) \
      .end_() \
      .if_("explore_enable_pack_badaudit_item == 1") \
        .pack_common_attr(
          input_common_attrs = [
            "retrieval_only_bad_cover_input_item_key_list",
            "retrieval_only_bad_sense_input_item_key_list",
            "retrieval_only_bad_hot_audit_input_item_key_list",
            "embedding_unaudit_pids"
          ],
          output_common_attr = "audit_all_pids",
          deduplicate = True
        ) \
      .end_() \
      .if_("explore_enable_get_bad_audit_item_emb == 1") \
        .get_remote_embedding_lite(
          kess_service = "grpc_hotRerankMmuEmbServerV3",
          shard_num = 4,
          id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
          input_attr_name = "audit_all_pids",
          output_attr_name = "audit_all_pids_embedding",
          query_source_type = "common_attr",
          size = 64,
          client_side_shard=True
        ) \
      .end_() \
      .if_("explore_mc_enable_cal_bad_cover_similarity_score == 1") \
        .explore_custom_embedding_score_enricher(
          user_info_ptr_attr = "user_info_ptr",
          embedding_list_attr = "audit_all_pids_embedding",
          source_pids_list_attr = "audit_all_pids",
          target_pids_list_attr = "retrieval_only_bad_cover_input_item_key_list",
          calc_type = "list_similarity",
          enable_fix_low_hit_rate = True,
          dim_size = 64,
          export_item_attr = "explore_mc_bad_cover_similarity_score",
          select_item = {
            "attr_name": "audit_hot_cover_level",
            "compare_to": 0,
            "select_if": "<=",
            "select_if_attr_missing": True
          }
        ) \
      .end_() \
      .if_("explore_mc_enable_cal_bad_sense_similarity_score == 1") \
        .explore_custom_embedding_score_enricher(
          user_info_ptr_attr = "user_info_ptr",
          embedding_list_attr = "audit_all_pids_embedding",
          source_pids_list_attr = "audit_all_pids",
          target_pids_list_attr = "retrieval_only_bad_sense_input_item_key_list",
          calc_type = "list_similarity",
          dim_size = 64,
          export_item_attr = "explore_mc_bad_sense_similarity_score",
          enable_fix_low_hit_rate = True,
          select_item = {
            "attr_name": "audit_b_second_tag",
            "compare_to": 0,
            "select_if": "<=",
            "select_if_attr_missing": True
          }
        ) \
      .end_() \
      .if_("explore_mc_enable_cal_bad_hot_audut_similarity_score == 1") \
        .explore_custom_embedding_score_enricher(
          user_info_ptr_attr = "user_info_ptr",
          embedding_list_attr = "audit_all_pids_embedding",
          source_pids_list_attr = "audit_all_pids",
          enable_fix_low_hit_rate = True,
          target_pids_list_attr = "retrieval_only_bad_hot_audit_input_item_key_list",
          calc_type = "list_similarity",
          dim_size = 64,
          export_item_attr = "explore_mc_bad_hot_audit_similarity_score",
          select_item = {
            "attr_name": "audit_hot_high_tag_level",
            "compare_to": 0,
            "select_if": "<=",
            "select_if_attr_missing": True
          }
        ) \
      .end_() \
      .if_("explore_mc_enable_cal_marketing_compensation_positive_trigger_similarity_score == 1") \
        .explore_custom_embedding_score_enricher(
          enable_fix_low_hit_rate = "{{explore_mc_enable_marketing_fix_mmu_embedding_low_hit_rate}}",
          user_info_ptr_attr = "user_info_ptr",
          embedding_list_attr = "explore_mc_mmu_embeddings",
          source_pids_list_attr = "explore_mc_mmu_embedding_marketing_source_pids",
          target_pids_list_attr = "explore_marketing_compensation_positive_trigger",
          calc_type = "list_similarity",
          dim_size = 64,
          export_item_attr = "explore_marketing_compensation_positive_trigger_similarity_score",
          target_item = {
            "is_marketing_compensation_photo": 1
          }
        ) \
      .end_()
