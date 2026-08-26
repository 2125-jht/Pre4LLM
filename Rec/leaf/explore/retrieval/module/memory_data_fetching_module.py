from retrieval import CommonModule

class MemoryDataFetchingModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .explore_memory_data_enrich(
        data_key = "explore_mc_dynamic_xtr_debias_map",
        data_type = "string_double_vector_map",
        save_data_ptr_to_attr = "explore_mc_hourly_xtr_debias_map_ptr",
      ) \
      .explore_memory_data_enrich(
        data_key = "explore_fullrank_dynamic_xtr_debias_map",
        data_type = "string_double_vector_map",
        save_data_ptr_to_attr = "explore_fullrank_hourly_xtr_debias_map_ptr",
      ) \
      .explore_memory_data_enrich(
        data_key="explore_support_author_tgi_score",
        data_type="uint64_uint64_double_map_map",
        save_data_ptr_to_attr="support_author_memory_data"
      ) \
      .explore_memory_data_enrich(
        data_key="prerank_duration_debias_bucket",
        data_type="string_double_vector_map",
        save_data_ptr_to_attr="prerank_duration_debias_bucket"
      ) \
      .explore_memory_data_enrich(
        data_key = "xhs_hetu_tags",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "xhs_hetu_memorydata_set",
      ) \
      .explore_memory_data_enrich(
        data_key = "livestream_merchant_author",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "merchant_author_list_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "merchant_live_authors_set",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "merchant_live_authors_set__memory_data",
      ) \
      .explore_memory_data_enrich(
        data_key = "explore_hetu_gender_debias",
        data_type = "string_double_vector_map",
        save_data_ptr_to_attr = "hetu_gender_to_norm_ctr_map_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "explore_personifed_author_pctr_ctr",
        data_type = "string_double_map",
        save_data_ptr_to_attr = "explore_personifed_author_boost_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "explore_pic_xtr_cluster_emp_map",
        data_type = "string_double_vector_map",
        save_data_ptr_to_attr = "explore_pic_xtr_cluster_emp_map_ptr",
      ) \
      .explore_memory_data_enrich(
        data_key = "explore_pic_xtr_emp_debias_map",
        data_type = "string_double_vector_map",
        save_data_ptr_to_attr = "explore_pic_xtr_emp_debias_map_ptr",
      ) \
      .explore_memory_data_enrich(
        data_key = "explore_pic_fr_pxtr_pcts",
        data_type = "string_double_map",
        save_data_ptr_to_attr = "explore_pic_fr_pxtr_pcts_ptr",
      ) \
      .explore_memory_data_enrich(
        data_key = "explore_pic_da_user_pref_score_data",
        data_type = "string_double_map",
        save_data_ptr_to_attr = "pic_da_user_pref_ptr",
      ) \
      .explore_memory_data_enrich(
        data_key = "explore_pic_fr_rel_score_pct_map",
        data_type = "string_double_vector_map",
        save_data_ptr_to_attr = "pic_fr_rel_score_pct_map",
      ) \
      .explore_memory_data_enrich(
        data_key = "explore_pic_rerank_pid_realshow_data_map",
        data_type = "string_double_vector_map",
        save_data_ptr_to_attr = "pic_rerank_pid_realshow_data_map",
      ) \
      .explore_memory_data_enrich(
        data_key = "explore_pic_xtr_bucket_calib_param_map",
        data_type = "string_double_vector_map",
        save_data_ptr_to_attr = "pic_bucket_calib_param_map"
      ) \
      .explore_memory_data_enrich(
        data_key = "hack_author_uid_list",
        data_type = "uint64_uint64_map",
        save_data_ptr_to_attr = "hack_author_uid_map_ptr"
      ) \
      .if_("explore_enable_fetch_rank_neg_photo == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rank_neg_photo_key_prefix", "as": "string_a"},
            {"name": "_DEVICE_ID_", "as": "string_b"}
          ],
          export_common_attr = [
            {"name": "final_string", "as": "explore_rank_neg_photo_redis_key"}
          ],
          function_name = "ConcatString",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .get_common_attr_from_redis( #上一刷精排结果过滤
          cluster_name = "recoExploreNegPhoto",
          redis_params = [
            {
              "redis_key": "{{explore_rank_neg_photo_redis_key}}",
              "output_attr_name": "rank_neg_photo_id_list_str"
            }
          ]
        ) \
      .end_() \
      .if_("explore_enable_fetch_rerank_neg_photo == 1", to_be_delete = "date=2024-05-29;committer=liuhao07") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rerank_neg_photo_key_prefix", "as": "string_a"},
            {"name": "_DEVICE_ID_", "as": "string_b"}
          ],
          export_common_attr = [
            {"name": "final_string", "as": "explore_rerank_neg_photo_redis_key"}
          ],
          function_name = "ConcatString",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .get_common_attr_from_redis( #上一刷精排结果过滤
          cluster_name = "recoExploreNegPhoto",
          redis_params = [
            {
              "redis_key": "{{explore_rerank_neg_photo_redis_key}}",
              "output_attr_name": "rerank_neg_photo_id_list_str"
            }
          ]
        ) \
      .end_() \
      .if_("explore_enable_fetch_explore_rank_pos_photo == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "explore_rank_pos_photo_key_prefix", "as": "string_a"},
            {"name": "_DEVICE_ID_", "as": "string_b"}
          ],
          export_common_attr = [
            {"name": "final_string", "as": "explore_rank_pos_photo_redis_key"}
          ],
          function_name = "ConcatString",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .get_common_attr_from_redis(
          cluster_name = "recoExploreNegPhoto",
          redis_params = [
            {
              "redis_key": "{{explore_rank_pos_photo_redis_key}}",
              "output_attr_name": "explore_rank_pos_photo_id_list_str"
            }
          ]
        ) \
        .split_string(
          input_common_attr = "explore_rank_pos_photo_id_list_str",
          output_common_attr = "explore_rank_pos_photo_id_retrieval_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
      .end_() \
      .if_("explore_enable_fetch_explore_rerank_pos_photo == 1") \
        .gen_common_attr_by_lua(
          attr_map={
            "explore_rerank_pos_photo_redis_key": "explore_rerank_pos_photo_key_prefix .. tostring(_USER_ID_)"
          }
        ) \
        .get_common_attr_from_redis(
          cluster_name = "recoExploreNegPhoto",
          redis_params = [
            {
              "redis_key": "{{explore_rerank_pos_photo_redis_key}}",
              "output_attr_name": "explore_rerank_pos_photo_id_list_str"
            }
          ]
        ) \
        .split_string(
          input_common_attr = "explore_rerank_pos_photo_id_list_str",
          output_common_attr = "explore_rerank_pos_photo_id_retrieval_list",
          delimiters = ",",
          trim_spaces = True,
          skip_empty_tokens = True,
          parse_to_int = True
        ) \
      .end_() \
      .explore_memory_data_enrich(
        data_key = "prior_author_5w",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "prior_author_5w_set_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "unbias_pid_5w",
        data_type = "string_uint64_vector_map",
        save_data_ptr_to_attr = "unbias_pid_5w_set_ptr",
      ) \
      .explore_memory_data_enrich(
        data_key = "high_photo_count_author_map",
        data_type = "uint64_uint64_map",
        save_data_ptr_to_attr = "high_photo_count_author_map_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "douyin_10w_author",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "douyin_10w_author_set_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "douyin_100w_author",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "douyin_100w_author_set_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "explore_low_pass_rate_photo_set",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "explore_low_pass_rate_photo_set"
      ) \
      .explore_memory_data_enrich(
        data_key = "xtr_fractile_score_map",
        data_type = "string_double_vector_map",
        save_data_ptr_to_attr = "explore_pftr_fractile_score_attr_from_redis_ptr",
      ) \
      .explore_memory_data_enrich(
        data_key="user_pic_unbiased_interest_map",
        data_type="string_uint64_vector_map",
        save_data_ptr_to_attr="user_pic_unbiased_interest_map_ptr",
      ) \
      .explore_memory_data_enrich(
        data_key = "unbias_interest_top_photo_map",
        data_type = "string_uint64_vector_map",
        save_data_ptr_to_attr = "unbias_interest_top_photo_map_ptr",
      ) \
      .explore_memory_data_enrich(
        data_key = "u2a_author_id_circle_id_detail_kuaishou",
        data_type = "uint64_uint64_map",
        save_data_ptr_to_attr = "u2a_author_id_circle_id_detail_kuaishou_ptr",
      ) \
      .if_("explore_enable_author_click_value_score_with_cover_quality_score == 1") \
        .explore_memory_data_enrich(
          data_key = "author_click_value_score_v2_low",
          data_type = "uint64_double_map",
          save_data_ptr_to_attr = "author_click_value_score_low_ptr",
        ) \
        .explore_memory_data_enrich(
          data_key = "author_click_value_score_v2_high",
          data_type = "uint64_double_map",
          save_data_ptr_to_attr = "author_click_value_score_high_ptr",
        ) \
      .else_() \
        .explore_memory_data_enrich(
          data_key = "author_click_value_score_low",
          data_type = "uint64_double_map",
          save_data_ptr_to_attr = "author_click_value_score_low_ptr",
        ) \
        .explore_memory_data_enrich(
          data_key = "author_click_value_score_high",
          data_type = "uint64_double_map",
          save_data_ptr_to_attr = "author_click_value_score_high_ptr",
        ) \
      .end_() \
      .explore_memory_data_enrich(
        data_key = "high_value_black_author_map",
        data_type = "uint64_double_vector_map",
        save_data_ptr_to_attr = "high_value_black_author_map_ptr",
      ) \
      .explore_memory_data_enrich(
        data_key = "explore_emp_xtr_decrease_photo",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "explore_emp_xtr_decrease_photo_set_ptr",
      ) \
      .explore_memory_data_enrich(
        data_key = "negative_aid",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "negative_aid_set_ptr",
      )\
      .explore_memory_data_enrich(
        data_key = "explore_emp_topson_decrease_down_photo",
        data_type = "uint64_double_map",
        save_data_ptr_to_attr = "explore_emp_topson_decrease_down_photo_map_ptr",
      ) \
      .explore_memory_data_enrich(
        data_key = "photo_collection_pids",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "photo_collection_pids_set_ptr",
      ) \
      .explore_memory_data_enrich(
        data_key = "interest_cid_collaborative_score",
        data_type = "uint64_double_vector_map",
        save_data_ptr_to_attr = "interest_cid_collaborative_score_map"
      ) \
      .explore_memory_data_enrich(
        data_key = "pid_neg_fb_ratio_list",
        data_type = "uint64_double_map",
        save_data_ptr_to_attr = "explore_emp_neg_feedback_photo_set_ptr",
      ) \
      .explore_memory_data_enrich(
        data_key = "high_upload_photo_author_map",
        data_type = "uint64_double_map",
        save_data_ptr_to_attr = "high_upload_photo_author_map_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "yanghao_disu_uids",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "sexy_induce_photo_set_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "gaofen_signs_uids",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "gaofen_signs_uids_set_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "kuaishou_official_accounts",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "kuaishou_official_account_set_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "personalization_authors",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "personalization_author_set_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "hierarchy_label_uids",
        data_type = "uint64_uint64_map",
        save_data_ptr_to_attr = "hierarchy_label_uids_map_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "android_app_package_id_cluster_mapping",
        data_type = "string_uint64_vector_map",
        save_data_ptr_to_attr = "android_app_package_id_cluster_id_map_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "ios_app_name_cluster_mapping",
        data_type = "string_uint64_vector_map",
        save_data_ptr_to_attr = "ios_app_name_cluster_id_map_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "user_hate_photo_mapping",
        data_type = "string_uint64_vector_map",
        save_data_ptr_to_attr = "user_hate_photo_id_map_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "highest_level_hetu_tag_map",
        data_type = "uint64_uint64_map",
        save_data_ptr_to_attr = "highest_hetu_tag_map_ptr",
      ) \
      .explore_memory_data_enrich(
        data_key = "cold_item_quality_score",
        data_type = "uint64_double_map",
        save_data_ptr_to_attr = "cold_item_quality_score_map_ptr",
      ) \
      .explore_memory_data_enrich(
        data_key = "ecommerce_good_author_show_case",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "ecommerce_good_author_show_case_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "ecommerce_good_author_e_commerce",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "ecommerce_good_author_e_commerce_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "illegal_word_pids",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "illegal_word_pids_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "explore_pic_operation_target_filter_hash_tagid",
        data_type = "uint64_set",
        save_data_ptr_to_attr = "pic_operation_filter_hash_tagid_ptr"
      ) \
      .explore_memory_data_enrich(
        data_key = "explore_pic_cid632_similar_map",
        data_type = "uint64_double_vector_map",
        save_data_ptr_to_attr = "pic_interest_cid_collaborative_score_map"
      ) \
      .explore_memory_data_enrich(
        data_key = "tagnex_lv3_to_lv2_map",
        data_type = "uint64_uint64_map",
        save_data_ptr_to_attr = "explore_no_bias_tagnex_lv3_to_lv2_map_ptr",
      )

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        common_attrs = ["explore_fullrank_hourly_xtr_debias_map_ptr", "hetu_gender_to_norm_ctr_map_ptr",
                        "pic_da_user_pref_ptr", "pic_fr_rel_score_pct_map", "pic_rerank_pid_realshow_data_map"],
        for_debug_request_only = True,
      )
