from ranking import CommonModule

class VariantDiversityModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def diversify(self, name = "explore_fr_diversify", target_item = {}, prev_items="explore_recent_play_list"):
        return self.flow \
        .explore_ranking_gen_part_diversity_tag(
          mmu_photo_low_quality_model_attr = "mmu_photo_low_quality_model",
          hetu_level_one_tag_list_attr = "hetu_tag_level_info_v2__hetu_level_one",
          merchant_item_id_list_attr = "merchant_item_info__item_id_list",
          merchant_photo_cart_relation_attr = "merchant_photo_cart_relation",
          hetu_level_two_tag_list_attr = "hetu_tag_level_info__hetu_level_two",
          hetu_tag_list_attr = "hetu_tag_level_info__hetu_tag",
          is_soft_porn_cover_attr = "is_soft_porn_cover",
          is_bad_feeling_cover_attr = "is_bad_feeling_cover",
          is_marketing_cover_attr = "is_marketing_cover",
          is_full_text_cover_attr = "is_full_text_cover",
          is_low_resolution_cover_attr = "is_low_resolution_cover",
          is_pure_text_cover_attr = "is_pure_text_cover",
          is_pure_porn_cover_attr = "is_pure_porn_cover",
          audit_hot_cover_attr = "audit_hot_cover_level",
          is_audit_gray_cover_attr = "is_audit_gray_cover",
          is_audit_gray_porn_cover_attr = "is_audit_gray_porn_cover",
          is_audit_gray_bad_feeling_cover_attr = "is_audit_gray_bad_feeling_cover",
          is_audit_gray_low_quality_cover_attr = "is_audit_gray_low_quality_cover",
          is_audit_gray_sensitive_word_cover_attr = "is_audit_gray_sensitive_word_cover",
          is_new_marketing_sense_attr = "is_new_marketing_sense",
          enable_cart_photo_scatter_attr = "{{enable_cart_photo_scatter}}",
          enable_mmu_score_fix_attr = "{{enable_mmu_score_fix}}",
          low_resolution_cover_threshold_attr = "{{low_resolution_cover_threshold}}",
          use_hetu_v3 = "{{use_hetu_v3}}",
          target_item = target_item
        ) \
        .enrich_attr_by_light_function(
          import_item_attr = [
            "hetu_tag_level_info__hetu_level_five"
          ],
          import_common_attr = [
            "specified_hetu5_str"
          ],
          export_item_attr = [
            "specified_hetu5_found"
          ],
          function_name = "DiversitySpecifiedHetu5",
          class_name = "ExploreLightFunctionSetV2",
        ) \
        .if_("enable_diversity_hetu_v2 == 1") \
          .explore_transform_hetu_tag(
            output_attrs = ["hetu_level_one_v2", "hetu_level_two_v2",  "hetu_level_three_v2"],
            hetu_tag_attrs = ["hetu_tag_level_info_v2__hetu_level_one", "hetu_tag_level_info_v2__hetu_level_two", "hetu_tag_level_info_v2__hetu_level_three"]
          ) \
          .explore_transform_hetu_tag(
            item_list_from_attr = prev_items,
            output_attrs = ["hetu_level_one_v2", "hetu_level_two_v2",  "hetu_level_three_v2"],
            hetu_tag_attrs = ["hetu_tag_level_info_v2__hetu_level_one", "hetu_tag_level_info_v2__hetu_level_two", "hetu_tag_level_info_v2__hetu_level_three"]
          ) \
        .end_() \
        .if_("enable_high_value_author_explore_diversity == 1 or (explore_ranking_enable_only_cold_start == 1 and refreshTimes == 0) or (explore_ranking_enable_only_first_page == 1 and page_index == 1)") \
          .gen_common_attr_by_lua(
            attr_map = {
              "enable_high_value_author_explore_diversity_sample": "util.Random() < explore_ranking_high_value_author_freq_thred and 1 or 0"
            }
          ) \
        .end_() \
        .if_("enable_friend_recommendation_explore_diversity == 1 or (enable_ranking_friend_recommendation_explore_only_cold_start == 1 and refreshTimes == 0) or (enable_ranking_friend_recommendation_explore_only_first_page == 1 and page_index == 1)") \
          .gen_common_attr_by_lua(
            attr_map = {
              "enable_friend_recommendation_explore_diversity_sample": "util.Random() < explore_ranking_friend_recommendation_freq_thred and 1 or 0"
            }
          ) \
        .end_() \
        .if_("enable_cluster_id_control_dynamic == 1") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              "uOldMmuClusterId300ListList",
              {"name": "fr_hetu_cluster_winsize", "as": "old_window_size"},
              "cluster_id_window_size_dynamic_mode",
              "cluster_id_window_size_upper_bound",
              "cluster_id_window_size_discount_coef",
              "cluster_id_window_size_lower_bound",
            ],
            export_common_attr = [
              {"name": "new_window_size", "as": "fr_hetu_cluster_winsize"},
            ],
            function_name = "DynamicClusterIdWindowSize",
            class_name = "ExploreLightFunctionSetV2"
          ) \
        .end_() \
        .if_("enable_explore_cocoon_cluster_id_control_dynamic == 1 and user_cocoon_flag == 1") \
          .multi_int_value_adjust(
            int_value_name_list = [
              "fr_hetu_cluster_winsize",
              "fr_refactoring_winsize6"
            ],
            strategy_name = "cocoon"
          ) \
        .end_() \
        .if_("explore_enable_user_need_break_cocoon_fr_s2 == 1 and user_need_break_cocoon_flag == 1") \
          .multi_int_value_adjust(
            int_value_name_list = [
              "fr_hetu_cluster_winsize",
              "fr_refactoring_winsize6",
              "fr_refactoring_max6",
            ],
            strategy_name = "user_need_break_cocoon"
          ) \
        .end_() \
        .if_("enable_diversity_hot_high_bad == 1") \
          .enrich_attr_by_light_function(
            import_item_attr = [
              "audit_hot_high_tag_level",
            ],
            export_item_attr = [
              "is_hot_high_bad",
            ],
            function_name = "IsHotHighBad",
            class_name = "ExploreLightFunctionSetV2",
          ) \
          .enrich_attr_by_light_function(
            item_list_from_attr = prev_items,
            import_item_attr = [
              "audit_hot_high_tag_level",
            ],
            export_item_attr = [
              "is_hot_high_bad",
            ],
            function_name = "IsHotHighBad",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .end_() \
        .if_("enable_opne_appearance_hetu1_diversity == 1", to_be_delete = "date=2024-05-29;committer=fengjingping") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              "explore_selected_hetu_category_map_str",
            ],
            import_item_attr = [
              "hetu_level_one_v2",
              "hetu_level_two_v2",
            ],
            export_item_attr = [
              "appearance_hetu_level_one",
            ],
            function_name = "SelectHetuCategoryDiversity",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .end_() \
        .if_("enable_explore_rank_gen_wide_screen_photo == 1", to_be_delete = "date=2024-05-29;committer=liuhao07") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              "height_width_ratio_thres",
            ],
            import_item_attr = [
              "width",
              "height",
            ],
            export_item_attr = [
              "is_wide_screen_photo",
            ],
            function_name = "IsWideScreenPhoto",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .end_() \
        .if_("enable_explore_merchant_impress_id == 1") \
          .split_string(
            input_common_attr = "audit_b_second_level_black_tags",
            output_common_attr = "audit_b_second_level_black_tags_list",
            delimiters = ",",
            trim_spaces = True,
            skip_empty_tokens = True,
            parse_to_int = True,
          ) \
          .enrich_attr_by_light_function(
            import_item_attr = [
              {"name": "audit_b_second_tag", "as": "attr"},
            ],
            import_common_attr = [
              {"name": "audit_b_second_level_black_tags_list", "as": "attr_list"},
            ],
            export_item_attr = [
              {"name": "is_in_set", "as": "is_merchant_impress_id"},
            ],
            function_name = "AttrIsInSet",
            class_name = "ExploreLightFunctionSetV2",
          ) \
          .enrich_attr_by_light_function(
            item_list_from_attr = prev_items,
            import_item_attr = [
              {"name": "audit_b_second_tag", "as": "attr"},
            ],
            import_common_attr = [
              {"name": "audit_b_second_level_black_tags_list", "as": "attr_list"},
            ],
            export_item_attr = [
              {"name": "is_in_set", "as": "is_merchant_impress_id"},
            ],
            function_name = "AttrIsInSet",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .end_() \
        .gen_common_attr_by_lua(
          attr_map = {
            "enable_explore_first_page_personalization_or_original_submission_author_diversity": 
              "page_index == 1 and enable_explore_personalization_or_original_submission_author_diversity"
          },
        ) \
        .if_("enable_explore_first_page_personalization_or_original_submission_author_diversity == 1") \
          .enrich_attr_by_light_function(
            import_item_attr = [
              {"name": "original_submission_author_tag", "as": "tag1"},
              {"name": "personalization_author_tag", "as": "tag2"},
            ],
            export_item_attr = [
              {"name": "final_tag", "as": "is_personalization_or_original_submission_tag"},
            ],
            function_name = "GenItemLogicalOrTag",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .end_() \
        .gen_common_attr_by_lua(
          attr_map = {
            "enable_first_refresh_good_photo_diversity": 
              "is_first_refresh == 1 and enable_first_refresh_good_photo_diversity"
          },
        ) \
        .diversify_by_rules(
          name = name,
          traceback = True,
          max_satisfied_pick="{{fr_variety_engineer_slot_num_shuanglie}}",
          range_end="{{fr_variety_gen_engineer_limit_thres}}",
          prev_items_from_attr = prev_items,
          rules=[
            dict(attr_name="is_first_refresh_good_photo",
                 enabled="{{enable_first_refresh_good_photo_diversity}}",
                 window_size= "{{explore_fr_first_refresh_good_diversity_winsize}}",
                 max_num="{{explore_fr_first_refresh_good_diversity_max_num}}",
                 min_num = "{{explore_fr_first_refresh_good_diversity_min_num}}",
                 priority="{{fr_refactoring_priority_level1_new}}"),
            dict(attr_name = "is_minority_photo",
                 enabled = "{{enable_minority_photo_diversity}}",
                 window_size = "{{minority_photo_diversity_winsize}}",
                 max_num = "{{minority_photo_diversity_max_num}}",
                 min_num = "{{minority_photo_diversity_min_num}}",
                 priority = "{{minority_photo_diversity_priority}}",
                 consider_prev_items = "{{enable_minority_photo_diversity_consider_prev_items}}"),
            dict(attr_name="is_hot_rank_photo",
                 enabled="{{enable_hot_photo_diversity}}",
                 window_size= "{{fr_refactoring_winsize2}}",
                 max_num="{{fr_refactoring_max2}}",
                 priority="{{fr_refactoring_priority_level2_new}}"),
            dict(attr_name="picture_variant_attr",
                 enabled="{{enable_picture_diversity}}",
                 window_size= "{{fr_refactoring_winsize3}}",
                 max_num="{{fr_refactoring_max3}}",
                 priority="{{fr_refactoring_priority_level3_new}}"),
            dict(attr_name="mounted_interest_cluster_id",
                 enabled="{{enable_hetu_cluster_diversity}}",
                 window_size= "{{fr_hetu_cluster_winsize}}",
                 max_num="{{fr_hetu_cluster_max}}",
                 priority="{{fr_refactoring_priority_hetu_cluster_level}}",
                 consider_prev_items="{{enable_hetu_cluster_consider_prev_items}}"),
            dict(attr_name="photo_dnn_cluster_id",
                 enabled="{{enable_dnn_cluster_id_diversity}}",
                 window_size= "{{fr_refactoring_winsize4}}",
                 max_num="{{fr_refactoring_max4}}",
                 priority="{{fr_refactoring_priority_photo_dnn_cluster_id_level}}"),
            dict(attr_name="hetu_tag_level_info__hetu_level_one",
                 enabled="{{enable_hetu1_diversity}}",
                 window_size= "{{fr_refactoring_winsize6}}",
                 max_num="{{fr_refactoring_max6}}",
                 priority="{{fr_refactoring_priority_photo_hetu_one_level}}"),
            dict(attr_name="hetu_tag_level_info__hetu_level_two",
                 enabled="{{enable_hetu2_diversity}}",
                 window_size= "{{fr_refactoring_winsize7}}",
                 max_num="{{fr_refactoring_max7}}",
                 priority="{{fr_refactoring_priority_hetu_two_level}}"),
            dict(attr_name="hetu_tag_level_info__hetu_level_three",
                 enabled="{{enable_hetu3_diversity}}",
                 window_size= "{{fr_refactoring_winsize8}}",
                 max_num="{{fr_refactoring_max8}}",
                 priority="{{fr_refactoring_priority_hetu_three_level}}"),
            dict(attr_name= "hetu_tag_level_info__hetu_level_five",
                 enabled="{{enable_hetu5_diversity}}",
                 window_size= "{{fr_refactoring_winsize9}}",
                 max_num="{{fr_refactoring_max9}}",
                 priority="{{fr_refactoring_priority_hetu_five_level}}"),
            dict(attr_name= "hetu_tag_level_info__hetu_face_id",
                 enabled="{{enable_hetu_faceid_diversity}}",
                 window_size= "{{fr_refactoring_faceid_winsize}}",
                 max_num="{{fr_refactoring_faceid_max}}",
                 priority="{{fr_refactoring_priority_hetu_five_level}}"),
            dict(attr_name="hetu_level_one_v2",
                 enabled="{{enable_hetu1_v2_diversity}}",
                 window_size= "{{fr_refactoring_winsize10}}",
                 max_num="{{fr_refactoring_max10}}",
                 priority="{{fr_refactoring_priority_level1_new}}",
                 consider_prev_items="{{enable_consider_prev_hetu1_v2_diversity}}"),
            dict(attr_name="appearance_hetu_level_one",
                 enabled="{{enable_appearance_hetu1_diversity}}",
                 window_size= "{{fr_appearance_hetu1_refactoring_winsize}}",
                 max_num="{{fr_refactoring_appearance_hetu1_max}}",
                 priority="{{fr_refactoring_appearance_hetu1_priority_level}}"),
            dict(attr_name="hetu_level_two_v2",
                 enabled="{{enable_hetu2_v2_diversity}}",
                 window_size= "{{fr_refactoring_winsize11}}",
                 max_num="{{fr_refactoring_max11}}",
                 priority="{{fr_refactoring_priority_level2_new}}",
                 consider_prev_items="{{enable_consider_prev_hetu2_v2_diversity}}"),
            dict(attr_name="hetu_level_three_v2",
                 enabled="{{enable_hetu3_v2_diversity}}",
                 window_size= "{{fr_refactoring_winsize12}}",
                 max_num="{{fr_refactoring_max12}}",
                 priority="{{fr_refactoring_priority_level1_new}}",
                 consider_prev_items="{{enable_consider_prev_hetu3_v2_diversity}}"),
            dict(attr_name= "is_follow_author",
                 enabled="{{enable_follow_author_diversity}}",
                 window_size= "{{fr_refactoring_winsize20}}",
                 max_num="{{fr_refactoring_max20}}",
                 priority="{{fr_refactoring_priority_follow_author_level}}"),
            dict(attr_name= "is_soft_porn_cover",
                 enabled="{{enable_soft_porn_cover_diversity}}",
                 window_size= "{{fr_refactoring_winsize29}}",
                 max_num="{{fr_refactoring_max29}}",
                 priority="{{fr_refactoring_priority_is_soft_porn_cover_level}}"),
            dict(attr_name= "is_bad_feeling_cover",
                 enabled="{{enable_bad_feeling_cover_diversity}}",
                 window_size= "{{fr_refactoring_winsize33}}",
                 max_num="{{fr_refactoring_max33}}",
                 priority="{{fr_refactoring_priority_is_bad_feeling_level}}"),
            dict(attr_name= "is_marketing_cover",
                 enabled="{{enable_marketing_cover_diversity}}",
                 window_size= "{{fr_refactoring_winsize34}}",
                 max_num="{{fr_refactoring_max34}}",
                 priority="{{fr_refactoring_priority_is_marketing_cover_level}}"),
            dict(attr_name= "is_full_text_cover",
                 enabled="{{enable_full_text_cover_diversity}}",
                 window_size= "{{fr_refactoring_winsize35}}",
                 max_num="{{fr_refactoring_max35}}",
                 priority="{{fr_refactoring_priority_is_full_text_cover_level}}"),
            dict(attr_name= "is_low_resolution_cover",
                 enabled="{{enable_low_resolution_cover_diversity}}",
                 window_size= "{{fr_refactoring_winsize36}}",
                 max_num="{{fr_refactoring_max36}}",
                 priority="{{fr_refactoring_priority_is_low_resolution_cover_level}}"),
            dict(attr_name= "is_pure_text_cover",
                 enabled="{{enable_pure_text_cover_diversity}}",
                 window_size= "{{fr_refactoring_winsize37}}",
                 max_num="{{fr_refactoring_max37}}",
                 priority="{{fr_refactoring_priority_is_pure_text_cover_level}}"),
            dict(attr_name= "is_pure_porn_cover",
                 enabled="{{enable_pure_porn_cover_diversity}}",
                 window_size= "{{fr_refactoring_winsize38}}",
                 max_num="{{fr_refactoring_max38}}",
                 priority="{{fr_refactoring_priority_is_pure_porn_cover_level}}"),
            dict(attr_name= "is_audit_gray_cover",
                 enabled="{{enable_audit_gray_cover_diversity}}",
                 window_size= "{{fr_refactoring_winsize39}}",
                 max_num="{{fr_refactoring_max39}}",
                 priority="{{fr_refactoring_priority_level1_new}}"),
            dict(attr_name= "is_audit_gray_porn_cover",
                 enabled="{{enable_audit_gray_porn_cover_diversity}}",
                 window_size= "{{fr_refactoring_winsize40}}",
                 max_num="{{fr_refactoring_max40}}",
                 priority="{{fr_refactoring_priority_level1_new}}"),
            dict(attr_name= "is_audit_gray_bad_feeling_cover",
                 enabled="{{enable_audit_gray_bad_feeling_cover_diversity}}",
                 window_size= "{{fr_refactoring_winsize41}}",
                 max_num="{{fr_refactoring_max41}}",
                 priority="{{fr_refactoring_priority_level1_new}}"),
            dict(attr_name= "is_audit_gray_low_quality_cover",
                 enabled="{{enable_audit_gray_low_quality_cover_diversity}}",
                 window_size= "{{fr_refactoring_winsize42}}",
                 max_num="{{fr_refactoring_max42}}",
                 priority="{{fr_refactoring_priority_level1_new}}"),
            dict(attr_name= "is_audit_gray_sensitive_word_cover",
                 enabled="{{enable_audit_gray_sensitive_word_cover_diversity}}",
                 window_size= "{{fr_refactoring_winsize43}}",
                 max_num="{{fr_refactoring_max43}}",
                 priority="{{fr_refactoring_priority_level1_new}}"),
            dict(attr_name= "is_new_marketing_sense",
                 enabled="{{enable_new_marketing_sense_photo_diversity}}",
                 window_size= "{{fr_refactoring_winsize44}}",
                 max_num="{{fr_refactoring_max44}}",
                 priority="{{fr_refactoring_priority_level1_new}}"),
            dict(attr_name= "mmu_content_ids_33",
                 enabled="{{enable_mmu_first_frame_diversity}}",
                 window_size= "{{fr_refactoring_winsize45}}",
                 max_num="{{fr_refactoring_max45}}",
                 priority="{{fr_refactoring_priority_level1_new}}"),
            dict(attr_name= "is_good_looking",
                 enabled="{{enable_fr_good_looking_diversity}}",
                 window_size= "{{fr_refactoring_winsize46}}",
                 max_num="{{fr_refactoring_max46}}",
                 priority="{{fr_refactoring_priority_level3_new}}"),
            dict(attr_name="is_hot_high_bad",
                 enabled="{{enable_hot_high_bad_diversity}}",
                 window_size="{{fr_hot_high_bad_refactoring_winsize}}",
                 max_num="{{fr_hot_high_bad_refactoring_max}}",
                 priority="{{fr_hot_high_bad_refactoring_priority}}",
                 consider_prev_items="{{enable_hot_high_bad_consider_prev_items}}"),
            dict(attr_name= "specified_hetu5_found",
                enabled="{{enable_fr_specified_hetu5_found}}",
                window_size= "{{fr_specified_hetu5_found_winsize}}",
                max_num="{{fr_specified_hetu5_found_max}}",
                priority="{{fr_specified_hetu5_found_priority}}"),
            dict(attr_name="is_new_interest_explore",
                 enabled="{{enable_new_interest_explore_diversity}}",
                 window_size="{{fr_new_interest_explore_refactoring_winsize}}",
                 max_num="{{fr_new_interest_explore_refactoring_max}}",
                 min_num="{{fr_new_interest_explore_refactoring_min}}",
                 priority="{{fr_new_interest_explore_refactoring_priority}}",
                 consider_prev_items="{{enable_new_interest_explore_consider_prev_items}}"),
            dict(attr_name="is_high_value_author",
                 enabled="{{enable_high_value_author_explore_diversity_sample}}",
                 window_size="{{fr_high_value_author_explore_refactoring_winsize}}",
                 max_num="{{fr_high_value_authort_explore_refactoring_max}}",
                 min_num="{{fr_high_value_author_explore_refactoring_min}}",
                 priority="{{fr_high_value_author_explore_refactoring_priority}}",
                 consider_prev_items="{{enable_high_value_author_explore_consider_prev_items}}"),
            dict(attr_name="is_recommend_by_friend",
                 enabled="{{enable_friend_recommendation_explore_diversity_sample}}",
                 window_size="{{fr_friend_recommendation_explore_refactoring_winsize}}",
                 max_num="{{fr_friend_recommendation_explore_refactoring_max}}",
                 min_num="{{fr_friend_recommendation_explore_refactoring_min}}",
                 priority="{{fr_friend_recommendation_explore_refactoring_priority}}"),
            dict(attr_name= "is_wide_screen_photo",
                 enabled="{{enable_wide_screen_photo_diversity}}",
                 window_size= "{{wide_screen_photo_diversity_winsize}}",
                 max_num="{{wide_screen_photo_diversity_max_num}}",
                 min_num="{{wide_screen_photo_diversity_min_num}}",
                 priority="{{wide_screen_photo_diversity_priority}}"),
            dict(attr_name = "is_merchant_hetu_tag_id",
                 enabled = "{{enable_merchant_hetu_tag_diversity}}",
                 window_size = "{{merchant_hetu_tag_diversity_winsize}}",
                 max_num = "{{merchant_hetu_tag_diversity_max_num}}",
                 min_num = "{{merchant_hetu_tag_diversity_min_num}}",
                 priority = "{{merchant_hetu_tag_diversity_priority}}",
                 consider_prev_items="{{enable_merchant_hetu_tag_explore_consider_prev_items}}"),
            dict(attr_name = "is_merchant_impress_id",
                 enabled = "{{enable_merchant_impress_id_diversity}}",
                 window_size = "{{merchant_impress_id_diversity_winsize}}",
                 max_num = "{{merchant_impress_id_diversity_max_num}}",
                 min_num = "{{merchant_impress_id_diversity_min_num}}",
                 priority = "{{merchant_impress_id_diversity_priority}}",
                 consider_prev_items="{{enable_merchant_impress_id_explore_consider_prev_items}}"),
            dict(attr_name = "is_marketing_compensation_photo",
                 enabled = "{{enable_marketing_compensation_photo_diversity}}",
                 window_size = "{{marketing_compensation_photo_diversity_winsize}}",
                 max_num = "{{marketing_compensation_photo_diversity_max_num}}",
                 min_num = "{{marketing_compensation_photo_diversity_min_num}}",
                 priority = "{{marketing_compensation_photo_diversity_priority}}"),
            dict(attr_name = "is_olympic_latest", # 奥运新内容保量
                 enabled = "{{enable_olympic_latest_diversity}}",
                 window_size = "{{olympic_latest_diversity_winsize}}",
                 min_num = "{{olympic_latest_diversity_min_num}}",
                 max_num = "{{olympic_latest_diversity_max_num}}",
                 priority = "{{olympic_latest_diversity_priority}}"),
            dict(attr_name = "is_olympic", # 奥运内容打散
                 enabled = "{{enable_olympic_diversity}}",
                 window_size = "{{olympic_diversity_winsize}}",
                 min_num = "{{olympic_diversity_min_num}}",
                 max_num = "{{olympic_diversity_max_num}}",
                 priority = "{{olympic_diversity_priority}}"),
            dict(attr_name = "is_pid_for_similar_author",
                 enabled = "{{enable_fr_similar_author_reason_retr}}",
                 window_size = "{{fr_similar_author_reason_retr_diversity_winsize}}",
                 min_num = "{{fr_similar_author_reason_retr_diversity_min_num}}",
                 max_num = "{{fr_similar_author_reason_retr_diversity_max_num}}",
                 priority = "{{fr_similar_author_reason_retr_diversity_priority}}"),
            dict(attr_name = "is_unbias_interest_pid_for_crows",
                 enabled = "{{enable_fr_unbias_interest_reason_retr}}",
                 window_size = "{{fr_unbias_interest_reason_retr_diversity_winsize}}",
                 min_num = "{{fr_unbias_interest_reason_retr_diversity_min_num}}",
                 max_num = "{{fr_unbias_interest_reason_retr_diversity_max_num}}",
                 priority = "{{fr_unbias_interest_reason_retr_diversity_priority}}"),
            dict(attr_name = "is_meinv_photo",
                 enabled = "{{enable_is_meinv_photo_diversity}}",
                 window_size = "{{is_meinv_photo_diversity_winsize}}",
                 min_num = "{{is_meinv_photo_diversity_min_num}}",
                 max_num = "{{is_meinv_photo_diversity_max_num}}",
                 priority = "{{is_meinv_photo_diversity_diversity_priority}}"),
            dict(attr_name = "is_protogenetic_advertise_photo",
                 enabled = "{{enable_protogenetic_advertise_photo_diversity}}",
                 window_size = "{{protogenetic_advertise_photo_diversity_winsize}}",
                 min_num = "{{protogenetic_advertise_photo_diversity_min_num}}",
                 max_num = "{{protogenetic_advertise_photo_diversity_max_num}}",
                 priority = "{{protogenetic_advertise_photo_diversity_diversity_priority}}",
                 consider_prev_items = "{{enable_protogenetic_advertise_photo_diversity_consider_prev_items}}"),
            dict(attr_name = "is_personalization_or_original_submission_tag",
                 enabled = "{{enable_explore_first_page_personalization_or_original_submission_author_diversity}}",
                 window_type = "top",
                 window_size = "{{explore_rank_personalization_or_original_submission_author_diversity_winsize}}",
                 min_num = "{{explore_rank_personalization_or_original_submission_author_diversity_min_num}}",
                 priority = "{{explore_rank_personalization_or_original_submission_author_diversity_priority}}",),
            dict(attr_name="is_senseview_lowcost_photo",
                 enabled="{{enable_explore_rank_is_senseview_lowcost_photo_diversity}}",
                 window_size= "{{explore_rank_senseview_lowcost_photo_diversity_winsize}}",
                 max_num="{{explore_rank_senseview_lowcost_photo_diversity_max_num}}",
                 min_num = "{{explore_rank_senseview_lowcost_photo_diversity_min_num}}",
                 priority="{{explore_rank_senseview_lowcost_photo_diversity_priority}}",
                 consider_prev_items="{{enable_explore_rank_senseview_lowcost_photo_explore_consider_prev_items}}"),
            dict(attr_name="reach_content",
                 enabled="{{enable_explore_reach_content_diversity}}",
                 window_size="{{explore_reach_content_diversity_winsize}}",
                 min_num="{{explore_reach_content_diversity_min_num}}",
                 max_num="{{explore_reach_content_diversity_max_num}}",
                 priority="{{explore_reach_content_diversity_priority}}"),
          ],
          target_item = target_item
        ) \
    
    def process(self) -> None:
        # 生成用于图文打散的attr，不再使用is_picture
        self.flow \
          .copy_attr(
            attrs=[{
              "from_item": "is_picture",
              "to_item": "picture_variant_attr"
            }],
            target_item = {
              "is_picture": 1
            }
          ) \
        
        self.flow.if_("enable_fr_good_looking_diversity == 1", to_be_delete = "date=2023-11-16;committer=lihaoliang") \
          .enrich_attr_by_light_function(
            import_item_attr = [
              "hetu_tag_level_info__hetu_tag",
            ],
            export_item_attr = [
              "is_good_looking",
            ],
            function_name = "IsGoodLooking",
            class_name = "ExploreLightFunctionSetV2",
          ) \
        .end_()
        
        self.flow \
        .if_("explore_ranking_diversity_enable_standard_explore_realshow_pid_list == 1") \
          .limit(
            size = "{{explore_ranking_diversity_max_keep_realshow_photoid_size}}",
            item_list_from_attr = "standard_explore_realshow_pid_list"
          ) \
        .end_() \

        # 只针对视频打散
        self.flow.if_("explore_ranking_diversity_enable_standard_explore_realshow_pid_list == 1")
        self.diversify(
          name = "explore_fr_diversify_video_standard_data",
          target_item = {
            "is_picture": 0
          },
          prev_items="standard_explore_realshow_pid_list"
        )
        self.flow.else_()
        self.diversify(
          name = "explore_fr_diversify_video",
          target_item = {
            "is_picture": 0
          },
          prev_items="explore_recent_play_list"
        )
        self.flow.end_()

        self.flow.if_("enable_explore_control_similarity_score == 1 and explore_is_low_diversity_status == 1") \
          .explore_control_similarity_score_arranger(
            common_similarity_pid_list_attr = "common_similarity_pid_list",
            item_similarity_score_attr = "item_similarity_score",
            max_satisfied_pick = "{{explore_similarity_max_satisfied_pick}}",
            similarity_score_max_threshold = "{{explore_similarity_score_max_threshold}}",
            export_common_similarity_size_attr = "explore_similarity_size"
          ) \
        .end_()

        self.flow \
          .pack_item_attr(  # 保存精排正样本
            item_source = {
              "reco_results": True,
              "total_limit": "{{mc_distill_sample_num}}",
            },
            mappings = [{
              "aggregator": "concat",
              "from_item_attr": "item_key",
              "to_common_attr": "ranking_pos_sample_list",
            }],
          )

    def calc_result_count_to_ab_metric(self):
      return self.flow \
        .cast_attr_type(
          attr_type_cast_configs=[
            {
              "to_type": "double",
              "from_item_attr": "prerank_final_index_photo",
              "to_item_attr": "prerank_final_index_double"
            },
            {
              "to_type": "double",
              "from_item_attr": "cascade_final_index",
              "to_item_attr": "cascade_final_index_double"
            },
            {
              "to_type": "double",
              "from_item_attr": "awesome_wtd_index",
              "to_item_attr": "awesome_wtd_index_double"
            },
            {
              "to_type": "double",
              "from_item_attr": "pctr_index",
              "to_item_attr": "pctr_index_double"
            },
            {
              "to_type": "double",
              "from_item_attr": "pltr_index",
              "to_item_attr": "pltr_index_double"
            },
            {
              "to_type": "double",
              "from_item_attr": "pwtr_index",
              "to_item_attr": "pwtr_index_double"
            },
            {
              "to_type": "double",
              "from_item_attr": "psvr_index",
              "to_item_attr": "psvr_index_double"
            },
          ]
        ) \
        .pack_item_attr(
          item_source = {
            "reco_results": True,
            "total_limit": 60,
          },
          mappings = [
            {
              "aggregator": "avg",
              "from_item_attr": "prerank_final_index_double",
              "to_common_attr": "rank_top60_prerank_index_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "cascade_final_index_double",
              "to_common_attr": "rank_top60_cascade_index_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "awesome_wtd_index_double",
              "to_common_attr": "rank_top60_awesome_wtd_index_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "pctr_index_double",
              "to_common_attr": "rank_top60_pctr_index_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "pltr_index_double",
              "to_common_attr": "rank_top60_pltr_index_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "pwtr_index_double",
              "to_common_attr": "rank_top60_pwtr_index_avg"
            },
            {
              "aggregator": "avg",
              "from_item_attr": "psvr_index_double",
              "to_common_attr": "rank_top60_psvr_index_avg"
            },
          ],
          target_item = {"is_picture" : 0}
        ) \
        .count_reco_result(
          save_count_to = "rank_top60_follow_author_count",
          target_item = {"is_follow_author": 1},
          range_end = 60,
        ) \
        .count_reco_result(
          save_count_to = "rank_top60_all_page_valid_interest_count",
          target_item = {"is_all_page_valid_interest": 1},
          range_end = 60
        ) \
        .count_reco_result(
          save_count_to = "rank_top60_new_interest_count",
          target_item = {"is_new_interest_explore": 1},
          range_end = 60
        ) \
        .count_reco_result(
          save_count_to = "rank_top60_outer_field_interest_count",
          target_item = {"is_outer_field_interest": 1},
          range_end = 60
        ) \
        .count_reco_result(
          save_count_to = "rank_top60_show_ration_level6_count",
          target_item = {"show_ration_level": 6},
          range_end = 60
        ) \
        .count_reco_result(
          save_count_to = "rank_top60_upload_time_day0_count",
          target_item = {"upload_time_day": 0},
          range_end = 60
        ) \
        .count_reco_result(
          save_count_to = "rank_top60_upload_time_day1_count",
          target_item = {"upload_time_day": 1},
          range_end = 60
        ) \
        .count_reco_result(
          save_count_to = "rank_top60_upload_time_day2_count",
          target_item = {"upload_time_day": 2},
          range_end = 60
        ) \
        .count_reco_result(
          save_count_to = "rank_top60_upload_time_day3_7_count",
          target_item = {"upload_time_day": [3, 4, 5, 6, 7]},
          range_end = 60
        ) \
        .count_reco_result(
          save_count_to = "rank_top60_upload_time_day30_180_count",
          select_item = {
            "attr_name": "upload_time_day",
            "compare_to": 30,
            "select_if": ">=",
          }
        ) \
        .count_reco_result(
          save_count_to = "rank_top60_explore_show_gt_show_ration_result_count",
          select_item = {
              "attr_name": "explore_stat__real_show_count",
              "compare_to": "{{show_ration_realshow_threshold}}",
              "select_if": ">"
          },
          range_end = 60
        ) \
        .count_reco_result(
          save_count_to = "rank_top60_explore_noncoverview_result_count",
          select_item = {
            "attr_name": "audit_hot_cover_level",
            "compare_to": 0,
            "select_if": "<=",
            "select_if_attr_missing": True
          },
          range_end = 60
        ) \
        .count_reco_result(
          save_count_to = "rank_top60_explore_nonsenseview_result_count",
          select_item = {
            "attr_name": "audit_b_second_tag",
            "compare_to": 0,
            "select_if": "<=",
            "select_if_attr_missing": True
          },
          range_end = 60
        ) \
        .count_reco_result(
          save_count_to = "rank_top60_bias_interest_count",
          target_item = {"is_bias_interest_tagnex": 1},
          range_end = 60
        ) \
        .send_abtest_metrics(
          metrics = [
            "rank_top60_bias_interest_count",
            "rank_top60_follow_author_count",
            "rank_top60_prerank_index_avg",
            "rank_top60_cascade_index_avg",
            "rank_top60_awesome_wtd_index_avg",
            "rank_top60_pctr_index_avg",
            "rank_top60_pltr_index_avg",
            "rank_top60_pwtr_index_avg",
            "rank_top60_psvr_index_avg",
            "rank_top60_all_page_valid_interest_count",
            "rank_top60_new_interest_count",
            "rank_top60_outer_field_interest_count",
            "rank_top60_show_ration_level6_count",
            "rank_top60_upload_time_day0_count",
            "rank_top60_upload_time_day1_count",
            "rank_top60_upload_time_day2_count",
            "rank_top60_upload_time_day3_7_count",
            "rank_top60_upload_time_day30_180_count",
            "rank_top60_explore_show_gt_show_ration_result_count",
            "rank_top60_explore_noncoverview_result_count",
            "rank_top60_explore_nonsenseview_result_count"
          ],
          metric_name_prefix = "explore_reco_leaf_",
        )

    def post_process(self) -> None:
        self.flow.if_("_IS_ABTEST_METRICS_SAMPLING_REQUEST_ == 1 and _IS_ONLINE_SERVICE_ == 1 and _IS_NOT_BACKUP_ == 1")
        self.calc_result_count_to_ab_metric()
        self.flow.end_()
        self.flow \
          .log_debug_info(
            item_num_limit = 20,
            common_attrs = [],
            item_attrs = [
              "is_picture",
              "picture_variant_attr",
              "explore_similarity_size",
              "hetu_tag_level_info__hetu_level_one",
            ],
            for_debug_request_only = True
          )
