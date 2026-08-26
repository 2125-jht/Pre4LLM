from ranking import CommonModule

class VariantDiversityModule(CommonModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def diversify(self, name = "explore_fr_diversify", target_item = {}):
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
        .if_("enable_diversity_hetu_v2 == 1") \
          .explore_transform_hetu_tag(
            output_attrs = ["hetu_level_one_v2", "hetu_level_two_v2",  "hetu_level_three_v2"],
            hetu_tag_attrs = ["hetu_tag_level_info_v2__hetu_level_one", "hetu_tag_level_info_v2__hetu_level_two", "hetu_tag_level_info_v2__hetu_level_three"]
          ) \
        .end_() \
        .if_("xlife_enable_fr_target_content_control == 1") \
          .get_kconf_params(
            kconf_configs = [{
              "kconf_key": "reco.eyeshot.LifeTabTargetHetu",
              "value_type": "list_int64",
              "export_common_attr": "target_hetu_list",
              "default_value": []
            }]
          ) \
          .get_kconf_params(
            kconf_configs = [{
              "kconf_key": "reco.eyeshot.LifeTabNotTargetHetu",
              "value_type": "list_int64",
              "export_common_attr": "not_target_hetu_list",
              "default_value": []
            }]
          ) \
          .enrich_attr_by_light_function(
            import_item_attr = [
              "photo_id",
              "hetu_tag_level_info__hetu_level_one",
              "hetu_tag_level_info__hetu_level_two"
            ],
            import_common_attr = [
              "target_hetu_list",
              "not_target_hetu_list"
            ],
            export_item_attr = [
              "gray_target", # 灰度 + 非生活打散，生活设为pid
              "not_life_target" # 非生活打散，灰度 + 生活设置为pid
            ],
            function_name = "ContentControlDiversifyTag",
            class_name = "ExploreLifeLightFunctionSet"
          ) \
        .end_() \
        .if_("enable_xlife_calc_second_tag_quality_level == 1") \
          .split_string(
            input_common_attr = "second_tag_quality_level1_str",
            output_common_attr = "second_tag_quality_level1_list",
            delimiters = ",",
            trim_spaces = True,
            skip_empty_tokens = True,
            parse_to_int = True
          ) \
          .split_string(
            input_common_attr = "second_tag_quality_level2_str",
            output_common_attr = "second_tag_quality_level2_list",
            delimiters = ",",
            trim_spaces = True,
            skip_empty_tokens = True,
            parse_to_int = True
          ) \
          .enrich_attr_by_light_function(
            import_common_attr = [
              "second_tag_quality_level1_list",
              "second_tag_quality_level2_list"
            ],
            import_item_attr = [
              "audit_b_second_tag",
            ],
            export_item_attr = [
              "second_tag_quality_level1",
              "second_tag_quality_level2",
            ],
            function_name = "CalcSecondTagQualityLevel",
            class_name = "ExploreLifeLightFunctionSet",
          ) \
        .end_() \
        .diversify_by_rules(
          name = name,
          traceback = True,
          max_satisfied_pick="{{fr_variety_engineer_slot_num_shuanglie}}",
          range_end="{{fr_variety_gen_engineer_limit_thres}}",
          rules=[
            dict(attr_name= "gray_target",
                  enabled="{{enable_xlife_fr_gray_control}}",
                  window_size="{{xlife_gray_control_window}}",
                  max_num="{{xlife_gray_control_max}}",
                  priority="{{fr_refactoring_priority_level4_new}}"),
            dict(attr_name= "not_life_target",
                  enabled="{{enable_xlife_fr_target_control}}",
                  window_size="{{xlife_target_control_window}}",
                  max_num="{{xlife_target_control_max}}",
                  priority="{{fr_refactoring_priority_level4_new}}"),
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
            dict(attr_name="photo_dnn_cluster_id",
                 enabled=True,
                 window_size= "{{fr_refactoring_winsize4}}",
                 max_num="{{fr_refactoring_max4}}",
                 priority="{{fr_refactoring_priority_level1_new}}"),
            dict(attr_name="hetu_tag_level_info__hetu_level_one",
                 enabled="{{enable_hetu1_diversity}}",
                 window_size= "{{fr_refactoring_winsize6}}",
                 max_num="{{fr_refactoring_max6}}",
                 priority="{{fr_refactoring_priority_level1_new}}"),
            dict(attr_name="hetu_tag_level_info__hetu_level_two",
                 enabled="{{enable_hetu2_diversity}}",
                 window_size= "{{fr_refactoring_winsize7}}",
                 max_num="{{fr_refactoring_max7}}",
                 priority="{{fr_refactoring_priority_level2_new}}"),
            dict(attr_name="hetu_tag_level_info__hetu_level_three",
                 enabled="{{enable_hetu3_diversity}}",
                 window_size= "{{fr_refactoring_winsize8}}",
                 max_num="{{fr_refactoring_max8}}",
                 priority="{{fr_refactoring_priority_level1_new}}"),
            dict(attr_name= "hetu_tag_level_info__hetu_level_five",
                 enabled=True,
                 window_size= "{{fr_refactoring_winsize9}}",
                 max_num="{{fr_refactoring_max9}}",
                 priority="{{fr_refactoring_priority_level1_new}}"),
            dict(attr_name="hetu_level_one_v2",
                 enabled="{{enable_hetu1_v2_diversity}}",
                 window_size= "{{fr_refactoring_winsize10}}",
                 max_num="{{fr_refactoring_max10}}",
                 priority="{{fr_refactoring_priority_level1_new}}"),
            dict(attr_name="hetu_level_two_v2",
                 enabled="{{enable_hetu2_v2_diversity}}",
                 window_size= "{{fr_refactoring_winsize11}}",
                 max_num="{{fr_refactoring_max11}}",
                 priority="{{fr_refactoring_priority_level2_new}}"),
            dict(attr_name="hetu_level_three_v2",
                 enabled="{{enable_hetu3_v2_diversity}}",
                 window_size= "{{fr_refactoring_winsize12}}",
                 max_num="{{fr_refactoring_max12}}",
                 priority="{{fr_refactoring_priority_level1_new}}"),
            dict(attr_name= "is_follow_author",
                 enabled="{{enable_follow_author_diversity}}",
                 window_size= "{{fr_refactoring_winsize20}}",
                 max_num="{{fr_refactoring_max20}}",
                 priority="{{fr_refactoring_priority_level3_new}}"),
            dict(attr_name= "is_soft_porn_cover",
                 enabled="{{enable_soft_porn_cover_diversity}}",
                 window_size= "{{fr_refactoring_winsize29}}",
                 max_num="{{fr_refactoring_max29}}",
                 priority="{{fr_refactoring_priority_level1_new}}"),
            dict(attr_name= "is_bad_feeling_cover",
                 enabled="{{enable_bad_feeling_cover_diversity}}",
                 window_size= "{{fr_refactoring_winsize33}}",
                 max_num="{{fr_refactoring_max33}}",
                 priority="{{fr_refactoring_priority_level1_new}}"),
            dict(attr_name= "is_marketing_cover",
                 enabled="{{enable_marketing_cover_diversity}}",
                 window_size= "{{fr_refactoring_winsize34}}",
                 max_num="{{fr_refactoring_max34}}",
                 priority="{{fr_refactoring_priority_level1_new}}"),
            dict(attr_name= "is_full_text_cover",
                 enabled="{{enable_full_text_cover_diversity}}",
                 window_size= "{{fr_refactoring_winsize35}}",
                 max_num="{{fr_refactoring_max35}}",
                 priority="{{fr_refactoring_priority_level1_new}}"),
            dict(attr_name= "is_low_resolution_cover",
                 enabled="{{enable_low_resolution_cover_diversity}}",
                 window_size= "{{fr_refactoring_winsize36}}",
                 max_num="{{fr_refactoring_max36}}",
                 priority="{{fr_refactoring_priority_level1_new}}"),
            dict(attr_name= "is_pure_text_cover",
                 enabled="{{enable_pure_text_cover_diversity}}",
                 window_size= "{{fr_refactoring_winsize37}}",
                 max_num="{{fr_refactoring_max37}}",
                 priority="{{fr_refactoring_priority_level1_new}}"),
            dict(attr_name= "is_pure_porn_cover",
                 enabled="{{enable_pure_porn_cover_diversity}}",
                 window_size= "{{fr_refactoring_winsize38}}",
                 max_num="{{fr_refactoring_max38}}",
                 priority="{{fr_refactoring_priority_level1_new}}"),
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
            dict(attr_name= "second_tag_quality_level1",
                 enabled="{{enable_xlife_fr_second_tag_quality_level1_diversity}}",
                 window_size= "{{xlife_fr_second_tag_quality_level1_winsize}}",
                 max_num="{{xlife_fr_second_tag_quality_level1_maxnum}}",
                 priority="{{xlife_fr_second_tag_quality_level1_priority}}"),
            dict(attr_name= "second_tag_quality_level2",
                 enabled="{{enable_xlife_fr_second_tag_quality_level2_diversity}}",
                 window_size= "{{xlife_fr_second_tag_quality_level2_winsize}}",
                 max_num="{{xlife_fr_second_tag_quality_level2_maxnum}}",
                 priority="{{xlife_fr_second_tag_quality_level2_priority}}"),
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
            dict(attr_name = "is_search_boost", # 最近搜索内容保量
                 enabled = "{{enable_life_fr_search_latest_diversity}}",
                 window_size = "{{life_fr_search_latest_diversity_winsize}}",
                 min_num = "{{life_fr_search_latest_diversity_min_num}}",
                 max_num = "{{life_fr_search_latest_diversity_max_num}}",
                 priority = "{{life_fr_search_latest_diversity_priority}}"),
            dict(attr_name = "is_minority_photo",
                 enabled = "{{enable_life_fr_minority_photo_diversity}}",
                 window_size = "{{life_fr_minority_photo_diversity_winsize}}",
                 max_num = "{{life_fr_minority_photo_diversity_max_num}}",
                 priority = "{{life_fr_minority_photo_diversity_priority}}"),
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
        
        self.flow.if_("enable_fr_good_looking_diversity == 1") \
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
          .copy_attr(
              attrs = [
                  {
                    "from_item": "photo_id",
                    "to_item": "is_good_looking"
                  }
              ],
              target_item = {"is_good_looking" : 0}
          ) \
        .end_()

        self.flow.if_("skip_variant_diversity_only_for_video == 0")
        
        # 只针对视频打散
        self.diversify(
          name = "explore_fr_diversify_video", 
          target_item = {
            "is_picture": 0
          }
        )

        self.flow.else_()

        # 全局打散
        self.diversify(
          name = "explore_fr_diversify", 
          target_item = {}
        )

        self.flow.end_()

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
          .count_reco_result(
            save_count_to = "final_variant_top200_pic_result_count",
            target_item = {"is_picture": 1},
            range_end = 200
          ) \
          .count_reco_result(
            save_count_to = "final_variant_top60_result_count",
            range_end = 60
          ) \
          .count_reco_result(
            save_count_to = "final_variant_top60_pic_result_count",
            target_item = {"is_picture": 1},
            range_end = 60
          ) \
          .send_abtest_metrics(
            metrics = [
              "final_variant_top200_pic_result_count",
              "final_variant_top60_result_count",
              "final_variant_top60_pic_result_count"
            ],
            metric_name_prefix = "explore_reco_leaf_",
          )

    def post_process(self) -> None:
        self.calc_result_count_to_ab_metric()
        self.flow \
          .log_debug_info(
            item_num_limit = 20,
            common_attrs = [],
            item_attrs = [
              "is_picture",
              "picture_variant_attr",
              "hetu_tag_level_info__hetu_level_one",
            ],
            for_debug_request_only = True
          )
