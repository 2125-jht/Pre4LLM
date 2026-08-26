from rerank import CommonModule

class FinalDiversityModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
    self.flow \
      .if_("explore_rerank_enable_gen_is_emotion_pic == 1") \
        .enrich_attr_by_light_function(
          import_item_attr = [
            "upload_type"
          ],
          export_item_attr = [
            "is_emotion_pic"
          ],
          function_name = "GenIsEmotionPic",
          class_name = "ExploreLightFunctionSetV2"
        ) \
        .enrich_attr_by_light_function(
          item_list_from_attr = "explore_recent_play_list",
          import_item_attr = [
            "upload_type"
          ],
          export_item_attr = [
            "is_emotion_pic"
          ],
          function_name = "GenIsEmotionPic",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_() \
      .if_("explore_rerank_enable_final_diversity == 1") \
        .diversify_by_rules(
          name = "explore_final_diversify",
          traceback = True,
          max_satisfied_pick = "{{explore_rerank_final_variety_engineer_max_pick_num}}",
          range_end = "{{explore_rerank_final_variety_engineer_limit_thres}}",
          prev_items_from_attr = "explore_recent_play_list",
          rules=[
            dict(attr_name = "is_minority_photo",
              enabled = "{{enable_minority_photo_diversity}}",
              window_size = "{{minority_photo_diversity_winsize}}",
              max_num = "{{minority_photo_diversity_max_num}}",
              min_num = "{{minority_photo_diversity_min_num}}",
              priority = "{{minority_photo_diversity_priority}}",
              consider_prev_items="{{enable_minority_photo_diversity_consider_prev_items}}"),
            dict(attr_name = "is_personalization_or_original_submission_tag",
              enabled = "{{enable_explore_first_page_personalization_or_original_submission_author_diversity}}",
              window_type = "top",
              window_size = "{{explore_rerank_personalization_or_original_submission_author_diversify_winsize}}",
              min_num = "{{explore_rerank_personalization_or_original_submission_author_diversify_min_num}}",
              priority = "{{explore_rerank_personalization_or_original_submission_author_diversify_diversity_priority}}",),
            dict(attr_name = "is_emotion_pic",
              enabled = "{{enable_is_emotion_pic_diversity}}",
              window_size = "{{emotion_pic_diversity_winsize}}",
              max_num = "{{emotion_pic_diversity_max_num}}",
              min_num = "{{emotion_pic_diversity_min_num}}",
              priority = "{{emotion_pic_diversity_priority}}",
              consider_prev_items="{{enable_emotion_pic_diversity_consider_prev_items}}"),
          ]
        ) \
      .end_() \
      .if_("is_traceback_request == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            {"name": "user_valid_interest_cid_list", "as": "attr_list"},
          ],
          import_item_attr = [
            {"name": "cluster_id_632", "as": "attr"},
          ],
          export_item_attr = [
            {"name": "is_in_set", "as": "is_valid_interest_explore"}
          ],
          range_end = 10,
          function_name = "AttrIsInSet",
          class_name = "ExploreLightFunctionSetV2"
        ) \
      .end_()
    return self
