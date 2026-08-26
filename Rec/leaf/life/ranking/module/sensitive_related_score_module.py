from ranking import CommonModule

class SensitiveRelatedScoreModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self) -> None:
      self.flow \
        .if_("enable_life_realtime_action_sensitive == 1") \
        .fountain_calc_related_score_v2(
          # 相关排序分
          enable_cal_photo_sim_by_intersect = "{{enable_life_realtime_related_score_calc}}",
          diversity_dim_weight = "{{life_sensitive_related_dim_weight}}",
          save_score_to_attr = "sensitive_related_score",
          int_source_attrs = [
            "sensi_hetu_sim_cluster_id", "sensi_hetu_cluster_id_v2",
            "sensiPidMmuImgClusterV3", "sensiPidMmuTextCluster", 
            "sensiPidAuthorId", "sensiPidFirstLevelCategory",
            "sensiPidSecondLevelCategory", "sensiPidThirdLevelCategory",
            "sensiPidTagId", "sensiPidUploadType",
          ],
          int_list_source_attrs = [
            "sensi_hetu_level_one_v2", "sensi_hetu_level_two_v2",
            "sensi_hetu_level_three_v2", "sensi_hetu_level_four_v2",
            "sensi_hetu_tag_v2", "sensi_hetu_face_id_v2"
          ],
          int_item_attrs = [
            "hetu_sim_cluster_id", "hetu_tag_level_info_v2__hetu_cluster_id",
            "mmu_img_cluster_v3", "mmu_text_cluster",
            "author__id", "author__category_detail__first_level_id",
            "author__category_detail__second_level_id", "author__category_detail__third_level_id",
            "tag", "upload_type",
          ],
          int_list_item_attrs = [
            "hetu_tag_level_info_v2__hetu_level_one", "hetu_tag_level_info_v2__hetu_level_two",
            "hetu_tag_level_info_v2__hetu_level_three", "hetu_tag_level_info_v2__hetu_level_four",
            "hetu_tag_level_info_v2__hetu_tag", "hetu_tag_level_info_v2__hetu_face_id",
          ],
        ) \
        .enrich_attr_by_lua(
            import_common_attr=["is_realtime_boost"],
            import_item_attr=["sensitive_related_score"],
            export_item_attr=["sensitive_related_score"],
            function_for_item="calculate",
            lua_script="""
            function calculate()
                local score = (sensitive_related_score or 0.0)
                if score < 0.03 then
                  score = 0.0
                end
                return score * (is_realtime_boost or 0)
            end
            """
        )\
        .end_() \

  def post_process(self) -> None:
      self.flow \
        .log_debug_info(
            common_attrs = [
                "colossus_photo_id_list_new",
                "colossus_play_time_list_new",
                "colossus_timestamp_list_new",
                "browse_screen__pid_list",
                "realtime_photo_id_list",
                "is_realtime_boost",
                "sensi_hetu_sim_cluster_id",
                "sensi_hetu_cluster_id_v2",
                "sensiPidAuthorId",
                "sensi_hetu_level_one_v2",
                "sensi_hetu_level_two_v2",
                "sensi_hetu_level_three_v2", 
                "sensi_hetu_level_four_v2",
                "sensi_hetu_tag_v2", 
                "sensi_hetu_face_id_v2",
            ],
            item_attrs = [
                "sensitive_related_score",
                "hetu_sim_cluster_id",
                "hetu_tag_level_info_v2__hetu_cluster_id",
                "author__id",
                "hetu_tag_level_info_v2__hetu_level_one", 
                "hetu_tag_level_info_v2__hetu_level_two",
                "hetu_tag_level_info_v2__hetu_level_three", 
                "hetu_tag_level_info_v2__hetu_level_four",
                "hetu_tag_level_info_v2__hetu_tag", 
                "hetu_tag_level_info_v2__hetu_face_id",
            ],
            for_debug_request_only = True,
            respect_sample_loggging = True,
        )