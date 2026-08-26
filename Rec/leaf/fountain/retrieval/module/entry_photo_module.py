from retrieval import CommonModule

class EntryPhotoModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)

  def process(self):
    self.flow \
      .pack_item_attr(
        item_source = {
          "reco_results": False,
          "common_attr": ["featureSourcePId"],
        },
        mappings = [
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info_v2__hetu_level_one",
            "to_common_attr": "source_hetu_level_one_v2_original",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info_v2__hetu_level_two",
            "to_common_attr": "source_hetu_level_two_v2_original",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info_v2__hetu_level_three",
            "to_common_attr": "source_hetu_level_three_v2_original",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info_v2__hetu_level_four",
            "to_common_attr": "source_hetu_level_four_v2_original",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info_v2__hetu_tag",
            "to_common_attr": "source_hetu_tag_v2_original",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info_v2__hetu_face_id",
            "to_common_attr": "source_hetu_face_id_v2_original",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info_v2__hetu_cluster_id",
            "to_common_attr": "source_hetu_cluster_id_v2",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info__hetu_level_one",
            "to_common_attr": "source_hetu_level_one",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info__hetu_level_two",
            "to_common_attr": "source_hetu_level_two",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info__hetu_level_three",
            "to_common_attr": "source_hetu_level_three",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info__hetu_level_five",
            "to_common_attr": "source_hetu_level_five",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_tag_level_info__hetu_face_id",
            "to_common_attr": "source_hetu_face_ids",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "hetu_sim_cluster_id",
            "to_common_attr": "source_hetu_sim_cluster_id",
          },
          {
            "aggregator": "copy",
            "from_item_attr": "user_hash_tag_id",
            "to_common_attr": "source_user_hash_tag_id",
          },
        ]
      ) \
      .enrich_attr_by_lua(
        import_common_attr = [
          "fountain_nn_retrieval_kess_service",
          "deviceType",
          "page",
          "featureUId",
          "photoTagBucket",
          "skip_top_subdivision_nn_retrieval_v2",
          "topSubdivisionBucket",
          "skip_explore_subdivision_nn_retrieval_v2",
          "exploreSubdivisionBucket",
          "colossusRetrievalTrigger",
          "skip_fountain_colossus_retr",
          "morePage",
          "fountain_enable_first_page_skip_u2i_retrieval",
          "fountainTopHetuBucket",
          "fountainEEHetuBucket",
          "source_hetu_level_one_v2_original",
          "source_hetu_level_two_v2_original",
          "source_hetu_level_three_v2_original",
          "source_hetu_level_four_v2_original",
          "userBrowseSetOriginalHetuLevel1",
          "userBrowseSetOriginalHetuLevel2",
          "userBrowseSetOriginalHetuLevel3",
          "source_hetu_tag_v2_original",
          "source_hetu_face_id_v2_original",
          "find_v4_skip_fountain_interact_author_retr",
          "skip_fountain_mid_photo_gnn_i2i_retr",
          "skip_fountain_colossus_retr_emb_fetch_new"
        ],
        export_common_attr = [
          "skip_top_subdivision_nn_retrieval_v2",
          "skip_explore_subdivision_nn_retrieval_v2",
          "skip_fountain_colossus_retr",
          "source_hetu_level_one_v2",
          "source_hetu_level_two_v2",
          "source_hetu_level_three_v2",
          "source_hetu_level_four_v2",
          "source_hetu_tag_v2",
          "source_hetu_face_id_v2",
          "userBrowseSetHetuLevel1",
          "userBrowseSetHetuLevel2",
          "userBrowseSetHetuLevel3",
          "skip_fountain_mid_photo_gnn_i2i_retr",
          "skip_fountain_colossus_retr_emb_fetch_old",
          "skip_fountain_colossus_retr_emb_fetch_new"
        ],
        function_for_common = "calculate",
        lua_script_file = "fountain/retrieval/lua/module/entry_photo__prepare.lua") \
      .if_("fountain_gen_source_label == 1") \
        .enrich_attr_by_light_function(
          import_common_attr = [
            "colossus_photo_id_list",
            "colossus_play_time_list",
            "colossus_label_list",
            "colossus_timestamp_list",
            {"name": "featureSourcePId", "as": "source_pid"},
          ],
          export_common_attr = [
            "source_playtime_s",
            "source_is_interacted",
          ],
          function_name = "ExtractSourceLabel",
          class_name = "ExploreLightFunctionSetV2",
        ) \
      .end_()