from retrieval.retrieval_module import RetrievalModule

class PicStRetrievalModule(RetrievalModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .if_("ft_pic_st_retr_delegate_retr_enable == 1") \
        .delegate_retrieve(
          kess_service="{{ft_pic_st_retr_delegate_retr_service}}",
          timeout_ms=50,
          request_type="fountain",
          request_num="{{ft_pic_st_retr_delegate_retr_num}}",
          send_common_attrs_in_request=False,
          send_common_attrs=[
            "userInfo",
            "uIsNicePicCsm"
          ],
        )\
      .else_()\
        .if_("ft_pic_st_retr_use_cache_emb == 1") \
          .switch_("ft_pic_st_retr_emb_server_mode") \
            .case_(1) \
              .get_remote_embedding_lite(
                kess_service = "{{ft_pic_st_retr_emb_service_name}}",
                timeout_ms = 10,
                query_source_type = "user_id",
                output_attr_name = "user_embedding_concat",
                id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
                slot = 999,
                size = 288,
              ) \
            .default_() \
              .get_remote_embedding_lite(
                kess_service = "{{ft_pic_st_retr_emb_service_name}}",
                timeout_ms = 10,
                query_source_type = "user_id",
                output_attr_name = "user_embedding_concat",
                id_converter = {"type_name": "kuibaEmbeddingIdConverter"},
                slot = 999,
                size = 256,
              ) \
          .end_()\
        .end_() \
        .if_("(user_embedding_concat == nil or #user_embedding_concat <= 0) and enable_ft_pic_st_retr_infer == 1") \
          .pack_common_attr(
            input_common_attrs = ["user_colossus_pic_play_pids"],
            output_common_attr = "featPlayPicPidLongTerm",
            limit_num = "{{ft_pic_st_retr_colossus_hisotry_num}}",
          ) \
          .pack_common_attr(
            input_common_attrs = ["user_colossus_pic_play_aids"],
            output_common_attr = "featPlayPicAidLongTerm",
            limit_num = "{{ft_pic_st_retr_colossus_hisotry_num}}",
          ) \
          .pack_common_attr(
            input_common_attrs = ["user_colossus_pic_play_rewards"],
            output_common_attr = "featPlayPicRewardLongTerm",
            limit_num = "{{ft_pic_st_retr_colossus_hisotry_num}}",
          ) \
          .pack_common_attr(
            input_common_attrs = ["user_colossus_pic_play_hetu_level1s"],
            output_common_attr = "featPlayPicHetu1LongTerm",
            limit_num = "{{ft_pic_st_retr_colossus_hisotry_num}}",
          ) \
          .delegate_enrich(
            kess_service = "{{ft_pic_st_retr_predict_service_name}}",
            recv_common_attrs = [
              {"name": "user_top_layer", "as": "user_embedding_concat"},
            ],
            timeout_ms = 40,
            send_common_attrs = [
              {"name": "_USER_ID_",   "as": "uId"},
              {"name": "_DEVICE_ID_", "as": "dId"},
              "featPlayPicPidLongTerm",
              "featPlayPicAidLongTerm",
              "featPlayPicRewardLongTerm",
              "featPlayPicHetu1LongTerm",
            ],
            request_type = "default"
          ) \
        .end_() \
        .if_("user_embedding_concat == nil or #user_embedding_concat == 0") \
          .return_() \
        .end_() \
        .gen_common_attr_by_lua(
          attr_map = {
            "uid_concat": "{_USER_ID_, _USER_ID_+1, _USER_ID_+2 ,_USER_ID_+3}",
          }
        ) \
        .retrieve_by_ann_embedding(
          reason = self.reason,
          kess_service = "{{ft_pic_st_retr_ann_service_name}}",
          space = "ip",
          timeout_ms = 50,
          items_from_attr=["uid_concat"],
          embeddings_from_attr=["user_embedding_concat"],
          bound_type={
            "top_k": "{{ft_pic_st_retr_ann_top_k}}"
          },
          algo_type={
            "scann": {},
          },
          src_bucket = "photo_src",
          dest_bucket = "{{ft_pic_st_retr_ann_dest_bucket}}",
          save_distance_to_attr = "ann_dist_list",
        ) \
        .if_("ft_pic_st_retr_ann_dist_threshold ~= nil and ft_pic_st_retr_ann_dist_threshold > 0.0", to_be_delete = "date=2024-05-29;committer=caozhong") \
          .enrich_attr_by_light_function(
            import_common_attr = [
              {"name": "ft_pic_st_retr_ann_dist_threshold", "as": "ann_dist_threshold"},
            ],
            import_item_attr = [
              "ann_dist_list",
            ],
            export_item_attr = [
              "ann_dist",
            ],
            function_name = "AnnCalThresholdValueForDistList",
            class_name = "ExploreLightFunctionSetV2",
          ) \
          .filter_by_attr(
            attr_name = "ann_dist",
            remove_if = "<",
            compare_to = "{{ft_pic_st_retr_ann_dist_threshold}}"
          ) \
        .end_()\
      .end_()
