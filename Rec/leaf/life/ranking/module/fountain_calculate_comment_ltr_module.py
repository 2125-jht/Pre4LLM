from ranking import CommonModule

class FountainCalculateCommentLtrModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .item_attr_enrich_from_redis_json(
        skip = "{{fountain_skip_cal_enrich_comment_ltr_item_attr_from_redis}}",
        cluster_name="mmuNlpComment",
        redis_key_prefix="nlp_photo_analysis",
        timeout_ms=30,
        json_queues=[
          {
            "name":"avg_like_with_show",
            "type":"double",
            "output_attr":"photo_avg_like_with_show"
          },
          {
            "name":"max_like_with_show",
            "type":"double",
            "output_attr":"photo_max_like_with_show"
          },
          {
            "name":"like_cnt",
            "type":"double",
            "output_attr":"photo_like_cnt"
          },
          {
            "name":"show_cnt",
            "type":"double",
            "output_attr":"photo_show_cnt"
          },
          {
            "name":"comment_cnt",
            "type":"double",
            "output_attr":"photo_comment_cnt"
          },
          {
            "name":"author_like_cnt",
            "type":"double",
            "output_attr":"photo_author_like_cnt"
          },
          {
            "name":"update_7d_cnt",
            "type":"double",
            "output_attr":"photo_update_7d_cnt"
          },
          {
            "name":"first_comment_cnt",
            "type":"double",
            "output_attr":"photo_first_comment_cnt"
          },
          {
            "name":"second_comment_cnt",
            "type":"double",
            "output_attr":"photo_second_comment_cnt"
          },
          {
            "name":"author_reply_cnt",
            "type":"double",
            "output_attr":"photo_author_reply_cnt"
          },
          {
            "name":"author_show_cnt",
            "type":"double",
            "output_attr":"photo_author_show_cnt"
          },
          {
            "name":"god_cnt",
            "type":"double",
            "output_attr":"photo_god_cnt"
          },
          {
            "name":"hot_cnt",
            "type":"double",
            "output_attr":"photo_hot_cnt"
          },
          {
            "name":"pre_god_cnt",
            "type":"double",
            "output_attr":"photo_pre_god_cnt"
          },
          {
            "name":"pre_hot_cnt",
            "type":"double",
            "output_attr":"photo_pre_hot_cnt"
          },
          {
            "name":"only_at_cnt",
            "type":"double",
            "output_attr":"photo_only_at_cnt"
          },
          {"name":"emoji_cnt",
            "type":"double",
            "output_attr":"photo_emoji_cnt"
          },
          {
            "name":"kmoji_cnt",
            "type":"double",
            "output_attr":"photo_kmoji_cnt"
          },
          {
            "name":"only_punctuation_cnt",
            "type":"double",
            "output_attr":"photo_only_punctuation_cnt"
          },
          {
            "name":"only_num_cnt",
            "type":"double",
            "output_attr":"photo_only_num_cnt"
          },
          {"name":"yuqi_cnt",
            "type":"double",
            "output_attr":"photo_yuqi_cnt"
          },
          {
            "name":"qiugoumai_cnt",
            "type":"double",
            "output_attr":"photo_qiugoumai_cnt"
          },
          {
            "name":"qiuziyuan_cnt",
            "type":"double",
            "output_attr":"photo_qiuziyuan_cnt"
          },
          {
            "name":"qiuhudong_cnt",
            "type":"double",
            "output_attr":"photo_qiuhudong_cnt"
          },
          {
            "name":"zhuixing_cnt",
            "type":"double",
            "output_attr":"photo_zhuixing_cnt"
          },
          {
            "name":"aicheng_cnt",
            "type":"double",
            "output_attr":"photo_aicheng_cnt"
          },
          {
            "name":"zanshang_cnt",
            "type":"double",
            "output_attr":"photo_zanshang_cnt"
          },
          {
            "name":"feiwenben_cnt",
            "type":"double",
            "output_attr":"photo_feiwenben_cnt"
          }
        ]
      ) \
      .common_attr_enrich_from_redis_json(
        skip = "{{fountain_skip_cal_enrich_comment_ltr_common_attr_from_redis}}",
        cluster_name="mmuNlpComment",
        redis_key_prefix="user_analysis",
        timeout_ms=30,
        json_queues=[
          {
            "name":"comment_count",
            "type":"double",
            "output_attr":"user_comment_count"
          },
          {
            "name":"like_count",
            "type":"double",
            "output_attr":"user_like_count"
          },
          {
            "name":"show_count",
            "type":"double",
            "output_attr":"user_show_count"
          },
          {
            "name":"comment_count_3d",
            "type":"double",
            "output_attr":"user_comment_count_3d"
          },
          {
            "name":"like_count_3d",
            "type":"double",
            "output_attr":"user_like_count_3d"
          },
          {
            "name":"show_count_3d",
            "type":"double",
            "output_attr":"user_show_count_3d"
          },
          {
            "name":"comment_count_1d",
            "type":"double",
            "output_attr":"user_comment_count_1d"
          },
          {
            "name":"like_count_1d",
            "type":"double",
            "output_attr":"user_like_count_1d"
          },
          {
            "name":"show_count_1d",
            "type":"double",
            "output_attr":"user_show_count_1d"
          },
        ]
      ) \
      .item_xgb_kml_predict_enrich(
        skip = "{{fountain_skip_cal_kml_comment_ltr}}",
        model_name = "{{fountain_kml_comment_ltr_model_name}}",
        kml_service_name = "{{fountain_kml_comment_ltr_kml_service_name}}",
        common_attrs_oredr_kconf = "{{fountain_comment_ltr_common_attr_order}}",
        item_attrs_oredr_kconf = "{{fountain_comment_ltr_item_attr_order}}",
        output_attr = "comment_ltr",
        timeout_ms = 100,
        common_feature_attrs=["user_comment_count","user_like_count","user_show_count","user_comment_count_3d","user_like_count_3d","user_show_count_3d","user_comment_count_1d","user_like_count_1d","user_show_count_1d"],
        item_feature_attrs=["photo_avg_like_with_show","photo_max_like_with_show","photo_like_cnt","photo_show_cnt","photo_update_7d_cnt","photo_comment_cnt","photo_first_comment_cnt","photo_second_comment_cnt","photo_author_reply_cnt","photo_author_like_cnt","photo_author_show_cnt","photo_god_cnt","photo_hot_cnt","photo_pre_god_cnt","photo_pre_hot_cnt","photo_only_at_cnt","photo_emoji_cnt","photo_kmoji_cnt","photo_only_punctuation_cnt","photo_only_num_cnt","photo_yuqi_cnt","photo_qiugoumai_cnt","photo_qiuziyuan_cnt","photo_qiuhudong_cnt","photo_zhuixing_cnt","photo_aicheng_cnt","photo_zanshang_cnt","photo_feiwenben_cnt"]
      )
  
  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_attrs = [
          "comment_ltr",
        ],
        for_debug_request_only = True
      )
