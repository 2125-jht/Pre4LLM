from ranking import CommonModule
from ranking.fountain_ranking_queues import fullrank_ensemble_filter_queues

class FountainRankingStageOneTruncateModule(CommonModule):
  def __init__(self, name: str) -> None:
    super().__init__(name)
  
  def process(self) -> None:
    self.flow \
      .count_reco_result(
        save_count_to="fountain_fullrank_result_count_before_stage1"
      ) \
      .explore_ensemble_filter_score_enricher(
        queues = fullrank_ensemble_filter_queues,
        filter_function = "{{fountain_ensemble_filter_function}}",
        score_with_rank = "{{fountain_ensemble_filter_score_with_rank}}",
        save_score_to_attr = "fullrank_ensemble_filter_score",
      ) \
      .sort(
        score_from_attr = "fullrank_ensemble_filter_score",
        stable_sort = True,
        desc = False
      ) \
      ._dump_attr_to_kafka( # filter 截断之前, 将全部item的重要 item attr 落盘
        stage_name = "fr_s1_score",
        dump_item_attr_list = [
          "fullrank_sim_pcmtr",
          "fullrank_sim_pfintr",
          "fullrank_sim_pevtr",
          "fullrank_sim_pltr",
          "fullrank_sim_pwtr",
          "fullrank_detail_new_pevtr_v2",
          "fullrank_sim_pvtr",
          "fullrank_ensemble_filter_score",
        ]
      ) \
      .gen_common_attr_by_lua(
        attr_map={
          "fullrank_stage1_limit_size": "math.floor(fountain_fullrank_result_count_before_stage1 * (1 - fountain_ensemble_filter_coeff))",
        }
      ) \
      .limit(
        name = "fountain_fr_stage1",
        traceback = True,
        size = "{{fullrank_stage1_limit_size}}"
      )

  def post_process(self) -> None:
    self.flow \
      .log_debug_info(
        item_attrs = [
          "fullrank_ensemble_filter_score"
        ],
        for_debug_request_only = True
      )

