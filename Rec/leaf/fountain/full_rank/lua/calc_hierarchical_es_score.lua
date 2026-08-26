function calc_hierarchical_es_score()
  local score = fullrank_ensemble_score and fullrank_ensemble_score or 0.0
  local seq_score = 1.0 / (fullrank_ensemble_rank + fullrank_ensemble_seq_rank_smooth)
  if (fountain_fullrank_xtr_ensemble_fusion_way == 1) then
    score = (seq_score^fountain_fullrank_hierarchical_es_es_seq_score_pow_weight)
                      * (fullrank_act_raw_score^fountian_fullrank_xtr_raw_score_pow_weight)
                      * (fullrank_wt_rank_score^fountain_fullrank_hierarchical_es_wt_rank_score_pow_weight)
                      * (fullrank_act_rank_score^fountain_fullrank_hierarchical_es_act_rank_score_pow_weight)
                      * (fullrank_vv_rank_score^fountain_fullrank_hierarchical_es_vv_rank_score_pow_weight)
  end
  return score
end