-- 为了做统一各种打分名称做的变换，主要是为了落 fullrank_detail_pxtr的特征
function fullrank_trans_pxtr()
  local pctr = fullrank_sim_pevtr and fullrank_sim_pevtr * 1.0 or 0.0
  local pltr = fullrank_sim_pltr and fullrank_sim_pltr * 1.0 or 0.0
  local pwtr = fullrank_sim_pwtr and fullrank_sim_pwtr * 1.0 or 0.0
  local pftr = fullrank_sim_pftr and fullrank_sim_pftr * 1.0 or 0.0
  local plvtr = fullrank_sim_plvtr and fullrank_sim_plvtr * 1.0 or 0.0
  local pvtr = fullrank_sim_pvtr and fullrank_sim_pvtr * 1.0 or 0.0
  local pout_ctr = fullrank_sim_out_pctr and fullrank_sim_out_pctr * 1.0 or 0.0
  local pcmtr = fullrank_sim_pcmtr and fullrank_sim_pcmtr * 1.0 or 0.0
  local pcmef = fullrank_sim_pcmef and fullrank_sim_pcmef * 1.0 or 0.0
  local pptr = fullrank_sim_pptr and fullrank_sim_pptr * 1.0 or 0.0
  local pepstr = fullrank_sim_pepstr and fullrank_sim_pepstr * 1.0 or 0.0
  local phtr = fullrank_sim_phtr and fullrank_sim_phtr * 1.0 or 0.0
  local lstr = fullrank_sim_lstr and fullrank_sim_lstr * 1.0 or 0.0
  local click_score = fullrank_sim_pevtr and fullrank_sim_pevtr * 1.0 or 0.0
  local like_score = fullrank_sim_pltr and fullrank_sim_pltr * 1.0 or 0.0
  local follow_score = fullrank_sim_pwtr and fullrank_sim_pwtr * 1.0 or 0.0
  local pcltr = fullrank_sim_pcltr and fullrank_sim_pcltr * 1.0 or 0.0
  local pwtd = fullrank_sim_pfintr and fullrank_sim_pfintr * 1.0 or 0.0

  return pctr, pltr, pwtr, pftr, plvtr, pvtr, pout_ctr, pcmtr, pcmef, pptr, pepstr, phtr, lstr, click_score, like_score, follow_score, pcltr, pwtd
end