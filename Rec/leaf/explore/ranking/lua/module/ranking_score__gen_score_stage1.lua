function gen_score_stage1()
    local report_discount_new = report_discount or 1.0
    local hate_discount_new = hate_discount or 1.0
    local pctr_new = corr_pctr or 0.0
    local ctr_power = explore_fr_ctr_power or 1.0
    local pltr_new = pltr or 0.0
    local pwtr_new = corr_pwtr or 0.0
    local pftr_new = pftr or 0.0
    local pcmtr_new = pcmtr or 0.0
    local petcm_new = petcm or 0.0
    local pptr_new = pptr or 0.0
    local psvr_new = psvr or 0.0
    local pdtr_new = pdtr or 0.0
    local pepstr_new = pepstr or 0.0
    local pcltr_new = pcltr or 0.0
    local pcmef_new = pcmef or 0.0
    local phtr_new = phtr or 0.0
    local fr_score2_new = fr_score2 or 0.0
    local psvr_power = explore_fr_svr_power or 0.0
    pctr_new = pctr_new^ctr_power * (1.0 - psvr_new)^psvr_power
    local score_pctr = report_discount_new * pctr_new
    local score_pltr = report_discount_new * pltr_new * pctr_new
    local score_pwtr = report_discount_new * pwtr_new * pctr_new
    local score_pftr = report_discount_new * pftr_new * hate_discount_new
    local score_pcmtr = report_discount_new * pcmtr_new * pctr_new
    local score_petcm = report_discount_new * petcm_new * pctr_new
    local score_pptr = report_discount_new * pptr_new
    local score_pdtr = report_discount_new * pdtr_new
    local score_pepstr = report_discount_new * pepstr_new
    local score_pcltr = report_discount_new * pcltr_new
    local score_pcmef = pctr_new * pcmef_new
    local score_phtr = phtr_new
    local score_psvr = psvr_new
    local fr_enable_neg_queue_report_discount = fr_enable_neg_queue_report_discount or 0
    if (fr_enable_neg_queue_report_discount > 0) then
        score_phtr = report_discount_new * phtr_new
        score_psvr = report_discount_new * psvr_new
    end
    local score_pctr_x_psvr = pctr_new * psvr_new

    local fetr = fetr or 0.0
    local fountain_eff = fountain_eff or 0.0
    local fetr_feff_ctr_power = fetr_feff_ctr_power or 0.5
    local explore_fr_fetr_feff_power = explore_fr_fetr_feff_power or 1.0
    fetr = pctr_new^fetr_feff_ctr_power * fetr^explore_fr_fetr_feff_power;
    fountain_eff = pctr_new^fetr_feff_ctr_power * fountain_eff^explore_fr_fetr_feff_power;
    return score_pctr, score_pltr, score_pwtr, score_pftr, score_pcmtr, score_petcm, score_pptr, score_psvr, score_pdtr, score_pepstr, score_pcltr, score_pcmef, score_phtr, score_pctr_x_psvr, fetr, fountain_eff
end