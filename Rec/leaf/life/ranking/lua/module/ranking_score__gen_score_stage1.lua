function gen_score_stage1()
    local report_discount_new = report_discount or 1.0
    local hate_discount_new = hate_discount or 1.0
    local pctr_new = corr_pctr or 0.0
    local pltr_new = pltr or 0.0
    local pwtr_new = corr_pwtr or 0.0
    local pftr_new = pftr or 0.0
    local pcmtr_new = pctr_x_pcmtr or 0.0
    local pptr_new = pptr or 0.0
    local psvr_new = psvr or 0.0
    local pdtr_new = pdtr or 0.0
    local pepstr_new = pepstr or 0.0
    local pcltr_new = pcltr or 0.0
    local pcmef_new = pcmef or 0.0
    local phtr_new = phtr or 0.0
    local fr_score2_new = fr_score2 or 0.0
    local score_pctr = report_discount_new * pctr_new
    local score_pltr = report_discount_new * pltr_new * pctr_new
    local score_pwtr = report_discount_new * pwtr_new * pctr_new
    local score_pftr = report_discount_new * pftr_new * hate_discount_new
    local score_pcmtr = report_discount_new * pcmtr_new
    local score_pptr = report_discount_new * pptr_new
    local score_psvr = report_discount_new * psvr_new
    local score_pdtr = report_discount_new * pdtr_new
    local score_pepstr = report_discount_new * pepstr_new
    local score_pcltr = report_discount_new * pcltr_new
    local score_pcmef = pctr_new * pcmef_new
    local score_phtr = phtr_new * report_discount_new
    return score_pctr, score_pltr, score_pwtr, score_pftr, score_pcmtr, score_pptr, score_psvr, score_pdtr, score_pepstr, score_pcltr, score_pcmef, score_phtr
end