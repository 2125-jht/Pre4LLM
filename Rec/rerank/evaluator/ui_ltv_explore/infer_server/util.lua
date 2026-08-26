function gen_item_attr()
    local pctr_list = pctr_list or {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    local pltr_list = pltr_list or {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    local pwtr_list = pwtr_list or {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    local plvtr_list = plvtr_list or {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    local pcmtr_list = pcmtr_list or {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    local pcmef_list = pcmef_list or {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    local pptr_list = pptr_list or {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    local pctr_index_list = pctr_index_list or {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    local pltr_index_list = pltr_index_list or {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    local pwtr_index_list = pwtr_index_list or {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    local pvtr_index_list = pvtr_index_list or {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    local plvtr_index_list = plvtr_index_list or {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    local emp_ctr_list = emp_ctr_list or {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    local emp_ltr_list = emp_ltr_list or {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    local emp_wtr_list = emp_wtr_list or {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    local emp_lvtr_list = emp_lvtr_list or {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}

    local pctr_index_list_double = {}
    for i = 1, #pctr_index_list do
        pctr_index_list_double[i] = pctr_index_list[i] + 0.0
    end
    local pltr_index_list_double = {}
    for i = 1, #pltr_index_list do
        pltr_index_list_double[i] = pltr_index_list[i] + 0.0
    end
    local pwtr_index_list_double = {}
    for i = 1, #pwtr_index_list do
        pwtr_index_list_double[i] = pwtr_index_list[i] + 0.0
    end
    local pvtr_index_list_double = {}
    for i = 1, #pvtr_index_list do
        pvtr_index_list_double[i] = pvtr_index_list[i] + 0.0
    end
    local plvtr_index_list_double = {}
    for i = 1, #plvtr_index_list do
        plvtr_index_list_double[i] = plvtr_index_list[i] + 0.0
    end


    return pctr_list, pltr_list, pwtr_list, plvtr_list, pcmtr_list, pcmef_list, pptr_list,
        pctr_index_list_double,pltr_index_list_double,pwtr_index_list_double, pvtr_index_list_double, plvtr_index_list,
        emp_ctr_list, emp_ltr_list, emp_wtr_list, emp_lvtr_list
end

function gen_return_attr()
    local pctr = pctr or 0.0
    local pltr = pltr or 0.0
    local pwtr = pwtr or 0.0

    local ctr = ctr or 0.0
    local cvr = cvr or 0.0
    local click = click or 0.0
    local action = action or 0.0

    return ctr*cvr, cvr
end
