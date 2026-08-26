function upload_time_filer()
    local is_filter = 0
    local upload_time_new = upload_time or 0.0
    local long_term_photo_new = long_term_photo or 0
    local normal_photo_life_time_hours = normal_photo_life_time_hours or 26
    local time_bound = (os.time() - normal_photo_life_time_hours * 60 * 60) * 1000
    if (upload_time_new < time_bound and long_term_photo_new < 1) then
        is_filter = 1
    end

    local enable_follow_author_retr_skip_26h_filter = enable_follow_author_retr_skip_26h_filter or 0
    if (enable_follow_author_retr_skip_26h_filter == 1 and is_follow_author and is_follow_author == 1) then
        is_filter = 0
    end

    return is_filter
  end