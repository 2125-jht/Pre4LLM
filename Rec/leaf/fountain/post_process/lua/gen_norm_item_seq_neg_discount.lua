function calculate()
    local normalized_item_seq_neg_discount = 1.0
    if (item_seq_neg_discount ~= nil and item_seq_neg_discount >= 0) then
        normalized_item_seq_neg_discount = 1.0 / (item_seq_neg_discount + 10)
    end
    return normalized_item_seq_neg_discount
end
