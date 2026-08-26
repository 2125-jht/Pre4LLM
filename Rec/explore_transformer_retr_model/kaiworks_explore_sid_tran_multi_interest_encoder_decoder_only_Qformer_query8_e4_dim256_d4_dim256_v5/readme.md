- 多兴趣生成式模型，encoder 输入多个 query token 和一个 router special token，输出用户的多个 interest token；每个 interest token 再送入 fine_item_model，自回归生成 sid0、sid1、sid2
    - 训练时，不再使用 `cos(preference_embedding, interest)` 硬挑一个 interest 做 loss
    - 训练时，先让每个 interest head 都单独解码一遍 ground truth sid，得到每个 head 的 NLL
    - 用 `q(h|x,y)=softmax(-NLL_h / tau)` 构造 posterior teacher
    - 再用 router token 和全部 interest token 构造 prior router `p(h|x,H)`，蒸馏学习 posterior teacher
    - sid 生成 loss 使用 posterior teacher 对所有 head 的 CE 做软加权
    - 推理时，先用 prior router 选 top-k interest head，再对选中的 head 做 beam search 生成 sid 序列

- coarse_interest_model 使用 encoder 结构，layer=4
- fine_item_model 使用 decoder only 结构，layer=4
- query token numb=8
- 训练时多个 fine_item_model 参数 share，但通过 reshape 的方式并行计算每个 head 的 teacher forcing loss
- 推理时多个 head 的 fine_item_model 参数 share，默认先取 router top2 head，再做 beam search

- router 结构
    - encoder 输入为 `[query_tokens, router_token, user_static_token, user_click_tokens]`
    - router token 经过 encoder 后得到 `router_token_output`
    - `router_token_output` 分别和每个 `interest_embed` 做 concat 和交叉特征，经过共享 MLP 后输出每个 head 的 logit
    - 所有 head 的 logits 做 softmax，得到 prior router 概率

- 线上导出
    - `user_sid_origin`：生成出的 sid 序列
    - `user_sid_prob`：逐 token 概率
    - `query_indices`：每条生成序列对应的 interest head 索引
    - `query_probs`：对应 head 的 prior router 概率

- 当前默认推理配置
    - `head_top_k=2`
    - `beam_size=500`
    - 理论最大产出 `2 * 500 = 1000` 条候选 sid 序列，最终按 `router logprob + beam score` 排序

- input 是 sid0、sid1+8192、sid2+8192+8192，label 没有加上 8192，就是 sid0、sid1、sid2
    - 训练的时候输出 dense 层是各层局部词表范围 `0~8192`
    - 推理的时候输出 dense 层也是各层局部词表范围 `0~8192`
    - 在保存结果时会加上对应的 offset，所以最终导出的 sid 是 `sid0,sid1+8192,sid2+8192+8192`
