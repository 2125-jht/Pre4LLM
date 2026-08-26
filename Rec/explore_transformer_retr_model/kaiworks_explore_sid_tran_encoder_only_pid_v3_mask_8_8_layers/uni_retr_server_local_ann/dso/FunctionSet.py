from dragonfly.matx.dragonfly_context import *




class FunctionSet:
  def __init__(self) -> None:
    pass

  def extract_sid_list_func(self, ctx: DragonflyContext) -> None:
    sid_list_sorted = ctx.GetIntList(b"sid_list_sorted")
    sid_limit_str = ''
    std_end_index = 20
    if len(sid_list_sorted) < std_end_index:
      sid_end_index = len(sid_list_sorted)
    total_len = 0
    extract_sid_list: FTList[int] = list()
    for i in range(len(sid_list_sorted)):
      extract_sid_list.append(sid_list_sorted[i])
      total_len = i
    ctx.SetIntList(b"extract_sid_list", extract_sid_list)
    ctx.SetInt(b"extract_sid_list_length", len(extract_sid_list))

  def processUserEmb(self, ctx: DragonflyContext) -> None:
    # Common Attr 处理逻辑
    user_emb = ctx.GetIntList(b"user_emb")
    loc = ctx.GetIntList(b"loc")
    user_emb_new: FTList[int] = list()
    for i in range(len(user_emb)):
      emb = user_emb[i]
      for j in range(3):
        user_emb_new.append((emb << 8) + loc[j])

    ctx.SetIntList(b'user_emb_append', user_emb_new)
