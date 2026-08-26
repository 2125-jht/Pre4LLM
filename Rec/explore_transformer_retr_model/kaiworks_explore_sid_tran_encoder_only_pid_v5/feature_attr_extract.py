import json
"""
特征属性提取模块
功能：从特征池中提取模型所需的特征，并配置特征的维度、共享策略等属性
主要用于深度学习模型的特征工程和特征管理
"""

# ================================
# 特征配置定义区域
# ================================

# 本次模型使用的所有特征，需要从feature pool抽取
# 特征配置参数说明：
# - dim: 特征嵌入维度，需要用户自定义，默认使用default_dim
# - share_id: 需要共享embedding的特征设置为相同的share id，具体值用户自定义
# - expand: list特征是否需要展开，如果自定义list长度则需要设置展开长度
default_dim = 128  # 默认特征嵌入维度

# 模型使用的所有特征配置
all_feats = {
    # === 用户基础特征 ===
    "user_id": {},                    # 用户ID，使用默认配置
    "user_gender": {},                # 用户性别
    "user_age_segment": {},           # 用户年龄段
    "user_level": {},                 # 用户等级
    
    # === 用户行为序列特征 ===
    # "user_profile_v1_click_pid_list": {"expand": 200},  # 用户点击视频ID列表，展开为128个元素
    # "user_profile_v1_click_aid_list": {"expand": 200},  # 用户点击作者ID列表，展开为128个元素
    
    "user_colossus_pid_list": {"expand": 1000},
    "user_colossus_aid_list": {"expand": 1000},

    # === 注释掉的视频相关特征（暂未使用） ===
    # "photo_id": {},                 # 视频ID
    # "photo_author_id": {},          # 视频作者ID

    # === 注释掉的视频作者统计特征（暂未使用） ===
    # "photo_author_fans_count": {"dim": 16},           # 作者粉丝数
    # "photo_author_fans_count_2": {"dim": 16},         # 作者粉丝数（版本2）
    # "photo_author_upload_count": {"dim": 16},         # 作者上传数
    # "photo_author_upload_count_2": {"dim": 16},       # 作者上传数（版本2）
    # "photo_author_click_count": {"dim": 16},          # 作者点击数
    # "photo_author_click_count_2": {"dim": 16},        # 作者点击数（版本2）
    # "photo_author_like_count": {"dim": 16},           # 作者点赞数
    # "photo_author_like_count_2": {"dim": 16},         # 作者点赞数（版本2）
    # "photo_author_follow_count": {"dim": 16},         # 作者关注数
    # "photo_author_follow_count_2": {"dim": 16},       # 作者关注数（版本2）
    # "photo_author_long_view_count": {"dim": 16},      # 作者长观看数
    # "photo_author_long_view_count_2": {"dim": 16},    # 作者长观看数（版本2）
    # "photo_author_emp_ctr": {"dim": 16},              # 作者经验点击率
    # "photo_author_emp_ltr": {"dim": 16},              # 作者经验点赞率
    # "photo_author_emp_wtr": {"dim": 16},              # 作者经验观看率
    # "photo_author_emp_lvtr": {"dim": 16},             # 作者经验长观看率
    # "photo_author_emp_svtr": {"dim": 16},             # 作者经验短观看率
    # "photo_author_emp_watch_time": {"dim": 16},       # 作者经验观看时长
}

# 特征复制配置：将某些特征复制为新特征，可设置不同的slot和维度
# 格式：原特征名: [(新特征名, 新slot, 新维度)]
copy_feats = {
    # 示例配置（已注释）：
    # "user_id": [("user_emb", 4006, 16)],             # 将user_id复制为user_emb
    # "photo_id": [
    #     ("photo_emb", 4103, 16),                     # 将photo_id复制为photo_emb
    # ],
    # "photo_author_id": [
    #     ("photo_author_id_v2", 4200),               # 将photo_author_id复制为新版本
    # ],
}

# ================================
# 特征池加载
# ================================

# 加载特征池配置文件，包含所有可用特征的元信息
feature_pool_config = json.load(open("./feature_pool.json", "r"))

# 推理时需要忽略的特征列表（这些特征在推理阶段不可用或不需要）
infer_ignore_feat = ["user_emb", "photo_emb", "last_step", "ave_step"]

# ================================
# 特征属性类定义
# ================================

class Attr:
    """
    特征属性类，用于存储每个特征的详细信息
    """
    def __init__(self, attr_name, slot, is_common, dim, expand):
        """
        初始化特征属性
        
        Args:
            attr_name (str): 模型中使用的特征名称
            slot (list): 特征对应的slot列表
            is_common (bool): 是否为通用属性
            dim (int): 特征嵌入维度
            expand (int): 序列特征的展开长度，None表示不展开
        """
        self.attr_name = attr_name   # 模型特征名称
        self.dim = dim              # 特征维度
        self.slots = slot           # 特征slot列表
        self.expand = expand        # 展开长度
        self.is_common = is_common  # 是否为通用属性

# ================================
# 特征属性提取主函数
# ================================

def get_all_feature_attrs(all_feats):
    """
    从特征配置中提取所有特征属性，并处理共享嵌入和特征复制
    
    Args:
        all_feats (dict): 特征配置字典
        
    Returns:
        tuple: (all_features, share_input_slots, share_output_slots, copy_input_slots, copy_output_slots)
            - all_features: 所有特征属性对象列表
            - share_input_slots: 需要共享嵌入的源slot列表
            - share_output_slots: 共享嵌入的目标slot列表
            - copy_input_slots: 需要复制的源slot列表
            - copy_output_slots: 复制的目标slot列表
    """
    all_features = []           # 存储所有特征属性
    all_share_id = {}          # 存储共享嵌入的分组信息
    share_input_slots = []     # 共享嵌入的输入slot
    share_output_slots = []    # 共享嵌入的输出slot
    copy_input_slots = []      # 特征复制的输入slot
    copy_output_slots = []     # 特征复制的输出slot
    
    # 遍历所有配置的特征
    for k in all_feats.keys():
        # 检查特征是否在特征池中存在
        if k in feature_pool_config.keys():
            # === 获取特征基本属性 ===
            slot = feature_pool_config[k].get("slot")                          # 获取slot
            is_common = feature_pool_config[k].get("use_common_attr_only", False)  # 是否使用通用属性
            dim = all_feats[k].get("dim", default_dim)                        # 获取维度，默认使用default_dim
            expand = all_feats[k].get("expand", None)                         # 获取展开配置
            
            # 创建特征属性对象
            all_features.append(Attr(k, [slot], is_common, dim, expand))
            
            # === 处理共享嵌入 ===
            share_id = all_feats[k].get("share_id", None)
            if share_id is not None:
                # 将具有相同share_id的特征分组
                if share_id in all_share_id:
                    all_share_id[share_id].append(slot)
                else:
                    all_share_id[share_id] = [slot]
            
            # === 处理特征复制 ===
            if k in copy_feats:
                for config in copy_feats[k]:
                    new_name = config[0]    # 新特征名
                    new_slot = config[1]    # 新slot
                    
                    # 记录复制关系
                    copy_input_slots.append(slot)
                    copy_output_slots.append(new_slot)
                    
                    # 如果指定了新维度，使用新维度；否则使用原维度
                    if len(config) > 2:
                        dim = config[2]
                    
                    # 创建复制后的特征属性对象
                    all_features.append(Attr(new_name, [new_slot], is_common, dim, expand))
                    print("--->>> {} copy from {}".format(new_name, k))
        else:
            # 特征不在特征池中，输出警告
            print(k, "feature is not in featue pool, please check!!!")

    # === 处理共享嵌入映射 ===
    # 共享嵌入策略：将具有相同share_id的特征映射到最小的slot上
    for k, v in all_share_id.items():
        v.sort()  # 按slot大小排序
        # 除了最小slot外，其他slot都需要映射到最小slot
        share_input_slots += v[1:]                    # 需要共享的源slot
        share_output_slots += [v[0]] * (len(v) - 1)  # 映射到的目标slot（最小slot）

    return all_features, share_input_slots, share_output_slots, copy_input_slots, copy_output_slots

# ================================
# 执行特征属性提取
# ================================

# 提取所有特征属性和映射关系
all_features, share_input_slots, share_output_slots, copy_input_slots, copy_output_slots = get_all_feature_attrs(all_feats)

# 输出共享嵌入配置信息
print("=== 共享嵌入配置 ===")
print("共享输入slots:", share_input_slots)
print("共享输出slots:", share_output_slots)

# 输出特征复制配置信息
print("=== 特征复制配置 ===")
print("复制输入slots:", copy_input_slots)
print("复制输出slots:", copy_output_slots)
