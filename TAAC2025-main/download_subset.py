"""
下载 TencentGR-1M 数据集的指定子集
使用 hf-mirror.com 镜像，支持断点续传
"""
import os
from huggingface_hub import list_repo_files, hf_hub_download

# 使用国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

REPO_ID = "TAAC2025/TencentGR-1M"
LOCAL_DIR = "./data/TencentGR-1M"

# 下载选项（True = 下载, False = 跳过）
DOWNLOAD_CONFIG = {
    "core": True,           # candidate + item_feat + seq + user_feat (~1 GB)
    "mm_emb_81_32": False,  # Bert, 32维 (~901 MB)
    "mm_emb_82_1024": False, # Conan, 1024维 (~9.4 GB)
    "mm_emb_83_3584": False, # gte-Qwen2, 3584维 (~31 GB)
    "mm_emb_84_4096": False, # hunyuan_mm, 4096维 (~30 GB)
    "mm_emb_85_3584": False, # QQMM, 3584维 (~31 GB)
    "mm_emb_86_3584": False, # UniME, 3584维 (~26 GB)
}

# 路径匹配规则
PATH_RULES = {
    "core": ["README.md", "indexer.pkl", "candidate/", "item_feat/", "seq/", "user_feat/"],
    "mm_emb_81_32": ["mm_emb/emb_81_32_parquet/"],
    "mm_emb_82_1024": ["mm_emb/emb_82_1024_parquet/"],
    "mm_emb_83_3584": ["mm_emb/emb_83_3584_parquet/"],
    "mm_emb_84_4096": ["mm_emb/emb_84_4096_parquet/"],
    "mm_emb_85_3584": ["mm_emb/emb_85_3584_parquet/"],
    "mm_emb_86_3584": ["mm_emb/emb_86_3584_parquet/"],
}


def should_download(filename: str) -> bool:
    """判断文件是否应该下载"""
    for key, enabled in DOWNLOAD_CONFIG.items():
        if not enabled:
            continue
        for prefix in PATH_RULES[key]:
            if filename.startswith(prefix) or filename == prefix.rstrip("/"):
                return True
    return False


def main():
    print("=" * 60)
    print("TencentGR-1M Subset Downloader")
    print("=" * 60)
    
    # 显示下载计划
    total_estimated_gb = 0.0
    print("\nDownload plan:")
    for key, enabled in DOWNLOAD_CONFIG.items():
        size_map = {
            "core": 1.0,
            "mm_emb_81_32": 0.9,
            "mm_emb_82_1024": 9.4,
            "mm_emb_83_3584": 31.0,
            "mm_emb_84_4096": 30.0,
            "mm_emb_85_3584": 31.0,
            "mm_emb_86_3584": 26.0,
        }
        size = size_map.get(key, 0)
        status = "YES" if enabled else "SKIP"
        total_estimated_gb += size if enabled else 0
        print(f"  [{status}] {key:20s} (~{size:5.1f} GB)")
    
    print(f"\nEstimated total: ~{total_estimated_gb:.1f} GB")
    print(f"Save to: {os.path.abspath(LOCAL_DIR)}")
    print("=" * 60)
    
    # 检查是否至少选了一项
    if not any(DOWNLOAD_CONFIG.values()):
        print("\nError: Nothing selected for download!")
        print("Please edit DOWNLOAD_CONFIG in this script to set at least one item to True.")
        return
    
    # 获取文件列表
    print("\nFetching file list from hf-mirror.com...")
    all_files = list(list_repo_files(REPO_ID, repo_type="dataset"))
    files_to_download = [f for f in all_files if should_download(f)]
    
    print(f"Total files in repo: {len(all_files)}")
    print(f"Files to download: {len(files_to_download)}")
    
    # 开始下载
    print("\nStarting download...")
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    for i, filename in enumerate(files_to_download, 1):
        print(f"[{i}/{len(files_to_download)}] {filename}")
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=filename,
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
