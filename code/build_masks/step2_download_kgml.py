import os
import requests
import time

# 创建存放 KGML 的目录
KGML_DIR = "kegg_kgml"
os.makedirs(KGML_DIR, exist_ok=True)

# 1. 获取小鼠所有通路列表
print("正在获取小鼠通路列表...")
url_list = "https://rest.kegg.jp/list/pathway/mmu"
r = requests.get(url_list)
if r.status_code != 200:
    print(f"获取通路列表失败: {r.status_code}")
    exit()

lines = r.text.strip().split("\n")
print(f"共获取 {len(lines)} 个小鼠通路")

# 解析通路ID和名称
pathways = []
for line in lines:
    parts = line.split("\t")
    if len(parts) >= 2:
        pathway_id = parts[0].replace("path:", "")  # 如 mmu04910
        pathway_name = parts[1]
        pathways.append((pathway_id, pathway_name))

# 保存完整通路列表（便于后续参考）
with open(os.path.join(KGML_DIR, "mmu_pathway_list.txt"), "w", encoding="utf-8") as f:
    for pid, pname in pathways:
        f.write(f"{pid}\t{pname}\n")
print(f"完整通路列表已保存到 {KGML_DIR}/mmu_pathway_list.txt")

# 2. 筛选你需要下载的通路
# 你可以手动指定，也可以用关键词自动筛选
# 这里我推荐手动指定关键通路，确保覆盖你的研究
target_ids = [
    "mmu04910",  # Insulin signaling pathway (胰岛素信号通路) —— 核心
    "mmu04012",  # ErbB signaling pathway
    "mmu04150",  # mTOR signaling pathway
    "mmu04010",  # MAPK signaling pathway
    "mmu04066",  # HIF-1 signaling pathway
    "mmu04931",  # Insulin resistance
    "mmu00010",  # Glycolysis / Gluconeogenesis
    "mmu00020",  # Citrate cycle (TCA cycle)
    "mmu00030",  # Pentose phosphate pathway
    "mmu00051",  # Fructose and mannose metabolism
    "mmu00500",  # Starch and sucrose metabolism
    "mmu04152",  # AMPK signaling pathway
    "mmu04922",  # Glucagon signaling pathway
]

# 也支持通过关键词自动筛选（如果你不想手动指定，可以取消下面注释）
# keywords = ["insulin", "metabolism", "signaling", "pathway"]
# target_ids = []
# for pid, pname in pathways:
#     if any(kw in pname.lower() for kw in keywords):
#         target_ids.append(pid)
# 但注意这样可能会漏掉一些重要通路，建议手动指定

print(f"\n目标下载 {len(target_ids)} 个通路")

# 3. 下载 KGML 文件
downloaded = 0
failed = []
for pid in target_ids:
    url = f"https://rest.kegg.jp/get/{pid}/kgml"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            # 检查内容是否有效（KEGG 返回错误信息可能仍返回200）
            if r.text.strip().startswith("<!DOCTYPE") or "ERROR" in r.text[:100]:
                failed.append((pid, "返回HTML或错误信息"))
            else:
                filename = os.path.join(KGML_DIR, f"{pid}.kgml")
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(r.text)
                downloaded += 1
                print(f"  ✅ {pid} 下载成功")
        else:
            failed.append((pid, f"HTTP {r.status_code}"))
    except Exception as e:
        failed.append((pid, str(e)))
    time.sleep(0.3)  # 避免请求过快

print(f"\n下载完成：成功 {downloaded} 个，失败 {len(failed)} 个")
if failed:
    print("失败的通路:")
    for pid, err in failed:
        print(f"  {pid}: {err}")

# 4. 列出已下载文件
print(f"\n已下载的 KGML 文件列表（保存在 {KGML_DIR} 目录）：")
for f in sorted(os.listdir(KGML_DIR)):
    if f.endswith(".kgml"):
        print(f"  {f}")