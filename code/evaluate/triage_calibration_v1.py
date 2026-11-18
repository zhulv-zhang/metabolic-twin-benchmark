# 仿真KO_分诊器标定v1.py —— 33分诊器初版阈值标定
# 输入: 仿真KO_v6_多KO池化.csv
# 输出: 分层准确率表 + 候选规则扫描（只打印，不写文件）
import csv, collections

rows = list(csv.DictReader(open('仿真KO_v6_多KO池化.csv')))
for r in rows:
    r['hop'] = int(r['hop']); r['conflict'] = int(r['conflict'])
    r['ce'] = int(r['correct_endpoint']); r['cy'] = int(r['correct_early'])
    r['r'] = float(r['r_traj'])

def stat(sub):
    n = len(sub)
    if n == 0: return None
    return (n,
            round(sum(x['ce'] for x in sub)/n, 3),
            round(sum(x['cy'] for x in sub)/n, 3),
            round(sorted(x['r'] for x in sub)[n//2], 3))

print('=== A. 跳数 × 对消 分层 (n, 终点正确率, 早期正确率, r中位) ===')
groups = collections.defaultdict(list)
for r in rows: groups[(r['hop'], r['conflict'])].append(r)
for k in sorted(groups): print(k, stat(groups[k]))

print('=== B. 候选分诊规则扫描 ===')
rules = {
  'T1定量: hop1且conflict0': lambda r: r['hop']==1 and r['conflict']==0,
  'T2预警: conflict1任意hop': lambda r: r['conflict']==1,
  'T3不报: hop>=2':          lambda r: r['hop']>=2,
}
for name, f in rules.items():
    sub = [r for r in rows if f(r)]
    print(name, stat(sub), f'覆盖率={len(sub)/len(rows):.2f}')