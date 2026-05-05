with open('notebooks/10_monitor.ipynb', encoding='utf-8') as f:
    lines = f.readlines()

changed = 0
for i, line in enumerate(lines):
    if '0.1276' in line and 'REF_RATES' in line:
        lines[i] = line.replace('0.1276', '0.1235').replace('0.1362', '0.1027').replace('0.1470', '0.1258')
        changed += 1

with open('notebooks/10_monitor.ipynb', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f'Changed {changed} line(s)')
