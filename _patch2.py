import json

path = 'notebooks/01_ingest.ipynb'
nb = json.load(open(path, encoding='utf-8'))

new_source = [
    '# -- Verificar presença da coluna de target --\n',
    '# Normaliza fg_demitido_voluntario_Nm -> fg_status_vol_Nm se necessário\n',
    '_rename_target = {\n',
    '    c: c.replace("fg_demitido_voluntario_", "fg_status_vol_")\n',
    '    for c in df.columns\n',
    '    if c.startswith("fg_demitido_voluntario_")\n',
    '}\n',
    'if _rename_target:\n',
    '    df = df.rename(columns=_rename_target)\n',
    '    print(f"  Renomeadas {len(_rename_target)} colunas fg_demitido_voluntario_ -> fg_status_vol_")\n',
    '\n',
    'target_cols = [c for c in df.columns if "fg_status_vol" in c.lower()]\n',
    'print(f"Colunas de target encontradas: {target_cols}")\n',
    '\n',
    'if COL_TARGET not in df.columns:\n',
    '    candidates = [c for c in df.columns if "fg_status_vol" in c]\n',
    '    if candidates:\n',
    '        COL_TARGET = candidates[0]\n',
    '        print(f"\\u26a0 Target ajustado para: {COL_TARGET}")\n',
    '    else:\n',
    '        raise ValueError(f"Coluna de target nao encontrada. Colunas: {list(df.columns)}")\n',
    '\n',
    'print(f"\\nDistribuicao do target ({COL_TARGET}):")\n',
    'print(df[COL_TARGET].value_counts(dropna=False).to_string())\n',
    'print(f"\\nTaxa de turnover: {df[COL_TARGET].mean():.2%}")\n',
]

for i, c in enumerate(nb['cells']):
    if c.get('cell_type') == 'code' and 'Verificar presen' in ''.join(c['source']):
        nb['cells'][i]['source'] = new_source
        print(f'Célula {i} atualizada.')
        break

json.dump(nb, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('Notebook salvo.')
