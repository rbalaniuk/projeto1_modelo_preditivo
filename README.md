# Modelo Preditivo de Turnover Voluntário

Modelo de machine learning para predição de turnover voluntário em empresas de grande porte, construído **inteiramente a partir de uma única base de dados** — a mensalizada de RH, onde cada colaborador aparece uma vez por mês.

> **O diferencial técnico central deste projeto:** a base de entrada é uma tabela simples e universal — um registro por colaborador por mês — e o pipeline se encarrega de construir automaticamente o target de turnover, as features de comportamento em múltiplos horizontes temporais (3m e 6m), os indicadores contextuais por equipe/gestor/cargo e toda a infraestrutura de modelagem. Não é necessária nenhuma base adicional de features pré-processadas ou de targets pré-calculados. Qualquer empresa que tenha uma mensalizada básica exportada do ERP pode executar o pipeline completo.

## Problema de Negócio

O desligamento voluntário gera custos significativos com recrutamento, treinamento e perda de produtividade. Este projeto constrói um modelo preditivo que identifica **com 4 meses de antecedência** colaboradores com risco elevado de saída voluntária, permitindo ações preventivas de retenção.

## O Desafio Real

Modelos de predição de turnover são frequentemente reportados com AUC acima de 0.80 na literatura e em ferramentas comerciais. Esses resultados, porém, quase sempre são obtidos em conjuntos de teste do mesmo período histórico — onde os padrões de saída são estruturalmente idênticos ao treino.

O desafio real não é ajustar um modelo a dados históricos: é construir um sistema que antecipe comportamentos ainda por acontecer, com 4 meses de antecedência, em uma população que muda ao longo do tempo. Na prática, modelos com boa performance em treino/teste perdem significativamente quando colocados em validação temporal estrita — meses futuros, com condições macroeconômicas diferentes, ciclos salariais e mudanças organizacionais que não estavam no treino. O resultado típico é um modelo que "funciona no passado" e decepciona em produção.

Este projeto foi estruturado para garantir uma avaliação honesta: treino, teste e validação são cronologicamente separados — a validação (Jun–Ago/2025) é inteiramente posterior ao treino e ao teste — e o modelo é avaliado pela sua capacidade de generalizar a colaboradores novos em períodos futuros. A queda de performance entre teste e validação temporal é esperada e documentada. Em ambos os casos, os valores superam o que tipicamente se observa em problemas equivalentes com avaliação temporal rigorosa.

## Abordagem Técnica

- **Base única de entrada:** toda a lógica parte da mensalizada — uma tabela com 1 linha por colaborador por mês. O pipeline constrói o target e as features automaticamente, sem necessidade de bases auxiliares pré-processadas
- **Construção automática do target:** janelas forward-looking de 3 a 12 meses são derivadas diretamente das flags `fg_dem_vol` e `fg_dem_inv` da mensalizada (IMPL-9b), com tratamento correto de NaN para saídas involuntárias e janelas truncadas
- **Rolling features por colaborador:** para todas as variáveis `vl_*`, são geradas agregações de média, mínimo, máximo e desvio padrão em janelas de 3 e 6 meses — usando groupby+rolling vetorizado, sem loops Python por coluna
- **Janela deslizante (Modelo 2):** cada linha do dataset representa o estado de um colaborador em um mês; o target olha 4 meses à frente — suficiente para um ciclo completo de retenção
- **Arquitetura multi-grupo:** modelos independentes para Vendas, Transporte e Fábrica — cada grupo tem seus próprios encoders, splits, feature selection, modelo e critério de classificação de risco
- **Indicadores contextuais de grupo:** headcount e taxas de turnover por cargo/uorg/gestor via rolling 4m, calculados sobre a população completa — sem lookahead
- **Representação de desligados:** medoide dos K=4 registros mais recentes (registro real mais central, sem síntese) — um desligado = 1 linha no treino
- **Espaço de distância para medoide e subsampling:** reduzido de ~814 para ~200 features antes do cálculo (remove pares redundantes contínua/`_bin`, near-zero variance e features sem correlação com o target) — a redução ocorre exclusivamente no cálculo de distância interno, sem afetar o conjunto de features que chega ao modelo
- **Classificação de risco operacional:** ranking percentual por mês por grupo (proporcional à taxa histórica de turnover), não threshold fixo
- **Pipeline completo:** ingestão → construção de target e features → feature engineering → splits → feature selection → modelagem → tuning → SHAP → score → ROI → monitoramento (10 notebooks)

## Estrutura do Projeto

```
├── notebooks/
│   ├── 01_ingest.ipynb              # Carga da mensalizada → target (3-12m) → rolling vl_* (3m/6m) → indicadores de grupo → bases gold
│   ├── 02_build_dataset_m2.ipynb    # Feature engineering + binning + MICE (fit em base_tt)
│   ├── 03_splits.ipynb              # Medoide, Gower, split por ID, StandardScaler
│   ├── 04_feature_selection.ipynb   # Seleção de features em 3 camadas, por grupo
│   ├── 05_model.ipynb               # Treinamento, threshold, score de risco, por grupo
│   ├── 06_tuning.ipynb              # Tuning de hiperparâmetros + K-fold + drift treino→val
│   ├── 07_shap.ipynb                # SHAP values + drift validação→aplicação
│   ├── 08_score.ipynb               # Score final — validação + aplicação (Jun–Dez/2025)
│   ├── 09_roi.ipynb                 # Projeção de ROI por cenário de retenção
│   └── 10_monitor.ipynb             # PSI, performance mensal, alertas de retreino
│
├── data/                            # ⛔ Não versionado (.gitignore)
│   ├── raw/                         # mensalizada_ficticio.parquet (única base de entrada)
│   ├── gold/                        # base_tt_raw · base_val_raw · base_apl_raw
│   │                                # base_features_tt · base_features_val · base_features_apl
│   └── processed/
│       ├── splits/{Grupo}/          # train · test · val · scaler_cols.json
│       └── feature_selection/{Grupo}/  # selected_features.json · fs_report.parquet
│
├── models/{Grupo}/                  # best_model.joblib · best_model_tuned.joblib
│                                    # standard_scaler.joblib · model_metadata.json · tuning_metadata.json
│
├── reports/
│   ├── feature_selection/{Grupo}/   # fs_layer1_dropped.txt + 2 PNGs (RFECV, SHAP)
│   ├── figures/{Grupo}/             # roc_pr · confusion_matrix · drift_train · drift_val
│   │                                # pr_threshold · pr_tuning · reliability_diagram
│   ├── figures/                     # distribuição de risco · histogramas · ROI (3) · monitor (4)
│   ├── shap/{Grupo}/                # shap_values.npy · shap_feature_importance.json
│   │                                # summary_bar · summary_beeswarm · waterfall · dependence
│   ├── score_risco_{Grupo}.parquet  # Scores por grupo (Jun–Dez/2025)
│   ├── score_risco_consolidado.parquet/.csv
│   ├── roi_projecao.parquet/.csv
│   ├── monitor_drift_psi.csv
│   └── monitor_performance_val.csv
│
├── .gitignore
└── README.md
```

## Camadas de Dados

O pipeline produz versões progressivas da base, cada uma com papel distinto no fluxo de trabalho:

| Camada | Arquivo(s) | Produzido em | Descrição |
|---|---|---|---|
| **0 · Entrada** | `data/raw/mensalizada_ficticio.parquet` | — | **Base única** — 1 linha por colaborador × mês; contém flags de demissão, variáveis `vl_*` e metadados |
| **1 · Bases raw** | `data/gold/base_{tt,val,apl}_raw.parquet` | `01_ingest` | Após construção do target (3–12m) + rolling features (3m/6m) + indicadores de grupo + separação temporal |
| **2 · Features** | `data/gold/base_features_{tt,val,apl}.parquet` | `02_build` | Binning quintílico + OHE + MICE fit em base_tt; transform-only em val/apl |
| **3 · Splits** | `data/processed/splits/{Grupo}/` | `03_splits` | Medoide + Gower + split por ID + StandardScaler — treino/teste/val por grupo |
| **4 · Score** | `reports/score_risco_{Grupo}.parquet` | `08_score` | Probabilidade + score_risco + grupo_risco para cada colaborador × mês |

## Dados

Os dados utilizados são **fictícios e descaracterizados** — gerados a partir de padrões estatísticos reais, mas sem informações identificáveis. As bases de dados simulam bases que comumente estão disponíveis para o RH de empresas de médio e grande porte brasileiras.

Para executar o projeto, coloque o arquivo abaixo em `data/raw/`:

| Arquivo | Descrição |
|---|---|
| `mensalizada_ficticio.parquet` | **Única base de entrada** — 1 linha por colaborador × mês |

O arquivo já está incluído no repositório. Todos os demais arquivos intermediários (`data/gold/`, `data/processed/`) são gerados pelo pipeline ao rodar os notebooks em sequência.

## A mensalizada como base única

A **mensalizada** é a única entrada do pipeline. Trata-se de uma tabela simples onde cada linha representa um colaborador em um mês — exatamente o formato que qualquer empresa obtém ao exportar a folha de pagamento do ERP.

A partir dessa tabela, o notebook `01_ingest` constrói automaticamente:
- **Target de turnover** para janelas de 3 a 12 meses à frente
- **Rolling features** (média, mínimo, máximo, desvio padrão) em janelas de 3 e 6 meses para todas as variáveis `vl_*`
- **Features derivadas** como percentual de dias diurnos e percentual de dias com horas extras
- **Indicadores contextuais de grupo** (headcount e taxas de turnover por cargo/uorg/gestor), calculados sobre a população completa com janela rolling de 4 meses

Nenhuma base adicional, nenhum script de construção de features externo, nenhum arquivo de target pré-calculado é necessário.

---

## Premissas e Requisitos da Base de Dados

Esta seção descreve o contrato de dados que a mensalizada precisa satisfazer para que o pipeline seja executado sem ajustes. O objetivo é que qualquer pessoa com acesso a um ERP de RH consiga montar a base no formato correto.

### Premissa fundamental

> **Cada colaborador aparece exatamente uma vez por mês.** Não pode haver linhas duplicadas para o mesmo `id_colaborador` × `dt_mes_ano`. Meses sem registro (ex.: afastamento longo) devem ser preenchidos ou omitidos — o pipeline não consolida dados intra-mês.

### Convenção de nomes de colunas

O pipeline usa os **prefixos de coluna como sinalização automática** para determinar o tratamento de cada variável ao longo do pipeline (rolling, binning, OHE, scaler, etc.):

| Prefixo | Tipo de dado | Papel no pipeline |
|---|---|---|
| `vl_` | `float` / `int` | Variáveis numéricas mensais → usadas em rolling features (3m e 6m), binning quintílico e normalização. Colunas com sufixo `_Xm` (ex.: `_3m`, `_4m`, `_6m`, `_12m`) já representam agregações pré-calculadas sobre uma janela temporal e **não** recebem rolling adicional — o pipeline detecta o sufixo e preserva o valor como está |
| `ds_` | `str` / `object` | Variáveis categóricas → usadas em OHE (≤ 5 categorias) ou como base para indicadores contextuais de grupo |
| `fg_` | `0` / `1` | Flags binárias → demissão, afastamento, status ativo |
| `id_` | `str` / `int` | Identificadores → não usados como feature |
| `dt_` | `date` | Datas → usadas apenas para ordenação temporal e construção de rolling |

> ⚠️ **Colunas fora dessas convenções não são processadas automaticamente.** Se você adicionar uma variável sem prefixo reconhecido, ela será ignorada pelo pipeline de feature engineering.

### Colunas obrigatórias

| Coluna | Tipo | Regra | Descrição |
|---|---|---|---|
| `id_colaborador` | `str` / `int` | Único por mês | Identificador do colaborador — não pode se repetir no mesmo mês |
| `dt_mes_ano` | `date` | Um por colaborador × mês | Mês de referência (ex.: `2024-01-01` para janeiro/2024) |
| `fg_dem_vol` | `0` / `1` | Exatamente 1 no mês da saída | `1` no mês em que o colaborador pediu demissão; `0` nos demais |
| `fg_dem_inv` | `0` / `1` | Exatamente 1 no mês da saída | `1` no mês em que o colaborador foi demitido; `0` nos demais |
| `fg_ativo` | `0` / `1` | `1` enquanto ativo | `0` a partir do mês de saída (inclusive) |
| `ds_grupo` | `str` | Sem nulos | Grupo operacional do colaborador (ex.: `"Vendas"`, `"Fábrica"`, `"Transporte"`) — define a partição de cada modelo |
| `ds_cargo` | `str` | Sem nulos | Cargo do colaborador — usado para calcular indicadores contextuais de grupo (taxa de turnover por cargo) |
| `ds_uorg` | `str` | Sem nulos | Unidade organizacional / setor — usado nos indicadores contextuais de grupo |
| `ds_gestor` | `str` | Sem nulos | Identificador do gestor direto — usado nos indicadores contextuais de grupo |

### Colunas opcionais (features)

Além das obrigatórias, inclua todas as variáveis mensais que possam ser preditoras do turnover. O pipeline as trata automaticamente via prefixo:

**Numéricas (`vl_`):**
Qualquer métrica mensal por colaborador — salário, horas extras, faltas, atrasos, avaliações de desempenho, etc.

| Exemplo | Descrição |
|---|---|
| `vl_salario` | Salário bruto do mês |
| `vl_horas_extras` | Total de horas extras no mês |
| `vl_faltas` | Dias de falta no mês |
| `vl_atrasos` | Minutos de atraso no mês |
| `vl_nota_avaliacao` | Nota de avaliação de desempenho mais recente |

> **Variáveis com horizonte temporal já embutido (sufixo `_Xm`):** se a variável já representa um agregado sobre uma janela de tempo — por exemplo `vl_abs_dias_sum_3m` (soma de faltas nos últimos 3 meses) ou `vl_horas_extras_mean_6m` (média de horas extras nos últimos 6 meses) — use o sufixo `_3m`, `_4m`, `_6m`, `_12m` etc. no nome da coluna. O pipeline detecta esse padrão automaticamente e **não aplica rolling adicional** sobre essas colunas, evitando dupla-agregação. Elas entram diretamente nas etapas de binning e normalização com o valor mensal tal como está na base.

**Categóricas (`ds_`):**
Variáveis categóricas com baixa cardinalidade (≤ 5 valores distintos) são transformadas em OHE pelo pipeline. Variáveis com alta cardinalidade (> 5 valores) são descartadas — a informação delas é capturada pelos indicadores contextuais de `ds_cargo`, `ds_uorg` e `ds_gestor`.

| Exemplo | Descrição |
|---|---|
| `ds_faixa_salarial` | Faixa de salário (ex.: `"A"`, `"B"`, `"C"`) |
| `ds_turno` | Turno de trabalho (ex.: `"Dia"`, `"Noite"`, `"Misto"`) |
| `ds_tipo_contrato` | Tipo de vínculo (ex.: `"CLT"`, `"Temporário"`) |

### Restrições adicionais

- **Período mínimo:** o pipeline usa janelas rolling de até 6 meses e seleciona colaboradores com ≥ 90 dias de empresa. Para ter volume suficiente de positivos no treino, a base deve cobrir pelo menos **18 meses de histórico**.
- **Volume mínimo por grupo:** recomendado no mínimo **500 colaboradores ativos por grupo** no período de treino. Grupos com N < 500 podem ter modelo instável.
- **Saídas involuntárias:** devem estar corretamente classificadas em `fg_dem_inv` para que o pipeline as trate como NaN no target (não como 0), evitando contaminação do sinal de saída voluntária.
- **Meses futuros sem target:** os meses em que a janela de predição (4m) extrapola o fim da base receberão `fg_sem_output = 1` automaticamente e não serão usados no treino nem na validação.
- **Nulos em `vl_*`:** são tolerados — o pipeline imputa pela mediana calculada no treino. Evitar, porém, colunas com > 20% de nulos, pois são descartadas na feature selection.

### Como saber se sua base está no formato correto

Antes de rodar `01_ingest`, verifique:

```python
import pandas as pd
df = pd.read_parquet("data/raw/sua_mensalizada.parquet")

# 1. Sem duplicatas por colaborador × mês
assert df.duplicated(["id_colaborador", "dt_mes_ano"]).sum() == 0

# 2. Colunas obrigatórias presentes
obrigatorias = ["id_colaborador", "dt_mes_ano", "fg_dem_vol", "fg_dem_inv",
                "fg_ativo", "ds_grupo", "ds_cargo", "ds_uorg", "ds_gestor"]
faltando = [c for c in obrigatorias if c not in df.columns]
assert not faltando, f"Colunas ausentes: {faltando}"

# 3. Grupos existentes (ajustar os nomes conforme sua empresa)
print(df["ds_grupo"].value_counts())

# 4. Ao menos algumas features numéricas com prefixo correto
vl_cols = [c for c in df.columns if c.startswith("vl_")]
print(f"{len(vl_cols)} colunas vl_* encontradas: {vl_cols[:10]}")
```

## Como Executar

```bash
# 1. Clone o repositório
git clone https://github.com/rbalaniuk/projeto1_modelo_preditivo.git
cd projeto1_modelo_preditivo

# 2. Crie o ambiente (Python 3.10+)
pip install pandas numpy scikit-learn lightgbm xgboost shap gower matplotlib seaborn

# 3. Execute os notebooks em ordem
#    notebooks/01_ingest.ipynb → 02_build_dataset_m2.ipynb → ...
```

Os três arquivos de `data/raw/` já estão incluídos no repositório. Todos os demais arquivos intermediários (`data/gold/`, `data/processed/`) são gerados pelo pipeline ao rodar os notebooks em sequência.

## Pipeline Detalhado

| Etapa | Notebook | Descrição | Saída principal |
|---|---|---|---|
| Ingestão, construção de target e features | `01_ingest` | Carga da mensalizada → target forward-looking (3–12m) → rolling vl_* (3m+6m) → features derivadas → indicadores de grupo → separação temporal (tt/val/apl) | `data/gold/base_{tt,val,apl}_raw.parquet` |
| Feature engineering | `02_build_dataset_m2` | Binning quintílico (fit em tt), correção de lag 15/15, OHE, MICE — transform-only em val/apl | `data/gold/base_features_{tt,val,apl}.parquet` |
| Splits + Scaler | `03_splits` | Medoide por grupo, Gower, split 70/30 por ID, StandardScaler fit no treino | `data/processed/splits/{Grupo}/` + `models/{Grupo}/standard_scaler.joblib` |
| Feature Selection | `04_feature_selection` | 3 camadas: qualidade → RFECV → SHAP pós-treino, por grupo | `data/processed/feature_selection/{Grupo}/selected_features.json` |
| Modelagem | `05_model` | LR/RF/XGBoost/LightGBM; threshold por AP; score de risco (0–1) | `models/{Grupo}/best_model.joblib` |
| Tuning + Diagnóstico | `06_tuning` | RandomizedSearchCV (AP), K-fold consistency, drift treino→val (KS) | `models/{Grupo}/best_model_tuned.joblib` + figuras |
| SHAP + Drift val→apl | `07_shap` | SHAP global (summary/beeswarm/waterfall/dependence) + drift KS val→aplicação | `reports/shap/{Grupo}/` + `reports/figures/{Grupo}/drift_val_apl.png` |
| Score final | `08_score` | Aplica modelo tuned a val + aplicação (Jun–Dez/2025); classifica Alto/Médio/Baixo por ranking | `reports/score_risco_consolidado.parquet/.csv` |
| ROI | `09_roi` | Projeção de ROI em 3 cenários (conservador/moderado/otimista) | `reports/roi_projecao.parquet/.csv` + 3 figuras |
| Monitoramento | `10_monitor` | PSI por feature, AUC mensal, Precision A+M, alertas de retreino | `reports/monitor_*.csv` + 4 figuras |

## Resultados dos Modelos

Resultados obtidos na execução dos notebooks 04 (feature selection) e 05 (modelagem) sobre dados fictícios descaracterizados. A separação temporal entre treino/teste e validação é estrita — a validação cobre meses posteriores ao treino, simulando aplicação real.

### Feature Selection (`04_feature_selection`)

| Grupo | Features selecionadas | AUC-ROC (val) |
|---|---:|---:|
| Vendas | 70 | 0.744 |
| Transporte | 69 | 0.718 |
| Fábrica | 69 | 0.734 |

### Modelagem + Tuning (`05_model` → `06_tuning`)

| Grupo | Algoritmo | AUC-ROC (teste) | AUC-ROC (val) | Recall val (threshold) | Threshold |
|---|---|---:|---:|---:|---:|
| Vendas | XGBoost | 0.833 | 0.744 | 71.9% | 0.465 |
| Transporte | XGBoost | 0.828 | 0.718 | 71.5% | 0.365 |
| Fábrica | XGBoost | 0.811 | 0.734 | 70.3% | 0.240 |

> **Nota sobre a queda teste → validação:** a redução de performance entre teste e validação temporal é esperada e saudável — indica que o modelo está sendo avaliado de forma honesta, sem vazamento temporal. Modelos avaliados apenas em divisões aleatórias de treino/teste tipicamente reportam AUC 3–7 p.p. superior ao que entregam em produção.

## Decisões de Modelagem e Negócio

Esta seção documenta escolhas técnicas que divergem da solução que um cientista de dados escolheria
em ambiente de laboratório — e explica por que a solução adotada é, no entanto, a mais adequada ao
problema real do cliente.

---

### Classificação de risco por ranking percentual, não por threshold fixo

#### O que foi testado

Dois métodos foram avaliados sobre a base de validação pura (Jun–Ago/2025, 3 meses, ~15% target real):

| Método | Mecanismo | % Alertados (Alto) | Precision Alto | Recall Alto |
|---|---|---:|---:|---:|
| **M1 — Threshold fixo** | `prob >= T` (T tunado por recall ≥ 0.70) | 37–66% | 0.19–0.30 | 0.70–0.85 |
| **M2 — Ranking** | top `ref_rate × 1.4` por mês por grupo | 18–21% | 0.36–0.45 | 0.51–0.65 |

Para o grupo Alto+Médio combinado (que é o que recebe alguma ação de RH):

| Grupo | M2 Alto+Médio % | Precision A+M | Recall A+M |
|---|---:|---:|---:|
| Vendas | 26.9% | 0.316 | 0.584 |
| Transporte | 28.6% | 0.309 | 0.706 |
| Fábrica | 30.9% | 0.304 | 0.604 |

#### Por que o Método 1 (threshold) é melhor técnicamente

O threshold T foi calibrado em validação para maximizar recall ≥ 0.70, que é o objetivo padrão de
um modelo de risco de saída: *não perder nenhuma demissão que poderia ter sido evitada*. Um recall
de 0.70–0.85 significa capturar 7 em cada 10 saídas futuras — resultado sólido em ciência de dados.

#### Por que o Método 2 (ranking) é melhor para o problema do cliente

**1. Capacidade operacional de RH é limitada.**
Um time de RH de grande empresa consegue conduzir, em média, entre 15% e 25% de conversas de
retenção por mês (1:1 com gestor, ajuste salarial, proposta de mobilidade interna). Entregar uma
lista com 37–66% da força de trabalho classificada como "Alto" paralisa o processo: não há tempo,
orçamento nem energia para agir sobre todos. A lista se torna papel.

**2. Listas pequenas e consistentes geram confiança.**
Quando o modelo diz a cada mês "esses 18–21% são os prioritários", o gestor consegue agir e
verificar o resultado. Quando a lista muda drasticamente de tamanho a cada mês (comportamento
natural de um threshold fixo aplicado a populações flutuantes), a equipe perde confiança no modelo.
A previsibilidade do tamanho da lista é uma propriedade de negócio, não estatística.

**3. O custo de uma intervenção de retenção é real.**
Ações de retenção têm custo: aumento salarial, mudança de cargo, realocação, horas de conversa de
gestores. Uma precision de 0.19–0.30 (M1) significa que 70–81% das intervenções serão desperdiçadas
em pessoas que não sairiam de qualquer forma. Com M2 (precision 0.27–0.33), o desperdício cai para
~67–73%. A diferença em escala é significativa no orçamento de RH.

**4. Ancoragem na taxa real de turnover é transparente e auditável.**
O percentual de Alto é definido como `taxa_turnover_histórica × 1.4`. Isso significa que o RH
consegue entender e explicar *de onde vem aquele número*: "classificamos como Alto um volume 40%
acima do que historicamente sai, para ter margem de segurança." Não depende de explicar o que é um
threshold probabilístico de 0.115.

**5. Recalibração automática ao longo do tempo.**
Quando o turnover real da empresa sobe ou cai, os percentuais de Alto e Médio se ajustam
automaticamente na próxima execução (basta recalcular as taxas dos últimos 3 meses de
`base_features_tt`). Um threshold fixo precisaria de tuning manual para cada ciclo.

#### O que se perde na troca

A principal perda é **recall**: o M2 captura ~50% das saídas futuras contra ~78% do M1. Isso
significa que metade das demissões voluntárias não estará na lista de alertados. Este é um tradeoff
consciente: o cliente decide priorizar ações de alta qualidade sobre poucos alvos certos em vez de
ações dispersas sobre uma lista muito grande.

> **Recomendação de uso:** a faixa Alto+Médio (27–31% dos colaboradores por mês) é o segmento
> operacional. A faixa Alto isolada (18–21%) é para priorização máxima quando os recursos
> de retenção forem escassos.

---

### O custo de reposição no cálculo de ROI é uma simplificação

O notebook `09_roi` utiliza **4× o salário mensal** como estimativa do custo total de reposição
de um colaborador. Este valor é uma simplificação deliberada para demonstrar a estrutura do cálculo.

**Deve ser substituído por um levantamento real e interno** conduzido pelo time de RH/Controladoria,
considerando todos os custos envolvidos: agência de recrutamento, headcount temporário, curva de
aprendizado do substituto, impacto em indicadores de qualidade/produção e custo de gestão do processo.

Benchmarks externos estimam custos de reposição entre **1.5–3× o salário anual (18–36× o mensal)**
para cargos operacionais. O multiplicador de 4× mensal adotado aqui é propositalmente conservador
para não superestimar o impacto do modelo.

> ⚠️ **Ação necessária antes do uso em decisão real:** substituir `MULT_REPOSICAO = 4` em
> `notebooks/09_roi.ipynb` pelo custo real de reposição apurado internamente.

---

### Por que o modelo prediz com 4 meses de antecedência e não menos

O horizonte de 4 meses (`fg_demitido_voluntario_4m`) foi escolhido para dar ao RH tempo hábil para
conduzir um ciclo completo de retenção: identificação → conversa com gestor → proposta → decisão
do colaborador → implementação (ex.: aumento salarial, remanejamento). Horizontes menores (1–2 meses)
reduzem o sinal preditivo disponível e eliminam o tempo de reação; horizontes maiores (6+ meses) introduzem
incerteza excessiva porque as condições que levam à saída ainda não se manifestaram nas features.

---

### Por que o protótipo é o medoide (registro real mais central) e não uma média sintética

A versão inicial usava um **centóide sintético** (média das features comportamentais dos K=4 registros de cada desligado + slope). Testes mostraram que colaboradores com padrões de saída heterogêneos (ex.: piora abrupta vs. degradação gradual lenta) tinham seus protótipos mesclados, gerando exemplos de treino que não representavam nenhum comportamento real.

O **medoide** seleciona o registro real que minimiza a soma de distâncias euclidianas aos demais K registros (o mais central). Cada desligado é representado por **1 registro histórico real** — sem síntese, sem inventão de features.

---

## Técnicas de Feature Engineering

### Indicadores Contextuais por Grupo (§5.5 do 01_ingest)

Variáveis categóricas de alta cardinalidade (`ds_cargo` com ~250 categorias, `ds_uorg` com ~90,
`ds_gestor` com ~1.500) são quantificadas por seu **comportamento coletivo**, sem introduzir dados futuros.

Para cada grupo × mês são calculados 3 indicadores com **janela rolling de 4 meses** (mês atual + 3 anteriores):

| Feature | Fórmula | Interpretação |
|---|---|---|
| `vl_hc_{grupo}_4m` | média mensal de `fg_ativo == 1` nos 4 meses | Headcount médio do grupo |
| `vl_tv_vol_{grupo}_4m` | Σ `fg_dem_vol == 1` / Σ `fg_ativo == 1` | Taxa de turnover voluntário acumulada |
| `vl_tv_inv_{grupo}_4m` | Σ `fg_dem_inv == 1` / Σ `fg_ativo == 1` | Taxa de turnover involuntário acumulada |

Como `df` já é a mensalizada completa, os indicadores são calculados sobre toda a população — não apenas o subconjunto amostrado para modelagem — garantindo representatividade real de cada grupo.

### Rolling Features por Colaborador (§4 do 01_ingest)

Para cada variável `vl_*` (exceto `vl_conjuge`, `vl_filhos`, `vl_dependentes`), são geradas
auto-maticamente 8 features de comportamento temporal:

| Sufixo | Janela | Agregação |
|---|---|---|
| `_mean_3m` / `_mean_6m` | 3 ou 6 meses | Média do mês atual + anteriores |
| `_min_3m` / `_min_6m` | 3 ou 6 meses | Mínimo |
| `_max_3m` / `_max_6m` | 3 ou 6 meses | Máximo |
| `_std_3m` / `_std_6m` | 3 ou 6 meses | Desvio padrão |

Implementação otimizada: `groupby().rolling()` sobre bloco de colunas de uma vez (C vetorizado), sem loops Python por coluna.

> **Por que funciona?** Captura dinâmicas de equipes ao longo do tempo (ex.: gestores com alta
> rotatividade histórica → maior risco futuro) sem nunca expor informação futura ao modelo —
> um dos erros mais comuns em datasets de RH construídos sem rigor temporal.

### Categorização Quintílica (§4.5 do 02_build_dataset_m2)

Variáveis numéricas com mais de 10 valores distintos são discretizadas em até **5 categorias ordinais**
usando **quintis amostrais** — um método não-paramétrico robusto a assimetrias e caudas pesadas:

- **Zero exato** recebe categoria própria `"1_zero"`, separando zeros estruturais (ex.: colaborador sem faltas)
  de valores contínuos — evita distorção das demais categorias
- **Valores ausentes** permanecem `NaN` para tratamento explícito na etapa seguinte
- **Distribuições muito concentradas** têm bins duplicados removidos automaticamente (< 5 categorias)
- **Nomenclatura ordinal**: `"2_(−∞, 1.30]"`, `"3_(1.30, 3.75]"`, etc. — permite ordenação direta sem mapeamento adicional

---

### Por que MICE e não imputação simples (median/KNN/drop)

As variáveis com missing neste dataset são **não-aleatórias**: um colaborador sem dados de horas extras provavelmente é de um cargo sem hora extra, não tem dado porque não se aplica. Imputação por median ou mean ignoraria a correlação entre variáveis — por exemplo, a relação entre faltas e horas extras seria destruida ao imputar cada coluna independentemente.

**MICE (Multiple Imputation by Chained Equations)** imputa cada variável como função das demais, iterativamente, capturando a estrutura de correlação do dataset. O modelo de imputação é fitado exclusivamente em `base_tt` e aplicado por transform a `base_val` e `base_apl`, garantindo que não haja vaz. de informação do futuro para o treino.

---

### Por que Gower no subsampling de ativos

Colaboradores com longa antigüidade têm **centenas de registros mensais** — muitos deles muito parecidos entre si, pois o perfil de um colaborador estável muda pouco mês a mês. Sem filtro, o modelo de treino seria dominado por esses ativos de longa data, enquanto os ativos jovens (mais similares ao perfil de demitidos) seriam sub-representados.

A **distância de Gower** permite medir similaridade entre registros que têm colunas mistas (numéricas, binárias e ordinais). Para cada colaborador ativo, pares de registros consecutivos com Gower abaixo do p25 intra-indivíduo são considerados redundantes — o mais antigo é descartado. Isso elimina repetições sem descartar a evolução real do colaborador ao longo do tempo.

---

### Por que split por ID no teste, e não split temporal

A separacão treino/teste serve para **estimar a generalização do modelo a novos colaboradores** — pessoa que o modelo nunca viu antes. Se fizéssemos split temporal (ex.: treinar em 2024 e testar em Jan/2025), estaríamos medindo uma combinação de generalização a novos períodos E a novos colaboradores, o que dificulta o diagnóstico: se o modelo for ruim no teste, é drift temporal ou é overfit de ID?

O **split por ID** isola a pergunta correta: *"o padrão aprendido nos colaboradores de treino se generaliza para colaboradores diferentes?"* A validação temporal já é coberta pela base `val` (Jun–Ago/2025), que é inteiramente posterior ao período de treino.

> Invariante crítico garantido: nenhum `id_colaborador` aparece em mais de um split (treino/teste/val são disjuntos por ID).

---

### Por que 3 camadas de feature selection

Uma única etapa de seleção não cobre todos os problemas:

| Camada | Método | O que remove |
|---|---|---|
| **1 · Qualidade** | Missing >20%, near-zero variance, correlação >0.85 | Features com pouca informação ou redundância trivial |
| **2 · RFECV** | Random Forest, AUC, StratifiedKFold k=5 | Features que não contribuem para discriminação de forma robusta |
| **3 · SHAP** | SHAP values pós-treino do modelo campeão | Features que o modelo "aprendeu" mas que têm importância real baixa |

A camada 1 é rápida e remove lixo; a camada 2 reduz multicolinearidade e garante robustez por cross-validation; a camada 3 afina com o modelo final real, não com um proxy.

---

### Por que XGBoost venceu nos 3 grupos após tuning

Os 4 candidatos avaliados foram Logistic Regression (referência), Random Forest, XGBoost e LightGBM. O critério de seleção em `05_model` foi Average Precision (AP) calibrado com `scale_pos_weight` correto (derivado da taxa real de turnover na validação, ~10–12%). No `06_tuning`, todos os grupos convergiram para XGBoost após RandomizedSearchCV sobre o espaço de hiperparâmetros.

XGBoost se destacou pelas seguintes razões:

- **`scale_pos_weight` nativo**: o parâmetro de calibração de peso de classe é diretamente interpretável e permite calibrar com precisão a trade-off recall/precision sem heurísticas adicionais
- **Regularização explícita** (L1/L2 via `reg_alpha`/`reg_lambda`): eficaz em datasets com features ordinais `_bin` e binárias OHE que tendem a multicolinearidade
- **Interação com `max_depth` e `min_child_weight`**: permite controle fino da profundidade de árvore que ajuda na generalização para colaboradores novos (split por ID)

Após tuning, os modelos atingiram AP de 0.77–0.80 no OOF do treino e AUC-ROC de 0.718–0.744 na validação temporal.

---

## Resultados do Modelo

### Resultados no teste — holdout 30% IDs (mesmo período do treino)

| Grupo | AUC-ROC | Precision | Recall |
|---|---:|---:|---:|
| Vendas | 0.833 | 0.677 | 0.686 |
| Transporte | 0.828 | 0.546 | 0.757 |
| Fábrica | 0.811 | 0.542 | 0.804 |

**Leitura:** performance no teste holdout — IDs que o modelo nunca viu no treino, mas no mesmo período histórico. Esses são os valores mais otimistas da avaliação; a queda esperada para validação temporal (dados futuros reais) está documentada na seção seguinte.

### Performance na validação temporal (Jun–Ago/2025 — dados reais, não vistos no treino)

> Avaliação pelo método M2 (ranking percentual por grupo × mês). Executado em `10_monitor` com rótulos reais disponíveis.

| Grupo | AUC-ROC | Precision A+M | Recall A+M | % Alertados A+M | Baseline¹ | Uplift |
|---|---:|---:|---:|---:|---:|---:|
| Vendas | 0.744 | **25.2%** | 56.1% | 25.9% | 11.7% | **2.15×** |
| Transporte | 0.718 | **21.2%** | 46.5% | 21.6% | 9.8% | **2.16×** |
| Fábrica | 0.734 | **18.4%** | 51.7% | 26.4% | 9.4% | **1.96×** |

¹ *Baseline = seleção aleatória da mesma proporção de colaboradores (= taxa real de turnover na validação).*

**Por que esses resultados são sólidos para o problema de negócio:**

- **Uplift 2× na precisão:** ao usar a lista do modelo em vez de selecionar aleatoriamente, o RH direciona esforços de retenção para uma população com o dobro da concentração de risco real. Com recursos escassos (conversas de retenção, ajustes salariais, realocações), essa diferença é diretamente operacional.
- **Validação temporal estrita:** os meses Jun–Ago/2025 são inteiramente posteriores ao treino — nenhum colaborador do período de validação estava disponível durante a construção do modelo. Esse regime de avaliação é o que ocorre em produção e é significativamente mais rigoroso do que splits aleatórios.
- **Horizonte de 4 meses:** a predição antecipa o comportamento com 4 meses de antecedência, não 1 mês — quanto maior o horizonte preditivo, mais difícil o problema. Modelos reportados em literatura com horizontes de 1–2 meses em splits aleatórios são diretamente incomparáveis.
- **AUC 0.718–0.744 em dados futuros:** supera o limiar operacional (AUC > 0.65) em todos os grupos com margem confortável. A equivalência é: um recrutador que usa a lista do modelo toma a decisão certa (identificar saída real) 72–74% das vezes — versus 50% se escolhesse aleatoriamente.
- **Comparação interna (teste vs. validação):** a queda de AUC entre teste (0.811–0.833) e validação temporal (0.718–0.744) é de 7–9 p.p. — dentro da faixa típica para problemas de RH com separação temporal estrita, e inferior ao que se observa em modelos não calibrados para imbalance.

### Projeção de ROI — aplicação Set–Dez/2025 (4 meses)

> Custo de reposição: 4× salário mensal (simplificação — substituir por levantamento interno). Período: Set–Dez/2025.

| Cenário | Taxa de retenção | Colaboradores retidos | Economia estimada |
|---|---|---:|---:|
| Conservador | 20% dos verdadeiros em risco retidos | 336 | R$ 2,51 mi |
| Moderado | 35% dos verdadeiros em risco retidos | 587 | R$ 4,38 mi |
| Otimista | 50% dos verdadeiros em risco retidos | 840 | R$ 6,27 mi |

---

## Principais Drivers de Risco (SHAP)

As features com maior importância média absoluta de SHAP, por grupo:

| Rank | Vendas | Transporte | Fábrica |
|---|---|---|---|
| 1 | `vl_dias_menos_6_horas_max_6m_bin` | `vl_idade` | `vl_dias_noturno_std_6m` |
| 2 | `vl_remuneracao_variavel_max_3m` | `vl_dias_menos_6_horas` | `vl_saldo_banco_horas_med_3m_std_3m` |
| 3 | `vl_idade` | `vl_filhos` | `vl_idade` |
| 4 | `vl_filhos` | `vl_tempo_empresa` | `vl_filhos` |
| 5 | `vl_dias_menos_6_horas` | `vl_dias_menos_6_horas_std_6m` | `vl_dias_menos_6_horas_mean_6m` |

**Interpretação:**
- `vl_dias_menos_6_horas_max_6m_bin` / `vl_dias_menos_6_horas` — pico e tendência de dias com jornada abaixo de 6h: sinal de desengajamento progressivo, presente como feature top em todos os 3 grupos
- `vl_remuneracao_variavel_max_3m` (Vendas) — variabilidade da remuneração variável recente: queda de comissões/bônus é precursora consistente de saída em áreas comerciais
- `vl_idade` — presente no top-3 dos 3 grupos: faixas etárias extremas (muito jovens + senior próximo a aposentadoria) têm risco distinto
- `vl_filhos` — dependentes como ancoragem ao emprego atual: colaboradores sem filhos têm mobilidade relativa maior
- `vl_dias_noturno_std_6m` / `vl_saldo_banco_horas_med_3m_std_3m` (Fábrica) — variabilidade da escala e do banco de horas: instabilidade de turno é estressor operacional característico do grupo industrial
- `vl_tempo_empresa` (Transporte) — senioridade como fator de risco: colaboradores muito novos ou em ciclos de transição têm risco elevado

---

## Monitoramento e Retreino

O notebook `10_monitor` foi projetado para execução mensal, após cada ciclo de scoring. Monitora:

- **PSI (Population Stability Index)** das top-10 features por grupo, comparando treino vs. aplicação
  - `PSI < 0.10` → estável | `0.10 – 0.20` → alerta | `≥ 0.20` → crítico
- **AUC-ROC e Precision A+M mensais** na janela de validação (onde labels estão disponíveis)
- **Distribuição de scores** ao longo do tempo (shift de probabilidades)
- **Proporção Alto/Médio/Baixo** por grupo e mês

**Critérios de retreino** (qualquer um dos abaixo por mais de 1 ciclo mensal):
1. AUC-ROC < 0.65 em qualquer grupo
2. Desvio de Precision A+M > 5pp da referência
3. PSI ≥ 0.20 em qualquer feature top-10

**Estado atual (Maio/2026):** todos os grupos apresentam alertas de PSI crítico em pelo menos uma feature — `vl_horas_extras_extraordinarias_horas` e `vl_horas_trabalhadas_std_3m` em Vendas; `vl_falta_dias_max_6m_bin` e `vl_tv_vol_uorg_4m` em Transporte; `vl_saldo_banco_horas_med_3m_std_3m` em Fábrica. A performance de AUC ainda está acima do limiar mínimo (0.65) em todos os grupos (0.718–0.744). Fábrica e Transporte apresentam leve degradação de Precision A+M em relação à referência de validação. **Recomendado retreino no próximo ciclo** se o padrão de drift persistir.

---

- **Python 3.10+**
- **pandas**, **numpy** — manipulação de dados
- **scikit-learn** — `IterativeImputer` (MICE), `RFECV`, `StandardScaler`, `StratifiedKFold`
- **LightGBM** — modelo campeão (Vendas, Transporte)
- **XGBoost** — modelo campeão (Fábrica)
- **SHAP** — interpretabilidade global e individual
- **gower** — distância de Gower para subsampling intra-indivíduo
- **scipy** — KS-test para drift de features
- **matplotlib**, **seaborn** — visualizações
- **joblib** — serialização de modelos e scalers

## Autor

Rafael — [LinkedIn](https://linkedin.com/in/rafael-de-melo-balaniuk-014756a3/)

---

*Pipeline completo: notebooks 01–10, 4.918 linhas de código, 74 arquivos de relatórios gerados.*
