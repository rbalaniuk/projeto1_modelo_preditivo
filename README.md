# Modelo Preditivo de Turnover Voluntário

Modelo de machine learning para predição de turnover voluntário em empresas de grande porte, utilizando dados mensalizados de RH e indicadores econômicos.

## Problema de Negócio

O desligamento voluntário gera custos significativos com recrutamento, treinamento e perda de produtividade. Este projeto constrói um modelo preditivo que identifica **com 4 meses de antecedência** colaboradores com risco elevado de saída voluntária, permitindo ações preventivas de retenção.

## O Desafio Real

Modelos de predição de turnover são frequentemente reportados com AUC acima de 0.80 na literatura e em ferramentas comerciais. Esses resultados, porém, quase sempre são obtidos em conjuntos de teste do mesmo período histórico — onde os padrões de saída são estruturalmente idênticos ao treino.

O desafio real não é ajustar um modelo a dados históricos: é construir um sistema que antecipe comportamentos ainda por acontecer, com 4 meses de antecedência, em uma população que muda ao longo do tempo. Na prática, modelos com boa performance em treino/teste perdem significativamente quando colocados em validação temporal estrita — meses futuros, com condições macroeconômicas diferentes, ciclos salariais e mudanças organizacionais que não estavam no treino. O resultado típico é um modelo que "funciona no passado" e decepciona em produção.

Este projeto foi estruturado para garantir uma avaliação honesta: treino, teste e validação são cronologicamente separados — a validação (Jun–Ago/2025) é inteiramente posterior ao treino e ao teste — e o modelo é avaliado pela sua capacidade de generalizar a colaboradores novos em períodos futuros. A queda de performance entre teste e validação temporal é esperada e documentada. Em ambos os casos, os valores superam o que tipicamente se observa em problemas equivalentes com avaliação temporal rigorosa.

## Abordagem Técnica

- **Janela deslizante (Modelo 2):** cada linha do dataset representa o estado de um colaborador em um mês; o target olha 4 meses à frente — suficiente para um ciclo completo de retenção
- **Arquitetura multi-grupo:** modelos independentes para Vendas, Transporte e Fábrica — cada grupo tem seus próprios encoders, splits, feature selection, modelo e critério de classificação de risco
- **Feature engineering:** indicadores contextuais de grupo (headcount e taxas de turnover por cargo/uorg/gestor via rolling 4m), discretização quintílica não-paramétrica, correção de lag para variáveis de ponto (15/15)
- **Representação de desligados:** medoide dos K=4 registros mais recentes (registro real mais central, sem síntese) — um desligado = 1 linha no treino
- **Classificação de risco operacional:** ranking percentual por mês por grupo (proporcional à taxa histórica de turnover), não threshold fixo
- **Pipeline completo:** ingestão → feature engineering → splits → feature selection → modelagem → tuning → SHAP → score → ROI → monitoramento (10 notebooks)

## Estrutura do Projeto

```
├── notebooks/
│   ├── 01_ingest.ipynb              # Integração das 3 bases + rolling stats de grupo
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
│   ├── raw/                         # Bases de entrada (.parquet)
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
| **1 · Entrada** | `data/raw/*.parquet` | — | 3 bases brutas: wide de features, target temporal, mensalizada |
| **2 · Bases raw** | `data/gold/base_{tt,val,apl}_raw.parquet` | `01_ingest` | Após merge + rolling stats + separação temporal (tt/val/apl) |
| **3 · Features** | `data/gold/base_features_{tt,val,apl}.parquet` | `02_build` | Binning quintílico + OHE + MICE fit em base_tt; transform-only em val/apl |
| **4 · Splits** | `data/processed/splits/{Grupo}/` | `03_splits` | Medoide + Gower + split por ID + StandardScaler — treino/teste/val por grupo |
| **5 · Score** | `reports/score_risco_{Grupo}.parquet` | `08_score` | Probabilidade + score_risco + grupo_risco para cada colaborador × mês |

## Dados

Os dados utilizados são **fictícios e descaracterizados** — gerados a partir de padrões estatísticos reais, mas sem informações identificáveis. As bases de dados simulam bases que comumente estão disponíveis para o RH de empresas de médio e grande porte brasileiras.

Para executar o projeto, coloque os arquivos abaixo em `data/raw/`:

| Arquivo | Descrição |
|---|---|
| `input_wide_ficticio.parquet` | Features mensalizadas (1 linha = colaborador × mês) |
| `temporal_output_ficticio.parquet` | Target de turnover voluntário (múltiplas janelas de predição) |
| `mensalizada_ficticio.parquet` | Mensalizada completa: fonte para indicadores contextuais de grupo (headcount e taxas de turnover por cargo / uorg / gestor) |

## Base mensalizada

A **base mensalizada** é uma base de registros mensais de colaboradores. Ela tem como premissa que cada colaborador tenha no máximo 1 registro por mês. Ela é baseada no fechamento da folha de pagamento, então é referente ao último dia do mês. Em geral, essas bases são retiradas diretamente dos ERPs e têm informações básicas, como dados demográficos, de cargo e lotação. A essa base, podem ser acrescentadas outras informações, como de bases de ponto e de folha de pagamento. Ao trazer para esta base, a informação deve ser transformada em uma informação mensal. Por exemplo, as faltas, que vêm de bases de registros de faltas, viram a quantidade de faltas no mês.

Na base fictícia também foram incluídas informações de indicadores macroeconômicos, simulando uma empresa que tem sedes em múltiplos municípios. Nesse caso, a informação dos indicadores macroeconômicos de cada município no tempo é aplicada a cada colaborador que faz parte do município no período.

## Base de input

A **base de inputs** é uma base que amplia o horizonte de tempoda base mensalizada. Isso é feito para fenômenos onde a informação mensal é granular demais, então são criados períodos como 3 e 6 meses. Por exemplo, a quantidade de horas extras de um colaborador em um determinado mês é muito volátil, portanto, a média dessas horas no mês presente e nos 2 anteriores (período de 3 meses) é mais relevante.

Os notebooks de tratamento de dados constroem mais features deste tipo e têm funções prontas para este fim.

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
| Ingestão & Separação temporal | `01_ingest` | Merge wide × target × rolling stats × ds_grupo; split em tt/val/apl via `fg_sem_output` | `data/gold/base_{tt,val,apl}_raw.parquet` |
| Feature engineering | `02_build_dataset_m2` | Binning quintílico (fit em tt), correção de lag 15/15, OHE, MICE — transform-only em val/apl | `data/gold/base_features_{tt,val,apl}.parquet` |
| Splits + Scaler | `03_splits` | Medoide por grupo, Gower, split 70/30 por ID, StandardScaler fit no treino | `data/processed/splits/{Grupo}/` + `models/{Grupo}/standard_scaler.joblib` |
| Feature Selection | `04_feature_selection` | 3 camadas: qualidade → RFECV → SHAP pós-treino, por grupo | `data/processed/feature_selection/{Grupo}/selected_features.json` |
| Modelagem | `05_model` | LR/RF/XGBoost/LightGBM; threshold por AP; score de risco (0–1) | `models/{Grupo}/best_model.joblib` |
| Tuning + Diagnóstico | `06_tuning` | RandomizedSearchCV (AP), K-fold consistency, drift treino→val (KS) | `models/{Grupo}/best_model_tuned.joblib` + figuras |
| SHAP + Drift val→apl | `07_shap` | SHAP global (summary/beeswarm/waterfall/dependence) + drift KS val→aplicação | `reports/shap/{Grupo}/` + `reports/figures/{Grupo}/drift_val_apl.png` |
| Score final | `08_score` | Aplica modelo tuned a val + aplicação (Jun–Dez/2025); classifica Alto/Médio/Baixo por ranking | `reports/score_risco_consolidado.parquet/.csv` |
| ROI | `09_roi` | Projeção de ROI em 3 cenários (conservador/moderado/otimista) | `reports/roi_projecao.parquet/.csv` + 3 figuras |
| Monitoramento | `10_monitor` | PSI por feature, AUC mensal, Precision A+M, alertas de retreino | `reports/monitor_*.csv` + 4 figuras |

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
| Vendas | 26.9% | 0.377 | 0.659 |
| Transporte | 28.6% | 0.302 | 0.653 |
| Fábrica | 30.9% | 0.355 | 0.768 |

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
| `vl_tv_vol_{grupo}_4m` | Σ `fg_dem_vol2 == 1` / Σ `fg_ativo == 1` | Taxa de turnover voluntário acumulada |
| `vl_tv_inv_{grupo}_4m` | Σ `fg_dem_inv2 == 1` / Σ `fg_ativo == 1` | Taxa de turnover involuntário acumulada |

Os indicadores são calculados sobre a **população completa da mensalizada** (não apenas o
subconjunto amostrado para modelagem), garantindo representatividade real de cada grupo.

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

### Por que LightGBM venceu em 2 dos 3 grupos (Transporte usou XGBoost)

Os 4 modelos avaliados foram Logistic Regression (baseline), Random Forest, XGBoost e LightGBM. O critério de seleção foi Average Precision (AP) na validação. LightGBM foi campeão em Vendas e Fábrica; XGBoost foi campeão em Transporte.

LightGBM superou os demais por algumas características do dataset:

- **Variáveis mistas** (numéricas escaladas + ordinais `_bin` encodadas + binárias OHE): o crescimento leaf-wise do LightGBM explora bem espaços de features mistas
- **Dados esparsas em algumas colunas** (zeros estruturais): o LightGBM lida nativamente com sparsidade via binning adaptativo
- **Dataset de tamanho médio** (4–10K exemplos de treino por grupo): o LightGBM generaliza bem nesse regime sem requerer a regularização agressiva do XGBoost

Após tuning (`06_tuning`), os modelos atingiram AP de 0.83–0.86 no OOF do treino e AUC-ROC de 0.777–0.842 na validação temporal.

---

## Resultados do Modelo

### Resultados no teste — holdout 30% IDs (mesmo período do treino)

| Grupo | AUC-ROC | Precision | Recall |
|---|---:|---:|---:|
| Vendas | 0.854 | 0.631 | 0.839 |
| Transporte | 0.865 | 0.606 | 0.871 |
| Fábrica | 0.817 | 0.601 | 0.755 |

**Leitura:** performance no teste holdout — IDs que o modelo nunca viu no treino, mas no mesmo período histórico. Esses são os valores mais otimistas da avaliação; a queda esperada para validação temporal está documentada na seção seguinte.

### Performance na validação temporal (Jun–Ago/2025 — dados reais, não vistos no treino)

| Grupo | AUC-ROC | Precision A+M | Recall A+M | % Alertados (Alto+Médio) |
|---|---:|---:|---:|---:|
| Vendas | 0.805 | 0.377 | 0.659 | 26.9% |
| Transporte | 0.777 | 0.302 | 0.653 | 28.6% |
| Fábrica | 0.842 | 0.355 | 0.768 | 30.9% |

**Leitura:** para cada 10 colaboradores classificados como Alto ou Médio, 3.0–3.8 realmente são demissões voluntárias nos próximos 4 meses. O modelo captura ~65–77% das demissões reais.

### Projeção de ROI (base fictícia — simplificação; ver nota abaixo)

| Cenário | Taxa de retenção | Estimativa de ROI |
|---|---|---|
| Conservador | 20% dos verdadeiros em risco retidos | R$ 3.4 mi |
| Moderado | 35% dos verdadeiros em risco retidos | R$ 5.9 mi |
| Otimista | 50% dos verdadeiros em risco retidos | R$ 8.4 mi |

---

## Principais Drivers de Risco (SHAP)

As features com maior importância média absoluta de SHAP, por grupo:

| Rank | Vendas | Transporte | Fábrica |
|---|---|---|---|
| 1 | `vl_dias_menos_6_horas_med_3m` | `vl_dias_menos_6_horas_med_3m` | `vl_horas_extras_horas_med_3m` |
| 2 | `vl_falta_horas_med_6m_bin` | `vl_tempo_promocao` | `vl_falta_horas_med_6m_bin` |
| 3 | `vl_atraso_horas_med_6m` | `vl_falta_horas_med_6m_bin` | `vl_dias_menos_6_horas_med_3m` |
| 4 | `vl_tempo_empresa` | `vl_horas_extras_horas_med_3m` | `vl_falta_dias_med_6m` |
| 5 | `vl_horas_extras_horas_med_3m` | `vl_atraso_horas_med_6m_bin` | `vl_atraso_horas_med_3m_y` |

**Interpretação:**
- `vl_dias_menos_6_horas_med_3m` — média de dias com jornada abaixo de 6h nos últimos 3 meses: consistente entre os 3 grupos como o principal sinal de desengajamento antes da saída
- `vl_tempo_empresa` / `vl_tempo_promocao` / `qt_tempo_funcao` — variáveis de senioridade e histórico de progressao: colaboradores muito novos ou sem promoção recente têm maior risco
- `vl_falta_horas_med_6m_bin` — padrão de faltas: indicador clássico de desengajamento, capturado pelo modelo como disicriminativo even em formato discretizado
- `vl_atraso_horas_med_6m` / `vl_atraso_horas_med_3m_y` (Vendas, Fábrica) — horas de atraso acumuladas: sinal de baixo engajamento e desalinhamento crescente com a rotina de trabalho

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

**Estado atual (Abril/2026):** todos os grupos apresentam alertas de PSI crítico em `vl_tempo_promocao`, `vl_tv_vol_cargo_4m` e `vl_falta_dias_med_3m_y`. A performance de modelo ainda está acima dos limiares (AUC 0.777–0.842). Recomendado retreino se o padrão persistir no próximo ciclo.

---

- **Python 3.10+**
- **pandas**, **numpy** — manipulação de dados
- **scikit-learn** — `IterativeImputer` (MICE), `RFECV`, `StandardScaler`, `StratifiedKFold`
- **LightGBM** — modelo campeão (Vendas, Fábrica)
- **XGBoost** — modelo campeão (Transporte)
- **SHAP** — interpretabilidade global e individual
- **gower** — distância de Gower para subsampling intra-indivíduo
- **scipy** — KS-test para drift de features
- **matplotlib**, **seaborn** — visualizações
- **joblib** — serialização de modelos e scalers

## Autor

Rafael — [LinkedIn](https://linkedin.com/in/rafael-de-melo-balaniuk-014756a3/)

---

*Pipeline completo: notebooks 01–10, 4.918 linhas de código, 74 arquivos de relatórios gerados.*
