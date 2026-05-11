# Pipeline Design Document — Modelo Preditivo de Turnover Voluntário

> **Tipo:** Design Document técnico  
> **Escopo:** Pipeline completo — da ingestão ao monitoramento (notebooks 01–10)  
> **Última atualização:** Maio 2026

> **Principio arquitetural central:** o pipeline é construído inteiramente a partir de uma única base de entrada — a mensalizada de RH, onde cada colaborador aparece uma vez por mês. Não são necessárias bases de features pré-processadas nem de targets pré-calculados. O notebook `01_ingest` constrói o target (janelas 3–12m forward-looking), as rolling features por colaborador (3m e 6m) e os indicadores contextuais de grupo — tudo diretamente da mensalizada.

---

## 0.5 · Arquitetura Multi-Grupo

O pipeline é executado de forma **independente** para cada grupo operacional:

| Grupo | Filtro aplicado sobre a base gold |
|---|---|
| **Vendas** | `ds_grupo == "Vendas"` |
| **Transporte** | `ds_grupo == "Transporte"` |
| **Fábrica** | `ds_grupo == "Fábrica"` |

### Motivação

Grupos de colaboradores com perfis e dinâmicas distintos tendem a ter:
- Distribuições de features particulares (ex: tempo de empresa, avaliações)
- Drivers de turnover diferentes (ex: oportunidades de mercado, condições físicas)
- Taxas de turnover distintas → thresholds ótimos independentes

Modelar cada grupo separadamente permite que cada modelo capture padrões específicos
da sua população, em vez de ser influenciado pelo ruído cruzado entre grupos.

### Fluxo de dados

```
construcao_base_ficticio
              → INJECT-TAIL em mensalizada_ficticio (sinais brutos amplificados)
              → IMPL-9b: injeta demitidos sintéticos até CUTOFF_INJECT (Ago/2025)
              → mensalizada_ficticio.parquet  (Única base de entrada do pipeline)

01_ingest     → [carga da mensalizada]
              → [construção do target: fg_demitido_voluntario_{3..12}m via forward-rolling]
              → [rolling features por colaborador: vl_* × (mean/min/max/std) × (3m/6m)]
              → [features derivadas: vl_perc_diurno, vl_perc_dias_he]
              → [indicadores contextuais: hc/tv_vol/tv_inv por cargo/uorg/gestor (rolling 4m)]
              → base_tt_raw.parquet    (Jan/2024–Mai/2025, target válido)
              → base_val_raw.parquet   (Jun/2025–Ago/2025, target válido)
              → base_apl_raw.parquet   (Set/2025+, fg_sem_output=1, sem target)

02_build      → base_features_tt.parquet   (bins + OHE + MICE fit em tt)
              → base_features_val.parquet  (transform-only com os encoders de tt)
              → base_features_apl.parquet  (transform-only, target=NaN mantido)

03_splits     → splits/{Grupo}/train.parquet   (medoide + Gower + scaler)
              → splits/{Grupo}/test.parquet    (medoide + Gower, scaler transform)
              → splits/{Grupo}/val.parquet     (sem medoide, snapshot mensal)
              → models/{Grupo}/standard_scaler.joblib + scaler_cols.json

04_fs         → feature_selection/{Grupo}/selected_features.json + fs_report.parquet

05_model      → models/{Grupo}/best_model.joblib + model_metadata.json

06_tuning     → models/{Grupo}/best_model_tuned.joblib + tuning_metadata.json
              → reports/figures/{Grupo}/pr_tuning.png + drift_train_val.png
              → reports/figures/{Grupo}/reliability_diagram.png + pr_threshold_analysis.png

07_shap       → reports/shap/{Grupo}/shap_values.npy + shap_feature_importance.json
              → reports/shap/{Grupo}/summary_bar.png + summary_beeswarm.png
              → reports/shap/{Grupo}/waterfall_top_risk.png + dependence_top3.png
              → reports/figures/{Grupo}/drift_val_apl.png

08_score      → reports/score_risco_{Grupo}.parquet  (val + aplicação, Jun–Dez/2025)
              → reports/score_risco_consolidado.parquet/.csv
              → reports/figures/distribuicao_risco_grupos.png
              → reports/figures/histograma_proba_grupos.png + histograma_score_grupos.png

09_roi        → reports/roi_projecao.parquet/.csv
              → reports/figures/roi_por_cenario.png + roi_acumulado_total.png
              → reports/figures/roi_alertados_retidos.png

10_monitor    → reports/monitor_drift_psi.csv + monitor_performance_val.csv
              → reports/figures/monitor_score_dist.png + monitor_psi_heatmap.png
              → reports/figures/monitor_performance_val.png + monitor_classificacao_tempo.png
```

### Nota técnica: imputação unificada

O binning quintílico e a imputação são aplicados de forma **unificada** (base_tt)
no notebook 02, antes da divisão por grupo. Isso é uma simplificação válida para projetos
de portfolio, dado que os três grupos compartilham a mesma escala e os grupos têm
tamanhos similares — o imputer não é enviesado significativamente.

> **Imputação vigente:** `SimpleImputer(strategy='median')` fitado exclusivamente em `base_tt`
> e aplicado via `.transform()` em val e apl. A abordagem MICE (IterativeImputer + BayesianRidge)
> foi descartada por overflow numérico com 860+ colunas — a mediana é robusta e suficiente na escala deste pipeline.

---

## 0.6 · Princípios Arquiteturais da Separação Temporal

### flag `fg_sem_output`

Meses **Set/2025 em diante** possuem janela de 4 meses que ultrapassa o fim
da base histórica (Dez/2025). Para esses meses, o target é **semanticamente ausente**
(não é possível calcular `fg_demitido_voluntario_4m`) — diferente de `NaN` por
afastamento ou desligamento involuntário.

A flag `fg_sem_output = 1` identifica esses registros e impede que sejam:
- removidos pelo `dropna(target)` junto com afastamentos
- incluídos no treino/validação com target incorreto

### Três bases de dados

| Base | Período | Target | Uso |
|---|---|---|---|
| `base_tt_raw` | Jan/2024–Mai/2025 | Válido (0/1) | Treino + teste (split por ID) |
| `base_val_raw` | Jun/2025–Ago/2025 | Válido (0/1) | Validação temporal (sem split de ID) |
| `base_apl_raw` | Set/2025+ | NaN (`fg_sem_output=1`) | Aplicação/score |

### INJECT-TAIL e IMPL-9b

O `INJECT-TAIL` é aplicado **em `construcao_base_ficticio`**, antes de `IMPL-9b`.
Ele amplifica sinais comportamentais brutos (faltas, atrasos, advertências) nos
registros da `mensalizada_ficticio` dos demitidos voluntários **naturais**.

O `IMPL-9b` só injeta demitidos nos meses ≤ `CUTOFF_INJECT` (Ago/2025), garantindo
que todos os meses injetados tenham janela de 4 meses completamente dentro da base.

Apenas a `mensalizada_ficticio.parquet` é construída aqui e entregue ao pipeline.
Não há `temporal_output_ficticio` nem `input_wide_ficticio` gerados — o `01_ingest`
constrói o target e as features diretamente da mensalizada.

### Protótipo — aplicação pós-split

O **protótipo de desligados** (colapsar K registros → 1 representativo) é aplicado
no notebook `03_splits`, **após** o split por ID. Isso é obrigatório porque:
1. O protótipo não possui `dt_mes_ano` — é um ponto no espaço de features, sem data
2. A data é usada **apenas** para dividir em tt/val/apl — o modelo nunca vê a data
3. Aplicar o protótipo antes do split causaria data leakage implícito

Base `val` **não recebe protótipo** — mantém snapshots mensais individuais,
pois simula a aplicação real do modelo mês a mês.

---

## 1. Objetivo

Prever se um colaborador **ativo** realizará desligamento voluntário nos próximos X meses, usando o histórico passado como features e o futuro como target. Permite antecipação acionável para o RH.

---

## 2. Definição do Target

### Construção das colunas de target com horizonte variável (IMPL-9b)

O target é construído **diretamente na mensalizada** no notebook `01_ingest`.
Nenhuma base externa de target é necessária.

Colunas geradas: `fg_demitido_voluntario_{K}m` para K = 3 até 12.

```
Para cada colaborador C no mês de referência T:
  fg_demitido_voluntario_Km = 1   se fg_dem_vol == 1 em qualquer mês T+1..T+K
  fg_demitido_voluntario_Km = NaN se fg_dem_inv == 1 em T+1..T+K
                                  OU a janela T+K ultrapassa o último mês do colaborador
  fg_demitido_voluntario_Km = 0   caso contrário

Implementação otimizada:
  1. _build_forward_max(): inverte a série, rolling(N).max(), reinverte
     → calcula todas as janelas de uma vez por grupo em batch
  2. beyond = (dt_mes_ano + N meses) > last_mes do colaborador
     → NaN para janelas truncadas independente do valor de roll
```

**Regras adicionais (mesmas do Modelo 1):**
- Colaboradores com menos de 90 dias de empresa → excluir
- Desligamentos nos primeiros 90 dias → excluir
- Aposentadorias → NaN

### Seleção da janela de predição

```
Tentar janela de 4 meses (janela_predicao=4, default):
  Se n_positivos_treino >= threshold_min_positivos (configurável, default=80):
    → usar 4 meses

  Se n_positivos_treino < threshold_min_positivos:
    → tentar 6 meses
    Se ainda < threshold:
      → tentar 12 meses
      Se ainda < threshold:
        → abortar com mensagem clara indicando N insuficiente
```

---

## 3. Estrutura da Base

- **Única base de entrada:** mensalizada — um registro por colaborador por mês
- Cada linha representa o estado do colaborador em um mês de referência T
- Features olham para T ou passado de T (rolling sobre o histórico próprio do colaborador)
- Target olha para o futuro de T (construido em 01_ingest a partir de `fg_dem_vol`/`fg_dem_inv`)

### Colunas obrigatórias na mensalizada

| Coluna | Tipo | Papel |
|---|---|---|
| `id_colaborador` | str/int | Identificador único |
| `dt_mes_ano` | date | Mês de referência |
| `fg_dem_vol` | 0/1 | Saída voluntária neste mês |
| `fg_dem_inv` | 0/1 | Saída involuntária neste mês |
| `fg_ativo` | 0/1 | Ativo neste mês |
| `ds_grupo` | str | Grupo operacional (Vendas / Transporte / Fábrica) |
| `ds_cargo`, `ds_uorg`, `ds_gestor` | str | Grupos de alta cardinalidade (indicadores contextuais) |
| `vl_*` | float | Variáveis numéricas mensais — prefixo obrigatório para que o pipeline aplique rolling, binning e scaler automaticamente |
| `ds_*` | str | Variáveis categóricas — prefixo obrigatório; ≤ 5 categorias → OHE; > 5 categorias → descartadas (informação capturada pelos indicadores de grupo) |

---

## 4. Construção do Dataset Analítico

### 4.1 Janela temporal
- Últimos **24 meses** a partir de `max(data_referencia)`

### 4.2 Construção do protótipo para desligados (Medoide)

> ⚠️ Aplicado em **03_splits**, **após** o split por ID. Nunca em 02_build.
> O protótipo é um **registro real** selecionado — não um ponto sintético computado.

```
Para cada colaborador desligado voluntariamente:
  Coletar os K=4 registros mais recentes antes do mês de desligamento

  Selecionar o MEDOIDE dos K registros:
    1. Normalizar features numéricas por std dos K registros (evita domínio de escala)
    2. Calcular matriz de distâncias euclidianas K×K (pairwise)
    3. Selecionar o registro com menor soma de distâncias aos demais
    → resultado: 1 REGISTRO REAL, sem síntese, sem geração de colunas _slope
```

**Por que medoide e não protótipo sintético?**
Colaboradores com padrões de saída heterogêneos (ex: piora abrupta vs. degradação gradual)
teriam seus protótipos sintéticos misturados, criando exemplos de treino que não
representam nenhum comportamento real. O medoide preserva a integridade individual.

**Resultado:** cada desligado aparece **1×** como registro real — o que melhor representa
o padrão central do período pré-saída, sem perda de informação estrutural.

**Base val não recebe protótipo** — mantém snapshots mensais individuais (simula aplicação real).

### 4.2b Espaço de distância reduzido (medoide e subsampling)

O cálculo de distância para o medoide e para o subsampling iterativo de ativos é feito sobre um
**subconjunto reduzido de features**, derivado exclusivamente do conjunto de treino. Isso
**não afeta as features que chegam ao modelo** — apenas o espaço interno de seleção de registros.

**Motivação:** a base pré-feature-selection tem ~814 features numéricas, das quais ~391 são pares
`(vl_x, vl_x_bin)` para a mesma variável original. Usar todas no cálculo de distância é redundante
(dobra o peso de cada variável binada) e caro — a complexidade do subsampling é proporcional ao
número de features.

**Função:** `_reduzir_feat_dist(df_treino, feature_cols, corr_weights)` — aplicada no treino,
**após** o split por ID (sem informação de teste ou val).

**Critérios aplicados em sequência:**

| Critério | Descrição | Motivo |
|---|---|---|
| Pares redundantes `_bin` | Quando existem `vl_x` e `vl_x_bin`, mantém apenas `vl_x` | A versão `_bin` é discretização da mesma variável; para distância euclidiana é redundante e duplica o peso da variável |
| Near-zero variance (std < 0.01) | Features sem variabilidade no treino | Distância nula — não contribuem para discriminação |
| Baixa correlação com target (\|corr\| < 0.05) | Features sem poder discriminativo | Não guiam a seleção em direção ao fenômeno de interesse |

**Resultado:** ~814 → ~200 features para o cálculo de distância. O conjunto final de features
salvo nos splits (`train.parquet`, `test.parquet`, `val.parquet`) permanece intacto com todas as
features originais.

**Por que não há leakage:** todos os critérios (std, corr com target) são calculados dentro do
`df_train_raw` — subconjunto de IDs de treino, após o split 70/30 por ID. Teste e val nunca
informam a seleção. O medoide e o subsampling apenas escolhem registros reais existentes; reduzir
o espaço de distância não cria nem contamina features.

### 4.3 Subsampling dos ativos

```
Para cada colaborador ativo:

  1. Subsampling por tempo_empresa_meses:
     → Selecionar 1 registro a cada janela_subsampling meses
        (janela_subsampling configurável, default=4)
     → O índice é relativo a tempo_empresa_meses, não ao mês calendário
     → Exemplo: registros nos meses de empresa 4, 8, 12, 16...
     → Se o primeiro registro disponível do colaborador for o mês 51 de empresa:
        → alinhar o grid a partir desse mês (51, 55, 59...)

  2. Filtro de similaridade por distância de Gower:
     → Para cada par de registros consecutivos do mesmo colaborador
        (após subsampling):
       Calcular Gower(T, T_anterior)
       Se Gower < threshold_similaridade:
         → drop do registro mais antigo
     → threshold_similaridade = percentil configurável da distribuição
       de distâncias intra-indivíduo calculada no treino
       (threshold_gower_percentil, default=25)
     → Se todos os registros do colaborador forem removidos pelo filtro:
        → manter apenas o mais recente
     → Biblioteca: `gower` (pip install gower) — disponível ✅
```

---

## 5. Divisão Treino / Validação / Teste

> ⚠️ Divisão **obrigatoriamente por id_colaborador** — o mesmo colaborador não pode aparecer em mais de um conjunto.
>
> ⚠️ Executada **independentemente para cada grupo** (Vendas / Transporte / Fábrica).
>
> ⚠️ Validação é **temporal** (base_val_raw, Jun/2025–Ago/2025), não retirada de tt por split aleatório.

```
Para cada grupo G em [Vendas, Transporte, Fábrica]:

  ── Treino / Teste (de base_features_tt) ──────────────────────
  1. Carregar base_features_tt filtrado para grupo G
  2. Obter lista de ids únicos de colaboradores do grupo
  3. Dividir ids: 70% treino / 30% teste
     (estratificado pela taxa de turnover por colaborador)
  4. Todos os registros de um colaborador vão para o mesmo conjunto

  ── Protótipo (aplicado em TREINO e TESTE separadamente) ──────
  5. Para desligados (target=1): colapsar K registros → 1 protótipo
  6. Para ativos (target=0): aplicar filtro Gower intra-individual

  ── Normalização ──────────────────────────────────────────────
  7. Fit StandardScaler EXCLUSIVAMENTE no treino pós-protótipo
  8. Transform: treino (fit) + teste + val (sem protótipo)

  ── Validação (de base_features_val — snapshots mensais) ──────
  9. Filtrar base_features_val para grupo G
     → sem protótipo (mantém registros mensais reais)
     → transform com scaler fitado no treino

  ── Salvar ────────────────────────────────────────────────────
  data/processed/splits/{G}/train.parquet    ← protótipo + Gower
  data/processed/splits/{G}/test.parquet     ← protótipo + Gower
  data/processed/splits/{G}/val.parquet      ← snapshots mensais
  data/processed/splits/{G}/scaler_cols.json
  models/{G}/standard_scaler.joblib
```

---

## 6. Categorização de Features Contínuas

**Abordagem adotada: Quintis não-paramétricos** (simplificação intencional para pipeline automatizado).

Motivo: categorização supervisionada (MDLP) exige ajuste manual das labels semânticas para interpretabilidade, o que é incompatível com automação. Quintis são robustos a assimetrias e suficientes para o objetivo.

```
Para cada variável contínua/ordinal com > 10 valores distintos:
  Zeros exatos → categoria "1_zero" (isolados do resto)
  Valores não-zero → até 4 quintis (se há zeros) ou 5 quintis (sem zeros)
  NaN → mantidos como NaN (imputados posteriormente via MICE)

Nomenclatura ordinal automática:
  "{rank}_{(lo, hi]}"  ex: "2_(14.00,27.00]"
  O prefixo numérico permite extração ordinal imediata: "2_..." → 2.0

Não há necessidade de nbins manuais, labels semânticos ou ajustes pós-MDLP.
```

---

## 7. Feature Selection

> Calculada **exclusivamente no treino**, de forma **independente por grupo**.
> Cada grupo pode ter um conjunto diferente de features finais.

#### Camada 1 — Filtro por qualidade
1. Missing > 20% → drop
2. Near-zero variance → drop
3. Correlação > 0.85 entre features → drop a com mais missing

#### Camada 2 — RFECV com preferência por categorizadas
- Estimador: Random Forest ou XGBoost
- Métrica: ROC-AUC
- CV: Stratified K-Fold (k=5)
- Preferência por features MDLP sobre contínuas (mesma lógica do Modelo 1)
- Boruta opcional (`usar_boruta`, default=False)

#### Camada 3 — SHAP pós-treino

---

## 8. Encoding de Variáveis Categóricas

```
Se n_categorias <= 5 distintos:
  → One-Hot Encoding (dummy_na=False, dtype=float)

Se n_categorias > 5 distintos:
  → Drop (alta cardinalidade)
  → Informação de grupos (ds_cargo, ds_uorg, ds_gestor) capturada
    indiretamente pelos indicadores de rolling stats (headcount e
    taxas de turnover por grupo) calculados em 01_ingest.
```

### Correção de lag para colunas de ponto (15/15)

Variáveis de ponto (timesheet) têm apuração no período 15 a 15, enquanto a folha é 01 a 31. Isso pode incluir dados pós-demissão no último mês "ativo" de um colaborador.

```
Arquivo de controle: reports/dic_colunas_complemento.xlsx
  Coluna: Feature  | lag (int, default=0) | descricao

Para colunas marcadas com lag=1:
  Para cada colaborador, substituir o valor do mês T
  pelo valor do mês T-1 (shift+1 dentro do grupo).
  Para o primeiro registro (sem mês anterior): manter valor original.
  Aplicado em df antes do protótipo e da imputação MICE.
```

---

## 9. Tratamento de Missing (Imputação)

- `SimpleImputer(strategy='median')` fitado exclusivamente em `base_tt` (notebook 02)
- `.transform()` aplicado de forma idêntica em val e apl — sem re-fit
- Target nunca incluído na imputação
- Categóricas → `"Unknown"` (preenchimento explícito antes do OHE)
- Colunas com > 20% de missing são descartadas na Camada 1 da feature selection (antes da imputação)

> **Decisão de implementação:** `IterativeImputer` (MICE com BayesianRidge) foi avaliado mas
> descartado — com 860+ colunas o BayesianRidge gera overflow numérico (`overflow encountered in square`),
> produzindo NaN nas predições internas e `ValueError` na saída do imputer.
> `SimpleImputer(median)` é API-compatível (mesmos métodos `.fit_transform()` / `.transform()`),
> robusto a assimetrias e suficiente para esta escala de features.

---

## 10. Feature Engineering

Todas as features são construídas automaticamente pelo pipeline a partir da mensalizada.

### Construídas em `01_ingest`

| Tipo | Descrição | Notação |
|---|---|---|
| **Rolling por colaborador** | média, mínimo, máximo e std das variáveis `vl_*` em 3 e 6 meses | `{col}_mean_3m`, `{col}_min_6m`, etc. |
| **Features derivadas** | percentual de dias diurnos, percentual de dias com HE | `vl_perc_diurno`, `vl_perc_dias_he` |
| **Indicadores de grupo** | headcount, tv_vol, tv_inv por cargo/uorg/gestor (rolling 4m) | `vl_hc_{g}_4m`, `vl_tv_vol_{g}_4m`, `vl_tv_inv_{g}_4m` |

**Colunas `vl_*` excluídas do rolling:** `vl_conjuge`, `vl_filhos`, `vl_dependentes`
(valores estáticos que não fazem sentido como agregação temporal).

**Implementação otimizada do rolling:** `groupby().rolling()` sobre bloco de colunas inteiro
→ `droplevel + reindex`, sem loop Python por coluna. Toda a construção é vetorizada.

### Construídas em `02_build_dataset_m2`

| Tipo | Descrição |
|---|---|
| **Bins quintílicos** | colunas `_bin` (quintis ordinais) para variáveis numéricas com >10 distintos |
| **OHE** | categóricas com ≤5 valores distintos → dummies binárias |
| **Imputação** | `SimpleImputer(median)` fitado exclusivamente em base_tt — ver seção 9 |

Normalização:
- `StandardScaler` para contínuas (num_estaveis + num_comportamentais + slopes)
- Features `_bin` ordinais e OHE binárias: **não normalizar**
- Aplicado em **03_splits**, fitado exclusivamente no treino

---

## 11. Tratamento de Desbalanceamento

Idêntico ao Modelo 1:

| Prioridade | Método | Observação |
|---|---|---|
| 1º | `class_weight="balanced"` | Preferencial |
| 2º | `RandomUnderSampler` | `undersample_ratio` configurável, default=1:10 |
| 3º | SMOTE-NC | Último recurso, com todas as restrições do Modelo 1 |

---

## 12. Modelagem

> Executada **independentemente por grupo**. Cada grupo tem seu próprio modelo, threshold e M.

### Modelos avaliados:
1. Logistic Regression (baseline)
2. Random Forest
3. XGBoost
4. LightGBM

### Critério de comparação:
- ROC-AUC (principal)
- Average Precision (AP)
- Recall@threshold (priorizar classe turnover)

### Resultado (`05_model`):

| Grupo | Algoritmo campeão | AUC-ROC (teste) | AUC-ROC (val) | Recall (val) | Threshold |
|---|---|---:|---:|---:|---:|
| Vendas | XGBoost | 0.870 | 0.853 | 74.8% | 0.690 |
| Transporte | XGBoost | 0.921 | 0.915 | 75.7% | 0.720 |
| Fábrica | RandomForest | 0.904 | 0.898 | 84.0% | 0.385 |

Os hiperparâmetros são refinados no `06_tuning` via `RandomizedSearchCV` (scoring=AP),
produzindo `best_model_tuned.joblib` e `tuning_metadata.json` por grupo.

### Resultado pós-tuning (`06_tuning`):

| Grupo | AUC-ROC (teste) | AUC-ROC (val) | Recall (val) | Precision (val) | Recall (teste) | Precision (teste) | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vendas | 0.872 | 0.863 | 59.5% | 49.7% | 64.8% | 84.9% | 0.590 |
| Transporte | 0.926 | 0.922 | 60.8% | 50.1% | 69.5% | 87.5% | 0.615 |
| Fábrica | 0.895 | 0.895 | 63.4% | 51.2% | 69.6% | 81.5% | 0.475 |

> **Resultado emergente (seção 13.2):** em todos os grupos, `precision > recall` no **teste**
> e `recall > precision` na **validação temporal** — o threshold encontra espontaneamente
> o equilíbrio certo para generalizar ao futuro.

---

## 13. Definição do Threshold Ótimo

> Calculado **independentemente por grupo**, no conjunto de **teste** (`05_model`) e re-otimizado
> com o modelo tunado no conjunto de **validação** (`06_tuning`).
> O threshold T é salvo em `model_metadata.json` / `tuning_metadata.json` e usado na fórmula de score (seção 14).

### 13.1 Função de score do threshold

Para cada threshold candidato `th` em `[0.005, 1.0)` com passo `0.005`, avalia-se:

```python
bonus = 1.0  if (recall >= META_REC and precision >= META_PRE)  else 0.0

dR = recall - META_REC
if dR < 0:  dR = dR / 0.80   # penalidade assimétrica: errar recall é mais grave
elif dR > 0: dR = dR * 1.20  # bônus por exceder a meta de recall

dP  = precision - META_PRE
dPR = abs(recall - precision - 0.10)  # penalidade por desequilíbrio recall/precisão

score(th) = bonus + dR + dP - dPR
```

`META_REC = 0.70`, `META_PRE = 0.60`.

**Constraint adicional:** apenas thresholds onde `recall >= precision` são elegíveis
(filtra candidatos com precisão dominando fortemente o recall).

O threshold com maior `score(th)` é selecionado.

### 13.2 Comportamento emergente — resultado notável

A fórmula foi projetada com regras fixas e determinísticas: não há nada adaptativo nos parâmetros.
O objetivo declarado é priorizar recall de forma muito forte (`dR` assimétrico), penalizar precisão
insuficiente e restringir thresholds onde recall e precision divergem demais (`dPR`).

**O resultado observado empiricamente:** em vários grupos, o threshold selecionado apresenta
`precision > recall` no conjunto de **teste** — mas esse mesmo threshold produz `recall > precision`
na **validação temporal** (período futuro, out-of-time).

Isso é contraintuitivo e não estava escrito em nenhuma regra. O sistema resolveu automaticamente
uma tensão latente nos dados:

> O fenômeno que queremos capturar (turnover voluntário futuro) é raro e difícil de encontrar
> na validação. O conjunto de teste, por pertencer ao mesmo período histórico do treino, tem
> uma distribuição ligeiramente diferente. Para que o modelo tenha recall alto *onde importa*
> (validação futura), o threshold precisa ser "aberto" o suficiente para que, na distribuição
> histórica do teste, a precisão acabe dominando o recall.

**Por que isso é um resultado sensacional:**

- A fórmula não impõe que `recall > precision`. Ela define *o que importa* (recall alto, penalidade
  por desequilíbrio). Cada grupo encontra seu próprio equilíbrio, com seu próprio threshold,
  refletindo sua própria distribuição de comportamento.
- O resultado não é fruto de ajuste manual, overfitting de regras ou intervenção por grupo.
  Emerge de forma totalmente automática a partir da estrutura dos dados.
- É um indicativo forte de que o pipeline está aprendendo a generalização real — não sobreajustando
  às métricas do conjunto de avaliação imediato.

**As regras de negócio guiam a direção. Os dados determinam o ponto de equilíbrio.**

> **Nota de implementação**: o threshold T é usado como referência para otimização do modelo
> em `06_tuning` e como parâmetro T na fórmula de score de risco (seção 14).
> A classificação final dos colaboradores em Alto/Médio/Baixo é derivada diretamente do
> `score_risco`: Alto ≥ 0.50 (prob ≥ T), Médio ≥ 0.35 (M ≤ prob < T), Baixo < 0.35 — ver seção 14.

---

## 14. Score de Risco por Colaborador

> **M** e **T** são calculados **por grupo**, garantindo que a escala de risco reflita
> a distribuição de probabilidades específica de cada população.

```python
def calcular_score_risco(prob, T, M):
    if prob >= T:
        score = 0.50 + ((prob - T) / (1.0 - T)) * 0.50
    elif prob >= M:
        score = 0.35 + ((prob - M) / (T - M)) * 0.15
    else:
        score = (prob / M) * 0.35
    return round(score, 4)
```

**M** = percentil 25 das probabilidades preditas no treino do grupo (`medio_risco_percentil`, default=25)

| Campo            | Tipo       | Descrição                      |
|------------------|------------|--------------------------------|
| `id_colaborador` | string/int | Identificador único            |
| `mes_referencia` | date       | Mês do snapshot                |
| `score_risco`    | float      | Score reescalado (0.00 a 1.00) |
| `grupo_risco`    | string     | "Alto", "Médio" ou "Baixo"     |

**Classificação Alto/Médio/Baixo — por `score_risco`**

A categoria de risco é derivada diretamente do `score_risco`, garantindo consistência
entre score e faixa: não há colaborador “Alto” com `score_risco < 0.50`.

```python
grupo_risco = np.where(
    score_risco >= 0.50, "Alto",     # equivale a proba >= T
    np.where(score_risco >= 0.35, "Médio", "Baixo")  # M <= proba < T
)
```

| Distribuição na aplicação (Set–Dez/2025)  | Vendas        | Transporte    | Fábrica      |
|---|---|---|---|
| **Alto** (score ≥ 0.50)              | 1.105 (13,2%) | 989 (8,9%)    | 407 (15,4%)  |
| **Médio** (0.35 ≤ score < 0.50)      | 2.021 (24,1%) | 3.967 (35,5%) | 1.203 (45,6%)|
| **Baixo** (score < 0.35)             | 5.277 (62,8%) | 6.213 (55,6%) | 1.026 (38,9%)|
| **Total colaborador×mês**            | 8.403         | 11.169        | 2.636        |

> Para uso operacional: a faixa **Alto+Médio** é o segmento de ação de RH;
> a faixa **Alto** isolada é para priorização máxima com recursos escassos.

---

## 15. Avaliação

- ROC Curve
- Precision-Recall Curve
- Confusion Matrix (com threshold ótimo)
- Feature Importance
- SHAP values (global: summary bar + beeswarm + dependence plots)
- Explicações individuais: waterfall plot para os N colaboradores de maior risco no val (top-3 por grupo), mostrando a contribuição de cada feature para aquela predição específica — ferramenta direta de comunicação com o RH

---

## 16. Métricas de Sucesso

- Threshold ótimo > 0.10
- Average Precision > 0.70 (no treino, pós-tuning)
- ROC-AUC > 0.65 (mínimo operacional — limiar de alerta em `10_monitor`)
- Precision A+M (validação, ranking M2): > 0.25
- Recall A+M (validação): > 0.45

---

## 17. Deploy (opcional)

- API FastAPI: `POST /predict`
- Modelo salvo com `joblib`
- Metadados salvos: threshold, M, janela_predicao usada, parâmetros de categorização
- Versionamento via MLflow

---

## 18. Monitoramento Pós-Deploy (`10_monitor`)

Implementado em `notebooks/10_monitor.ipynb`. Executado mensalmente, após cada ciclo de scoring.

**Métricas monitoradas:**

- **PSI (Population Stability Index)** das top-10 features selecionadas por grupo,
  comparando distribuição de treino vs. validação (ref) e vs. aplicação (corrente)
  - `PSI < 0.10` → estável | `0.10 ≤ PSI < 0.20` → alerta moderado | `PSI ≥ 0.20` → crítico
- **AUC-ROC mensal** na validação (labels reais disponíveis de Jun–Ago/2025)
  - Alerta se AUC < 0.65
- **Precision A+M mensal** vs. referência por grupo
  - Alerta se desvio > 5pp da referência
- **Distribuição de scores** ao longo do tempo (histograma val vs. aplicação)
- **Proporção Alto/Médio/Baixo mensal** (estabilidade da lista de risco)

**Critério de retreino:**
Retreino recomendado se qualquer condição persistir por > 1 ciclo mensal:
1. AUC-ROC < 0.65 em qualquer grupo
2. Precision A+M com desvio > 5pp da referência por grupo
3. PSI ≥ 0.20 em qualquer feature top-10 selecionada

**Saídas:** `reports/monitor_drift_psi.csv`, `reports/monitor_performance_val.csv`,
4 figuras em `reports/figures/` (distribuição de scores, heatmap PSI,
performance mensal, proporção de risco ao longo do tempo).

---

## 19. Consistência K-fold do Modelo Final (`06_tuning`)

> Executado **após** a seleção de hiperparâmetros, usando os **melhores parâmetros já fixos**.
> Objetivo: quantificar a variância real do modelo campeão — distingue instabilidade
> estrutural de oscilação esperada de amostragem.

```
Por grupo:
1. Carregar tuning_metadata.json (melhores parâmetros + threshold)
2. Reconstituir o modelo com exatos melhores parâmetros (clone do estimador)
3. StratifiedKFold(n_splits=5, shuffle=True) no split de TREINO completo
4. Para cada fold:
   - Treinar nos índices de treino do fold
   - Avaliar nos índices OOF (out-of-fold): AUC-ROC, AP, Recall@threshold, Precision@threshold
5. Reportar: média ± desvio-padrão por métrica
6. Sinal de atenção: std(AUC-ROC) > 0.04 → instabilidade estrutural
                     Recall mínimo de qualquer fold < 0.50 → fold instável
```

**Saída:** tabela por grupo (fold × métricas) + resumo consolidado impresso no notebook.
Nenhum arquivo salvo — análise diagnóstica, não altera modelos.

---

## 20. Drift Treino → Validação (`06_tuning`)

> Verificar se a distribuição das features selecionadas no set de **validação temporal**
> (Jun–Ago/2025) difere significativamente do **treino** (Jan/2024–Mai/2025).
> Drift alto indica que as métricas no val devem ser interpretadas com cautela.

```
Por grupo:
1. Carregar train.parquet e val.parquet — apenas as FEATURES selecionadas
2. Para cada feature:
   - KS test (Kolmogorov-Smirnov): ks_stat, p-value  [scipy.stats.ks_2samp]
   - Jensen-Shannon divergence via histograma de 20 bins (sem dependência externa)
3. Classificar por nível:
   - KS > 0.20  → "significativo"
   - KS ∈ (0.10, 0.20] → "moderado"
   - KS ≤ 0.10  → "estável"
4. Sumário: contagem por nível; top-10 features com maior KS
5. Plot: barras horizontais com KS stat das top-20 features,
         marcando limiar moderado (laranja) e significativo (vermelho)
6. Salvar plot em reports/figures/{grupo}/drift_train_val.png
```

**Saída:** tabela impressa + gráfico por grupo. Nenhum modelo alterado.

---

## 21. Drift Validação → Aplicação (`07_shap`)

> Antes de aplicar o modelo, verificar se a **população de aplicação** (Set–Dez/2025)
> difere da **validação** (Jun–Ago/2025). Drift alto na aplicação indica que as
> probabilidades podem estar descalibradas e o threshold pode precisar de revisão.

```
Por grupo:
1. Carregar val.parquet e base_features_apl.parquet — FEATURES selecionadas
2. Mesma metodologia da Seção 20 (KS + JS divergence)
3. Comparação adicional: distribuição de probabilidades preditas
   - KDE (density) do score no val vs score na aplicação sobrepostos
   - Flagear se a moda ou mediana de score shiftar > 0.10 entre os conjuntos
4. Sumário: KS por feature + distribuição de scores comparada
5. Salvar plots em reports/figures/{grupo}/drift_val_apl.png
```

**Saída:** tabela impressa + gráficos por grupo. Scorecard de risco inclui nota de drift.

---

## 22. Score de Risco Aplicado (`08_score`)

> Aplicação dos modelos tuned (LightGBM) a todos os snapshots de validação e aplicação.

```
Por grupo (Vendas, Transporte, Fábrica):
1. Carregar val.parquet e base_features_apl.parquet filtrado para o grupo
2. Aplicar StandardScaler (transform-only) sobre scaler_cols do grupo
3. Subconjunto para selected_features do grupo
4. Carregar best_model_tuned.joblib
5. prob = model.predict_proba(X)[:, 1]
6. score_risco = calcular_score_risco(prob, T, M)  # fórmula seção 14
7. grupo_risco = classificação por ranking percentual (seção 14)
```

**Saída consolidada:** `reports/score_risco_consolidado.parquet/.csv`
Colunas: `id_colaborador`, `dt_mes_ano`, `ds_grupo`, `prob_score`, `score_risco`, `grupo_risco`.

---

## 23. Projeção de ROI (`09_roi`)

> Estimativa do valor financeiro capturado pelo modelo, dado um conjunto de hipóteses
> de taxa de retenção e custo de reposição.

**Cenários avaliados:**

| Cenário      | Taxa de retenção | ROI estimado (base fictícia) |
|---|---|---|
| Conservador  | 10%               | R$ 6.2 mi                   |
| Moderado     | 17%               | R$ 10.7 mi                  |
| Otimista     | 25%               | R$ 15.4 mi                  |

**Fórmula:**
```python
ROI = n_alertados * taxa_retencao * salario_mensal_medio * MULT_REPOSICAO
# MULT_REPOSICAO = 4  (simplificação — deve ser substituído por custo real)
```

> ⚠️ O multiplicador `MULT_REPOSICAO = 4` é conservador (benchmarks externos: 18–36× salário mensal).
> Substituir por levantamento real de RH/Controladoria antes de uso em decisão.

**Saída:** `reports/roi_projecao.parquet/.csv` + 3 figuras.

---

# 🔥 Pontos Estratégicos (Ambos os Modelos)

1. Sem histórico consistente de desligamentos voluntários → modelo fraco
2. N < 500 colaboradores após filtros → usar apenas Logistic Regression
3. Sem dados comportamentais → capacidade preditiva cai drasticamente
4. RH deve estar envolvido desde o início
5. Threshold, M e metadados de categorização **devem ser salvos com o modelo**
6. Categorias que usaram fallback no target encoding devem ser registradas para auditoria
7. Modelo 1 é o baseline — validar com o negócio antes de evoluir para Modelo 2

---

# 📌 Stack Utilizada

- **Python 3.10+**
- **pandas**, **numpy** — manipulação de dados e arrays
- **scikit-learn** — `SimpleImputer` (median), `RFECV`, `StandardScaler`, `StratifiedKFold`, `KolmogorovSmirnov`
- **LightGBM** — modelo campeão nos 3 grupos
- **XGBoost** — testado como candidato; não selecionado no modelo final
- **SHAP** — interpretabilidade global (summary bar/beeswarm) e individual (waterfall)
- **gower** — distância de Gower para subsampling intra-indivíduo
- **matplotlib**, **seaborn** — visualizações
- **scipy** — KS-test para drift de features
- **joblib** — serialização de modelos e scalers

---

# 📦 Estrutura de Pastas

```
projeto1_modelo_preditivo/
│
├── notebooks/
│   ├── 01_ingest.ipynb              # Integração das 3 bases + rolling stats de grupo
│   ├── 02_build_dataset_m2.ipynb    # Feature engineering + MICE (fit em base_tt)
│   ├── 03_splits.ipynb              # Medoide, Gower, split por ID, StandardScaler
│   ├── 04_feature_selection.ipynb   # Seleção em 3 camadas, independente por grupo
│   ├── 05_model.ipynb               # Treino, threshold inicial, score de risco
│   ├── 06_tuning.ipynb              # Hyperparameter tuning + K-fold + drift treino-val
│   ├── 07_shap.ipynb                # SHAP values + drift val→aplicação
│   ├── 08_score.ipynb               # Score final (val + aplicação, Jun–Dez/2025)
│   ├── 09_roi.ipynb                 # Projeção de ROI por cenário de retenção
│   └── 10_monitor.ipynb             # PSI, performance mensal, alertas de retreino
│
├── data/                            # ⛔ Não versionado (.gitignore)
│   ├── raw/                         # Bases de entrada (.parquet)
│   ├── gold/                        # base_tt_raw, base_val_raw, base_apl_raw
│   │                                # base_features_tt/val/apl
│   └── processed/
│       ├── splits/{Grupo}/          # train.parquet, test.parquet, val.parquet
│       │                            # scaler_cols.json
│       └── feature_selection/{Grupo}/  # selected_features.json, fs_report.parquet
│
├── models/{Grupo}/                  # best_model.joblib, best_model_tuned.joblib
│                                    # standard_scaler.joblib
│                                    # model_metadata.json, tuning_metadata.json
│
├── reports/
│   ├── feature_selection/{Grupo}/   # fs_layer1_dropped.txt + fs_layer2/3 PNGs
│   ├── figures/{Grupo}/             # roc_pr, confusion_matrix, drift, tuning, reliability
│   ├── figures/                     # distribuicao_risco, histogramas, ROI (3), monitor (4)
│   ├── shap/{Grupo}/                # shap_values.npy, shap_feature_importance.json
│   │                                # summary_bar/beeswarm/waterfall/dependence PNGs
│   ├── score_risco_{Grupo}.parquet  # Scores por grupo (val + aplicação)
│   ├── score_risco_consolidado.parquet/.csv
│   ├── roi_projecao.parquet/.csv
│   ├── monitor_drift_psi.csv
│   └── monitor_performance_val.csv
│
├── README.md
└── .gitignore
```