# ==============================================================
# T1 - Spot-Checking de Algoritmos de Aprendizado de Máquina
# Dataset: Online Shoppers Purchasing Intention (UCI, ID=468)
# URL: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
# Disciplina: INF01017 - Aprendizado de Máquina (UFRGS)
# ==============================================================

# ==============================================================
# SEÇÃO 0: INSTALAÇÃO DE DEPENDÊNCIAS E IMPORTS
# ==============================================================
# No Google Colab, descomente e execute a linha abaixo antes de tudo:
# !pip install ucimlrepo imbalanced-learn -q

import sys
import io
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline

import warnings
warnings.filterwarnings('ignore')

# Semente global — fixada para reprodutibilidade de todos os passos aleatórios
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ==============================================================
# LOG DE EXECUÇÃO — captura tudo que vai ao terminal
# ==============================================================

class _Tee:
    """Escreve simultaneamente para o stdout original e para um buffer em memória."""
    def __init__(self, stream):
        self._stream = stream
        self._buf = io.StringIO()

    def write(self, data):
        self._stream.write(data)
        self._buf.write(data)

    def flush(self):
        self._stream.flush()

    def getvalue(self):
        return self._buf.getvalue()

_tee = _Tee(sys.stdout)
sys.stdout = _tee
_exec_start = datetime.now()

# ==============================================================
# SEÇÃO 1: CARREGAMENTO DOS DADOS
# ==============================================================

from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=468)
X_raw = dataset.data.features          
y_raw = dataset.data.targets.squeeze()

# OperatingSystems, Browser, Region e TrafficType são códigos inteiros sem
# relação de ordem (Region=9 não é "maior" que Region=1): convertidos para
# texto para serem tratados como categóricos na EDA e no pré-processamento.
CODED_CAT_COLS = ['OperatingSystems', 'Browser', 'Region', 'TrafficType']
X_raw = X_raw.copy()
X_raw[CODED_CAT_COLS] = X_raw[CODED_CAT_COLS].astype(str)

print("=== Dados carregados ===")
print(f"Instâncias: {X_raw.shape[0]} | Features: {X_raw.shape[1]}")
print(f"Distribuição do alvo:\n{y_raw.value_counts()}\n")

# ==============================================================
# SEÇÃO 2: ANÁLISE EXPLORATÓRIA DOS DADOS (EDA)
# ==============================================================

df_eda = X_raw.copy()
df_eda['Revenue'] = y_raw.values

# --- 2.1 Visão geral ---
print("=== Primeiras linhas ===")
print(df_eda.head())

print("\n=== Informações gerais ===")
df_eda.info()

print("\n=== Estatísticas descritivas (variáveis numéricas) ===")
print(df_eda.describe().T)

# --- 2.2 Valores faltantes ---
missing = df_eda.isnull().sum()
print("\n=== Valores faltantes ===")
if missing.sum() == 0:
    print("Nenhum valor faltante encontrado.")
else:
    print(missing[missing > 0])

# --- 2.2b Linhas duplicadas ---
n_dup = df_eda.duplicated().sum()
n_dup_feat = X_raw.duplicated().sum()
print("\n=== Linhas duplicadas ===")
print(f"Duplicatas exatas: {n_dup} ({n_dup/len(df_eda)*100:.2f}%)")
print(f"Duplicatas com alvo conflitante: {n_dup_feat - n_dup}")
if n_dup > 0:
    print("Exemplos mais frequentes:")
    print(df_eda[df_eda.duplicated(keep=False)]
          .value_counts().head(3).to_string())

# --- 2.3 Distribuição da variável alvo ---
fig, ax = plt.subplots(figsize=(6, 4))
counts = y_raw.value_counts()
ax.bar(['Não comprou (False)', 'Comprou (True)'], counts.values,
       color=['steelblue', 'salmon'])
for i, v in enumerate(counts.values):
    ax.text(i, v + 30, f"{v} ({v/len(y_raw)*100:.1f}%)", ha='center', fontsize=11)
ax.set_title("Distribuição da Variável Alvo (Revenue)")
ax.set_ylabel("Contagem")
plt.tight_layout()
plt.savefig(LOGS_DIR / "eda_01_target_distribution.png", dpi=150)
plt.close()

# --- 2.4 Identificar tipos de colunas ---
num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
# exclude=[np.number] em vez de include=['object', ...]: no pandas 3 texto tem
# dtype 'str', e incluí-lo via 'object' está depreciado.
cat_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"\nVariáveis numéricas ({len(num_cols)}): {num_cols}")
print(f"Variáveis categóricas/booleanas ({len(cat_cols)}): {cat_cols}")

# --- 2.5 Distribuições das variáveis numéricas ---
n_cols_grid = 3
n_rows_grid = int(np.ceil(len(num_cols) / n_cols_grid))
fig, axes = plt.subplots(n_rows_grid, n_cols_grid,
                          figsize=(15, n_rows_grid * 4))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    axes[i].hist(X_raw[col].dropna(), bins=40, color='steelblue', edgecolor='white')
    axes[i].set_title(col, fontsize=9)
    axes[i].set_xlabel("Valor")
    axes[i].set_ylabel("Freq.")
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Distribuição das Variáveis Numéricas", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(LOGS_DIR / "eda_02_num_distributions.png", dpi=150, bbox_inches='tight')
plt.close()

# --- 2.6 Análise de variáveis categóricas ---
print("\n=== Contagem por categoria ===")
for col in cat_cols:
    print(f"\n{col}:\n{X_raw[col].value_counts()}")

# --- 2.7 Matriz de correlação ---
fig, ax = plt.subplots(figsize=(14, 10))
corr = X_raw[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
            linewidths=0.5, vmin=-1, vmax=1, ax=ax)
ax.set_title("Matriz de Correlação — Variáveis Numéricas")
plt.tight_layout()
plt.savefig(LOGS_DIR / "eda_03_correlation_matrix.png", dpi=150)
plt.close()

# --- 2.8 Relação de variáveis-chave com o alvo ---
# Selecionadas as métricas do Google Analytics (PageValues, BounceRates,
# ExitRates) e as durações por tipo de página: são as variáveis numéricas com
# maior correlação com Revenue e as que melhor descrevem o engajamento na sessão.
highlight_features = ['PageValues', 'BounceRates', 'ExitRates',
                       'ProductRelated_Duration', 'Administrative_Duration',
                       'Informational_Duration']
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(highlight_features):
    df_eda.boxplot(column=col, by='Revenue', ax=axes[i], grid=False)
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel("Revenue")
    axes[i].set_ylabel("Valor")
plt.suptitle("Distribuição de Variáveis-Chave por Classe (Revenue)", fontsize=12)
plt.tight_layout()
plt.savefig(LOGS_DIR / "eda_04_feature_vs_target.png", dpi=150)
plt.close()

# --- 2.9 Síntese da EDA ---
# 1. Alvo desbalanceado: 84,5% sem compra x 15,5% com compra (5,46:1) ->
#    Accuracy é enganosa; usar ROC-AUC/F1 e balancear o treino (SMOTENC).
# 2. Sem valores faltantes; 125 duplicatas exatas (1,01%), sem alvo
#    conflitante, quase todas sessões de 1 página sem compra -> mantidas.
# 3. Numéricas com forte assimetria à direita (skew de 2 a 7,6) e escalas
#    muito diferentes (ProductRelated_Duration chega a ~64 mil s; taxas em
#    [0, 0,2]) -> padronização necessária para kNN, LR e SMOTENC.
# 4. PageValues é o atributo mais informativo (corr. 0,49 com o alvo): é zero
#    em 77,9% das sessões, e a taxa de compra é 56,3% quando PageValues > 0
#    contra 3,9% quando = 0. Como é calculado a partir de páginas visitadas
#    antes de transações, pode carregar informação próxima do próprio alvo
#    (possível vazamento) — ponto a discutir no relatório.
# 5. ExitRates (-0,21) e BounceRates (-0,15) se relacionam negativamente com a
#    compra; as durações/contagens de páginas, positivamente (0,07 a 0,16).
# 6. Pares redundantes: BounceRates x ExitRates (0,91) e ProductRelated x
#    ProductRelated_Duration (0,86). Mantidos no spot-checking (modelos de
#    árvore não são afetados), mas relevantes para LR e NB.
# 7. Categóricas: taxa de compra varia por mês (1,6% em Feb a 25,4% em Nov;
#    Jan e Apr ausentes) e é maior em novos visitantes (24,9%) que em
#    recorrentes (13,9%). OperatingSystems, Browser, Region e TrafficType são
#    códigos nominais (one-hot); TrafficType, Browser e OperatingSystems têm
#    categorias raras (< 30 sessões) -> handle_unknown='ignore' no One-Hot.

# ==============================================================
# SEÇÃO 3: PRÉ-PROCESSAMENTO DOS DADOS
# ==============================================================

X = X_raw.copy()
y = y_raw.astype(int)  

# --- 3.1 Tratamento de valores faltantes ---
# O dataset não possui valores faltantes (ver 2.2); o bloco abaixo é mantido
# por robustez. Atribuição explícita (X[col] = ...) em vez de inplace=True:
# no pandas 3 (Copy-on-Write), X[col].fillna(..., inplace=True) não altera X.
for col in X.select_dtypes(include=[np.number]).columns:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].median())
for col in X.select_dtypes(exclude=[np.number]).columns:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].mode()[0])

# --- 3.2 Codificação de variáveis categóricas ---
# Atributos nominais que recebem One-Hot Encoding (dentro do pipeline, após o
# SMOTENC — ver 4.4): 'Month', 'VisitorType', 'Weekend', 'OperatingSystems',
# 'Browser', 'Region' e 'TrafficType'. Aqui eles são mantidos como texto.
X['Weekend'] = X['Weekend'].astype(str)
CAT_FEATURES = ['Month', 'VisitorType', 'Weekend'] + CODED_CAT_COLS
NUM_FEATURES = [c for c in X.columns if c not in CAT_FEATURES]
X = X[NUM_FEATURES + CAT_FEATURES]

n_onehot = len(NUM_FEATURES) + sum(X[c].nunique() for c in CAT_FEATURES)
print("=== Atributos para modelagem ===")
print(f"Numéricos ({len(NUM_FEATURES)}): {NUM_FEATURES}")
print(f"Categóricos ({len(CAT_FEATURES)}): {CAT_FEATURES}")
print(f"Colunas após One-Hot (no pipeline): {n_onehot}")

# --- 3.3 Verificação de desbalanceamento ---
class_dist = pd.Series(y).value_counts().sort_index()
ratio = class_dist[0] / class_dist[1]
print(f"\n=== Distribuição de classes ===")
print(class_dist)
print(f"Razão negativo:positivo = {ratio:.2f}:1")

X_final = X.copy()
y_final = y.copy()

print(f"\n=== Dados prontos para modelagem: X{X_final.shape}, y{y_final.shape} ===")

# ==============================================================
# SEÇÃO 4: DEFINIÇÃO DOS ALGORITMOS E ESTRATÉGIA DE AVALIAÇÃO
# ==============================================================

# --- 4.1 Problema e abordagem ---
# Tarefa: Classificação binária (Revenue: compra=1 / não compra=0)
# Justificativa: a variável alvo é booleana (a sessão terminou ou não em
# compra), portanto discreta com duas classes — não há valor contínuo a
# prever, o que descarta regressão. Do ponto de vista do negócio, a pergunta
# é "este visitante vai comprar?", e a probabilidade estimada pelo
# classificador permite ordenar sessões por propensão de compra (ex.: para
# ofertas direcionadas), o que também motiva o uso da ROC-AUC como métrica.

# --- 4.2 Métricas de avaliação ---
# Dataset desbalanceado (~84/16) -> Accuracy sozinha é enganosa.
# Métrica PRINCIPAL: ROC-AUC (robusto a desbalanceamento, mede capacidade
#   discriminativa geral sem depender de threshold).
# Métricas secundárias: F1-macro, Precision, Recall (para análise complementar).
SCORING = {
    'roc_auc':   'roc_auc',
    'f1_macro':  'f1_macro',
    'accuracy':  'accuracy',
    'precision': 'precision',
    'recall':    'recall',
}
METRIC_PRINCIPAL = 'test_roc_auc'

# --- 4.3 Estratégia de validação cruzada ---
N_SPLITS  = 10  # k-folds
N_REPEATS = 10  # repetições com shuffles diferentes

# Processos paralelos no cross_validate. Cada processo carrega sua própria
# cópia dos dados; com -1 (todos os núcleos) houve MemoryError. Pode ser
# aumentado se houver RAM sobrando, ou reduzido em caso de MemoryError.
N_JOBS = 4
cv_strategy = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_SEED
)

# --- 4.4 Função auxiliar de pipeline ---
# Após o 1º ColumnTransformer as colunas ficam na ordem NUM_FEATURES +
# CAT_FEATURES, então as categóricas ocupam os índices abaixo.
CAT_IDX = list(range(len(NUM_FEATURES), len(NUM_FEATURES) + len(CAT_FEATURES)))

def to_float(X):
    """Converte a saída do One-Hot (dtype object, herdado do SMOTENC) para
    float64 — ocupa ~4x menos memória em cada processo paralelo."""
    return np.asarray(X, dtype=np.float64)

def make_pipeline(model):
    """
    Cria ImbPipeline: StandardScaler -> SMOTENC -> One-Hot -> modelo.
    - O scaler vem antes do SMOTENC porque ele usa distâncias (kNN) para
      gerar amostras sintéticas; sem padronização, atributos de grande escala
      (ex.: ProductRelated_Duration) dominariam a vizinhança.
    - SMOTENC interpola apenas os atributos numéricos; nos categóricos, a
      amostra sintética recebe a categoria mais frequente entre os vizinhos,
      evitando valores impossíveis (ex.: Month_Nov = 0.37).
    - O One-Hot vem depois, sobre as categorias originais. handle_unknown=
      'ignore' cobre categorias raras ausentes no fold de treino.
    Todas as etapas são ajustadas apenas no fold de treino a cada iteração
    de CV, eliminando data leakage.
    """
    scaler = ColumnTransformer([
        ('num', StandardScaler(), NUM_FEATURES),
        ('cat', 'passthrough',    CAT_FEATURES),
    ])
    onehot = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
         CAT_IDX),
    ], remainder='passthrough')
    return ImbPipeline(steps=[
        ('scaler',  scaler),
        ('smotenc', SMOTENC(categorical_features=CAT_IDX, random_state=RANDOM_SEED)),
        ('onehot',  onehot),
        ('float',   FunctionTransformer(to_float)),
        ('model',   model)
    ])

# --- 4.5 Algoritmos selecionados ---
algorithms = [
    # Linear / Paramétrico / Otimização
    ("Logistic Regression",
     make_pipeline(LogisticRegression(
         max_iter=1000, random_state=RANDOM_SEED))),

    # Árvore / Não-paramétrico / Instâncias
    ("Decision Tree",
     make_pipeline(DecisionTreeClassifier(
         random_state=RANDOM_SEED))),

    # Árvore com poda leve (variação mínima de hiperparâmetro)
    ("Decision Tree (depth=10)",
     make_pipeline(DecisionTreeClassifier(
         max_depth=10, random_state=RANDOM_SEED))),

    # Baseado em instâncias / Não-paramétrico
    ("K-Nearest Neighbors (k=5)",
     make_pipeline(KNeighborsClassifier(n_neighbors=5))),

    # Probabilístico / Generativo
    ("Naive Bayes",
     make_pipeline(GaussianNB())),

    # Ensemble / Não-linear / Bagging
    ("Random Forest",
     make_pipeline(RandomForestClassifier(
         n_estimators=100, random_state=RANDOM_SEED))),

    # Ensemble / Não-linear / Boosting
    ("Gradient Boosting",
     make_pipeline(GradientBoostingClassifier(
         n_estimators=100, random_state=RANDOM_SEED))),
]

print(f"\n=== {len(algorithms)} algoritmos definidos ===")
for name, _ in algorithms:
    print(f"  - {name}")

# ==============================================================
# SEÇÃO 5: SPOT-CHECKING — TREINAMENTO E AVALIAÇÃO
# ==============================================================

results = {}  

print(f"\n=== Iniciando Spot-Checking ===")
print(f"Validação: RepeatedStratifiedKFold "
      f"({N_SPLITS} folds × {N_REPEATS} repetições = "
      f"{N_SPLITS * N_REPEATS} avaliações/algoritmo)\n")

for name, pipeline in algorithms:
    print(f"[...] {name}", end="  ->  ", flush=True)
    cv_res = cross_validate(
        estimator=pipeline,
        X=X_final,
        y=y_final,
        cv=cv_strategy,
        scoring=SCORING,
        n_jobs=N_JOBS,
        return_train_score=False
    )
    results[name] = cv_res
    mean_auc = cv_res[METRIC_PRINCIPAL].mean()
    std_auc  = cv_res[METRIC_PRINCIPAL].std()
    print(f"ROC-AUC = {mean_auc:.4f} ± {std_auc:.4f}")

print("\nSpot-checking concluído!")

# ==============================================================
# SEÇÃO 6: SUMARIZAÇÃO DOS RESULTADOS
# ==============================================================

# --- 6.1 Tabela resumo (média ± desvio-padrão por métrica) ---
metric_map = {
    'test_roc_auc':   'ROC-AUC',
    'test_f1_macro':  'F1-Macro',
    'test_accuracy':  'Accuracy',
    'test_precision': 'Precision',
    'test_recall':    'Recall',
}

rows = []
for algo_name, cv_res in results.items():
    row = {'Algoritmo': algo_name}
    for key, label in metric_map.items():
        vals = cv_res[key]
        row[f'{label} Média'] = round(float(vals.mean()), 4)
        row[f'{label} DP']    = round(float(vals.std()),  4)
    rows.append(row)

df_summary = (pd.DataFrame(rows)
               .set_index('Algoritmo')
               .sort_values('ROC-AUC Média', ascending=False))

print("\n=== Tabela de Resultados (ordenada por ROC-AUC) ===")
print(df_summary.to_string())
df_summary.to_csv(LOGS_DIR / "resultados_spot_checking.csv")

# --- 6.2 Box plot — ROC-AUC ---
auc_df = pd.DataFrame(
    {name: res[METRIC_PRINCIPAL] for name, res in results.items()}
)
# Reordenar colunas pelo ranking da tabela
auc_df = auc_df[df_summary.index.tolist()]

fig, ax = plt.subplots(figsize=(14, 6))
colors = sns.color_palette("muted", n_colors=len(algorithms))
bp = ax.boxplot(
    [auc_df[col].values for col in auc_df.columns],
    tick_labels=auc_df.columns,
    patch_artist=True,
    medianprops=dict(color='black', linewidth=2)
)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_title("Distribuição do ROC-AUC por Algoritmo", fontsize=13)
ax.set_xlabel("Algoritmo")
ax.set_ylabel("ROC-AUC")
ax.tick_params(axis='x', rotation=30)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(LOGS_DIR / "resultados_01_boxplot_auc.png", dpi=150)
plt.close()

# --- 6.3 Violin plot — F1-Macro ---
f1_df = pd.DataFrame(
    {name: res['test_f1_macro'] for name, res in results.items()}
)
f1_df = f1_df[df_summary.index.tolist()]
f1_melted = f1_df.melt(var_name='Algoritmo', value_name='F1-Macro')

fig, ax = plt.subplots(figsize=(14, 6))
sns.violinplot(data=f1_melted, x='Algoritmo', y='F1-Macro',
               order=df_summary.index.tolist(),
               palette='muted', inner='quartile', ax=ax)
ax.set_title("Distribuição do F1-Macro por Algoritmo", fontsize=13)
ax.set_xlabel("Algoritmo")
ax.set_ylabel("F1-Macro")
ax.tick_params(axis='x', rotation=30)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(LOGS_DIR / "resultados_02_violin_f1.png", dpi=150)
plt.close()

# --- 6.4 Heatmap das médias de todas as métricas ---
mean_cols = [c for c in df_summary.columns if 'Média' in c]
df_heat = df_summary[mean_cols].copy()
df_heat.columns = [c.replace(' Média', '') for c in df_heat.columns]

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(df_heat, annot=True, fmt=".4f", cmap='YlGnBu',
            linewidths=0.5, ax=ax, vmin=0, vmax=1)
ax.set_title("Médias das Métricas de Avaliação por Algoritmo", fontsize=12)
plt.tight_layout()
plt.savefig(LOGS_DIR / "resultados_03_heatmap_metrics.png", dpi=150)
plt.close()

# ==============================================================
# SEÇÃO 7: SELEÇÃO DOS ALGORITMOS MAIS PROMISSORES
# ==============================================================

print("\n=== TOP 3 Algoritmos Mais Promissores (por ROC-AUC) ===")
top3 = df_summary.head(3)
display_cols = ['ROC-AUC Média', 'ROC-AUC DP', 'F1-Macro Média', 'F1-Macro DP']
print(top3[display_cols].to_string())

# --- Gráfico de barras para facilitar visualização do TOP 3 ---
fig, ax = plt.subplots(figsize=(8, 5))
top3['ROC-AUC Média'].plot(
    kind='barh', xerr=top3['ROC-AUC DP'],
    color=['gold', 'silver', '#cd7f32'], edgecolor='grey', ax=ax
)
ax.set_title("TOP 3 Algoritmos — ROC-AUC Médio ± DP")
ax.set_xlabel("ROC-AUC")
ax.set_xlim(0, 1)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(LOGS_DIR / "resultados_04_top3.png", dpi=150)
plt.close()

print("\n=== Script concluído. Verifique os arquivos CSV e PNG gerados. ===")

# ==============================================================
# SALVAR LOG DA EXECUÇÃO
# ==============================================================

sys.stdout = _tee._stream  

log_path = LOGS_DIR / "execution_log.txt"
with open(log_path, 'w', encoding='utf-8') as _f:
    _f.write(f"Execução: {_exec_start.strftime('%Y-%m-%d %H:%M:%S')} "
             f"→ {datetime.now().strftime('%H:%M:%S')}\n")
    _f.write("=" * 60 + "\n\n")
    _f.write(_tee.getvalue())

print(f"Log da execução salvo em: {log_path}")
