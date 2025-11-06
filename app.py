# app.py — auto-load CSV & auto insights (no upload needed)
# -*- coding: utf-8 -*-
# ✅ 일반인도 이해할 수 있도록 주요 전문용어 옆에 쉬운 설명(주석)을 추가했습니다.
# ✅ 그래프/표 아래의 자동 해석 문장을 더 자세히, 실무 시사점까지 나오도록 개선했습니다.

import os
import io
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 표준화: 각 특징을 평균0, 표준편차1로 맞춰 비교 쉽게 함
from sklearn.metrics import (
    classification_report,   # 정밀도/재현율/F1/정확도 등 요약표
    confusion_matrix,        # 혼동행렬: 예측과 실제가 맞/틀린 개수 표
    roc_auc_score,           # AUC: 0~1, 1에 가까울수록 좋은 분류 성능(면적)
    RocCurveDisplay,         # ROC 곡선: 민감도(재현율)와 위양성률 관계
)
from sklearn.linear_model import LogisticRegression   # 로지스틱 회귀: 확률을 예측하는 선형 분류기
from sklearn.ensemble import RandomForestClassifier   # 랜덤포레스트: 여러 결정트리를 합쳐 예측하는 앙상블

st.set_page_config(page_title="Software Defect Dataset Explorer", layout="wide")

# -----------------------
# Locate CSV automatically (same folder preferred)
# -----------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "software_defects_multilang_ast_1000.csv"),
    os.path.join(SCRIPT_DIR, "data", "software_defects_multilang_ast_1000.csv"),
    "/mnt/data/software_defects_multilang_ast_1000.csv",
]

def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

CSV_PATH = _first_existing(CSV_CANDIDATES)

st.title("🧪 Software Defects (Multilang) — 시각 분석 & 간단 모델")

# Small status banner at the top-right
status_col1, status_col2 = st.columns([1, 2])
with status_col2:
    if CSV_PATH:
        st.caption(f"데이터 소스: `{os.path.basename(CSV_PATH)}` (자동 로드)")
    else:
        st.caption("데이터 소스: (없음) — 같은 폴더에 CSV를 두면 자동으로 로드됩니다.")

@st.cache_data(show_spinner=True)
def load_data_from_path(path: str) -> pd.DataFrame:
    # 메모리에 읽어와서 Pandas로 파싱
    with open(path, "rb") as f:
        buf = io.BytesIO(f.read())
    df = pd.read_csv(buf)
    return df

# -----------------------
# Glossary (쉬운 말 용어 설명)
# -----------------------

def glossary_md() -> str:
    return (
        """
**용어 설명 (쉬운 말)**
- **결함(defect)**: 문제가 있는 코드(버그가 있을 확률이 높다고 표시된 항목)
- **LOC(lines_of_code)**: 코드 줄 수. 길수록 복잡해질 수 있음
- **순환 복잡도(cyclomatic_complexity)**: 분기(If/반복) 등으로 복잡한 정도를 수치로 표현
- **토큰 수(token_count)**: 코드 단위를 쪼갠 최소 단위 개수(길이/복잡도의 다른 표현)
- **if/return/함수호출 수**: 각각 조건문/반환/다른 함수 부르는 횟수
- **AST 노드 수(ast_nodes)**: 코드를 트리로 표현했을 때의 요소 개수(구조적 복잡도)
- **표본 샘플(sample)**: 데이터의 한 행(하나의 함수/코드 조각)
- **클래스 불균형**: 결함(1)과 정상(0)의 비율 차이가 큰 상태
- **상관관계**: 두 수치가 같이 오르내리는 경향(인과관계와는 다름)
- **ROC/AUC**: 임계값을 바꿔가며 살펴본 분류 성능 곡선/면적(0.5=운, 0.7~0.8 무난, 0.8~0.9 양호)
- **정밀도(Precision)**: 결함이라고 한 것 중 실제 결함 비율(거짓 경보를 얼마나 줄였나)
- **재현율(Recall)**: 실제 결함 중 찾아낸 비율(놓친 결함이 얼마나 적은가)
- **F1**: 정밀도와 재현율의 균형 지표
- **혼동행렬**: 예측과 실제의 맞춤/틀림을 표로 요약
- **표준화(Standardization)**: 특징들을 같은 스케일로 맞춤(공정한 비교)
- **로지스틱 회귀**: 선형 방식으로 결함일 확률을 계산하는 간단하고 빠른 모델
- **랜덤포레스트**: 여러 결정나무를 묶어 과적합을 줄이고 성능을 높이는 모델
        """
    )

with st.expander("ℹ️ 용어 설명 열기(초보자용)"):
    st.markdown(glossary_md())

# -----------------------
# Load data or stop with a helpful message
# -----------------------
if not CSV_PATH:
    st.error(
        "CSV 파일을 찾을 수 없습니다. `app.py`와 같은 폴더에 "
        "`software_defects_multilang_ast_1000.csv`를 두고 다시 실행하세요."
    )
    st.stop()

try:
    data = load_data_from_path(CSV_PATH)
except Exception as e:
    st.exception(e)
    st.stop()

st.caption("다국어 함수 코드의 정적 분석 지표로 결함(defect)을 탐색하고 분류 모델을 시도합니다.")

# -----------------------
# Basic validation
# -----------------------
required_cols = [
    "function_name","code","language","lines_of_code","cyclomatic_complexity",
    "token_count","num_ifs","num_returns","num_func_calls","ast_nodes","defect"
]
missing = [c for c in required_cols if c not in data.columns]
if missing:
    st.error(f"필수 컬럼이 없습니다: {missing}")
    st.stop()

# 숫자형으로 변환(에러는 NaN으로)
numeric_cols = [
    "lines_of_code","cyclomatic_complexity","token_count",
    "num_ifs","num_returns","num_func_calls","ast_nodes","defect"
]
for c in numeric_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce")

data = data.dropna(subset=numeric_cols).copy()
if data.empty:
    st.error("모든 행이 누락값으로 제거되어 데이터가 비었습니다. 전처리를 확인하세요.")
    st.stop()

# 타깃은 0/1 정수로 보정
try:
    data["defect"] = data["defect"].astype(int)
except Exception:
    st.error("`defect` 컬럼을 정수형으로 변환할 수 없습니다. 0/1로 구성되어 있는지 확인하세요.")
    st.stop()

# -----------------------
# Top metrics (쉽게 읽히는 핵심 숫자)
# -----------------------
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("전체 샘플 수", f"{len(data):,}")
col_b.metric("프로그래밍 언어 수", data["language"].nunique())
col_c.metric("결함 비율(%)", f"{data['defect'].mean()*100:.1f}")
col_d.metric("평균 LOC", f"{data['lines_of_code'].mean():.2f}")

st.markdown("---")

# -----------------------
# Filters (sidebar)
# -----------------------
st.sidebar.header("필터")
langs = sorted(data["language"].unique().tolist())
sel_langs = st.sidebar.multiselect("언어 선택", langs, default=langs)
loc_max = int(data["lines_of_code"].max())
loc_range = st.sidebar.slider("라인 수(LOC) 범위", 0, loc_max, (0, loc_max))
show_code = st.sidebar.checkbox("표 미리보기에 code 컬럼 포함", value=False)

# 필터 적용
_df = data.query("language in @sel_langs").copy()
_df = _df[(_df["lines_of_code"] >= loc_range[0]) & (_df["lines_of_code"] <= loc_range[1])]

if _df.empty:
    st.info("현재 필터로는 행이 없습니다. 필터를 조정하세요.")
    st.stop()

st.subheader("📄 데이터 미리보기")
preview_cols = [c for c in _df.columns if c != "code"] if not show_code else _df.columns
st.dataframe(_df[preview_cols].head(50), use_container_width=True)

# -----------------------
# Insight helpers (상세 해석)
# -----------------------

def _pct(n, d):
    return 0 if d == 0 else round(n / d * 100, 1)


def _skew_text(skew: float) -> str:
    if skew > 0.5:
        return "분포의 꼬리가 오른쪽(큰 값)으로 길어 평균이 중앙값보다 커지는 경향"
    if skew < -0.5:
        return "분포의 꼬리가 왼쪽(작은 값)으로 길어 평균이 중앙값보다 작아지는 경향"
    return "대체로 좌우 대칭에 가까운 분포"


def insight_language_counts(df: pd.DataFrame) -> str:
    total = int(df.shape[0])
    vc = df["language"].value_counts()
    top = vc.head(3)
    parts = [
        "표본 수 상위 언어: " + ", ".join([f"{idx}({_pct(cnt, total)}%)" for idx, cnt in top.items()])
    ]
    if vc.nunique() > 1:
        share = _pct(vc.iloc[0], total)
        if share >= 60:
            parts.append("특정 언어에 표본이 많이 몰려 있어 결과가 그 언어에 치우칠 수 있습니다. 가능하면 표본을 보강하거나 가중치/교차검증을 권장합니다.")
        elif share <= 35:
            parts.append("언어별 표본이 비교적 고르게 분포하여 일반화에 유리합니다.")
    parts.append("샘플 불균형은 모델이 소수 언어를 과소학습할 위험을 키웁니다.")
    return "\n- " + "\n- ".join(parts)


def insight_class_ratio(df: pd.DataFrame) -> str:
    pos = int((df["defect"] == 1).sum())
    neg = int((df["defect"] == 0).sum())
    total = pos + neg
    pos_p = _pct(pos, total)
    parts = [
        f"결함(1) 비율: {pos_p}%(결함 {pos}개 / 정상 {neg}개).",
        ("심한 클래스 불균형 가능성 → 학습 시 `class_weight='balanced'` 또는 리샘플링(SMOTE/언더샘플링) 고려." if pos_p <= 30 or pos_p >= 70 else "불균형이 크지 않아 기본 설정으로도 무난합니다."),
        "업무상 중요한 오류 유형이 있다면 재현율(놓치지 않기)을 우선할지, 정밀도(거짓 경보 줄이기)를 우선할지 목표를 정하세요."
    ]
    return "\n- " + "\n- ".join(parts)


def insight_histogram(df: pd.DataFrame, col: str) -> str:
    q1, q2, q3 = df[col].quantile([0.25, 0.5, 0.75])
    mean = df[col].mean()
    skew = float(df[col].skew())
    iqr = float(q3 - q1)
    parts = [
        f"중앙값 {q2:.2f}, 평균 {mean:.2f} → {_skew_text(skew)}.",
        f"사분위 범위(IQR) {iqr:.2f}. IQR 기준으로 이상치가 많다면 품질 규칙/코드 스타일 점검이 필요할 수 있습니다.",
    ]
    if iqr > 0 and (df[col] > q3 + 1.5 * iqr).any():
        parts.append("상한(Upper fence) 밖의 큰 값 이상치가 존재합니다 → 과도한 복잡도/긴 함수 가능성.")
    if iqr > 0 and (df[col] < q1 - 1.5 * iqr).any():
        parts.append("하한(Lower fence) 밖의 작은 값 이상치가 존재합니다 → 자동 생성/템플릿 코드 여부 확인.")
    return "\n- " + "\n- ".join(parts)


def insight_box_by_lang(df: pd.DataFrame, col: str) -> str:
    med = df.groupby("language")[col].median().sort_values(ascending=False)
    var = df.groupby("language")[col].var().sort_values(ascending=False)
    parts = []
    if not med.empty:
        parts.append(f"중앙값 최댓값: {med.index[0]} ({med.iloc[0]:.2f}) → 해당 언어에서 상대적으로 {col}가 높습니다.")
    if not var.empty and not np.isnan(var.iloc[0]):
        parts.append(f"분산 최댓값: {var.index[0]} ({var.iloc[0]:.2f}) → 언어별 편차가 큰 편입니다. 표준/가이드 정비를 검토하세요.")
    if not parts:
        parts.append("언어별 차이가 크지 않습니다 → 공통 정책으로 관리해도 무방.")
    return "\n- " + "\n- ".join(parts)


def insight_corr(corr_df: pd.DataFrame) -> str:
    if "defect" not in corr_df.columns:
        return "- 결함 열이 없어 상관 분석을 생략했습니다."
    s = corr_df["defect"].drop(labels=["defect"])  # type: ignore
    s_sorted = s.sort_values(ascending=False)
    s_abs_top = s.abs().sort_values(ascending=False).head(3)
    parts = [
        "결함(defect)과의 상관 상위 특징(절대값 기준): " + ", ".join([f"{idx}(r={val:.2f})" for idx, val in s_abs_top.items()]),
        "상관이 높다고 원인이 되는 것은 아닙니다(인과와는 다름). 다만 규칙/리뷰 포인트로 우선 검토하기에 적합합니다.",
    ]
    if not s_sorted.empty and s_sorted.iloc[0] > 0:
        parts.append("양(+)의 상관은 값이 클수록 결함일 가능성이 높다는 뜻, 음(-)의 상관은 반대입니다.")
    return "\n- " + "\n- ".join(parts)


def insight_importances(imp_df: pd.DataFrame) -> str:
    if imp_df.empty:
        return "- 중요도 계산 결과가 없습니다."
    top = imp_df.sort_values("importance", ascending=False).head(5)
    parts = [
        "모델이 중요하게 본 특징(상위 5): " + ", ".join([f"{r.feature}({r.importance:.3f})" for r in top.itertuples()]),
        "중요도는 데이터와 모델에 따라 달라집니다. 정책/리뷰 체크리스트를 만들 때 상위 특징부터 반영해 보세요.",
    ]
    return "\n- " + "\n- ".join(parts)


def insight_roc(auc: float) -> str:
    if np.isnan(auc):
        return "- AUC 계산 불가 (확률 예측 없음)."
    tier = (
        "양호" if auc >= 0.80 else ("무난" if auc >= 0.70 else ("개선 필요" if auc >= 0.60 else "낮음"))
    )
    parts = [
        f"AUC={auc:.3f} → 성능 등급: {tier}.",
        "AUC는 임계값 전반의 평균적 성능을 나타냅니다. 실제 업무에서는 정밀도·재현율의 균형(F1)이나 원하는 목표치에 맞춘 임계값 조정이 중요합니다.",
    ]
    return "\n- " + "\n- ".join(parts)


def insight_per_lang_table(grp: pd.DataFrame) -> str:
    if grp.empty:
        return "- 언어별 집계 결과가 없습니다."
    hi = grp.sort_values("defect_rate", ascending=False).iloc[0]
    lo = grp.sort_values("defect_rate", ascending=True).iloc[0]
    parts = [
        f"결함률 최고: {hi['language']} ({hi['defect_rate']:.2f}%) → 우선 개선 대상.",
        f"결함률 최저: {lo['language']} ({lo['defect_rate']:.2f}%) → 모범 사례 벤치마크.",
    ]
    big = grp.sort_values("samples", ascending=False).iloc[0]
    parts.append(f"표본 최다 언어: {big['language']} ({int(big['samples'])}개) → 정책 변경 시 영향도가 큼.")
    return "\n- " + "\n- ".join(parts)

# -----------------------
# 분포 & 카운트 시각화
# -----------------------
st.subheader("📊 분포 & 카운트")
left, right = st.columns(2)
with left:
    st.markdown("**언어별 샘플 수**")
    lang_count = _df["language"].value_counts().reset_index()
    lang_count.columns = ["language", "count"]
    chart1 = alt.Chart(lang_count).mark_bar().encode(
        x=alt.X("language:N", sort="-y"),
        y="count:Q",
        tooltip=["language", "count"],
    )
    st.altair_chart(chart1, use_container_width=True)
    st.markdown("**그래프 해석**")
    st.markdown(insight_language_counts(_df))

with right:
    st.markdown("**결함(1) / 정상(0) 비율**")
    cls_count = _df["defect"].value_counts().sort_index().rename_axis("defect").reset_index(name="count")
    cls_count["label"] = cls_count["defect"].map({0: "정상(0)", 1: "결함(1)"})
    chart2 = alt.Chart(cls_count).mark_arc(innerRadius=40).encode(
        theta="count:Q",
        color=alt.Color("label:N", legend=None),
        tooltip=["label", "count"],
    )
    st.altair_chart(chart2, use_container_width=True)
    st.markdown("**그래프 해석**")
    st.markdown(insight_class_ratio(_df))

st.markdown("---")

# -----------------------
# Feature Explorer
# -----------------------
st.subheader("🧭 특징(Feature) 탐색")
feature_cols = [
    "lines_of_code","cyclomatic_complexity","token_count",
    "num_ifs","num_returns","num_func_calls","ast_nodes"
]
feat = st.selectbox("분포 확인할 수치 컬럼", feature_cols, index=0)

hist = alt.Chart(_df).mark_bar(opacity=0.8).encode(
    x=alt.X(f"{feat}:Q", bin=alt.Bin(maxbins=30)),
    y="count()",
    tooltip=[feat, alt.Tooltip("count()", title="count")],
)
st.altair_chart(hist, use_container_width=True)
st.markdown("**그래프 해석**")
st.markdown(insight_histogram(_df, feat))

box = alt.Chart(_df).mark_boxplot().encode(
    x="language:N",
    y=alt.Y(f"{feat}:Q"),
    tooltip=["language", feat],
)
st.altair_chart(box, use_container_width=True)
st.markdown("**그래프 해석**")
st.markdown(insight_box_by_lang(_df, feat))

st.markdown("---")

# -----------------------
# Correlation Heatmap
# -----------------------
st.subheader("🧪 상관관계 히트맵")
corr = _df[feature_cols + ["defect"]].corr(numeric_only=True)
corr_df = corr.reset_index().melt("index")
corr_df.columns = ["feature_x", "feature_y", "corr"]
heat = alt.Chart(corr_df).mark_rect().encode(
    x=alt.X("feature_x:N", sort=feature_cols + ["defect"]),
    y=alt.Y("feature_y:N", sort=feature_cols + ["defect"]),
    color=alt.Color("corr:Q", scale=alt.Scale(scheme="redyellowblue")),
    tooltip=["feature_x", "feature_y", alt.Tooltip("corr:Q", format=".2f")],
).properties(height=360)
st.altair_chart(heat, use_container_width=True)
st.markdown("**그래프 해석**")
st.markdown(insight_corr(_df[feature_cols + ["defect"]].corr(numeric_only=True)))

st.markdown("---")

# -----------------------
# Simple Modeling
# -----------------------
st.subheader("🤖 결함 예측 (Quick Baselines)")
st.caption("로지스틱 회귀/랜덤포레스트로 **기준선 성능**을 빠르게 확인합니다. 실제 적용 전, 데이터/목표에 맞게 임계값과 가중치를 조정하세요.")

# Features + target
X = _df[feature_cols].copy()
y = _df["defect"].copy()

# 클래스가 한쪽뿐이면 학습 불가
if y.nunique() < 2:
    st.warning("현재 필터에서 타깃 클래스가 하나뿐입니다. 다른 필터를 선택해 주세요.")
    st.stop()

# 데이터 분할(검증을 위해 일부를 테스트로 남김)
test_size = st.slider("테스트 세트 비율", 0.1, 0.5, 0.2, 0.05)
random_state = st.number_input("random_state", min_value=0, value=42, step=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

# 모델 선택
model_name = st.selectbox("모델 선택", ["LogisticRegression", "RandomForestClassifier"], index=1)

# 학습
if model_name == "LogisticRegression":
    scaler = StandardScaler()  # 표준화: 각 특징을 같은 스케일로 만들어 선형 모델에 유리
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)[:, 1] if hasattr(clf, "predict_proba") else None
else:
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

# 지표 계산
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
acc = report.get("accuracy", 0.0)                # 정확도: 전체 중 맞춘 비율
precision = report.get("1", {}).get("precision", 0.0)  # 정밀도: 결함이라고 한 것 중 실제 결함 비율
recall = report.get("1", {}).get("recall", 0.0)        # 재현율: 실제 결함 중 찾아낸 비율
f1 = report.get("1", {}).get("f1-score", 0.0)          # F1: 정밀도/재현율의 균형
auc = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy", f"{acc:.3f}")
m2.metric("Precision(Defect=1)", f"{precision:.3f}")
m3.metric("Recall(Defect=1)", f"{recall:.3f}")
m4.metric("F1(Defect=1)", f"{f1:.3f}")
m5.metric("ROC AUC", f"{auc:.3f}" if not np.isnan(auc) else "N/A")

# 혼동행렬 표
st.markdown("**Confusion Matrix (혼동행렬)**")
cm_df = pd.DataFrame(cm, index=["True 0(정상)", "True 1(결함)"], columns=["Pred 0(정상)", "Pred 1(결함)"])
st.dataframe(cm_df, use_container_width=True)
st.markdown(
    "- 좌상단: 정상으로 맞춤, 우하단: 결함으로 맞춤\n"
    "- 우상단: 거짓 경보(False Positive), 좌하단: 놓친 결함(False Negative)"
)

# 특징 중요도(랜덤포레스트만)
if hasattr(clf, "feature_importances_"):
    st.markdown("**Feature Importances (RandomForest)**")
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)
    bar = alt.Chart(imp).mark_bar().encode(
        x=alt.X("importance:Q"),
        y=alt.Y("feature:N", sort="-x"),
        tooltip=["feature", alt.Tooltip("importance:Q", format=".4f")],
    )
    st.altair_chart(bar, use_container_width=True)
    st.markdown("**그래프 해석**")
    st.markdown(insight_importances(imp))

# ROC Curve (확률 예측이 있을 때)
if y_proba is not None:
    st.markdown("**ROC Curve**")
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    st.pyplot(fig, clear_figure=True)
    st.markdown("**그래프 해석**")
    st.markdown(insight_roc(auc))

st.markdown("---")

# -----------------------
# Per-language breakdown
# -----------------------
st.subheader("🧩 언어별 지표")
grp = _df.groupby("language").agg(
    samples=("defect", "size"),
    defect_rate=("defect", "mean"),
    avg_loc=("lines_of_code", "mean"),
    avg_cc=("cyclomatic_complexity", "mean"),
    avg_tokens=("token_count", "mean"),
).reset_index()
grp["defect_rate"] = (grp["defect_rate"] * 100).round(2)
st.dataframe(grp, use_container_width=True)
st.markdown("**표 해석**")
st.markdown(insight_per_lang_table(grp))

st.success("CSV를 자동으로 불러와 분석을 완료했습니다. 좌측 필터와 옵션을 바꿔가며 탐색해 보세요!")
