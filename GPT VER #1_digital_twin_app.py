"""
digital_twin_app.py
--------------------

This Streamlit application exposes the digital twin therapy‑selection workflow
as a simple web interface.  Users can upload a gene expression matrix or
fallback to a built‑in synthetic dataset and then run the drug selection
analysis.  Results include a summary of how often each therapy is selected
across patients and a per‑patient report.  Users can also download the report
as a CSV file.

To run the app locally, install Streamlit (`pip install streamlit`) and
execute:

    streamlit run digital_twin_app.py

If you do not provide an expression file, the app will use a synthetic
dataset for demonstration purposes.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st


###############################################################################
# Pathway definitions and helper functions (copied from simulate_digital_twin)
###############################################################################

SIGS = {
    "REACTOME_SIGNALING_BY_EGFR": [
        "EGFR", "ERBB2", "ERBB3", "GRB2", "SOS1", "SHC1", "PTPN11", "KRAS",
        "NRAS", "HRAS", "BRAF", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3",
        "PLCG1", "PIK3CA", "PIK3R1", "AKT1", "AKT2", "AKT3", "GAB1",
    ],
    "REACTOME_SIGNALING_BY_ALK": [
        "ALK", "EML4", "GRB2", "SHC1", "PIK3CA", "PIK3R1", "AKT1", "AKT2",
        "AKT3", "STAT3", "MAP2K1", "MAPK1", "MAPK3",
    ],
    "REACTOME_MAPK1_MAPK3_SIGNALING": [
        "BRAF", "RAF1", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3", "DUSP6",
        "DUSP4", "FOS", "JUN", "EGFR",
    ],
    "REACTOME_PI3K_AKT_SIGNALING": [
        "PIK3CA", "PIK3CB", "PIK3CD", "PIK3R1", "PIK3R2", "AKT1", "AKT2",
        "AKT3", "PTEN", "MTOR", "RHEB",
    ],
    "REACTOME_MTORC1_MEDIATED_SIGNALLING": [
        "MTOR", "RPTOR", "MLST8", "RHEB", "TSC1", "TSC2", "EIF4EBP1",
        "RPS6KB1", "RPS6",
    ],
    "REACTOME_PD1_SIGNALING": [
        "PDCD1", "CD274", "PDCD1LG2", "JAK1", "JAK2", "STAT1", "IFNG",
        "GZMB", "LAG3", "TIGIT", "CXCL9", "CXCL10",
    ],
    "REACTOME_VEGFA_VEGFR2_SIGNALING_PATHWAY": [
        "VEGFA", "KDR", "FLT1", "FLT4", "PTPRB", "PLCG1", "MAP2K1",
        "MAPK1", "NOS3",
    ],
    "REACTOME_SIGNALING_BY_FGFR": [
        "FGFR1", "FGFR2", "FGFR3", "FGFR4", "FRS2", "PLCG1", "PIK3CA",
        "PIK3R1", "MAP2K1", "MAPK1",
    ],
}


def generate_synthetic_expression(
    genes: list[str], n_samples: int, random_state: int | None = None
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    data = rng.lognormal(mean=1.0, sigma=1.0, size=(len(genes), n_samples))
    columns = [f"Sample_{i+1:02d}" for i in range(n_samples)]
    return pd.DataFrame(data, index=genes, columns=columns)


def zscore_by_gene(expr_symbols: pd.DataFrame) -> pd.DataFrame:
    E = expr_symbols.apply(pd.to_numeric, errors="coerce")
    E = E.loc[~E.isna().all(axis=1)]
    gene_means = E.mean(axis=1)
    E = E.apply(lambda col: col.fillna(gene_means), axis=0)
    mu = E.mean(axis=1)
    sd = E.std(axis=1) + 1e-8
    return (E.sub(mu, axis=0)).div(sd, axis=0)


def pathway_scores(expr_symbols: pd.DataFrame, signatures: dict) -> pd.DataFrame:
    Z = zscore_by_gene(expr_symbols)
    rows = []
    for pw, genes in signatures.items():
        present = [g for g in genes if g in Z.index]
        if present:
            s = Z.loc[present].mean(axis=0)
        else:
            s = pd.Series([np.nan] * Z.shape[1], index=Z.columns)
        s.name = pw
        rows.append(s)
    return pd.DataFrame(rows)


def example_drug_panel() -> dict[str, list[str]]:
    return {
        "EGFRi": ["REACTOME_SIGNALING_BY_EGFR"],
        "ALKi": ["REACTOME_SIGNALING_BY_ALK"],
        "MEKi": ["REACTOME_MAPK1_MAPK3_SIGNALING"],
        "PI3Ki": ["REACTOME_PI3K_AKT_SIGNALING"],
        "mTORi": ["REACTOME_MTORC1_MEDIATED_SIGNALLING"],
        "PD1i": ["REACTOME_PD1_SIGNALING"],
        "VEGFi": ["REACTOME_VEGFA_VEGFR2_SIGNALING_PATHWAY"],
        "FGFRi": ["REACTOME_SIGNALING_BY_FGFR"],
    }


def patient_vector(P: pd.DataFrame, sample_id: str) -> pd.Series:
    z = (P - P.mean(axis=1).values.reshape(-1, 1)) / (
        P.std(axis=1).values.reshape(-1, 1) + 1e-8
    )
    return z[sample_id].fillna(0.0)


def drug_benefit_prior(z_path: pd.Series, panel: dict[str, list[str]]) -> pd.Series:
    s = pd.Series(
        {
            d: float(sum(max(z_path.get(p, 0.0), 0.0) for p in pws))
            for d, pws in panel.items()
        }
    )
    return s / s.max() if s.max() > 0 else s


def build_penalty_matrix(
    drugs: list[str], panel: dict[str, list[str]], base_overlap: float = 0.25, sparsity: float = 0.10
) -> np.ndarray:
    K = len(drugs)
    R = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(i + 1, K):
            overlap = len(set(panel[drugs[i]]) & set(panel[drugs[j]]))
            if overlap > 0:
                R[i, j] = R[j, i] = base_overlap * overlap
    for i in range(K):
        R[i, i] += sparsity
    return R


def exact_qubo_solve(b_hat: np.ndarray, R: np.ndarray, lam: float = 1.0) -> tuple[np.ndarray, float]:
    K = len(b_hat)
    Q = lam * R.copy()
    q_vec = -b_hat.copy() + np.diag(Q)
    np.fill_diagonal(Q, 0.0)
    best_e = float("inf")
    best_x: np.ndarray | None = None
    upper_idx = np.triu_indices(K, 1)
    for mask in range(1 << K):
        x = np.fromiter(((mask >> i) & 1 for i in range(K)), dtype=np.int8)
        e = float(np.dot(q_vec, x) + 2.0 * np.sum(Q[upper_idx] * (x[upper_idx[0]] * x[upper_idx[1]])))
        if e < best_e:
            best_e, best_x = e, x
    assert best_x is not None
    return best_x.astype(int), best_e


def run_workflow(expr: pd.DataFrame, n_patients: int = 25) -> tuple[list[list[str]], list[pd.Series], pd.DataFrame]:
    P = pathway_scores(expr, SIGS)
    panel = example_drug_panel()
    drugs = list(panel.keys())
    selections: list[list[str]] = []
    benefit_series_list: list[pd.Series] = []
    for sid in P.columns[:n_patients]:
        z_path = patient_vector(P, sid)
        b_hat = drug_benefit_prior(z_path, panel).reindex(drugs).fillna(0.0)
        R = build_penalty_matrix(drugs, panel, base_overlap=0.25, sparsity=0.10)
        x_star, _ = exact_qubo_solve(b_hat.to_numpy(float), R, lam=1.0)
        sel = [drugs[i] for i, xi in enumerate(x_star) if xi == 1]
        selections.append(sel)
        benefit_series_list.append(b_hat)
    return selections, benefit_series_list, P


def build_patient_report(
    P: pd.DataFrame, selections: list[list[str]], benefit_series_list: list[pd.Series]
) -> pd.DataFrame:
    panel = example_drug_panel()
    drugs = list(panel.keys())
    rows = []
    for idx, sid in enumerate(P.columns[: len(selections)]):
        sel = selections[idx]
        b_hat = benefit_series_list[idx]
        z_path = patient_vector(P, sid)
        top_pw = z_path.abs().sort_values(ascending=False).head(5)
        rows.append(
            {
                "patient": sid,
                "selected_drugs": ", ".join(sel) if sel else "(none)",
                "top_pathways": "; ".join(f"{p}:{z_path[p]:+.2f}" for p in top_pw.index),
                **{f"b_hat.{d}": float(b_hat.get(d, 0.0)) for d in drugs},
            }
        )
    return pd.DataFrame(rows)


###############################################################################
# Streamlit UI
###############################################################################

st.title("Digital Twin Therapy Selection")
st.write(
    "This app demonstrates the digital twin workflow for selecting potential drug therapies "
    "based on gene expression data. Upload your own TPM‑formatted expression matrix (genes in rows, "
    "samples in columns) or use the built‑in synthetic dataset to explore the analysis."
)

# File uploader
uploaded = st.file_uploader(
    "Upload a gene expression matrix (CSV or TSV with genes in rows)", type=["csv", "tsv"]
)

# Parameter: number of patients to analyse
n_patients = st.number_input(
    "Number of patients to analyse (must be <= number of samples)", min_value=1, value=25, step=1
)

# When the user clicks the button, run the analysis
if st.button("Run Analysis"):
    # Load or generate expression matrix
    if uploaded is not None:
        bytes_data = uploaded.read()
        # Try CSV first, then TSV
        try:
            expr = pd.read_csv(io.BytesIO(bytes_data), index_col=0)
        except Exception:
            expr = pd.read_csv(io.BytesIO(bytes_data), sep="\t", index_col=0)
        # Keep only signature genes to avoid mismatches
        expr = expr.loc[expr.index.intersection(set(g for genes in SIGS.values() for g in genes))]
        # If too few samples, warn
        if expr.shape[1] < n_patients:
            st.error(f"The uploaded dataset has only {expr.shape[1]} samples; adjust the number of patients.")
            st.stop()
    else:
        # Use synthetic data for demonstration
        st.info("No file uploaded. Using a synthetic dataset for demonstration.")
        genes = sorted({g for genes in SIGS.values() for g in genes})
        expr = generate_synthetic_expression(genes, max(n_patients, 10), random_state=0)
        expr = expr.loc[~expr.index.duplicated(keep="first")]

    # Run the workflow
    selections, b_hat_list, P = run_workflow(expr, n_patients=int(n_patients))

    # Compute selection frequency
    flat = [d for sel in selections for d in sel]
    freq = Counter(flat)
    panel = example_drug_panel()
    drugs = list(panel.keys())
    freq_df = pd.DataFrame(
        {
            "Drug": drugs,
            "Frequency": [freq[d] for d in drugs],
        }
    )
    freq_df["Frequency (% patients)"] = 100.0 * freq_df["Frequency"] / max(1, len(selections))
    freq_df = freq_df.sort_values("Frequency (% patients)", ascending=False).reset_index(drop=True)

    st.subheader("Drug Selection Frequency")
    st.dataframe(freq_df)

    # Build and display patient report
    report_df = build_patient_report(P, selections, b_hat_list)
    st.subheader("Patient‑Level Report")
    st.dataframe(report_df)

    # Download button
    csv_bytes = report_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Report as CSV",
        data=csv_bytes,
        file_name="patient_report.csv",
        mime="text/csv",
    )