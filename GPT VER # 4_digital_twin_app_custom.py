# Customizable version of the Digital Twin Therapy Selection app.
#
# This version extends the original proof‑of‑concept by allowing users to
# upload their own pathway definitions (gene sets) and choose how to
# aggregate gene expression into pathway scores. It retains the original
# workflow (pathway scoring, drug benefit scoring, QUBO selection) but
# removes hard‑coded limitations on which genes can be used. If no
# custom pathway file is provided, the app falls back to the original
# Reactome‑derived gene signatures. A missing import for `Counter` has
# also been fixed.

import io
from collections import Counter
from functools import partial
import numpy as np
import pandas as pd
import streamlit as st

###############################################################################
# Default pathway definitions and helper functions
###############################################################################

# Original Reactome pathway signatures used as a default when the user does not
# supply a custom pathway file. Each entry maps a pathway name to a list of
# gene symbols.
DEFAULT_SIGS = {
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


def load_custom_signatures(file: io.BytesIO) -> dict[str, list[str]]:
    """Parse a user‑uploaded pathway definition file into a signature dictionary.

    The file can be CSV/TSV with the first column as pathway names and the
    remaining columns (or comma‑separated values) as gene symbols. It can
    also be JSON with a mapping of pathway names to lists of genes. If
    parsing fails, a ValueError will be raised.
    """
    try:
        # Try JSON format first
        import json

        content = file.read().decode("utf-8")
        try:
            data = json.loads(content)
            if not isinstance(data, dict):
                raise ValueError("JSON pathway file must be a dict of lists")
            # ensure all values are lists of strings
            sigs = {
                str(k): [str(x).strip() for x in v if str(x).strip()]
                for k, v in data.items()
                if isinstance(v, (list, tuple))
            }
            if sigs:
                return sigs
        except json.JSONDecodeError:
            pass
        # Reset pointer if JSON parsing failed
        file.seek(0)
        # Fallback to tabular format (CSV/TSV)
        df = pd.read_csv(file, sep=None, engine="python", header=None)
        if df.shape[1] < 2:
            raise ValueError(
                "Pathway definition file must have at least two columns: pathway and genes"
            )
        sigs: dict[str, list[str]] = {}
        for _, row in df.iterrows():
            pw = str(row.iloc[0]).strip()
            genes = []
            for col in row.iloc[1:]:
                if pd.isna(col):
                    continue
                # split comma‑separated lists if present
                parts = str(col).replace(";", ",").split(",")
                genes.extend([p.strip() for p in parts if p.strip()])
            if genes:
                sigs[pw] = genes
        return sigs
    finally:
        # reset pointer for any subsequent reads
        file.seek(0)


def zscore_by_gene(expr_symbols: pd.DataFrame) -> pd.DataFrame:
    """Compute Z‑scores for each gene across samples."""
    E = expr_symbols.apply(pd.to_numeric, errors="coerce")
    E = E.loc[~E.isna().all(axis=1)]
    gene_means = E.mean(axis=1)
    E = E.apply(lambda col: col.fillna(gene_means), axis=0)
    mu = E.mean(axis=1)
    sd = E.std(axis=1) + 1e-8
    return (E.sub(mu, axis=0)).div(sd, axis=0)


def pathway_scores(expr_symbols: pd.DataFrame, signatures: dict[str, list[str]], agg_fn) -> pd.DataFrame:
    """Calculate aggregated z‑scored expression per pathway per sample.

    Parameters
    ----------
    expr_symbols : pd.DataFrame
        Gene expression matrix with genes as index and samples as columns.
    signatures : dict[str, list[str]]
        Mapping from pathway names to lists of genes. Only genes present in
        expr_symbols will be used.
    agg_fn : callable
        Aggregation function applied to the z‑scored expression of the genes
        belonging to a pathway (e.g. np.mean, np.median, np.sum).
    """
    Z = zscore_by_gene(expr_symbols)
    rows = []
    for pw, genes in signatures.items():
        present = [g for g in genes if g in Z.index]
        if present:
            s = Z.loc[present].aggregate(agg_fn, axis=0)
        else:
            s = pd.Series([np.nan] * Z.shape[1], index=Z.columns)
        s.name = pw
        rows.append(s)
    return pd.DataFrame(rows)


def example_drug_panel(signatures: dict[str, list[str]]) -> dict[str, list[str]]:
    """Define a simple drug panel by mapping drugs to pathway(s).

    For custom signatures, we assign each pathway to its own drug (e.g.
    drug name = pathway name + '_i'), otherwise use the default panel.
    """
    # If the signatures equal the default, return default panel
    if signatures == DEFAULT_SIGS:
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
    # Otherwise create a trivial panel: one drug per pathway
    return {f"{pw}_i": [pw] for pw in signatures.keys()}


def patient_vector(P: pd.DataFrame, sample_id: str) -> pd.Series:
    """Compute a normalized pathway activity vector for a patient (sample)."""
    z = (P - P.mean(axis=1).values.reshape(-1, 1)) / (
        P.std(axis=1).values.reshape(-1, 1) + 1e-8
    )
    return z[sample_id].fillna(0.0)


def drug_benefit_prior(z_path: pd.Series, panel: dict[str, list[str]]) -> pd.Series:
    """Compute a prior benefit score for each drug based on pathway activity."""
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
    """Construct a penalty matrix to discourage overlapping pathways and enforce sparsity."""
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
    """Solve a small QUBO exactly by brute force over all binary configurations."""
    K = len(b_hat)
    Q = lam * R.copy()
    q_vec = -b_hat.copy() + np.diag(Q)
    np.fill_diagonal(Q, 0.0)
    best_e = float("inf")
    best_x: np.ndarray | None = None
    upper_idx = np.triu_indices(K, 1)
    for mask in range(1 << K):
        x = np.fromiter(((mask >> i) & 1 for i in range(K)), dtype=np.int8)
        e = float(
            np.dot(q_vec, x) + 2.0 * np.sum(Q[upper_idx] * (x[upper_idx[0]] * x[upper_idx[1]]))
        )
        if e < best_e:
            best_e, best_x = e, x
    assert best_x is not None
    return best_x.astype(int), best_e


def run_workflow(
    expr: pd.DataFrame,
    signatures: dict[str, list[str]],
    n_patients: int = 25,
    agg_fn = np.mean,
) -> tuple[list[list[str]], list[pd.Series], pd.DataFrame]:
    """Run the digital twin workflow and return selections, benefit series, and pathway scores."""
    P = pathway_scores(expr, signatures, agg_fn)
    panel = example_drug_panel(signatures)
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
    P: pd.DataFrame,
    selections: list[list[str]],
    benefit_series_list: list[pd.Series],
    panel: dict[str, list[str]],
) -> pd.DataFrame:
    """Build a per‑patient report summarizing selected drugs and top pathways."""
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

st.title("Customizable Digital Twin Therapy Selection")
st.write(
    "This app demonstrates an extensible digital twin workflow for selecting potential drug therapies "
    "based on gene expression data. Upload your own TPM‑formatted expression matrix (genes in rows, "
    "samples in columns) and, optionally, a pathway definition file to define custom gene sets. "
    "You can also choose how to aggregate gene expression within each pathway."
)

# File uploader for expression matrix
expr_file = st.file_uploader(
    "Upload a gene expression matrix (CSV or TSV with genes in rows)", type=["csv", "tsv"], key="expr"
)

# Optional file uploader for custom pathway definitions
sig_file = st.file_uploader(
    "Upload a pathway definition file (optional: CSV/TSV or JSON)", type=["csv", "tsv", "json"], key="sig"
)

# Pathway aggregation method selection
agg_method = st.selectbox(
    "Select pathway aggregation method", options=["mean", "median", "sum"], index=0
)

# Parameter: number of patients to analyse
n_patients = st.number_input(
    "Number of patients to analyse (must be <= number of samples)", min_value=1, value=25, step=1
)

# When the user clicks the button, run the analysis
if st.button("Run Analysis"):
    # Validate and load expression matrix
    if expr_file is None:
        st.error("Please upload a gene expression matrix.")
        st.stop()
    bytes_data = expr_file.read()
    try:
        expr = pd.read_csv(io.BytesIO(bytes_data), index_col=0)
    except Exception:
        try:
            expr = pd.read_csv(io.BytesIO(bytes_data), sep="\t", index_col=0)
        except Exception as e:
            st.error(f"Failed to parse expression file: {e}")
            st.stop()
    # Load or set pathway signatures
    if sig_file is not None:
        try:
            signatures = load_custom_signatures(sig_file)
        except Exception as e:
            st.error(f"Failed to parse pathway definition file: {e}")
            st.stop()
    else:
        signatures = DEFAULT_SIGS
    # Determine aggregation function
    agg_fn = {"mean": np.mean, "median": np.median, "sum": np.sum}[agg_method]
    # Keep only genes present in signatures to avoid mismatches
    sig_genes = set(g for genes in signatures.values() for g in genes)
    expr = expr.loc[expr.index.intersection(sig_genes)]
    # Check sample count
    if expr.shape[1] < n_patients:
        st.error(f"The uploaded dataset has only {expr.shape[1]} samples; adjust the number of patients.")
        st.stop()
    # Run workflow
    selections, b_hat_list, P = run_workflow(expr, signatures, n_patients=int(n_patients), agg_fn=agg_fn)
    panel = example_drug_panel(signatures)
    # Compute frequency
    flat = [d for sel in selections for d in sel]
    freq = Counter(flat)
    drugs = list(panel.keys())
    freq_df = pd.DataFrame({"Drug": drugs, "Frequency": [freq[d] for d in drugs]})
    freq_df["Frequency (% patients)"] = 100.0 * freq_df["Frequency"] / max(1, len(selections))
    freq_df = freq_df.sort_values("Frequency (% patients)", ascending=False).reset_index(drop=True)
    # Display results
    st.subheader("Drug Selection Frequency")
    st.dataframe(freq_df)
    report_df = build_patient_report(P, selections, b_hat_list, panel)
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