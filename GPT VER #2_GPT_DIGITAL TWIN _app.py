import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import plotly.express as px


"""
Streamlit Digital Twin App
==========================

This Streamlit application implements a simplified digital twin workflow
for gene‑expression based drug discovery.  Users can upload a gene
expression matrix with labels, perform basic preprocessing,
fit a logistic regression model to distinguish between classes, and
identify candidate drugs based on the top predictive genes.  While not
a one‑to‑one port of the original Jupyter notebook (which contained
complex quantum optimisation and pathway analysis code), this app
demonstrates the core interactions: data upload, normalisation,
classification, visualisation and heuristic drug suggestions.

**Usage**

1. Prepare a CSV file where rows correspond to samples and columns
   correspond to gene expressions.  Include a column containing
   class labels (e.g. 0/1, or case/control) and ensure it is
   explicitly named.  Each gene should be numeric.
2. Launch this application via `streamlit run app.py`.
3. Upload your CSV file using the file uploader.
4. Select which column represents the label and which columns represent
   gene expressions.  If you do not specify gene columns, all
   non‑label columns are used by default.
5. Click the **Run Analysis** button to fit a logistic regression
   classifier.  The app will display accuracy and AUC metrics, a
   scatter plot of the first two principal components, a bar chart of
   top gene coefficients, and a heuristic list of candidate drugs.

**Limitations**

This simplified version omits the detailed pathway scoring,
penalisation matrices and quantum solvers present in the original
notebook.  Instead, it uses a simple logistic regression model and
maps the most predictive genes to a small, hard‑coded gene–drug
dictionary.  For production use you should integrate a comprehensive
gene–drug database and the full optimisation logic.
"""


def main():
    st.title("Digital Twin: Gene Expression to Drug Selection")
    st.markdown(
        """
        Upload a gene expression dataset (CSV) with a column containing
        class labels (e.g. responder vs non‑responder).  This app will
        normalise the gene data, fit a logistic regression classifier
        and identify the most important genes.  A heuristic mapping
        provides candidate drugs targeting those genes.
        """
    )

    # File upload
    uploaded_file = st.file_uploader("Upload gene expression CSV", type=["csv", "tsv", "txt"])

    if uploaded_file is not None:
        # Determine delimiter automatically
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        delimiter = "," if "," in content.splitlines()[0] else "\t"
        df = pd.read_csv(uploaded_file, sep=delimiter)

        st.subheader("Preview of uploaded data")
        st.write(df.head())

        # Select label column
        label_col = st.selectbox(
            "Select the label column (target)",
            options=df.columns,
            help="Choose the column that contains class labels"
        )

        # Select gene columns; default all except label
        default_gene_cols = [col for col in df.columns if col != label_col]
        gene_columns = st.multiselect(
            "Select gene columns (predictors)",
            options=[col for col in df.columns if col != label_col],
            default=default_gene_cols,
            help="Choose the columns to use as gene features.  If you leave this empty, all non‑label columns are used."
        )
        if not gene_columns:
            gene_columns = default_gene_cols

        if st.button("Run Analysis"):
            st.info("Running logistic regression…")
            try:
                # Extract X and y
                X = df[gene_columns].select_dtypes(include=[np.number]).fillna(0).values
                y = df[label_col].values
                # Encode labels if they're strings
                y_unique = np.unique(y)
                if not np.issubdtype(y.dtype, np.number):
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                # Standardise features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.3, random_state=42, stratify=y
                )
                # Fit model
                model = LogisticRegression(max_iter=500, solver='lbfgs')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                acc = accuracy_score(y_test, y_pred)
                try:
                    auc = roc_auc_score(y_test, y_proba)
                except Exception:
                    auc = float('nan')

                st.success(f"Accuracy: {acc:.3f}, AUC: {auc:.3f}")
                st.subheader("Classification report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.write(report_df)

                # PCA for visualisation
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                pca_df = pd.DataFrame(
                    {
                        'PC1': X_pca[:, 0],
                        'PC2': X_pca[:, 1],
                        'Label': y
                    }
                )
                st.subheader("PCA scatter plot")
                fig = px.scatter(
                    pca_df, x='PC1', y='PC2', color=pca_df['Label'].astype(str),
                    title='First two principal components',
                    labels={'color': 'Class'}
                )
                st.plotly_chart(fig, use_container_width=True)

                # Feature importance (absolute coefficients)
                coef = np.abs(model.coef_[0])
                gene_importance = pd.DataFrame({
                    'Gene': gene_columns,
                    'Importance': coef
                }).sort_values(by='Importance', ascending=False)
                st.subheader("Top predictive genes")
                fig_bar = px.bar(
                    gene_importance.head(30),
                    x='Gene',
                    y='Importance',
                    title='Top gene coefficients (absolute values)',
                    labels={'Importance': 'Coefficient magnitude'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # Heuristic gene–drug mapping (toy example)
                gene_to_drugs = {
                    'TP53': ['Nutlin-3', 'APR-246'],
                    'BRCA1': ['Olaparib', 'Talazoparib'],
                    'EGFR': ['Gefitinib', 'Erlotinib'],
                    'VEGFA': ['Bevacizumab'],
                    'KRAS': ['Sotorasib'],
                    'ALK': ['Crizotinib'],
                    'MYC': ['Omomyc'],
                    'PIK3CA': ['Alpelisib'],
                    'BRAF': ['Vemurafenib'],
                    'CDK4': ['Palbociclib']
                }
                # Select top genes and map to drugs
                top_genes = gene_importance['Gene'].tolist()[:20]
                candidate_drugs = {}
                for gene in top_genes:
                    if gene in gene_to_drugs:
                        candidate_drugs[gene] = gene_to_drugs[gene]
                st.subheader("Candidate drugs for top genes")
                if candidate_drugs:
                    drug_df = pd.DataFrame([
                        {'Gene': gene, 'Drugs': ", ".join(drugs)}
                        for gene, drugs in candidate_drugs.items()
                    ])
                    st.table(drug_df)
                else:
                    st.write(
                        "No candidate drugs were found for the top genes in the built‑in dictionary."
                    )

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                import traceback
                st.text(traceback.format_exc())

    st.markdown(
        """
        \n---\n
        #### About this app
        
        This application was generated as part of a demonstration on converting
        a complex Jupyter notebook into a self‑contained web application
        using Streamlit.  It summarises the digital twin workflow by
        combining data upload, normalisation, modelling and a simple
        gene–drug mapping into an interactive interface.
        
        The original notebook performed pathway analysis and quantum
        optimisation to suggest drug combinations.  Those features are
        beyond the scope of this simplified app but could be added by
        extending the code and incorporating appropriate libraries.
        
        """
    )


if __name__ == "__main__":
    main()