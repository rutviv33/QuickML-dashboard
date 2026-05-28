import sys
import streamlit as st
import numpy as np
import pandas as pd
import pdfplumber
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# UI Config
st.set_page_config(page_title="AutoML Dashboard", layout="wide", page_icon="🤖")

st.title("🤖 QuickML Dashboard")
st.markdown("Upload data, visualize insights, and train ML models instantly.")
st.write("---")

# Upload
st.header("📁 Data Source")
uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx", "pdf"])

df = None

# ================= FILE LOADING =================
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        elif uploaded_file.name.endswith(".pdf"):
            rows = []
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    table = page.extract_table()
                    if table:
                        rows.extend(table)

            if len(rows) > 1:
                df = pd.DataFrame(rows[1:], columns=rows[0])

        if df is not None:
            df = df.dropna(how="all")
            df = df.infer_objects()

    except Exception as e:
        st.error(f"Error loading file: {e}")

# ================= MAIN APP =================
if df is not None and not df.empty:

    tab1, tab2, tab3 = st.tabs(["📊 Data", "📈 Visualization", "⚙️ ML"])

    # -------- TAB 1 --------
    with tab1:
        st.dataframe(df, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing", int(df.isna().sum().sum()))

    # -------- TAB 2 (FIXED) --------
    with tab2:
        cols = df.columns.tolist()

        gtype = st.selectbox("Chart", ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot"])
        x_axis = st.selectbox("X-axis", cols)

        y_axis = None
        if gtype != "Histogram":
            y_axis = st.selectbox("Y-axis", cols)

        color_by = st.selectbox("Color", ["None"] + cols)
        color_param = None if color_by == "None" else color_by

        try:
            plot_df = df.copy()

            # Clean symbols
            plot_df[x_axis] = plot_df[x_axis].astype(str).str.replace(r"[^\d.-]", "", regex=True)

            if gtype != "Histogram":
                plot_df[y_axis] = plot_df[y_axis].astype(str).str.replace(r"[^\d.-]", "", regex=True)

            # Convert
            plot_df[x_axis] = pd.to_numeric(plot_df[x_axis], errors='coerce')

            if gtype != "Histogram":
                plot_df[y_axis] = pd.to_numeric(plot_df[y_axis], errors='coerce')

            # Drop NaN
            if gtype == "Histogram":
                plot_df = plot_df.dropna(subset=[x_axis])
            else:
                plot_df = plot_df.dropna(subset=[x_axis, y_axis])

            if plot_df.empty:
                st.error("No valid numeric data.")
                st.stop()

            # Plot
            if gtype == "Scatter Plot":
                fig = px.scatter(plot_df, x=x_axis, y=y_axis, color=color_param)

            elif gtype == "Line Chart":
                fig = px.line(plot_df, x=x_axis, y=y_axis, color=color_param)

            elif gtype == "Bar Chart":
                fig = px.bar(plot_df, x=x_axis, y=y_axis, color=color_param)

            elif gtype == "Histogram":
                fig = px.histogram(plot_df, x=x_axis, color=color_param)

            elif gtype == "Box Plot":
                fig = px.box(plot_df, x=x_axis, y=y_axis, color=color_param)

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Visualization error: {e}")

    # -------- TAB 3 (FULL FIXED ML) --------
    with tab3:

        y_col = st.selectbox("Target (Y)", df.columns)
        x_cols = st.multiselect("Features (X)", [c for c in df.columns if c != y_col])

        if len(x_cols) > 0:

            ml_df = df[[y_col] + x_cols].dropna()

            X = ml_df[x_cols]
            y = ml_df[y_col]

            X = pd.get_dummies(X, drop_first=True)

            task = st.radio("Task", ["Regression", "Classification"])

            if st.button("🚀 Train"):

                # Clean numeric
                X = X.apply(pd.to_numeric, errors='coerce')

                if task == "Regression":
                    y = pd.to_numeric(y, errors='coerce')

                valid = X.notna().all(axis=1)
                if task == "Regression":
                    valid = valid & y.notna()

                X, y = X[valid], y[valid]

                if X.empty:
                    st.error("No valid data.")
                    st.stop()

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # ===== REGRESSION =====
                if task == "Regression":

                    best_deg = 0
                    best_score = -999
                    scores = []

                    for d in range(1, 11):
                        try:
                            poly = PolynomialFeatures(degree=d)

                            Xtr = poly.fit_transform(X_train)
                            Xte = poly.transform(X_test)

                            model = LinearRegression()
                            model.fit(Xtr, y_train)

                            pred = model.predict(Xte)
                            score = r2_score(y_test, pred)

                            scores.append((d, score))

                            if score > best_score:
                                best_score = score
                                best_deg = d

                        except:
                            continue

                    st.metric("Best Degree", best_deg)
                    st.metric("Best R²", f"{round(best_score*100,2)}%")

                    res_df = pd.DataFrame(scores, columns=["Degree", "R2"])
                    fig = px.line(res_df, x="Degree", y="R2")
                    st.plotly_chart(fig, use_container_width=True)

                # ===== CLASSIFICATION =====
                else:

                    models = {
                        "KNN": KNeighborsClassifier(),
                        "Decision Tree": DecisionTreeClassifier(),
                        "SVM": SVC(),
                        "Random Forest": RandomForestClassifier()
                    }

                    results = {}

                    for name, m in models.items():
                        m.fit(X_train, y_train.astype(str))
                        pred = m.predict(X_test)
                        acc = accuracy_score(y_test.astype(str), pred)
                        results[name] = acc

                    res_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])

                    st.dataframe(res_df)

                    best = max(results, key=results.get)
                    st.metric("Best Model", f"{best} ({round(results[best]*100,2)}%)")

                    fig = px.bar(res_df, x="Model", y="Accuracy", color="Model")
                    st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Select features.")

else:
    st.info("Upload a file to start.")