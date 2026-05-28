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
    with tab_model:
        st.subheader("Model Configuration")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            y_col = st.selectbox("Select Target Variable (Y)", options=df.columns, key="target_y")
            
        with col_m2:
            select_all = st.checkbox("Select all columns as features", value=True)
            feature_choices = [col for col in df.columns if col != y_col]
            
            if select_all:
                x_cols = st.multiselect("Select Independent Variables (X)", options=feature_choices, default=feature_choices)
            else:
                x_cols = st.multiselect("Select Independent Variables (X)", options=feature_choices)

        # Process chosen features
        if len(x_cols) > 0:
            
            # Drop rows where critical modeling variables are null
            ml_df = df[[y_col] + x_cols].dropna()
            
            X = ml_df[x_cols]
            y = ml_df[y_col]
            
            # Auto convert object categories to categorical dummy flags
            X = pd.get_dummies(X, drop_first=True)
            
            task_type = st.radio("Select Task Type", ["Regression", "Classification"], horizontal=True)
            
            if st.button("🚀 Train & Evaluate Models", use_container_width=True):
                # Ensure numeric compatibility for regression/target conversions
                if task_type == "Regression":
                    y = pd.to_numeric(y, errors='coerce')
                    X = X.apply(pd.to_numeric, errors='coerce')
                    
                    # Eliminate any nan gaps caused by forced conversions
                    valid_idx = y.notna() & X.notna().all(axis=1)
                    X, y = X[valid_idx], y[valid_idx]
                
                    if not X.empty and not y.empty:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        st.write("---")
                        
                        best_degree = None
                        best_score = -999
                        scores = []
                        
                        for d in range(1, 11):
                            try:
                                poly = PolynomialFeatures(degree=d)
                                x_train_poly = poly.fit_transform(X_train)
                                x_test_poly = poly.transform(X_test)
                            
                                model = LinearRegression()
                                model.fit(x_train_poly, y_train)
                                preds = model.predict(x_test_poly)
                            
                                score = r2_score(y_test, preds)
                                scores.append((d, score))
                            
                                if score > best_score:
                                    best_score = score
                                    best_degree = d
                    
                            except Exception as e:
                                # Skip if model fails (important for high degrees)
                                continue
                        
                        # Display best result
                        st.metric(
                            label=f"🏆 Best Polynomial Degree: {best_degree}",
                            value=f"{round(best_score * 100, 2)}%"
                        )
                
                        # Optional: plot
                        fig_deg = px.line(res_df, x="Degree", y="R2 Score", title="Degree vs R² Score")
                        st.plotly_chart(fig_deg, use_container_width=True)
                        
                    else:
                        # Classification configuration
                        class_models = {
                            "KNN": KNeighborsClassifier(n_neighbors=3),
                            "Decision Tree": DecisionTreeClassifier(criterion="entropy", random_state=42, max_depth=3),
                            "SVM": SVC(kernel="rbf", C=50, random_state=42),
                            "Random Forest": RandomForestClassifier(criterion="entropy", n_estimators=5, max_depth=4, random_state=42)
                        }
                        
                        c = {}
                        for name, clf in class_models.items():
                            clf.fit(X_train, y_train.astype(str)) # Ensure labels treat as explicit strings/classes
                            preds = clf.predict(X_test)
                            score = accuracy_score(y_test.astype(str), preds)
                            c[name] = score
                        
                        st.subheader("Model Leaderboard")
                        res_df = pd.DataFrame(list(c.items()), columns=["Model", "Accuracy"])
                        res_df["Accuracy %"] = res_df["Accuracy"].apply(lambda val: f"{round(val*100,2)}%")
                        
                        col_lead1, col_lead2 = st.columns([1, 2])
                        with col_lead1:
                            st.dataframe(res_df[["Model", "Accuracy %"]], hide_index=True)
                            best_model = max(c, key=c.get)
                            st.metric(label=f"Best Performance: {best_model}", value=f"{round(c[best_model]*100, 2)}%")
                        with col_lead2:
                            fig_res = px.bar(res_df, x="Model", y="Accuracy", color="Model", title="Classifier Performance Comparison")
                            st.plotly_chart(fig_res, use_container_width=True)
                else:
                    st.error("❌ The processed numbers contain null values. Check that your selected variables contain numbers.")
        else:
            st.warning("⚠️ Please select at least one feature column (X) to train the model.")

else:
    st.info("Upload a file to start.")