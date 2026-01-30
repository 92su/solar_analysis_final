# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from datetime import datetime
from PIL import Image
import zipfile
import io
import os
from fpdf import FPDF

# ============================================================
# ----------------- INSIGHT + SUMMARY FUNCTIONS --------------
# ============================================================
def generate_anomaly_insight(row, vars_selected, df_clean):
    insights = []
    for var in vars_selected:
        # skip if var not in df_clean (safety)
        if var not in df_clean.columns or pd.isna(row.get(var, np.nan)):
            continue
        mean = df_clean[var].mean()
        std = df_clean[var].std()
        if std == 0 or pd.isna(std):
            continue
        z = (row[var] - mean) / std
        if abs(z) > 3:
            insights.append(f"{var} is extremely unusual (z-score {z:.2f}) compared to normal values.")
        elif abs(z) > 2:
            insights.append(f"{var} is outside normal operating range (z-score {z:.2f}).")
        else:
            insights.append(f"{var} deviates slightly from normal (z-score {z:.2f}).")
    return " | ".join(insights)

def generate_summary(anomalies_df, vars_selected, df_clean):
    if anomalies_df.empty:
        return "No anomalies detected."
    has_insight = "Insight" in anomalies_df.columns
    total = len(anomalies_df)
    summary_text = f"### 🔎 Summary of Anomaly Detection\n\n"
    summary_text += f"- Total anomalies detected: **{total}**\n"
    # Variable impact
    freq = {}
    for var in vars_selected:
        if var not in df_clean.columns:
            continue
        mean = df_clean[var].mean()
        std = df_clean[var].std()
        if std == 0 or pd.isna(std):
            continue
        freq[var] = anomalies_df[var].apply(lambda x: abs((x - mean) / std)).mean()
    if freq:
        sorted_vars = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        most_affected = sorted_vars[0][0]
        summary_text += f"- Most affected variable: **{most_affected}**\n"
    # severity
    if has_insight:
        severe_count = (anomalies_df["Insight"].str.contains("extremely", na=False)).sum()
        moderate_count = (anomalies_df["Insight"].str.contains("outside normal", na=False)).sum()
        mild_count = total - severe_count - moderate_count
    else:
        severe_count = moderate_count = mild_count = 0
    summary_text += f"- Severe anomalies (critical): **{severe_count}**\n"
    summary_text += f"- Moderate anomalies: **{moderate_count}**\n"
    summary_text += f"- Mild anomalies: **{mild_count}**\n\n"
    if severe_count > 0:
        summary_text += "⚠️ **System Status: Critical** — Immediate attention required.\n"
    elif moderate_count > 0:
        summary_text += "⚠️ **System Status: Warning** — Some values exceed normal limits.\n"
    else:
        summary_text += "✅ **System Status: Stable** — Only minor irregularities detected.\n"
    return summary_text

# ----------------- Page Setup -----------------
st.set_page_config(page_title="Solar Analysis Dashboard", layout="wide")

# ================== LOGIN SYSTEM ======================
USERS = {"admin": "admin", "user1": "abcd"}

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def login_page():
    st.markdown("""
        <style>
            .main {display: flex; justify-content: center; align-items: center; height: 100vh;}
            .login-card {
                background: #ffffff;
                padding: 40px 50px;
                width: 450px;
                border-radius: 40px;
                box-shadow: 0 5px 25px rgba(0,0,0,0.10);
                text-align: center;
            }
            .stButton>button {width: 100%; padding: 10px; border-radius: 10px;
                background: #4a90e2; color: white; font-weight: bold; border: none;}
            .stButton>button:hover {background: #357ABD;}
            .login-card input {border-radius: 10px !important;}
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
        .block-container { max-width: 650px; padding-top: 5%; }
    </style>
    """, unsafe_allow_html=True)

    try:
        logo = Image.open("logo.jpg")
        st.image(logo, width=280)
    except:
        st.info("Upload 'logo.jpg' to show logo")

    st.markdown("### Login Username and Password Here!")
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")
    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.success("Login successful 🎉")
            st.rerun()
        else:
            st.error("Invalid username or password ❌")

if not st.session_state["authenticated"]:
    login_page()
    st.stop()

# ================== LOGOUT ======================
st.sidebar.markdown("### 🔓 Account")
if st.sidebar.button("Logout"):
    st.session_state["authenticated"] = False
    st.rerun()

# -------------------------------------------------
#                  FILE UPLOAD SECTION
# -------------------------------------------------
st.sidebar.markdown("## 📤 Upload Data")
upload_type = st.sidebar.radio("Upload type:", ["Single/Multiple Files", "ZIP File"])
dfs = []

if upload_type == "Single/Multiple Files":
    uploaded_files = st.sidebar.file_uploader("📂 Upload CSV files", type=["csv"], accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files:
            try:
                dfs.append(pd.read_csv(f))
            except UnicodeDecodeError:
                dfs.append(pd.read_csv(f, encoding="ISO-8859-1"))



elif upload_type == "ZIP File":
    zip_file = st.sidebar.file_uploader("📂 Upload ZIP containing CSV files", type=["zip"])

    if zip_file:
        with zipfile.ZipFile(zip_file) as z:
            for filename in z.namelist():

                # ⛔ Skip macOS metadata files
                if (
                    not filename.lower().endswith(".csv")
                    or filename.startswith("__MACOSX/")
                    or filename.split("/")[-1].startswith("._")
                ):
                    continue

                with z.open(filename) as f:
                    try:
                        df_temp = pd.read_csv(
                            io.TextIOWrapper(f, encoding="utf-8"),
                            on_bad_lines="skip"
                        )
                    except UnicodeDecodeError:
                        f.seek(0)
                        df_temp = pd.read_csv(
                            io.TextIOWrapper(f, encoding="ISO-8859-1"),
                            on_bad_lines="skip"
                        )

                    if not df_temp.empty:
                        dfs.append(df_temp)



if len(dfs) == 0:
    st.error("❌ Please Upload valid CSV Files or Zip File.")
    st.stop()

df = pd.concat(dfs, ignore_index=True)
st.success(f"Total records combined: **{len(df)}**")

# ----------------- Preprocessing -----------------
if "Timestamp" not in df.columns:
    st.error("❌ Missing required column: Timestamp")
    st.stop()

# ensure Timestamp is datetime
#df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")


# ---- Timestamp cleanup (ZIP-safe) ----
df["Timestamp"] = (
    df["Timestamp"]
    .astype(str)
    .str.strip()
    .str.replace("\ufeff", "", regex=False)
)

df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)


df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

# Create combined label column (Label_Section) so dropdowns show e.g. ACCurrent_g1
# Ensure Section & Label exist; fill missing with "unknown"
df["Section"] = df.get("Section", "").fillna("").astype(str)
df["Label"] = df.get("Label", "").fillna("").astype(str)
df["Label_Full"] = df["Label"].astype(str) + "_" + df["Section"].astype(str)
# If Section is empty string produce just Label (avoid trailing underscore)
df["Label_Full"] = df.apply(lambda r: r["Label"] if (r["Section"] == "" or pd.isna(r["Section"])) else f"{r['Label']}_{r['Section']}", axis=1)

# Try convert Value to numeric if possible
if "Value" in df.columns:
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

# ----------------- Date-Time Filter (Sidebar) -----------------
st.sidebar.subheader("📅 Date & ⏰ Time Filter")
min_dt = df["Timestamp"].min()
max_dt = df["Timestamp"].max()
now = max_dt

preset = st.sidebar.selectbox("Quick Select Range", ["Custom Range", "Today", "Last 24 Hours", "Last 7 Days"])
if preset == "Today":
    start_datetime = now.replace(hour=0, minute=0, second=0)
    end_datetime = now
elif preset == "Last 24 Hours":
    start_datetime = now - pd.Timedelta(hours=24)
    end_datetime = now
elif preset == "Last 7 Days":
    start_datetime = now - pd.Timedelta(days=7)
    end_datetime = now
else:
    start_date = st.sidebar.date_input("Start Date", min_dt.date())
    end_date = st.sidebar.date_input("End Date", max_dt.date())
    start_time = st.sidebar.time_input("Start Time", min_dt.time())
    end_time = st.sidebar.time_input("End Time", max_dt.time())
    start_datetime = datetime.combine(start_date, start_time)
    end_datetime = datetime.combine(end_date, end_time)

if start_datetime > end_datetime:
    st.sidebar.error("❌ Start DateTime must be earlier")
    st.stop()

df = df[(df["Timestamp"] >= start_datetime) & (df["Timestamp"] <= end_datetime)].copy()
st.sidebar.success(f"⏳ Showing {len(df)} records")

if df.empty:
    st.error("❌ No data in selected range")
    st.stop()

# ----------------- SENSOR SELECTION (NEW DROPDOWNS) -----------------
st.sidebar.subheader("🔎 Sensor Filters (Section → Label_Section)")

# Ensure Section and Label_Full columns exist in the tall data
if "Section" in df.columns and "Label" in df.columns and "Value" in df.columns:
    sections = list(df["Section"].fillna("Unknown").unique())
    sections.insert(0, "All Sections")
    section_selected = st.sidebar.selectbox("Select Section", sections, index=0)

    if section_selected == "All Sections":
        available_labels = sorted(df["Label_Full"].dropna().unique().tolist())
    else:
        # when a section is chosen, show only labels for that section in combined form
        available_labels = sorted(df.loc[df["Section"] == section_selected, "Label_Full"].dropna().unique().tolist())

    # default select some sensible labels if many exist
    labels_selected = st.sidebar.multiselect(
        "Select Labels (choose one or more) — format: Label_Section e.g. ACCurrent_g1",
        options=available_labels,
        default=available_labels[:6] if len(available_labels) > 0 else []
    )

    if len(labels_selected) == 0:
        st.sidebar.info("Pick one or more combined labels to enable variable selection.")
else:
    st.sidebar.error("Dataframe missing one of required columns: Section, Label, Value")
    st.stop()

# ----------------- PIVOT to WIDE format after sensor selection -----------------
if len(labels_selected) > 0:
    try:
        # filter rows by Label_Full selection
        df_filtered = df[df["Label_Full"].isin(labels_selected)].copy()

        # Create a pivot using the Label_Full as column names
        df_wide = df_filtered.pivot_table(index="Timestamp", columns="Label_Full", values="Value", aggfunc="mean").reset_index()
        df_pivot = df_wide.copy()
    except Exception as e:
        st.error(f"Pivot failed: {e}")
        df_pivot = df.copy()
else:
    # If user didn't select labels, fall back to pivoting all Label_Full (safe default)
    if "Label_Full" in df.columns and "Value" in df.columns:
        try:
            df_pivot = df.pivot_table(index="Timestamp", columns="Label_Full", values="Value", aggfunc="mean").reset_index()
        except Exception:
            # last resort: try pivot by Label (original)
            try:
                df_pivot = df.pivot_table(index="Timestamp", columns="Label", values="Value", aggfunc="mean").reset_index()
            except:
                df_pivot = df.copy()
    else:
        df_pivot = df.copy()

# Ensure Timestamp column exists in df_pivot
if "Timestamp" not in df_pivot.columns and df_pivot.index.name == "Timestamp":
    df_pivot = df_pivot.reset_index()

# numeric columns to choose from
numeric_cols = df_pivot.select_dtypes(include=[np.number]).columns.tolist()

# ----------------- Sidebar Navigation -----------------
st.sidebar.markdown("## 📂 Navigation")
pages = [
    "Data Preview",
    "Statistics & Correlation",
    "Time-Series & Efficiency",
    "Anomalies",
    "Solar vs Grid Analysis"

]
selected_page = st.sidebar.radio("Go to:", pages)
st.session_state["active_page"] = selected_page

# ----------------- Page Title -----------------
st.title("📊 Solar-based EV Charging Analysis Dashboard")
st.markdown("Upload CSV/ZIP with all sensor data (Timestamp, Section, Label, Value). Choose sensors (Label_Section) in the sidebar, then analyze.")

# ---------------------------------------------------------------
#                     ⬇️ PAGE CONTENT SWITCHER ⬇️
# ---------------------------------------------------------------
page = st.session_state["active_page"]

# ============================================================
# ----------------- PAGE 1: DATA PREVIEW ---------------------
# ============================================================
if page == "Data Preview":
    st.header("📋 Data Preview")

    show_preview = st.checkbox("Show Data Preview (Top 10 Rows)", value=False)
    if show_preview:
        st.dataframe(df.head(10))

    show_full = st.checkbox("Show Full Filtered Data", value=False)
    if show_full:
        st.dataframe(df)

    # Show pivot preview
    st.subheader("Pivoted Wide Data (Timestamp x Labels)")
    st.dataframe(df_pivot.head(10))

    st.download_button("⬇️ Download CSV (pivoted)", df_pivot.to_csv(index=False).encode("utf-8"), file_name="filtered_pivot.csv", mime="text/csv")

    # Timestamp distribution
    st.subheader("📊 Timestamp Distribution (Data Density)")
    hist_df = df.copy()
    fig_hist = px.histogram(hist_df, x="Timestamp", nbins=50, title="Timestamp Distribution")
    fig_hist.update_layout(bargap=0.05)
    st.plotly_chart(fig_hist, width='stretch')

    # (rest of page 1 unchanged) ...
    # keep your previous advanced timestamp analysis code here (unchanged)

    # st.subheader("📊 Advanced Timestamp Analysis")


    # ======================================================
    ##Data Count per Hour & Minutes Coverage

    st.markdown("### 📋 Data Count per Hour & Minutes Coverage")

        # Extract hour and minute
    df['Hour'] = df['Timestamp'].dt.hour
    df['Minute'] = df['Timestamp'].dt.minute

        # --- Count number of records per hour ---
    hourly_records = (
        df.groupby('Hour')
            .size()
            .reset_index(name='Number of Records')
    )

    # --- Count unique minutes per hour ---
    hourly_minutes = (
        df.groupby('Hour')['Minute']
            .nunique()
            .reset_index(name='Minutes Covered')
    )

        # Merge
    hourly_summary = pd.merge(hourly_records, hourly_minutes, on='Hour')

        # Compute missing minutes + missing percentage
    hourly_summary['Missing Minutes'] = 60 - hourly_summary['Minutes Covered']
    hourly_summary['Missing %'] = (hourly_summary['Missing Minutes'] / 60 * 100).round(2)

        # Rename columns
    hourly_summary = hourly_summary.rename(columns={
        'Hour': 'Hour of Day'
    })

        # Sort by hour
    hourly_summary = hourly_summary.sort_values('Hour of Day')

        # ---------------------------------------
        # 🎨 COLOR-CODING RULES
        # ---------------------------------------
    def highlight_row(row):
        missing = row['Missing %']

        if missing == 0:       # perfect data
            color = 'background-color: #d4f8d4'   # green
        elif missing <= 30:    # acceptable
            color = 'background-color: #fff3b0'   # yellow
        else:                  # poor data
            color = 'background-color: #f8d4d4'   # red

        return [color] * len(row)

    styled_table = hourly_summary.style.apply(highlight_row, axis=1)

    st.dataframe(styled_table, use_container_width=True)


    # --------------------------------------------
    # RIGHT: Missing Data Detector
    # --------------------------------------------

    st.markdown("### 🕒 Missing Data Summary")

    # --- Minute-level ---
    full_range_min = pd.date_range(df['Timestamp'].min(), df['Timestamp'].max(), freq='1min')
    missing_min_count = len(full_range_min.difference(df['Timestamp']))

    # --- Hour-level ---
    df['Hour'] = df['Timestamp'].dt.floor('H')
    full_range_hour = pd.date_range(df['Timestamp'].min(), df['Timestamp'].max(), freq='1H')
    missing_hour_count = len(full_range_hour.difference(df['Hour']))

    # Summary table
    summary = pd.DataFrame({
        "Category": ["Minutes Missing", "Hours Missing"],
        "Count": [missing_min_count, missing_hour_count]
    })

    # ✅ Fix: Ensure Count is integer
    summary["Count"] = summary["Count"].astype(int)

    # Color function
    def color_missing(val):
        if val > 1000:
            return "background-color: #ffcccc"   # red
        elif val > 0:
            return "background-color: #fff3cd"   # yellow
        return "background-color: #d4edda"        # green

    # Show colored summary table
    st.dataframe(summary.style.applymap(color_missing, subset=["Count"]), height=120)

# ============================================================
# --------- PAGE 2: STATISTICS & CORRELATION -----------------
# ============================================================
elif page == "Statistics & Correlation":
    st.header("📈 Statistics & Correlation")

    if len(numeric_cols) == 0:
        st.warning("No numeric columns available. Make sure labels are selected in the sidebar.")
    else:
        selected_vars = st.multiselect("Select variables:", numeric_cols, default=numeric_cols[:4])
        if len(selected_vars) >= 2:
            corr = df_pivot[selected_vars].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(corr)

            st.subheader("Data Insights")
            insights = []
            high_threshold = 0.7
            medium_threshold = 0.5
            for col1 in selected_vars:
                for col2 in selected_vars:
                    if col1 != col2:
                        value = corr.loc[col1, col2]
                        if value >= high_threshold:
                            insights.append(f" **{col1}** and **{col2}** have a **strong positive correlation** ({value:.2f}).")
                        elif value <= -high_threshold:
                            insights.append(f" **{col1}** and **{col2}** have a **strong negative correlation** ({value:.2f}).")
                        elif abs(value) >= medium_threshold:
                            insights.append(f" **{col1}** and **{col2}** show a **moderate relationship** ({value:.2f}).")
            insights = list(dict.fromkeys(insights))
            if len(insights) == 0:
                st.info("No meaningful correlations detected between selected variables.")
            else:
                for text in insights:
                    st.markdown(text)
        else:
            st.info("Select at least 2 variables to compute correlation.")

# ============================================================
# --------- PAGE 3: TIME SERIES & EFFICIENCY -----------------
# ============================================================
elif page == "Time-Series & Efficiency":
    st.header("📆 Time Series")

    def generate_insights(df_local, timestamp_col, value_col):
        insights = []
        if timestamp_col not in df_local.columns or value_col not in df_local.columns:
            return ["Column not found in dataframe."]
        series = df_local[value_col].dropna()
        ts = df_local[timestamp_col].loc[series.index]
        if len(series) < 2:
            return ["Not enough data to generate insights."]
        if series.iloc[-1] > series.iloc[0]:
            insights.append(f"{value_col} is trending upward over time.")
        elif series.iloc[-1] < series.iloc[0]:
            insights.append(f"{value_col} is trending downward over time.")
        else:
            insights.append(f"{value_col} remains stable during this period.")
        insights.append(f"Highest {value_col} = {series.max():.2f} at {ts.iloc[series.idxmax()]}.")
        insights.append(f"Lowest {value_col} = {series.min():.2f} at {ts.iloc[series.idxmin()]}.")
        if series.std() > (series.mean() * 0.5):
            insights.append("High variability detected.")
        else:
            insights.append("Variability is within normal range.")
        diff = series.diff().abs()
        spikes = diff[diff > (series.std() * 1.5)]
        if len(spikes) > 0:
            insights.append(f"Detected {len(spikes)} sudden spikes/drops.")
        else:
            insights.append("No significant spikes detected.")
        return insights

    # --- FUNCTION: Create PDF (uses temp image files, then removes them) ---
    def create_pdf_report_with_temp_images(ts_fig, eff_fig, insights_dict, filename="Solar_EV_Report.pdf", logo_path=None):
        import os
        import tempfile
        from fpdf import FPDF

        class PDFWithFooter(FPDF):
            def footer(self):
                # Position at 1.5 cm from bottom
                self.set_y(-15)
                self.set_font("Arial", "I", 8)
                self.cell(0, 10, f"Page {self.page_no()}", align="C")

        # Create temp files for charts
        tmp_ts = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_eff = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_ts.close()
        tmp_eff.close()
        ts_path = tmp_ts.name
        eff_path = tmp_eff.name

        # Save figures to PNG (requires kaleido)
        ts_fig.write_image(ts_path, width=800, height=400)
        eff_fig.write_image(eff_path, width=800, height=400)

        # Build PDF
        pdf = PDFWithFooter()
        pdf.add_page()

        # Add logo if provided
        if logo_path and os.path.exists(logo_path):
            pdf.image(logo_path, x=10, y=8, w=30)
            pdf.ln(25)  # space after logo
        else:
            pdf.ln(10)

        # Title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Solar EV System Analysis Report", ln=True, align="C")
        pdf.ln(8)

        # Insert charts
        pdf.image(ts_path, x=10, w=pdf.w - 20)
        pdf.ln(5)
        pdf.image(eff_path, x=10, w=pdf.w - 20)
        pdf.ln(10)

        # Insights section
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Auto Insights", ln=True)
        pdf.ln(4)
        pdf.set_font("Arial", "", 11)

        left_margin = 10
        pdf.set_left_margin(left_margin)
        pdf.set_right_margin(left_margin)

        for key in ["Efficiency", "Loss", "Utilization"]:
            lines = insights_dict.get(key, [])
            pdf.set_font("Arial", "B", 11)
            pdf.set_x(left_margin)  # Reset X position for heading
            pdf.cell(0, 8, f"* {key} Insights:", ln=True)
            pdf.set_font("Arial", "", 10)

            if not lines:
                pdf.set_x(left_margin)
                pdf.multi_cell(0, 6, "No insights available.", align="L")
            else:
                for line in lines:
                    safe_line = str(line).replace("\r", " ").replace("\n", " ")
                    pdf.set_x(left_margin)
                    pdf.multi_cell(0, 6, safe_line, align="L")
                    # Add new page if running out of space
                    if pdf.get_y() > pdf.h - 40:
                        pdf.add_page()


            pdf.ln(4)

        # Save PDF
        pdf.output(filename)

        # Read file into bytes
        with open(filename, "rb") as f:
            pdf_bytes = f.read()

        # Clean temp images
        try: os.remove(ts_path)
        except: pass
        try: os.remove(eff_path)
        except: pass

        return pdf_bytes


    if len(numeric_cols) == 0:
        st.warning("No numeric columns available. Choose labels in the sidebar.")
    else:
        var = st.selectbox("Select variable for time-series:", numeric_cols)
        fig_ts = px.line(df_pivot, x="Timestamp", y=var, title=f"{var} Over Time")
        st.plotly_chart(fig_ts, use_container_width=True)
        ts_insights = generate_insights(df_pivot, "Timestamp", var)
        st.subheader("📌 Time-Series Insights")
        for line in ts_insights:
            st.markdown(line)

        # Efficiency (requires at least two numeric series)
        st.subheader("⚡ Efficiency, Loss & Utilization")
        if len(numeric_cols) >= 2:
            solar_col = st.selectbox("Solar Input (choose solar power/current/voltage)", numeric_cols, index=0, key="solar_input")
            battery_col = st.selectbox("Battery Output (choose battery metric)", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="battery_output")
            df_eff = df_pivot.copy()
            max_solar = df_eff[solar_col].max() if solar_col in df_eff.columns else 0

            def safe_div(a, b):
                try:
                    return (a / b) if b and b != 0 and not pd.isna(b) else np.nan
                except Exception:
                    return np.nan

            df_eff["Efficiency (%)"] = df_eff.apply(lambda row: safe_div(row.get(battery_col, np.nan), row.get(solar_col, np.nan)) * 100, axis=1)
            df_eff["Efficiency (%)"] = df_eff["Efficiency (%)"].fillna(0)
            df_eff["Loss (%)"] = 100 - df_eff["Efficiency (%)"]
            df_eff["Utilization (%)"] = df_eff.apply(lambda row: (safe_div(row.get(battery_col, np.nan), max_solar) * 100) if max_solar and max_solar != 0 else 0, axis=1)

            fig_eff = px.line(df_eff, x="Timestamp", y=["Efficiency (%)", "Loss (%)", "Utilization (%)"], title="System Efficiency, Loss, and Utilization Over Time")
            st.plotly_chart(fig_eff, use_container_width=True)

            insights_dict = {
                "Efficiency": generate_insights(df_eff, "Timestamp", "Efficiency (%)"),
                "Loss": generate_insights(df_eff, "Timestamp", "Loss (%)"),
                "Utilization": generate_insights(df_eff, "Timestamp", "Utilization (%)"),
            }

            for key, title in zip(["Efficiency", "Loss", "Utilization"], ["Efficiency Insights", "Loss Insights", "Utilization Insights"]):
                st.subheader(f"📌 {title}")
                for line in insights_dict[key]:
                    st.markdown(line)

            st.download_button(label="Download CSV", data=df_eff.to_csv(index=False).encode("utf-8"), file_name="time_series_efficiency_results.csv", mime="text/csv")

            # PDF generation (unchanged)...
            # (kept your previous create_pdf_report_with_temp_images implementation)


            # PDF download - create on click and stream bytes
            st.subheader("📄 Download Insights as PDF")
            if st.button("Generate PDF"):
                try:
                    pdf_bytes = create_pdf_report_with_temp_images(fig_ts, fig_eff, insights_dict, filename="Solar_EV_Report.pdf")
                    st.success("PDF generated — click Download below.")
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name="Solar_EV_Report.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

        else:
            st.info("At least 2 numeric columns are required to calculate efficiency.")

# ============================================================
# ----------------- PAGE 4: ANOMALIES -------------------------
# ============================================================
elif page == "Anomalies":
    st.header("🚨 Anomaly Detection")

    if len(numeric_cols) == 0:
        st.warning("No numeric columns available. Choose labels in the sidebar.")
    else:
        vars_selected = st.multiselect(
            "Select variables for anomaly detection:",
            numeric_cols,
            default=numeric_cols[:3]
        )
        contamination = st.slider(
            "Contamination (expected fraction of anomalies)",
            0.001, 0.2, 0.02, step=0.001
        )

        if st.button("Run Detection"):
            if not vars_selected:
                st.error("Please select variables first.")
            else:
                df_clean = df_pivot.dropna(subset=vars_selected).copy()
                if df_clean.empty:
                    st.error("No data after dropping NaNs for selected variables.")
                else:
                    model = IsolationForest(
                        contamination=float(contamination), random_state=42
                    )
                    model.fit(df_clean[vars_selected])
                    df_clean["anomaly"] = model.predict(df_clean[vars_selected])
                    anomalies_df = df_clean[df_clean["anomaly"] == -1].copy()
                    anomalies_df["Insight"] = anomalies_df.apply(
                        lambda row: generate_anomaly_insight(row, vars_selected, df_clean),
                        axis=1
                    )
                    st.session_state["anomaly_full"] = df_clean
                    st.session_state["anomaly_only"] = anomalies_df
                    st.session_state["selected_vars"] = vars_selected

        if "anomaly_only" in st.session_state:
            df_clean = st.session_state["anomaly_full"]
            anomalies_df = st.session_state["anomaly_only"]
            vars_selected = st.session_state.get("selected_vars", [])

            if len(vars_selected) == 0:
                st.error("No variables selected.")
            else:
                # Melt dataframe for multiple variables plotting
                df_melt = df_clean.melt(
                    id_vars=["Timestamp", "anomaly"],
                    value_vars=vars_selected,
                    var_name="Variable",
                    value_name="Value"
                )

                # Map anomaly colors
                df_melt["Anomaly_Flag"] = df_melt["anomaly"].map({1: "Normal", -1: "Anomaly"})
                color_map = {"Normal": "green", "Anomaly": "red"}

                fig = px.scatter(
                    df_melt,
                    x="Timestamp",
                    y="Value",
                    color="Anomaly_Flag",
                    facet_col="Variable",
                    color_discrete_map=color_map,
                    title="Anomaly Detection (Red = Anomaly, Green = Normal)"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Summary
                summary_text = generate_summary(anomalies_df, vars_selected, df_clean)
                st.markdown(summary_text)

                # Highlight anomalies in table
                def highlight_anomalies(row):
                    return ['background-color: #ffcccc'] * len(row) if row.get("anomaly", 1) == -1 else ['background-color: #d4ffd4'] * len(row)

                st.subheader("📄 Anomaly Table")
                st.dataframe(
                    anomalies_df.style.apply(highlight_anomalies, axis=1),
                    use_container_width=True
                )

                # CSV Download
                csv_data = anomalies_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Anomaly Results (CSV)",
                    data=csv_data,
                    file_name="anomalies_with_insights.csv",
                    mime="text/csv"
                )




# ============================================================
# --------- PAGE: Solar vs Grid Analysis -------------------
# ============================================================
elif page == "Solar vs Grid Analysis":

    import streamlit as st
    import pandas as pd
    import plotly.express as px

    st.header("☀️ Solar vs Grid Analysis")

    # =====================================================
    # Helper Functions
    # =====================================================
    def combine_energy(df, label):
        """
        Combine g1 + g2 power labels and convert to kWh per minute
        """
        tmp = df[
            (df["Section"].isin(["g1", "g2"])) &
            (df["Label"] == label)
        ].copy()

        if tmp.empty:
            return None

        tmp["Timestamp"] = pd.to_datetime(tmp["Timestamp"])
        tmp["Energy_kWh"] = tmp["Value"] / 1000 / 60  # W → kWh per minute

        return tmp.groupby("Timestamp")["Energy_kWh"].sum().reset_index().sort_values("Timestamp")

    def myanmar_grid_cost(units_kwh):
        """Myanmar slab-based electricity tariff (Kyat)"""
        if units_kwh <= 50:
            return units_kwh * 50
        elif units_kwh <= 100:
            return (50 * 50) + (units_kwh - 50) * 100
        elif units_kwh <= 200:
            return (50 * 50) + (50 * 100) + (units_kwh - 100) * 150
        else:
            return (50 * 50) + (50 * 100) + (100 * 150)

    # =====================================================
    # Data Preparation
    # =====================================================
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    solar_energy = combine_energy(df, "solarPower")
    grid_energy = combine_energy(df, "ACWatt")
    load_energy = combine_energy(df, "outputWatt")

    ev = df[(df["Section"] == "kws") & (df["Label"] == "outputPower")].copy()
    ev["Energy_kWh"] = ev["Value"] / 1000 / 60

    # =====================================================
    # DATE RANGE SELECTION
    # =====================================================
    st.subheader("🗓 Select Date Range")
    min_date = df["Timestamp"].min().date()
    max_date = df["Timestamp"].max().date()
    start_date, end_date = st.date_input(
        "Select Start and End Date",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if start_date > end_date:
        st.error("❌ Start date must be before end date.")
    else:
        # Filter each dataframe for the selected date range
        solar_agg = solar_energy[
            (solar_energy["Timestamp"].dt.date >= start_date) &
            (solar_energy["Timestamp"].dt.date <= end_date)
        ]
        grid_agg = grid_energy[
            (grid_energy["Timestamp"].dt.date >= start_date) &
            (grid_energy["Timestamp"].dt.date <= end_date)
        ]
        load_agg = load_energy[
            (load_energy["Timestamp"].dt.date >= start_date) &
            (load_energy["Timestamp"].dt.date <= end_date)
        ]
        ev_agg = ev[
            (ev["Timestamp"].dt.date >= start_date) &
            (ev["Timestamp"].dt.date <= end_date)
        ]

        # =====================================================
        # Daily Aggregation (to reduce clutter)
        # =====================================================
        solar_agg = solar_agg.resample("D", on="Timestamp").sum().reset_index()
        grid_agg = grid_agg.resample("D", on="Timestamp").sum().reset_index()
        load_agg = load_agg.resample("D", on="Timestamp").sum().reset_index()
        ev_agg = ev_agg.resample("D", on="Timestamp").sum().reset_index()

        # Rename columns
        solar_agg.rename(columns={"Energy_kWh": "Solar_kWh"}, inplace=True)
        grid_agg.rename(columns={"Energy_kWh": "Grid_kWh"}, inplace=True)
        load_agg.rename(columns={"Energy_kWh": "Load_kWh"}, inplace=True)
        ev_agg.rename(columns={"Energy_kWh": "EV_kWh"}, inplace=True)

        # Grid cost
        grid_agg["Grid_Cost_Kyat"] = grid_agg["Grid_kWh"].apply(myanmar_grid_cost)

        # Merge all
        merged = (
            solar_agg
            .merge(grid_agg, on="Timestamp", how="left")
            .merge(load_agg, on="Timestamp", how="left")
            .merge(ev_agg, on="Timestamp", how="left")
            .fillna(0)
        )

        # =====================================================
        # KEY INSIGHTS (TOP + TOTAL SUM)
        # =====================================================
        total_solar = merged["Solar_kWh"].sum()
        total_grid = merged["Grid_kWh"].sum()
        total_load = merged["Load_kWh"].sum()
        total_ev = merged["EV_kWh"].sum()

        primary_source = "Solar" if total_solar >= total_grid else "Grid"

        st.subheader("📌 Key Insights (Total for Selected Range)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("☀️ Solar Energy (kWh)", round(total_solar, 2))
        col2.metric("🔌 Grid Energy (kWh)", round(total_grid, 2))
        col3.metric("🏠 Total Load (kWh)", round(total_load, 2))
        col4.metric("🚗 EV Charging (kWh)", round(total_ev, 2))

        st.success(f"Primary energy source: **{primary_source}**")

        # =====================================================
        # Visualization 1: Solar vs Grid (Bar Chart)
        # =====================================================
        fig1 = px.bar(
        merged,
        x="Timestamp",
        y=["Solar_kWh", "Grid_kWh"],
        title="Solar Vs Grid Energy Usage",
        labels={"value": "Energy (kWh)", "Timestamp": "Time"},
        color_discrete_map={
            "Solar_kWh": "#FFD700",
            "Grid_kWh": "#888888"
        }
        )
        st.plotly_chart(fig1, use_container_width=True)

        # =====================================================
        # Visualization 2: Total Load vs EV Charging (Bar Chart)
        # =====================================================
        fig2 = px.bar(
        merged,
        x="Timestamp",
        y=["Load_kWh", "EV_kWh"],
        title="Total Load Vs EV Charging Energy",
        labels={"value": "Energy (kWh)", "Timestamp": "Time"},
        color_discrete_map={
            "Load_kWh": "#1f77b4",
            "EV_kWh": "#2ca02c"
        }
        )
        st.plotly_chart(fig2, use_container_width=True)

        # =====================================================
        # Optional Data Table
        # =====================================================
        with st.expander("📋 View Aggregated Energy Summary Table"):
            st.dataframe(merged)

        # =====================================================
        # CSV Download
        # =====================================================
        st.download_button(
            "⬇️ Download Aggregated Data (CSV)",
            data=merged.to_csv(index=False).encode("utf-8"),
            file_name=f"solar_vs_grid_{start_date}_{end_date}.csv",
            mime="text/csv"
        )
