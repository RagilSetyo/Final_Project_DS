import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# --------------------------
# KONFIGURASI HALAMAN
# --------------------------
st.set_page_config(page_title="Analisis Harga Rumah", layout="wide")
st.title("🏠 Analisis Harga Rumah di Washington")

# --------------------------
# LOAD DATA
# --------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data.csv")
        return df
    except FileNotFoundError:
        st.error("❌ File data.csv tidak ditemukan.")
        return None

house = load_data()

if house is not None:
    st.success("✅ Dataset berhasil dimuat!")

    # Tabs utama
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 EDA",
        "🧩 Preprocessing",
        "🤖 Modeling",
        "📈 Feature Importance",
        "🏠 Kesimpulan"
    ])

    # ==========================================================
    # ====================== TAB EDA ============================
    # ==========================================================
    with tab1:
        st.header("📊 Exploratory Data Analysis (EDA)")

        with st.expander("📄 Lihat Data Awal"):
            st.dataframe(house.head())

        # Data Cleaning
        st.subheader("🧹 Data Cleaning")

        before_rows = len(house)
        house_clean = house[house["price"] > 0]
        after_zero_removed = len(house_clean)

        Q1 = house_clean["price"].quantile(0.25)
        Q3 = house_clean["price"].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        house_clean = house_clean[(house_clean["price"] >= lower) & (house_clean["price"] <= upper)]
        after_outlier_removed = len(house_clean)

        st.write(f"💾 Jumlah baris awal: **{before_rows:,}**")
        st.write(f"🚫 Setelah hapus harga 0: **{after_zero_removed:,}**")
        st.write(f"🧹 Setelah hapus outlier: **{after_outlier_removed:,}**")

        # Visualisasi
        st.subheader("🏙️ Rata-rata Harga Rumah per Kota & StateZip")

        top_n = st.slider("Tampilkan berapa kota teratas?", 5, 30, 10)
        avg_city = house_clean.groupby("city")["price"].mean().sort_values(ascending=False).head(top_n)
        avg_statezip = house_clean.groupby("statezip")["price"].mean().sort_values(ascending=False).head(top_n)

        # Plot City
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.barplot(x=avg_city.values, y=avg_city.index, palette="viridis", ax=ax1)
        ax1.set_title("Rata-rata Harga Rumah per Kota", fontsize=14, fontweight='bold')
        for i, v in enumerate(avg_city.values):
            ax1.text(v, i, f"${v:,.0f}", va="center")
        st.pyplot(fig1)

        # Plot StateZip
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.barplot(x=avg_statezip.values, y=avg_statezip.index, palette="coolwarm", ax=ax2)
        ax2.set_title("Rata-rata Harga Rumah per Statezip", fontsize=14, fontweight='bold')
        for i, v in enumerate(avg_statezip.values):
            ax2.text(v, i, f"${v:,.0f}", va="center")
        st.pyplot(fig2)

        # BOX PLOT
        st.subheader("📦 Distribusi Harga Berdasarkan Kondisi Rumah")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.boxplot(x="condition", y="price", data=house_clean, palette="cool", ax=ax3)
        st.pyplot(fig3)

        # SCATTERPLOT
        st.subheader("📈 Hubungan Luas Bangunan dengan Harga Rumah")
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=house_clean, x="sqft_living", y="price", alpha=0.6, ax=ax4)
        st.pyplot(fig4)

    # ==========================================================
    # ================= TAB PREPROCESSING =======================
    # ==========================================================
    with tab2:
        st.header("🧩 Data Preprocessing")

        X = house_clean.drop(columns=["price"])
        y = house_clean["price"]

        st.subheader("✂️ Train-Test Split")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

        st.write(f"🧠 Data latih: **{len(X_train):,}**")
        st.write(f"🧪 Data uji: **{len(X_test):,}**")

        # Heatmap Korelasi
        st.subheader("🔥 Heatmap Korelasi")
        num_train = X_train.select_dtypes(include=['int64', 'float64'])
        fig_corr, ax_corr = plt.subplots(figsize=(10,7))
        sns.heatmap(num_train.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
        st.pyplot(fig_corr)

        # Seleksi fitur & VIF
        st.subheader("📉 Seleksi Fitur dan VIF")
        st.write("Pada kasus ini, kolom sqft_living akan di drop")
        num_train = num_train.drop(columns=['sqft_living'])
        X_test = X_test.drop(columns=['sqft_living'])
        X_const = add_constant(num_train)
        vif_df = pd.DataFrame([vif(X_const.values, i) for i in range(X_const.shape[1])],
                              index=X_const.columns).reset_index()
        vif_df.columns = ['feature', 'vif_score']
        vif_df = vif_df.loc[vif_df.feature != 'const']
        st.dataframe(vif_df)

        # One Hot Encoding + Normalisasi
        cat_cols = X_train.select_dtypes(include='object').columns
        num_cols = num_train.select_dtypes(include=['int64','float64']).columns

        X_train_cat = X_train[cat_cols]
        X_train_num = X_train[num_cols]

        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_train_cat_enc = encoder.fit_transform(X_train_cat)
        X_train_cat_enc = pd.DataFrame(
            X_train_cat_enc,
            columns=encoder.get_feature_names_out(cat_cols),
            index=X_train.index
        )
        X_train_final = pd.concat([X_train_num, X_train_cat_enc], axis=1)

        X_test_cat = X_test[cat_cols]
        X_test_num = X_test[num_cols]
        X_test_cat_enc = encoder.transform(X_test_cat)
        X_test_cat_enc = pd.DataFrame(
            X_test_cat_enc,
            columns=encoder.get_feature_names_out(cat_cols),
            index=X_test.index
        )
        X_test_final = pd.concat([X_test_num, X_test_cat_enc], axis=1)

        # Normalisasi MinMax
        scaler = MinMaxScaler()
        minmax_scaler = scaler.fit(X_train_final)
        X_train_scaled = minmax_scaler.transform(X_train_final)
        X_test_scaled = minmax_scaler.transform(X_test_final)

        st.session_state.update({
            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "X_train_final": X_train_final
        })

        st.success("✅ Data berhasil di-preprocessing dan siap untuk modeling!")

    # ==========================================================
    # ===================== TAB MODELING ========================
    # ==========================================================
    with tab3:
        st.header("🤖 Modeling — Linear Regression, Random Forest & XGBoost")

        if "X_train_scaled" in st.session_state:
            X_train_scaled = st.session_state["X_train_scaled"]
            X_test_scaled = st.session_state["X_test_scaled"]
            y_train = st.session_state["y_train"]
            y_test = st.session_state["y_test"]

            results = []

            # ===================== Linear Regression =====================
            model_LR = LinearRegression()
            model_LR.fit(X_train_scaled, y_train)

            y_pred_train = model_LR.predict(X_train_scaled)
            y_pred_test = model_LR.predict(X_test_scaled)

            results.append({
                "Model": "Linear Regression",
                "R² Train": r2_score(y_train, y_pred_train),
                "MAE Train": mean_absolute_error(y_train, y_pred_train),
                "RMSE Train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
                "R² Test": r2_score(y_test, y_pred_test),
                "MAE Test": mean_absolute_error(y_test, y_pred_test),
                "RMSE Test": np.sqrt(mean_squared_error(y_test, y_pred_test))
            })

            # ===================== Random Forest =====================
            model_RF = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=10)
            model_RF.fit(X_train_scaled, y_train)

            y_pred_train = model_RF.predict(X_train_scaled)
            y_pred_test = model_RF.predict(X_test_scaled)

            results.append({
                "Model": "Random Forest Regressor",
                "R² Train": r2_score(y_train, y_pred_train),
                "MAE Train": mean_absolute_error(y_train, y_pred_train),
                "RMSE Train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
                "R² Test": r2_score(y_test, y_pred_test),
                "MAE Test": mean_absolute_error(y_test, y_pred_test),
                "RMSE Test": np.sqrt(mean_squared_error(y_test, y_pred_test))
            })

            # ===================== XGBoost =====================
            model_XGB = XGBRegressor(
                n_estimators=500,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            model_XGB.fit(X_train_scaled, y_train)

            y_pred_train = model_XGB.predict(X_train_scaled)
            y_pred_test = model_XGB.predict(X_test_scaled)

            results.append({
                "Model": "XGBoost Regressor",
                "R² Train": r2_score(y_train, y_pred_train),
                "MAE Train": mean_absolute_error(y_train, y_pred_train),
                "RMSE Train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
                "R² Test": r2_score(y_test, y_pred_test),
                "MAE Test": mean_absolute_error(y_test, y_pred_test),
                "RMSE Test": np.sqrt(mean_squared_error(y_test, y_pred_test))
            })

            # ===================== Hasil Akhir =====================
            st.subheader("📊 Hasil Evaluasi Semua Model")
            st.dataframe(pd.DataFrame(results).style.format({
                "R² Train": "{:.4f}",
                "R² Test": "{:.4f}",
                "MAE Train": "{:,.2f}",
                "MAE Test": "{:,.2f}",
                "RMSE Train": "{:,.2f}",
                "RMSE Test": "{:,.2f}"
            }))
            st.session_state["model_xgb"] = model_XGB

            st.success("✅ Evaluasi 3 model selesai!")
        else:
            st.warning("⚠️ Jalankan preprocessing dulu sebelum modeling.")

    # ==========================================================
    # ===================== TAB FEATURE IMPORTANCE ==============
    # ==========================================================
    with tab4:
        st.header("📈 Feature Importance — XGBoost Regressor")

        if "model_xgb" in st.session_state:
            model_xgb = st.session_state["model_xgb"]
            X_train_final = st.session_state["X_train_final"]

            feature_names = X_train_final.columns
            importances = model_xgb.feature_importances_

            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

            top_features = feature_importance.head(10)

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(top_features['Feature'][::-1], top_features['Importance'][::-1], color='skyblue')
            ax.set_title("Top 10 Feature Importance - XGBoost", fontsize=14)
            ax.set_xlabel("Importance Score", fontsize=12)
            ax.set_ylabel("Feature", fontsize=12)

            max_imp = top_features['Importance'].max()
            for bar in bars:
                width = bar.get_width()
                ax.text(width + max_imp * 0.02, bar.get_y() + bar.get_height()/2,
                        f"{width:.4f}", va='center', ha='left')

            st.pyplot(fig)
            st.markdown("""
### Feature Importance : 

Diketahui bahwa **fitur yang paling berpengaruh** pada model adalah **Statezip** dan **City**,  
terutama pada **statezip_WA 98038**, di mana fitur ini memiliki pengaruh paling kuat dengan nilai **0.0285**.

📍 **Statezip_WA 98038** adalah rumah yang berada di **kota Maple Valley**  
dengan **kode pos 98038**.
""")
        else:
            st.warning("⚠️ Jalankan modeling dulu agar XGBoost tersedia.")

    with tab5:
        st.header("Kesimpulan")
        st.markdown("""
**Model XGBoost Regressor** menunjukkan performa terbaik.

- Faktor utama yang mempengaruhi: **Statezip_WA** dan **City**  
- **MAE:** $66,794 per rumah  
- **RMSE:** $95,761 per rumah  
- **R² Score:** 80.00%  
- **Rata-rata harga rumah:** $487,456.90  

Model ini mampu memprediksi harga rumah dengan akurasi yang cukup tinggi,
menunjukkan bahwa faktor lokasi memiliki peranan yang sangat penting.
""")
        st.title("📊 Evaluasi Akurasi Model")

        # Menampilkan gambar
        st.image("hasil akurasi mae rsme.PNG", 
         caption="Hasil Perhitungan Akurasi Berdasarkan MAE dan RMSE", 
         use_container_width=True)
else:
    st.stop()
