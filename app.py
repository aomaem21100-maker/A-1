"""
Streamlit Web Application for Loan Status Prediction (Self-Contained - Final Version)
รันด้วยคำสั่ง: streamlit run 2_streamlit_app.py
ผู้พัฒนา: นายจตุรภัทร สถาปิตานนท์ 664245024
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

# ============================================================
# Custom CSS (เพิ่มสไตล์สำหรับรูปโปรไฟล์)
# ============================================================
st.markdown("""
<style>
    /* Profile card - จัดกึ่งกลาง */
    .profile-card {
        text-align: center;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    
    /* บังคับให้รูปอยู่กึ่งกลางและเป็นวงกลม */
    [data-testid="stImage"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
    }
    
    [data-testid="stImage"] img {
        border-radius: 50% !important;
        object-fit: cover !important;
        border: 4px solid white !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
        margin: 0 auto !important;
        display: block !important;
    }
    
    /* ซ่อน caption ของรูป (กันชื่อโผล่ใต้รูป) */
    [data-testid="stImage"] figcaption,
    [data-testid="stImage"] p {
        display: none !important;
    }
    
    /* Fallback: รูปวงกลมตัวอักษร (ธีมสีน้ำเงินสำหรับงานการเงิน) */
    .profile-initials {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: linear-gradient(135deg, #0066cc 0%, #004499 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        border: 4px solid white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .profile-initials span {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Embedded Data & Model Training Logic
# ============================================================
CSV_DATA = """person_age,person_gender,person_education,person_income,person_emp_exp,person_home_ownership,loan_amnt,loan_intent,loan_int_rate,loan_percent_income,cb_person_cred_hist_length,credit_score,previous_loan_defaults_on_file,loan_status
23.0,female,Associate,53395.0,1.0,OWN,15000.0,EDUCATION,11.01,0.28,3.0,574.0,Yes,0
23.0,female,Bachelor,60080.0,1.0,RENT,9000.0,MEDICAL,11.01,0.15,2.0,562.0,No,1
24.0,male,Master,68121.0,1.0,RENT,9000.0,HOMEIMPROVEMENT,11.01,0.13,2.0,651.0,Yes,0
24.0,male,Bachelor,67890.0,0.0,RENT,9000.0,PERSONAL,7.29,0.13,2.0,548.0,Yes,0
23.0,female,High School,68170.0,0.0,RENT,9000.0,EDUCATION,13.35,0.13,3.0,579.0,Yes,0
23.0,female,High School,69638.0,0.0,RENT,9000.0,PERSONAL,7.14,0.13,2.0,660.0,No,0
23.0,male,Bachelor,46345.0,1.0,MORTGAGE,10000.0,DEBTCONSOLIDATION,9.63,0.22,3.0,646.0,No,1
24.0,male,Master,37260.0,2.0,MORTGAGE,6000.0,DEBTCONSOLIDATION,13.57,0.16,2.0,634.0,No,0
26.0,male,Master,37000.0,4.0,MORTGAGE,3500.0,VENTURE,8.0,0.09,4.0,569.0,No,0"""

FEATURE_COLUMNS = [
    'person_age', 'person_gender', 'person_education', 'person_income',
    'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
    'credit_score', 'previous_loan_defaults_on_file'
]

def get_or_train_model():
    model_path = 'svm_loan_model.pkl'
    
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.warning(f"⚠️ ไม่สามารถโหลดโมเดลเดิมได้ ({type(e).__name__}) จะทำการสร้างและฝึกโมเดลใหม่...")
    
    with st.spinner("🔧 กำลังเตรียมโครงสร้างข้อมูลและสร้างโมเดล SVM ใหม่..."):
        df = pd.read_csv(StringIO(CSV_DATA))
        X = df[FEATURE_COLUMNS]
        y = df['loan_status']
        
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        try:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', encoder)
                ]), categorical_features)
            ])
            
        svm_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
        ])
        
        svm_pipeline.fit(X, y)
        joblib.dump(svm_pipeline, model_path)
        st.success("✅ บันทึกโมเดลจำลองลงเครื่องเรียบร้อยแล้ว!")
        
    return svm_pipeline

@st.cache_resource
def load_cached_model():
    return get_or_train_model()

model = load_cached_model()

# ============================================================
# Sidebar (เพิ่มรูปผู้พัฒนา + ข้อมูล)
# ============================================================
st.sidebar.title("💰 Loan Prediction App")
st.sidebar.markdown("---")

# 👤 รูปผู้พัฒนา - ลบ caption + จัดกึ่งกลาง
st.sidebar.markdown("<div class='profile-card'>", unsafe_allow_html=True)

image_files = ["developer.jpg", "developer.png", "profile.jpg", "profile.png", "author.jpg"]
image_loaded = False

for img_file in image_files:
    if os.path.exists(img_file):
        # ✅ ไม่มี parameter caption เพื่อป้องกันชื่อโผล่ใต้รูป
        st.sidebar.image(img_file, width=120, use_container_width=False)
        image_loaded = True
        break

if not image_loaded:
    # Fallback: แสดงตัวอักษรแทน หากไม่พบไฟล์รูป
    st.sidebar.markdown("""
    <div class='profile-initials'>
        <span>จส</span>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("</div>", unsafe_allow_html=True)

# ข้อมูลผู้พัฒนา
st.sidebar.markdown("""
---
👨‍💻 **ผู้พัฒนา:** นายจตุรภัทร สถาปิตานนท์  
🆔 **รหัสนักศึกษา:** 664245024  
📚 วิชา Machine Learning / Data Science
""")

st.sidebar.markdown("---")
st.sidebar.info("""
**โมเดลหลัก:** Support Vector Machine (SVM)  
**Kernel:** RBF  
**เป้าหมาย:** ทำนายการผ่านสินเชื่อ (0=อนุมัติ, 1=ผิดนัดชำระ)
""")

# ============================================================
# Main Title
# ============================================================
st.title("🏦 ระบบทำนายสถานะการอนุมัติเงินกู้")
st.markdown("กรุณาป้อนข้อมูลของผู้ขอสินเชื่อเพื่อประเมินความเสี่ยงและการอนุมัติด้วยระบบ AI (SVM)")

# ============================================================
# Input Form
# ============================================================
st.header("📝 ส่วนกรอกข้อมูลรายบุคคล")

col1, col2, col3 = st.columns(3)

with col1:
    person_age = st.number_input("อายุ (Age)", min_value=18, max_value=100, value=25, step=1)
    person_gender = st.selectbox("เพศ (Gender)", ["male", "female"])
    person_education = st.selectbox("ระดับการศึกษา (Education)", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
    person_income = st.number_input("รายได้ต่อปี (Annual Income)", min_value=0, value=50000, step=1000, help="หน่วย: ดอลลาร์สหรัฐ ($)")

with col2:
    person_emp_exp = st.number_input("ปีประสบการณ์ทำงาน (Work Experience)", min_value=0, max_value=50, value=3, step=1)
    person_home_ownership = st.selectbox("สถานะที่อยู่อาศัย (Home Ownership)", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    loan_amnt = st.number_input("จำนวนเงินที่ขอกู้ (Loan Amount)", min_value=0, value=10000, step=500)
    loan_intent = st.selectbox("วัตถุประสงค์การใช้เงินกู้ (Loan Intent)", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])

with col3:
    loan_int_rate = st.number_input("อัตราดอกเบี้ย (%)", min_value=0.0, max_value=30.0, value=10.0, step=0.1)
    loan_percent_income = st.number_input("สัดส่วนเงินกู้ต่อรายได้ (Loan-to-Income Ratio)", min_value=0.0, max_value=1.0, value=0.10, step=0.01, help="เช่น 0.10 หมายถึง เงินกู้คิดเป็น 10% ของรายได้ต่อปี")
    cb_person_cred_hist_length = st.number_input("ระยะเวลาประวัติเครดิต (ปี)", min_value=0, max_value=50, value=5, step=1)
    credit_score = st.number_input("คะแนนเครดิต (Credit Score)", min_value=300, max_value=850, value=650, step=10)
    previous_loan_defaults = st.selectbox("เคยมีประวัติผิดนัดชำระหนี้มาก่อนหรือไม่?", ["No", "Yes"])

# ============================================================
# Single Prediction Logic
# ============================================================
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    predict_button = st.button("🔮 เริ่มทำนายผล", use_container_width=True, type="primary")

if predict_button:
    input_data = pd.DataFrame({
        'person_age': [float(person_age)],
        'person_gender': [person_gender],
        'person_education': [person_education],
        'person_income': [float(person_income)],
        'person_emp_exp': [float(person_emp_exp)],
        'person_home_ownership': [person_home_ownership],
        'loan_amnt': [float(loan_amnt)],
        'loan_intent': [loan_intent],
        'loan_int_rate': [float(loan_int_rate)],
        'loan_percent_income': [float(loan_percent_income)],
        'cb_person_cred_hist_length': [float(cb_person_cred_hist_length)],
        'credit_score': [float(credit_score)],
        'previous_loan_defaults_on_file': [previous_loan_defaults]
    })
    
    input_data = input_data[FEATURE_COLUMNS]
    
    with st.spinner("ระบบกำลังประมวลผลผ่านโมเดล SVM..."):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
    
    st.markdown("---")
    st.header("📊 ผลการวิเคราะห์จากระบบ AI")
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        if prediction == 0:
            st.success("✅ **สถานะที่แนะนำ:** ได้รับการอนุมัติ (Approved)")
            st.metric("โอกาสผ่านการอนุมัติสินเชื่อ", f"{probability[0]*100:.2f}%")
        else:
            st.error("❌ **สถานะที่แนะนำ:** ไม่ได้รับการอนุมัติ / เสี่ยงผิดชำระหนี้ (Default)")
            st.metric("โอกาสที่ผู้กู้จะเบี้ยวหนี้", f"{probability[1]*100:.2f}%")
            
    with col_res2:
        st.subheader("📈 แผนภูมิแสดงเปอร์เซ็นต์ความน่าจะเป็น")
        prob_df = pd.DataFrame({
            'ผลลัพธ์': ['Approved', 'Default'],
            'ความน่าจะเป็น (%)': [probability[0]*100, probability[1]*100]
        })
        st.bar_chart(prob_df.set_index('ผลลัพธ์'))
    
    with st.expander("🔍 ตรวจสอบชุดตัวแปรที่ส่งเข้าวิเคราะห์"):
        st.dataframe(input_data.T.rename(columns={0: 'ค่าตัวแปร'}))

# ============================================================
# Batch Prediction
# ============================================================
st.markdown("---")
st.header("📂 ทำนายผลแบบกลุ่มผ่านไฟล์ CSV")
st.write("คุณสามารถโยนไฟล์ฐานข้อมูลลูกค้ารายการใหญ่ เพื่อทำนายสถานะพร้อมกันในรอบเดียวได้")

uploaded_file = st.file_uploader("เลือกไฟล์ข้อมูลผู้กู้ยืม (.csv)", type=['csv'])

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        st.write(f"📊 ตรวจพบข้อมูลในไฟล์ทั้งหมด: `{df_input.shape[0]}` รายการ")
        
        missing_cols = [col for col in FEATURE_COLUMNS if col not in df_input.columns]
        
        if len(missing_cols) > 0:
            st.error(f"❌ โครงสร้างข้อมูลไม่ครบถ้วน! ขาดคอลัมน์ดังนี้: `{', '.join(missing_cols)}` โปรดตั้งชื่อคอลัมน์ให้ตรงกัน")
        else:
            st.success("👍 โครงสร้างคอลัมน์ถูกต้อง ครบถ้วนตามมาตรฐานโมเดล!")
            st.dataframe(df_input.head())
            
            if st.button("🚀 ประมวลผลกลุ่มทั้งหมด", type="primary"):
                with st.spinner("ระบบกำลังคำนวณผลลัพธ์จากข้อมูลชุดใหญ่..."):
                    df_to_predict = df_input[FEATURE_COLUMNS].copy()
                    num_cols = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 
                                'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']
                    for col in num_cols:
                        df_to_predict[col] = df_to_predict[col].astype(float)
                        
                    batch_predictions = model.predict(df_to_predict)
                    batch_probabilities = model.predict_proba(df_to_predict)
                
                df_result = df_input.copy()
                df_result['Prediction'] = batch_predictions
                df_result['Prediction_Status'] = df_result['Prediction'].map({0: 'Approved', 1: 'Default'})
                df_result['Prob_Approved (%)'] = np.round(batch_probabilities[:, 0] * 100, 2)
                df_result['Prob_Default (%)'] = np.round(batch_probabilities[:, 1] * 100, 2)
                
                st.markdown("---")
                st.subheader("🏁 สรุปผลการประมวลผลภาพรวม")
                
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("จำนวนลูกหนี้ทั้งหมด", len(df_result))
                with col_s2:
                    approved_count = (df_result['Prediction'] == 0).sum()
                    st.metric("ได้รับการอนุมัติ (Approved)", f"{approved_count} คน", f"{(approved_count/len(df_result))*100:.1f}%")
                with col_s3:
                    default_count = (df_result['Prediction'] == 1).sum()
                    st.metric("ปฏิเสธ/ความเสี่ยงสูง (Default)", f"{default_count} คน", f"-{(default_count/len(df_result))*100:.1f}%", delta_color="inverse")
                
                st.dataframe(df_result)
                
                csv_download = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 ดาวน์โหลดตารางผลลัพธ์การอนุมัติ (CSV)",
                    data=csv_download,
                    file_name="loan_status_batch_predictions.csv",
                    mime="text/csv"
                )
                
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดระหว่างจัดระเบียบข้อมูล: {e}")

# ============================================================
# Footer (เพิ่มข้อมูลผู้พัฒนา)
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <p>🎓 พัฒนาเป็นส่วนหนึ่งของวิชา การโปรแกรมสำหรับการเรียนรู้ด้วยเครื่องด้วยภาษาไพทอน</p>
        <p>
            👨‍💻 <strong style='color: #333;'>ผู้พัฒนา:</strong> นายจตุรภัทร สถาปิตานนท์ 
            &nbsp;|&nbsp; 
            🆔 <strong style='color: #333;'>รหัสนักศึกษา:</strong> 664245024
        </p>
        <small>ระบบถูกพัฒนาและพร้อมรันแบบสมบูรณ์บนเซิร์ฟเวอร์ | อัปเดตโครงสร้างความปลอดภัยแล้ว</small>
    </div>
    """,
    unsafe_allow_html=True
)