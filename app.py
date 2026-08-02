import streamlit as st
import pandas as pd
import joblib as jb


# -----------------------------
# Load ML Model
# -----------------------------
model = jb.load("KNN_heart.pkl")
scaler = jb.load("scaler.pkl")
expected_columns = jb.load("columns.pkl")


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>

    .main {
        background-color: #f7f9fc;
    }

    .header-card {
        background: linear-gradient(135deg,#0f766e,#14b8a6);
        padding: 25px;
        border-radius: 18px;
        color:white;
        margin-bottom:25px;
    }


    .medical-card {
        background:white;
        padding:25px;
        border-radius:18px;
        box-shadow:0px 5px 20px rgba(0,0,0,0.08);
        margin-top:20px;
    }


    .risk-high {
        background:#fee2e2;
        border-left:8px solid #dc2626;
        padding:25px;
        border-radius:15px;
        color:#991b1b;
        font-size:22px;
        font-weight:600;
    }


    .risk-low {
        background:#dcfce7;
        border-left:8px solid #16a34a;
        padding:25px;
        border-radius:15px;
        color:#166534;
        font-size:22px;
        font-weight:600;
    }


    div.stButton > button {
        width:100%;
        background:#0f766e;
        color:white;
        height:50px;
        border-radius:12px;
        font-size:18px;
        font-weight:bold;
    }


    </style>

    """,
    unsafe_allow_html=True
)



# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2966/2966486.png",
        width=100
    )

    st.title("HeartCare AI")

    st.write(
        """
        🩺 AI-powered heart disease
        prediction system.

        Built using:
        - Machine Learning
        - KNN Classification
        - Medical Feature Analysis
        """
    )


    st.divider()

    st.caption(
        "Developed by Ayush Jaiswal"
    )



# -----------------------------
# Header
# -----------------------------

st.markdown(
    """
    <div class="header-card">

    <h1>❤️ Heart Disease Risk Assessment</h1>

    <p>
    Enter patient information below to estimate
    cardiovascular disease risk using Machine Learning.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)



# -----------------------------
# Prediction Form
# -----------------------------

with st.form("medical_form"):


    # Patient Profile

    with st.container():

        st.subheader("👤 Patient Profile")


        col1,col2,col3 = st.columns(3)


        with col1:
            age = st.slider(
                "Age",
                18,
                100,
                40
            )


        with col2:

            try:
                sex = st.pills(
                    "Gender",
                    ["M","F"],
                    default="M"
                )
            except:
                sex = st.selectbox(
                    "Gender",
                    ["M","F"]
                )


        with col3:

            chest_pain = st.selectbox(
                "Chest Pain Type",
                [
                    "ATA",
                    "NAP",
                    "TA",
                    "ASY"
                ]
            )



    st.divider()


    # Clinical Tests

    st.subheader("🧪 Clinical Measurements")


    col1,col2,col3 = st.columns(3)


    with col1:

        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            80,
            200,
            120
        )


    with col2:

        cholesterol = st.number_input(
            "Cholesterol (mg/dL)",
            100,
            600,
            200
        )


    with col3:

        fasting_bs = st.selectbox(
            "Fasting Blood Sugar >120",
            [0,1]
        )



    col1,col2,col3 = st.columns(3)


    with col1:

        max_hr = st.slider(
            "Maximum Heart Rate",
            50,
            220,
            150
        )


    with col2:

        old_peak = st.slider(
            "OldPeak (ST Depression)",
            0.0,
            6.0,
            1.0
        )


    with col3:

        resting_ecg = st.selectbox(
            "Resting ECG",
            [
                "Normal",
                "ST",
                "LHV"
            ]
        )


    st.divider()


    # ECG Section

    st.subheader("❤️ Electrocardiogram Analysis")


    col1,col2 = st.columns(2)


    with col1:

        try:

            exercise_angina = st.pills(
                "Exercise Induced Angina",
                [
                    "Y",
                    "N"
                ],
                default="N"
            )

        except:

            exercise_angina = st.selectbox(
                "Exercise Induced Angina",
                [
                    "Y",
                    "N"
                ]
            )


    with col2:

        st_slope = st.selectbox(
            "ST Slope",
            [
                "Up",
                "Flat",
                "Down"
            ]
        )


    st.divider()


    predict_button = st.form_submit_button(
        "🔍 Analyze Heart Risk"
    )




# -----------------------------
# Prediction Logic
# -----------------------------

if predict_button:


    with st.spinner(
        "Analyzing patient data using AI model..."
    ):


        raw_input = {

            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': old_peak,

            'Sex_' + sex:1,

            'ChestPainType_' + chest_pain:1,

            'RestingECG_' + resting_ecg:1,

            'ExerciseAngina_' + exercise_angina:1,

            'ST_Slope_' + st_slope:1
        }


        input_df = pd.DataFrame(
            [raw_input]
        )


        for col in expected_columns:

            if col not in input_df.columns:

                input_df[col]=0



        input_df = input_df[
            expected_columns
        ]


        scaled_input = scaler.transform(
            input_df
        )


        prediction = model.predict(
            scaled_input
        )[0]



    # -----------------------------
    # Result Card
    # -----------------------------


    st.subheader(
        "📊 Prediction Result"
    )


    if prediction == 1:


        st.markdown(
            """
            <div class="risk-high">

            ⚠️ High Risk of Heart Disease

            <br><br>

            Recommendation:
            Consult a healthcare professional
            for further medical evaluation.

            </div>

            """,
            unsafe_allow_html=True
        )


    else:
        st.markdown(
            """
            <div class="risk-low">
            
            ✅ Low Risk of Heart Disease

            <br><br>

            Current indicators suggest lower
            cardiovascular risk.

            </div>

            """,
            unsafe_allow_html=True
        )