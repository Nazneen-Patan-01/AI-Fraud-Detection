import streamlit as st
import pandas as pd
import joblib

# PAGE SETTINGS — must come first
st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# LOGIN — demo only; accounts are stored for this session
if "users" not in st.session_state:
    st.session_state.users = {}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 AI Fraud Detection")
    st.write("Create an account or sign in to continue.")

    tab1, tab2 = st.tabs(["Create Account", "Sign In"])

    with tab1:
        new_user = st.text_input("Create User ID", key="new_user")
        new_password = st.text_input(
            "Create Password", type="password", key="new_password"
        )
        confirm_password = st.text_input(
            "Confirm Password", type="password", key="confirm_password"
        )

        if st.button("Create Account"):
            if not new_user or not new_password:
                st.warning("Please fill all fields.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif new_user in st.session_state.users:
                st.error("User ID already exists.")
            else:
                st.session_state.users[new_user] = new_password
                st.success("Account created! Now sign in.")

    with tab2:
        user_id = st.text_input("User ID", key="login_user")
        password = st.text_input(
            "Password", type="password", key="login_password"
        )

        if st.button("Sign In"):
            if (
                user_id in st.session_state.users
                and st.session_state.users[user_id] == password
            ):
                st.session_state.logged_in = True
                st.session_state.current_user = user_id
                st.rerun()
            else:
                st.error("Invalid User ID or Password.")

    st.stop()

# LOAD MODEL AND SMALL SAMPLE DATASET
@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("sample_transactions.csv")

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Could not load the model or sample dataset: {e}")
    st.stop()

if "Class" not in df.columns:
    st.error("The sample dataset must contain a Class column.")
    st.stop()

# DESIGN
st.markdown("""
<style>
html, body {
    overflow-x: hidden;
}
.stApp {
    background: linear-gradient(120deg, #17145c, #090b1c, #5b174f);
}
.main .block-container {
    padding-top: 20px;
    padding-bottom: 20px;
}
h1 {
    background: linear-gradient(90deg, #ffffff, #a9eaff, #ff9edc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 42px !important;
}
.stButton button {
    background: linear-gradient(90deg, #00c6ff, #7b5cff, #ff3cac);
    color: white;
    border: none;
    border-radius: 10px;
    min-height: 42px;
    font-weight: bold;
    width: 100%;
}
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #17295c, #3b174f);
    padding: 12px;
    border-radius: 15px;
    border: 1px solid #ffffff25;
}
#MainMenu, footer, header {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.title("💳 AI Fraud Detection")
st.write(
    "Intelligent credit card transaction analysis powered by Machine Learning"
)

st.subheader("🏦 My Bank Dashboard")
st.caption(f"Welcome back, {st.session_state.current_user} 👋")

col1, col2, col3 = st.columns(3)
col1.metric("💰 Demo Account Balance", "$12,450.80")
col2.metric("💳 Card Status", "ACTIVE")
col3.metric("🛡️ Security", "PROTECTED")

st.caption("Demo dashboard — not connected to a real bank.")

# RECENT SAMPLE TRANSACTIONS
st.subheader("💳 Sample Transactions")

recent = df[["Time", "Amount", "Class"]].head(5).copy()
recent["Status"] = recent["Class"].map({
    0: "🟢 Normal",
    1: "🚨 Fraud"
})
recent = recent.drop(columns=["Class"])
st.dataframe(recent, width="stretch", hide_index=True)

# TRANSACTION ANALYSIS
st.subheader("🔍 Transaction Analysis")
st.caption("Choose a sample transaction for the Random Forest model to analyze.")

transaction_number = st.number_input(
    "Transaction number",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)

transaction = df.iloc[int(transaction_number)]

col1, col2, col3 = st.columns(3)
col1.metric("💰 Transaction Amount", f"${transaction['Amount']:,.2f}")
col2.metric("⏱ Transaction Time", f"{transaction['Time']:.0f}")
col3.metric("🛡️ AI Protection", "ACTIVE")

if st.button("🔎 Analyze Transaction"):
    try:
        input_data = transaction.drop("Class").to_frame().T
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        classes = list(model.classes_)
        fraud_probability = probabilities[classes.index(1)] * 100

        if prediction == 1:
            st.error("🚨 FRAUDULENT TRANSACTION DETECTED")
            st.metric("Fraud Probability", f"{fraud_probability:.2f}%")
            st.warning("⚠️ Risk Level: HIGH")
        else:
            st.success("🟢 TRANSACTION APPEARS NORMAL")
            st.metric("Fraud Probability", f"{fraud_probability:.2f}%")
            st.success("✅ Risk Level: LOW")

        st.caption(
            "Prediction is based on a sample dataset and is for demonstration only."
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption("Random Forest AI • Credit Card Fraud Detection • Portfolio Project")

if st.button("Log Out"):
    st.session_state.logged_in = False
    st.session_state.pop("current_user", None)
    st.rerun()
