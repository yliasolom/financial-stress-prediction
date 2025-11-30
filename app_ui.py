import streamlit as st
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Financial Stress Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, clean design
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Clean header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Result card */
    .result-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .stress-low {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        color: white;
    }
    
    .stress-moderate {
        background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
        color: white;
    }
    
    .stress-high {
        background: linear-gradient(135deg, #ef4444 0%, #f87171 100%);
        color: white;
    }
    
    .result-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    
    .result-value {
        font-size: 3rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Info boxes */
    .info-box {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    /* Metric cards */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    
    /* Sidebar improvements */
    [data-testid="stSidebar"] {
        background: #f8fafc;
    }
    
    /* Input improvements */
    .stSlider > div > div {
        background: white;
    }
    
    /* Button improvements */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.2s;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'auto_predict' not in st.session_state:
    st.session_state.auto_predict = False

# API URL
API_URL = "http://localhost:8000"

# Header
st.markdown('<h1 class="main-header">💰 Financial Stress Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered assessment for gig economy workers</p>', unsafe_allow_html=True)

# Sidebar with organized inputs
with st.sidebar:
    st.markdown("### 📝 Enter Data")
    
    # Auto-predict toggle
    auto_predict = st.checkbox("🔄 Auto Prediction", value=st.session_state.auto_predict)
    st.session_state.auto_predict = auto_predict
    
    if auto_predict:
        st.info("💡 Results update automatically when data changes")
    
    st.markdown("---")
    
    # Demographics
    st.markdown('<div class="section-header">👤 Demographics</div>', unsafe_allow_html=True)
    worker_age = st.slider("Age", 18, 70, 30, help="Worker age")
    job_sector = st.selectbox("Job Sector", [
        "Driver", "Writer", "Engineer", "Doctor", "Teacher", 
        "Designer", "Developer", "Consultant", "Freelancer", "Other"
    ], help="Type of gig work")
    
    st.markdown("---")
    
    # Income & Savings
    st.markdown('<div class="section-header">💵 Income & Savings</div>', unsafe_allow_html=True)
    monthly_income = st.number_input(
        "Monthly Gig Income ($)", 
        0, 50000, 3000, 
        step=100,
        help="Monthly income from gig work"
    )
    annual_income = st.number_input(
        "Estimated Annual Income ($)", 
        0, 500000, 36000, 
        step=1000,
        help="Estimated annual income"
    )
    num_savings = st.slider("Savings Accounts", 0, 10, 1)
    monthly_investments = st.number_input(
        "Monthly Investments ($)", 
        0, 10000, 100, 
        step=50
    )
    
    st.markdown("---")
    
    # Credit Information
    st.markdown('<div class="section-header">💳 Credit Information</div>', unsafe_allow_html=True)
    num_credit_cards = st.slider("Credit Cards", 0, 15, 2)
    credit_utilization = st.slider("Credit Utilization (%)", 0, 100, 30, help="Percentage of credit limit used")
    avg_credit_interest = st.slider("Avg Credit Interest (%)", 0, 40, 15)
    num_active_loans = st.slider("Active Loans", 0, 20, 1)
    
    st.markdown("---")
    
    # Payment History
    st.markdown('<div class="section-header">📅 Payment History</div>', unsafe_allow_html=True)
    missed_payments = st.slider("Missed Payment Events", 0, 50, 5, help="Number of missed or late payments")
    avg_loan_delay = st.slider("Avg Loan Delay (days)", 0, 180, 10)
    recent_credit_checks = st.slider("Recent Credit Checks", 0, 20, 2, help="Credit inquiries in past 3 months")
    
    st.markdown("---")
    
    # Financial Status
    st.markdown('<div class="section-header">📊 Financial Status</div>', unsafe_allow_html=True)
    total_liability = st.number_input(
        "Total Liability ($)", 
        0, 100000, 2000, 
        step=100,
        help="Total amount of debt"
    )
    end_of_month_balance = st.number_input(
        "End of Month Balance ($)", 
        -10000, 50000, 500, 
        step=100,
        help="Account balance at month end"
    )
    
    st.markdown("---")
    
    # Predict button
    predict_button = st.button("🔮 Get Prediction", type="primary", use_container_width=True)

# Main content area
col1, col2 = st.columns([1.5, 1])

with col1:
    # Key metrics dashboard
    st.markdown("### 📊 Key Metrics")
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.metric(
            "Monthly Income",
            f"${monthly_income:,}",
            help="Monthly income from gig work"
        )
    
    with metrics_col2:
        utilization_color = "normal"
        if credit_utilization > 70:
            utilization_color = "inverse"
        elif credit_utilization > 50:
            utilization_color = "off"
        st.metric(
            "Credit Utilization",
            f"{credit_utilization}%",
            delta=None if credit_utilization <= 30 else "High" if credit_utilization > 70 else "Medium",
            delta_color=utilization_color
        )
    
    with metrics_col3:
        missed_color = "normal"
        if missed_payments > 10:
            missed_color = "inverse"
        elif missed_payments > 5:
            missed_color = "off"
        st.metric(
            "Missed Payments",
            missed_payments,
            delta=None if missed_payments <= 5 else "Many" if missed_payments > 10 else "Some",
            delta_color=missed_color
        )
    
    with metrics_col4:
        st.metric(
            "Total Liability",
            f"${total_liability:,}"
        )
    
    st.markdown("---")
    
    # Financial health indicators
    st.markdown("### 📈 Financial Health Indicators")
    
    health_col1, health_col2, health_col3 = st.columns(3)
    
    with health_col1:
        # Income to liability ratio
        if total_liability > 0:
            income_liability_ratio = (monthly_income * 12) / total_liability
            st.metric("Income/Debt", f"{income_liability_ratio:.1f}x")
        else:
            st.metric("Income/Debt", "N/A")
    
    with health_col2:
        # Savings indicator
        savings_score = "Low" if num_savings == 0 else "Medium" if num_savings <= 2 else "High"
        st.metric("Savings Level", savings_score)
    
    with health_col3:
        # Payment reliability
        if missed_payments == 0:
            reliability = "Excellent"
        elif missed_payments <= 3:
            reliability = "Good"
        elif missed_payments <= 10:
            reliability = "Fair"
        else:
            reliability = "Poor"
        st.metric("Payment Reliability", reliability)

with col2:
    st.markdown("### 🎯 Prediction Result")
    
    # Check if we should predict
    should_predict = predict_button or (auto_predict and st.session_state.prediction_result is None)
    
    if predict_button or (auto_predict and st.session_state.prediction_result is None):
        with st.spinner("🔄 Analyzing financial data..."):
            payload = {
                "features": {
                    "worker_age": float(worker_age),
                    "job_sector": job_sector,
                    "estimated_annual_income": float(annual_income),
                    "monthly_gig_income": float(monthly_income),
                    "num_savings_accounts": num_savings,
                    "num_credit_cards": num_credit_cards,
                    "avg_credit_interest": float(avg_credit_interest),
                    "num_active_loans": num_active_loans,
                    "avg_loan_delay_days": float(avg_loan_delay),
                    "missed_payment_events": missed_payments,
                    "recent_credit_checks": recent_credit_checks,
                    "current_total_liability": float(total_liability),
                    "credit_utilization_rate": float(credit_utilization),
                    "monthly_investments": float(monthly_investments),
                    "end_of_month_balance": float(end_of_month_balance)
                }
            }
            
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.prediction_result = result
                    stress_level = result.get("predicted_stress_level", "Unknown")
                    probabilities = result.get("prediction_probabilities", {})
                    
                    # Display stress level card
                    if stress_level == "Low":
                        color_class = "stress-low"
                        emoji = "✅"
                        title = "Low Stress Level"
                        description = "Financial status is stable"
                    elif stress_level == "Moderate":
                        color_class = "stress-moderate"
                        emoji = "⚠️"
                        title = "Moderate Stress Level"
                        description = "Financial attention required"
                    else:
                        color_class = "stress-high"
                        emoji = "🚨"
                        title = "High Stress Level"
                        description = "Urgent action needed"
                    
                    st.markdown(f"""
                    <div class="result-card {color_class}">
                        <div class="result-title">{title}</div>
                        <div class="result-value">{emoji} {stress_level}</div>
                        <div style="opacity: 0.9; font-size: 0.9rem; margin-top: 0.5rem;">{description}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability visualization
                    st.markdown("#### 📊 Probabilities")
                    
                    # Create a more informative chart
                    fig = make_subplots(
                        rows=1, cols=2,
                        specs=[[{"type": "bar"}, {"type": "indicator"}]],
                        subplot_titles=("Probability Distribution", "Confidence Level")
                    )
                    
                    # Bar chart
                    colors_map = {'Low': '#10b981', 'Moderate': '#f59e0b', 'High': '#ef4444'}
                    bar_colors = [colors_map.get(k, '#64748b') for k in probabilities.keys()]
                    
                    fig.add_trace(
                        go.Bar(
                            x=list(probabilities.keys()),
                            y=[v * 100 for v in probabilities.values()],
                            marker_color=bar_colors,
                            text=[f"{v*100:.1f}%" for v in probabilities.values()],
                            textposition='auto',
                            name="Probability",
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    
                    # Gauge chart for confidence
                    max_prob = max(probabilities.values())
                    fig.add_trace(
                        go.Indicator(
                            mode="gauge+number+delta",
                            value=max_prob * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Confidence"},
                            delta={'reference': 50},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': colors_map.get(stress_level, '#64748b')},
                                'steps': [
                                    {'range': [0, 33], 'color': "lightgray"},
                                    {'range': [33, 66], 'color': "gray"},
                                    {'range': [66, 100], 'color': "darkgray"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ),
                        row=1, col=2
                    )
                    
                    fig.update_layout(
                        height=300,
                        showlegend=False,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed probabilities
                    with st.expander("📋 Detailed Probabilities"):
                        prob_col1, prob_col2, prob_col3 = st.columns(3)
                        for i, (level, prob) in enumerate(probabilities.items()):
                            with [prob_col1, prob_col2, prob_col3][i]:
                                st.metric(
                                    level,
                                    f"{prob*100:.1f}%",
                                    delta=None
                                )
                    
                    # Recommendations based on stress level
                    st.markdown("---")
                    st.markdown("#### 💡 Recommendations")
                    
                    if stress_level == "Low":
                        st.success("""
                        ✅ **Excellent financial status!**
                        - Continue maintaining current level
                        - Consider increasing investments
                        - Maintain emergency fund
                        """)
                    elif stress_level == "Moderate":
                        st.warning("""
                        ⚠️ **Attention required:**
                        - Reduce credit utilization to 30%
                        - Create debt repayment plan
                        - Increase savings
                        - Consider loan consolidation
                        """)
                    else:
                        st.error("""
                        🚨 **Critical situation:**
                        - Contact financial advisor immediately
                        - Prioritize high-interest debt repayment
                        - Consider debt restructuring
                        - Create emergency action plan
                        """)
                
                elif response.status_code == 503:
                    st.error("⚠️ Model not loaded. Make sure API server is running and model is available.")
                else:
                    st.error(f"❌ Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("""
                ❌ **Failed to connect to API**
                
                Make sure:
                1. API server is running on http://localhost:8000
                2. Run command: `uvicorn app.main:app --reload`
                """)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    elif st.session_state.prediction_result:
        # Show cached result
        result = st.session_state.prediction_result
        stress_level = result.get("predicted_stress_level", "Unknown")
        probabilities = result.get("prediction_probabilities", {})
        
        if stress_level == "Low":
            color_class = "stress-low"
            emoji = "✅"
            title = "Low Stress Level"
        elif stress_level == "Moderate":
            color_class = "stress-moderate"
            emoji = "⚠️"
            title = "Moderate Stress Level"
        else:
            color_class = "stress-high"
            emoji = "🚨"
            title = "High Stress Level"
        
        st.markdown(f"""
        <div class="result-card {color_class}">
            <div class="result-title">{title}</div>
            <div class="result-value">{emoji} {stress_level}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 Click button to update prediction")
    else:
        st.info("""
        👆 **Start Analysis**
        
        Fill in the data in the sidebar and click "Get Prediction" to analyze financial stress.
        
        Or enable "Auto Prediction" for instant results.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 1rem 0;">
    Built with Streamlit • Powered by RandomForest ML Model • Financial Stress Prediction API
</div>
""", unsafe_allow_html=True)
