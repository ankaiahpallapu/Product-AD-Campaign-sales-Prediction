
import streamlit as st
import pandas as pd
import numpy as np
import joblib, pickle

model   = joblib.load('model_xgb.pkl')
columns = pickle.load(open('columns.pkl', 'rb'))
scaler  = joblib.load('scaler.pkl')
st.title('Campaign Orders Prediction')
st.markdown('Fill in campaign details to predict **orders**.')

with st.form('pred_form'):
    c1, c2 = st.columns(2)
    with c1:
        email_rate     = st.number_input('Email Rate (0-1.00)',    0.08,0.84,0.30)
        campaign_fee   = st.number_input('Campaign Fee',        0.0, value=5000.0)
        price          = st.number_input('Price',               0.0, value=999.0)
        discount_rate  = st.number_input('Discount Rate (0-1.00)',0.49, 0.98,0.60)
    with c2:
        campaign_type  = st.selectbox('Campaign Type',   ['A','B','C','D','E','F'])
        campaign_level = st.selectbox('Campaign Level',  ['1','2'])
        product_level  = st.selectbox('Product Level',   ['low','mid','high'])
        resource_amount= st.selectbox('Resource Amount', ['small','medium','large'])
    submitted = st.form_submit_button('Predict Orders')

if submitted:
    raw = {'email_rate'     : email_rate,
           'campaign_fee'   : np.log1p(campaign_fee),
           'price'          : np.log1p(price),
           'discount_rate'  : discount_rate,
           'campaign_type'  : campaign_type,
           'campaign_level' : campaign_level,
           'product_level'  : product_level,
           'resource_amount': resource_amount}
    inp = pd.DataFrame([raw])
    inp = pd.get_dummies(inp)
    inp = inp.reindex(columns=columns, fill_value=0)
    pred = model.predict(inp.values)[0]
    st.success(f'Predicted Orders: {pred:.0f}')
