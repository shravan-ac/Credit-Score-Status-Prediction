import streamlit as st
import pandas as pd
import numpy as np
import pickle
#Load saved files
with open("model.pkl","rb") as f:
    model=pickle.load(f)
with open("le_encoder.pkl","rb") as f:
    label_encoders=pickle.load(f)
with open("oh_encoder.pkl","rb") as f:
    onehot_encoder=pickle.load(f)
with open("feature_names.pkl","rb") as f:
    feature_names=pickle.load(f)
edu=pd.read_csv("Credit Score Classification Dataset.csv")['Education'].unique().tolist()
#streamlit ui
st.title("Credit Score Predition")

age=st.number_input("Age",min_value=18,max_value=100)
gender=st.selectbox("Gender",["Female","Male"])
income=st.number_input("Income",min_value=20000,max_value=200000)
education=st.selectbox("Education",edu)
maritial=st.selectbox("Maritial Status",["Married","Single"])
children=st.number_input("Number of children",min_value=0,max_value=10)
home_ownership=st.selectbox("Home Ownership",["Owned","Rented"])

input_df=pd.DataFrame({"Age":[age],"Gender":[gender],"Income":[income],"Education":[education],"Marital Status":[maritial],"Number of Children":[children],"Home Ownership":[home_ownership]})

for i,le in label_encoders.items():
    input_df[i]=le.transform(input_df[i])
onehot_cols=['Education']
onehot_encoded=onehot_encoder.transform(input_df[onehot_cols])
onehot_encoded_df=pd.DataFrame(onehot_encoded,columns=onehot_encoder.get_feature_names_out(onehot_cols))
input_df=input_df.drop(onehot_cols,axis=1).join(onehot_encoded_df)
input_df=input_df[feature_names]

if st.button("Predict"):
    target_mapping = {0: "Low", 1: "Average", 2: "High"} 
    pred=model.predict(input_df)
    pred_label = target_mapping[pred[0]]
    st.success(f"Prediction: {pred_label}")