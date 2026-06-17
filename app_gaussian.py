import streamlit as st
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

st.title("Prediksi Coimbra Breast Cancer dataset dengan Gaussian Naive Bayes")

data=pd.read_excel('data_ujian_akhir.xlsx', engine='openpyxl')
st.write("Data Training",data.head(50))
st.write("jumlah data tiap kelas: ")
st.write(data.iloc[:,-1].value_counts())
hu=data.Age==39
haa=data.Classification==1
st.write(data[hu && haa])

x=data.iloc[:,[0,1,2,3]].values
y=data.iloc[:,-1].values

model=GaussianNB()
model.fit(x,y)

x1= st.number_input("Masukkan Nilai Age",format="%.3f")
x2= st.number_input("Masukkan Nilai BMI",format="%.3f")
x3= st.number_input("Masukkan Nilai Resistin",format="%.3f")
x4= st.number_input("Masukkan Nilai MCP.1",format="%.3f")

x_input=np.array([[x1,x2,x3,x4]])
prediksi=st.button("Hasil Prediksi")

if prediksi:
    y_predik=model.predict(x_input)
    if (y_predik==1):
        st.write("Hasil Prediksi: Normal/Sehat")
    else:
        st.write("Hasil Prediksi: Sakit")
else:
    st.write("Hasil Prediksi ",0)
