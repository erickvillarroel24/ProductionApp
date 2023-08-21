#Import Python libraries

import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import plotly.express as px
from PIL import Image

# Insert an icon
icon = Image.open("Resources/image1.jpg")

# state the design of the app
st.set_page_config(page_title="DE App", page_icon=icon)

# Insert css codes to improve the design of the app
st.markdown(
    """
<style>
h1 {text-align: center;
}
body {background-color: #DCE3D5;
      width: 1400px;
      margin: 15px auto;
}
footer {
  display: none;
}
</style>""",
    unsafe_allow_html=True,
)

# Insert title of the app
st.title("Production Engineering App ®")

st.write("---")

# Add information of the app
st.markdown(""" This app is used to see production history,IPR graphs curve, nodal 
analysis for single phase flow and reservoir potential calculations.

***Python:*** Pandas, NumPy, Streamlit, PIL, Plotly.
""")

# Add additional information
expander = st.expander("Information")
expander.write("This is an open-source web app fully programmed in Python for calculating"
               " production parameters.")

# Insert image
image = Image.open("Resources/image2.jpg")
st.image(image, width=100, use_column_width=True)

# Insert subheader
st.subheader("**Production Fundamentals**")

file = st.sidebar.file_uploader("Upload your csv file")

def data(dataframe):
    st.subheader("**View dataframe**")
    st.write(dataframe.head())
    st.subheader("**Statistical summary**")
    st.write(dataframe.describe())

if file:
    df = pd.read_excel(file)
    df1 = pd.DataFrame(df)
    #data(df1)
    dcol=df1[["date","oil_rate","water_rate"]]
    coldate= list(dcol["date"])
    coloil= list(dcol["oil_rate"])
    colwat= list(dcol["water_rate"])
    listdate=[]
    cant= len(coldate)
    for i in range(cant):
        date1 = coldate[i].strftime('%Y/%m/%d')
        date2 = date1[0:4]
        listdate.append(date2)
    listdate_int = list(map(int, listdate))
    date_uni= list(set(listdate_int))
    date_uni.sort()
    bucl= len(date_uni)
    listend=[]
    listoil=[]
    listwat=[]
    for i in range(bucl):
        listoil1 = []
        listwat2 = []
        for x in range(cant):
            if int(listdate[x]) == date_uni[i]:
                listoil1.append(coloil[x])
                listwat2.append(colwat[x])
        listend.append(date_uni[i])
        listoil.append(np.mean(listoil1))
        listwat.append(np.mean(listwat2))
    df2= pd.DataFrame({'date':listend,'oil_rate':listoil,'oil_water':listwat})
    data(df2)
    #







