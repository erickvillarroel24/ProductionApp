#Import Python libraries

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import plotly.express as px
from PIL import Image
import seaborn as sns

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
st.subheader("**Production**")

file = st.sidebar.file_uploader("Upload your csv file")

def data(dataframe):

    st.write(dataframe)

    st.subheader("***Production History***")
    fig1, ax1 = plt.subplots()
    ax1.bar(list(dataframe['date']), list(dataframe['oil_rate']))
    plt.title('Annual Oil Production')
    plt.xlabel('Years')
    plt.ylabel('Rate (BBL/D)')
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.bar(list(dataframe['date']), list(dataframe['water_rate']))
    plt.title('Annual Water Production')
    plt.xlabel('Years')
    plt.ylabel('Rate (BBL/D)')
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.bar(list(dataframe['date']), list(dataframe['total_rate']))
    plt.title('Annual Total Production')
    plt.xlabel('Years')
    plt.ylabel('Rate (BBL/D)')
    st.pyplot(fig3)


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
    df2= pd.DataFrame({'date':listend,'oil_rate':listoil,'water_rate':listwat,
                       'total_rate': list(np.array(listoil)+np.array(listwat))})
    data(df2)








