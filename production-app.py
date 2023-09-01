#Import Python libraries

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import plotly.express as px
from PIL import Image
from collections import namedtuple
import seaborn as sns
from Model.utilities import qo_darcy, j, Qb, aof, qo_vogel, qo_darcy, qo_standing, qo, \
    IPR_Curve, IPR_curve_methods, pwf_darcy, pwf_vogel, f_darcy, sg_oil, sg_avg, gradient_avg

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

#Create Menu
st.sidebar.title("⬇ Navigation")
file = st.sidebar.file_uploader("Upload your csv file")

# Add sections of the app
with st.sidebar:
    options = option_menu(
        menu_title="Menu",
        options=["Home", "Data", "3D Plots", "calculations", "Analisis nodal"],
        icons=["house", "tv-fill", "box", "calculator"],)

#Qo(bpd) @ all conditions


def plots(dataframe):

    st.write(dataframe)

    st.subheader("***Production History***")
    fig1, ax1 = plt.subplots()
    ax1.plot(list(dataframe['date']), list(dataframe['oil_rate']))
    plt.title('Annual Oil Production')
    plt.xlabel('Years')
    plt.ylabel('Rate (BBL/D)')
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(list(dataframe['date']), list(dataframe['water_rate']))
    plt.title('Annual Water Production')
    plt.xlabel('Years')
    plt.ylabel('Rate (BBL/D)')
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    aq = dataframe[['date', 'oil_rate', 'water_rate']]
    ax3.plot(list(aq['date']), list(aq['water_rate']))
    ax3.plot(list(aq['date']), list(aq['oil_rate']))
    plt.title('Annual Total Production')
    plt.xlabel('Years')
    plt.ylabel('Rate (BBL/D)')
    st.pyplot(fig3)




Data = namedtuple("Input", "q_test pwf_test pr pwf pf ef ef2")
Output = namedtuple("Output", "qo")


if file:
    df = pd.read_excel(file, index_col=0)
    df1 = pd.DataFrame(df)
    if options == "Data":
        df1
    elif options == "3D Plots":
        plots(df)
if options == "calculations":
    if st.checkbox("Potential resevoir"):
        Data = namedtuple("Input", "q_test pwf_test pr pwf pb ef ef2")
        st.subheader("**Enter input values**")
        q_test = st.number_input("Enter q_test value: ")
        pwf_test = st.number_input("Enter pw_test value: ")
        pr = st.number_input("Enter pr value: ")
        pwf = st.number_input("Enter pwf value")
        pb = st.number_input("Enter pb value")
        ef = st.number_input("Enter ef value")
        ef2 = st.number_input("Enter ef2 value")
        st.subheader("**Show results**")
        qo = qo(q_test, pwf_test, pr, pwf, pb, ef=1, ef2=None)
        st.success(f"{'Qo'} -> {qo:.3f} scf/Dia ")
        Qmax = aof(q_test, pwf_test, pr, pb, ef=1, ef2=None)
        st.success(f"{'Caudal maximo'} -> {Qmax:.3f} scf/Dia ")
        idp = j(q_test, pwf_test, pr, pb, ef=1, ef2=None)
        st.success(f"{'Indice de productividad'} -> {idp:.3f}  ")

    elif st.checkbox("IPR Curve"):
        st.subheader("**Select method**")
        method = st.selectbox("Method", ("Darcy", "Vogel", "IPR Compuesto", "Standing"))
        Data = namedtuple("Input", "q_test pwf_test pr pwf pb ef ef2")
        st.subheader("**Enter input values**")
        q_test = st.number_input("Enter q_test value: ")
        pwf_test = st.number_input("Enter pw_test value: ")
        pr = st.number_input("Enter pr value: ")
        pwf = st.number_input("Enter pwf value")
        pb = st.number_input("Enter pb value")
        ef = st.number_input("Enter ef value")
        ef2 = st.number_input("Enter ef2 value")
        q = IPR_curve_methods(q_test, pwf_test, pr, pwf, pb, method, ef=1, ef2=None)

if options == "Analisis nodal":
    Data = namedtuple("Input", "q_test pwf_test q pr pb sg_h2o API Q ID c wc")
    st.subheader("**Enter input values**")
    q_test = st.number_input("Enter q_test Value: ")
    pwf_test = st.number_input("Enter pwf test value: ")
    sg_h2o = st.number_input("Enter sg_h20 value: ")
    API = st.number_input("Enter API value")
    Q = st.number_input("Enter Q value")
    ID = st.number_input("Enter ID value")
    q = st.number_input("Enter q value")
    pr = st.number_input("Enter pr value")
    c = st.number_input("Enter c value")
    pb = st.number_input("Enter pb value")
    wc = st.number_input("Enter wc value")
    st.subheader("**Show results**")
    if pr > pb:
        pw_darcy = pwf_darcy(q_test, pwf_test, q, pr, pb)
        st.success(f"{'Pwf Darcy'} -> {pw_darcy:.3f} psi ")
    else:
        pw_vogel = pwf_vogel(q_test, pwf_test, q, pr, pb)
        st.success(f"{'Pwf Vogel'} -> {pw_vogel:.3f} psi ")

    fric= f_darcy(Q, ID, C=120)
    st.success(f"{'Friccion'} -> {fric:.3f} psi ")
    sg_oil = sg_oil(API)
    st.success(f"{'Sg Oil'} -> {sg_oil:.3f} psi ")
    sg_f = sg_avg(API, wc, sg_h2o)
    st.success(f"{'Sg fluids'} -> {sg_f:.3f} psi ")
    gra = gradient_avg(API, wc, sg_h2o)
    st.success(f"{'Average Gradient'} -> {gra:.3f} psi ")






