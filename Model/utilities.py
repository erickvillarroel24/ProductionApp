 Import Python libraries

import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import plotly.express as px
from PIL import Image

# Insert an icon
icon = Image.open("Resources/image1.avif")

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
st.subheader("**Drilling Fundamentals**")
