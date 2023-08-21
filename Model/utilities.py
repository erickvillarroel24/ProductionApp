import pandas as pd
import numpy as np
import openpyxl


archivo= "C:/Users/darwi/OneDrive/Documentos/Software en Ingenieria/data_clean_volve.xlsx"
exl=  openpyxl.load_workbook(archivo)
dataframe= exl.active
