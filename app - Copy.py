import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import pickle
from sklearn.ensemble import RandomForestClassifier ,VotingClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from xgboost import XGBClassifier
from sklearn.neighbors import LocalOutlierFactor

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

#set_png_as_page_bg('Data/bg.jpeg')

# Streamlit app
def main():
    st.title("Data Analysis App")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        Data = pd.read_csv(uploaded_file)
        tab1, tab2,tab3, tab4 = st.tabs(["Raw Data","Descriptive statistics", "EDA",'Model'])
        with tab1:
            st.subheader("Raw Data")
            st.dataframe(Data)
        with tab2:
            st.subheader("Descriptive statistics")
            st.write(Data.describe())
        with tab3:
            cat_columns = st.multiselect('Select Categoricalvaribales',Data.columns)
            Data[cat_columns] = Data[cat_columns].astype('category') 
            num_cols = Data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            st.subheader("Plots")
            option = st.selectbox(label='Plot Type',options=['Correlation plot','Frequency plot','Bar Plot','Box Plot','Scatter Plot'])
            if option == 'Correlation plot':
                plt.figure(figsize=(10, 7))
                sns.heatmap(Data[num_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
                st.pyplot(plt)
            elif option == 'Frequency plot':
                col = st.selectbox(label='Select column',options=num_cols)
                if col:
                    bins = st.slider('Select number of bins:', min_value=5, max_value=50, value=10)
                    plt.figure(figsize=(10, 6))
                    sns.histplot(Data[col], bins=bins, kde=False)
                    plt.title(f'Frequency Plot for {col}')
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    st.pyplot(plt)
                else:
                    st.warning('No category columns')
            elif option == 'Bar Plot':
                col1 = st.selectbox(label='Select X',options=cat_columns,index=1)
                col2 = st.selectbox(label='Select Hue',options=[None]+cat_columns)
                plt.figure(figsize=(10, 6))
                sns.countplot(data=Data, x=col1,hue=col2)
                plt.title(f'Bar Plot for {col1}')
                plt.ylabel('Count')
                st.pyplot(plt)            
            elif option == 'Box Plot':
                col1 = st.selectbox(label='Select X',options=cat_columns,index=1)
                col2 = st.selectbox(label='Select Y',options=num_cols)
                col3 = st.selectbox(label='Select Hue',options=[None]+cat_columns)
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=Data, x=col1, y=col2,hue=col3)
                plt.title(f'Box Plot of {col2} by {col1}')
                plt.ylabel(col2)
                plt.xlabel(col1)
                st.pyplot(plt)
            elif option == 'Scatter Plot':
                available_columns = Data.select_dtypes(include=['float64', 'int64']).columns.tolist()
                x_axis = st.selectbox('Select the X-axis:', available_columns)
                y_axis = st.selectbox('Select the Y-axis:', available_columns, index=1 if len(available_columns) > 1 else 0)
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=Data, x=x_axis, y=y_axis)
                plt.title(f'Scatter Plot of {x_axis} vs {y_axis}')
                st.pyplot(plt) 
        with tab4:

            result = {
                    'Label': ['Fatal', 'Serious', 'Slight', 'Accuracy', 'Macro Avg', 'Weighted Avg'],
                    'Precision': [0.05, 0.28, 0.80, None, 0.38, 0.68],
                    'Recall': [0.10, 0.29, 0.78, None, 0.39, 0.67],
                    'F1-score': [0.07, 0.29, 0.79, 0.6652, 0.38, 0.67],
                    'Support': [673, 10226, 37687,48586, 48586, 48586]
                }

            df = pd.DataFrame(result)
            st.subheader("Ensemble model summary")
            st.dataframe(df)

            with open('./Notebook/xgb_model.pkl', 'rb') as f:
                ensemble_model = pickle.load(f,fix_imports=True)
            
            sample = {
                "Time": 0,
                "V1": 0,
                "V2": 0.0,
                "V3": 0,
                "V4": 0.0,
                "V5": 0.0,
                "V6": 0.0,
                "V7": 0.0,
                "V8": 0.0,
                "V9": 0.0,
                "V10": 0.0,
                "V11": 0.0,
                "V12": 0.0,
                "V13": 0.0,
                "V14": 0.0,
                "V15": 0.0,
                "V16": 0.0,
                "V17": 0.0,
                "V18": 0.0,
                "V19": 0.0,
                "V20": 0.0,
                "V21": 0.0,
                "V22": 0.0,
                "V23": 0.0,
                "V24": 0.0,
                "V25": 0.0,
                "V26": 0.0,
                "V27": 0.0,
                "V28": 0.0,
                "Amount": 0.0
            }
            input_df = pd.DataFrame(sample, index=[0])

            st.subheader("Data Input")

            with st.form("input_form"):
                col1, col2, col3,col4,col5,col6 = st.columns(6)
                with col1:
                    Time = st.number_input('Time', value=1, min_value=0)
                    V1 = st.number_input('V1', value=0.4)
                    V2 = st.number_input('V2', value=1)
                    V3 = st.number_input('V3', value=0.4)
                    V4 = st.number_input('V4', value=1)
                    V5 = st.number_input('V5', value=0.4)

                with col2:
                    V6 = st.number_input('V6', value=0.4)
                    V7 = st.number_input('V7', value=1)
                    V8 = st.number_input('V8', value=0.4)
                    V9 = st.number_input('V9', value=1)
                    V10 = st.number_input('V10', value=0.4)
                    V11 = st.number_input('V11', value=0.4)

                with col3:
                    V12 = st.number_input('V12', value=1)
                    V13 = st.number_input('V13', value=0.4)
                    V14 = st.number_input('V14', value=1)
                    V15 = st.number_input('V15', value=0.4)
                    V16 = st.number_input('V16', value=0.4)
                    V17 = st.number_input('V17', value=1)

                with col4:
                    V18 = st.number_input('V18', value=1)
                    V19 = st.number_input('V19', value=0.4)
                    V20 = st.number_input('V20', value=0.4)
                    V21 = st.number_input('V21', value=1)
                    V22 = st.number_input('V22', value=0.4)
                    V23 = st.number_input('V23', value=1)

                with col5:
                    
                    V24 = st.number_input('V24', value=0.4)
                    V25 = st.number_input('V25', value=0.4)
                    V26 = st.number_input('V26', value=1)
                    V27 = st.number_input('V27', value=0.4)
                    V28 = st.number_input('V28', value=1)
                    Amount = st.number_input('Amount', value=0)                  

                submitted = st.form_submit_button("Submit")


            if submitted:
                input_df['Time'] = Time
                input_df['V1'] = V1
                input_df['V2'] = V2
                input_df['V3'] = V3
                input_df['V4'] = V4
                input_df['V5'] = V5
                input_df['V6'] = V6
                input_df['V7'] = V7
                input_df['V8'] = V8
                input_df['V9'] = V9
                input_df['V10'] = V10
                input_df['V11'] = V11
                input_df['V12'] = V12
                input_df['V13'] = V13
                input_df['V14'] = V14
                input_df['V15'] = V15
                input_df['V16'] = V16
                input_df['V17'] = V17
                input_df['V18'] = V18
                input_df['V19'] = V19
                input_df['V20'] = V20
                input_df['V21'] = V21
                input_df['V22'] = V22
                input_df['V23'] = V23
                input_df['V24'] = V24
                input_df['V25'] = V25
                input_df['V26'] = V26
                input_df['V27'] = V27
                input_df['V28'] = V28
                input_df['Amount'] = Amount
                
                ensemble_prediction = ensemble_model.predict(input_df)
                ans_maps = {0:'Not Fraud',1:'Fraud'}
                ans = 'This transaction might be '+str(ans_maps[ensemble_prediction[0]])
                st.subheader(f':orange[{ans}]',divider='rainbow')

if __name__ == "__main__":
    main()
