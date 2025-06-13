# Graduation Project App - Multiple Disease Prediction 

# ---- Imports ----
import os
import pickle
import requests
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie

# Function to load a local lottie file
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# ---- Page Config ----
st.set_page_config(
    page_title='Graduation Project',
    page_icon='🏥',
    #layout='wide'
)

# ---- Hide Streamlit Default Menu ----
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# ---- Function to Read URL for Lottie Animations ----
def get_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def load_lottiefile(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# ---- Navigation ----
selected = option_menu(
    menu_title=None,
    options=["Home", "Multiple Disease Prediction"],
    icons=["house-door-fill", "patch-check-fill"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

# ---- Home Section ----
if selected == "Home":
    st.markdown('<h2 style="color:#ff7f7f;">Comprehensive Illness Diagnosis with AI</h2>', unsafe_allow_html=True)
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown('<h2 style="color:#ff7f7f;">Medical Predictions</h2>', unsafe_allow_html=True)

        #st.write("##")
        st.write(
            """
        Medical predictions use data and advanced technologies (like AI and big data) to forecast health outcomes, enabling earlier disease detection, personalized treatments, and more efficient healthcare delivery. In today’s era of digital transformation, they help manage vast health data, support clinical decisions, optimize resources, and promote preventive care. 
        This makes medical predictions essential for improving patient outcomes and addressing the challenges of modern, technology-driven healthcare systems....
            
            """)
    with right_column:
        url1 = get_url("https://assets7.lottiefiles.com/packages/lf20_ecvwbhww.json")
        if url1:
            st_lottie(url1)
        else:
            st.error("Failed to load Lottie animation.")


            #---------------------------------#
# About
    expander_bar =  st.expander("About")
    expander_bar.markdown("""
* **Python libraries:** Streamlit, Numpy , Pandas , Matplotlib , Plotly , Pickle , os , Json , Requests.
* **Data source:** [diabetes.csv]("diabetes.csv") ,  [heart.csv]("heart.csv") , [kidney.csv]("kidney_disease.csv"), [Parkinsons.csv]("parkinsons.csv").
* **Inspiration:** [Medium](https://medium.com/?tag=machine-learning) , [Kaggle](https://www.kaggle.com/) ,  [Geeksforgks](https://www.geeksforgeeks.org/machine-learning/) ,  [stack overflow](https://stackoverflow.com/),  [Streamlit](https://docs.streamlit.io/), [lottiefiles](https://lottiefiles.com/).
""")


    # ---- 2 Home section ----
    with st.container():
        st.write("---")
        st.markdown('<h2 style="color:#ff7f7f;">The Technology That I Used</h2>', unsafe_allow_html=True)
        st.header("Machine Learning Technology?")
        text_column, right_column = st.columns((1, 2))
        with text_column:
            st.write(
            """
Machine Learning (ML) is a core subfield of Artificial Intelligence (AI) that empowers computers to learn from data, identify patterns, and make decisions with minimal human intervention. 
Unlike traditional programming, where explicit instructions are given for every task, ML uses algorithms to analyze large datasets, improving performance over time based on experience. This adaptive capability allows systems to predict outcomes, detect anomalies, and automate complex decisions, making ML a powerful tool across industries like healthcare, finance, and more."""
            )
        with right_column:
            # Load your local JSON file (adjust the path if needed)
            lottie_ML = load_lottiefile("WDfItVJUqT.json")  # or just "WDfItVJUqT.json" if same folder
        
            # Display the animation
            if lottie_ML:
                st_lottie(lottie_ML, speed=1, height=400, key="ML")
            else:
                st.error("❌ Failed to load Lottie animation. Make sure the path is correct.")
    # ---- 3 Home section ----
    with st.container():
        st.write("---")
        st.header("The Following Diseases Predictions Are Available")
        st.markdown('<h2 style="color:#ff7f7f;">1. Diabetes Prevention Predictions</h2>', unsafe_allow_html=True) 
        st.write("##")
        text_column, right_column = st.columns((1, 2))
        with text_column:
            st.write(
 """
    Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into energy.
    Your body breaks down most of the food you eat into sugar (glucose) and releases it into your bloodstream. When your blood sugar goes up, 
    it signals your pancreas to release insulin. Insulin acts like a key to let the blood sugar into your body’s cells for use as energy.
                """
            )
        with right_column:
            url3 = get_url("https://assets9.lottiefiles.com/packages/lf20_tbjuenb2.json")
            if url3:
                st_lottie(url3)
            else:
                st.error("Failed to load Lottie animation.")

    # ---- 4 Home section ----------------
    with st.container():
        st.write("---")
        st.markdown('<h2 style="color:#ff7f7f;">2. Heart Disease Predictions</h2>', unsafe_allow_html=True) 
        st.write("##")
        text_column, right_column = st.columns((1, 2))
        with text_column:
            st.write(
"""
Heart disease, also known as cardiovascular disease, refers to a group of conditions that affect the heart and blood vessels. It encompasses various disorders that can affect the heart's structure and function, leading to impaired blood flow and potential complications.
The most common type of heart disease is coronary artery disease (CAD), which involves the narrowing or blockage of the coronary arteries that supply blood to the heart muscle. This narrowing is usually caused by the buildup of fatty deposits called plaques within the artery walls, a condition known as atherosclerosis.
"""
            )
        with right_column:
            url4 = get_url("https://assets1.lottiefiles.com/packages/lf20_NDRSDjCFia.json")
            if url4:
                st_lottie(url4)
            else:
                st.error("Failed to load Lottie animation.")

    # ---- 5 Home section --------------            
    with st.container():
        st.write("---")
        st.markdown('<h2 style="color:#ff7f7f;">3. Kidney Disease Predictions</h2>', unsafe_allow_html=True) 
        st.write("##")
        text_column, right_column = st.columns((1, 2))
        with text_column:
            st.write(
            """
Kidney disease is a medical condition characterized by gradual or sudden damage to the kidneys, impairing their ability to filter waste, excess fluids, and toxins from the bloodstream effectively. This reduced kidney function can cause harmful substances to accumulate in the body, leading to symptoms such as fatigue, swelling, and changes in urination. If left untreated, kidney disease can progress to kidney failure, requiring dialysis or a kidney transplant to sustain life. It can be caused by conditions like diabetes, high blood pressure, infections, or inherited disorders."""
            )

        with right_column:
            st.header("   ")
            # Load your local JSON file (use the correct path!)
            lottie_kidney = load_lottiefile("vQoJtK3Eqi.json")

            # Display the animation
            if lottie_kidney:
                st_lottie(lottie_kidney, speed=1, height=400, key="kidney")
            else:
                st.error("Failed to load Lottie animation.")
                
                    # ---- 5 Home section --------------            
    with st.container():
        st.write("---")
        st.markdown('<h2 style="color:#ff7f7f;">4. Parkinson Disease Disease Predictions</h2>', unsafe_allow_html=True) 
        st.write("##")
        text_column, right_column = st.columns((1, 2))
        with text_column:
            st.write(
            """
Parkinson's disease is a chronic and progressive neurological disorder that affects movement. It occurs when nerve cells in the brain, especially those in the area called the substantia nigra, become damaged or die, leading to a decrease in dopamine production. This causes symptoms such as tremors, stiffness, slow movements, and balance problems. Parkinson’s can also affect mood, sleep, and thinking abilities. While there is no cure, treatments like medication and therapy can help manage symptoms."""
            )
        with right_column:
            # Load your local JSON file (use the correct path!)
            lottie_Parkinson = load_lottiefile("n6ZTTbeABa.json")

            # Display the animation
            if lottie_Parkinson:
                st_lottie(lottie_Parkinson, speed=1, height=400, key="Parkinson")
            else:
                st.error("Failed to load Lottie animation.")
                
                
    # ---- SOCIAL MEDIA  ------------------------
    with st.container():
        st.write("---")
        st.markdown('<h2 style="color:#ff7f7f;">Discover MY SOCIAL MEDIA</h2>', unsafe_allow_html=True) 
        st.write("##")
        text_column, right_column = st.columns((1, 2))
    with right_column:
        # Load your local JSON file (use the correct path!)
        lottie_kidney = load_lottiefile("1ycP8TXE9C.json")

        # Display the animation
        if lottie_kidney:
            st_lottie(lottie_kidney, speed=1, height=400, key="socialmedia")
        else:
            st.error("Failed to load Lottie animation.")
        with text_column:
            st.header("   ")
            st.header("   ")
            st.header("   ")
            st.button(""". Get In Touch With Me!""") 

    with st.container():
        text_column, right_column = st.columns((1, 2))
    with text_column:
        EMAIL = "loughmarimad@gmail.com"
        st.write("📧", EMAIL)
    with right_column:
        SOCIAL_MEDIA = {
            "🧑‍💼 LinkedIn": "www.linkedin.com/in/imad-loughmari-67b5aa1b3",
            "👨🏾‍💻 GitHub": "https://github.com/imadeddin2000",
            "📲 Twitter": "https://twitter.com",
        }
        cols = st.columns(len(SOCIAL_MEDIA))
        for index, (platform, link) in enumerate(SOCIAL_MEDIA.items()):
            cols[index].write(f"[{platform}]({link})")
            
            

# ---- Multiple Disease Prediction Section ----
else:
    # Load Models
    working_dir = os.path.dirname(os.path.abspath(__file__))
    

    diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes.pkl', 'rb'))
    heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart.pkl', 'rb'))
    kidney_disease_model = pickle.load(open(f'{working_dir}/saved_models/kidney.pkl', 'rb'))
    parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

    # Sidebar Menu
    with st.sidebar:
        selected_disease = option_menu(
            "Multiple Disease Prediction",
            ['Diabetes Disease', 'Heart Disease', 'Kidney Disease', "Parkinson's Disease", 'Plots'],
            menu_icon='hospital-fill',
            icons=['shield-plus', 'heart-pulse-fill', 'droplet-half', 'file-earmark-medical-fill', 'bar-chart-line-fill'],
            default_index=0
        )

    # ---- Diabetes Disease ----
    if selected_disease == 'Diabetes Disease':
        st.markdown('<h2 style="color:#ff7f7f;">Discover If You Have Diabetes Disease Using ML Technology</h2>', unsafe_allow_html=True)

        st.markdown("""
            <div style="background-color:rgb(30, 35, 45); color:white; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <b style="color:#ff7f7f;">Input Ranges:</b><br>
                - Pregnancies: 0 – 17<br>
                - Glucose: 0 – 199<br>
                - Blood Pressure: 0 – 122<br>
                - Skin Thickness: 0 – 99<br>
                - Insulin: 0 – 846<br>
                - BMI: 0 – 67.1<br>
                - Diabetes Pedigree Function: 0.078 – 2.42<br>
                - Age: 21 – 81
            </div>
        """, unsafe_allow_html=True)

        NewBMI_Overweight = NewBMI_Underweight = NewBMI_Obesity_1 = 0
        NewBMI_Obesity_2 = NewBMI_Obesity_3 = NewInsulinScore_Normal = 0
        NewGlucose_Low = NewGlucose_Normal = NewGlucose_Overweight = NewGlucose_Secret = 0

        col1, col2, col3 = st.columns(3)
        with col1:
            Pregnancies = st.text_input("Number of Pregnancies")
        with col2:
            Glucose = st.text_input("Glucose Level")
        with col3:
            BloodPressure = st.text_input("Blood Pressure Value")
        with col1:
            SkinThickness = st.text_input("Skin Thickness Value")
        with col2:
            Insulin = st.text_input("Insulin Value")
        with col3:
            BMI = st.text_input("BMI Value")
        with col1:
            DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
        with col2:
            Age = st.text_input("Age")

        diabetes_result = ""
        if st.button("Diabetes Test Result"):
            try:
                bmi_val = float(BMI) if BMI else 0
                insulin_val = float(Insulin) if Insulin else 0
                glucose_val = float(Glucose) if Glucose else 0

                if bmi_val <= 18.5:
                    NewBMI_Underweight = 1
                elif 24.9 < bmi_val <= 29.9:
                    NewBMI_Overweight = 1
                elif 29.9 < bmi_val <= 34.9:
                    NewBMI_Obesity_1 = 1
                elif 34.9 < bmi_val <= 39.9:
                    NewBMI_Obesity_2 = 1
                elif bmi_val > 39.9:
                    NewBMI_Obesity_3 = 1

                if 16 <= insulin_val <= 166:
                    NewInsulinScore_Normal = 1

                if glucose_val <= 70:
                    NewGlucose_Low = 1
                elif 70 < glucose_val <= 99:
                    NewGlucose_Normal = 1
                elif 99 < glucose_val <= 126:
                    NewGlucose_Overweight = 1
                elif glucose_val > 126:
                    NewGlucose_Secret = 1

                user_input = [
                    Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
                    DiabetesPedigreeFunction, Age, NewBMI_Underweight, NewBMI_Overweight,
                    NewBMI_Obesity_1, NewBMI_Obesity_2, NewBMI_Obesity_3, NewInsulinScore_Normal,
                    NewGlucose_Low, NewGlucose_Normal, NewGlucose_Overweight, NewGlucose_Secret
                ]

                user_input = [float(x) if x else 0.0 for x in user_input]
                prediction = diabetes_model.predict([user_input])
                diabetes_result = "The person has diabetes." if prediction[0] == 1 else "The person does not have diabetes."
            except ValueError:
                diabetes_result = "Please enter valid numeric values."
        st.success(diabetes_result)

    # ---- Heart Disease ----
    elif selected_disease == 'Heart Disease':
        st.markdown('<h2 style="color:#ff7f7f;">Discover If You Have Heart Disease Using ML Technology</h2>', unsafe_allow_html=True)
        
        st.markdown("""
            <div style="background-color:rgb(30, 35, 45); color:white; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <b style="color:#ff7f7f;">Input Ranges:</b><br>
                - Age: 29 – 77<br>
                - Sex: 0 = female, 1 = male<br>
                - Chest Pain (cp): 0 – 3<br>
                - Resting BP: 94 – 200<br>
                - Cholesterol: 126 – 564<br>
                - Fasting BS: 1 = true or 0 = false<br>
                - Rest ECG: 0 – 2<br>
                - Max Heart Rate: 71 – 202<br>
                - Exercise Angina: 1 = Yes or 0 = No<br>
                - ST Depression: 0.0 – 6.2<br>
                - ST Slope: 0 – 2<br>
                - Major Vessels (ca): 0 – 3<br>
                - Thalassemia (thal): 1 = normal, 2 = fixed, 3 = reversible
            </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.text_input("Age")
        with col2:
            sex = st.text_input("Sex (1 = male; 0 = female)")
        with col3:
            cp = st.text_input("Chest Pain Types (0-3)")
        with col1:
            trestbps = st.text_input("Resting Blood Pressure")
        with col2:
            chol = st.text_input("Serum Cholesterol (mg/dl)")
        with col3:
            fbs = st.text_input("Fasting Blood Sugar > 120 mg/dl")
        with col1:
            restecg = st.text_input("Resting ECG results (0-2)")
        with col2:
            thalach = st.text_input("Max Heart Rate Achieved")
        with col3:
            exang = st.text_input("Exercise Induced Angina ")
        with col1:
            oldpeak = st.text_input("ST depression by exercise")
        with col2:
            slope = st.text_input("Slope of peak ST segment (0-2)")
        with col3:
            ca = st.text_input("Number of major vessels (0-3)")
        with col1:
            thal = st.text_input("Thalassemia ")

        heart_result = ""
        if st.button("Heart Disease Test Result"):
            try:
                user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                              exang, oldpeak, slope, ca, thal]
                user_input = [float(x) if x else 0.0 for x in user_input]
                prediction = heart_disease_model.predict([user_input])
                heart_result = "The person has heart disease." if prediction[0] == 1 else "The person does not have heart disease."
            except ValueError:
                heart_result = "Please enter valid numeric values."
        st.success(heart_result)

    # ---- Kidney Disease ----
    elif selected_disease == 'Kidney Disease':
        st.markdown('<h2 style="color:#ff7f7f;">Discover If You Have Kidney Disease Using ML Technology</h2>', unsafe_allow_html=True)

        st.markdown("""
            <div style="background-color:rgb(30, 35, 45); color:white; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <b style="color:#ff7f7f;">Input Ranges:</b><br>
                - Age: 0 – 90<br>
                - Blood Pressure: 50 – 180<br>
                - Specific Gravity: 1.005 – 1.025<br>
                - Albumin & Sugar: 0 – 5<br>
                - Blood Glucose: 22 – 490<br>
                - Blood Urea: 1.5 – 391<br>
                - Serum Creatinine: 0.4 – 76<br>
                - Sodium: 111 – 163<br>
                - Potassium: 2.5 – 47.0<br>
                - Hemoglobin: 3.1 – 17.8<br>
                - Packed Cell Volume: 9 – 54<br>
                - WBC: 2200 – 26400<br>
                - RBC Count: 2.1 – 6.9<br>
            </div>
        """, unsafe_allow_html=True)

        fields = [
            "Age", "Blood Pressure", "Specific Gravity", "Albumin", "Sugar", "Red Blood Cells",
            "Pus Cell", "Pus Cell Clumps", "Bacteria", "Blood Glucose", "Blood Urea",
            "Serum Creatinine", "Sodium", "Potassium", "Hemoglobin", "Packed Cell Volume",
            "White Blood Cell", "Red Blood Cell Count", "Hypertension", "Diabetes Mellitus",
            "Coronary Artery", "Appetite", "Pedal Edema", "Anemia"
        ]

        user_values = []
        for i in range(0, len(fields), 5):
            cols = st.columns(5)
            for j in range(5):
                if i + j < len(fields):
                    user_values.append(cols[j].text_input(fields[i + j]))

        kidney_result = ""
        if st.button("Kidney Disease Test Result"):
            try:
                input_floats = [float(x) if x else 0.0 for x in user_values]
                prediction = kidney_disease_model.predict([input_floats])
                kidney_result = "The person has kidney disease." if prediction[0] == 1 else "The person does not have kidney disease."
            except ValueError:
                kidney_result = "Please enter valid numeric values."
        st.success(kidney_result)

    # ---- Parkinson's Disease ----
    elif selected_disease == "Parkinson's Disease":
        st.markdown('<h2 style="color:#ff7f7f;">Discover If You Have Parkinson Disease Using ML Technology</h2>', unsafe_allow_html=True)

        st.markdown("""
            <div style="background-color:rgb(30, 35, 45); color:white; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <b style="color:#ff7f7f;">Input Ranges:</b><br>
                - MDVP:Fo(Hz): 80 – 262<br>
                - MDVP:Fhi(Hz): 110 – 320<br>
                - MDVP:Flo(Hz): 65 – 220<br>
                - MDVP:Jitter(%): 0.001 – 0.02<br>
                - MDVP:Jitter(Abs): 0.00001 – 0.00015<br>
                - MDVP:RAP: 0.001 – 0.02<br>
                - MDVP:PPQ: 0.001 – 0.02<br>
                - Jitter:DDP: 0.002 – 0.05<br>
                - MDVP:Shimmer: 0.01 – 0.1<br>
                - MDVP:Shimmer(dB): 0.1 – 1.5<br>
                - Shimmer:APQ3: 0.005 – 0.08<br>
                - Shimmer:APQ5: 0.007 – 0.09<br>
                - MDVP:APQ: 0.006 – 0.1<br>
                - Shimmer:DDA: 0.02 – 0.15<br>
                - NHR: 0.01 – 0.04<br>
                - HNR: 10 – 30<br>
                - RPDE: 0.3 – 1.0<br>
                - DFA: 0.5 – 1.0<br>
                - spread1: -8 – -1<br>
                - spread2: 0 – 2<br>
                - D2: 1.5 – 3.0<br>
                - PPE: 0.1 – 0.9
            </div>
        """, unsafe_allow_html=True)

        parkinsons_fields = [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
            'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
            'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
            'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
        ]

        parkinsons_inputs = []
        for i in range(0, len(parkinsons_fields), 5):
            cols = st.columns(5)
            for j in range(5):
                if i + j < len(parkinsons_fields):
                    parkinsons_inputs.append(cols[j].text_input(parkinsons_fields[i + j]))

        parkinsons_diagnosis = ""
        if st.button("Parkinson's Test Result"):
            try:
                input_floats = [float(x) if x else 0.0 for x in parkinsons_inputs]
                prediction = parkinsons_model.predict([input_floats])
                parkinsons_diagnosis = "The person has Parkinson's disease." if prediction[0] == 1 else "The person does not have Parkinson's disease."
            except ValueError:
                parkinsons_diagnosis = "Please enter valid numeric values."
        st.success(parkinsons_diagnosis)

# ---- Plots Section ----
    elif selected_disease == 'Plots':
        st.markdown('<h1 style="color:#ff7f7f;">📊 Disease Data Visualization</h1>', unsafe_allow_html=True)

        # Load datasets (make sure they exist in the correct path)
        df_diabetes = pd.read_csv("dataset/diabetes.csv")
        df_heart = pd.read_csv("dataset/heart.csv")
        df_kidney = pd.read_csv("dataset/kidney_disease.csv")
        df_parkinsons = pd.read_csv("dataset/parkinsons.csv")

        # Dropdown menu for choosing disease
        plot_choice = st.selectbox(
            "Which Disease Data Would You Like to Explore?",
            ['Diabetes', 'Heart Disease', 'Kidney Disease', "Parkinson's Disease"]
        )

        # Plotting based on selection
        if plot_choice == 'Diabetes': 
            st.subheader("Diabetes Data Distribution")
            fig = px.density_heatmap(df_diabetes, x="Glucose", y="Outcome",
                                    title="Glucose Level vs Diabetes Outcome")
            fig.update_layout(title_font=dict(color="#ff7f7f"))  # Set title color here
            st.plotly_chart(fig)
            st.subheader("Detailed Diabetes Data")
            st.dataframe(df_diabetes)

        elif plot_choice == 'Heart Disease':
            st.subheader("Heart Disease Data Distribution")
            fig = px.scatter(df_heart, x='thalach', y='chol', color='target',
                             title="Max Heart Rate vs Cholesterol by Heart Disease Status",
                             labels={'thalach': 'Max Heart Rate', 'chol': 'Cholesterol'})
            fig.update_layout(title_font=dict(color="#ff7f7f"))  # Title color set here
            st.plotly_chart(fig)
            st.subheader("Detailed Heart Disease Data")
            st.dataframe(df_heart)

        elif plot_choice == 'Kidney Disease':
            st.subheader("Kidney Disease Data Distribution")
            fig = px.pie(df_kidney, names='classification', title='Kidney Disease Classification Proportion')
            fig.update_layout(title_font=dict(color="#ff7f7f"))  # Title color styling
            st.plotly_chart(fig)
            st.subheader("Detailed Kidney Disease Data")
            st.dataframe(df_kidney)

        elif plot_choice == "Parkinson's Disease": 
            st.subheader("Parkinson's Disease Data Distribution")
            fig = px.scatter_3d(
                df_parkinsons,
                x="MDVP:Fo(Hz)",
                y="MDVP:Jitter(%)",
                z="MDVP:Shimmer",
                color="status",
                title="3D Scatter Plot of Parkinson's Features"
            )
            fig.update_layout(title_font=dict(color="#ff7f7f"))  # Set title color
            st.plotly_chart(fig)
            st.subheader("Detailed Parkinson's Disease Data")
            st.dataframe(df_parkinsons)
