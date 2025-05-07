import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Chargement des données
data = pd.read_csv('Telco_Customer_Churn.csv', delimiter=',', header=0)

# Suppression de la colonne customerID
data = data.drop('customerID', axis=1, errors='ignore')

# Gestion des valeurs manquantes (TotalCharges)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].mean())

# Encodage des variables catégorielles
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = data[column].astype('category')
        data[column] = data[column].cat.codes

# Normalisation des variables numériques
scaler = StandardScaler()
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Division des données
X = data.drop('Churn', axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle (Régression Logistique)
model = LogisticRegression()
model.fit(X_train, y_train)

# Sauvegarde du modèle, scaler, numerical_cols and all columns
all_columns = X.columns.tolist()
with open("churn_model6.pkl", "wb") as f:
    pickle.dump((model, scaler, numerical_cols, all_columns), f)

# Charger le modèle
with open("churn_model6.pkl", "rb") as f:
    model, scaler, numerical_cols, all_columns = pickle.load(f)

# App title and description
st.markdown("<h1 style='color:#FF4B4B;'>ORANGE CUSTOMERS APPLI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='color:#1E90FF;'>Entrez les informations du client pour prédire le churn.</h4>", unsafe_allow_html=True)

# Sidebar for user input
st.sidebar.header("Informations Client")

# Define input fields with appropriate labels and options
gender = st.sidebar.selectbox("Genre", ["Male", "Female"])
seniorcitizen = st.sidebar.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.sidebar.selectbox("Partenaire", ["Yes", "No"])
dependents = st.sidebar.selectbox("Personnes à charge", ["Yes", "No"])
tenure = st.sidebar.slider("Ancienneté (mois)", 0, 72, 12)
phoneservice = st.sidebar.selectbox("Service Téléphonique", ["Yes", "No"])
multiplelines = st.sidebar.selectbox("Lignes Multiples", ["Yes", "No", "No phone service"])
internetservice = st.sidebar.selectbox("Service Internet", ["DSL", "Fiber optic", "No"])
onlinesecurity = st.sidebar.selectbox("Sécurité en Ligne", ["Yes", "No", "No internet service"])
onlinebackup = st.sidebar.selectbox("Sauvegarde en Ligne", ["Yes", "No", "No internet service"])
deviceprotection = st.sidebar.selectbox("Protection de l'appareil", ["Yes", "No", "No internet service"])
techsupport = st.sidebar.selectbox("Support Technique", ["Yes", "No", "No internet service"])
streamingtv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streamingmovies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.sidebar.selectbox("Contrat", ["Month-to-month", "One year", "Two year"])
paperlessbilling = st.sidebar.selectbox("Facturation dématérialisée", ["Yes", "No"])
paymentmethod = st.sidebar.selectbox("Mode de Paiement", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthlycharges = st.sidebar.number_input("Frais Mensuels", min_value=0.0)
totalcharges = st.sidebar.number_input("Frais Totaux", min_value=0.0)

# Prepare input data for prediction
input_data = {
    "gender": 1 if gender == "Male" else 0,
    "SeniorCitizen": 1 if seniorcitizen == "Yes" else 0,
    "Partner": 1 if partner == "Yes" else 0,
    "Dependents": 1 if dependents == "Yes" else 0,
    "tenure": tenure,
    "PhoneService": 1 if phoneservice == "Yes" else 0,
    "MultipleLines": 0 if multiplelines == "No phone service" else (1 if multiplelines == "Yes" else 0),
    "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2}[internetservice],
    "OnlineSecurity": 0 if onlinesecurity == "No internet service" else (1 if onlinesecurity == "Yes" else 0),
    "OnlineBackup": 0 if onlinebackup == "No internet service" else (1 if onlinebackup == "Yes" else 0),
    "DeviceProtection": 0 if deviceprotection == "No internet service" else (1 if deviceprotection == "Yes" else 0),
    "TechSupport": 0 if techsupport == "No internet service" else (1 if techsupport == "Yes" else 0),
    "StreamingTV": 0 if streamingtv == "No internet service" else (1 if streamingtv == "Yes" else 0),
    "StreamingMovies": 0 if streamingmovies == "No internet service" else (1 if streamingmovies == "Yes" else 0),
    "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract],
    "PaperlessBilling": 1 if paperlessbilling == "Yes" else 0,
    "PaymentMethod": {"Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card (automatic)": 3}[paymentmethod],
    "MonthlyCharges": monthlycharges,
    "TotalCharges": totalcharges,
}

input_df = pd.DataFrame([input_data])

# Convert categorical features to numerical values based on your original encoding
for column in input_df.columns:
    if input_df[column].dtype == 'object':
        try:
            input_df[column] = input_df[column].astype('category')
            input_df[column] = input_df[column].cat.codes
        except:
            pass

# Add missing columns to input_df with 0 values
for col in all_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Ensure the order of columns is the same as the training data
input_df = input_df[all_columns]

# Identify numerical columns in the input data that were used for scaling
numerical_cols_input = input_df.columns.intersection(numerical_cols)

# Scale numerical features
input_df[numerical_cols_input] = scaler.transform(input_df[numerical_cols_input])


# Make prediction
if st.button("Prédire"):
    try:
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]

        if prediction[0] == 1:
            st.markdown(f"<h2 style='color:red;'>Le client est susceptible de partir. (Probabilité: {probability:.2f})</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color:green;'>Le client n'est pas susceptible de partir. (Probabilité: {1 - probability:.2f})</h2>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Une erreur s'est produite lors de la prédiction: {e}")

# Add some styling
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
        font-family: sans-serif;
    }
    .stApp {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        color: #ffffff;
        background-color: #007bff;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    </style>
    """, unsafe_allow_html=True)