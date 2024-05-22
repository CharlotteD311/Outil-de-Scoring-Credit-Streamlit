import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
import os

# URL de l'API
API_URL = "https://my-app-scoring-api-660b74752f36.herokuapp.com/"

st.title("Demande de financement")

# Charger les données de test pour obtenir les identifiants des clients
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'Data', 'test_selected_data.csv')
df_test = pd.read_csv(data_path, index_col=0)
client_ids = df_test.index.tolist()

selected_client_id = st.selectbox("Sélectionner l'identifiant du client", client_ids)
st.write(f"ID Client Sélectionné : {selected_client_id}")

# Actions à effectuer
st.sidebar.title("Informations")
show_credit_decision_button = st.sidebar.checkbox("Afficher la décision de crédit")
show_client_info = st.sidebar.checkbox("Afficher les informations du client")
show_shap_details = st.sidebar.checkbox("Afficher les détails de la décision")
show_feature_description = st.sidebar.checkbox("Aide descriptions")


# Vérifier l'état de santé de l'API
response_health = requests.get(f"{API_URL}/health")
if response_health.status_code == 200:
    st.sidebar.success("L'API est en ligne")
else:
    st.sidebar.error("L'API est hors ligne")

# Affichage de la décision de crédit
if show_credit_decision_button:
    client_data = df_test.loc[selected_client_id].to_dict()
    client_data["client_id"] = int(selected_client_id)  # Inclure l'ID du client dans les données

    # Envoyer la requête à l'API
    response = requests.post(f"{API_URL}/predict", json=client_data)

    if response.status_code == 200:
        prediction = response.json()
        prob = prediction['probability']
        st.write("### Scoring et décision du modèle")

        # Jauge de probabilité avec étiquettes de risque
        fig = go.Figure(go.Indicator(
            mode="gauge",
            value=prob,
            title={'text': "Probabilité de défault"},
            gauge={
                'axis': {'range': [0, 100], 'tickvals': [20, 50, 75], 'ticktext': ['Risque faible', 'Risque modéré', 'Risque élevé']},
                'bar': {'color': "rgba(0, 0, 0, 0)"},
                'steps': [
                    {'range': [0, 40], 'color': "green"},
                    {'range': [40, 60], 'color': "orange"},
                    {'range': [60, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': prob
                }
            }
        ))

        st.plotly_chart(fig)

        # Bouton pour afficher la décision de crédit
        decision_text = f"Décision de crédit : {prediction['class_prediction']}"
        if st.button("Afficher la décision de crédit"):
            if prediction['class_prediction'] == "Accepté":
                st.markdown(
                    f"""
                    <div style='text-align: center;'>
                        <h3 style='color: green;'>{decision_text}</h3>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(
                    f"""
                    <div style='text-align: center;'>
                        <h3 style='color: red;'>{decision_text}</h3>
                    </div>
                    """, unsafe_allow_html=True)

        # Affichage des valeurs SHAP pour les caractéristiques
        if show_shap_details:
            st.write("### Explication de la décision à partir des informations")
            shap_values_dict = prediction['shap_values']
            shap_df = pd.DataFrame(list(shap_values_dict.items()), columns=['Caractéristique', 'Valeur'])

            # Visualisation des valeurs SHAP
            fig_shap = px.bar(shap_df, x='Valeur', y='Caractéristique', orientation='h',
                              color='Valeur', color_continuous_scale=px.colors.diverging.RdBu)
            fig_shap.update_layout(title="Impact des caractéristiques sur la prédiction", yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_shap)

    else:
        st.error(f"Erreur : {response.status_code}")
        st.error(response.text)

# Affichage des informations du client
if show_client_info:
    st.write("### Informations du client")
    client_info = df_test.loc[selected_client_id]

    financial_columns = ["TX_ENDETTEMENT", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_GOODS_PRICE"]
    personal_columns = ["CODE_GENDER", "NAME_FAMILY_STATUS_Married"]

    existing_personal_columns = [col for col in personal_columns if col in client_info]
    existing_financial_columns = [col for col in financial_columns if col in client_info]

    if existing_personal_columns:
        st.write("#### Informations personnelles")

        # Transformation des valeurs pour CODE_GENDER et NAME_FAMILY_STATUS_Married
        if "CODE_GENDER" in client_info:
            client_info["CODE_GENDER"] = "Femme" if client_info["CODE_GENDER"] == 0 else "Homme"
        if "NAME_FAMILY_STATUS_Married" in client_info:
            client_info["NAME_FAMILY_STATUS_Married"] = "Marié" if client_info["NAME_FAMILY_STATUS_Married"] == 1 else "Non Marié"

        st.write(client_info[existing_personal_columns])
    else:
        st.write("#### Informations personnelles")
        st.write("Aucune information personnelle disponible pour ce client.")

    if existing_financial_columns:
        st.write("#### Informations financières")
        st.write(client_info[existing_financial_columns])
    else:
        st.write("#### Informations financières")
        st.write("Aucune information financière disponible pour ce client.")

    st.write("#### Autres informations")
    other_columns = [col for col in client_info.index if col not in financial_columns + personal_columns]
    st.write(client_info[other_columns])

if show_feature_description:
    st.write("### Aide description des features")
    # Vous pouvez ajouter du code ici pour afficher la description des features
    st.write("""
    - **AMT_CREDIT**: Montant total emprunté en cours 
    - **AMT_INCOME_TOTAL**: Revenu total annuel du client
    - **YEARS_LAST_PHONE_CHANGE**: Années depuis le dernier changement de téléphone
    - **AVG_CREDIT_GRANTED**: Crédit moyen accordé
    - **EXT_SOURCE_2 et 3**: Source externe (autres établissements)
    - **AVG_DAYS_PAST_DUE**: Jours moyens de retard de paiement
    """)