import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
import os
import joblib

# URL de l'API
API_URL = "https://my-app-scoring-api-660b74752f36.herokuapp.com/"

st.title("Demande de financement")

# Chargement des données
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'Data', 'test_selected_data.csv')
original_data_path = os.path.join(BASE_DIR, 'Data', 'data_original.csv')
df_original = pd.read_csv(original_data_path, index_col='SK_ID_CURR')
df_test = pd.read_csv(data_path, index_col=0)
client_ids = df_test.index.tolist()


# CSS pour respect des normes WCAG
st.markdown("""
<style>
h1 {
    font-size: 50px !important;  /* Taille de la police du titre */
}
.stMarkdown {
    font-family: 'Arial', sans-serif; /* Police lisible */
    font-size: 22px; /* Taille augmentée pour lisibilité */
    color: #333333; /* Couleur de texte pour contraste suffisant */
    background-color: #ffffff; /* Fond clair pour contraste */
}
.stButton>button:focus {
    outline: 3px solid #333; /* Focus visible pour navigation clavier */
}


</style>
""", unsafe_allow_html=True)

# Vérifier l'état de santé de l'API
response_health = requests.get(f"{API_URL}/health")
if response_health.status_code == 200:
    st.sidebar.success("L'API est en ligne")
else:
    st.sidebar.error("L'API est hors ligne")

show_credit_decision_button = st.sidebar.checkbox("Décision de crédit")

selected_client_id = st.selectbox("Sélectionner l'identifiant du client", client_ids)
st.write(f"ID Client Sélectionné : {selected_client_id}")

# Afficher les informations du client

if selected_client_id in df_original.index:
    client_info_original = df_original.loc[selected_client_id]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Informations personnelles")
        st.write(f"**Sexe:** {client_info_original['CODE_GENDER']}")
        st.write(f"**Âge:** {round(client_info_original['AGE_YEARS'])}")
        st.write(f"**Statut familial:** {client_info_original['NAME_FAMILY_STATUS']}")
        st.write(f"**Années employées:** {client_info_original['YEARS_EMPLOYED']:.1f}")
        st.write(f"**Niveau d'études:** {client_info_original['NAME_EDUCATION_TYPE']}")
        st.write(f"**Type d'organisation:** {client_info_original['ORGANIZATION_TYPE']}")
        st.write(f"**Type de revenu:** {client_info_original['NAME_INCOME_TYPE']}")

    with col2:
        st.subheader("Informations financières")
        st.write(f"**Revenu total annuel:** {client_info_original['AMT_INCOME_TOTAL']}")
        st.write(f"**Dette totale:** {client_info_original['TOTAL_DEBT']}")
        total_active_credits = client_info_original['TOTAL_ACTIVE_CREDITS']
        if pd.notna(total_active_credits):
            st.write(f"**Crédits actifs:** {round(total_active_credits)}")
        else:
            st.write("**Crédits actifs:** non disponible")

        avg_days_past_due = client_info_original['AVG_DAYS_PAST_DUE']
        if pd.notna(avg_days_past_due):
            st.write(f"**Jours moyens de paiement en retard:** {round(avg_days_past_due)}")
        else:
            st.write("**Jours moyens de paiement en retard:** Information non disponible")
        st.write(f"**Taux d'endettement (en %):** {client_info_original['TX_ENDETTEMENT']:.2f}")

else:
    st.error("Les données originales pour cet ID client ne sont pas disponibles.")

# Affichage de la décision de crédit
if show_credit_decision_button:
    client_data = df_test.loc[selected_client_id].to_dict()
    client_data["client_id"] = int(selected_client_id)

    response = requests.post(f"{API_URL}/predict", json=client_data)
    if response.status_code == 200:
        prediction = response.json()
        prob = prediction['probability']

        st.write("## Évaluation du Risque de Crédit")
        # Expander pour des explications supplémentaires
        with st.expander("Explications"):
            st.markdown("""
            La jauge ci-dessous représente l'estimation quant au **risque** que le client ne respecte pas ses engagements de crédit.
            - Un score inférieur à **30%** est considéré comme un **risque faible**,
            - Un score entre **30%** et **52%** est considéré comme un **risque modéré** et **nécessite une attention**,
            - Un score supérieur à **52%** est considéré comme un **risque élevé**.
            """)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            number={'suffix': "%", 'font': {'size': 73, 'color': "black"}},
            domain={'x': [0.1, 0.9], 'y': [0.2, 0.9]},
            title={'text': "Probabilité de défaut", 'font': {'size': 20, 'color': 'black'}},
            gauge={
                'axis': {'range': [0, 100], 'tickvals': [15, 40, 75], 'tickfont': {'size': 16, 'color': 'black'},
                         'ticktext': ['Risque faible', 'Risque modéré', 'Risque élevé']},
                'bar': {'color': "rgba(0, 0, 0, 0)"},
                'steps': [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 52], 'color': "yellow"},
                    {'range': [52, 100], 'color': "red"}
                ],
                'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.8, 'value': prob}
            }
        ))
        st.plotly_chart(fig)

        decision_text = f"Décision de crédit : {prediction['class_prediction']}"
        st.markdown(
            f"<h3 style='text-align: center; color: {'green' if prediction['class_prediction'] == 'Accepté' else 'red'};'>{decision_text}</h3>",
            unsafe_allow_html=True)

        # Explications de la décision
        st.write("## Explication détaillée de la décision")
        shap_values_dict = prediction['shap_values']
        shap_df = pd.DataFrame(list(shap_values_dict.items()), columns=['Caractéristique', 'Valeur'])

        # Descriptions des caractéristiques
        descriptions = {
            "CODE_GENDER": "Sexe ",
            "DUREE": "Durée du prêt",
            "TX_ENDETTEMENT": "Taux d'endettement",
            "AVG_INSURANCE_STATUS": "Statut moyen d'assurance",
            "NAME_INCOME_TYPE_State servant": "Fonctionnaire",
            "ORGANIZATION_TYPE_Legal Services": "Services juridiques",
            "ORGANIZATION_TYPE_Self-employed": "Travailleur indépendant",
            "NAME_INCOME_TYPE_Student": "Étudiant",
            "TOTAL_LOAN_APP": "Nombre total de demandes de prêt",
            "YEARS_LAST_PHONE_CHANGE": "Années depuis le dernier changement de téléphone",
            "NAME_FAMILY_STATUS_Married": " Statut familial",
            "AVG_DAYS_PAST_DUE": "Jours moyens de retard de paiement",
            "REGION_RATING_CLIENT_W_CITY": "Évaluation régionale démographique",
            "ORGANIZATION_TYPE_Security Ministries": "Ministères de la sécurité",
            "NAME_EDUCATION_TYPE_Higher education": "Enseignement supérieur",
            "AGE_YEARS": "Âge",
            "NAME_INCOME_TYPE_Commercial associate": "Associé commercial",
            "TOTAL_ACTIVE_CREDITS": "Nombre total de crédits actifs",
            "NAME_CONTRACT_TYPE": "Type de contrat",
            "YEARS_EMPLOYED": "Années d'emploi",
            "AMT_CREDIT": "Montant du crédit",
            "NAME_INCOME_TYPE_Pensioner": "Retraité",
            "TOTAL_DEBT": "Dette totale",
            "AMT_INCOME_TOTAL": "Revenu annuel total",
            "AVG_CREDIT_GRANTED": "Crédit moyen accordé",
            "ORGANIZATION_TYPE_Construction": "Construction",
            "ORGANIZATION_TYPE_Military": "Militaire",
            "EXT_SOURCE_3": "Score 3 risque autre établissement",
            "ORGANIZATION_TYPE_Realtor": "Agent immobilier",
            "NAME_INCOME_TYPE_Unemployed": "Sans Emploi",
            "NAME_INCOME_TYPE_Working": "En activité",
            "EXT_SOURCE_2": "Score 2 risque autre établissement"
        }

        shap_df['Description'] = shap_df['Caractéristique'].map(descriptions)

        shap_df['Impact absolu'] = shap_df['Valeur'].abs()
        shap_df_sorted = shap_df.sort_values(by='Impact absolu', ascending=False).head(10)

        # Graphique en barres
        fig_bar = go.Figure()

        fig_bar.add_trace(go.Bar(
            y=shap_df_sorted['Caractéristique'],
            x=shap_df_sorted['Valeur'].clip(lower=0),
            orientation='h',
            marker=dict(color='rgba(255, 0, 0, 0.6)'),
            name='Impact négatif'
        ))

        fig_bar.add_trace(go.Bar(
            y=shap_df_sorted['Caractéristique'],
            x=shap_df_sorted['Valeur'].clip(upper=0),
            orientation='h',
            marker=dict(color='rgba(0, 128, 0, 0.6)'),
            name='Impact positif'
        ))

        fig_bar.update_layout(
            title={
                'text': "Top 10 des informations importantes",
                'font': {'size': 20, 'color': 'black'}
            },
            xaxis_title={
                'text': 'Impact',
                'font': {'size': 18, 'color': 'black'}
            },
            yaxis_title={
                'text': 'Informations',
                'font': {'size': 18, 'color': 'black'}
            },
            barmode='relative',
            bargap=0.1,
            showlegend=True,
            legend=dict(
                font=dict(size=14, color='black')
            ),
            height=600
        )

        fig_bar.update_xaxes(
            tickfont=dict(size=14, color='black'),
            title_font=dict(size=18, color='black')
        )

        fig_bar.update_yaxes(
            tickfont=dict(size=14, color='black'),
            title_font=dict(size=18, color='black')
        )

        st.plotly_chart(fig_bar)

        # Caractéristiques les plus influentes
        st.write("## Caractéristiques décisives")
        top_features = shap_df.sort_values(by='Valeur', ascending=False).head(5)
        for idx, row in top_features.iterrows():
            direction = "augmente" if row['Valeur'] > 0 else "diminue"
            st.write(
                f"**{row['Description']}:** {direction} la probabilité de défaut de paiement.")

    else:
        st.error(f"Erreur : {response.status_code}")
        st.error(response.text)

show_shap_details = st.sidebar.checkbox("Analyse détaillée")

if show_shap_details:
    df_original['Score de stabilité financière'] = (df_original['AMT_CREDIT'] / df_original['AMT_INCOME_TOTAL']) * 100

    client_info = df_original.loc[selected_client_id]

    # Score du client
    st.write("## Score de stabilité financière")
    with st.expander("Explications"):
        st.markdown("""
        Le **score de stabilité financière** est calculé en divisant le montant total du crédit par le revenu annuel total et en multipliant le résultat par 100.  
        Cette formule permet de mesurer la proportion du crédit par rapport au revenu annuel d'un client  

        - Un **score bas** indique une bonne stabilité financière
        - Un **score autour de la moyenne** indique une stabilité financière modérée,
        - Un **score élevé** indique un risque financier important.  

        Le **graphique** représente la distribution des scores pour l'ensemble des clients. Il permet de comparer le client à la fois à la moyenne mais également aux autres clients de manière anonyme.
        """)

    # Déterminer la couleur du score en fonction de sa valeur
    score_color = "green" if client_info['Score de stabilité financière'] < 300 else "orange" if client_info[
                                                                                                     'Score de stabilité financière'] < 450 else "red"

    st.write(
        f"#### Le Score de Stabilité financière du client est de : <span style='color:{score_color}'>{round(client_info['Score de stabilité financière'])}</span>",
        unsafe_allow_html=True)

    mean_score = df_original['Score de stabilité financière'].mean()

    fig_score_distribution = px.histogram(df_original, x='Score de stabilité financière',
                                          title='Distribution des scores de stabilité financière')

    fig_score_distribution.add_vline(x=mean_score, line_width=5, line_dash="dash", line_color="black")

    fig_score_distribution.add_vline(x=client_info['Score de stabilité financière'], line_width=5, line_dash="dash",
                                     line_color="red")

    fig_score_distribution.add_annotation(x=mean_score, y=0.9, yref="paper", showarrow=True, arrowhead=2, ax=20, ay=-40,
                                          text="Moyenne", font=dict(size=16, color="black"))
    fig_score_distribution.add_annotation(x=client_info['Score de stabilité financière'], y=0.9, yref="paper",
                                          showarrow=True, arrowhead=2, ax=-20, ay=-40, text="Client",
                                          font=dict(size=16, color="black"))

    fig_score_distribution.update_layout(
        width=900,
        height=500
    )

    st.plotly_chart(fig_score_distribution)

    # Comparaison aux niveaux des sources 2 et 3
    st.write("## Scores de confiance externes")
    with st.expander("Explications"):
        st.markdown("""
        Les **scores de confiance externes**, sont des **indicateurs** pour évaluer la probabilité qu'un client rembourse un prêt.  
        Ces scores sont fournis par des agences de crédit externes et sont basés sur diverses données financières et comportementales.
        - Un **score bas** indique un **risque élevé** de non-remboursement
        - Un **score élevé** indique un **risque faible** de non-remboursement  

        **Pour le client**, comprendre son score et comment il se compare à la moyenne peut fournir des indications sur **sa position de risque et les actions possibles** pour **améliorer** son score avec son conseiller.""")

    # Statistiques pour les sources 2 et 3
    stats = {
        'Valeurs': ['Min', 'Max', 'Moyenne', 'Client'],
        'Autre établissement 2': [
            df_original['EXT_SOURCE_2'].min(),
            df_original['EXT_SOURCE_2'].max(),
            df_original['EXT_SOURCE_2'].mean(),
            client_info['EXT_SOURCE_2']
        ],
        'Autre établissement 3': [
            df_original['EXT_SOURCE_3'].min(),
            df_original['EXT_SOURCE_3'].max(),
            df_original['EXT_SOURCE_3'].mean(),
            client_info['EXT_SOURCE_3']
        ]
    }

    df_stats = pd.DataFrame(stats)
    st.table(df_stats)

# Explication des features
show_feature_description = st.sidebar.checkbox("Aide descriptions")

if show_feature_description:
    st.write("## Aide description des features")
    st.markdown('''
    - **CODE_GENDER**: Sexe
    - **DUREE**: Durée du prêt
    - **TX_ENDETTEMENT**: Taux d'endettement
    - **AVG_INSURANCE_STATUS**: Statut moyen d'assurance
    - **NAME_INCOME_TYPE_State servant**: Fonctionnaire
    - **ORGANIZATION_TYPE_Legal Services**: Services juridiques
    - **ORGANIZATION_TYPE_Self-employed**: Travailleur indépendant
    - **NAME_INCOME_TYPE_Student**: Étudiant
    - **TOTAL_LOAN_APP**: Nombre total de demandes de prêt
    - **YEARS_LAST_PHONE_CHANGE**: Années depuis le dernier changement de téléphone
    - **NAME_FAMILY_STATUS_Married**: Statut familial
    - **AVG_DAYS_PAST_DUE**: Jours moyens de retard de paiement
    - **REGION_RATING_CLIENT_W_CITY**: Évaluation régionale démographique
    - **ORGANIZATION_TYPE_Security Ministries**: Ministères de la sécurité
    - **NAME_EDUCATION_TYPE_Higher education**: Enseignement supérieur
    - **AGE_YEARS**: Âge
    - **NAME_INCOME_TYPE_Commercial associate**: Associé commercial
    - **TOTAL_ACTIVE_CREDITS**: Nombre total de crédits actifs
    - **NAME_CONTRACT_TYPE**: Type de contrat
    - **YEARS_EMPLOYED**: Années d'emploi
    - **AMT_CREDIT**: Montant du crédit
    - **NAME_INCOME_TYPE_Pensioner**: Retraité
    - **TOTAL_DEBT**: Dette totale
    - **AMT_INCOME_TOTAL**: Revenu annuel total
    - **AVG_CREDIT_GRANTED**: Crédit moyen accordé
    - **ORGANIZATION_TYPE_Construction**: Construction
    - **ORGANIZATION_TYPE_Military**: Militaire
    - **EXT_SOURCE_3**: Score 3 risque autre établissement
    - **ORGANIZATION_TYPE_Realtor**: Agent immobilier
    - **NAME_INCOME_TYPE_Unemployed**: Sans Emploi
    - **NAME_INCOME_TYPE_Working**: En activité
    - **EXT_SOURCE_2**: Score 2 risque autre établissement
    ''')