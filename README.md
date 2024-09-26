import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Affichage du menu dans la barre latérale avec des boutons
st.sidebar.title("Sommaire")
pages = ["Introduction", "Exploration de données", "DataVizualization", "Machine Learning (ML)"]
page = st.sidebar.radio("Aller vers", pages)

# Création d'une zone de texte pour chaque page
if page == "Introduction":
    # On utilise du HTML/CSS pour le style du titre et des images
  st.markdown("""
    <style>
    .title-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .title-container img {
        width: 100px;
        height: 100px;
        margin: 0 20px;
    }
    .title-container h1 {
        margin: 0;
        font-size: 3em;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)
# Titre centré avec deux images
  st.markdown("""
    <div class="title-container">
        <img src="data:image/jpeg;base64,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" alt="Image de gauche">
        <h1>Projet MPG x DataScientest</h1>
        <img src="data:image/jpeg;base64,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" alt="Image de droite">
    </div>
""", unsafe_allow_html=True)
#Texte
  st.write("Ce projet a été réalisé dans le cadre du cursus “Data Analyst” dispensé par l’organisme de formation https://datascientest.com.")
  st.write("L’objectif de ce projet consistait à effectuer une analyse de données sur le jeu de football fantasy Mon Petit Gazon pour proposer une aide à la composition de la meilleure équipe possible aux utilisateurs du jeu.")
  st.write("Afin de nous familiariser avec les données (issues du site MPG Stats), un contact a été établi avec l’équipe support. Cet échange nous a permis de mieux appréhender les variables contenues dans les différents datasets (un par championnat).")
  st.write("Notre analyse se découpera en 3 parties :")
  st.write("- Exploration des données via un export réalisé sur le site https://www.mpgstats.fr")
  st.write("- DataVizualization afin de dégager d'éventuels corrélations entres les variables de notre dataset")
  st.write("- Modélisation afin de choisir le meilleur modèle pour prédire notre variable cible")
  st.markdown("<br>", unsafe_allow_html=True)
  st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <img src="https://www.mpg-torreip.fr/wp-content/uploads/2023/08/bonus-capitaine-mon-petit-gazon-mpg.png" width="50">
        </div>
        <h4 style="text-align: center; flex-grow: 1;">Membres du projet</h4>
        <div>
            <img src="https://www.mpg-torreip.fr/wp-content/uploads/2023/08/bonus-capitaine-mon-petit-gazon-mpg.png" width="50">
        </div>
    </div>
    <div style="text-align: center; font-size: 16px;">
        <br>
        <span>BUTTIGIEG Jérémy</span>
        <span style="margin: 0 14px;">|</span>
        <span>CHEVASSIER Arnaud</span>
        <span style="margin: 0 14px;">|</span>
        <span>VAIDIE Florian</span>
        <span style="margin: 0 14px;">|</span>
        <span>MAITRALAIN Pierre</span>
    </div>
    """, unsafe_allow_html=True)

elif page == "Exploration de données":
    st.markdown("<h1 style='text-align: center;'>Exploration de données</h1>", unsafe_allow_html=True)
    st.markdown("Comme évoqué dans l'introduction la compréhension du jeu de données a d'abord demandé un échange entre MPG et le groupe pour une meilleure compréhension des intitulés des colonnes.")
    st.markdown("Une fois les données à disposition, nous avons choisi de fusionner l’ensemble des datasets relatifs à chaque championnat, afin de disposer d’un jeu de données suffisant. Puis d'effectuer un travail préparatoire avant toutes 'Dataviz' en décrivant la signification des colonnes, identifiant les types de variables, la distribution des valeurs pour certains colonnes, etc.")
    st.write("- Ligue 1 et Ligue 2 - France")
    st.write("- Liga - Espagne")
    st.write("- Premier League – Angleterre")
    st.write("- Serie A – Italie")
    def load_data():
        return pd.read_excel('exploration_des_donnees_projet_MPG.xlsx')
    data = load_data()
    st.write(data.head(20))
    st.markdown("Il s’agira de la seule source de données utilisée pour le projet. Le dataset final fusionné (avant nettoyage des données) comporte 121 colonnes et 1225 lignes.")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ-ttzY2jTs4CzBHYUdNu_BapLMQ_h8X53KDg&s" width="50">
        </div>
        <h5 style="text-align: center; flex-grow: 1;">DATASET FINAL</h5>
        <div>
            <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ-ttzY2jTs4CzBHYUdNu_BapLMQ_h8X53KDg&s" width="50">
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    def load_data1():
        return pd.read_excel('dataset_final_mpgstats_ML (1).xlsx')
    data1 = load_data1()
    st.write(data1.head(20))

elif page == "DataVizualization":
    st.markdown("<h1 style='text-align: center;'>DataVizualization</h1>", unsafe_allow_html=True)
    st.markdown("Cette section, consacrée à la Datavisualisation vise à présenter différents graphiques pour mieux appréhender le jeu dans un premier temps.")
    st.markdown("Dans un second temps les relations qu’entretiennent les variables de notre jeu de données entre elles, et plus particulièrement les facteurs clés pour la définition de la note du joueur. Ceci constitue un prémisse à l’étape de Machine Learning qui suivra.")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <img src="https://www.mpg-torreip.fr/wp-content/uploads/2023/08/bonus-tonton-pat-mon-petit-gazon-mpg.png" width="50">
        </div>
        <h6 style="text-align: center; flex-grow: 1;">[JEU] Boxplot des enchères moyennes pour les 5 championnats MPG</h6>
        <div>
            <img src="https://www.mpg-torreip.fr/wp-content/uploads/2023/08/bonus-tonton-pat-mon-petit-gazon-mpg.png" width="50">
        </div>
    """, unsafe_allow_html=True)
    dataset = pd.read_excel('dataset_final_mpgstats_all.xlsx')
    data_sup60 = dataset.loc[
    (dataset['Tps moy 1 an'] > 60) & 
    (dataset['Note 1 an'] > 5) & 
    (dataset['%Titu'] > 0.5)
    ]
    data_sup60 = data_sup60.rename(columns={'Enchère moy': 'Enchère moyenne'})
    fig = px.box(data_sup60, x="Championnat", y="Enchère moyenne", color="Championnat")
    st.plotly_chart(fig)
    st.markdown("Une des étapes clés dans la construction d'une équipe MPG est la gestion du budget (500 millions d'€ au démarrage) et donc des enchères pour acheter des joueurs. Ce graphique permet d'identifier par championnant les enchères moyennes de l'ensemble des joueurs. Par exemple nous pouvons considérer que pour la Liga, enchérir au dela de 38 millions pour un joueur est 'excessif'.")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <img src="https://www.mpg-torreip.fr/wp-content/uploads/2023/08/bonus-tonton-pat-mon-petit-gazon-mpg.png" width="50">
        </div>
        <h6 style="text-align: center; flex-grow: 1;">[JEU] Détection des "pépites" offensives avec la mise en relation "Cotation du joueur - Buts/Minutes jouées</h6>
        <div>
            <img src="https://www.mpg-torreip.fr/wp-content/uploads/2023/08/bonus-tonton-pat-mon-petit-gazon-mpg.png" width="50">
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    dataset = pd.read_excel('dataset_final_mpgstats_all.xlsx')
    attaquant = dataset.loc[
    (dataset['Poste'] == "A") & 
    (dataset['Min/But'] < 500) & 
    (dataset['Tps moy 1 an'] > 60) & 
    (dataset['Note 1 an'] > 5) & 
    (dataset['%Titu'] > 0.5)
    ]
    fig = px.scatter(
    attaquant, 
    x="Min/But", 
    y="Cote", 
    color="Joueur", 
    size='But', 
    color_discrete_sequence=px.colors.qualitative.Dark24
    )
    st.plotly_chart(fig)
    st.markdown("L'intéret de ce graphique réside principalement dans le fait de trouver des joueurs fiables et qui passent 'sous les radars' ou qui n'ont pas une grande notoriété médiatique avec un prix abordable à l'achat. C'est le cas par exemple Jean-Philippe Mateta. D'autant plus que ces joueurs peuvent l'année d'après performer encore plus.")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <img src="https://static.wixstatic.com/media/f084b3_c67c88b2bb194c8e89ee485b3fe20ecd~mv2.png/v1/fill/w_663,h_740,al_c,q_90,enc_auto/f084b3_c67c88b2bb194c8e89ee485b3fe20ecd~mv2.png" width="50">
        </div>
        <h6 style="text-align: center; flex-grow: 1;">[NOTES] Corrélation buts, tirs cadrés et corners gagnés avec note du joueur</h6>
        <div>
            <img src="https://static.wixstatic.com/media/f084b3_c67c88b2bb194c8e89ee485b3fe20ecd~mv2.png/v1/fill/w_663,h_740,al_c,q_90,enc_auto/f084b3_c67c88b2bb194c8e89ee485b3fe20ecd~mv2.png" width="50">
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.image("C:\\Users\\pmaitralain\\OneDrive - EVERNEX\\Attachments\\Bureau\\Evernex\\Formation\\Datascientest\\Soutenance\\Graph1.png")
    st.markdown("Ce graphique permet d'identifier une réelle corrélation entre les buts, les tirs cadrés et corners gagnés avec la note du joueur. Parmi nos autres graphiques présents dans le rapport final, elles constituent les variables les plus liées à la note finale du joueur sur la partie 'offensive'.")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <img src="https://static.wixstatic.com/media/f084b3_c67c88b2bb194c8e89ee485b3fe20ecd~mv2.png/v1/fill/w_663,h_740,al_c,q_90,enc_auto/f084b3_c67c88b2bb194c8e89ee485b3fe20ecd~mv2.png" width="50">
        </div>
        <h6 style="text-align: center; flex-grow: 1;">[NOTES - SPECIFIQUE GARDIEN] Etude de la relation entre les buts concédés des gardiens et la note du joueur</h6>
        <div>
            <img src="https://static.wixstatic.com/media/f084b3_c67c88b2bb194c8e89ee485b3fe20ecd~mv2.png/v1/fill/w_663,h_740,al_c,q_90,enc_auto/f084b3_c67c88b2bb194c8e89ee485b3fe20ecd~mv2.png" width="50">
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    second_viz = pd.read_excel('dataset_final_mpgstats_all.xlsx')
    gardiens = second_viz[second_viz['Poste'] == 'G']
    fig = px.scatter(
    gardiens, 
    x='Note', 
    y='But concédé', 
    facet_col='Erreur>But',  # Facettes basées sur 'Erreur>But'
    facet_col_wrap=3,        # Nombre de colonnes pour le wrapping
    labels={
        'Note': 'Note du gardien',
        'But concédé': 'Buts concédés',
        'Erreur>But': 'Erreurs menant à des buts'
    }
    )
    fig.update_layout(
    height=800,  # Ajuster la hauteur pour que les graphiques ne soient pas trop serrés
    width=1200   # Ajuster la largeur pour un meilleur affichage
    )
    st.plotly_chart(fig)
    st.markdown("Assez basiquement, nous pouvons penser qu'un gardien qui commet des erreurs aura une mauvaise note et impactera l'issue du match, idem pour les buts encaissés. Pourtant nous pouvons constater via ce visuel que les erreurs qui se sont soldées par un but ne semblement pas avoir de corrélation avec la note du joueur mais qu'un 'cleansheet' augmentera sa note. Soit un faible nombre de buts encaissés semble augmenter la note du joueur.")

elif page == "Machine Learning (ML)":
    st.markdown("<h1 style='text-align: center;'>Machine Learning (ML)</h1>", unsafe_allow_html=True)

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import LabelEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    import sklearn.metrics
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, multilabel_confusion_matrix

    dataset_ml = pd.read_csv('dataset_final_mpgstats_ML v2.csv', sep = ';', encoding = 'latin_1')

    st.write("Afin de pouvoir prédire au mieux notre variable cible 'Note du joueur', nous allons entrainer puis tester plusieurs modèles de machine learning.")
    st.write("De prime abord, notre problématique semble s'apparenter à un problème de type régression, cependant et dans l'objectif d'être le plus exhaustif possible, nous proposons d'aborder la modélisation sous le prisme de la régression ET de la classification.")
    st.write("Nous découpons ici notre travail en 3 parties :")
    st.write("- Régression")
    st.write("- Classification")
    st.write("- Améliorations et ouverture")
    st.write("Avant d'attaquer la modélisation à proprement parlée, nous retravaillons quelque peu notre dataset en supprimant certaines colonnes (Variables trop corrélées à la note et qui peuvent fausser les résultats ou variables inutiles), en remplacant les valeurs Na de la colonne 'Plonge&stop' par des 0 et en changeant le type de la variable 'Plonge&stop' en int64.")
    
    dataset_ml['Plonge&stop'] = dataset_ml['Plonge&stop'].replace(to_replace = "Na", value = 0)
    
    if st.checkbox("Affichage des dimensions du dataset après ajustements") :
        st.write(dataset_ml.shape)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <img src="https://play-lh.googleusercontent.com/B8bOYefL3rXvs1LU6yZxwAcJrfN_rqcdyRYjWhkZdg4oJ7mZIMV6pQpTfh2fOLhH188=w240-h480-rw" width="50">
        </div>
        <h5 style="text-align: center; flex-grow: 1;">Régression</h5>
        <div>
            <img src="https://play-lh.googleusercontent.com/B8bOYefL3rXvs1LU6yZxwAcJrfN_rqcdyRYjWhkZdg4oJ7mZIMV6pQpTfh2fOLhH188=w240-h480-rw" width="50">
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("Nous décidons de séparer notre dataset au format 80/20 afin d'avoir 80% dédié à l'entrainement et 20% dédié au test de nos modèles de machine learning")

    feats = dataset_ml.drop(columns = 'Note')
    target = dataset_ml['Note']
    feats['Plonge&stop'] = feats['Plonge&stop'].astype('int64')
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.20, random_state=42)

    cols_quanti = []

    for c in feats:
        cols_quanti.append(c)

    cols_quanti.remove("Poste")

    col_cat = ['Poste']

    X_train = X_train.reset_index(drop=True)  
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    if st.checkbox("Affichage des dimensions du jeu d'entrainement") :
        st.write(X_train.shape)
    if st.checkbox("Affichage des dimensions du jeu de test") :
        st.write(X_test.shape)

    scaler = StandardScaler()
    X_train[cols_quanti] = scaler.fit_transform(X_train[cols_quanti])
    X_test[cols_quanti] = scaler.transform(X_test[cols_quanti])

    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    def predreg(regression):
        if regression == 'Régression linéaire':
            reg = LinearRegression()
        elif regression == 'Arbre de décision de régression':
            reg = DecisionTreeRegressor(random_state=42, max_depth = 3)
        elif regression == 'Random forest':
            reg = RandomForestRegressor()
        reg.fit(X_train, y_train)
        return reg
    
    def regscores(reg, choice):
        if choice == 'Accuracy':
            return reg.score(X_test, y_test)
        elif choice == 'MAE':
            return mean_absolute_error(y_test, reg.predict(X_test))
        elif choice == 'MSE':
            return mean_squared_error(y_test, reg.predict(X_test), squared=True)
        elif choice == 'RMSE':
            return mean_squared_error(y_test, reg.predict(X_test), squared=False)

    
    st.write("En guise de pré-traitement, nous effectuons un scaling des données afin de les mettre à la même échelle, puis un encodage de la variable catégorielle 'Poste'")
    if st.checkbox("Affichage des dimensions du jeu d'entrainement après pré-traitement") :
        st.write(X_train.shape) 
    if st.checkbox("Affichage du jeu d'entrainement une fois le pré-traitement effectué") :
        st.dataframe(X_train.head(10))
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("Pour la régression, nous décidons d'utiliser 3 modèles de machine learning :")
    st.write("- La régression linéaire")
    st.write("- L'arbre de décision de régression")
    st.write("- La Random forest")

    st.markdown("<br>", unsafe_allow_html=True)
    choix = ['Régression linéaire', 'Arbre de décision de régression', 'Random forest']
    option = st.selectbox('Choix du modèle', choix)

    st.write('Le modèle choisi est :', option)

    st.write("Nous pouvons ensuite choisir d'afficher le métrique qui nous intéresse afin d'apprécier la performance du modèle choisi.")
    st.markdown("<br>", unsafe_allow_html=True)
    reg = predreg(option)
    display = st.radio('Quel métrique souhaitez-vous afficher ?', ('Accuracy', 'MAE', 'MSE', 'RMSE'))
    if display == 'Accuracy':
        st.write("Accuracy :", regscores(reg, display))
    elif display == 'MAE':
        st.write("MAE :", regscores(reg, display))
    elif display == 'MSE':
        st.write("MSE :", regscores(reg, display))
    elif display == 'RMSE':
        st.write("RMSE :", regscores(reg, display))
    
    linreg = LinearRegression()
    linreg.fit(X_train,y_train)
    model_dtr = DecisionTreeRegressor(random_state=42, max_depth = 3)
    model_dtr.fit(X_train, y_train)
    rfr = RandomForestRegressor()
    rfr.fit(X_train, y_train)
    y_pred_linreg = linreg.predict(X_test)
    y_pred_train_linreg = linreg.predict(X_train)
    mae_linreg_train = mean_absolute_error(y_train,y_pred_train_linreg)
    mse_linreg_train = mean_squared_error(y_train,y_pred_train_linreg,squared=True)
    rmse_linreg_train = mean_squared_error(y_train,y_pred_train_linreg,squared=False)
    mae_linreg_test = mean_absolute_error(y_test,y_pred_linreg)
    mse_linreg_test = mean_squared_error(y_test,y_pred_linreg,squared=True)
    rmse_linreg_test = mean_squared_error(y_test,y_pred_linreg,squared=False)
    y_pred_decision_tree = model_dtr.predict(X_test)
    y_pred_train_decision_tree = model_dtr.predict(X_train)
    mae_decision_tree_train = mean_absolute_error(y_train,y_pred_train_decision_tree)
    mse_decision_tree_train = mean_squared_error(y_train,y_pred_train_decision_tree,squared=True)
    rmse_decision_tree_train = mean_squared_error(y_train,y_pred_train_decision_tree,squared=False)
    mae_decision_tree_test = mean_absolute_error(y_test,y_pred_decision_tree)
    mse_decision_tree_test = mean_squared_error(y_test,y_pred_decision_tree,squared=True)
    rmse_decision_tree_test = mean_squared_error(y_test,y_pred_decision_tree,squared=False)
    y_pred_random_forest = rfr.predict(X_test)
    y_pred_random_forest_train = rfr.predict(X_train)
    mae_random_forest_train = mean_absolute_error(y_train,y_pred_random_forest_train)
    mse_random_forest_train = mean_squared_error(y_train,y_pred_random_forest_train,squared=True)
    rmse_random_forest_train = mean_squared_error(y_train,y_pred_random_forest_train,squared=False)
    mae_random_forest_test = mean_absolute_error(y_test,y_pred_random_forest)
    mse_random_forest_test = mean_squared_error(y_test,y_pred_random_forest,squared=True)
    rmse_random_forest_test = mean_squared_error(y_test,y_pred_random_forest,squared=False)
    
    # Creation d'un dataframe pour comparer les metriques des deux algorithmes 
    data = {'MAE train': [mae_linreg_train, mae_decision_tree_train, mae_random_forest_train],
            'MAE test': [mae_linreg_test, mae_decision_tree_test, mae_random_forest_test],
            'MSE train': [mse_linreg_train, mse_decision_tree_train,mse_random_forest_train],
            'MSE test': [mse_linreg_test, mse_decision_tree_test,mse_random_forest_test],
            'RMSE train': [rmse_linreg_train, rmse_decision_tree_train, rmse_random_forest_train],
            'RMSE test': [rmse_linreg_test, rmse_decision_tree_test, rmse_random_forest_test]}
    
    # Creer DataFrame
    df_metriques = pd.DataFrame(data, index = ['Linear Regression', 'Decision Tree', 'Random Forest '])
    if st.checkbox("Affichage des métriques pour l'ensemble des modèles") :
        st.dataframe(df_metriques.head())
    st.markdown("""
    <h6 style="text-align: center;">Conclusion</h6>
    """, unsafe_allow_html=True)
    st.write("Après analyse des métriques, nous sommes en capacité de confirmer que la régression linéaire semble être le modèle le plus partinent pour prédire la note globale des joueurs et ainsi composer la meilleure équipe possible pour la prochaine saison")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <img src="https://pic.clubic.com/v1/images/2084961/raw" width="50">
        </div>
        <h5 style="text-align: center; flex-grow: 1;">Classification</h5>
        <div>
            <img src="https://pic.clubic.com/v1/images/2084961/raw" width="50">
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("Comme pour la régression, nous séparons notre dataset au format 80/20 afin d'avoir 80% dédié à l'entrainement et 20% dédié au test de nos modèles de machine learning")

    X = dataset_ml.drop(columns = 'Note')
    y = dataset_ml['Note']
    X['Plonge&stop'] = X['Plonge&stop'].astype('int64')
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y, test_size = 0.20, random_state=42)

    cols_quanti_clf = []

    for c in X:
        cols_quanti_clf.append(c)

    cols_quanti_clf.remove("Poste")

    col_cat_clf = ['Poste']

    X_train_clf = X_train_clf.reset_index(drop=True)  
    X_test_clf = X_test_clf.reset_index(drop=True)
    y_train_clf = y_train_clf.reset_index(drop=True)
    y_test_clf = y_test_clf.reset_index(drop=True)

    if st.checkbox("Affichage des dimensions de notre jeu d'entrainement") :
        st.write(X_train_clf.shape)
    if st.checkbox("Affichage des dimensions de notre jeu de test") :
        st.write(X_test_clf.shape)

    st.write("Nous encodons la variable 'Poste' et scalons les données afin de les avoir à la même échelle")
    st.write("Pour la classification, nous choisissons d'encoder notre variable 'Poste' en remplacant les postes par des chiffres allant de 0 à 3")

    scaler_clf = StandardScaler()
    X_train_clf[cols_quanti_clf] = scaler_clf.fit_transform(X_train_clf[cols_quanti_clf])
    X_test_clf[cols_quanti_clf] = scaler_clf.transform(X_test_clf[cols_quanti_clf])

    X_train_clf['Poste'] = X_train_clf['Poste'].apply(lambda x: 0 if x == 'DC' or x == 'DL' else (1 if x == 'MO' or x == 'MD' else (2 if x == 'G' else 3)))
    X_test_clf['Poste'] = X_test_clf['Poste'].apply(lambda x: 0 if x == 'DC' or x == 'DL' else (1 if x == 'MO' or x == 'MD' else (2 if x == 'G' else 3)))

    if st.checkbox("Affichage du jeu d'entrainement après les pré-traitements") :
        st.dataframe(X_train_clf.head())

    st.write("Concernant notre variable cible 'Note', nous la répartissons dans les classes suivantes :")
    st.write("- Si le score est entre 0 et 5.18, alors le score est 'mauvais'")
    st.write("- Si le score est entre 5.18 et 5.43, alors le score est 'moyen'")
    st.write("- Si le score est entre 5.43 et 10, alors le score est 'bon'")

    y_train_discret = pd.cut(y_train_clf, bins=[0, 5.18, 5.43, 10], labels=['mauvais', 'moyen', 'bon'])
    y_test_discret = pd.cut(y_test_clf, bins=[0, 5.18, 5.43, 10], labels=['mauvais', 'moyen', 'bon'])

    if st.checkbox("Affichage du jeu d'entrainement de la variable cible après répartition") :
        st.dataframe(y_train_discret.head())
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("Pour la classification, nous décidons d'utiliser 3 modèles de machine learning :")
    st.write("- La régression logistique")
    st.write("- Le support vector machine SVM")
    st.write("- Les KNeighbors")
    st.markdown("<br>", unsafe_allow_html=True)
    def predclf(classification):
        if classification == 'Régression logistique':
            clf = LogisticRegression()
        elif classification == 'Support vector machine SVM':
            clf = SVC(gamma=0.01, kernel='poly')
        elif classification == 'KNeighbors':
            clf = KNeighborsClassifier()
        clf.fit(X_train_clf, y_train_discret)
        return clf
    
    def clfscores(clf, clfchoice):
        if clfchoice == 'Accuracy':
            return clf.score(X_test_clf, y_test_discret)
        elif clfchoice == 'Rapport de classification':
            return classification_report(y_test_discret, clf.predict(X_test_clf))
        elif clfchoice == 'Matrice de confusion':
            return pd.crosstab(y_test_discret, clf.predict(X_test_clf))
        
    choixclf = ['Régression logistique', 'Support vector machine SVM', 'KNeighbors']
    st.markdown("<br>", unsafe_allow_html=True)
    optionclf = st.selectbox('Choix du modèle', choixclf)
    st.write('Le modèle choisi est :', optionclf)

    st.write("Nous pouvons ensuite choisir d'afficher le métrique qui nous intéresse afin d'apprécier la performance du modèle choisi.")
    clf = predclf(optionclf)
    st.markdown("<br>", unsafe_allow_html=True)
    
    displayclf = st.radio('Quel métrique souhaitez-vous afficher ?', ('Accuracy', 'Rapport de classification', 'Matrice de confusion'))
    if displayclf == 'Accuracy':
        st.write("Accuracy :", clfscores(clf, displayclf))
    elif displayclf == 'Rapport de classification':
        st.write("Rapport de classification :", clfscores(clf, displayclf))
    elif displayclf == 'Matrice de confusion':
        st.write("Matrice de confusion :", clfscores(clf, displayclf))

    st.markdown("""
    <h6 style="text-align: center;">Conclusion</h6>
    """, unsafe_allow_html=True)
    st.write("Après analyse des métriques, nous sommes en capacité de confirmer que la régression logistique semble être le modèle le plus partinent pour prédire la note globale des joueurs et ainsi composer la meilleure équipe possible pour la prochaine saison")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <img src="https://europarchive.org/wp-content/uploads/2019/07/Regles-Mon-Petit-Gazon-1200x958.jpg" width="50">
        </div>
        <h5 style="text-align: center; flex-grow: 1;">Amélioration et ouverture</h5>
        <div>
            <img src="https://europarchive.org/wp-content/uploads/2019/07/Regles-Mon-Petit-Gazon-1200x958.jpg" width="50">
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("En termes d’amélioration, nous pourrions déployer une stratégie plus élaborée de récupération des données. Cela  nous permettrait d’entraîner le modèle retenu sur un dataset “mouvant” (en opposition au dataset “figé” utilisé) et d’approfondir également l’aspect forme du joueur pour chercher à prédire la note du prochain match et non pas la note globale. Et dans l'optique de proposer un outil capable de sélection automatiquement les meilleurs joueurs, nous pourrions aussi conserver la variable 'joueur' que nous pourrions mettre en index. Cela constituerait un réel atout pour les utilisateurs")
