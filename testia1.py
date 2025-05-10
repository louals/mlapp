import streamlit as st 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score , f1_score
from sklearn.metrics import mean_absolute_error , mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression , LogisticRegression 
from sklearn.ensemble import GradientBoostingRegressor , RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor , DecisionTreeClassifier 
from sklearn.svm import SVR 
from sklearn.neighbors import KNeighborsClassifier



st.set_page_config(page_title="ML App", layout="wide")
st.title(" ML App - Classification & Regression")

with st.container():
    file = st.file_uploader(" Uploade ton fichier CSV", type=["csv"])
    if file:
        with st.spinner("Chargement des données... "):
            data = pd.read_csv(file)
            st.success("Données chargées avec succès ")
            st.write(" Aperçu des données:")
            st.dataframe(data.head())

        columns = data.columns.tolist()
        target = st.selectbox(" Sélectionnez la colonne cible (output)", columns)
        task = st.radio(" Choisissez la tâche :", ["Classification", "Regression"], horizontal=True)

        if target:
            with st.spinner("Préparation et encodage des données..."):
                X = data.drop(columns=[target])
                Y = data[target]

                for col in X.select_dtypes(include=['object']).columns:
                    X[col] = LabelEncoder().fit_transform(X[col])

                if Y.dtype == 'object':
                    Y = LabelEncoder().fit_transform(Y)

                x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=29)

            st.markdown("---")
            st.subheader(" Modèles et Évaluation")

            with st.spinner("Entraînement des modèles..."):
                result = []

                if task == "Classification":
                    models = {
                        "Decision Tree": DecisionTreeClassifier(),
                        "Logistic Regression": LogisticRegression(),
                        "KNN": KNeighborsClassifier(),
                        "Random Forest": RandomForestClassifier() 
                    }

                    for name, model in models.items():
                        model.fit(x_train , y_train)
                        y_predit = model.predict(x_test)
                        result.append({
                            "Modèle": name,
                            "Accuracy": round(accuracy_score(y_test,y_predit),2),
                            "Precision": round(precision_score(y_test,y_predit,average='macro'),2),
                            "Recall": round(recall_score(y_test,y_predit,average='macro'),2),
                            "F1 Score": round(f1_score(y_test,y_predit,average='macro'),2)
                        })

                else:  # Regression
                    models = {
                        "Linear Regression": LinearRegression(),
                        "Gradient Boosting": GradientBoostingRegressor(),
                        "Decision Tree": DecisionTreeRegressor(),
                        "SVR": SVR()
                    }

                    for name, model in models.items():
                        model.fit(x_train, y_train)
                        y_pred = model.predict(x_test)
                        result.append({
                            "Modèle": name,
                            "MAE": round(mean_absolute_error(y_test,y_pred),2),
                            "MSE": round(mean_squared_error(y_test,y_pred),2),
                            "R2 Score": round(r2_score(y_test, y_pred), 2)
                        })

                st.dataframe(pd.DataFrame(result),  use_container_width=True)

            
            st.markdown("---")
            st.subheader(" Prédiction ")

            user_input = {}
            cols = st.columns(len(X.columns))

            for i, col in enumerate(X.columns):
                with cols[i % len(cols)]:
                    val = st.number_input(f"{col}", value=float(X[col].mean()))
                    user_input[col] = val

            input_df = pd.DataFrame([user_input])

            model_name = st.selectbox("Choisissez un modèle pour la prédiction", list(models.keys()))
            model = models[model_name]

            if st.button("Faire la prédiction"):
                with st.spinner("Prédiction en cours..."):
                    predictions = model.predict(input_df)[0]
                st.success(f" Prédiction : {predictions}")
