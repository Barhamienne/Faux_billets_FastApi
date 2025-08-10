# ---------------------
# API FastAPI : Détection de faux billets
# ---------------------
from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
import io

# Charger le modèle et le normaliseur
modele = joblib.load("meilleur_modele.joblib")
normaliseur = joblib.load("normaliseur.joblib")

# Création de l'application FastAPI
app = FastAPI(title="API Détection de Faux Billets",
              description="API pour prédire si un billet est vrai ou faux à partir de ses dimensions",
              version="1.0")

@app.post("/predire")
async def predire_billets(fichier: UploadFile = File(...)):
    """
    Endpoint pour prédire si les billets sont vrais ou faux à partir d'un fichier CSV
    Le fichier ne doit pas contenir la colonne 'is_genuine'
    """
    try:
        # Lire le fichier CSV envoyé
        contenu = await fichier.read()
        df = pd.read_csv(io.BytesIO(contenu), sep=None, engine="python")
        
        # Normalisation des données
        X_normalise = normaliseur.transform(df)

        # Prédictions
        predictions = modele.predict(X_normalise)

        # Conversion en texte (optionnel)
        etiquettes = ["Vrai" if p == 1 else "Faux" for p in predictions]

        # Retourner résultats en JSON
        return {
            "nombre_billets": len(predictions),
            "predictions_numeriques": predictions.tolist(),
            "predictions_texte": etiquettes
        }

    except Exception as e:
        print("Erreur API :", str(e))
        return {"erreur": str(e)}

