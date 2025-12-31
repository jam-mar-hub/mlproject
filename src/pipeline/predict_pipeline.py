import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Chemins vers les fichiers (modèle et preprocessor)
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            
            # 1. Transformation des données
            data_scaled = preprocessor.transform(features)
            
            # CORRECTION CRITIQUE : Gestion Matrice Sparse
            if hasattr(data_scaled, "toarray"):
                data_scaled = data_scaled.toarray()

            # --- CALCUL DU SCORE DE RISQUE (Hazard) ---
            risk_score = model.predict(data_scaled)[0]

            # --- CALCUL DE LA PROBABILITE DE SURVIE ---
            surv_funcs = model.predict_survival_function(data_scaled)
            
            # On fixe un horizon de temps (Ici : 3 ans)
            # Tu peux changer ce chiffre (ex: 1, 2, 5) selon tes besoins
            TIME_HORIZON = 3 
            
            # On évalue la fonction au temps t = 3
            survival_prob = surv_funcs[0](TIME_HORIZON)
            percentage = survival_prob * 100

            # --- RESULTAT COMBINÉ ---
            # On renvoie une phrase contenant les deux infos pour l'affichage
            final_result = (
                f"Score de Risque : {risk_score:.2f}  |  "
                f"Probabilité de survie à {TIME_HORIZON} ans : {percentage:.2f} %"
            )
            
            # On renvoie une liste car app.py attend results[0]
            return [final_result]
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
        BM_BLAST: float,
        WBC: float,
        ANC: float,
        MONOCYTES: float,
        HB: float,
        PLT: float,
        Nmut: float,
        CENTER: str):

        self.BM_BLAST = BM_BLAST
        self.WBC = WBC
        self.ANC = ANC
        self.MONOCYTES = MONOCYTES
        self.HB = HB
        self.PLT = PLT
        self.Nmut = Nmut
        self.CENTER = CENTER

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "BM_BLAST": [self.BM_BLAST],
                "WBC": [self.WBC],
                "ANC": [self.ANC],
                "MONOCYTES": [self.MONOCYTES],
                "HB": [self.HB],
                "PLT": [self.PLT],
                "Nmut": [self.Nmut],
                "CENTER": [self.CENTER]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)