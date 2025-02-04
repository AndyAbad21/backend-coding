from django.shortcuts import render
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import joblib
import numpy as np
import pandas as pd
from io import BytesIO
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import openai
import os

# Configurar la API Key de OpenAI de forma directa en el código
openai.api_key = ""  # Reemplaza "TU_API_KEY_AQUI" por tu clave real

# Cargar el modelo previamente entrenado
modelo = joblib.load("/home/andy/Escritorio/Universidad/Aprendizaje_Automatico/ProyectoFinal/rf/rfcv2.pkl")

# Inicializar SHAP explainer
explainer = shap.Explainer(modelo)

# Etiquetas de las clases
class_labels = {0: "Sin Defectos", 1: "Con Defectos"}

FEATURE_NAMES = [
    "LOC_BLANK", "BRANCH_COUNT", "LOC_CODE_AND_COMMENT", "LOC_COMMENTS",
    "CYCLOMATIC_COMPLEXITY", "DESIGN_COMPLEXITY", "ESSENTIAL_COMPLEXITY",
    "LOC_EXECUTABLE", "HALSTEAD_CONTENT", "HALSTEAD_DIFFICULTY", "HALSTEAD_EFFORT",
    "HALSTEAD_ERROR_EST", "HALSTEAD_LENGTH", "HALSTEAD_LEVEL",
    "HALSTEAD_PROG_TIME", "HALSTEAD_VOLUME", "NUM_OPERANDS",
    "NUM_OPERATORS", "NUM_UNIQUE_OPERANDS", "NUM_UNIQUE_OPERATORS",
    "LOC_TOTAL"
]

class PrediccionView(APIView):
    def post(self, request):
        datos = request.data
        try:
            # Extraer características
            features = np.array([[datos[feature] for feature in FEATURE_NAMES]])
            features_df = pd.DataFrame(features, columns=FEATURE_NAMES)

            # Realizar la predicción
            prediccion = modelo.predict(features_df)

            # Generar explicaciones con SHAP
            shap_values = explainer(features_df)

            target_class = 1
            expected_value = explainer.expected_value[target_class]
            class_shap_values = shap_values.values[0, :, target_class]

            # Generar gráfico de SHAP
            shap.force_plot(expected_value, class_shap_values, features_df.iloc[0], matplotlib=True)
            buffer = BytesIO()
            plt.savefig(buffer, format="png", bbox_inches='tight', pad_inches=0)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()
            plt.clf()

            # Preparar explicaciones para la tabla
            explicaciones = []
            for feature, value, shap_value in zip(features_df.columns, features_df.iloc[0], class_shap_values):
                explicacion = {
                    "caracteristica": feature,
                    "valor": value,
                    "impacto": shap_value,
                    "interpretacion": self.generar_interpretacion(shap_value)
                }
                explicaciones.append(explicacion)

            # Generar explicación usando OpenAI con la nueva sintaxis
            explicacion_chatgpt = self.generar_explicacion_chatgpt(prediccion[0], explicaciones)

            return Response({
                "prediction": class_labels[prediccion[0]],
                "image": "data:image/png;base64," + image_base64,
                "explicaciones": explicaciones,
                "explicacion_chatgpt": explicacion_chatgpt
            }, status=status.HTTP_200_OK)

        except KeyError as e:
            return Response({"error": f"Falta la característica: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": f"Error interno: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @staticmethod
    def generar_interpretacion(valor):
        """Genera una interpretación basada en el impacto del valor SHAP."""
        if valor > 0.2:
            return "Aumenta significativamente la probabilidad de defectos."
        elif valor > 0:
            return "Aumenta ligeramente la probabilidad de defectos."
        elif valor < -0.2:
            return "Reduce significativamente la probabilidad de defectos."
        elif valor < 0:
            return "Reduce ligeramente la probabilidad de defectos."
        else:
            return "No tiene un impacto notable en la predicción."

    def generar_explicacion_chatgpt(self, prediccion, explicaciones):
        """Envía la información de la predicción a ChatGPT para generar una explicación más detallada."""
        prompt = f"""
        El modelo de predicción ha determinado que el módulo de software es: {'Con Defectos' if prediccion == 1 else 'Sin Defectos'}.
        
        Aquí está la tabla con los factores que influenciaron la predicción:
        
        | Característica | Valor | Impacto | Interpretación |
        |--------------|--------|---------|----------------|
        """
        for exp in explicaciones:
            prompt += f"| {exp['caracteristica']} | {exp['valor']} | {exp['impacto']:.2f} | {exp['interpretacion']} |\n"

        prompt += "\nExplica en lenguaje sencillo por qué el modelo ha determinado este resultado y cómo estos factores afectan la predicción, pero quiero que solo tomes en cuenta las 10 variables mas importantes"

        # Nueva sintaxis para la API de OpenAI v1.0.0+
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "Eres un experto en software explicando predicciones de defectos."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1
            )

            if "choices" in response and len(response["choices"]) > 0:
                return response["choices"][0]["message"]["content"]
            else:
                return "No se pudo generar una explicación en este momento."

        except Exception as e:
            return f"Error al generar la explicación: {str(e)}"
