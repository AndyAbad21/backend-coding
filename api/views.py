from django.shortcuts import render
import shap
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo
import matplotlib.pyplot as plt
import base64
import joblib
import numpy as np
import pandas as pd
from io import BytesIO
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

# Cargar el modelo previamente entrenado
modelo = joblib.load("/home/andy/Escritorio/Universidad/Aprendizaje_Automatico/ProyectoFinal/rf/rfcv2.pkl")

# Inicializar SHAP explainer
explainer = shap.Explainer(modelo)

# Etiquetas de las clases
class_labels = {0: "Sin Defectos", 1: "Con Defectos"}

# Nueva lista de características que espera el modelo
FEATURE_NAMES = [
    "LOC_BLANK", "BRANCH_COUNT", "LOC_CODE_AND_COMMENT", "LOC_COMMENTS",
    "CYCLOMATIC_COMPLEXITY", "DESIGN_COMPLEXITY", "ESSENTIAL_COMPLEXITY",
    "LOC_EXECUTABLE", "HALSTEAD_CONTENT", "HALSTEAD_DIFFICULTY","HALSTEAD_EFFORT",
    "HALSTEAD_ERROR_EST", "HALSTEAD_LENGTH", "HALSTEAD_LEVEL",
    "HALSTEAD_PROG_TIME", "HALSTEAD_VOLUME", "NUM_OPERANDS",
    "NUM_OPERATORS", "NUM_UNIQUE_OPERANDS", "NUM_UNIQUE_OPERATORS",
    "LOC_TOTAL"
]



class PrediccionView(APIView):
    def post(self, request):
        datos = request.data
        try:
            # Extraer las características desde el request
            features = np.array([[datos[feature] for feature in FEATURE_NAMES]])

            # Convertir a DataFrame de pandas con los nuevos nombres de características
            features_df = pd.DataFrame(features, columns=FEATURE_NAMES)

            # Realizar la predicción
            prediccion = modelo.predict(features_df)

            # Generar explicaciones con SHAP
            shap_values = explainer(features_df)

            # Seleccionar la clase objetivo
            target_class = 1
            expected_value = explainer.expected_value[target_class]
            class_shap_values = shap_values.values[0, :, target_class]

            # Generar gráfico estático con Matplotlib
            shap.force_plot(
                expected_value,
                class_shap_values,
                features_df.iloc[0].values,
                matplotlib=True
            )

            # Guardar el gráfico en un buffer como PNG
            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

            # Limpiar el gráfico de Matplotlib
            plt.clf()

            # Preparar las explicaciones para la tabla
            explicaciones = []
            for feature, value, shap_value in zip(
                features_df.columns, features_df.iloc[0].values, class_shap_values
            ):
                explicacion = {
                    "caracteristica": feature,
                    "valor": value,
                    "impacto": shap_value,
                    "interpretacion": self.generar_interpretacion(shap_value)
                }
                explicaciones.append(explicacion)

            return Response({
                "prediction": class_labels[prediccion[0]],
                "image": "data:image/png;base64," + image_base64,
                "explicaciones": explicaciones
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
