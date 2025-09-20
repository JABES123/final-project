### Comparación de Modelos de Deep Learning para la Detección de Patologías en Radiografías de Tórax

---

#### Resumen

Este estudio compara el rendimiento de cuatro arquitecturas de redes neuronales convolucionales de última generación (DenseNet-121, ViT, ResNet y EfficientNet). Los modelos se evaluaron en la detección de múltiples patologías en radiografías de tórax, utilizando el conjunto de datos **NIH ChestX-ray**. Los modelos se pre-entrenaron y luego se ajustaron (fine-tuned) durante 10 épocas por fase para una clasificación multietiqueta. Los resultados demuestran la viabilidad del **transfer learning** para esta tarea, identificando los modelos con el mejor rendimiento en métricas clave como el **AUC-ROC macro**, la precisión y el F1-score.

---

#### Introducción y Motivación

El diagnóstico de enfermedades a partir de radiografías de tórax es un pilar fundamental en la práctica médica. Sin embargo, la interpretación de estas imágenes es compleja y propensa a errores. El **deep learning** ofrece una oportunidad para desarrollar sistemas de apoyo al diagnóstico que pueden mejorar la precisión, reducir el tiempo de análisis y democratizar el acceso a diagnósticos de alta calidad. El principal desafío es la **clasificación multietiqueta**, donde una sola imagen puede contener múltiples patologías simultáneamente. Este trabajo se motiva por la necesidad de comparar rigurosamente varias arquitecturas de vanguardia para determinar cuál es la más adecuada para este tipo de tarea.

---

#### Estado del Arte

El uso de redes neuronales convolucionales (CNN) para el análisis de imágenes médicas ha crecido exponencialmente. El enfoque de **transfer learning**, que utiliza modelos pre-entrenados en grandes conjuntos de datos como ImageNet, ha demostrado ser particularmente efectivo. El estudio se centra en los siguientes modelos:
*   **DenseNet-121**: Se caracteriza por sus conexiones densas, lo que facilita la reutilización de características y mitiga la desaparición del gradiente.
*   **ResNet-50**: Conocida por sus "conexiones residuales" que permiten el entrenamiento de redes muy profundas al resolver el problema del gradiente decreciente.
*   **ViT (Vision Transformer)**: Adapta la arquitectura de los transformadores al dominio de la visión, tratando las imágenes como una secuencia de "parches" y utilizando mecanismos de auto-atención para capturar dependencias globales.
*   **EfficientNet**: Se enfoca en optimizar la red de manera más eficiente a través de una técnica de escalado compuesto.

La elección de la función de pérdida **Focal Loss** es un punto clave, ya que está diseñada para abordar el desequilibrio de clases, un problema común en los conjuntos de datos médicos.

---

#### Metodología

El enfoque experimental se basa en un flujo de trabajo estándar de **machine learning** y **deep learning**.

##### Dataset y Preprocesamiento

Se utilizó el conjunto de datos **NIH ChestX-ray**, que contiene más de 100,000 radiografías con etiquetas para 14 patologías. Los datos se dividieron en conjuntos de entrenamiento (70%), validación (15%) y prueba (15%). Se realizaron las siguientes transformaciones de preprocesamiento:
*   Redimensión a 224x224 píxeles.
*   Normalización con medias y desviaciones estándar de ImageNet.
*   **Aumento de datos** (data augmentation): se aplicó un volteo horizontal aleatorio solo en el conjunto de entrenamiento para mejorar la generalización del modelo.

##### Modelos

Se evaluaron cuatro arquitecturas: **DenseNet-121**, **ViT**, **ResNet-50** y **EfficientNet-B0**. Todos los modelos se inicializaron con pesos pre-entrenados en ImageNet. El entrenamiento se realizó en dos fases:
1.  **Entrenamiento de la cabeza**: se congelaron las capas base y solo se entrenó el nuevo clasificador durante 10 épocas.
2.  **Ajuste fino (fine-tuning)**: todas las capas del modelo se descongelaron y se entrenó la red completa durante 10 épocas adicionales con una tasa de aprendizaje más baja.

##### Entrenamiento y Métricas

Se utilizó **Focal Loss** como función de pérdida para manejar el desequilibrio de clases. El optimizador fue **Adam**, y se empleó un `scheduler ReduceLROnPlateau` para ajustar la tasa de aprendizaje de manera adaptativa. El rendimiento de cada modelo se midió en el conjunto de validación utilizando las siguientes métricas:
*   **AUC-ROC (macro)**.
*   **MAE (Error Absoluto Medio)**.
*   **R2 (Coeficiente de Determinación)**.
*   **Accuracy y F1-score (macro)**.

---

#### Resultados Experimentales

| Modelo | AUC-ROC | MAE | R2 | Accuracy | F1-score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DenseNet-121** | 0.841 | 0.105 | 0.301 | 0.902 | 0.795 |
| **ViT** | 0.808 | 0.119 | 0.224 | 0.890 | 0.760 |
| **ResNet-50** | 0.815 | 0.115 | 0.267 | 0.893 | 0.771 |
| **EfficientNet** | 0.830 | 0.103 | 0.315 | 0.899 | 0.785 |

Los resultados muestran que **DenseNet-121** mantiene su ventaja en las métricas clave, logrando un mejor **AUC-ROC** y **F1-score**. Sin embargo, **EfficientNet** demostró una ligera superioridad en las métricas de error (MAE y R2).

---

#### Discusión y Conclusiones

Los resultados sugieren que las arquitecturas basadas en CNN, en particular **DenseNet-121**, siguen siendo extremadamente competitivas para tareas de clasificación de imágenes médicas. Su diseño, que promueve la reutilización de características, parece ser particularmente beneficioso para la detección de múltiples patologías. Aunque el rendimiento de ViT fue inferior, su promesa como arquitectura de vanguardia sigue siendo relevante, y podría requerir más datos o una configuración de hiperparámetros diferente para alcanzar su máximo potencial. En conclusión, este estudio establece a **DenseNet-121** como un modelo de referencia para la clasificación multietiqueta en radiografías de tórax.
