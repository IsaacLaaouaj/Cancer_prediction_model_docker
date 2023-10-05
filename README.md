# Cancer_prediction_model_docker
Construcción y ejecución de un modelo de aprendizaje supervisado muy básico de aprendizaje automático (Machine Learning), a partir de un conjunto de datos (dataset) que representan el carácter (Beningno o Maligno) del tumor cancerígeno detectado en 569 pacientes. Lo que tratamos de predecir es si el cáncer de los 569 pacientes es Benigno o Maligno, en base a una serie de características definidas en el dataset.

Para ello implementaremos los datos en dos modelos de clasificación mediante la librería scikit-learn, con estas técnicas de aprendizaje automático: Linear Discriminant Analysis y Neural Networks multilayer perceptron.

Finalmente se implementará el modelo en una imagen de docker llamada “docker-ml-model”.

Se puede descargar y aplicar mediante docker pull isaac31120/cancer_prediction_docker además esto asegurará que la última versión de la imagen esté disponible localmente en tu sistema.


