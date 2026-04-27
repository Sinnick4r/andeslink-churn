# CONCLUSIONES

El analisis exploratorio muestra que el dataset  no presenta valores nulos, posee variables numéricas y categóricas relevantes para el caso de churn, y el target presenta un desbalance moderado de 66% no churn contra 34% churn. Por este motivo, la evaluación del modelo no debería basarse, en principio, unicamente en accuracy, sino incluir métricas como F1-score, recall, precision, ROC-AUC y matriz de confusión.

Las variables numéricas presentan correlaciones individuales débiles con churn, lo que indica que el abandono no está determinado por una única variable aislada. Las señales más relevantes observadas son la menor antigüedad del cliente, mayor cantidad de tickets de soporte, pagos atrasados y mayor cargo mensual.

En variables categóricas, `contract_type` muestra la mayor discriminación: tasa de churn de 47.5% en contratos mensuales vs 11.6% en bianuales (diferencia de ~ 36%). `internet_service` también discrimina fuertemente: servicio móvil con 50.9% de churn vs fibra con 26.8%. En contraste, `region` no aporta señal discriminante (todas las regiones entre 33.5% y 34.5%), aunque se incluirá en el pipeline por si captura interacciones no lineales con otras variables.

Los resultados de este EDA justifican, en mi opinion, la creación de dos features derivadas: `charges_per_month` (gasto real histórico promedio, reemplaza a `total_charges`) y `tickets_per_year` (tickets de soporte anualizados por antigüedad, normaliza la señal de `support_tickets` para clientes con distintos tiempos en la empresa).

Justificación de `charges_per_month`:

En primer lugar, `total_charges` se aproxima algebraicamente al producto `tenure_months × monthly_charge` (correlación 0.87, ratio mediana = 1.000). Su inclusión puede introducier multicolinealidad sin agregar información independiente y relevante. Por eso desde ya se elimina del futuro pipeline y se reemplaza por la feature derivada `charges_per_month = total_charges / tenure_months`, que captura el gasto real promedio mensual. Esta feature nueva puede correlacionar fuertemente con `monthly_charge` dado el comportamiento algebraico del dataset; su utilidad real se va a evaluar por importancia de features post-entrenamiento.

Justificación de `tickets_per_year`:

Por otro lado, se justifica la creación de una segunda feature derivada: `tickets_per_year = support_tickets / (tenure_months / 12)`, que normaliza la señal de soporte por la antigüedad del cliente y permite comparar clientes con distintos tiempos en la empresa en igualdad de condiciones. SE puede hacer con una proteccion para clientes con baja antigüedad (clip en 6 meses mínimo) para evitar inflación artificial de la tasa anualizada en clientes recientes.

*Siguiente etapa: se va a desarrollar una clasificación binaria con: separación estratificada de datos (64/16/20), codificación OHE para las 4 variables categóricas de baja cardinalidad, escalado estándar para numericas, `class_weight='balanced'` por el desbalance moderado, y comparación de tres modelos supervisados con tracking en MLflow. El objetivo es seleccionar un modelo serializable y reproducible apto para ser consumido por una API de inferencia. Ver PLAN.md