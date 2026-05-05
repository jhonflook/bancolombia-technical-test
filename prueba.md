PRUEBA TÉCNICA – DÉBITOS RECURRENTES

El objetivo de esta prueba es evaluar, de manera rigurosa e integral, la capacidad del candidato para entender un problema de negocio, traducirlo a un planteamiento analítico, diseñar una arquitectura de datos escalable, desarrollar un modelo y proponer su operacionalización para respaldar procesos de toma de decisiones, siguiendo buenas prácticas.
Se recomienda revisar cuidadosamente cada uno de los enunciados, documentar de forma transparente los supuestos adoptados y presentar soluciones fundamentadas y reproducibles.

Contexto
El banco administra actualmente una cartera activa de clientes clasificados según distintos segmentos, niveles de riesgo y perfiles comerciales. Cada cliente posee una variedad de productos financieros, tanto vigentes como recientemente adquiridos, entre los cuales se incluyen tarjetas de crédito, créditos de consumo, obligaciones específicas y otros productos derivados de su relación con la entidad.
La Gerencia de Inteligencia de Originación y Cobranza busca optimizar la asignación de clientes en mora temprana (1–30 días) mediante la identificación de obligaciones con alta probabilidad de pago exclusivamente por débito recurrente. Con el propósito de reducir costos de gestión de cobranza, mejorar la eficiencia operativa y maximizar el ahorro financiero.
Así que, buscando garantizar una adecuada gestión de esta cartera, es necesario integrar y analizar fuentes de información, cuyo entendimiento en conjunto permitirá evaluar las condiciones financieras de los clientes y generar insumos de valor para la toma de decisiones estratégicas dentro de la organización.

Bases de datos
Para el desarrollo de la prueba técnica, se entregan 6 archivos en formato csv. Estas fuentes de información deberán ser integradas, transformadas y analizadas para el cumplimiento del ejercicio. 
Los archivos proporcionados son:
Canales: Contiene el detalle de la información transaccional, cantidades y montos en canales como APPs, Sucursal Virtual, retiros, sucursal física, etc
Moras: Incluye moras de las obligaciones en saldos y días
Excedentes: Expone los pagos adicionales en monto que se dan sobre la cuota, y los porcentajes correspondientes al valor de la cuota.
Gestiones: Corresponde a la gestión de cobranza que se le ha realizado previamente a los clientes (acuerdos de pagos o el rechazo del cliente en la realización de estos acuerdos)
Pagos (tanque): Contiene la agrupación de los pagos que ha realizado el cliente a sus obligaciones
Clientes: Incluye la base de clientes a analizar, junto con el valor de la variable respuesta real (var_rta=1: La obligación presento pagos por débito, var_rta=0: La obligación presento pagos por otros canales)
Nota: Tenga presente que las fuentes de información entregadas incluyen el detalle correspondiente al nivel cliente–obligación.

Requerimientos
Diseñar, entrenar y desplegar un modelo de clasificación binaria que estime la probabilidad de pago por obligación únicamente por débito recurrente, definiendo un umbral operativo de recurrencia ≥ 40%, garantizando robustez estadística y alineación con decisiones de negocio.
Variable respuesta = 1, cuando la obligación presenta pagos únicamente por débito y una recurrencia ≥ 40%
Variable respuesta = 0, cuando existen pagos por otros canales o combinaciones
Para lo anterior, construya un código utilizando la herramienta de su preferencia (orquestador, base calendarizable, notebook) que permita ejecutar de manera integrada el procesamiento de datos, la construcción del modelo y su posterior evaluación. El desarrollo debe centrarse en la correcta lectura, transformación y análisis de las fuentes de información entregadas, aplicando buenas prácticas de ingeniería de datos, modelado analítico y la definición explícita de supuestos de negocio coherentes con el propósito del ejercicio.
El flujo de trabajo deberá permitir resolver, como mínimo, los siguientes requerimientos:
Diseñar un modelo de datos analítico a partir de los módulos de información suministrados, garantizando granularidad adecuada, consistencia entre identificadores y trazabilidad de las variables utilizadas.
Construir un pipeline escalable que cubra el proceso de extracción, limpieza, transformación y enriquecimiento de datos, así como la generación y entrega final de los resultados del modelo, asegurando modularidad y capacidad de ejecución programada o en ambientes productivos.
Definir un esquema de particionamiento Train/Test/OOT, justificando su estructura temporal y asegurando una correcta evaluación fuera de muestra que permita medir la estabilidad del modelo.
Evaluar el desempeño del modelo previamente estructurado, entrenado y documentado empleando como métrica principal el AUC. Esta evaluación debe complementarse con otros indicadores que considere pertinentes para medir el impacto del modelo en los procesos de cobranza y soportar la toma de decisiones operativas.

Utilizando los resultados obtenidos en el punto anterior y los datos disponibles en el modelo de datos desarrollado, construya un tablero o interfaz (empleando la herramienta o librería de su preferencia) que permita generar un análisis descriptivo con la información y variables generadas. 
Sea libre de incluir indicadores, variables, métricas, visualizaciones u otros elementos que aporten valor al análisis y que permitan definir accionables o sugerencias para el negocio.

Importante
Este ejercicio será sustentado en una sesión corta (20min), la idea es que muestren como abordaron el reto y respondan un par de preguntas.
El espacio estará divido en dos. La primera parte mostrarán el desarrollo/análisis realizado y en la segunda una sesión de preguntas.
Recuerda resolver la prueba basándote en tus fortalezas, sea ingeniera o ciencia o ambas.
Le agradecemos su participación en este ejercicio 






CARDS


Cards de negocio / EDA
                                                                                                                                   
  ┌─────┬────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────┐
  │  #  │                Card                │                                   Qué te dice                                   │   
  ├─────┼────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ 1   │ Evolución % Débito Exclusivo por   │ Si la proporción de clase 1 sube o baja en el tiempo → detecta cambios en el    │   
  │     │ Período                            │ comportamiento de pago de la cartera                                            │
  ├─────┼────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────┤   
  │ 2   │ Volumen de Obligaciones por        │ Cuántas obligaciones llegaron a mora en cada cohorte → tamaño de la cartera por │
  │     │ Período                            │  período                                                                        │   
  ├─────┼────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ 3   │ Segmentos de Cobranza —            │ Qué tan grande es cada segmento A/B/C/D sobre el total → dónde se concentra el  │   
  │     │ Distribución Global                │ esfuerzo de cobranza                                                            │   
  ├─────┼────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ 4   │ Mix de Canales de Pago por Clase   │ Cómo difiere el uso de débito, físico, virtual y otros entre clase 0 y clase 1  │   
  │     │                                    │ → valida que clase 1 usa más débito                                             │   
  ├─────┼────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ 5   │ Efectividad de Gestiones de        │ Si las gestiones, RPC, acuerdos y promesas difieren entre clases → si clase 0   │   
  │     │ Cobranza por Clase                 │ requiere más intervención humana                                                │   
  ├─────┼────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ 6   │ Segmentos de Cobranza por Período  │ Cómo evolucionan los segmentos A/B/C/D en el tiempo → tendencias de riesgo de   │   
  │     │                                    │ la cartera                                                                      │   
  ├─────┼────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ 7   │ KPI Resumen — Último Período       │ Foto ejecutiva del período más reciente: total obligaciones, % clase 1, % clase │   
  │     │                                    │  0                                                                              │   
  └─────┴────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────┘
                                                                                                                                   
  ---             
  Cards de calidad del pipeline
                                                                                                                                   
  ┌─────┬────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────┐
  │  #  │              Card              │                                     Qué te dice                                     │   
  ├─────┼────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ 11  │ Retención de Datos en Carga    │ Qué % de filas brutas del CSV llegaron a la BD → detecta pérdidas en la carga       │
  │     │ por Fuente                     │                                                                                     │
  ├─────┼────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤   
  │ 12  │ Duplicados Eliminados en Carga │ Cuántos duplicados exactos y por conflicto se descartaron por fuente → calidad del  │
  │     │                                │ ETL origen                                                                          │   
  ├─────┼────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ 13  │ Funnel de Selección de         │ Cuántas features quedan en cada etapa (varianza → ANOVA → ElasticNet), por          │   
  │     │ Features                       │ estrategia de split → qué tan agresiva fue la selección                             │   
  ├─────┼────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ 14  │ Dimensiones de Particiones     │ Filas, features candidatas/seleccionadas, tasa clase 1 y memoria por partición →    │   
  │     │ Train / Test / OOT             │ sanidad del split                                                                   │   
  ├─────┼────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ 15  │ Balance de Clases por          │ Clase 1 vs clase 0 en cada partición → si el split respetó el balance 78/22 %       │   
  │     │ Partición                      │                                                                                     │   
  ├─────┼────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ 16  │ Features Seleccionadas por     │ Cuántas features de cada grupo (gestiones, pagos, moras…) quedaron al final → qué   │   
  │     │ Grupo Temático                 │ fuente de datos aporta más al modelo                                                │   
  ├─────┼────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ 17  │ Última Fecha de Carga de Datos │ Cuándo se ejecutó por última vez el loader → freshness de los datos en BD           │   
  ├─────┼────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤   
  │ 18  │ Última Fecha de Entrenamiento  │ Cuándo se entrenó el modelo por última vez → si el modelo está desactualizado       │
  └─────┴────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────┘   
                  
  ---                                                                                                                              
  Cards de modelos

  ┌─────┬──────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────┐
  │  #  │             Card             │                                      Qué te dice                                      │
  ├─────┼──────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ 8   │ Comparativa AUC y KS por     │ AUC y KS en CV/Train/Test/OOT por modelo y configuración (Opción A/B) → cuál modelo   │
  │     │ Modelo                       │ generaliza mejor                                                                      │
  ├─────┼──────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤   
  │ 9   │ Estabilidad Train vs Test vs │ Si el AUC cae mucho de train a test/OOT → detecta sobreajuste o degradación temporal  │
  │     │  OOT                         │                                                                                       │   
  ├─────┼──────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ 10  │ Top 20 Features por Modelo   │ Qué variables impulsan más la predicción según SHAP → interpretabilidad y validación  │   
  │     │                              │ de negocio                                                                            │   
  └─────┴──────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────┘
                                                   
