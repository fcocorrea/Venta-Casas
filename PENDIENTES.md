# Estado del proyecto y trabajo pendiente

Documento de traspaso. Última actualización: **2026-07-23**.

Todo lo que dice "medido" acá se corrió de verdad contra `casas.json` (5.603 avisos, antes
`deptos.json` -- renombrado 2026-07-23 junto con `proyecto_deptos.py` -> `proyecto_casas.py` y
`deptos_scraper/` -> `casas_scraper/`, ver sección 2) en el venv
del proyecto. Lo que no está medido dice explícitamente que es estimación.

---

## 1. Dónde quedó el proyecto

`proyecto_casas.py` (2.200+ líneas) corre limpio de punta a punta. Pipeline completo:

```
PASO 1  Lectura de casas.json
PASO 2  Limpieza determinista + filtrado de filas
PASO 2b SPLIT train/test + ajustar_imputacion()/aplicar_imputacion() (refit por fold, ver PASO 5a)
PASO 3  Exploración (EDA) + casas_limpios.xlsx
PASO 4a Selección de columnas          4g Colinealidad (VIF, sin PCA)
PASO 4b Codificación de categóricas    4h Baseline mediana precio/m² por barrio
PASO 4c Validación de texto (rechazado) 4i Binaria gastos_comunes_informado
PASO 4d Escalado numérico              4j Densidad local de oferta
PASO 4e Features de distancia          4k Selección por importancia de permutación
PASO 5  a-i HECHOS (2026-07-23) -- CV, lineal, tuning, evaluación final, intervalos, ranking,
        export, mapa. Solo 5j (límites, ya documentado como comentario) es puramente descriptivo.
```

Desde el 2026-07-23, PASO 2b-4j corren como pares `ajustar_*()`/`aplicar_*()` reusables (no solo
código de una corrida), para que PASO 5a pueda reajustar TODO el preprocesamiento por fold de CV en
vez de reusar los parámetros del split fijo 80/20. Ver sección 3, ítem P3 (resuelto). El script
ahora entrena tres modelos (HistGB por defecto, HistGB tuned -- solo diagnóstico, no en producción
--, y Ridge log-log) más los tres de cuantil de PASO 5e; `requirements.txt` congela el venv.

### Números de referencia de la última corrida

Sirven para detectar si un cambio futuro rompió algo. Los de imputación/features (filas, columnas,
MdAPE CV) están verificados bit-a-bit contra la corrida previa al refactor de PASO 5a. Los de
MAE/RMSE/residuo son de PASO 5d (implementado 2026-07-23, primera vez que se miden con código
persistente en el repo -- los que aparecían acá antes eran de un cálculo ad hoc de la sesión
anterior, nunca commiteado).

| Métrica | Valor |
|---|---|
| Filas tras limpieza | 5.284 (train 4.227 / test 1.057) |
| Features finales | 75 (82 antes de la selección con CV anidada de P3, 2026-07-23) |
| MdAPE modelo (HistGB, split fijo 80/20) | **8,39 %** (PASO 4c) / **8,61 %** (PASO 5d, `modelo_q50`) |
| MdAPE baseline (mediana precio/m² por barrio) | 21,50 % |
| MdAPE lineal (Ridge log-log, PASO 5b) | 12,15 % |
| MdAPE HistGB tuned (PASO 5c, CV barata, NO en producción) | 7,84 % (default: 8,54 %) |
| MdAPE CV repetida (5 folds × 3 repeticiones, refit completo por fold) | **8,36 % ± 0,35 %** (min 7,77 %, max 8,87 %) |
| MAE (PASO 5d) | 162.688.857 CLP |
| RMSE (PASO 5d) | 319.759.311 CLP (RMSE/MAE = 1,97) |
| Residuo p10 / p50 / p90 / p99 (PASO 5d) | −16,5 % / −0,5 % / +22,7 % / **+71,8 %** |
| MdAPE por decil de precio (PASO 5d) | 8,4 % en baratos -> **13,6 %** (decil 8) -> **18,5 %** (decil 9) |
| Cobertura del intervalo (PASO 5e, segmentado + q05/q95 tuned) | 91,8 % global, 97,2 % / 89,6 % en deciles 8/9 |
| Ancho del intervalo, mediana (PASO 5e) | 53,1 % del precio predicho (antes de tuning: 56,1 %) |

El desvío de la CV (±0,35 pp) coincide con el rango ya medido variando la semilla del split fijo
(8,15/8,29/8,51 %, rango 0,36 pp) -- señal de que el refit por fold funciona de verdad y no es un
"teatro" que reusa parámetros fijos (si lo fuera, el desvío entre folds sería ~0).

Dos métricas independientes (MdAPE por decil de PASO 5d y cobertura por decil de PASO 5e) apuntan
al mismo punto débil: el modelo es notablemente menos confiable en el segmento de precio alto.

### Nada está commiteado

El último commit es `6ec308d`. Todo el trabajo de esta sesión y la anterior está en el working tree
sin commitear. Sin trackear: `.agents/`, `.claude/skills/`, `venv/` — evaluar si van a `.gitignore`.

---

## 2. Qué se arregló en esta sesión

### Bug de pandas 3.0 (rompía el script entero)

`pandas 3.0.3` activa `future.infer_string=True` por defecto: las columnas de texto llegan con
dtype `str`, ya no `object`. Dos chequeos `dtype == object` dejaron de matchear y las columnas de
medición nunca se parseaban a float. Corregido con `pd.api.types.is_numeric_dtype()`, que es
agnóstico al backend y sobrevive cambios futuros de pandas.

### Fuga de target en la imputación (crítico)

`attribute_correlations()` usaba una matriz de correlación que incluía `precio`/`clp`/
`precio unitario`. `Estacionamientos` tenía `clp` como su match de mayor correlación.

**Medido: 39 filas (20 de `Estacionamientos`, 19 de `Dormitorios`) se imputaban directamente desde
el precio.** Violaba la regla que la propia cabecera del archivo declara. Ahora es 0.

### Contaminación train/test (crítico)

Toda la limpieza que aprende de los datos corría sobre el dataset completo. Se movió el split a
**PASO 2b**, antes de las 6 imputaciones que estiman parámetros:

- reparación de comuna desde barrio
- KDTree de comuna/barrio
- umbrales de outlier (`bad_ranges`)
- matriz de correlación que guía la imputación
- `allocate_values` (tablas cruzadas, medianas agrupadas, moda)
- Orientación (KDTree + moda por barrio + moda global)

Todas ajustan sus parámetros solo con `indice_train` y los aplican a ambos folds. También se
adelantaron los descartes de filas (sin coordenadas, fuera de la RM) para que **todo filtrado de
filas cierre antes del split**.

> **Hallazgo inesperado:** el MdAPE casi no se movió (8,43 % → 8,39 %). La contaminación existía
> pero caía sobre columnas que el modelo casi no usa: las de más nulos (`Bodegas` 1.446,
> `Cantidad de pisos` 708) tienen importancia ≈ 0, y las que dominan (`Superficie total`,
> `m2_por_dormitorio`) no tenían nulos que imputar.

---

## 3. Pendiente, por prioridad

### ✅ P1 — Intervalos de predicción (resuelto 2026-07-23)

Antes no se podía distinguir "esta casa está barata" de "el modelo no supo predecir esta casa".
Medido: con umbral fijo de ±10 % se hubiera flagueado el 43 % del inventario -- señal ≈ ruido.

**Resuelto:** PASO 5e implementa CQR (conformalized quantile regression) -- tres
`HistGradientBoostingRegressor(loss="quantile")` a q=0,05/0,50/0,95, calibrados sobre un fold de
calibración recortado de train (20 %, nunca visto por el fit). Medido en test:

- Cobertura empírica: **90,5 %** (nominal 90 %) -- bien calibrado.
- Ancho del intervalo (mediana): **56,1 %** del precio predicho (p25: 43,1 %, p75: 76,4 %) -- ancho,
  la herramienta solo señala casos extremos con confianza, no matices finos. Es un hallazgo, no un
  fracaso, pero hay que comunicarlo.

**Corrección conformal segmentada por precio (2026-07-23, a pedido explícito):** la corrección
original era GLOBAL (un solo número para todo el inventario) y daba 91,7 % de cobertura promedio
pero solo 85,8 % en los dos deciles más caros -- la corrección promediaba sobre todo el inventario
y quedaba angosta justo donde el modelo es peor. Se reemplazó por calibración Mondrian/group-
conditional (Vovk 2003): 4 segmentos por q50 predicho (nunca por precio real, para que sea
calculable en producción), cada uno con su propia corrección conformal ajustada solo con sus
~211 filas de calibración. Medido el efecto:

| Decil de precio | Antes (corrección global) | Después (por segmento) |
|---|---|---|
| 8 (caro) | 85,8 % | **94,3 %** |
| 9 (más caro) | 85,8 % | **89,6 %** |

**Trade-off honesto, no escondido:** arreglar los deciles caros corrió algo de ruido a los deciles
medios (decil 3: 80,2 %, antes más cerca de 90 %) -- efecto de borde esperable con solo 4 segmentos
y ~211 filas de calibración por segmento (un decil puede caer justo en el límite entre dos
segmentos, donde la corrección salta discretamente en vez de variar suave). Global sigue mejor
calibrado (90,5 % vs 91,7 % antes), y el punto que motivó el cambio (decil 9) mejoró 3,8 pp -- neto
positivo, pero no una solución perfecta. Balance de flags también mejoró: 4,4 %/5,1 %
sub/sobrevaloradas (antes 2,9 %/5,4 %, más cerca del ~5 %/5 % esperado por construcción).

PASO 5f (residuo + regla de flag fuera-de-intervalo + ranking por distancia al borde) y 5g (filtros
del comprador, placeholders configurables, aplicados sobre el ranking) también implementados.

### ✅ P2 — PASO 5e-5i implementados (2026-07-23); solo 5b-5d siguen pendientes

PASO 5 tiene un bloque de diseño completo (comentarios) en `proyecto_casas.py` con sub-pasos
5a-5j. Implementados: **5a** (CV, ver P3), **5e** (intervalos, ver P1), **5f** (residuo/flag/
ranking), **5g** (filtros del comprador), **5h** (export `casas_candidatas.xlsx` -- solo filas
flaggeadas de test, 88 filas medido: 31 subvaloradas + 57 sobrevaloradas), **5i** (mapa interactivo,
`gráficos/mapa_intervalos.html` -- 1.057 casas de test, 88 flaggeadas, folium/OpenStreetMap;
dependencia nueva, instalada en el venv). `CLAUDE.md` ya se corrigió para describir esto con
precisión.

Falta solo:

1. 5b modelo lineal log-log, 5c tuning de hiperparámetros, 5d evaluación final en test (no bloquea
   el objetivo de negocio -- el modelo actual con intervalos ya es accionable)

### ✅ P3 — Validación cruzada (protocolo Y selección de features resueltos 2026-07-23)

**Resuelto:** PASO 2b-4j se refactorizaron a pares `ajustar_*()`/`aplicar_*()` (comuna/barrio/
orientación por KDTree, `bad_ranges`, `allocate_values`, codificación categórica, 3 escaladores
numéricos, distancias, ratios, densidad), y PASO 5a implementa `ejecutar_cv_repetida()`:
`StratifiedKFold` por `comuna`, 5 folds × 3 repeticiones, cada fold reajusta TODO el pipeline con
sus propios índices de train (no reusa los parámetros del split fijo 80/20). Medido: **8,36 % ±
0,35 %** -- desvío del mismo orden que el 0,36 pp ya medido variando semilla, confirmando que el
refit por fold funciona de verdad.

Alcance deliberadamente excluido de este refit (documentado en el propio código, PASO 5a):
- PASO 4k (selección por importancia de permutación) NO se repite por fold -- cada fold usa las 82
  columnas completas. Recalcularla por fold es nested-selection caro y no era el objetivo de esta
  primera pasada.
- PASO 4g (VIF/drop de 'Superficie útil') y PASO 4h (baseline) tampoco se recalculan por fold: son
  reglas estructurales fijas o diagnóstico de un solo split.

**Selección de features con CV anidada (resuelto 2026-07-23):** `medir_importancia_anidada()`
repite, dentro de cada uno de los 15 folds de `ejecutar_cv_repetida`, el split interno 80/20 +
importancia por permutación de PASO 4k. Por fold, la cantidad de columnas "sin señal" varió mucho
(41 a 54 de 82) -- confirma que un solo split es ruido, como ya se sabía. Exigiendo consistencia
TOTAL (sin señal en el 100% de los folds donde la columna existió, con presencia en al menos el
80% de los folds), solo **7 columnas** califican: `gastos_comunes_informado`,
`onehot__Orientación_P`, `onehot__Tipo de casa_Otro`, y las 4 canchas deportivas raras (básquetbol/
paddle/fútbol/polideportiva -- ya identificadas en PASO 4 como <1% de las casas). Comparando la CV
completa (15 folds) con y sin esas 7: 8,36 % vs 8,38 % (diferencia +0,02 pp, muy por debajo del
desvío de 0,35 %) -- **se aplicó el drop a producción**: `X_train_final`/`X_test_final` pasan de 82
a **75 columnas**, reemplazando la conclusión "se conservan las 82" de PASO 4k (que sigue en el
código como diagnóstico de referencia, sin tocar, con su advertencia actualizada apuntando acá).

**Bug encontrado y corregido en el proceso:** el mapa completo de PASO 5i (extra) reconstruye el
feature set completo desde `X` con los `aplicar_*` de PASO 4, que no sabían del drop -- rompía con
`ValueError: feature names unseen at fit time` al predecir con `modelo_q05`/`q50`/`q95` (que sí
quedaron fit con 75 columnas). Corregido reindexando `X_final_completo` a `X_train_final.columns`
antes de predecir, en vez de asumir que ambos coinciden.

### ✅ P4 — MAE y RMSE reportados (resuelto 2026-07-23, PASO 5d)

PASO 5d consolida la evaluación final sobre test (después de PASO 5e, reusa `modelo_q50`): MdAPE
8,61 %, **MAE 162.688.857 CLP**, **RMSE 319.759.311 CLP** (RMSE/MAE = 1,97 -- confirma la cola de
errores grandes que la mediana escondía). Residuo p10/p50/p90/**p99: +71,8 %**.

**Hallazgo confirmado** (era hipótesis, ahora está medido): el error se concentra en el decil de
precio más caro. MdAPE por decil: 8,4 % en los baratos, **13,6 % en el decil 8, 18,5 % en el decil
9** -- más del doble del error global (8,61 %) justo donde un mismo % de error cuesta más plata.
Conclusión de producto: la herramienta es notablemente menos confiable en el segmento alto, y eso
ya se veía también en la cobertura del intervalo de PASO 5e (85,8 % en esos mismos deciles, contra
90 % nominal). Dos métricas distintas señalando el mismo punto débil.

### ✅ P5 — Modelo lineal log-log entrenado (resuelto 2026-07-23, PASO 5b)

Ridge (alfa elegido por CV: 100.0) sobre X_train_final con 'target__barrio' relogueada/escalada
(arreglo local, ver código). MdAPE test: **12,15 %** -- le gana claro al baseline (21,50 %) pero
pierde claro contra HistGB (8,61 %, 3,5 pp de diferencia, muy por sobre el ruido de 0,35 pp de
PASO 5a) -- la complejidad del árbol se justifica.

**Elasticidad precio-superficie total medida: 0,068** (1 % más de superficie -> 0,07 % más de
precio). Sospechosamente baja para una elasticidad hedónica -- lectura más probable: colinealidad
residual entre Superficie total/Dormitorios/Baños (VIF 10-14, tolerado en PASO 4g) diluye el
coeficiente individual de Ridge entre variables correlacionadas. No se puede leer como "la
superficie casi no le importa al precio" sin ese matiz -- es una limitación de interpretación del
modelo lineal con estas features, no un hallazgo de mercado.

**Confirmado (2026-07-23):** se reajustó el mismo Ridge (misma búsqueda de alfa) excluyendo
Dormitorios/Baños/los 3 ratios estructurales, dejando 'Superficie total' como única variable de
tamaño. La elasticidad subió a **0,288** -- 4,2 veces más alta, confirma que 0,068 era dilución por
colinealidad y no un efecto de mercado genuinamente chico. Costo: el MdAPE de este modelo reducido
empeora (12,15 % -> 14,76 %), esperable porque esas columnas sí aportan señal predictiva real (el
propio HistGB las usa) -- 0,288 es el número a citar si se necesita una elasticidad hedónica
defendible, pero el modelo reducido que la produce es solo un diagnóstico, no reemplaza a
`modelo_lineal` en ningún lado del pipeline.

### ✅ P6 — Tuning de hiperparámetros (resuelto 2026-07-23, PASO 5c)

`RandomizedSearchCV` (20 combinaciones, CV barata de 5 folds sin reajustar preprocesamiento --
mismo criterio "(b)" que el diseño original de PASO 5a ya autorizaba para comparar modelos entre
sí). Medido: MdAPE CV -- default 8,54 %, tuned **7,84 %** (mejora de 0,70 pp, **supera el ruido de
0,35 pp** medido en 5a -- a diferencia de la estimación original de "1-2 puntos, no medido", esto
sí está medido y sí es una mejora real, no ruido).

Mejores hiperparámetros encontrados: `min_samples_leaf=10, max_leaf_nodes=127, max_features=0.9,
max_bins=128, learning_rate=0.1, l2_regularization=1.0`.

**Aplicado a los modelos de cuantil (2026-07-23, PASO 5e):** estos hiperparámetros se buscaron con
scorer MdAPE, un objetivo distinto al de `modelo_q05`/`q50`/`q95` (`loss='quantile'`) -- aplicarlos
tal cual habría mezclado el óptimo de un problema con la solución de otro. Se repitió la búsqueda
con **pinball loss** (la métrica que `loss='quantile'` sí optimiza), una vez por cuantil, con CV
sobre `X_train_ajuste` (la partición exacta de producción, no train+calibración). Resultado,
diferenciado por cuantil -- señal de que tratarlos igual habría sido un error:

| Cuantil | Pinball default | Pinball tuned | Mejora | Decisión |
|---|---|---|---|---|
| 0,05 | 0,0206 | 0,0195 | 5,3 % | **Se adopta** |
| 0,50 | 0,0628 | 0,0616 | 1,9 % | Se mantiene default (bajo el umbral de 2 %) |
| 0,95 | 0,0237 | 0,0219 | 7,5 % | **Se adopta** |

Efecto medido en el intervalo final (test): ancho mediano **56,1 % -> 53,1 %** del precio
predicho (más angosto, más útil) con cobertura empírica **90,5 % -> 91,8 %** (se mantiene bien
calibrado, incluso más conservador). El decil 3, que había quedado con ruido de borde tras
segmentar por precio (80,2 %), también mejoró a 85,8 % -- la mejora de q05/q95 ayudó ahí también,
no solo en los deciles caros.

### ✅ P7 — Corregir `tiene_gastos_comunes` (resuelto 2026-07-23)

Renombrada a `gastos_comunes_informado` en PASO 4i (línea ~1582 de `proyecto_casas.py`), que es lo
que la columna mide de verdad (ver hallazgo original abajo). Se optó por renombrar en vez de
eliminar: aunque su importancia por permutación medida es 0,000000 hoy, "el corredor no informó el
dato" puede seguir siendo señal en otro modelo o con más datos, y el nombre ya no engaña sobre qué
representa.

Hallazgo original: `Gastos comunes` nulo → `0` sobre 2.473 nulos (47 % de las filas). La binaria
entonces no separaba "paga gasto común" de "no paga", separaba "el corredor informó el dato". Medía
comportamiento del publicador, no atributo de la casa.

---

## 4. Límites de fondo del enfoque actual

No son bugs. Son techos metodológicos que conviene tener presentes.

### El target es precio de lista, no de transacción

El modelo responde *"¿esta casa pide más o menos que casas comparables que también están
pidiendo?"*, **no** *"¿vale más o menos de lo que pide?"*.

Consecuencia: si todo el sector oriente está listado 12 % sobre el precio de transacción real, el
modelo lo absorbe como media y no detecta nada. No hay ancla externa.

**Mayor salto de valor disponible:** cruzar contra transacciones reales del **CBR (Conservador de
Bienes Raíces)**. Convierte esto de "detector de listings anómalos" en un AVM de verdad.

### Sin dimensión temporal

Verificado sobre los 77 campos del scrape crudo: **no hay fecha de publicación, ni días en mercado,
ni historial de precio.**

Días-en-mercado es la señal de sobreprecio por preferencia revelada más fuerte del dominio: una
casa 14 meses publicada sin vender está sobrevaluada, y eso es un hecho, no una predicción.

**Acción sugerida — empezar ya:** agregar `fecha_publicacion` al spider. Es un cambio chico y el
valor es acumulativo en el tiempo; cuanto antes se empiece a acumular histórico, mejor.

### Otras variables ausentes del sector

Vista / pendiente / exposición (determinante en Lo Barnechea), calidad de terminaciones más allá de
binarios, distancia a colegios (se descartó por no tener coordenada única — se podría usar distancia
al colegio *más cercano* de una lista, en vez de a un centroide inventado).

---

## 5. Entorno (tropezones ya resueltos)

- **Usar el venv del proyecto**: `./venv/Scripts/python.exe`. El `python` del PATH es el de la
  Microsoft Store y **no tiene sklearn**.
- `pandas 3.0.3` — ojo con `future.infer_string` (ver sección 2).
- Se instalaron en el venv: `openpyxl` y el corpus `stopwords` de nltk (sesión anterior,
  `python -c "import nltk; nltk.download('stopwords')"`), y `folium` (esta sesión, para PASO 5i).
- `TargetEncoder` emite un `FutureWarning` por `random_state` (deprecado en sklearn 1.9, se elimina
  en 1.11). Inofensivo hoy; migrar a pasar un `cv` cuando moleste.
- `requirements.txt` congelado (2026-07-23, `pip freeze`). Reinstalar con
  `./venv/Scripts/python.exe -m pip install -r requirements.txt` si se recrea el venv.

---

## 6. Decisiones tomadas — no volver a abrirlas sin motivo nuevo

| Decisión | Por qué |
|---|---|
| **Sin PCA** para la colinealidad | Mata la interpretabilidad del modelo hedónico y mezcla dummies con continuas. Documentado en PASO 4g. |
| Solo se dropeó `Superficie útil` | Redundancia exacta (`útil = total × ratio_construido`). VIF residual 10-14 se tolera: es señal real, no identidad matemática. |
| **Sin TF-IDF/SVD** de `descripcion` | Medido: no aportó (8,39 % vs 8,40 %). Revisar solo si cambia el modelo o hay muchos más datos. |
| EDA de PASO 3 sobre el dataset completo | Es material descriptivo del mercado, no ajusta parámetros del modelo. |
| Split estratificado por `comuna` | Los tres niveles de precio son muy distintos entre comunas. |
| Target = `log(clp)`, no `precio unitario` | Dividir por superficie impone elasticidad = 1, que es falso en terrenos. |

---

## 7. Arranque sugerido para la próxima sesión

P1-P7 están todos resueltos. El entregable de negocio (ranking con intervalos, export y mapa)
existe y corre de punta a punta; PASO 5 completo hasta 5i; `requirements.txt` congelado. Lo que
queda es refinamiento, ninguno bloquea nada:

1. ~~Segmentar por precio~~ -- **resuelto 2026-07-23**: calibración conformal Mondrian/group-
   conditional por segmento de precio (ver P1).
2. ~~Decidir sobre el tuning de 5c~~ -- **resuelto 2026-07-23**: se repitió la búsqueda con pinball
   loss por cuantil (ver P6) en vez de aplicar el tuning de MdAPE tal cual. q05/q95 adoptaron
   tuning (mejoras de 5,3 %/7,5 %), q50 se quedó con el default (1,9 %, bajo el umbral). Efecto
   combinado con la segmentación de precio: cobertura 91,8 % global, ancho mediano del intervalo
   bajó de 56,1 % a **53,1 %** (más angosto, más útil), y el decil 3 (que había quedado con ruido de
   borde tras segmentar solo) mejoró de 80,2 % a 85,8 %.
3. **Punto pendiente real que queda de la ronda de segmentación/tuning**: el MdAPE puntual del
   segmento caro (P4/P5d, `modelo_q50`, que se quedó con default) sigue en ~18,5 % en el decil 9 --
   los cambios de esta ronda mejoraron la CALIBRACIÓN del intervalo ahí, no la PRECISIÓN del punto
   predicho. Si se necesita bajar ese número, hace falta un modelo de punto separado para el
   segmento alto (la opción descartada al elegir el enfoque de calibración, ver P1), no más ajuste
   de intervalo. Revisar también si conviene más granularidad de segmentos (ej. 6-8 en vez de 4)
   cuando haya más datos de calibración.
4. ~~Selección de features CON la CV (nested)~~ -- **resuelto 2026-07-23** (ver P3): 7 columnas
   sin señal en el 100% de 15 folds (`gastos_comunes_informado`, 4 canchas deportivas raras,
   `onehot__Orientación_P`, `onehot__Tipo de casa_Otro`). CV completo con/sin ellas: 8,36% vs
   8,38% (+0,02 pp, dentro del ruido) -- se aplicó el drop, `X_train_final`/`X_test_final` pasan de
   82 a **75 columnas**. De paso se encontró y corrigió un bug real: el mapa completo de PASO 5i no
   sabía del drop y rompía al predecir.
5. ~~Elasticidad precio-superficie~~ -- **resuelto 2026-07-23** (ver P5): confirmado que 0,068 era
   dilución por colinealidad -- sin Dormitorios/Baños/ratios, sube a **0,288** (4,2x). Costo:
   MdAPE de ese modelo diagnóstico empeora a 14,76% -- 0,288 es el número a citar para uso hedónico,
   pero ese modelo reducido no reemplaza a `modelo_lineal` en el pipeline.
