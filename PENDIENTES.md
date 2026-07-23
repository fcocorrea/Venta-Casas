# Estado del proyecto y trabajo pendiente

Documento de traspaso. Última actualización: **2026-07-23**.

Todo lo que dice "medido" acá se corrió de verdad contra `deptos.json` (5.603 avisos) en el venv
del proyecto. Lo que no está medido dice explícitamente que es estimación.

---

## 1. Dónde quedó el proyecto

`proyecto_deptos.py` (2.207 líneas) corre limpio de punta a punta. Pipeline completo:

```
PASO 1  Lectura de deptos.json
PASO 2  Limpieza determinista + filtrado de filas
PASO 2b SPLIT train/test + ajustar_imputacion()/aplicar_imputacion() (refit por fold, ver PASO 5a)
PASO 3  Exploración (EDA) + deptos_limpios.xlsx
PASO 4a Selección de columnas          4g Colinealidad (VIF, sin PCA)
PASO 4b Codificación de categóricas    4h Baseline mediana precio/m² por barrio
PASO 4c Validación de texto (rechazado) 4i Binaria gastos_comunes_informado
PASO 4d Escalado numérico              4j Densidad local de oferta
PASO 4e Features de distancia          4k Selección por importancia de permutación
PASO 5a CV repetida (HECHO, 2026-07-23) 5b-5j Modelo lineal/intervalos/ranking/export -- PENDIENTES
```

Desde el 2026-07-23, PASO 2b-4j corren como pares `ajustar_*()`/`aplicar_*()` reusables (no solo
código de una corrida), para que PASO 5a pueda reajustar TODO el preprocesamiento por fold de CV en
vez de reusar los parámetros del split fijo 80/20. Ver sección 3, ítem P3 (resuelto).

### Números de referencia de la última corrida

Sirven para detectar si un cambio futuro rompió algo. Verificado bit-a-bit contra la corrida previa
al refactor de PASO 5a (mismo `deptos.json`, mismo `random_state`): estos números NO cambiaron.

| Métrica | Valor |
|---|---|
| Filas tras limpieza | 5.284 (train 4.227 / test 1.057) |
| Features finales | 82 |
| MdAPE modelo (HistGB, split fijo 80/20) | **8,39 %** |
| MdAPE baseline (mediana precio/m² por barrio) | 21,50 % |
| MdAPE CV repetida (5 folds × 3 repeticiones, refit completo por fold) | **8,36 % ± 0,35 %** (min 7,77 %, max 8,87 %) |
| MAE | 150.912.000 CLP |
| RMSE | 295.966.827 CLP |
| Residuo p10 / p50 / p90 | −15,8 % / −0,5 % / +20,1 % |

El desvío de la CV (±0,35 pp) coincide con el rango ya medido variando la semilla del split fijo
(8,15/8,29/8,51 %, rango 0,36 pp) -- señal de que el refit por fold funciona de verdad y no es un
"teatro" que reusa parámetros fijos (si lo fuera, el desvío entre folds sería ~0).

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

### 🔴 P1 — Intervalos de predicción (bloquea el objetivo de negocio)

### ✅ P1 — Intervalos de predicción (resuelto 2026-07-23)

Antes no se podía distinguir "esta casa está barata" de "el modelo no supo predecir esta casa".
Medido: con umbral fijo de ±10 % se hubiera flagueado el 43 % del inventario -- señal ≈ ruido.

**Resuelto:** PASO 5e implementa CQR (conformalized quantile regression) -- tres
`HistGradientBoostingRegressor(loss="quantile")` a q=0,05/0,50/0,95, calibrados sobre un fold de
calibración recortado de train (20 %, nunca visto por el fit). Medido en test:

- Cobertura empírica: **91,7 %** (nominal 90 %) -- bien calibrado.
- Ancho del intervalo (mediana): **56,1 %** del precio predicho (p25: 46,7 %, p75: 71,3 %) -- ancho,
  la herramienta solo señala casos extremos con confianza, no matices finos. Es un hallazgo, no un
  fracaso, pero hay que comunicarlo.
- Cobertura por decil de precio: 93-95 % en deciles baratos, **85,8 %** en los dos deciles más
  caros -- el intervalo es menos confiable justo donde una mala calibración cuesta más plata.

PASO 5f (residuo + regla de flag fuera-de-intervalo + ranking por distancia al borde) y 5g (filtros
del comprador, placeholders configurables, aplicados sobre el ranking) también implementados.

### ✅ P2 — PASO 5e-5i implementados (2026-07-23); solo 5b-5d siguen pendientes

PASO 5 tiene un bloque de diseño completo (comentarios) en `proyecto_deptos.py` con sub-pasos
5a-5j. Implementados: **5a** (CV, ver P3), **5e** (intervalos, ver P1), **5f** (residuo/flag/
ranking), **5g** (filtros del comprador), **5h** (export `casas_candidatas.xlsx` -- solo filas
flaggeadas de test, 88 filas medido: 31 subvaloradas + 57 sobrevaloradas), **5i** (mapa interactivo,
`gráficos/mapa_intervalos.html` -- 1.057 casas de test, 88 flaggeadas, folium/OpenStreetMap;
dependencia nueva, instalada en el venv). `CLAUDE.md` ya se corrigió para describir esto con
precisión.

Falta solo:

1. 5b modelo lineal log-log, 5c tuning de hiperparámetros, 5d evaluación final en test (no bloquea
   el objetivo de negocio -- el modelo actual con intervalos ya es accionable)

### ✅ P3 — Validación cruzada (protocolo resuelto 2026-07-23; feature selection con CV sigue pendiente)

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

**Sigue pendiente** (la parte de P3 que todavía no se resuelve): usar esta CV para decidir
features de verdad -- descartar una columna solo si su pérdida de importancia es consistente entre
folds, reemplazando la comparación de un solo split que PASO 4k ya marcó como ruido (antes
"descarta 44 columnas", ahora "conserva las 82" sin tocar ninguna feature). Requiere nested CV
(selección de importancia dentro de cada fold externo) -- más caro que el protocolo de arriba.

### 🟠 P4 — Reportar MAE y RMSE, no solo MdAPE

`RMSE ≈ 2 × MAE` delata una cola de errores grandes que la mediana esconde por completo
(residuo p99: **+80,8 %**). Para un negocio donde equivocarse 300M CLP es catastrófico, reportar
solo la mediana es engañoso.

### 🟡 P5 — Entrenar el modelo lineal log-log

La cabecera lo promete (línea 36) y nunca se implementó. Solo existe `HistGradientBoostingRegressor`.

Importa además porque **se rechazó PCA argumentando preservar la interpretabilidad de los
coeficientes** (PASO 4g) — y hoy no hay ningún modelo con coeficientes. El argumento es correcto en
lo estadístico pero está sin respaldo empírico.

### 🟡 P6 — Tuning de hiperparámetros

`HistGradientBoostingRegressor` corre con defaults, sin búsqueda. Estimación (no medida): 1-2 puntos
de MdAPE disponibles.

### ✅ P7 — Corregir `tiene_gastos_comunes` (resuelto 2026-07-23)

Renombrada a `gastos_comunes_informado` en PASO 4i (línea ~1582 de `proyecto_deptos.py`), que es lo
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
- No hay `requirements.txt`. Valdría la pena congelar el venv.

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

P1, P2 (5e-5i) y P3 (protocolo de CV) ya están resueltos -- el entregable de negocio (ranking con
intervalos, export y mapa) existe y corre de punta a punta. Lo que sigue, sin bloquear nada:

1. **5b-5d** (lineal log-log, tuning, evaluación final) -- mejoran el número y dan un chequeo de
   sanidad de coeficientes, pero el sistema ya es accionable sin ellos.
2. La parte de P3 que sigue abierta (selección de features CON la CV, no solo el protocolo de
   medición) -- reemplazar la decisión de un solo split de PASO 4k.
3. Congelar el venv (`pip freeze`) -- ya son dos dependencias nuevas esta sesión (`openpyxl`,
   `folium`) sin `requirements.txt` que las registre (ver sección 5).
