# Estado del proyecto y trabajo pendiente

Documento de traspaso. Última actualización: **2026-07-22**.

Todo lo que dice "medido" acá se corrió de verdad contra `deptos.json` (5.603 avisos) en el venv
del proyecto. Lo que no está medido dice explícitamente que es estimación.

---

## 1. Dónde quedó el proyecto

`proyecto_deptos.py` (1.732 líneas) corre limpio de punta a punta. Pipeline completo:

```
PASO 1  Lectura de deptos.json
PASO 2  Limpieza determinista + filtrado de filas
PASO 2b SPLIT train/test  ← todo lo que aprende de los datos va DESPUÉS de acá
PASO 3  Exploración (EDA) + deptos_limpios.xlsx
PASO 4a Selección de columnas          4g Colinealidad (VIF, sin PCA)
PASO 4b Codificación de categóricas    4h Baseline mediana precio/m² por barrio
PASO 4c Validación de texto (rechazado) 4i Binaria de gastos comunes
PASO 4d Escalado numérico              4j Densidad local de oferta
PASO 4e Features de distancia          4k Selección por importancia de permutación
PASO 5  NO IMPLEMENTADO  ← ver sección 3
```

### Números de referencia de la última corrida

Sirven para detectar si un cambio futuro rompió algo.

| Métrica | Valor |
|---|---|
| Filas tras limpieza | 5.284 (train 4.227 / test 1.057) |
| Features finales | 82 |
| MdAPE modelo (HistGB, test) | **8,39 %** |
| MdAPE baseline (mediana precio/m² por barrio) | 21,50 % |
| MAE | 150.912.000 CLP |
| RMSE | 295.966.827 CLP |
| Residuo p10 / p50 / p90 | −15,8 % / −0,5 % / +20,1 % |

### Nada está commiteado

El último commit es `6ec308d`. Todo el trabajo de esta sesión está en el working tree sin commitear.
Sin trackear: `.agents/`, `.claude/skills/`, `venv/` — evaluar si van a `.gitignore`.

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

Hoy no se puede distinguir "esta casa está barata" de "el modelo no supo predecir esta casa".

Medido: con umbral fijo de ±10 % se flagearía el **43 % del inventario** (20,6 % subvaluadas +
22,6 % sobrevaluadas). El error del modelo (±8-20 %) es del mismo orden que el margen de
negociación típico en Chile (5-10 %) — señal ≈ ruido.

**Qué hacer:** `HistGradientBoostingRegressor(loss="quantile", quantile=...)` a q=0,05 / 0,50 / 0,95,
o conformal prediction. Flagear una casa **solo si su precio real cae fuera del intervalo**, nunca
contra un umbral fijo.

### 🔴 P2 — Implementar PASO 5 (el entregable no existe)

La cabecera lo describe (líneas 40-45) pero no hay código. Falta:

1. Residuo `(precio real − predicho) / predicho` por casa
2. Ranking por mayor descuento
3. Filtros del comprador (dormitorios, baños) **sobre** el ranking, no como criterio principal
4. Export `casas_candidatas.xlsx`
5. Mapa coloreado rojo (sobrevalorada) → verde (subvalorada)

> `CLAUDE.md` afirma que el script produce `casas_candidatas.xlsx`. **Es falso hoy** — el archivo no
> existe ni se genera. Corregir `CLAUDE.md` al implementar esto.

### 🟠 P3 — Validación cruzada

Hoy: una sola partición 80/20. Medido sobre 3 semillas, el mismo modelo da MdAPE de
**8,15 % / 8,29 % / 8,51 %** — un rango de 0,36 puntos.

Eso invalida la decisión de PASO 4k, cuya diferencia es de 0,20 puntos. **Prueba concreta:** el
criterio ya se dio vuelta solo por arreglar la contaminación, sin tocar ninguna feature
(antes "descarta 44 columnas", ahora "conserva las 82"). La decisión la tomaba el ruido.

**Qué hacer:** `RepeatedKFold`; descartar una columna solo si su pérdida es consistente entre folds.
Ya hay una advertencia escrita al final de PASO 4k.

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

### 🟡 P7 — Corregir `tiene_gastos_comunes`

`Gastos comunes` nulo → `0` sobre 2.473 nulos (47 % de las filas). La binaria entonces no separa
"paga gasto común" de "no paga", separa **"el corredor informó el dato"**. Mide comportamiento del
publicador, no atributo de la casa.

Confirmado: importancia por permutación exactamente **0,000000**. Renombrar a
`gastos_comunes_informado` (que es lo que mide) o eliminarla.

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
- Se instalaron en el venv durante la sesión: `openpyxl`, y el corpus `stopwords` de nltk
  (`python -c "import nltk; nltk.download('stopwords')"`).
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

1. Commitear el trabajo actual (está todo sin commitear, ver sección 1).
2. **P1 + P2 juntos** — son el objetivo de negocio y se complementan: los intervalos de predicción
   son justamente lo que hace que el ranking de PASO 5 sea accionable en vez de ruido.
3. Después P3 (`RepeatedKFold`), que vuelve defendible toda decisión posterior de features.
