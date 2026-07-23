# Análisis para la venta de casas
#
# Problemática:
# No siempre sabemos si el precio de una casa está sobre o bajo el mercado dadas sus
# características. Para decidir bien necesitamos datos. Muchos datos.
#
# Contexto:
# El estudio se realiza en Santiago de Chile, en tres comunas del sector oriente: Las Condes,
# Vitacura y Lo Barnechea. Las ofertas se obtienen de Portal Inmobiliario, considerando solo
# publicaciones de venta de casas.
#
# Objetivos:
# Predecir el precio de una casa a partir de sus características y comparar ese precio predicho
# con el precio real, para determinar si está sobre o bajo su precio de mercado. Es un problema
# de regresión: "sobre/bajo mercado" no se entrena como un modelo aparte, se deriva del residuo
# (precio real - predicho) / predicho de un único modelo de precio.
#
# Target: se modela log(precio en CLP), no "precio unitario" (CLP/m²). Dividir por superficie
# para normalizar la escala impone elasticidad = 1 entre precio y superficie (un lote de 200 m²
# valdría exactamente el doble que uno de 100 m², todo lo demás igual), lo que no es cierto en
# terrenos (retornos decrecientes a la superficie). El modelo debe aprender esa elasticidad a
# partir de log(Superficie total) y log(Superficie útil) como features, no asumirla de antemano.
# "precio unitario" se mantiene como feature/diagnóstico para detección de outliers -- tal como ya
# se usa más abajo -- pero no como variable objetivo.
#
# Implementación, en cinco grandes pasos (numeración alineada 1:1 con los PASO del código: el
# paso N de acá es PASO N más abajo, sin desfase):
# 1) Obtención de datos: se extraen vía web scraping con Scrapy (ver deptos_scraper/spiders/deptos.py;
#    se ejecuta con `scrapy crawl deptos -O deptos.json`, documentado en CLAUDE.md).
# 2) Limpieza de datos: homologar y transformar los datos, que no vienen expresados de forma
#    consistente. Las imputaciones que aprenden de los datos (KDTree de comuna/barrio, moda por
#    barrio, match por correlación en atributos discretos, umbrales de outlier) se ajustan SOLO
#    con el fold de entrenamiento -- por eso el split vive en PASO 2b, antes de todas ellas, y no
#    en PASO 4 -- y ninguna usa precio/clp/precio unitario como predictor, para no filtrar el
#    target hacia adentro de una feature (ver COLUMNAS_TARGET en la matriz de correlación).
# 3) Exploración: entender los datos para adaptarlos a un modelo de regresión.
# 4) Preprocesamiento: selección de columnas, codificación de categóricas, escalado numérico,
#    features de distancia/densidad/ratios, colinealidad (VIF, sin PCA) y selección final por
#    importancia de permutación. Deja armados X_train_final/X_test_final -- el input ya listo
#    para cualquier modelo, sin entrenar todavía ninguno de producción.
# 5) Modelamiento y producción: validación cruzada repetida, modelo lineal log-log (interpretable)
#    y uno de árboles con boosting (captura interacciones comuna × superficie × amenities),
#    evaluados con MdAPE/MAE/RMSE en CLP (no solo en escala log) contra el baseline de mediana por
#    barrio. Con el modelo elegido: intervalos de predicción, residuo (precio real - predicho) /
#    predicho, y ranking por distancia al borde del intervalo -- no por residuo crudo, y no contra
#    un umbral fijo -- que reemplaza el filtro original (precio UF < 15000, Dormitorios >= 3, etc.).
#    Las restricciones del comprador (dormitorios, baños, cercanía) se aplican como filtro sobre el
#    ranking, no como criterio principal. El resultado se ubica en un mapa, coloreado de rojo
#    (sobre el intervalo, sobrevalorada) a verde (bajo el intervalo, subvalorada).

import os
import re
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

GRAFICOS_DIR = 'gráficos'
os.makedirs(GRAFICOS_DIR, exist_ok=True)

# =======================================================================================
# PASO 1 -- OBTENCIÓN DE DATOS
# =======================================================================================
# El scraping en sí corre fuera de este script, vía Scrapy (deptos_scraper/spiders/deptos.py,
# `scrapy crawl deptos -O deptos.json`). Este archivo retoma el proceso desde
# el JSON ya scrapeado: no hace requests HTTP, solo lee lo que el crawl dejó en disco.

deptos_df = pd.read_json('deptos.json')
print('Cantidad de observaciones: {}.\nCantidad de atributos: {}.'.format(*deptos_df.shape))
print('Columnas:', ', '.join(deptos_df.columns.tolist()))

# =======================================================================================
# PASO 2 -- LIMPIEZA DE DATOS
# =======================================================================================
# Homologa y transforma los datos, que no vienen expresados de forma consistente entre
# publicaciones (formatos de texto distintos para el mismo tipo de dato, columnas invertidas,
# nulos que en realidad significan "no informado" en vez de "desconocido", etc.).

# Atributos:
# El dataframe tiene muchos atributos y hay que reducir la dimensionalidad para un análisis más
# centrado. Muchos son categóricos binarios (indican si la casa tiene o no cierta característica).
# Algunos no aportan mucho valor para elegir una casa o llegan casi vacíos.

remove_columns = [c for c in ['Chimenea', 'Fecha de entrega', 'Tipo de seguridad'] if c in deptos_df.columns]
deptos_df.drop(remove_columns, axis=1, inplace=True)

# Atributos categóricos binarios:
# indican si la casa tiene o no una característica (piscina, quincho, jardín, etc.). Consideramos
# binario un atributo con 3 valores únicos o menos (el 3 contempla los nulos).

binary_features = deptos_df.loc[:, deptos_df.nunique(dropna=False) < 4]
count_binary_attributes = binary_features.shape[1]
count_deptos_attributes = deptos_df.shape[1]
print(f'Los atributos binarios son {count_binary_attributes} y representan el '
      f'{round((count_binary_attributes / count_deptos_attributes) * 100, 4)} % del total de atributos')

# Definir un vector objetivo:
# el precio por metro cuadrado es un buen indicador porque tiene una escala homogénea (precio en
# pesos chilenos dividido por los metros cuadrados totales).
#
# Moneda (UM): 'precio' y 'clp' llegan del scraper como números puros (sin la palabra "pesos" ni
# "unidades de fomento" -- esas columnas nunca contienen texto, son el fragmento numérico del
# monto). La detección de moneda original buscaba esas palabras y por lo tanto nunca matcheaba
# nada (siempre caía en 'Unknown'). El dato real está en si 'clp' viene informado o no: el sitio
# solo muestra un monto equivalente en pesos cuando el precio principal está en UF.

deptos_df['UM'] = np.where(deptos_df['clp'].isna(), 'CLP', 'UF')
print('Distribución de moneda:', deptos_df['UM'].value_counts().to_dict())


NUMERO_REGEX = re.compile(r'-?\d+(?:\.\d{3})*(?:,\d+)?')


def parse_measurement(value):
    """Convierte 'N', 'N a M' (rango, se usa el mínimo) o 'N.NNN unidad' (separador de miles) a
    float. Devuelve NaN si no hay nada parseable (ej. "Precio a consultar")."""
    if pd.isna(value):
        return np.nan
    numeros = NUMERO_REGEX.findall(str(value))
    if not numeros:
        return np.nan
    token = numeros[0].replace('.', '').replace(',', '.')
    try:
        return float(token)
    except ValueError:
        return np.nan


# Columnas de medición: se separan en las que el JSON ya trajo como float64 (no requieren parseo
# de texto) y las que llegaron como texto y hay que convertir. Se deriva del dtype real en vez de
# asumir de memoria cuáles son numéricas, porque esa lista cambia según el schema de casas vs.
# departamentos.

measurement_columns = [c for c in ['precio', 'clp', 'Superficie total', 'Superficie útil',
                                    'Dormitorios', 'Baños', 'Estacionamientos', 'Bodegas',
                                    'Cantidad de pisos', 'Antigüedad'] if c in deptos_df.columns]
numeric_attributes = [c for c in measurement_columns if not pd.api.types.is_numeric_dtype(deptos_df[c])]
print('Columnas de medición ya numéricas:', [c for c in measurement_columns if c not in numeric_attributes])
print('Columnas de medición a parsear desde texto:', numeric_attributes)

for column in numeric_attributes:
    deptos_df[column] = deptos_df[column].apply(parse_measurement)

# clp es el precio en pesos chilenos equivalente, y solo viene informado cuando la venta está en
# una moneda distinta a CLP. Cuando el precio ya está en CLP, ese valor vive en la columna precio.

deptos_df['clp'] = deptos_df['clp'].fillna(deptos_df['precio'])

# Publicaciones sin precio parseable (ej. "Precio a consultar", frecuente en casas de alto valor)
# no sirven como variable objetivo: se descartan.

sin_precio = deptos_df[deptos_df['clp'].isna()]
print(f'Casas sin precio parseable: {len(sin_precio)}')
deptos_df = deptos_df.drop(sin_precio.index)

# Con el precio listo, falta la superficie total (denominador). Algunas superficies vienen como
# rango (ej. "45 a 52") cuando se vende más de una casa en el mismo proyecto; parse_measurement ya
# toma el valor mínimo del rango, igual que antes.
#
# Valores de superficie implausiblemente chicos (ej. "1,18 m²", "1 m²") no son medidas reales --
# ninguna casa se vende con 1-4 m² de superficie útil o total, así que es el mismo tipo de dato
# roto que un nulo, no una casa real. Se detectaron 6 casos donde un campo trae un valor así
# mientras el otro trae uno plausible (160-553 m²): se anulan ANTES del fill cruzado de abajo,
# para que se resuelvan con el valor bueno del otro campo -- si no, el swap útil/total de más
# abajo (pensado para columnas invertidas, no para basura) los "arregla" dejando la basura adentro
# igual, solo que en la columna equivocada.

SUPERFICIE_MINIMA_PLAUSIBLE = 5  # ponytail: umbral conservador (menor que un baño chico); subir si aparecen más casos límite
deptos_df['Superficie útil'] = np.where(deptos_df['Superficie útil'] <= SUPERFICIE_MINIMA_PLAUSIBLE,
                                         np.nan, deptos_df['Superficie útil'])
deptos_df['Superficie total'] = np.where(deptos_df['Superficie total'] <= SUPERFICIE_MINIMA_PLAUSIBLE,
                                          np.nan, deptos_df['Superficie total'])

# Cuando falta superficie total o superficie útil pero no la otra, rellenamos con la que sí está.
# Las casas sin ninguna de las dos se descartan, porque no hay forma de saber si están con
# sobreprecio.

util_missing = deptos_df['Superficie útil'].isna() | (deptos_df['Superficie útil'] == 0.0)
total_missing = deptos_df['Superficie total'].isna() | (deptos_df['Superficie total'] == 0.0)
deptos_df['Superficie útil'] = np.where(util_missing & ~total_missing, deptos_df['Superficie total'], deptos_df['Superficie útil'])
deptos_df['Superficie total'] = np.where(total_missing & ~util_missing, deptos_df['Superficie útil'], deptos_df['Superficie total'])
index_deptos_sin_superficie = deptos_df[deptos_df['Superficie útil'].isna() | (deptos_df['Superficie útil'] == 0.0)].index
print(f'Casas sin superficie: {len(index_deptos_sin_superficie)}')
deptos_df.drop(index_deptos_sin_superficie, inplace=True)

# Cuando la superficie útil es mayor que la total (no tiene sentido), las columnas vienen
# invertidas en el aviso: se intercambian.

util, total = deptos_df['Superficie útil'], deptos_df['Superficie total']
deptos_df['Superficie útil'], deptos_df['Superficie total'] = np.minimum(util, total), np.maximum(util, total)

# Con precio y superficie listos, calculamos la variable objetivo.

deptos_df['precio unitario'] = (deptos_df['clp'] / deptos_df['Superficie total']).round(4)

# Valores extremos:
# revisamos errores de tipeo antes de seguir con el análisis.

import matplotlib.pyplot as plt

plt.hist(deptos_df['precio unitario'])
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.xlabel('Precio por metro cuadrado')
plt.ylabel('Frecuencia')
plt.title('Histograma de Precio por metro cuadrado.')
plt.xticks(rotation=90)
plt.savefig(os.path.join(GRAFICOS_DIR, 'histograma_precio_unitario.png'))



def percentil_limits(serie: pd.Series, percentils: tuple) -> tuple:
    """Límites inferior y superior para considerar un valor extremo."""
    lower_limit = serie.quantile(percentils[0])
    upper_limit = serie.quantile(percentils[1])
    percentil_ranges = upper_limit - lower_limit
    lower_bound = lower_limit - 1.5 * percentil_ranges
    upper_bound = upper_limit + 1.5 * percentil_ranges
    return lower_bound, upper_bound


def get_interquartile_range(serie: pd.Series) -> pd.Series:
    """Valores de la serie dentro del rango intercuartílico."""
    lower_bound, upper_bound = percentil_limits(serie, (0.25, 0.75))
    return serie[(serie >= lower_bound) & (serie <= upper_bound)]


unitary_price = deptos_df['precio unitario']
interquartile_range = get_interquartile_range(unitary_price)
plt.figure()
plt.xlabel('Precio por metro cuadrado')
plt.ylabel('Frecuencia')
plt.title('Histograma de precio por metro cuadrado.')
plt.hist(interquartile_range, bins='sturges')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.savefig(os.path.join(GRAFICOS_DIR, 'histograma_precio_unitario_iqr.png'))



def show_outliers(serie: pd.Series) -> tuple:
    """Valores extremos (bajos y altos) de `serie`."""
    lower_bound, upper_bound = percentil_limits(serie, (0.25, 0.75))
    return serie.loc[serie < lower_bound], serie.loc[serie > upper_bound]


lower_values, higher_values = show_outliers(unitary_price)
print(f'Valores extremos bajos: {len(lower_values)}. Valores extremos altos: {len(higher_values)}.')


def plot_sorted_attribute(serie, ascending, ylabel, xlabel, filename, **plot_config):
    plt.figure()
    serie = serie.sort_values(ascending=ascending)
    plt.ticklabel_format(style='plain', axis='both')
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.plot(serie.values, **plot_config)
    plt.savefig(os.path.join(GRAFICOS_DIR, filename))
    


plot_sorted_attribute(lower_values, False, 'Precio Unitario', 'Casa (N°)', 'outliers_bajos_precio_unitario.png',
                      color='darkblue', marker='o', linestyle='dashed', linewidth=2, markersize=8)
plot_sorted_attribute(higher_values, True, 'Precio Unitario', 'Casa (N°)', 'outliers_altos_precio_unitario.png',
                      color='darkblue', linestyle='dashed')


def cut_after_relative_jump(sorted_values: pd.Series, threshold: float):
    """Recorre `sorted_values` (ya ordenada) y devuelve el primer valor tras un salto relativo
    (en valor absoluto) mayor a `threshold` respecto al valor anterior. None si no hay tal salto.
    Reemplaza los índices fijos que antes se leían a mano de un gráfico (ej. "el N°141"), que
    quedaban pegados a la escala de precios de un dataset específico y no se adaptaban a uno
    nuevo."""
    changes = sorted_values.pct_change().abs()
    jump_idx = changes[changes > threshold].index
    return sorted_values.loc[jump_idx[0]] if len(jump_idx) else None


# Comunas:
# en la página de origen, el último elemento de la ruta de navegación corresponde al barrio y el
# penúltimo a la comuna. Si la casa no tiene barrio, la comuna termina ocupando el lugar del
# barrio en el dataset, así que hay que corregirlo.

print('Comunas encontradas:', deptos_df.comuna.unique().tolist())

comunas_a_modificar = deptos_df[(deptos_df.comuna == 'RM (Metropolitana)') | (deptos_df.comuna == 'Propiedades usadas')]

# En estos casos el atributo "barrio" en realidad contiene la comuna, y "comuna" contiene la región
# o la sección "propiedades usadas" de la página, porque el barrio real no está detallado.
# - Si la comuna es "Propiedades usadas", la propiedad no tiene ni comuna ni barrio: dejamos el
#   barrio como nulo.
# - Para "Propiedades usadas" y "RM (Metropolitana)", la comuna queda como valor nulo (se imputa
#   más abajo).

deptos_df.barrio = np.where(deptos_df.comuna == 'Propiedades usadas', np.nan, deptos_df.barrio)
deptos_df.loc[comunas_a_modificar.index, 'comuna'] = np.nan
print(f'Hay {deptos_df.comuna.isna().sum()} comunas por imputar.')

# La IMPUTACIÓN de esas comunas nulas (mapeo barrio -> comuna) aprende de la distribución de los
# datos, así que no puede correr acá: va después del split, ajustada solo con train. Ver
# "PASO 2b -- SPLIT" más abajo. Lo que queda arriba es la corrección determinista del artefacto
# del scraper (poner en nulo lo que no es una comuna), que no aprende nada y no necesita el split.

# Signo de latitud/longitud: Santiago de Chile cae siempre en latitud y longitud negativas
# (hemisferio sur/oeste). Alguna fila llega con el signo invertido (error del scraper, no un dato
# realmente distinto -- la magnitud sigue siendo la correcta para la comuna), así que se corrige
# el signo en vez de anular el valor.

lat_lon_signo_invertido = (deptos_df['latitud'] > 0) | (deptos_df['longitud'] > 0)
if lat_lon_signo_invertido.any():
    print(f'Corregimos el signo de latitud/longitud en {lat_lon_signo_invertido.sum()} filas.')
    deptos_df.loc[lat_lon_signo_invertido, 'latitud'] = -deptos_df.loc[lat_lon_signo_invertido, 'latitud'].abs()
    deptos_df.loc[lat_lon_signo_invertido, 'longitud'] = -deptos_df.loc[lat_lon_signo_invertido, 'longitud'].abs()

# Filas sin coordenadas: latitud/longitud es la única columna que no se puede imputar por vecino
# más cercano, porque justamente se necesita la coordenada de la propia fila para buscar al vecino.
# Es una sola casa en todo el dataset: se descarta.
#
# Coordenadas fuera de la Región Metropolitana: se detectó una publicación ("Casa, Oficina Y Local
# En Venta Lujan, Mendoza, Argentina", etiquetada como comuna "Las Condes") con coordenadas en
# Mendoza, Argentina -- a más de 130 km al otro lado de la cordillera, un error de scraping/
# clasificación del sitio de origen, no una casa real del mercado de Santiago. Se descartan filas
# con longitud > -69,5°: la propiedad más extrema pero real del dataset (un refugio en
# Farellones/La Parva, dentro del territorio de Lo Barnechea que sí llega hasta la cordillera)
# queda en -70,22°, muy por debajo de ese corte.
#
# Ambos descartes se hacen ACÁ, antes del split, y no más abajo: definir qué filas son datos
# válidos delimita el universo del problema, no aprende un parámetro de los datos. Todo filtrado de
# filas tiene que quedar cerrado antes de partir en train/test, para que ambos folds salgan del
# mismo universo y los índices no se muevan después.

sin_coordenadas = deptos_df['latitud'].isna() | deptos_df['longitud'].isna()
print(f'Eliminamos {sin_coordenadas.sum()} casa(s) sin coordenadas (no imputable por vecino cercano).')
deptos_df = deptos_df.drop(deptos_df[sin_coordenadas].index)

fuera_de_rm = deptos_df['longitud'] > -69.5
print(f'Eliminamos {fuera_de_rm.sum()} casa(s) con coordenadas fuera de la Región Metropolitana.')
deptos_df = deptos_df.drop(deptos_df[fuera_de_rm].index)

# Duplicados:
# una misma casa puede quedar publicada más de una vez (mismo corredor republicando el aviso,
# distintas inmobiliarias vendiendo la misma propiedad, etc.). Si 'precio', 'latitud', 'longitud'
# y 'Superficie total' coinciden exactamente, es la misma casa repetida -- se mantiene la primera
# aparición y se descarta el resto.

columnas_duplicado = ['precio', 'latitud', 'longitud', 'Superficie total']
duplicados = deptos_df[deptos_df.duplicated(subset=columnas_duplicado, keep='first')]
print(f'Casas duplicadas: {len(duplicados)}')
deptos_df = deptos_df.drop(duplicados.index)

# Casi duplicados: la misma casa republicada por otro corredor, con el precio o la superficie
# levemente distintos (redondeo, un dato corregido), no cae en el match exacto de arriba. Se
# agrupan primero por ubicación casi idéntica (lat/lon redondeados a 4 decimales, ~10-20 m de
# radio -- alcanza para "el mismo lote", no para "la misma cuadra") y, dentro de cada grupo, se
# tratan como la misma casa las que además difieren en 2 m² o menos tanto en Superficie total como
# en Superficie útil. Se validó a mano contra 'dirección': los pares que caen en esta regla
# comparten calle (ej. "Av Vitacura 6560" y "Av Vitacura 6300 - 6600"); los que solo comparten
# ubicación redondeada pero tienen superficies muy distintas son casas vecinas reales, no duplicados,
# y la regla los deja afuera.


def encontrar_casi_duplicados(df: pd.DataFrame, tolerancia_m2: float = 2.0) -> pd.Index:
    """Índices a descartar: dentro de cada grupo con la misma lat/lon redondeada, se mantiene la
    primera aparición y se marcan como casi duplicadas las filas cuya Superficie total y
    Superficie útil están a `tolerancia_m2` o menos de alguna fila ya vista en el grupo."""
    lat_redondeada = df['latitud'].round(4)
    lon_redondeada = df['longitud'].round(4)
    a_descartar = []
    for _, grupo in df.groupby([lat_redondeada, lon_redondeada]):
        if len(grupo) < 2:
            continue
        vistas = []
        for idx, fila in grupo.iterrows():
            es_casi_duplicada = any(
                abs(fila['Superficie total'] - vista['Superficie total']) <= tolerancia_m2 and
                abs(fila['Superficie útil'] - vista['Superficie útil']) <= tolerancia_m2
                for vista in vistas
            )
            if es_casi_duplicada:
                a_descartar.append(idx)
            else:
                vistas.append(fila)
    return pd.Index(a_descartar)


casi_duplicados = encontrar_casi_duplicados(deptos_df)
print(f'Casas casi duplicadas (misma ubicación y superficie, distinto corredor/precio): {len(casi_duplicados)}')
deptos_df = deptos_df.drop(casi_duplicados)

# =======================================================================================
# PASO 2b -- SPLIT TRAIN/TEST (antes de cualquier imputación que aprenda de los datos)
# =======================================================================================
# Hasta acá todo lo hecho es determinista fila por fila (parseo de texto, swap útil/total, signo
# de coordenadas) o filtrado de filas (sin precio, sin superficie, sin coordenadas, fuera de la
# RM, duplicados). Nada de eso estima un parámetro a partir de la distribución de los datos, así
# que puede correr sobre el dataset completo sin contaminar nada.
#
# Lo que viene después SÍ aprende: modas, medianas, tablas cruzadas, umbrales de outlier y árboles
# de vecino más cercano. Si esos parámetros se calculan sobre el dataset entero, las filas de test
# participan de la estadística que después se usa para completarlas, y la métrica final queda
# optimista sin que se pueda saber en cuánto. Por eso el split va acá y no en PASO 4: de este
# punto en adelante, todo lo que se estima se estima SOLO con las filas de `indice_train`.
#
# El deduplicado tiene que ir antes del split, no después: la misma casa publicada dos veces,
# repartida una copia a cada fold, es fuga directa -- el modelo vería en entrenamiento la
# respuesta exacta de una fila de test.
#
# Estratificación por 'comuna': los tres niveles de precio son muy distintos y ambos folds deben
# representarlos en la misma proporción. Las comunas todavía nulas (el artefacto del scraper que
# se acaba de poner en nulo) forman su propio estrato 'Desconocida' para no perder esas filas ni
# romper el estratificador -- se imputan más abajo, ya con el split hecho.

from sklearn.model_selection import train_test_split

indice_train, indice_test = train_test_split(
    deptos_df.index,
    test_size=0.2,
    stratify=deptos_df['comuna'].fillna('Desconocida'),
    random_state=42,
)
print(f'Split previo a la limpieza que aprende -> train: {len(indice_train)} filas. '
      f'test: {len(indice_test)} filas.')

# Snapshot post-PASO 2 (filtrado/deduplicado), pre-imputación: PASO 5a (más abajo) reutiliza esta
# copia para reajustar TODA la imputación de este bloque con los índices de train de cada fold de
# validación cruzada, en vez de reusar los parámetros ajustados para este split fijo 80/20.
deptos_df_limpio = deptos_df.copy()

from scipy.spatial import cKDTree
from datetime import datetime

import seaborn as sns

COLUMNAS_TARGET = ['precio', 'clp', 'precio unitario']
ORIENTACION_VECINO_MAX_M = 150  # ponytail: umbral fijo (~una cuadra corta); ajustar si el sector tiene lotes más grandes


def haversine_m(lat1, lon1, lat2, lon2):
    """Distancia en línea recta (metros) entre dos puntos lat/lon."""
    radio_tierra_m = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    return 2 * radio_tierra_m * np.arcsin(np.sqrt(a))


def _nulificar_antiguedad_atipica(df: pd.DataFrame) -> None:
    """Año de construcción -> antigüedad (cuantil 75% ~20 años, pero el máximo puede ser un año
    como 2013), y descarte de valores imposibles (>1.000 años restantes o negativos). Regla
    determinista y universal -- no depende de qué filas son train, así que corre igual dentro del
    ajuste sobre train (para que bad_ranges/corr_sin_target vean el dato ya corregido, igual que en
    el código original) y sobre cualquier dataframe en aplicar_imputacion(). Muta `df` in-place."""
    year = datetime.now().year
    df['Antigüedad'] = np.where(df['Antigüedad'] >= 1800, year - df['Antigüedad'], df['Antigüedad'])
    df['Antigüedad'] = np.where((df['Antigüedad'] > 1000) | (df['Antigüedad'] < 0), np.nan, df['Antigüedad'])


def _nulificar_negativos_y_ceros(df: pd.DataFrame) -> None:
    """Ningún atributo de conteo puede ser negativo, y ninguno salvo `zero_ok_columns` puede ser
    cero -- salvo latitud/longitud, negativas por definición en Santiago. Regla determinista y
    universal, mismo motivo que _nulificar_antiguedad_atipica. Muta `df` in-place."""
    coordenadas = {'latitud', 'longitud'}
    zero_ok_columns = {'Bodegas', 'Estacionamientos', 'Antigüedad'}
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]) or column in coordenadas:
            continue
        invalid = df[column] < 0
        if column not in zero_ok_columns:
            invalid |= df[column] == 0
        if invalid.any():
            df.loc[invalid, column] = np.nan


def _binarizar_amenities(df: pd.DataFrame) -> None:
    """'Sí' -> 1, cualquier otro valor (incluido nulo) -> 0, para las columnas de amenities
    (`binary_features`, calculado en PASO 2 sobre el dataframe crudo -- ver más arriba). Regla
    determinista y universal: no depende de train, pero tiene que correr ANTES de calcular
    `corr_sin_target`, que necesita estas columnas ya numéricas. Muta `df` in-place."""
    for binary_feature in binary_features.columns:
        df[binary_feature] = (df[binary_feature] == 'Sí').astype(int)


@dataclass
class AllocateRule:
    """Una regla congelada de allocate_values(): probar `match_attribute` para resolver nulos de
    un atributo discreto. 'discrete': `mapping` es la moda de una tabla cruzada (valor de
    match_attribute -> valor imputado). 'continuous': `mapping` son las medianas agrupadas (valor
    del atributo -> mediana de match_attribute), y se imputa el grupo cuya mediana esté más cerca
    del match_attribute de la fila pendiente."""
    match_attribute: str
    kind: str
    mapping: pd.Series


@dataclass
class ImputacionParams:
    """Todo lo que PASO 2b aprende de `indice_train`. Los cKDTree exponen sus puntos de fit en
    `.data`, así que no hace falta guardar coordenadas de los vecinos conocidos por separado --
    solo el árbol y el array de valores conocidos, alineado al mismo orden que `.data`."""
    barrio_a_comuna: pd.Series
    comunas_validas: set
    tree_comuna: object
    valores_comuna: np.ndarray
    tree_barrio: object
    valores_barrio: np.ndarray
    bad_range_cuts: dict
    allocate_rules: dict = field(default_factory=dict)
    allocate_modo_fallback: dict = field(default_factory=dict)
    tree_orientacion: object = None
    valores_orientacion: np.ndarray = None
    moda_orientacion_por_barrio: pd.Series = None
    moda_orientacion_global: object = None


def _fit_kdtree_conocido(df: pd.DataFrame, column: str):
    """cKDTree fit con las filas de `df` que ya tienen `column` y coordenadas conocidas. Devuelve
    (árbol, valores) o (None, None) si no hay filas conocidas."""
    has_coords = df['latitud'].notna() & df['longitud'].notna()
    known = df.loc[df[column].notna() & has_coords]
    if known.empty:
        return None, None
    tree = cKDTree(known[['latitud', 'longitud']].to_numpy())
    return tree, known[column].to_numpy()


def _aplicar_kdtree_conocido(df: pd.DataFrame, column: str, tree, valores) -> None:
    """Imputa `column` en `df` con el valor del vecino conocido más cercano (distancia euclidiana
    en grados -- a la escala de Santiago el orden de vecino-más-cercano no cambia). Sin filtro de
    distancia máxima, a diferencia de la imputación de Orientación más abajo. Muta `df` in-place."""
    if tree is None:
        return
    has_coords = df['latitud'].notna() & df['longitud'].notna()
    missing = df.loc[df[column].isna() & has_coords]
    if missing.empty:
        return
    _, nearest_idx = tree.query(missing[['latitud', 'longitud']].to_numpy())
    df.loc[missing.index, column] = valores[nearest_idx]


def ajustar_imputacion(df: pd.DataFrame, indice_train: pd.Index, verbose: bool = True) -> ImputacionParams:
    """Aprende TODOS los parámetros de la imputación de PASO 2b usando solo `indice_train`. Corre
    la secuencia sobre una copia aislada de esas filas (`train`) -- el resultado sobre filas de
    train es idéntico al del código original (cada paso ya filtraba "conocido" a indice_train), y
    de paso deja fijo cada parámetro aprendido para poder reproducirlo en aplicar_imputacion()
    sobre cualquier dataframe (el fold completo train+val, o el dataset entero en la corrida
    única). Los pasos deterministas que ocurren DESPUÉS de allocate_values() en el código original
    (drop de 'Unidades totales'/'dirección', fills de 'Gastos comunes'/'Tipo de casa'/
    'descripcion') no afectan ningún parámetro aprendido más abajo (Orientación no los usa), así
    que no hace falta replicarlos acá -- solo en aplicar_imputacion(), donde sí producen el
    dataframe final."""
    # Boolean mask, no fancy-indexing por `indice_train`: .loc[lista] reordena las filas según el
    # orden de la lista (que train_test_split entrega barajado), mientras que el máscara booleana
    # preserva el orden original de `df`. Importa para reproducibilidad bit-a-bit: los cKDTree de
    # más abajo indexan sus puntos por posición, y con coordenadas duplicadas/casi-duplicadas
    # (ver "casi duplicados" de PASO 2) un desempate por índice interno distinto puede asignar un
    # vecino distinto aunque el CONJUNTO de puntos conocidos sea idéntico.
    train = df.loc[df.index.isin(indice_train)].copy()

    barrios_conocidos = train.dropna(subset=['barrio', 'comuna'])
    barrio_a_comuna = barrios_conocidos.groupby('barrio')['comuna'].agg(lambda s: s.value_counts().idxmax())
    comunas_validas = set(train['comuna'].dropna().unique())

    comuna_nula = train['comuna'].isna()
    desde_barrio = comuna_nula & train['barrio'].isin(barrio_a_comuna.index)
    train.loc[desde_barrio, 'comuna'] = train.loc[desde_barrio, 'barrio'].map(barrio_a_comuna)
    barrio_es_comuna = train['comuna'].isna() & train['barrio'].isin(comunas_validas)
    train.loc[barrio_es_comuna, 'comuna'] = train.loc[barrio_es_comuna, 'barrio']
    train.loc[barrio_es_comuna, 'barrio'] = np.nan
    if verbose:
        print(f'[ajustar_imputacion] Comunas imputadas desde el barrio (train): {desde_barrio.sum()}. '
              f'Barrios que eran nombre de comuna: {barrio_es_comuna.sum()}.')

    tree_comuna, valores_comuna = _fit_kdtree_conocido(train, 'comuna')
    _aplicar_kdtree_conocido(train, 'comuna', tree_comuna, valores_comuna)

    train['barrio'] = train['barrio'].str.title()
    tree_barrio, valores_barrio = _fit_kdtree_conocido(train, 'barrio')
    _aplicar_kdtree_conocido(train, 'barrio', tree_barrio, valores_barrio)

    _nulificar_antiguedad_atipica(train)

    bad_ranges = [c for c in ['Baños', 'Estacionamientos', 'Bodegas', 'Cantidad de pisos', 'Antigüedad']
                  if c in train.columns]
    bad_range_cuts = {}
    for column in bad_ranges:
        top_20 = train[column].dropna().nlargest(20).sort_values()
        cut_value = cut_after_relative_jump(top_20, threshold=1.0)
        if cut_value is not None:
            bad_range_cuts[column] = cut_value
            train.loc[train[column] >= cut_value, column] = np.nan

    _nulificar_negativos_y_ceros(train)
    _binarizar_amenities(train)

    corr_sin_target = (train.drop(columns=COLUMNAS_TARGET, errors='ignore')
                       .corr(numeric_only=True)
                       .dropna(axis=0, how='all').dropna(axis=1, how='all'))

    def _atributos_correlacionados(attribute):
        columna = corr_sin_target[attribute]
        return columna[columna.index != attribute].sort_values(ascending=False, key=lambda x: abs(x)).index

    discrete_values = [c for c in ['Dormitorios', 'Baños', 'Estacionamientos', 'Bodegas',
                                     'Cantidad de pisos', 'Antigüedad'] if c in train.columns]
    allocate_rules = {}
    allocate_modo_fallback = {}
    for attribute in discrete_values:
        reglas_atributo = []
        pending_index = train[train[attribute].isna()].index
        for match_attribute in _atributos_correlacionados(attribute):
            if pending_index.empty:
                break
            if match_attribute in discrete_values:
                cross_table = pd.crosstab(train[match_attribute], train[attribute])
                if cross_table.empty:
                    continue
                modes = cross_table.idxmax(axis=1)
                match_values = train.loc[pending_index, match_attribute]
                resolvable = match_values[match_values.isin(modes.index)]
                train.loc[resolvable.index, attribute] = resolvable.map(modes)
                pending_index = pending_index.difference(resolvable.index)
                reglas_atributo.append(AllocateRule(match_attribute, 'discrete', modes))
            else:
                grouped_medians = train.groupby(attribute)[match_attribute].median().dropna()
                if grouped_medians.empty:
                    continue
                match_values = train.loc[pending_index, match_attribute].dropna()
                nearest = match_values.apply(lambda v: (grouped_medians - v).abs().idxmin())
                train.loc[nearest.index, attribute] = nearest
                pending_index = pending_index.difference(nearest.index)
                reglas_atributo.append(AllocateRule(match_attribute, 'continuous', grouped_medians))
        if not pending_index.empty:
            mode = train[attribute].mode()
            if not mode.empty:
                allocate_modo_fallback[attribute] = mode.iloc[0]
                train.loc[pending_index, attribute] = mode.iloc[0]
        allocate_rules[attribute] = reglas_atributo
        if verbose:
            print(f'[ajustar_imputacion] {attribute}: {train[attribute].isna().sum()} nulos restantes en train.')

    tree_orientacion, valores_orientacion = _fit_kdtree_conocido(train, 'Orientación')
    if tree_orientacion is not None:
        has_coords = train['latitud'].notna() & train['longitud'].notna()
        missing = train.loc[train['Orientación'].isna() & has_coords]
        if not missing.empty:
            _, nearest_idx = tree_orientacion.query(missing[['latitud', 'longitud']].to_numpy())
            nearest_coords = tree_orientacion.data[nearest_idx]
            distance_m = haversine_m(missing['latitud'].to_numpy(), missing['longitud'].to_numpy(),
                                      nearest_coords[:, 0], nearest_coords[:, 1])
            close_enough = distance_m <= ORIENTACION_VECINO_MAX_M
            train.loc[missing.index[close_enough], 'Orientación'] = valores_orientacion[nearest_idx][close_enough]

    moda_orientacion_por_barrio = (train.groupby('barrio')['Orientación']
                                   .agg(lambda s: s.mode().iat[0] if not s.mode().empty else np.nan))
    orientacion_na = train['Orientación'].isna()
    train.loc[orientacion_na, 'Orientación'] = train.loc[orientacion_na, 'barrio'].map(moda_orientacion_por_barrio)
    orientacion_na = train['Orientación'].isna()
    moda_orientacion_global = train['Orientación'].mode().iat[0] if orientacion_na.any() else None
    if orientacion_na.any():
        train.loc[orientacion_na, 'Orientación'] = moda_orientacion_global
        if verbose:
            print(f'[ajustar_imputacion] {orientacion_na.sum()} casas de train sin barrio ni Orientación: '
                  f'moda global ({moda_orientacion_global}).')

    return ImputacionParams(
        barrio_a_comuna=barrio_a_comuna, comunas_validas=comunas_validas,
        tree_comuna=tree_comuna, valores_comuna=valores_comuna,
        tree_barrio=tree_barrio, valores_barrio=valores_barrio,
        bad_range_cuts=bad_range_cuts,
        allocate_rules=allocate_rules, allocate_modo_fallback=allocate_modo_fallback,
        tree_orientacion=tree_orientacion, valores_orientacion=valores_orientacion,
        moda_orientacion_por_barrio=moda_orientacion_por_barrio,
        moda_orientacion_global=moda_orientacion_global,
    )


def aplicar_imputacion(df: pd.DataFrame, params: ImputacionParams, verbose: bool = True) -> pd.DataFrame:
    """Reproduce sobre una COPIA de `df` (nunca muta el dataframe recibido -- indispensable para
    que los folds de PASO 5a no se contaminen entre sí) la imputación que `params` aprendió de
    train. Mismo orden de pasos que el código original de PASO 2b."""
    df = df.copy()

    comuna_nula = df['comuna'].isna()
    desde_barrio = comuna_nula & df['barrio'].isin(params.barrio_a_comuna.index)
    df.loc[desde_barrio, 'comuna'] = df.loc[desde_barrio, 'barrio'].map(params.barrio_a_comuna)
    barrio_es_comuna = df['comuna'].isna() & df['barrio'].isin(params.comunas_validas)
    df.loc[barrio_es_comuna, 'comuna'] = df.loc[barrio_es_comuna, 'barrio']
    df.loc[barrio_es_comuna, 'barrio'] = np.nan
    if verbose:
        print(f'Comunas imputadas desde el barrio: {desde_barrio.sum()}. '
              f'Barrios que eran nombre de comuna: {barrio_es_comuna.sum()}.')
        print(f'Quedan {df["comuna"].isna().sum()} comunas nulas para el vecino más cercano.')

    _aplicar_kdtree_conocido(df, 'comuna', params.tree_comuna, params.valores_comuna)
    if verbose:
        print(f'Ahora hay {df["comuna"].isna().sum()} comunas con valores nulos')

    df['barrio'] = df['barrio'].str.title()
    if verbose:
        print(f'Hay {df["barrio"].isna().sum()} casas sin barrio.')
    _aplicar_kdtree_conocido(df, 'barrio', params.tree_barrio, params.valores_barrio)
    if verbose:
        print(f'Ahora hay {df["barrio"].isna().sum()} casas sin barrio.')

    _nulificar_antiguedad_atipica(df)

    for column, cut_value in params.bad_range_cuts.items():
        idx = df[df[column] >= cut_value].index
        if verbose:
            print(f'Asignamos a nan {len(idx)} valores de la columna "{column}" (>= {cut_value})')
        df.loc[idx, column] = np.nan

    _nulificar_negativos_y_ceros(df)
    _binarizar_amenities(df)

    for attribute, reglas in params.allocate_rules.items():
        pending_index = df[df[attribute].isna()].index
        for regla in reglas:
            if pending_index.empty:
                break
            if regla.kind == 'discrete':
                match_values = df.loc[pending_index, regla.match_attribute]
                resolvable = match_values[match_values.isin(regla.mapping.index)]
                df.loc[resolvable.index, attribute] = resolvable.map(regla.mapping)
                pending_index = pending_index.difference(resolvable.index)
            else:
                match_values = df.loc[pending_index, regla.match_attribute].dropna()
                nearest = match_values.apply(lambda v: (regla.mapping - v).abs().idxmin())
                df.loc[nearest.index, attribute] = nearest
                pending_index = pending_index.difference(nearest.index)
        if not pending_index.empty and attribute in params.allocate_modo_fallback:
            df.loc[pending_index, attribute] = params.allocate_modo_fallback[attribute]
        if verbose:
            print(f'{attribute}: {df[attribute].isna().sum()} nulos restantes')

    unidades_totales_pct_null = df['Unidades totales'].isna().mean() * 100
    if verbose:
        print(f'"Unidades totales": {unidades_totales_pct_null:.1f}% nulo (solo aplica a fichas de proyecto, '
              f'no a casas individuales). No es imputable de forma confiable: se descarta la columna.')
    df = df.drop(columns=['Unidades totales'])
    df = df.drop(columns=['dirección'])
    df['Gastos comunes'] = df['Gastos comunes'].apply(parse_measurement).fillna(0)
    df['Tipo de casa'] = df['Tipo de casa'].fillna('Casa')
    df['descripcion'] = df['descripcion'].fillna(df['titulo'])

    if params.tree_orientacion is not None:
        has_coords = df['latitud'].notna() & df['longitud'].notna()
        missing = df.loc[df['Orientación'].isna() & has_coords]
        if not missing.empty:
            _, nearest_idx = params.tree_orientacion.query(missing[['latitud', 'longitud']].to_numpy())
            nearest_coords = params.tree_orientacion.data[nearest_idx]
            distance_m = haversine_m(missing['latitud'].to_numpy(), missing['longitud'].to_numpy(),
                                      nearest_coords[:, 0], nearest_coords[:, 1])
            close_enough = distance_m <= ORIENTACION_VECINO_MAX_M
            df.loc[missing.index[close_enough], 'Orientación'] = params.valores_orientacion[nearest_idx][close_enough]
    if verbose:
        print(f'Orientación: {df["Orientación"].isna().sum()} nulos restantes tras imputar por vecino cercano.')

    orientacion_na = df['Orientación'].isna()
    df.loc[orientacion_na, 'Orientación'] = df.loc[orientacion_na, 'barrio'].map(params.moda_orientacion_por_barrio)
    orientacion_na = df['Orientación'].isna()
    if orientacion_na.any() and params.moda_orientacion_global is not None:
        if verbose:
            print(f'{orientacion_na.sum()} casas sin barrio ni Orientación: '
                  f'se imputan con la moda global ({params.moda_orientacion_global}).')
        df.loc[orientacion_na, 'Orientación'] = params.moda_orientacion_global
    if verbose:
        print(f'Orientación: {df["Orientación"].isna().sum()} nulos restantes')

    return df


params_imputacion = ajustar_imputacion(deptos_df, indice_train)
deptos_df = aplicar_imputacion(deptos_df, params_imputacion)

# Diagnóstico descriptivo (no alimenta ninguna decisión del modelo -- ver más abajo): nulos por
# columna y matriz de correlación sobre el dataset YA imputado completo, con el bloque de target
# incluido a propósito (entender qué correlaciona con el precio es el objetivo del EDA de PASO 3).

na_values = deptos_df.isnull().sum()
percent_na = ((na_values / len(deptos_df)) * 100).round(2)
na_summary = pd.concat([na_values, percent_na], axis=1)
na_summary.columns = ['Cantidad de Nulos', 'Porcentaje de Nulos']
na_summary = na_summary[na_summary['Cantidad de Nulos'] > 0]
na_summary.sort_values('Cantidad de Nulos', ascending=False, inplace=True)
print(na_summary)

sns.set(style='whitegrid')
plt.figure(figsize=(16, 8))
sns.set(font_scale=.75)
if not na_summary.empty:
    ax = sns.barplot(x='Porcentaje de Nulos', y=na_summary.index, data=na_summary)
    plt.title('Número de valores nulos por variable')
    plt.xlabel('Porcentaje valores nulos')
    plt.ylabel('Atributos')
    ax.set_xlim(0, 100)
    plt.savefig(os.path.join(GRAFICOS_DIR, 'valores_nulos_por_variable.png'))

corr = deptos_df.corr(numeric_only=True)
corr = corr.dropna(axis=0, how='all').dropna(axis=1, how='all')
plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap="Blues", annot=True)
plt.savefig(os.path.join(GRAFICOS_DIR, 'correlacion_atributos.png'))

# =======================================================================================
# PASO 3 -- EXPLORACIÓN
# =======================================================================================
# Con los datos limpios, se exploran las relaciones entre precio unitario y el resto de los
# atributos, para entender qué features van a aportarle señal al modelo de regresión.

# Exploración:
# con los datos limpios, exploramos el precio por metro cuadrado por comuna y su relación con la
# superficie.

g = sns.catplot(x='precio unitario', y='comuna', data=deptos_df, kind='box', showfliers=False)
g.savefig(os.path.join(GRAFICOS_DIR, 'precio_unitario_por_comuna.png'))


# hipótesis: a medida que aumenta la superficie útil, disminuye el precio unitario
#
# Unas pocas casas con terrenos enormes (ej. un lote de 400.000 m²) estiran tanto el eje de
# Superficie total que los datos centrales quedan aplastados contra el borde inferior. Se filtra
# al rango intercuartílico de ambas variables (mismo criterio de percentil_limits ya usado para
# los outliers de precio unitario) solo para esta visualización, sin tocar deptos_df.

precio_lower, precio_upper = percentil_limits(deptos_df['precio unitario'], (0.25, 0.75))
superficie_lower, superficie_upper = percentil_limits(deptos_df['Superficie total'], (0.25, 0.75))
plot_data = deptos_df[
    deptos_df['precio unitario'].between(precio_lower, precio_upper) &
    deptos_df['Superficie total'].between(superficie_lower, superficie_upper)
]
g = sns.relplot(x='precio unitario', y='Superficie total', data=plot_data, kind='scatter', hue='comuna')
g.savefig(os.path.join(GRAFICOS_DIR, 'precio_unitario_vs_superficie.png'))


# ¿El precio unitario varía según la cantidad de dormitorios? Útil para decidir si conviene
# incluir Dormitorios como feature categórica en el modelo, más allá de su correlación lineal.

g = sns.catplot(x='Dormitorios', y='precio unitario', data=deptos_df, kind='box', showfliers=False,
                order=sorted(deptos_df['Dormitorios'].dropna().unique()))
g.savefig(os.path.join(GRAFICOS_DIR, 'precio_unitario_por_dormitorios.png'))


# hipótesis: una casa más antigua debería tener un precio unitario menor. El heatmap ya muestra
# una correlación casi nula (~0) entre Antigüedad y precio unitario; este gráfico lo confirma
# visualmente en vez de solo en un número. Se filtra al rango intercuartílico de Antigüedad por el
# mismo motivo que en el gráfico anterior: un puñado de casas muy antiguas o muy nuevas no debería
# aplastar el resto de los puntos.

antiguedad_lower, antiguedad_upper = percentil_limits(deptos_df['Antigüedad'], (0.25, 0.75))
plot_data_antiguedad = deptos_df[deptos_df['Antigüedad'].between(antiguedad_lower, antiguedad_upper)]
g = sns.lmplot(x='Antigüedad', y='precio unitario', data=plot_data_antiguedad,
               scatter_kws={'alpha': 0.4}, line_kws={'color': 'darkred'})
g.savefig(os.path.join(GRAFICOS_DIR, 'precio_unitario_vs_antiguedad.png'))


# Generamos un CSV con los datos limpios, para no tener que repetir todo el proceso de limpieza cada vez.
# Se ordena por precio unitario ascendente por defecto.
deptos_df.sort_values('precio unitario').to_excel('deptos_limpios.xlsx', index=False)

# Análisis de la base ya limpia:
# antes de diseñar el preprocesamiento (PASO 4), revisamos la forma final del dataset -- dtypes,
# nulos y estadísticos descriptivos -- con datos medidos, no supuestos.

print(deptos_df.info())
print(deptos_df.describe())

# Lo que llama la atención de info() (5.284 filas x 74 columnas, tras descartar la única casa sin
# coordenadas, los casi duplicados y la publicación con coordenadas en Mendoza -- ver PASO 2): el
# dataset quedó 100% completo, sin nulos en ninguna columna.
# - 52 columnas son int64 en {0, 1} (los amenities binarios). Con 'Sí' -> 1 / resto -> 0 en vez de
#   regresión logística, ya ninguna quedó constante, pero sí muy desbalanceada: 4 amenities
#   (Cancha de básquetbol, Con cancha polideportiva, Cancha de paddle, Con cancha de fútbol) están
#   presentes en menos del 1% de las casas (3 a 29 de 5.284) y aportan casi nada de señal.
#
# Lo que llama la atención de describe():
# - 'precio' tiene skew ~13 y rango de 3.790 a 1.240.000.000: mezcla UF y CLP sin convertir según
#   'UM', así que sus estadísticos no son comparables entre sí -- 'clp' (siempre en pesos) es la
#   columna homogénea.
# - 'Superficie útil' (skew ~70, máx 250.000 m²) y 'Superficie total' (skew ~59, máx 400.000 m²)
#   siguen con la asimetría extrema que documenta el PASO 4 más abajo: sin recorte de outliers,
#   esos máximos son lotes/proyectos atípicos reales, no errores, pero exigen log1p antes de
#   cualquier modelo lineal.
# - 'Antigüedad': el caso de 939 años (imposible, ver "Rangos de valores" en PASO 2) ya se anula
#   ahí mismo con el mismo mecanismo de salto relativo que usan Baños/Estacionamientos/Bodegas/
#   Cantidad de pisos, y se reimputa como el resto de las discretas -- máximo actual 371 años.
#   Sigue siendo alto para un valor real, pero ya no es un error de tipeo evidente como 939; el
#   recorte a un rango plausible (0-120) queda para el escalado del PASO 4, no para la limpieza.
# - 'Cantidad de pisos' llega a 25 -- valor extremo que ya no se descarta al no recortar
#   outliers, y queda pendiente para el PASO 4.
# - 'Gastos comunes' es 0 en 83,7 % de las filas (el valor con el que se rellenó el nulo): conviene
#   tratarla también como binaria "tiene o no gasto común", además de su valor continuo.

# =======================================================================================
# PASO 4 -- ESTRATEGIA DE PREPROCESAMIENTO PARA EL MODELO (diseño, aún no implementado)
# =======================================================================================
#
# Punto de partida medido sobre deptos_limpios.xlsx (5.284 filas x 74 columnas, ver análisis de
# info()/describe() más arriba), no supuesto: sin nulos en ninguna columna, pero con tres
# problemas que condicionan todo lo que sigue.
#
#   (a) Amenities binarios MUY desbalanceados. Con 'Sí' -> 1 / resto -> 0 (ver imputación más
#       arriba) ninguna columna quedó constante, pero varias están cerca: 4 amenities (Cancha de
#       básquetbol, Con cancha polideportiva, Cancha de paddle, Con cancha de fútbol) aparecen en
#       menos del 1% de las casas (3 a 29 de 5.284). Aportan casi nada de señal y son candidatas a
#       agruparse en una única feature ("tiene cancha deportiva") o descartarse.
#   (b) Asimetría extrema en las superficies: skew de 70 en 'Superficie útil' (máx 250.000 m²)
#       y 59 en 'Superficie total' (máx 400.000 m²), más 'Cantidad de pisos' con skew 12 (máx
#       25). La limpieza ya no recorta outliers de ninguna columna (ni siquiera 'precio
#       unitario'), así que todas siguen sucias. 'Antigüedad' ya bajó de skew 16 (máx 939, un
#       error de tipeo evidente, anulado y reimputado en la limpieza) a skew ~2,1 (máx 371) --
#       sigue alta para un valor real, así que igual se acota a 0-120 antes de estandarizar.
#   (c) Fuga de target: 'precio', 'clp', 'precio unitario' y 'UM' son todas el target o
#       transformaciones suyas. 'UM' además codifica el tramo de precio (solo las publicaciones
#       caras se listan en UF), así que es un proxy y no una feature.
#
# ORDEN DE OPERACIONES (no negociable)
# ------------------------------------
# Primero el split, después todo lo demás. Cada paso que "aprende" algo de los datos (media
# y desvío del escalado, categorías del encoder, vocabulario e IDF del texto, medianas de
# imputación, umbral de colinealidad) se ajusta SOLO con el fold de entrenamiento y se aplica
# al de test. Split estratificado por 'comuna': los tres niveles de precio son muy distintos y
# hay que asegurar que ambos folds los representen.
#
# Esto ya está implementado, y el split no vive acá sino en PASO 2b: la limpieza de PASO 2 también
# aprende de los datos (modas, medianas, tablas cruzadas, KDTree, umbrales de outlier), así que
# partir recién en PASO 4 dejaba a test participando de su propia imputación. Lo que queda antes
# del split es solo transformación determinista fila por fila y filtrado de filas.
#
# Efecto medido de haberlo corregido: MdAPE 8,43% -> 8,39%, prácticamente sin cambio. Vale la pena
# dejarlo escrito, porque el resultado no era el esperado: la contaminación existía pero casi no
# inflaba la métrica. La razón es que las columnas con muchos nulos (Bodegas con 1.446, Cantidad de
# pisos con 708) resultaron tener importancia por permutación cercana a cero, mientras que las
# features que dominan el modelo ('Superficie total', y 'm2_por_dormitorio' que se deriva de ella)
# no tenían nulos que imputar. La corrección igual se justifica: sin ella, ningún número del
# proyecto era defendible, y no había forma de saber de antemano que el sesgo era chico.
#
# 1. ESCALADO DE FEATURES (normalización vs. estandarización)
# -----------------------------------------------------------
# La decisión no es una sola para todo el dataset, va por forma de la distribución:
#
#   - Superficies ('Superficie útil', 'Superficie total') y 'Gastos comunes' (skew 7,4):
#     log1p PRIMERO, StandardScaler DESPUÉS. Estandarizar una distribución con skew 69 no
#     la arregla -- deja la masa de los datos aplastada en un rango mínimo y el outlier a 40
#     desvíos igual de dominante. El log es lo que corrige la forma; el escalado solo centra.
#     Además el log es lo correcto por teoría: en un modelo hedónico el precio responde a la
#     superficie de forma multiplicativa, y log(precio) ~ log(superficie) da directamente la
#     elasticidad (ver cabecera del archivo).
#   - Conteos discretos y acotados ('Dormitorios' 1-14, 'Baños' 1-18, 'Estacionamientos',
#     'Bodegas', 'Cantidad de pisos'): StandardScaler a secas. Ya son de rango chico y
#     aproximadamente simétricos (skew < 1,2 en dormitorios y baños); el log no aporta.
#   - 'Antigüedad': acotar primero a un rango plausible (0-120 años; el caso de 939 ya se anuló
#     y reimputó en la limpieza, pero quedan valores de 140-371 años que siguen siendo demasiado
#     altos para una casa real) y luego estandarizar.
#   - 'latitud'/'longitud': NO escalar por separado ni tratarlas como dos features numéricas
#     independientes -- ver punto 4, se convierten en features de distancia.
#   - Binarias 0/1: no se escalan, ya están en [0,1].
#
#   Normalización (MinMax) queda descartada como default: es sensible al máximo, y con
#   máximos de 400.000 m² comprimiría el 99% de los datos a un rango casi nulo. Solo tendría
#   sentido si se cambia a una red neuronal, y aun así después del log. Si tras el log
#   quedaran colas pesadas, la alternativa robusta es RobustScaler (usa mediana e IQR, que es
#   el mismo criterio de percentil_limits() que ya usa este script).
#
#   Nota: para el modelo de árboles con boosting el escalado es indiferente (los splits son
#   invariantes a transformaciones monótonas). Se mantiene igual para que ambos modelos
#   compartan el mismo ColumnTransformer, y porque el log sí cambia lo que aprende el lineal.
#
# 2. CODIFICACIÓN DE CATEGÓRICAS
# -------------------------------
# Medida la cardinalidad real, aquí no hay ninguna categórica verdaderamente alta:
#
#   - Baja cardinalidad -> One-Hot con drop='first' (evita la trampa de la variable ficticia
#     en el modelo lineal) y handle_unknown='infrequent_if_exist':
#       * 'comuna' (3 niveles)
#       * 'Tipo de casa' (5: Casa 4.902, Chalet 491, Dúplex 162, Tríplex 41, Cabaña 1)
#         -> agrupar Tríplex y Cabaña en una categoría 'Otro'; con 1 sola cabaña, esa columna
#         one-hot es un identificador de fila disfrazado de feature.
#       * 'Orientación' (8 niveles). Es cíclica en teoría (N-NE-E-...), pero como categórica
#         de 8 niveles con 5.597 filas no vale la pena el seno/coseno; one-hot y listo.
#   - Cardinalidad media -> 'barrio' (41 niveles) es el caso a decidir. One-Hot lo deja en 41
#     columnas para 5.597 filas, y 7 barrios tienen menos de 30 casas (Puente Nuevo tiene 1).
#     Estrategia: Target Encoding sobre log(precio) con suavizado bayesiano hacia la media de
#     la comuna, ajustado dentro de CV anidado (out-of-fold) para no filtrar el target. El
#     barrio es la feature de ubicación más fuerte del dataset, y el target encoding conserva
#     ese orden de precios en una sola columna en vez de 41 dispersas. Los barrios con <30
#     casas colapsan casi por completo hacia la media de su comuna, que es exactamente el
#     comportamiento deseado. Alternativa más simple si el encoding out-of-fold da problemas
#     de fuga: One-Hot agrupando los 7 barrios raros en 'Otro' (35 columnas).
#   - Ordinal Encoding: no se usa. Ninguna categórica tiene orden natural, y asignarle uno
#     arbitrario (barrio 0..40) le inventaría al modelo lineal una relación monótona falsa.
#     Los conteos que sí son ordinales ('Dormitorios', 'Baños') ya vienen como enteros.
#   - 'UM': se descarta (fuga, ver punto (c) arriba).
#
# 3. VECTORIZACIÓN DE TEXTO
# --------------------------
# Hay dos campos de texto libre: 'titulo' (4.520 valores únicos) y 'descripcion' (5.486).
# El texto es la única fuente de señal para atributos que la ficha estructurada no captura:
# estado de conservación, remodelaciones, vista, calidad de terminaciones, urgencia de venta.
#
#   - Faltantes: ya resuelto aguas arriba -- 'descripcion' nula se rellena con 'titulo'
#     (siempre presente). Para el modelo se concatenan ambos campos en un solo documento
#     ('titulo' + ' ' + 'descripcion'), lo que además hace que las filas imputadas no queden
#     con el título duplicado pesando doble en el vector.
#   - Baseline: TF-IDF con ngram_range=(1,2), min_df=5 (descarta typos y direcciones únicas),
#     max_df=0.7 (descarta el boilerplate de la inmobiliaria, que se repite en miles de
#     avisos), lista de stopwords en español, y normalización previa de acentos y minúsculas.
#     Sobre eso, TruncatedSVD a ~50-100 componentes: la matriz TF-IDF dispersa se lleva mal
#     con los modelos de árboles y con una matriz densa de features numéricas.
#   - Alternativa a evaluar contra el baseline: embeddings de un modelo multilingüe
#     (p. ej. sentence-transformers, paraphrase-multilingual-MiniLM) que capturan sinónimos
#     que TF-IDF no ve ("impecable" ~ "excelente estado"). Con 5.597 documentos cortos y en
#     español el costo es bajo. Se adopta solo si mejora el MdAPE de forma medible; si no,
#     queda TF-IDF, que además es interpretable.
#   - Riesgo a controlar: la descripción suele mencionar el precio o la superficie en texto
#     ("vendo en 12.000 UF"). Eso es fuga directa del target. Antes de vectorizar hay que
#     borrar del documento los patrones numéricos de precio (UF, CLP, $, millones) con una
#     regex. Sin ese filtro el modelo "predice" leyendo la respuesta.
#
# 4. INGENIERÍA Y SELECCIÓN DE FEATURES
# --------------------------------------
# Features de dominio a construir:
#
#   - ratio_construido = Superficie útil / Superficie total. Distingue la casa grande en lote
#     chico de la casa chica en lote grande, que a igual superficie total valen muy distinto.
#     Es la feature que el target 'precio unitario' original borraba por construcción.
#   - superficie_terreno = Superficie total - Superficie útil (terreno no construido).
#   - baños_por_dormitorio y m2_por_dormitorio: proxies de estándar/calidad de la casa,
#     independientes del tamaño absoluto.
#   - Distancias reales (haversine) a los polos de valor del sector -- Clínica Alemana,
#     Estadio Español, Portal La Dehesa, los colegios del corredor Manquehue-La Dehesa --
#     calculadas desde 'latitud'/'longitud'. Esto reemplaza y mejora el antiguo campo
#     'cercania' 'Cerca'/'Lejos', que era una lista de barrios escrita a mano: una distancia
#     continua en metros conserva el gradiente que la etiqueta binaria tiraba a la basura.
#     Ahora es viable porque las coordenadas quedaron completas (99,98%) tras corregir el bug
#     del filtro de negativos.
#   - Alternativa/complemento a las distancias: distancia al centroide de la comuna, y
#     densidad local de la oferta (número de casas en un radio de 500 m vía cKDTree), que
#     aproxima "qué tan consolidado está el sector".
#   - 'Gastos comunes' > 0 como binaria (indica condominio con administración) además del
#     valor continuo: el 50% de las casas tiene 0, así que la columna es medio indicador y
#     medio monto.
#
# Selección y eliminación de redundancia:
#
#   - Eliminar de entrada las 37 columnas constantes. No aportan información, inflan la
#     matriz y ensucian cualquier ranking de importancia. Un VarianceThreshold(0) lo hace
#     automáticamente y sigue funcionando si un crawl futuro les da varianza.
#   - Eliminar identificadores: 'url' (5.597 valores únicos = una fila cada uno).
#   - Eliminar el bloque de fuga: 'precio', 'clp', 'precio unitario', 'UM'.
#   - Colinealidad: 'Superficie útil' y 'Superficie total' están fuertemente correlacionadas
#     entre sí y con 'Dormitorios'/'Baños' (el heatmap ya generado lo muestra). Para el modelo
#     lineal, calcular VIF y descartar iterativamente lo que supere ~10, o directamente
#     quedarse con log(Superficie total) + ratio_construido en vez de las dos superficies
#     crudas (misma información, sin la correlación de 0,9). Para el modelo de árboles la
#     colinealidad no rompe las predicciones, pero sí reparte la importancia entre features
#     gemelas y hace ilegible el SHAP, así que conviene igual.
#   - Selección final: importancia por permutación sobre el fold de validación (no la
#     importancia por impureza de sklearn, que sobrevalora las features de alta cardinalidad
#     como el target encoding de barrio). Descartar lo que tenga importancia no distinguible
#     de cero y reentrenar, verificando que el MdAPE no empeore.
#   - Feature de control obligatoria: comparar siempre contra el baseline de mediana de
#     precio/m² por barrio. Si toda esta ingeniería no le gana a esa línea de una sola
#     columna, el modelo no justifica su complejidad.

# =======================================================================================
# PASO 4a -- SELECCIÓN DE COLUMNAS DE ENTRENAMIENTO
# =======================================================================================
# Primer paso concreto del preprocesamiento: definir qué entra al modelo antes de imputar,
# escalar o codificar nada (eso se ajusta dentro del split, ver "ORDEN DE OPERACIONES" arriba).
# Target: 'clp' (se modela log(clp), ver cabecera del archivo). 'precio', 'precio unitario' y
# 'UM' quedan fuera por ser el target o transformaciones/proxies suyas -- 'precio unitario' en
# particular es clp / Superficie total, así que dejarla como feature junto a 'Superficie total'
# filtraría el target casi exactamente. Los binarios se derivan por dtype (igual que
# measurement_columns más arriba) en vez de listarlos a mano, porque son exactamente las
# columnas que la imputación de amenities dejó en 'Sí'/resto -> 1/0.

def seleccionar_columnas(df: pd.DataFrame, verbose: bool = True) -> tuple:
    """Extrae PASO 4a a función pura (sin fit, sin dependencia de indice_train) para que la
    corrida única y el driver de CV de PASO 5a compartan la misma selección de columnas."""
    columnas_binarias = [c for c in df.columns if df[c].dtype == 'int64']
    columnas_entrenamiento = [
        'latitud', 'longitud', 'descripcion',
        'Superficie útil', 'Superficie total',
        'Dormitorios', 'Baños', 'Estacionamientos', 'Bodegas',
        'Antigüedad', 'Cantidad de pisos',
        'Orientación', 'Gastos comunes',
        'comuna', 'barrio', 'Tipo de casa',
    ] + columnas_binarias
    if verbose:
        print(f'Columnas de entrenamiento: {len(columnas_entrenamiento)}')
    return df[columnas_entrenamiento].copy(), df['clp'].copy()


X, y = seleccionar_columnas(deptos_df)

# =======================================================================================
# PASO 4b -- CODIFICACIÓN DE CATEGÓRICAS
# =======================================================================================
# El split ya está hecho: se define en PASO 2b, antes de toda la limpieza que aprende de los
# datos. Acá solo se materializa sobre X/y usando esos mismos índices -- NO se vuelve a partir.
# Repartir de nuevo acá rompería la garantía de PASO 2b: filas que fueron test durante la
# imputación podrían caer en train ahora, y las medianas/modas con las que se las completó ya
# habrían sido calculadas sin ellas (o peor, con ellas, según cómo cayera el sorteo).
#
# El encoding de 'barrio' (Target Encoding) aprende del target, así que también necesita el split
# hecho antes de ajustarse: si se ajustara sobre todo X, el valor codificado de cada fila
# filtraría información de su propio target hacia adentro de la feature.

X_train, X_test = X.loc[indice_train], X.loc[indice_test]
y_train, y_test = y.loc[indice_train], y.loc[indice_test]
print(f'Train: {len(X_train)} filas. Test: {len(X_test)} filas.')

# Categóricas de baja cardinalidad ('comuna' 3 niveles, 'Tipo de casa' 4 tras agrupar,
# 'Orientación' 8): One-Hot con drop='first' (evita la trampa de la variable ficticia en el
# modelo lineal) y handle_unknown='infrequent_if_exist' (si test trae una categoría que no
# apareció en train, no rompe el transform). sparse_output=False para no mezclar una matriz
# sparse con las columnas de texto/numéricas que pasan sin tocar (remainder='passthrough').
#
# 'barrio' (41 niveles, varios con menos de 30 casas): Target Encoding sobre 'clp' (target
# continuo). TargetEncoder de sklearn hace cross-fitting internamente dentro de fit_transform
# (K-fold sobre TRAIN, calculando el encoding de cada fila sin ver su propio target) y aplica
# suavizado hacia la media global automáticamente -- es exactamente el suavizado bayesiano y la
# CV anidada que describe el PASO 4, sin tener que implementarlos a mano.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, StandardScaler

columnas_onehot = ['comuna', 'Tipo de casa', 'Orientación']
columnas_target_encoding = ['barrio']
TIPOS_DE_CASA_RAROS = {'Tríplex', 'Cabaña'}


@dataclass
class CodificacionParams:
    """`encoder_categoricas` ya viene fitted con train -- IMPORTANTE: `TargetEncoder.fit_transform`
    hace cross-fitting interno (K-fold dentro de train), así que llamar después `.transform(X_train)`
    con el encoder ya fitted daría un resultado DISTINTO (más optimista, sin cross-fitting) para las
    mismas filas. Por eso `X_train_codificado` se guarda acá, tal como salió del propio
    fit_transform -- aplicar_codificacion() nunca lo recalcula, solo se usa para filas que el
    encoder nunca vio en su fit (val/test)."""
    encoder_categoricas: ColumnTransformer
    columnas_numericas_codificadas: list
    X_train_codificado: pd.DataFrame


def _agrupar_tipos_de_casa_raros(X: pd.DataFrame) -> pd.DataFrame:
    """Tríplex (40 casas) y Cabaña (1 casa) son demasiado raras para su propia columna one-hot --
    la de Cabaña sería un identificador de fila disfrazado de feature (ver PASO 4). Renombrada fija
    de categorías (no aprende nada del target ni de la distribución de train), así que se aplica
    igual en cualquier X sin riesgo de fuga."""
    X = X.copy()
    X['Tipo de casa'] = X['Tipo de casa'].replace(TIPOS_DE_CASA_RAROS, 'Otro')
    return X


def _restaurar_dtypes_codificados(X_codificado: pd.DataFrame, columnas_numericas: list) -> pd.DataFrame:
    """El array que devuelve el ColumnTransformer mezcla números con el texto crudo de
    'remainder__descripcion' en un solo bloque, así que numpy lo tipa todo como dtype=object
    (incluido 'target__barrio', que es un float pero queda encapsulado en un objeto). Se restaura
    el dtype numérico columna por columna, dejando 'descripcion' como texto -- si no, cualquier
    describe()/cálculo posterior sobre las columnas numéricas se rompe silenciosamente."""
    X_codificado[columnas_numericas] = X_codificado[columnas_numericas].apply(pd.to_numeric)
    return X_codificado


def ajustar_codificacion(X_train: pd.DataFrame, y_train: pd.Series, verbose: bool = True) -> CodificacionParams:
    X_train = _agrupar_tipos_de_casa_raros(X_train)
    encoder_categoricas = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(drop='first', handle_unknown='infrequent_if_exist', sparse_output=False),
             columnas_onehot),
            ('target', TargetEncoder(target_type='continuous', random_state=42), columnas_target_encoding),
        ],
        remainder='passthrough',
    )
    X_train_codificado = pd.DataFrame(
        encoder_categoricas.fit_transform(X_train, y_train),
        columns=encoder_categoricas.get_feature_names_out(),
        index=X_train.index,
    )
    columnas_numericas_codificadas = [c for c in X_train_codificado.columns if c != 'remainder__descripcion']
    X_train_codificado = _restaurar_dtypes_codificados(X_train_codificado, columnas_numericas_codificadas)
    if verbose:
        print(f'Columnas tras codificar categóricas: {X_train_codificado.shape[1]} (antes: {X_train.shape[1]})')
    return CodificacionParams(encoder_categoricas, columnas_numericas_codificadas, X_train_codificado)


def aplicar_codificacion(X: pd.DataFrame, params: CodificacionParams) -> pd.DataFrame:
    X = _agrupar_tipos_de_casa_raros(X)
    X_codificado = pd.DataFrame(
        params.encoder_categoricas.transform(X),
        columns=params.encoder_categoricas.get_feature_names_out(),
        index=X.index,
    )
    return _restaurar_dtypes_codificados(X_codificado, params.columnas_numericas_codificadas)


params_codificacion = ajustar_codificacion(X_train, y_train)
X_train_codificado = params_codificacion.X_train_codificado
X_test_codificado = aplicar_codificacion(X_test, params_codificacion)

# =======================================================================================
# PASO 4c -- ¿VALE LA PENA VECTORIZAR 'descripcion'? VALIDACIÓN ANTES DE IMPLEMENTAR
# =======================================================================================
# Antes de invertir en limpiar y vectorizar texto (con su propio riesgo de fuga, ver más abajo),
# se mide cuánto aporta: se entrena el mismo modelo con y sin el bloque de texto y se compara
# MdAPE en clp -- en la escala real, no en log, para que el número sea interpretable como
# porcentaje de error sobre el precio. Se usa HistGradientBoostingRegressor (ya viene con
# sklearn, no agrega dependencias) porque es invariante a escala: no hace falta esperar a que el
# escalado numérico (todavía sin implementar, ver PASO 4) esté listo para esta comparación.

from sklearn.ensemble import HistGradientBoostingRegressor


def mdape(y_true, y_pred):
    """Mediana del error porcentual absoluto -- el criterio que PASO 4 ya definía para evaluar
    en la escala real (CLP), no en log."""
    return np.median(np.abs((np.asarray(y_true) - y_pred) / np.asarray(y_true))) * 100


columnas_sin_texto = [c for c in X_train_codificado.columns if c != 'remainder__descripcion']

modelo_sin_texto = HistGradientBoostingRegressor(random_state=42)
modelo_sin_texto.fit(X_train_codificado[columnas_sin_texto], np.log(y_train))
prediccion_sin_texto = np.exp(modelo_sin_texto.predict(X_test_codificado[columnas_sin_texto]))
mdape_sin_texto = mdape(y_test, prediccion_sin_texto)
print(f'MdAPE sin texto: {mdape_sin_texto:.2f}%')

# Limpieza de 'descripcion' antes de vectorizar: el texto libre suele mencionar precio o
# superficie (ej. "terreno de 306 m²", "$457.000", "18.000 UF") -- eso es fuga directa del
# target, ya que 'clp'/'Superficie total' son justamente lo que se quiere predecir/ya está en el
# feature set. Se valida contra una muestra real de deptos.json: 200 de 400 descripciones (50%)
# traen al menos una mención así, y el patrón scrubbea números seguidos de m²/m2/metros/UF/CLP/$
# sin tocar el resto del texto (algún falso positivo inofensivo, ej. "jardín de 10 metros de
# fondo", se pierde -- preferible a dejar pasar una fuga real). Además se bajan a minúscula y se
# sacan tildes (unidecode, ya instalado) porque nltk trae sus stopwords en español sin tildes.

import unidecode
from nltk.corpus import stopwords

PATRON_PRECIO_SUPERFICIE = re.compile(
    r'\$\s?\d[\d.,]*'
    r'|\d[\d.,]*\s?(?:uf|clp|pesos?|millones?|m2|m²|mts2|mts²|metros?)\b',
    re.IGNORECASE,
)


def limpiar_descripcion(texto: str) -> str:
    texto = PATRON_PRECIO_SUPERFICIE.sub(' ', str(texto))
    return unidecode.unidecode(texto).lower()


descripcion_train_limpia = X_train['descripcion'].apply(limpiar_descripcion)
descripcion_test_limpia = X_test['descripcion'].apply(limpiar_descripcion)
stopwords_es = [unidecode.unidecode(palabra) for palabra in stopwords.words('spanish')]

# TF-IDF + SVD, no embeddings: es la opción más barata (sin dependencias nuevas) y la más
# proporcional a los ~4.200 documentos de train que hay para ajustarla. min_df/max_df descartan
# typos/direcciones únicas y el boilerplate de la inmobiliaria que se repite en miles de avisos.
# Se ajusta (fit) SOLO sobre train, igual que el resto de los encoders del PASO 4b.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

N_COMPONENTES_SVD = 50

tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.7, stop_words=stopwords_es)
tfidf_train = tfidf.fit_transform(descripcion_train_limpia)
tfidf_test = tfidf.transform(descripcion_test_limpia)

svd = TruncatedSVD(n_components=N_COMPONENTES_SVD, random_state=42)
texto_train = svd.fit_transform(tfidf_train)
texto_test = svd.transform(tfidf_test)

columnas_texto = [f'texto_svd_{i}' for i in range(N_COMPONENTES_SVD)]
texto_train_df = pd.DataFrame(texto_train, columns=columnas_texto, index=X_train.index)
texto_test_df = pd.DataFrame(texto_test, columns=columnas_texto, index=X_test.index)

X_train_con_texto = pd.concat([X_train_codificado[columnas_sin_texto], texto_train_df], axis=1)
X_test_con_texto = pd.concat([X_test_codificado[columnas_sin_texto], texto_test_df], axis=1)

modelo_con_texto = HistGradientBoostingRegressor(random_state=42)
modelo_con_texto.fit(X_train_con_texto, np.log(y_train))
prediccion_con_texto = np.exp(modelo_con_texto.predict(X_test_con_texto))
mdape_con_texto = mdape(y_test, prediccion_con_texto)
print(f'MdAPE con texto: {mdape_con_texto:.2f}%')
print(f'Diferencia: {mdape_sin_texto - mdape_con_texto:.2f} puntos porcentuales '
      f'({"mejora" if mdape_con_texto < mdape_sin_texto else "empeora"} al agregar texto)')

# CONCLUSIÓN (medida, no supuesta): MdAPE sin texto 8,39% vs. con texto 8,40% -- el bloque de
# TF-IDF+SVD empeora levemente en vez de mejorar. No se adopta: la señal que 'descripcion'
# podría aportar (estado, remodelaciones, vista) ya queda cubierta por las features estructurales
# para este dataset/modelo, o el ruido de 50 componentes SVD sobre ~4.200 documentos le pesa más
# de lo que aporta. 'descripcion' queda fuera del set de entrenamiento final: los pasos
# siguientes deben construirse sobre X_train_codificado[columnas_sin_texto] /
# X_test_codificado[columnas_sin_texto], no sobre X_train_con_texto/X_test_con_texto. Si más
# adelante se prueba con más datos, otro modelo, u otro número de componentes y cambia el
# resultado, revisar este bloque en vez de descartarlo -- por ahora la medición dice que no vale
# la complejidad.

# =======================================================================================
# PASO 4d -- ESCALADO DE VARIABLES NUMÉRICAS
# =======================================================================================
# Se parte de columnas_sin_texto (PASO 4c: 'descripcion' queda afuera, no aportó). El criterio
# para decidir qué escalar no es "continua vs. discreta", sino la forma de la distribución:
#
#   - Superficies y 'Gastos comunes' (skew ~70/~59/~7,4): log1p primero, StandardScaler después.
#     Estandarizar sin corregir la forma no arregla nada -- solo centra la masa de datos
#     aplastada contra el extremo, con el outlier igual de dominante a 40 desvíos.
#   - Conteos discretos ('Dormitorios', 'Baños', 'Estacionamientos', 'Bodegas', 'Cantidad de
#     pisos'): ya son de rango chico y aproximadamente simétricos (skew < 1,2 en dormitorios y
#     baños), así que se estandarizan directo, sin log.
#   - 'Antigüedad': se acota primero a un rango plausible (0-120 años) y luego se estandariza
#     igual que los conteos. Los 140-371 años que quedan tras la limpieza (ver PASO 2 -- el caso
#     evidente de 939 ya se anuló ahí) siguen siendo demasiado altos para una casa real, así que
#     el recorte va acá en vez de descartar esas filas.
#   - Binarias: no se tocan. Ya están en [0,1], y estandarizarlas rompería la lectura directa del
#     coeficiente ("tener piscina cambia el precio en X").
#   - 'latitud'/'longitud': tampoco se escalan -- no se usan crudas, se convertirán en features de
#     distancia (PASO 4, punto 4, todavía sin implementar).
#   - Categóricas ya codificadas (PASO 4b: one-hot y 'target__barrio') quedan tal cual -- el
#     escalado que se pidió acá es solo sobre las numéricas, no sobre lo que ya se codificó.
#
# Para el modelo de árboles el escalado es indiferente (invariante a transformaciones monótonas);
# se aplica igual porque comparte el mismo set de features que el modelo lineal, que sí lo
# necesita. Los StandardScaler se ajustan SOLO con train, igual que el resto del PASO 4.

columnas_log_scale = ['remainder__Superficie útil', 'remainder__Superficie total', 'remainder__Gastos comunes']
columnas_solo_scale = ['remainder__Dormitorios', 'remainder__Baños', 'remainder__Estacionamientos',
                       'remainder__Bodegas', 'remainder__Cantidad de pisos']
columna_antiguedad = 'remainder__Antigüedad'


@dataclass
class EscaladoNumericoParams:
    escalador_log: StandardScaler
    escalador_conteos: StandardScaler
    escalador_antiguedad: StandardScaler
    X_train_final: pd.DataFrame


def ajustar_escalado_numerico(X_train_codificado: pd.DataFrame, verbose: bool = True) -> EscaladoNumericoParams:
    # Recalculado localmente (no el global de PASO 4c) -- las columnas codificadas pueden variar
    # levemente entre folds de PASO 5a, así que no hay que asumir que coinciden con las del split
    # fijo de la corrida única.
    columnas_sin_texto_local = [c for c in X_train_codificado.columns if c != 'remainder__descripcion']
    X_train_final = X_train_codificado[columnas_sin_texto_local].copy()

    escalador_log = StandardScaler()
    X_train_final[columnas_log_scale] = escalador_log.fit_transform(np.log1p(X_train_final[columnas_log_scale]))

    escalador_conteos = StandardScaler()
    X_train_final[columnas_solo_scale] = escalador_conteos.fit_transform(X_train_final[columnas_solo_scale])

    X_train_final[columna_antiguedad] = X_train_final[columna_antiguedad].clip(0, 120)
    escalador_antiguedad = StandardScaler()
    X_train_final[[columna_antiguedad]] = escalador_antiguedad.fit_transform(X_train_final[[columna_antiguedad]])

    if verbose:
        print(f'X_train_final: {X_train_final.shape}.')
        print(X_train_final[columnas_log_scale + columnas_solo_scale + [columna_antiguedad]].describe())
    return EscaladoNumericoParams(escalador_log, escalador_conteos, escalador_antiguedad, X_train_final)


def aplicar_escalado_numerico(X_codificado: pd.DataFrame, params: EscaladoNumericoParams) -> pd.DataFrame:
    columnas_sin_texto_local = [c for c in X_codificado.columns if c != 'remainder__descripcion']
    X_final = X_codificado[columnas_sin_texto_local].copy()
    X_final[columnas_log_scale] = params.escalador_log.transform(np.log1p(X_final[columnas_log_scale]))
    X_final[columnas_solo_scale] = params.escalador_conteos.transform(X_final[columnas_solo_scale])
    X_final[columna_antiguedad] = X_final[columna_antiguedad].clip(0, 120)
    X_final[[columna_antiguedad]] = params.escalador_antiguedad.transform(X_final[[columna_antiguedad]])
    return X_final


params_escalado = ajustar_escalado_numerico(X_train_codificado)
X_train_final = params_escalado.X_train_final
X_test_final = aplicar_escalado_numerico(X_test_codificado, params_escalado)
print(f'X_train_final: {X_train_final.shape}. X_test_final: {X_test_final.shape}.')

# =======================================================================================
# PASO 4e -- FEATURES DE DISTANCIA DESDE LATITUD/LONGITUD
# =======================================================================================
# 'latitud'/'longitud' crudas no se usan como feature -- un modelo lineal no puede aprender una
# relación de valor a partir de dos coordenadas por sí solas (ver PASO 4). En su lugar se calculan
# distancias reales (haversine, ya definido en PASO 2 para imputar Orientación) a los polos de
# valor del sector: Clínica Alemana, Estadio Español y Portal La Dehesa -- lugares que las propias
# publicaciones mencionan como argumento de venta ("a pasos de Portal La Dehesa", ver
# 'descripcion') y que reemplazan al viejo campo 'cercania' ('Cerca'/'Lejos', eliminado en un
# commit anterior) con una medida continua en vez de una lista de barrios escrita a mano. Con eso,
# el modelo ya tiene la señal de "qué tan bien ubicada" está la casa antes de calcular el residuo,
# así que ese residuo no confunde "buena ubicación no capturada" con sobre/subvaloración real.
#
# Coordenadas verificadas por búsqueda web, no de memoria:
#   - Clínica Alemana (Av. Vitacura 5951, Vitacura): -33,3918 / -70,5729
#   - Estadio Español (Nevería 4855, Las Condes): -33,4147 / -70,5771
#   - Portal La Dehesa (Av. La Dehesa, Lo Barnechea): -33,3579 / -70,5152
#
# Se deja afuera "colegios del corredor Manquehue-La Dehesa" (ver PASO 4): no es un punto único,
# es una zona con decenas de colegios, y no correspondía inventarle una coordenada
# "representativa" a algo que no tiene una.
#
# También se agrega distancia al centroide de la propia comuna (aproxima qué tan central o
# periférica es la casa dentro de su zona). El centroide se calcula SOLO con train -- es una
# estadística que "aprende" de los datos, igual que el resto de lo que ya se ajusta solo con train
# en PASO 4 -- y se aplica a test con esos mismos valores, sin recalcularlo ahí.

PUNTOS_DE_INTERES = {
    'clinica_alemana': (-33.3918, -70.5729),
    'estadio_espanol': (-33.4147, -70.5771),
    'portal_la_dehesa': (-33.3579, -70.5152),
}
columnas_distancia = [f'distancia_{nombre}_m' for nombre in PUNTOS_DE_INTERES] + ['distancia_centroide_comuna_m']


@dataclass
class DistanciaParams:
    centroide_comuna: pd.DataFrame
    escalador_distancias: StandardScaler
    X_train_final: pd.DataFrame


def _agregar_distancias(df: pd.DataFrame, comuna_por_fila: pd.Series, centroide_comuna: pd.DataFrame) -> None:
    for nombre, (lat_punto, lon_punto) in PUNTOS_DE_INTERES.items():
        df[f'distancia_{nombre}_m'] = haversine_m(df['remainder__latitud'], df['remainder__longitud'],
                                                    lat_punto, lon_punto)
    lat_centroide = comuna_por_fila.map(centroide_comuna['remainder__latitud'])
    lon_centroide = comuna_por_fila.map(centroide_comuna['remainder__longitud'])
    df['distancia_centroide_comuna_m'] = haversine_m(df['remainder__latitud'], df['remainder__longitud'],
                                                       lat_centroide, lon_centroide)


def ajustar_features_distancia(X_train_final: pd.DataFrame, comuna_train: pd.Series,
                                verbose: bool = True) -> DistanciaParams:
    X_train_final = X_train_final.copy()
    centroide_comuna = X_train_final.groupby(comuna_train)[['remainder__latitud', 'remainder__longitud']].mean()
    _agregar_distancias(X_train_final, comuna_train, centroide_comuna)
    if verbose:
        print(X_train_final[columnas_distancia].describe())
        print('skew:', X_train_final[columnas_distancia].skew().to_dict())

    # Con las distancias calculadas, 'latitud'/'longitud' crudas ya cumplieron su propósito y se
    # eliminan del set de features -- dejarlas junto con las distancias sería redundante y
    # volvería a exponer al modelo lineal a la coordenada cruda que se quería evitar.
    X_train_final = X_train_final.drop(columns=['remainder__latitud', 'remainder__longitud'])

    X_train_final[columnas_distancia] = np.log1p(X_train_final[columnas_distancia])
    escalador_distancias = StandardScaler()
    X_train_final[columnas_distancia] = escalador_distancias.fit_transform(X_train_final[columnas_distancia])
    return DistanciaParams(centroide_comuna, escalador_distancias, X_train_final)


def aplicar_features_distancia(X_final: pd.DataFrame, comuna: pd.Series, params: DistanciaParams) -> pd.DataFrame:
    X_final = X_final.copy()
    _agregar_distancias(X_final, comuna, params.centroide_comuna)
    X_final = X_final.drop(columns=['remainder__latitud', 'remainder__longitud'])
    X_final[columnas_distancia] = np.log1p(X_final[columnas_distancia])
    X_final[columnas_distancia] = params.escalador_distancias.transform(X_final[columnas_distancia])
    return X_final


params_distancia = ajustar_features_distancia(X_train_final, X_train['comuna'])
X_train_final = params_distancia.X_train_final
X_test_final = aplicar_features_distancia(X_test_final, X_test['comuna'], params_distancia)
print(f'X_train_final: {X_train_final.shape}. X_test_final: {X_test_final.shape}.')

# =======================================================================================
# PASO 4f -- RATIOS ESTRUCTURALES
# =======================================================================================
# Se calculan sobre los valores crudos de X_train/X_test -- antes del log1p y el StandardScaler
# que PASO 4d ya aplicó in-place a 'Superficie útil'/'Superficie total'/'Dormitorios'/'Baños'
# dentro de X_train_final. Dividir columnas ya estandarizadas no tiene lectura como ratio, solo
# reproduce el cociente de dos z-scores.
#
#   - ratio_construido = Superficie útil / Superficie total: distingue la casa grande en lote
#     chico de la chica en lote grande, que a igual superficie total valen muy distinto (ver
#     PASO 4). Va de 0 a 1 por construcción -- la limpieza ya invirtió los pares donde útil >
#     total (ver commit "Swap Superficie útil/total..." en PASO 2), así que no hace falta acotarlo.
#   - m2_por_dormitorio = Superficie útil / Dormitorios: proxy de estándar de la casa,
#     independiente de su tamaño absoluto. Se usa 'útil' y no 'total': el dormitorio se vive en
#     el área construida, no en el terreno.
#   - baños_por_dormitorio = Baños / Dormitorios: mismo criterio, otro proxy de estándar.
#
# Sin guard contra división por cero: 'Dormitorios' y 'Superficie total' no tienen ceros en todo
# el dataset limpio, medido, no supuesto (ver mín. de ambas en el describe() de PASO 3).

columnas_ratios = ['ratio_construido', 'm2_por_dormitorio', 'banos_por_dormitorio']
columnas_ratio_simetricos = ['ratio_construido', 'banos_por_dormitorio']


@dataclass
class RatiosParams:
    escalador_ratios: StandardScaler
    escalador_m2_dormitorio: StandardScaler
    X_train_final: pd.DataFrame


def _agregar_ratios(X_final: pd.DataFrame, X_crudo: pd.DataFrame) -> None:
    X_final['ratio_construido'] = X_crudo['Superficie útil'] / X_crudo['Superficie total']
    X_final['m2_por_dormitorio'] = X_crudo['Superficie útil'] / X_crudo['Dormitorios']
    X_final['banos_por_dormitorio'] = X_crudo['Baños'] / X_crudo['Dormitorios']


def ajustar_ratios(X_train_final: pd.DataFrame, X_train_crudo: pd.DataFrame, verbose: bool = True) -> RatiosParams:
    X_train_final = X_train_final.copy()
    _agregar_ratios(X_train_final, X_train_crudo)
    if verbose:
        print(X_train_final[columnas_ratios].describe())
        print('skew:', X_train_final[columnas_ratios].skew().to_dict())

    escalador_ratios = StandardScaler()
    X_train_final[columnas_ratio_simetricos] = escalador_ratios.fit_transform(X_train_final[columnas_ratio_simetricos])

    escalador_m2_dormitorio = StandardScaler()
    X_train_final[['m2_por_dormitorio']] = escalador_m2_dormitorio.fit_transform(
        np.log1p(X_train_final[['m2_por_dormitorio']]))

    return RatiosParams(escalador_ratios, escalador_m2_dormitorio, X_train_final)


def aplicar_ratios(X_final: pd.DataFrame, X_crudo: pd.DataFrame, params: RatiosParams) -> pd.DataFrame:
    X_final = X_final.copy()
    _agregar_ratios(X_final, X_crudo)
    X_final[columnas_ratio_simetricos] = params.escalador_ratios.transform(X_final[columnas_ratio_simetricos])
    X_final[['m2_por_dormitorio']] = params.escalador_m2_dormitorio.transform(np.log1p(X_final[['m2_por_dormitorio']]))
    return X_final


params_ratios = ajustar_ratios(X_train_final, X_train)
X_train_final = params_ratios.X_train_final
X_test_final = aplicar_ratios(X_test_final, X_test, params_ratios)
print(f'X_train_final: {X_train_final.shape}. X_test_final: {X_test_final.shape}.')

# =======================================================================================
# PASO 4g -- COLINEALIDAD
# =======================================================================================
# Con los ratios ya en el set, 'Superficie útil' queda redundante por construcción: útil = total
# × ratio_construido. VIF por columna (regresión OLS de cada feature contra el resto -- misma
# fórmula que statsmodels, sin agregar esa dependencia al proyecto) sobre las columnas ya
# escaladas de X_train_final cuantifica el problema en vez de suponerlo.


def calcular_vif(frame: pd.DataFrame) -> dict:
    valores = frame.to_numpy()
    unos = np.ones((valores.shape[0], 1))
    vif = {}
    for i, columna in enumerate(frame.columns):
        y_columna = valores[:, i]
        resto = np.delete(valores, i, axis=1)
        diseño = np.hstack([unos, resto])
        coeficientes, *_ = np.linalg.lstsq(diseño, y_columna, rcond=None)
        prediccion = diseño @ coeficientes
        r2 = 1 - np.sum((y_columna - prediccion) ** 2) / np.sum((y_columna - y_columna.mean()) ** 2)
        vif[columna] = 1 / (1 - r2) if r2 < 0.999999 else np.inf
    return vif


columnas_vif = ['remainder__Superficie útil', 'remainder__Superficie total',
                'remainder__Dormitorios', 'remainder__Baños'] + columnas_ratios
print('VIF con los 3 ratios agregados:',
      {c: round(v, 1) for c, v in calcular_vif(X_train_final[columnas_vif]).items()})

# VIF medido: útil=113,3 / m2_por_dormitorio=84,5 / dormitorios=21,6 / baños=17,1 / total=14,6 /
# ratio_construido=5,8. 'Superficie útil' y 'm2_por_dormitorio' son las más afectadas -- cada una
# comparte a 'útil' como numerador con otra feature que ya está en el set (total×ratio_construido
# y Dormitorios×m2_por_dormitorio respectivamente), así que su VIF se dispara.
#
# Iterar VIF>10 al pie de la letra (como plantea PASO 4) no se detiene en 'útil': también
# descarta 'Superficie total' y 'Baños' (recalculando el VIF tras cada drop, 'Dormitorios' sí
# termina sobreviviendo), y al modelo lineal no le queda ninguna medida absoluta de superficie,
# solo 'Dormitorios' + los 3 ratios. Eso contradice la premisa hedónica del propio archivo (el
# precio responde a la superficie, ver cabecera) -- perder la señal más fuerte del dataset por
# seguir un umbral de regla de dedo es peor que tolerar VIF residual.
#
# Se aplica entonces la alternativa que PASO 4 ya dejaba planteada: dropear SOLO 'Superficie
# útil' (redundancia exacta, cero información perdida -- se reconstruye con total×ratio) y
# conservar 'Superficie total'/'Dormitorios'/'Baños' pese a que su VIF post-drop siga por encima
# de 10 (medido: total=14,3 / baños=14,2 / dormitorios=10,0 / banos_por_dormitorio=9,8 /
# m2_por_dormitorio=6,7): a diferencia de 'útil', esa correlación no es una identidad
# matemática, es señal real (cantidad absoluta de baños/dormitorios) que ninguna otra columna
# reemplaza. Para el modelo de árboles esto es indiferente de por sí (ver PASO 4); para el
# lineal, infla la varianza de esos coeficientes puntuales sin invalidar la predicción global.
#
# CONCLUSIÓN (evaluada, no solo medida): se descarta PCA como alternativa para bajar este VIF
# residual. Motivos:
#   - Mata la interpretabilidad que es el objetivo del modelo lineal hedónico (ver cabecera del
#     archivo): los componentes son combinaciones lineales de columnas heterogéneas (one-hot de
#     comuna, target encoding de barrio, binarias de amenities, ratios, distancias), así que
#     ningún coeficiente queda legible como elasticidad o efecto de un amenity.
#   - Mezcla tipos que PCA no debería recibir juntos: 82 columnas incluyen dummies 0/1 y target
#     encoding además de las continuas escaladas. La varianza de una dummy no es señal continua
#     correlacionada, es una categoría -- PCA sobre eso no produce componentes con sentido.
#   - Ya hay precedente en este archivo de cuándo SÍ conviene reducir dimensionalidad: PASO 4c
#     aplicó TruncatedSVD (mismo espíritu que PCA), pero solo al bloque disperso de TF-IDF,
#     separado de las features estructuradas -- y ahí la medición dijo que ni siquiera convenía
#     (MdAPE empeoró). No hay caso análogo para aplicarlo sobre las features estructuradas.
#   - El VIF residual no es grave: infla varianza de coeficientes puntuales, no invalida la
#     predicción, y es indiferente para el modelo de árboles. No justifica el costo.

COLUMNAS_DROP_ESTRUCTURAL = ['remainder__Superficie útil']  # regla fija, no se recalcula por fold (ver PASO 5a)

X_train_final = X_train_final.drop(columns=COLUMNAS_DROP_ESTRUCTURAL)
X_test_final = X_test_final.drop(columns=COLUMNAS_DROP_ESTRUCTURAL)

columnas_vif_post = ['remainder__Superficie total', 'remainder__Dormitorios',
                     'remainder__Baños'] + columnas_ratios
print('VIF tras dropear Superficie útil:',
      {c: round(v, 1) for c, v in calcular_vif(X_train_final[columnas_vif_post]).items()})
print(f'X_train_final: {X_train_final.shape}. X_test_final: {X_test_final.shape}.')

# =======================================================================================
# PASO 4h -- BASELINE: MEDIANA DE PRECIO/M² POR BARRIO
# =======================================================================================
# Última pieza pendiente de PASO 4: la feature de control obligatoria. Antes de invertir más en
# ingeniería de features, hay que fijar el piso que cualquier modelo tiene que superar para
# justificar su complejidad -- si toda la codificación, escalado y ratios de arriba no le ganan a
# una sola columna (mediana de precio/m² del barrio), el modelo no vale la pena.
#
# 'precio unitario' = clp / Superficie total (misma definición que ya usa el resto del archivo,
# ver PASO 4a). La mediana se calcula SOLO con train, igual que el resto de lo que "aprende" de
# los datos en PASO 4, y se aplica a test multiplicando por su propia 'Superficie total' -- nunca
# se usa una mediana calculada sobre test.
#
# Fallback en dos niveles para barrios de test ausentes en train (el split es estratificado por
# 'comuna', no por 'barrio' -- ver PASO 4b -- así que train no queda garantizado con los 41
# barrios): si el barrio no está en la mediana de train, se usa la mediana de su 'comuna'; si
# ninguna fila de esa comuna cayó en train (no debería pasar con el split 80/20 actual, pero el
# fallback cubre el caso), la mediana global de train.

precio_unitario_train = y_train / X_train['Superficie total']

mediana_por_barrio = precio_unitario_train.groupby(X_train['barrio']).median()
mediana_por_comuna = precio_unitario_train.groupby(X_train['comuna']).median()
mediana_global = precio_unitario_train.median()

precio_m2_baseline_test = (
    X_test['barrio'].map(mediana_por_barrio)
    .fillna(X_test['comuna'].map(mediana_por_comuna))
    .fillna(mediana_global)
)
prediccion_baseline = precio_m2_baseline_test * X_test['Superficie total']

mdape_baseline = mdape(y_test, prediccion_baseline)
print(f'MdAPE baseline (mediana precio/m² por barrio): {mdape_baseline:.2f}%')
print(f'MdAPE modelo HistGB sin texto (PASO 4c): {mdape_sin_texto:.2f}%')
print(f'Diferencia: {mdape_baseline - mdape_sin_texto:.2f} puntos porcentuales '
      f'({"el modelo le gana al baseline" if mdape_sin_texto < mdape_baseline else "el baseline le gana al modelo"})')

# =======================================================================================
# PASO 4i -- 'GASTOS COMUNES' > 0 COMO BINARIA
# =======================================================================================
# El nulo de 'Gastos comunes' se rellenó con 0 en la carga (línea 741), junto con los avisos que sí
# declararon 0 CLP de gasto común. 47% de las filas en train son ese nulo relleno, así que esta
# binaria no distingue "está en condominio con administración" de "no lo está" -- distingue si el
# corredor informó el dato o no. Confirmado: importancia por permutación 0,000000 (ver PASO 4k).
# Se deja el nombre honesto sobre lo que mide, en vez de 'tiene_gastos_comunes'.
#
# Se calcula sobre el valor CRUDO de X_train/X_test (antes del log1p+scale que PASO 4d ya aplicó
# in-place a 'remainder__Gastos comunes' en X_train_final), mismo criterio que los ratios de
# PASO 4f. Es binaria: no se escala (ver PASO 4d).

def agregar_gastos_comunes_informado(X_final: pd.DataFrame, X_crudo: pd.DataFrame) -> pd.DataFrame:
    """Regla fija, sin fit: no depende de train ni de fold. Ver nota arriba sobre por qué es
    'informado' y no 'tiene'."""
    X_final = X_final.copy()
    X_final['gastos_comunes_informado'] = (X_crudo['Gastos comunes'] > 0).astype(int)
    return X_final


X_train_final = agregar_gastos_comunes_informado(X_train_final, X_train)
X_test_final = agregar_gastos_comunes_informado(X_test_final, X_test)

print('gastos_comunes_informado en train:',
      X_train_final['gastos_comunes_informado'].value_counts(normalize=True).round(3).to_dict())
print(f'X_train_final: {X_train_final.shape}. X_test_final: {X_test_final.shape}.')

# =======================================================================================
# PASO 4j -- DENSIDAD LOCAL DE OFERTA
# =======================================================================================
# Complementa las distancias de PASO 4e: cuántas otras casas hay a menos de 500 m aproxima qué
# tan consolidado está el sector -- dos casas a la misma distancia de Portal La Dehesa pueden
# estar en un sector denso o en un loteo aislado, y esa diferencia le importa al precio. Se
# cuenta contra TRAIN únicamente (el árbol se construye solo con esas coordenadas, igual que el
# centroide de comuna en PASO 4e), para no filtrar información de test.
#
# cKDTree opera en distancia euclidiana, no haversine, así que lat/lon se proyectan primero a
# metros locales (proyección equirrectangular: a la escala de Santiago -- unas pocas decenas de
# km, latitud casi constante -- el error frente a haversine es despreciable, y evita recalcular
# haversine por pares para un simple conteo por radio). El punto de referencia de la proyección
# (latitud/longitud media) se toma de TRAIN, mismo criterio de "se ajusta solo con train" del
# resto de PASO 4.

RADIO_DENSIDAD_M = 500
RADIO_TIERRA_M = 6_371_000


def proyectar_metros(lat: pd.Series, lon: pd.Series, lat0: float, lon0: float) -> np.ndarray:
    x = RADIO_TIERRA_M * np.radians(lon - lon0) * np.cos(np.radians(lat0))
    y = RADIO_TIERRA_M * np.radians(lat - lat0)
    return np.column_stack([x, y])


@dataclass
class DensidadParams:
    lat0: float
    lon0: float
    arbol_densidad: object
    escalador_densidad: StandardScaler
    X_train_final: pd.DataFrame


def ajustar_densidad(X_train_final: pd.DataFrame, X_train_crudo: pd.DataFrame, verbose: bool = True) -> DensidadParams:
    X_train_final = X_train_final.copy()
    lat0, lon0 = X_train_crudo['latitud'].mean(), X_train_crudo['longitud'].mean()
    coords_train_m = proyectar_metros(X_train_crudo['latitud'], X_train_crudo['longitud'], lat0, lon0)
    arbol_densidad = cKDTree(coords_train_m)

    # Cada casa de train está en el propio árbol -- se resta 1 para no contarse a sí misma.
    densidad_train = arbol_densidad.query_ball_point(coords_train_m, r=RADIO_DENSIDAD_M, return_length=True) - 1
    X_train_final['densidad_oferta_500m'] = densidad_train
    if verbose:
        print(X_train_final['densidad_oferta_500m'].describe())
        print('skew:', X_train_final['densidad_oferta_500m'].skew())

    escalador_densidad = StandardScaler()
    X_train_final[['densidad_oferta_500m']] = escalador_densidad.fit_transform(X_train_final[['densidad_oferta_500m']])
    return DensidadParams(lat0, lon0, arbol_densidad, escalador_densidad, X_train_final)


def aplicar_densidad(X_final: pd.DataFrame, X_crudo: pd.DataFrame, params: DensidadParams) -> pd.DataFrame:
    """A diferencia de train, las filas de `X_crudo` NO están en `arbol_densidad` (que solo
    contiene coordenadas de train) -- no hace falta restar 1."""
    X_final = X_final.copy()
    coords_m = proyectar_metros(X_crudo['latitud'], X_crudo['longitud'], params.lat0, params.lon0)
    X_final['densidad_oferta_500m'] = params.arbol_densidad.query_ball_point(
        coords_m, r=RADIO_DENSIDAD_M, return_length=True)
    X_final[['densidad_oferta_500m']] = params.escalador_densidad.transform(X_final[['densidad_oferta_500m']])
    return X_final


params_densidad = ajustar_densidad(X_train_final, X_train)
X_train_final = params_densidad.X_train_final
X_test_final = aplicar_densidad(X_test_final, X_test, params_densidad)
print(f'X_train_final: {X_train_final.shape}. X_test_final: {X_test_final.shape}.')

# =======================================================================================
# PASO 4k -- SELECCIÓN FINAL POR IMPORTANCIA DE PERMUTACIÓN
# =======================================================================================
# Último paso de PASO 4: con las columnas de X_train_final ya armadas (categóricas codificadas,
# numéricas escaladas, ratios, distancias, densidad, binaria de gastos comunes), se mide qué
# aporta cada una y se descarta lo que no distingue de ruido.
#
# Importancia por PERMUTACIÓN, no por impureza de sklearn: la de impureza sobrevalora columnas de
# alta cardinalidad como 'target__barrio' (más posibilidades de split = más impureza explicada
# por azar), y acá 'target__barrio' es justamente una de las features más fuertes del dataset
# (ver PASO 4), así que ese sesgo sí distorsionaría el ranking.
#
# Se mide sobre un fold de VALIDACIÓN separado de train, nunca sobre test: decidir qué features
# entran al modelo mirando el desempeño en test sería la misma fuga que ajustar un hiperparámetro
# ahí, y test dejaría de ser un holdout limpio para la evaluación final. Split 80/20 estratificado
# por 'comuna', mismo criterio que el split original de PASO 4b.

X_train_sel, X_val_sel, y_train_sel, y_val_sel = train_test_split(
    X_train_final, y_train, test_size=0.2, stratify=X_train['comuna'], random_state=42
)

modelo_seleccion = HistGradientBoostingRegressor(random_state=42)
modelo_seleccion.fit(X_train_sel, np.log(y_train_sel))

mdape_val_completo = mdape(y_val_sel, np.exp(modelo_seleccion.predict(X_val_sel)))
print(f'MdAPE en validación ({X_train_sel.shape[1]} columnas): {mdape_val_completo:.2f}%')

from sklearn.inspection import permutation_importance

resultado_permutacion = permutation_importance(
    modelo_seleccion, X_val_sel, np.log(y_val_sel), n_repeats=10, random_state=42, n_jobs=-1
)

importancias = pd.DataFrame({
    'feature': X_val_sel.columns,
    'importancia_media': resultado_permutacion.importances_mean,
    'importancia_std': resultado_permutacion.importances_std,
}).sort_values('importancia_media')
print(importancias.to_string(index=False))

# No distinguible de cero: el intervalo [media - std, media + std] incluye 0 -- la caída de score
# al permutar la columna no es consistente entre repeticiones, así que no hay evidencia de que el
# modelo la use de verdad (mismo criterio que documentación de sklearn recomienda para interpretar
# permutation_importance).

columnas_sin_senal = importancias.loc[
    importancias['importancia_media'] - importancias['importancia_std'] <= 0, 'feature'
].tolist()
print(f'Columnas sin señal distinguible de cero ({len(columnas_sin_senal)}):', columnas_sin_senal)

columnas_final_reducidas = [c for c in X_train_final.columns if c not in columnas_sin_senal]

modelo_reducido = HistGradientBoostingRegressor(random_state=42)
modelo_reducido.fit(X_train_sel[columnas_final_reducidas], np.log(y_train_sel))
mdape_val_reducido = mdape(y_val_sel, np.exp(modelo_reducido.predict(X_val_sel[columnas_final_reducidas])))
print(f'MdAPE en validación ({len(columnas_final_reducidas)} columnas): {mdape_val_reducido:.2f}%')
print(f'Diferencia: {mdape_val_reducido - mdape_val_completo:.2f} puntos porcentuales')

# Solo se descartan las columnas si el MdAPE en validación no empeora -- si empeora, alguna de
# esas columnas "sin señal" para el modelo de este split sí aportaba, y se prefiere conservar el
# feature set completo antes que perder precisión por ahorrar columnas.

if mdape_val_reducido <= mdape_val_completo:
    X_train_final = X_train_final.drop(columns=columnas_sin_senal)
    X_test_final = X_test_final.drop(columns=columnas_sin_senal)
    print(f'MdAPE no empeora: se descartan {len(columnas_sin_senal)} columnas sin señal.')
else:
    print('MdAPE empeora al descartar: se conservan todas las columnas.')

print(f'X_train_final: {X_train_final.shape}. X_test_final: {X_test_final.shape}.')

# ADVERTENCIA SOBRE ESTE CRITERIO -- la comparación de arriba NO es concluyente.
#
# Se mide una sola partición de validación, y la diferencia entre quedarse con 82 o con 40
# columnas (0,2 puntos porcentuales) es más chica que la variación que produce cambiar la semilla
# del split. Medido sobre tres semillas distintas, el mismo modelo y el mismo feature set dan
# MdAPE de 8,15% / 8,29% / 8,51%: un rango de 0,36 puntos, casi el doble de la "diferencia" que
# acá decide si se descartan 42 columnas o ninguna.
#
# La prueba está en que el criterio ya cambió de resultado una vez sin que se tocara ninguna
# feature: antes de corregir la contaminación train/test de PASO 2 concluía "no empeora, se
# descartan 44 columnas", y con la limpieza ya ajustada solo con train concluye lo contrario.
# La decisión la estaba tomando el ruido de partición, no la señal.
#
# Para que esto sea una decisión y no un sorteo hay que reemplazar la validación simple por
# RepeatedKFold y descartar una columna solo si su pérdida de importancia es consistente entre
# folds -- pendiente. Mientras tanto se conserva el feature set completo, que es la opción
# conservadora: una columna irrelevante le cuesta poco a un modelo de árboles, y descartar una
# que sí aportaba no se recupera.

# =======================================================================================
# PASO 4 -- WRAPPERS DE ORQUESTACIÓN (ajustar_features / aplicar_features)
# =======================================================================================
# Para el driver de CV de PASO 5a: encadenan los 5 pares finos de arriba (4b, 4d, 4e, 4f, 4j) más
# el drop estructural de 4g y la binaria de 4i, en el mismo orden que la corrida única de más
# arriba. NO incluyen PASO 4c (validación de texto, no llega a producción) ni PASO 4k (selección
# por importancia de permutación) -- ver la nota de scope en PASO 5a más abajo sobre por qué la
# selección de columnas se deja fuera del refit por fold.


@dataclass
class FeatureParams:
    codificacion: CodificacionParams
    escalado: EscaladoNumericoParams
    distancia: DistanciaParams
    ratios: RatiosParams
    densidad: DensidadParams
    X_train_final: pd.DataFrame


def ajustar_features(X_train: pd.DataFrame, y_train: pd.Series, verbose: bool = False) -> FeatureParams:
    codificacion = ajustar_codificacion(X_train, y_train, verbose=verbose)
    escalado = ajustar_escalado_numerico(codificacion.X_train_codificado, verbose=verbose)
    distancia = ajustar_features_distancia(escalado.X_train_final, X_train['comuna'], verbose=verbose)
    ratios = ajustar_ratios(distancia.X_train_final, X_train, verbose=verbose)

    X_train_final = ratios.X_train_final.drop(columns=COLUMNAS_DROP_ESTRUCTURAL)
    X_train_final = agregar_gastos_comunes_informado(X_train_final, X_train)

    densidad = ajustar_densidad(X_train_final, X_train, verbose=verbose)
    return FeatureParams(codificacion, escalado, distancia, ratios, densidad, densidad.X_train_final)


def aplicar_features(X: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    X_codificado = aplicar_codificacion(X, params.codificacion)
    X_final = aplicar_escalado_numerico(X_codificado, params.escalado)
    X_final = aplicar_features_distancia(X_final, X['comuna'], params.distancia)
    X_final = aplicar_ratios(X_final, X, params.ratios)
    X_final = X_final.drop(columns=COLUMNAS_DROP_ESTRUCTURAL)
    X_final = agregar_gastos_comunes_informado(X_final, X)
    X_final = aplicar_densidad(X_final, X, params.densidad)
    return X_final


# =======================================================================================
# PASO 5 -- ESTRATEGIA DE MODELAMIENTO (diseño, aún no implementado)
# =======================================================================================
#
# Punto de partida: PASO 4k deja X_train_final / X_test_final ya armados (82 columnas: categóricas
# codificadas, numéricas log+escaladas, ratios, distancias, densidad) y el target en CLP en
# y_train / y_test, que se modela como log(clp) (ver cabecera del archivo). Tres piezas ya existen
# y no hay que reconstruirlas: mdape() (PASO 4c), el baseline de mediana de precio/m² por barrio
# (PASO 4h) y un HistGradientBoostingRegressor con defaults, entrenado en PASO 4k como herramienta
# de selección de features -- no como modelo final.
#
# Lo que falta NO es "entrenar un modelo": eso ya ocurre de facto. Lo que falta es lo que convierte
# una predicción en una decisión de compra: saber cuánto vale esa predicción caso a caso
# (intervalos), cuán estable es la medición (validación cruzada) y cómo se traduce el error en una
# lista accionable (ranking).
#
# El entregable del proyecto no es el modelo, es la lista de casas a visitar. Medido: un modelo con
# 8,39% de MdAPE que no sabe cuándo NO sabe produce, con un umbral fijo de ±10%, una lista con el
# 43% del inventario -- que es lo mismo que no producir nada.
#
# PUNTO DE PARTIDA MEDIDO (última corrida completa; sirve para detectar regresiones)
# ---------------------------------------------------------------------------------
#   Filas 5.284 (train 4.227 / test 1.057)   |   features 82
#   MdAPE HistGB (test)                            8,39 %
#   MdAPE baseline mediana precio/m² por barrio   21,50 %
#   MAE                                          150.912.000 CLP
#   RMSE                                         295.966.827 CLP
#   Residuo p10 / p50 / p90               -15,8 % / -0,5 % / +20,1 %      (p99: +80,8 %)
#   Mismo modelo, tres semillas de split   8,15 % / 8,29 % / 8,51 %  -> ruido de 0,36 puntos
#
# ORDEN DE OPERACIONES (no negociable)
# ------------------------------------
# 1. Todo lo que decida algo -- features, hiperparámetros, familia de modelo -- se decide con CV
#    sobre TRAIN. Un hiperparámetro elegido mirando test es la misma fuga que una feature elegida
#    mirando test.
# 2. test se evalúa UNA sola vez, al final, y no se vuelve atrás. Si se mira test y después se
#    cambia algo, test dejó de ser holdout y el número deja de estimar el error fuera de muestra.
# 3. Primero el protocolo de medición (5a), después todo lo demás: sin conocer el ruido de medición
#    no se puede saber si una mejora es una mejora. Ya pasó una vez -- ver la advertencia al final
#    de PASO 4k, donde el criterio de selección de features se dio vuelta solo por el ruido.
# 4. Los intervalos (5e) van ANTES del ranking (5f). Un ranking sin intervalo es el umbral fijo que
#    ya se descartó por medición.
# 5. La calibración de los intervalos necesita datos que el modelo no vio: sale de un fold de
#    calibración recortado de TRAIN, nunca de test.
#
# SUB-PASOS
# ---------
#   5a  Protocolo de evaluación: validación cruzada repetida
#   5b  Modelo lineal log-log (el modelo interpretable que la cabecera promete)
#   5c  Tuning de hiperparámetros del modelo de árboles
#   5d  Evaluación final en test: MdAPE + MAE + RMSE + percentiles, desagregado
#   5e  Intervalos de predicción                        <- bloquea el objetivo de negocio
#   5f  Residuo, regla de flag y ranking
#   5g  Filtros del comprador sobre el ranking
#   5h  Export de casas_candidatas.xlsx
#   5i  Mapa
#   5j  Límites: qué NO puede responder este modelo

# =======================================================================================
# PASO 5a -- PROTOCOLO DE EVALUACIÓN: VALIDACIÓN CRUZADA REPETIDA
# =======================================================================================
# Va primero porque ningún resultado posterior es interpretable sin saber cuánto ruido tiene la
# medición. Medido: el mismo modelo con el mismo feature set, cambiando solo la semilla del split,
# da 8,15% / 8,29% / 8,51%. Cualquier mejora menor a ~0,4 puntos reportada sobre una sola partición
# es indistinguible de haber tenido suerte con la semilla.
#
# Ya no queda la "TRAMPA ARQUITECTÓNICA" que advertía la versión anterior de este bloque: PASO
# 2b-4j se refactorizaron a pares ajustar()/aplicar() (ver más arriba), así que cada fold de abajo
# reajusta TODO -- KDTree de comuna/barrio/orientación, bad_ranges, allocate_values, TargetEncoder
# de barrio, los escaladores, centroide de comuna, densidad local -- con SUS PROPIOS índices de
# train, no con los de `indice_train` del split fijo.
#
# Alcance deliberadamente fuera de esta CV (ver PASO 5a en el bloque de diseño original, ahora
# resuelto en la práctica):
#   - PASO 4k (selección por importancia de permutación) NO se repite por fold: cada fold usa el
#     feature set completo (82 columnas menos el drop de 4g). Recalcularla por fold sería nested-
#     selection carísimo, y el propósito de esta CV es medir el modelo, no repetir esa decisión --
#     que además la propia advertencia de 4k ya mostró que es ruido, no señal.
#   - PASO 4g (VIF) tampoco se recalcula: el drop de 'Superficie útil' es una regla estructural
#     fija (redundancia exacta con total×ratio_construido), no depende de qué filas cayeron en
#     train.
#   - PASO 4h (baseline mediana precio/m² por barrio) sigue siendo diagnóstico de un solo split.
#
# Universo de la CV: `deptos_df_limpio.loc[indice_train]` -- las 4.227 filas de train fijo (nunca
# `indice_test`, que se preserva intacto como holdout final para PASO 5d). La CV reparte de nuevo
# esas filas en folds internos, sobre la versión SIN la imputación ya fijada por el split 80/20,
# para poder reajustarla por fold.

from sklearn.model_selection import StratifiedKFold


def ejecutar_cv_repetida(universo: pd.DataFrame, n_splits: int = 5, n_repeats: int = 3,
                          random_state: int = 42) -> pd.Series:
    """CV repetida (n_splits x n_repeats folds) estratificada por 'comuna' -- mismo criterio del
    split fijo de PASO 2b/4b. Cada fold reajusta TODO el pipeline (ajustar_imputacion +
    ajustar_features) con sus propios índices de train, y evalúa MdAPE sobre su propia porción de
    validación. verbose=False en los ajustar_*/aplicar_* internos: con 15 folds, el log completo
    de cada uno (idéntico al de la corrida única) inundaría la salida."""
    etiquetas = universo['comuna'].fillna('Desconocida')
    scores = []
    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + repeat)
        for fold, (pos_train, pos_val) in enumerate(skf.split(universo, etiquetas)):
            idx_train_fold = universo.index[pos_train]
            idx_val_fold = universo.index[pos_val]

            params_imp_fold = ajustar_imputacion(universo, idx_train_fold, verbose=False)
            df_fold = aplicar_imputacion(universo, params_imp_fold, verbose=False)

            X_fold, y_fold = seleccionar_columnas(df_fold, verbose=False)
            X_tr, X_val = X_fold.loc[idx_train_fold], X_fold.loc[idx_val_fold]
            y_tr, y_val = y_fold.loc[idx_train_fold], y_fold.loc[idx_val_fold]

            params_feat_fold = ajustar_features(X_tr, y_tr, verbose=False)
            X_val_final = aplicar_features(X_val, params_feat_fold)

            if X_val_final.shape[1] != params_feat_fold.X_train_final.shape[1]:
                print(f'[repeat {repeat} fold {fold}] ADVERTENCIA: columnas desalineadas -- '
                      f'train {params_feat_fold.X_train_final.shape[1]} vs val {X_val_final.shape[1]}')
                X_val_final = X_val_final.reindex(columns=params_feat_fold.X_train_final.columns, fill_value=0)

            modelo_fold = HistGradientBoostingRegressor(random_state=42)
            modelo_fold.fit(params_feat_fold.X_train_final, np.log(y_tr))
            pred_fold = np.exp(modelo_fold.predict(X_val_final))
            score = mdape(y_val, pred_fold)
            scores.append(score)
            print(f'[repeat {repeat} fold {fold}] MdAPE: {score:.2f}% (train: {len(idx_train_fold)}, '
                  f'val: {len(idx_val_fold)})')

    scores = pd.Series(scores)
    print(f'MdAPE CV ({n_splits}x{n_repeats}={len(scores)} folds): '
          f'{scores.mean():.2f}% ± {scores.std():.2f}% '
          f'(min {scores.min():.2f}%, max {scores.max():.2f}%)')
    return scores


resultados_cv = ejecutar_cv_repetida(deptos_df_limpio.loc[indice_train])

# =======================================================================================
# PASO 5b -- MODELO LINEAL LOG-LOG (pendiente)
# =======================================================================================
# La cabecera lo promete (línea 36) y nunca se implementó: hoy el único modelo del script es
# HistGB. Importa por tres razones concretas, ninguna de ellas "porque estaba en el plan":
#   1. PASO 4g rechazó PCA explícitamente para preservar la interpretabilidad de los coeficientes.
#      Hoy no hay ningún modelo con coeficientes que preservar: el argumento está sin respaldo.
#   2. Es la cota inferior honesta de complejidad. Si HistGB no le gana por un margen mayor al ruido
#      medido en 5a, la ganancia no paga la opacidad.
#   3. Los coeficientes son el chequeo de sanidad del feature engineering de PASO 4: un signo
#      invertido (más baños -> menos precio) delata colinealidad residual o un bug de construcción
#      que el modelo de árboles esconde sin avisar.
#
# Detalle de implementación:
#   - Las superficies ya vienen log1p + escaladas de PASO 4d, así que regresar log(clp) contra ellas
#     YA es la especificación log-log; no hay que volver a transformar nada. Al interpretar, ojo:
#     el coeficiente está en unidades de desvío estándar de log(superficie), así que la elasticidad
#     es beta / escalador_log.scale_[i], no beta.
#   - Ridge, no OLS. PASO 4g toleró VIF residual de 10-14 por ser señal real y no identidad
#     matemática, y hay 82 columnas con varios one-hot casi constantes: OLS devuelve coeficientes de
#     varianza enorme y signo inestable. El alfa se elige con la CV de 5a.
#   - PROBLEMA DE ESCALA A RESOLVER ANTES: 'target__barrio' se ajusta contra y_train en CLP (PASO 4b,
#     línea ~1154) y NO se escala en PASO 4d. Entra al modelo en unidades de ~10^8 mientras el resto
#     de las features tiene varianza 1. Para árboles da lo mismo; para Ridge no, porque la
#     penalización es sensible a la escala y aplastaría esa columna -- que es una de las features más
#     fuertes del dataset. Para el lineal hay que pasarla a log y estandarizarla (ajustando el
#     escalador solo con train, como todo lo demás). Es un arreglo local al modelo lineal, no un
#     cambio a PASO 4.
#   - Interpretación de binarias: el efecto sobre el precio es exp(beta) - 1 (semi-elasticidad), no
#     beta. Documentarlo junto a la tabla de coeficientes o se va a leer mal.
#   - Evaluar en CLP (exp de la predicción), no en log, para que sea comparable con todo lo demás.
#     Ojo con el sesgo de retransformación: E[exp(log y)] != exp(E[log y]). Para el ranking importa
#     poco (el sesgo es multiplicativo y casi constante), pero si se reporta MAE en CLP hay que
#     corregirlo (Duan smearing) o declararlo explícitamente.

# =======================================================================================
# PASO 5c -- TUNING DE HIPERPARÁMETROS (pendiente)
# =======================================================================================
# HistGB corre con defaults en todo el script. Estimación NO medida: 1-2 puntos de MdAPE
# disponibles. Es la última optimización que conviene hacer, no la primera: mejora el número pero no
# cambia lo que el proyecto puede responder, a diferencia de 5e.
#
#   - Búsqueda: RandomizedSearchCV o HalvingRandomSearchCV sobre learning_rate, max_leaf_nodes,
#     min_samples_leaf, l2_regularization, max_features y max_bins. max_iter no se busca: se fija
#     alto y se deja que early_stopping + validation_fraction lo corten.
#   - Scorer: MdAPE en CLP, no el neg_MSE por defecto sobre log. El objetivo del negocio es el error
#     porcentual mediano; optimizar MSE sobre log optimiza otra cosa (media geométrica) y pondera la
#     cola de forma distinta. Se envuelve mdape() con make_scorer(greater_is_better=False) y se
#     invierte el log DENTRO del scorer.
#   - Solo sobre train, con la CV de 5a.
#   - Criterio de parada: si la mejor combinación no le gana al default por más que el desvío entre
#     folds, se queda el default. Menos superficie de mantenimiento por la misma métrica.

# =======================================================================================
# PASO 5d -- EVALUACIÓN FINAL EN TEST (pendiente)
# =======================================================================================
# Se corre UNA vez, cuando 5a-5c ya cerraron. Reportar el set completo de métricas, no solo MdAPE:
#
#   MdAPE   error porcentual mediano -- la métrica de negocio, robusta a outliers
#   MAE     error absoluto medio en CLP -- cuánta plata, en promedio
#   RMSE    penaliza la cola -- hoy RMSE ~ 2 x MAE, que delata errores grandes concentrados
#   p10/p50/p90/p99 del residuo -- la forma completa del error, no su centro
#
# Por qué las cuatro: con MdAPE 8,39% y p99 de +80,8%, la mediana esconde por completo la cola. En
# un negocio donde equivocarse por 300 millones de CLP es catastrófico, reportar solo la mediana es
# engañoso.
#
# Desagregar por comuna y por decil de precio. Hipótesis a verificar (no medida): el error se
# concentra en el decil superior, donde las casas son pocas, heterogéneas (vista, terreno,
# arquitectura) y donde un mismo error porcentual cuesta mucha más plata. Si se confirma, la
# conclusión de producto es acotar el alcance de la herramienta a un rango de precio, no promediar
# sobre todo el inventario y aparentar una precisión que no se tiene donde más importa.
#
# Comparar siempre contra los dos pisos: el baseline de mediana precio/m² por barrio (21,50%) y el
# lineal de 5b. Un modelo que no le gana a ambos no justifica su complejidad.

# =======================================================================================
# PASO 5e -- INTERVALOS DE PREDICCIÓN (CQR)
# =======================================================================================
# El problema, medido: con un umbral fijo de ±10% se flaggearía el 43% del inventario (20,6%
# subvaluadas + 22,6% sobrevaluadas). El error del modelo (p10-p90: -15,8% a +20,1%) es del mismo
# orden que el margen de negociación típico en Chile (5-10%). Un flag así no distingue "esta casa
# está barata" de "el modelo no supo predecir esta casa", que es la única pregunta que el proyecto
# necesita responder. Sin esto, todo lo que sigue es ruido ordenado.
#
# CQR (conformalized quantile regression, Romano et al. 2019) sobre las opciones descartadas:
#   - Cuantiles sin calibrar (solo los tres HistGB) no dan garantía de cobertura, y los cuantiles
#     pueden cruzarse (q05 > q95 en filas raras).
#   - Conformal simple sobre el residuo absoluto da ancho CONSTANTE para todas las casas, que no
#     sirve acá: la incertidumbre de una casa estándar en Las Condes y la de un terreno de 5.000 m²
#     en Lo Barnechea no son comparables.
# CQR combina ambas: ancho ADAPTATIVO de los cuantiles (se ensancha donde el modelo es malo) más
# la garantía de cobertura marginal en muestra finita de conformal prediction, sin supuestos de
# distribución del error.
#
# Split adicional DENTRO de train (80% ajuste / 20% calibración, mismo criterio 80/20 y
# estratificado por 'comuna' que el resto del archivo): los tres modelos de cuantil se entrenan
# SOLO con el fold de ajuste, y la corrección conformal se mide SOLO con el fold de calibración,
# que ninguno de los tres vio en su fit -- si se calibrara con datos que el modelo ya vio, la
# corrección subestimaría el error real fuera de muestra. Ninguno de los dos ve test.

ALPHA_INTERVALO = 0.10  # cobertura nominal 90% (q05/q95)
FRACCION_CALIBRACION = 0.2

X_train_ajuste, X_train_calib, y_train_ajuste, y_train_calib = train_test_split(
    X_train_final, y_train, test_size=FRACCION_CALIBRACION, stratify=X_train['comuna'], random_state=42,
)
print(f'Ajuste de cuantiles: {len(X_train_ajuste)} filas. Calibración conformal: {len(X_train_calib)} filas.')


def entrenar_modelo_cuantil(quantile: float) -> HistGradientBoostingRegressor:
    """Igual que el resto del archivo: se entrena sobre log(clp), no CLP directo (ver cabecera) --
    los cuantiles son equivariantes ante transformaciones monótonas, así que exp(cuantil de
    log(clp)) es el cuantil de clp."""
    modelo = HistGradientBoostingRegressor(loss='quantile', quantile=quantile, random_state=42)
    modelo.fit(X_train_ajuste, np.log(y_train_ajuste))
    return modelo


modelo_q05 = entrenar_modelo_cuantil(0.05)
modelo_q50 = entrenar_modelo_cuantil(0.50)
modelo_q95 = entrenar_modelo_cuantil(0.95)

# Conformity score de CQR: cuánto se pasa cada casa de calibración fuera del intervalo crudo
# [q05, q95] -- positivo si cae afuera (por cualquiera de los dos lados), negativo si cae adentro
# con margen. Se mide en CLP (escala real), no en log, porque el intervalo final se reporta en CLP.

q05_calib = np.exp(modelo_q05.predict(X_train_calib))
q95_calib = np.exp(modelo_q95.predict(X_train_calib))
conformity_scores = np.maximum(q05_calib - y_train_calib.to_numpy(), y_train_calib.to_numpy() - q95_calib)

# Corrección conformal: cuantil (1-alpha)(1+1/n) de los conformity scores -- la corrección finita-
# muestra de Romano et al., no simplemente el cuantil (1-alpha), para que la garantía de cobertura
# valga con n_calib finito y no solo asintóticamente.

n_calib = len(conformity_scores)
nivel_ajustado = min(1.0, (1 - ALPHA_INTERVALO) * (1 + 1 / n_calib))
correccion_conformal = np.quantile(conformity_scores, nivel_ajustado)
print(f'Corrección conformal (n_calib={n_calib}, nivel={nivel_ajustado:.4f}): '
      f'{correccion_conformal:,.0f} CLP')

# Intervalo final sobre TEST (nunca visto por el ajuste de cuantiles ni por la calibración):
# predicción cruda +- la corrección conformal, en CLP. Salvaguarda de orden: la corrección es
# simétrica y no debería cruzar los cuantiles, pero si `correccion_conformal` sale negativa (la
# calibración encontró que el intervalo crudo ya sobraba margen) un caso límite podría invertir
# q05/q95 -- se fuerza el orden por seguridad, mismo criterio que el diseño original advertía para
# cuantiles sin calibrar.

q05_test = np.exp(modelo_q05.predict(X_test_final)) - correccion_conformal
q50_test = np.exp(modelo_q50.predict(X_test_final))
q95_test = np.exp(modelo_q95.predict(X_test_final)) + correccion_conformal
q05_test, q95_test = np.minimum(q05_test, q95_test), np.maximum(q05_test, q95_test)

# Validación obligatoria del intervalo:
#   - Cobertura empírica: debe dar ~90%. Si da 70%, el intervalo miente y el ranking de PASO 5f
#     construido sobre él también.
#   - Ancho mediano como % del precio predicho: ESTE número es el resultado real del proyecto. Si
#     el ancho mediano es ±25%, la conclusión honesta es que la herramienta solo puede señalar
#     casos extremos -- es un hallazgo, no un fracaso, y hay que decirlo antes de que alguien
#     decida una compra con ella.
#   - Desagregada por comuna y por decil de precio: un intervalo con 90% global puede tener 60% en
#     el decil caro, que es donde se toman las decisiones más caras.

dentro_del_intervalo = (y_test.to_numpy() >= q05_test) & (y_test.to_numpy() <= q95_test)
cobertura_empirica = dentro_del_intervalo.mean() * 100
ancho_pct = (q95_test - q05_test) / q50_test * 100

print(f'Cobertura empírica en test: {cobertura_empirica:.1f}% (nominal: {(1 - ALPHA_INTERVALO) * 100:.0f}%)')
print(f'Ancho del intervalo (mediana): {np.median(ancho_pct):.1f}% del precio predicho '
      f'(p25: {np.percentile(ancho_pct, 25):.1f}%, p75: {np.percentile(ancho_pct, 75):.1f}%)')

intervalos_test = pd.DataFrame({
    'comuna': X_test['comuna'].to_numpy(),
    'q05': q05_test, 'q50': q50_test, 'q95': q95_test,
    'dentro_del_intervalo': dentro_del_intervalo,
    'ancho_pct': ancho_pct,
}, index=X_test.index)

print('Cobertura por comuna (%):')
print((intervalos_test.groupby('comuna')['dentro_del_intervalo'].mean() * 100).round(1))

intervalos_test['decil_precio'] = pd.qcut(intervalos_test['q50'], 10, labels=False, duplicates='drop')
print('Cobertura por decil de precio predicho (0=más barato, 9=más caro) (%):')
print((intervalos_test.groupby('decil_precio')['dentro_del_intervalo'].mean() * 100).round(1))

# =======================================================================================
# PASO 5f -- RESIDUO, REGLA DE FLAG Y RANKING
# =======================================================================================
# residuo = (precio real - predicho) / predicho, contra q50 (la predicción puntual). Es la
# magnitud DESCRIPTIVA que va en el export -- va en 5h -- pero NO es la regla de decisión: dos
# casas con el mismo residuo negativo pueden estar en lados opuestos del intervalo (una mal
# predicha, con intervalo ancho, y sí puede ser normal; otra bien predicha, con intervalo angosto,
# y el precio real igual se sale) -- ver PASO 5e.

intervalos_test['precio_real'] = y_test.to_numpy()
intervalos_test['residuo_pct'] = (
    (intervalos_test['precio_real'] - intervalos_test['q50']) / intervalos_test['q50'] * 100
)

# Regla de flag: una casa se marca solo si su precio real cae FUERA del intervalo de 5e. Bajo q05
# -> candidata a subvaluada; sobre q95 -> sobrevaluada. Nunca contra un umbral fijo -- es
# precisamente lo que 5e vino a reemplazar.

intervalos_test['flag'] = np.select(
    [intervalos_test['precio_real'] < intervalos_test['q05'],
     intervalos_test['precio_real'] > intervalos_test['q95']],
    ['subvalorada', 'sobrevalorada'],
    default='dentro_del_intervalo',
)

# Ranking: distancia relativa al borde del intervalo, (q05 - precio real) / predicho para
# subvaloradas y (precio real - q95) / predicho para sobrevaloradas -- NO el residuo crudo.
# Ordenar por residuo mezclaría de nuevo "barata" con "mal predicha", que es el error que 5e
# corrige. Positivo y más grande = mejor candidata a subvaluada; negativo y más chico = más
# sobrevalorada; cero = dentro del intervalo, sin rankear ("el modelo no encontró nada raro").
# Ordenar de mayor a menor score da directamente "de mayor a menor descuento" (ver cabecera).

distancia_bajo_q05 = (intervalos_test['q05'] - intervalos_test['precio_real']) / intervalos_test['q50'] * 100
distancia_sobre_q95 = (intervalos_test['precio_real'] - intervalos_test['q95']) / intervalos_test['q50'] * 100
intervalos_test['score_ranking'] = np.select(
    [intervalos_test['flag'] == 'subvalorada', intervalos_test['flag'] == 'sobrevalorada'],
    [distancia_bajo_q05, -distancia_sobre_q95],
    default=0.0,
)
intervalos_test = intervalos_test.sort_values('score_ranking', ascending=False)

# Calibración de expectativas: si el intervalo está bien calibrado, ~5% cae bajo q05 y ~5% sobre
# q95 POR CONSTRUCCIÓN (alpha/2 cada lado). Un flag no es evidencia de ganga, es una candidata a
# inspección -- el residuo negativo puede ser oportunidad real o un defecto que el modelo no
# observa (sin vista, mala orientación, mal estado de conservación, problema legal, uso de suelo).
# El ranking genera una lista de visitas, no una tasación.

print(intervalos_test['flag'].value_counts())
pct_subvalorada = (intervalos_test['flag'] == 'subvalorada').mean() * 100
pct_sobrevalorada = (intervalos_test['flag'] == 'sobrevalorada').mean() * 100
print(f'% subvaloradas: {pct_subvalorada:.1f}% (esperado ~{ALPHA_INTERVALO / 2 * 100:.1f}% por construcción)')
print(f'% sobrevaloradas: {pct_sobrevalorada:.1f}% (esperado ~{ALPHA_INTERVALO / 2 * 100:.1f}% por construcción)')

print('Top 10 candidatas subvaloradas (test):')
print(intervalos_test.loc[intervalos_test['flag'] == 'subvalorada',
                           ['comuna', 'precio_real', 'q05', 'q50', 'q95', 'residuo_pct', 'score_ranking']].head(10))

# =======================================================================================
# PASO 5g -- FILTROS DEL COMPRADOR SOBRE EL RANKING
# =======================================================================================
# Dormitorios y baños mínimos, comuna, presupuesto máximo, distancia máxima a un punto de interés.
# Se aplican SOBRE el ranking ya construido en 5f, nunca como criterio de selección principal --
# filtrar antes cambiaría la población y con ella los percentiles sobre los que se calculó el
# intervalo (5e) y el ranking (5f).
#
# Reemplaza el filtro fijo original (precio UF < 15.000, Dormitorios >= 3): ese filtro no se ajusta
# por comuna ni por superficie, y "barato" en Lo Barnechea y en Las Condes no es el mismo número --
# que es exactamente lo que el modelo sí sabe y el filtro fijo tiraba a la basura.
#
# Todavía no hay un comprador real, así que los campos quedan como PLACEHOLDERS configurables --
# todos en None por defecto (sin filtro). Se ajustan acá cuando exista un caso de uso concreto.


@dataclass
class FiltrosComprador:
    dormitorios_min: int = None
    banos_min: int = None
    comunas: list = None  # None = todas; si no, lista de nombres tal como aparecen en 'comuna'
    presupuesto_max_clp: float = None
    distancia_max_m: float = None
    punto_interes: str = None  # clave de PUNTOS_DE_INTERES (PASO 4e); requerido junto con distancia_max_m


def aplicar_filtros_comprador(ranking: pd.DataFrame, X_crudo: pd.DataFrame,
                               filtros: FiltrosComprador) -> pd.DataFrame:
    """Filtra `ranking` (ya ordenado por score_ranking, ver PASO 5f) según las restricciones del
    comprador -- nunca antes de rankear. `X_crudo` es el dataframe SIN codificar (X_test), para
    leer Dormitorios/Baños/latitud/longitud tal como vienen, no las columnas escaladas de
    X_test_final."""
    mascara = pd.Series(True, index=ranking.index)

    if filtros.dormitorios_min is not None:
        mascara &= X_crudo.loc[ranking.index, 'Dormitorios'] >= filtros.dormitorios_min
    if filtros.banos_min is not None:
        mascara &= X_crudo.loc[ranking.index, 'Baños'] >= filtros.banos_min
    if filtros.comunas is not None:
        mascara &= ranking['comuna'].isin(filtros.comunas)
    if filtros.presupuesto_max_clp is not None:
        mascara &= ranking['precio_real'] <= filtros.presupuesto_max_clp
    if filtros.distancia_max_m is not None:
        if filtros.punto_interes not in PUNTOS_DE_INTERES:
            raise ValueError(f"punto_interes debe ser uno de {list(PUNTOS_DE_INTERES)}, "
                              f"no {filtros.punto_interes!r}")
        lat_punto, lon_punto = PUNTOS_DE_INTERES[filtros.punto_interes]
        distancia_m = haversine_m(X_crudo.loc[ranking.index, 'latitud'], X_crudo.loc[ranking.index, 'longitud'],
                                   lat_punto, lon_punto)
        mascara &= distancia_m <= filtros.distancia_max_m

    return ranking.loc[mascara]


filtros_comprador = FiltrosComprador()  # sin filtro por defecto -- ajustar acá según el comprador
ranking_filtrado = aplicar_filtros_comprador(intervalos_test, X_test, filtros_comprador)
print(f'Ranking sin filtros: {len(intervalos_test)} filas. Con filtros del comprador: {len(ranking_filtrado)} filas.')

# =======================================================================================
# PASO 5h -- EXPORT casas_candidatas.xlsx
# =======================================================================================
# Solo TEST: las predicciones sobre train son in-sample (los tres modelos de cuantil las vieron en
# su fit -- o en la calibración conformal) y tienen residuos artificialmente chicos. La alternativa
# correcta -- predicciones out-of-fold para train reusando la CV de 5a -- necesitaría que
# `ejecutar_cv_repetida` devolviera las predicciones por fila además del MdAPE agregado; queda
# pendiente (ver PENDIENTES.md). Mezclar train sin marcarlo produciría un ranking donde esas casas
# parecen sistemáticamente mejor tasadas de lo que están.
#
# Valores CRUDOS, no las columnas de X_test_final: nadie puede leer "Superficie total = -0,42".
# 'barrio' y las columnas físicas salen de X_test (ya imputado en PASO 2b, pero sin escalar ni
# codificar); 'url' y 'precio'/'UM' no están en X_test (PASO 4a las excluyó por ser identificador o
# fuga) y se buscan en `deptos_df` por el mismo índice. 'precio' se deja en su moneda original (UF
# o CLP según 'UM') en vez de forzar una conversión a UF: no hay tipo de cambio UF/CLP histórico en
# este proyecto, e inventar uno sería peor que mostrar el dato tal como se publicó.
#
# Solo filas FLAGGEADAS (fuera del intervalo, ver PASO 5f): el archivo se llama "casas candidatas",
# no "todas las casas de test" -- CLAUDE.md ya lo describe como "subset flagged". Las ~92% que
# caen dentro del intervalo ("el modelo no encontró nada raro") no son candidatas a nada y solo
# ensuciarían el archivo. PASO 5i (mapa) sí necesita el set completo -- usa `ranking_filtrado`
# directamente, no este export.

CASAS_CANDIDATAS_XLSX = 'casas_candidatas.xlsx'

columnas_export_crudas = ['barrio', 'Superficie total', 'Superficie útil', 'Dormitorios', 'Baños',
                          'Estacionamientos', 'Antigüedad', 'latitud', 'longitud']

casas_candidatas = ranking_filtrado.loc[ranking_filtrado['flag'] != 'dentro_del_intervalo']
casas_candidatas = casas_candidatas.join(X_test[columnas_export_crudas])
casas_candidatas = casas_candidatas.join(deptos_df.loc[casas_candidatas.index, ['url', 'precio', 'UM']])

columnas_finales = [
    'url', 'comuna', 'barrio', 'flag', 'score_ranking',
    'precio_real', 'precio', 'UM', 'q05', 'q50', 'q95', 'ancho_pct', 'residuo_pct',
    'Superficie total', 'Superficie útil', 'Dormitorios', 'Baños', 'Estacionamientos', 'Antigüedad',
    'latitud', 'longitud',
]
casas_candidatas = casas_candidatas[columnas_finales].sort_values('score_ranking', ascending=False)
casas_candidatas.to_excel(CASAS_CANDIDATAS_XLSX, index=False)

print(f'{CASAS_CANDIDATAS_XLSX}: {len(casas_candidatas)} filas exportadas '
      f'-- {(casas_candidatas["flag"] == "subvalorada").sum()} subvaloradas, '
      f'{(casas_candidatas["flag"] == "sobrevalorada").sum()} sobrevaloradas.')

# =======================================================================================
# PASO 5i -- MAPA
# =======================================================================================
# folium (Leaflet.js) sobre plotly: la única necesidad acá es un scatter geográfico con
# tooltip -- no hace falta la superficie de features de plotly, y folium da un mapa base de calles
# real (OpenStreetMap) en vez de un scatter plano, que para ubicar CASAS concretas importa. Se
# agrega como dependencia nueva (no estaba en el venv); instalada con `pip install folium`.
#
# Coloreado de rojo (sobrevalorada, precio real sobre q95) a verde (subvalorada, precio real bajo
# q05), gris para todo lo que cae dentro del intervalo -- el gris es la mayoría (91,7% medido en
# PASO 5e) y tiene que verse como tal: se dibuja con radio chico y opacidad baja para no competir
# visualmente con los puntos flaggeados, en vez de omitirse (omitirlo exageraría la cantidad de
# señal disponible, viendo solo casos "interesantes").
#
# Se usa `ranking_filtrado` (el set de test completo, no `casas_candidatas` que 5h ya filtró a
# solo flaggeadas) -- el mapa necesita el gris de fondo para que la proporción se lea bien. Las
# coordenadas ya están completas (99,98%) tras la corrección de PASO 2, no hace falta filtrar filas.
# gráficos/ hoy solo guarda PNG y está commiteado -- el HTML se guarda ahí mismo, mismo criterio de
# "artefacto de una corrida, commiteado y sobreescrito" que ya aplica a los PNG y a los .xlsx.

import folium

MAPA_HTML = os.path.join(GRAFICOS_DIR, 'mapa_intervalos.html')


def color_por_residuo(precio_real: float, q50: float) -> str:
    """Rojo si el precio real está sobre la predicción puntual (sobrevalorada respecto a q50),
    verde si está bajo (subvalorada). A pedido explícito: sin gris -- toda casa se clasifica, no
    solo las que caen fuera del intervalo de 5e."""
    return 'red' if precio_real >= q50 else 'green'


mapa_datos = ranking_filtrado.join(X_test[['latitud', 'longitud']])
mapa_datos = mapa_datos.join(deptos_df.loc[mapa_datos.index, ['url']])

mapa = folium.Map(location=[mapa_datos['latitud'].mean(), mapa_datos['longitud'].mean()],
                   zoom_start=12, tiles='OpenStreetMap')

for _, fila in mapa_datos.iterrows():
    # El color ya no distingue flaggeada/no-flaggeada (ver color_por_residuo) -- esa distinción se
    # conserva en el tamaño/opacidad: las flaggeadas (fuera del intervalo de 5e) se ven más grandes
    # y sólidas, el resto más chicas y tenues, para no perder del todo la señal de "esto es un caso
    # extremo" vs. "esto está apenas a un lado de la mediana".
    es_flaggeada = fila['flag'] != 'dentro_del_intervalo'
    tooltip = (f"<b>{fila['comuna']}</b><br>"
               f"Precio real: {fila['precio_real']:,.0f} CLP<br>"
               f"Predicho (q50): {fila['q50']:,.0f} CLP<br>"
               f"Intervalo 90%: [{fila['q05']:,.0f}, {fila['q95']:,.0f}] CLP<br>"
               f"<a href='{fila['url']}' target='_blank'>Ver aviso</a>")
    folium.CircleMarker(
        location=[fila['latitud'], fila['longitud']],
        radius=6 if es_flaggeada else 3,
        color=color_por_residuo(fila['precio_real'], fila['q50']),
        fill=True, fill_opacity=0.75 if es_flaggeada else 0.35,
        opacity=0.85 if es_flaggeada else 0.4,
        tooltip=tooltip,
    ).add_to(mapa)

mapa.save(MAPA_HTML)
print(f'{MAPA_HTML}: {len(mapa_datos)} casas de test graficadas '
      f'({(mapa_datos["flag"] != "dentro_del_intervalo").sum()} flaggeadas).')

# =======================================================================================
# PASO 5i (extra) -- MAPA CON TODO EL DATASET (train + test), a pedido explícito
# =======================================================================================
# El mapa de arriba usa solo test porque es el único fold con una garantía de cobertura válida.
# Este segundo mapa aplica los mismos tres modelos de cuantil + la misma corrección conformal a
# TODO deptos_df (5.284 filas), sin reajustar nada -- diagnóstico/exploratorio, no una métrica a
# reportar. Dentro de las filas de train hay dos grupos con un problema distinto cada uno:
#   - Ajuste (3.381 filas): el modelo las vio directo en su fit -- predicción in-sample, se ve
#     "mejor calibrada" de lo real casi por definición, haya o no overfitting.
#   - Calibración (846 filas): no las vio el fit, pero DEFINIERON la corrección conformal --
#     aplicarles esa misma corrección es circular, no es su desempeño fuera de muestra real.
# El tooltip marca 'train'/'test' por punto para que quede transparente cuál es cuál.

MAPA_COMPLETO_HTML = os.path.join(GRAFICOS_DIR, 'mapa_intervalos_completo.html')

X_codificado_completo = aplicar_codificacion(X, params_codificacion)
X_final_completo = aplicar_escalado_numerico(X_codificado_completo, params_escalado)
X_final_completo = aplicar_features_distancia(X_final_completo, X['comuna'], params_distancia)
X_final_completo = aplicar_ratios(X_final_completo, X, params_ratios)
X_final_completo = X_final_completo.drop(columns=COLUMNAS_DROP_ESTRUCTURAL)
X_final_completo = agregar_gastos_comunes_informado(X_final_completo, X)
X_final_completo = aplicar_densidad(X_final_completo, X, params_densidad)

q05_completo = np.exp(modelo_q05.predict(X_final_completo)) - correccion_conformal
q50_completo = np.exp(modelo_q50.predict(X_final_completo))
q95_completo = np.exp(modelo_q95.predict(X_final_completo)) + correccion_conformal
q05_completo, q95_completo = np.minimum(q05_completo, q95_completo), np.maximum(q05_completo, q95_completo)

precio_real_completo = y.to_numpy()
flag_completo = np.select(
    [precio_real_completo < q05_completo, precio_real_completo > q95_completo],
    ['subvalorada', 'sobrevalorada'],
    default='dentro_del_intervalo',
)

mapa_datos_completo = pd.DataFrame({
    'comuna': X['comuna'].to_numpy(), 'latitud': X['latitud'].to_numpy(), 'longitud': X['longitud'].to_numpy(),
    'precio_real': precio_real_completo, 'q05': q05_completo, 'q50': q50_completo, 'q95': q95_completo,
    'flag': flag_completo,
}, index=X.index)
mapa_datos_completo['fold'] = np.where(mapa_datos_completo.index.isin(indice_train), 'train', 'test')
mapa_datos_completo = mapa_datos_completo.join(deptos_df.loc[mapa_datos_completo.index, ['url']])

mapa_completo = folium.Map(location=[mapa_datos_completo['latitud'].mean(), mapa_datos_completo['longitud'].mean()],
                            zoom_start=12, tiles='OpenStreetMap')

# Color por signo del residuo contra q50 (ver color_por_residuo, definida junto al mapa de solo
# test) -- sin gris, toda casa se clasifica rojo/verde. 'fold' nunca entra en el color, train y
# test comparten el mismo criterio. Igual que en el mapa de test, el tamaño/opacidad sí distingue
# flaggeada (fuera del intervalo) de no-flaggeada, para no perder esa señal. Con 5.284 puntos (5x
# más que el mapa de solo test) hay mucha más superposición geográfica, por eso la opacidad de
# ambos grupos es más baja acá que en el mapa de test.

for _, fila in mapa_datos_completo.iterrows():
    es_flaggeada = fila['flag'] != 'dentro_del_intervalo'
    tooltip = (f"<b>{fila['comuna']}</b> ({fila['fold']})<br>"
               f"Precio real: {fila['precio_real']:,.0f} CLP<br>"
               f"Predicho (q50): {fila['q50']:,.0f} CLP<br>"
               f"Intervalo 90%: [{fila['q05']:,.0f}, {fila['q95']:,.0f}] CLP<br>"
               f"<a href='{fila['url']}' target='_blank'>Ver aviso</a>")
    folium.CircleMarker(
        location=[fila['latitud'], fila['longitud']],
        radius=6 if es_flaggeada else 3,
        color=color_por_residuo(fila['precio_real'], fila['q50']),
        fill=True, fill_opacity=0.55 if es_flaggeada else 0.2,
        opacity=0.6 if es_flaggeada else 0.25,
        tooltip=tooltip,
    ).add_to(mapa_completo)

mapa_completo.save(MAPA_COMPLETO_HTML)
print(f'{MAPA_COMPLETO_HTML}: {len(mapa_datos_completo)} casas graficadas (train+test) '
      f'({(mapa_datos_completo["flag"] != "dentro_del_intervalo").sum()} flaggeadas) -- '
      f'diagnóstico, no tiene garantía de cobertura para las filas de train.')

# =======================================================================================
# PASO 5j -- LÍMITES: QUÉ NO PUEDE RESPONDER ESTE MODELO
# =======================================================================================
# No son bugs ni pendientes, son techos del enfoque. Van escritos acá para que el ranking no se
# sobre-interprete.
#
#   - EL TARGET ES PRECIO DE LISTA, NO DE TRANSACCIÓN. El modelo responde "¿esta casa pide más o
#     menos que casas comparables que también están pidiendo?", no "¿vale más o menos de lo que
#     pide?". Si todo el sector oriente está listado 12% sobre el precio de transacción real, el
#     modelo absorbe ese 12% como media y no detecta nada: no hay ancla externa. El mayor salto de
#     valor disponible es cruzar contra transacciones reales del CBR (Conservador de Bienes Raíces),
#     que convierte esto de detector de listings anómalos en un AVM de verdad.
#   - NO HAY DIMENSIÓN TEMPORAL. Verificado sobre los 77 campos del scrape crudo: no hay fecha de
#     publicación, ni días en mercado, ni historial de precio. Días-en-mercado es la señal de
#     sobreprecio por preferencia revelada más fuerte del dominio -- una casa 14 meses publicada sin
#     vender está sobrevaluada, y eso es un hecho, no una predicción. Agregar fecha_publicacion al
#     spider es un cambio chico y su valor es acumulativo: conviene empezar a acumular histórico
#     cuanto antes, aunque el modelo todavía no lo use.
#   - VARIABLES AUSENTES DEL SECTOR: vista, pendiente y exposición (determinantes en Lo Barnechea),
#     calidad de terminaciones más allá de binarios, estado de conservación. Parte de eso vive en
#     'descripcion', que PASO 4c midió y descartó (8,39% vs 8,40% con TF-IDF+SVD) -- revisable solo
#     si cambia el modelo o si crece mucho el volumen de datos.

