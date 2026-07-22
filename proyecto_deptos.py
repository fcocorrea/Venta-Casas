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
# Implementación, en cinco grandes pasos:
# 1) Obtención de datos: se extraen vía web scraping con Scrapy (ver deptos_scraper/spiders/deptos.py;
#    se ejecuta con `scrapy crawl deptos -O deptos.json`, documentado en CLAUDE.md).
# 2) Limpieza de datos: homologar y transformar los datos, que no vienen expresados de forma
#    consistente. Las imputaciones que aprenden de los datos (KDTree de comuna/barrio, moda por
#    barrio, match por correlación en atributos discretos, umbrales de outlier) se ajustan SOLO
#    con el fold de entrenamiento -- por eso el split vive en PASO 2b, antes de todas ellas, y no
#    en PASO 4 -- y ninguna usa precio/clp/precio unitario como predictor, para no filtrar el
#    target hacia adentro de una feature (ver COLUMNAS_TARGET en la matriz de correlación).
# 3) Exploración: entender los datos para adaptarlos a un modelo de regresión.
# 4) Modelos: separar train/test, probar un modelo lineal (log-log, interpretable) y uno de
#    árboles con boosting (captura interacciones comuna × superficie × amenities), evaluar con
#    MdAPE/MAE/RMSE en CLP (no solo en escala log) contra un baseline de mediana por barrio, y
#    quedarnos con el más adecuado.
# 5) Producción: para cada casa, calcular el residuo (precio real - predicho) / predicho y
#    ordenar de mayor a menor descuento -- ese ranking reemplaza el filtro fijo actual (precio UF
#    < 15000, Dormitorios >= 3, etc.), que no se ajusta por comuna ni superficie. Las
#    restricciones del comprador (dormitorios, baños, cercanía) se aplican como filtro sobre el
#    ranking, no como criterio principal de selección. El resultado se ubica en un mapa,
#    coloreado de rojo (residuo positivo, sobrevalorada) a verde (residuo negativo, subvalorada).

import os
import re

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

# Imputación de comuna (aprende de los datos -> ajustada SOLO con train):
# - comuna nula y barrio conocido: si en train ese barrio identifica una comuna, se usa la más
#   frecuente de train.
# - comuna nula y el "barrio" es en realidad el nombre de una comuna (el artefacto de la ruta de
#   navegación descrito arriba): se promueve barrio -> comuna y el barrio queda nulo.
# - lo que no resuelva ninguna de las dos reglas queda nulo y lo toma el vecino más cercano.
# Un barrio que solo aparece en test no está en el mapeo y cae al vecino más cercano, que es el
# comportamiento correcto: train no tiene nada que decir sobre él.

barrios_train = deptos_df.loc[indice_train].dropna(subset=['barrio', 'comuna'])
barrio_a_comuna = barrios_train.groupby('barrio')['comuna'].agg(lambda s: s.value_counts().idxmax())
comunas_validas = set(deptos_df.loc[indice_train, 'comuna'].dropna().unique())

comuna_nula = deptos_df['comuna'].isna()
desde_barrio = comuna_nula & deptos_df['barrio'].isin(barrio_a_comuna.index)
deptos_df.loc[desde_barrio, 'comuna'] = deptos_df.loc[desde_barrio, 'barrio'].map(barrio_a_comuna)

barrio_es_comuna = deptos_df['comuna'].isna() & deptos_df['barrio'].isin(comunas_validas)
deptos_df.loc[barrio_es_comuna, 'comuna'] = deptos_df.loc[barrio_es_comuna, 'barrio']
deptos_df.loc[barrio_es_comuna, 'barrio'] = np.nan

print(f'Comunas imputadas desde el barrio: {desde_barrio.sum()}. '
      f'Barrios que eran nombre de comuna: {barrio_es_comuna.sum()}.')
print(f'Quedan {deptos_df.comuna.isna().sum()} comunas nulas para el vecino más cercano.')

# Para los registros donde comuna y/o barrio siguen nulos, se imputan con el valor de la casa
# conocida más cercana en latitud/longitud -- más preciso que fuzzy-matching de texto sobre
# 'dirección' (que además viene vacía en el crawl actual de casas) y aprovecha que casi todas las
# filas sí tienen coordenadas (ver deptos.json: solo 1 de 5603 sin latitud/longitud).

from scipy.spatial import cKDTree


def impute_nearest_by_location(df: pd.DataFrame, column: str) -> None:
    """El árbol de vecinos se construye SOLO con filas de train (`indice_train`): son las únicas
    que pueden "enseñar" un valor. Las filas a completar son de ambos folds -- a una fila de test
    se le puede asignar la comuna de su vecino de train, pero nunca la de otro vecino de test."""
    has_coords = df['latitud'].notna() & df['longitud'].notna()
    known = df.loc[df.index.isin(indice_train) & df[column].notna() & has_coords]
    missing = df.loc[df[column].isna() & has_coords]
    if known.empty or missing.empty:
        return
    # ponytail: distancia euclidiana en grados, no haversine -- a la escala de Santiago (unas
    # decenas de km) el orden de vecino-más-cercano no cambia; pasar a haversine solo si se
    # necesitan distancias reales en algún otro cálculo.
    tree = cKDTree(known[['latitud', 'longitud']].to_numpy())
    _, nearest_idx = tree.query(missing[['latitud', 'longitud']].to_numpy())
    df.loc[missing.index, column] = known[column].to_numpy()[nearest_idx]


impute_nearest_by_location(deptos_df, 'comuna')
print(f'Ahora hay {deptos_df.comuna.isna().sum()} comunas con valores nulos')

# Barrios:
# mismo problema que con las comunas. Un mismo barrio puede aparecer escrito de formas distintas
# (ej. "las condes" vs "Las Condes"), así que se capitalizan todos.

deptos_df.barrio = deptos_df.barrio.str.title()
print(f'Hay {deptos_df.barrio.isna().sum()} casas sin barrio.')

impute_nearest_by_location(deptos_df, 'barrio')
print(f'Ahora hay {deptos_df.barrio.isna().sum()} casas sin barrio.')

# Rangos de valores:
# revisamos qué atributos tienen valores inverosímiles (mal tipeados o sin sentido) y los
# corregimos.

summary = deptos_df.describe()
print(summary)

# Último cuartil:
# en las filas 75% y max, si la diferencia entre ambas es enorme para un atributo numérico, suele
# ser un error de tipeo o de contexto. Por ejemplo, en "Antigüedad" el cuantil 75% son ~20 años,
# pero el máximo puede ser un año como 2013: imposible como antigüedad, pero tiene sentido si en
# realidad es el año de construcción.

from datetime import datetime

year = datetime.now().year
deptos_df['Antigüedad'] = np.where(deptos_df['Antigüedad'] >= 1800, year - deptos_df['Antigüedad'], deptos_df['Antigüedad'])

# Con la antigüedad corregida, valores sobre 1.000 años restantes o negativos no tienen sentido
# (ambos aparecen en el scrape: años mal tipeados y algún valor negativo corrupto): se dejan nulos.

deptos_df['Antigüedad'] = np.where((deptos_df['Antigüedad'] > 1000) | (deptos_df['Antigüedad'] < 0),
                                    np.nan, deptos_df['Antigüedad'])

# El mismo tipo de revisión aplica a otros atributos discretos propios de una casa (a diferencia
# de un departamento, no hay "número de piso de la unidad" ni "departamentos por piso"; en cambio
# "Cantidad de pisos" puede tener el mismo tipo de error de tipeo, ej. 2013 en vez de 2). Se
# incluye también 'Antigüedad' acá: el filtro de arriba (>1.000 o <0) no descarta el caso de 939
# años, que sigue siendo un error de tipeo evidente para una casa en estas comunas -- el mismo
# corte por salto relativo que ya se usa para las demás lo anula sin tener que hardcodear "939".

bad_ranges = [c for c in ['Baños', 'Estacionamientos', 'Bodegas', 'Cantidad de pisos', 'Antigüedad'] if c in deptos_df.columns]

# El umbral de corte se lee del top-20 de TRAIN, no del dataset completo: es un parámetro estimado
# a partir de la distribución (dónde está el salto que delata el error de tipeo), y estimarlo con
# test adentro sería dejar que test defina su propio criterio de limpieza. Una vez fijado con
# train, se aplica a ambos folds -- un valor imposible en test se anula igual, pero según el
# umbral que train consideró imposible.

for column in bad_ranges:
    top_20 = deptos_df.loc[indice_train, column].dropna().nlargest(20).sort_values()
    cut_value = cut_after_relative_jump(top_20, threshold=1.0)
    if cut_value is not None:
        idx = deptos_df[deptos_df[column] >= cut_value].index
        print(f'Asignamos a nan {len(idx)} valores de la columna "{column}" (>= {cut_value})')
        deptos_df.loc[idx, column] = np.nan

# Valores negativos o ceros:
# no tiene sentido que una casa tenga 0 baños o dormitorios, ni que ningún atributo de conteo sea
# negativo. Se exceptúan bodegas, estacionamientos y antigüedad, donde 0 es válido (una casa puede
# no tener bodega/estacionamiento, o ser recién construida). Como no se puede saber el valor real,
# se dejan como nulos para imputar después.
#
# latitud/longitud quedan fuera de esta regla: en Santiago ambas son negativas por definición
# (hemisferio sur/oeste), así que "negativo" no es un error para ellas -- ya se corrigió el único
# caso real (signo invertido) más arriba, antes de las imputaciones por vecino más cercano.

coordenadas = {'latitud', 'longitud'}
zero_ok_columns = {'Bodegas', 'Estacionamientos', 'Antigüedad'}
for column in summary.columns:
    if not pd.api.types.is_numeric_dtype(deptos_df[column]) or column in coordenadas:
        continue
    invalid = deptos_df[column] < 0
    if column not in zero_ok_columns:
        invalid |= deptos_df[column] == 0
    if invalid.any():
        print(f'Anulamos {invalid.sum()} valores inválidos en "{column}"')
        deptos_df.loc[invalid, column] = np.nan

# Valores Nulos:
# al ser datos anotados manualmente por cada vendedor, es normal tener muchos valores nulos,
# especialmente en los atributos categóricos de las tablas de cada publicación.

total_observations = len(deptos_df)
na_values = deptos_df.isnull().sum()
percent_na = ((na_values / total_observations) * 100).round(2)
na_summary = pd.concat([na_values, percent_na], axis=1)
na_summary.columns = ['Cantidad de Nulos', 'Porcentaje de Nulos']
na_summary = na_summary[na_summary['Cantidad de Nulos'] > 0]
na_summary.sort_values('Cantidad de Nulos', ascending=False, inplace=True)
print(na_summary)

import seaborn as sns

sns.set(style='whitegrid')
plt.figure(figsize=(16, 8))
sns.set(font_scale=.75)
ax = sns.barplot(x='Porcentaje de Nulos', y=na_summary.index, data=na_summary)
plt.title('Número de valores nulos por variable')
plt.xlabel('Porcentaje valores nulos')
plt.ylabel('Atributos')
ax.set_xlim(0, 100)
plt.savefig(os.path.join(GRAFICOS_DIR, 'valores_nulos_por_variable.png'))


# Imputación de valores nulos en variables categóricas binarias:
# el scrape solo marca un amenity cuando la casa lo tiene ('Sí'); un nulo no significa "desconocido",
# significa que el vendedor no lo marcó porque la casa no lo tiene. Por eso no hay nada que predecir
# con un modelo: 'Sí' -> 1, cualquier otro valor (incluido nulo) -> 0.

for binary_feature in binary_features:
    deptos_df[binary_feature] = (deptos_df[binary_feature] == 'Sí').astype(int)


def columns_with_missing_values() -> pd.DataFrame:
    null_values = {column: deptos_df[column].isna().sum()
                   for column in deptos_df.columns
                   if deptos_df[column].isna().sum() > 0}
    null_values = pd.DataFrame(list(null_values.items()), columns=['Atributo', 'Valores Nulos'])
    null_values = null_values.set_index('Atributo').sort_values('Valores Nulos')
    null_values['Tipo de Dato'] = [deptos_df[column].dtype for column in null_values.index]
    return null_values


print(columns_with_missing_values())

# Relación entre atributos con datos faltantes:
# se revisa la correlación entre los atributos numéricos para poder imputar un atributo nulo a
# partir de otro con alta correlación. numeric_only=True evita que .corr() falle o descarte
# columnas de texto (comuna, barrio, dirección, titulo, url) de forma implícita.
#
# Un atributo sin varianza (ej. un amenity que, tras la imputación, quedó con el mismo valor en
# todas las filas -- Amoblado solo trae 'Sí' en todo el scrape) no tiene correlación definida con
# nada: la fórmula de Pearson queda 0/0 (NaN) para toda su fila y columna. Se descartan esos
# atributos del heatmap y de la búsqueda de matches para imputar, en vez de mostrarlos en blanco.

corr = deptos_df.corr(numeric_only=True)
corr = corr.dropna(axis=0, how='all').dropna(axis=1, how='all')
plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap="Blues", annot=True)
plt.savefig(os.path.join(GRAFICOS_DIR, 'correlacion_atributos.png'))

# El heatmap SÍ debe mostrar el bloque de target ('precio', 'clp', 'precio unitario'): entender
# qué atributo correlaciona con el precio es justamente el objetivo del análisis exploratorio.
#
# La búsqueda de matches para IMPUTAR, en cambio, no puede verlo. Si un atributo nulo se rellena
# a partir del precio, ese atributo deja de ser una característica de la casa y pasa a ser una
# función del target: el modelo lo "usa para predecir" un precio que ya está adentro de la
# feature. Es la regla que declara la cabecera de este archivo ("ninguna [imputación] debe usar
# precio/clp/precio unitario como predictor"), y sin este filtro el código la incumplía --
# medido: 'Estacionamientos' tenía 'clp' como su match de mayor correlación y 20 de sus filas se
# imputaban directamente desde el precio, más 19 de 'Dormitorios'.

# Además, la matriz que guía la imputación se calcula SOLO con train: qué atributo correlaciona
# con cuál es un parámetro aprendido de los datos, igual que una media o una moda. El heatmap de
# arriba sí usa el dataset completo -- es material descriptivo del mercado, no alimenta ninguna
# decisión del modelo.

COLUMNAS_TARGET = ['precio', 'clp', 'precio unitario']

corr_sin_target = (deptos_df.loc[indice_train]
                   .drop(columns=COLUMNAS_TARGET, errors='ignore')
                   .corr(numeric_only=True)
                   .dropna(axis=0, how='all').dropna(axis=1, how='all'))


# Imputación en variables discretas:
# variables numéricas que toman un número finito de valores, propias de una casa (a diferencia del
# departamento no hay "número de piso de la unidad" ni "departamentos por piso").

discrete_values = [c for c in ['Dormitorios', 'Baños', 'Estacionamientos', 'Bodegas',
                                'Cantidad de pisos', 'Antigüedad'] if c in deptos_df.columns]
na_columns = columns_with_missing_values()
print(na_columns[na_columns.index.isin(discrete_values)])


def attribute_correlations(attribute) -> pd.Index:
    """Atributos ordenados de mayor a menor correlación (absoluta) respecto a `attribute`, sobre
    la matriz SIN el bloque de target -- ver COLUMNAS_TARGET más arriba."""
    columna = corr_sin_target[attribute]
    return columna[columna.index != attribute].sort_values(ascending=False, key=lambda x: abs(x)).index


# Las tres funciones de abajo estiman su regla de imputación (tabla cruzada, medianas por grupo,
# moda) SOLO con las filas de `indice_train`, y la aplican a las filas pendientes de ambos folds.
# Es el mismo contrato que un transformer de sklearn: fit(train), transform(todo).

def impute_from_discrete_match(attribute: str, match_attribute: str, pending_index: pd.Index) -> pd.Index:
    """Imputa vía la combinación más frecuente en una tabla cruzada con `match_attribute`
    (ej. si 2 dormitorios se combina más seguido con 2 baños, un baño nulo con 2 dormitorios se
    imputa como 2). La tabla cruzada se arma con train. Devuelve los índices sin resolver."""
    train_df = deptos_df.loc[indice_train]
    cross_table = pd.crosstab(train_df[match_attribute], train_df[attribute])
    if cross_table.empty:
        return pending_index
    modes = cross_table.idxmax(axis=1)
    match_values = deptos_df.loc[pending_index, match_attribute]
    resolvable = match_values[match_values.isin(modes.index)]
    deptos_df.loc[resolvable.index, attribute] = resolvable.map(modes)
    return pending_index.difference(resolvable.index)


def impute_from_continuous_match(attribute: str, match_attribute: str, pending_index: pd.Index) -> pd.Index:
    """Agrupa los valores discretos por la mediana de `match_attribute` (que sí es continuo) y
    asigna el grupo cuya mediana esté más cerca. Las medianas por grupo se calculan con train.
    Devuelve los índices que siguen sin resolver."""
    grouped_medians = deptos_df.loc[indice_train].groupby(attribute)[match_attribute].median().dropna()
    if grouped_medians.empty:
        return pending_index
    match_values = deptos_df.loc[pending_index, match_attribute].dropna()
    nearest = match_values.apply(lambda v: (grouped_medians - v).abs().idxmin())
    deptos_df.loc[nearest.index, attribute] = nearest
    return pending_index.difference(nearest.index)


def allocate_values():
    """Para cada atributo discreto con nulos, prueba sus atributos mejor correlacionados en
    orden (tabla cruzada si el match es discreto, agrupación por mediana si es continuo). Si
    ningún match resuelve una fila, se imputa la moda del atributo."""
    for attribute in discrete_values:
        pending_index = deptos_df[deptos_df[attribute].isna()].index
        if pending_index.empty:
            continue
        for match_attribute in attribute_correlations(attribute):
            if pending_index.empty:
                break
            if match_attribute in discrete_values:
                pending_index = impute_from_discrete_match(attribute, match_attribute, pending_index)
            else:
                pending_index = impute_from_continuous_match(attribute, match_attribute, pending_index)
        if not pending_index.empty:
            mode = deptos_df.loc[indice_train, attribute].mode()
            if not mode.empty:
                deptos_df.loc[pending_index, attribute] = mode.iloc[0]
        print(f'{attribute}: {deptos_df[attribute].isna().sum()} nulos restantes')


allocate_values()

# Imputación en variables continuas:
# Superficie total y Superficie útil ya quedaron sin nulos más arriba (fill cruzado entre ambas +
# descarte de las filas que no traían ninguna de las dos). Verificando qué float64 sigue con nulos
# a esta altura, el único caso real es "Unidades totales" (99.8% nulo, 12 de 5.603 filas).
#
# No es una columna "difícil de imputar", es una columna que no aplica a casi ninguna fila: esas 12
# filas son publicaciones de PROYECTOS inmobiliarios completos (mismas filas con 'Tipo de casa' y
# 'En condominio cerrado' nulos, ficha de página distinta a una casa individual -- ver el fix del
# selector de dirección), no casas individuales. "Unidades totales" describe al proyecto, no a una
# casa, así que no existe un valor real que imputarle a las 5.584 casas restantes: ni 0 ni la
# mediana serían ciertos, el atributo simplemente no les corresponde. Imputarlo sería inventar un
# número para reducir el conteo de nulos, no describir mejor los datos. Se descarta la columna.

unidades_totales_pct_null = deptos_df['Unidades totales'].isna().mean() * 100
print(f'"Unidades totales": {unidades_totales_pct_null:.1f}% nulo (solo aplica a fichas de proyecto, '
      f'no a casas individuales). No es imputable de forma confiable: se descarta la columna.')
deptos_df = deptos_df.drop(columns=['Unidades totales'])

# Resto de los atributos de texto que seguían con nulos:
# - dirección: 100% nula (el selector del scraper no la puebla para casas -- ya corregido en el
#   spider, pero este JSON es de antes del fix). No aporta nada al dataset: se descarta.
# - Gastos comunes: llega como texto ("290.000 CLP", "0 CLP"), se parsea con la misma
#   parse_measurement() del resto de las columnas de medición. El nulo se trata como "sin gasto
#   común informado", es decir 0 -- razonable en casas (a diferencia de un departamento, la mayoría
#   no paga gasto común de edificio).
# - Tipo de casa: nulo en las mismas 12 fichas de proyecto sin 'Unidades totales'; siguen siendo
#   casas, se etiquetan como 'Casa' (la categoría más común y no hay una mejor para esas fichas).
# - descripcion: nula quiere decir que el vendedor no escribió texto libre, no que no haya nada
#   que decir de la casa -- se rellena con 'titulo' (siempre presente), que es la mejor
#   aproximación textual disponible.

deptos_df = deptos_df.drop(columns=['dirección'])
deptos_df['Gastos comunes'] = deptos_df['Gastos comunes'].apply(parse_measurement).fillna(0)
deptos_df['Tipo de casa'] = deptos_df['Tipo de casa'].fillna('Casa')
deptos_df['descripcion'] = deptos_df['descripcion'].fillna(deptos_df['titulo'])

# Orientación: casas en la misma cuadra (misma calle, lote vecino) casi siempre comparten
# orientación -- se prioriza imputar desde la casa conocida más cercana en línea recta, y solo si
# esa distancia real es chica (calle/cuadra, no "mismo barrio en general"). Para eso hace falta la
# distancia real en metros (a diferencia de comuna/barrio, donde solo importaba el orden del
# vecino más cercano), así que aquí sí se usa haversine en vez de distancia euclidiana en grados.

ORIENTACION_VECINO_MAX_M = 150  # ponytail: umbral fijo (~una cuadra corta); ajustar si el sector tiene lotes más grandes


def haversine_m(lat1, lon1, lat2, lon2):
    """Distancia en línea recta (metros) entre dos puntos lat/lon."""
    radio_tierra_m = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    return 2 * radio_tierra_m * np.arcsin(np.sqrt(a))


def impute_orientacion_by_nearest_neighbor(df: pd.DataFrame, column: str, max_distance_m: float) -> None:
    """Imputa `column` con el valor de la casa conocida más cercana, solo si esa casa está a menos
    de `max_distance_m`. Deja sin imputar (NaN) los casos sin un vecino lo bastante cerca, para que
    el fallback de moda por barrio los resuelva."""
    has_coords = df['latitud'].notna() & df['longitud'].notna()
    known = df.loc[df.index.isin(indice_train) & df[column].notna() & has_coords]
    missing = df.loc[df[column].isna() & has_coords]
    if known.empty or missing.empty:
        return
    tree = cKDTree(known[['latitud', 'longitud']].to_numpy())
    _, nearest_idx = tree.query(missing[['latitud', 'longitud']].to_numpy())
    nearest = known.iloc[nearest_idx]
    distance_m = haversine_m(missing['latitud'].to_numpy(), missing['longitud'].to_numpy(),
                              nearest['latitud'].to_numpy(), nearest['longitud'].to_numpy())
    close_enough = distance_m <= max_distance_m
    df.loc[missing.index[close_enough], column] = nearest[column].to_numpy()[close_enough]


impute_orientacion_by_nearest_neighbor(deptos_df, 'Orientación', ORIENTACION_VECINO_MAX_M)
print(f'Orientación: {deptos_df["Orientación"].isna().sum()} nulos restantes tras imputar por vecino cercano.')

# Orientación (fallback): la moda cambia bastante de un barrio a otro (la orientación "buena"
# depende de cómo está trazada la calle), así que para lo que el vecino cercano no resolvió,
# imputar con la moda del barrio es más certero que la moda global. Los 41 barrios del dataset
# tienen al menos una Orientación conocida, así que esto resuelve casi todo; las filas que además
# no tienen barrio caen a la moda global como último recurso.

moda_por_barrio = (deptos_df.loc[indice_train]
                   .groupby('barrio')['Orientación']
                   .agg(lambda s: s.mode().iat[0] if not s.mode().empty else np.nan))
orientacion_na = deptos_df['Orientación'].isna()
deptos_df.loc[orientacion_na, 'Orientación'] = deptos_df.loc[orientacion_na, 'barrio'].map(moda_por_barrio)

orientacion_na = deptos_df['Orientación'].isna()
if orientacion_na.any():
    moda_global = deptos_df.loc[indice_train, 'Orientación'].mode().iat[0]
    print(f'{orientacion_na.sum()} casas sin barrio ni Orientación: se imputan con la moda global ({moda_global}).')
    deptos_df.loc[orientacion_na, 'Orientación'] = moda_global

print(f'Orientación: {deptos_df["Orientación"].isna().sum()} nulos restantes')

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

columnas_binarias = [c for c in deptos_df.columns if deptos_df[c].dtype == 'int64']

columnas_entrenamiento = [
    'latitud', 'longitud', 'descripcion',
    'Superficie útil', 'Superficie total',
    'Dormitorios', 'Baños', 'Estacionamientos', 'Bodegas',
    'Antigüedad', 'Cantidad de pisos',
    'Orientación', 'Gastos comunes',
    'comuna', 'barrio', 'Tipo de casa',
] + columnas_binarias

print(f'Columnas de entrenamiento: {len(columnas_entrenamiento)}')

X = deptos_df[columnas_entrenamiento].copy()
y = deptos_df['clp'].copy()

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

# 'Tipo de casa': Tríplex (40 casas) y Cabaña (1 casa) son demasiado raras para su propia columna
# one-hot -- la de Cabaña sería un identificador de fila disfrazado de feature (ver PASO 4). Se
# agrupan en 'Otro' antes de codificar. Es una renombrada fija de categorías (no aprende nada del
# target ni de la distribución de train), así que se aplica igual en train y test sin riesgo de
# fuga -- a diferencia del Target Encoding de 'barrio' más abajo, esto no necesita ir después del
# split por seguridad, pero se deja acá junto al resto de la codificación para que quede junto.

tipos_raros = {'Tríplex', 'Cabaña'}
X_train['Tipo de casa'] = X_train['Tipo de casa'].replace(tipos_raros, 'Otro')
X_test['Tipo de casa'] = X_test['Tipo de casa'].replace(tipos_raros, 'Otro')

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

encoder_categoricas = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first', handle_unknown='infrequent_if_exist', sparse_output=False),
         columnas_onehot),
        ('target', TargetEncoder(target_type='continuous', random_state=42), columnas_target_encoding),
    ],
    remainder='passthrough',
)

# El resto de las columnas (superficies, conteos, binarios, 'descripcion' como texto crudo)
# quedan sin tocar en esta etapa -- escalado y vectorización de texto son pasos aparte del PASO 4
# que todavía no se pidieron, así que X_*_codificado no es todavía el input final del modelo.

X_train_codificado = pd.DataFrame(
    encoder_categoricas.fit_transform(X_train, y_train),
    columns=encoder_categoricas.get_feature_names_out(),
    index=X_train.index,
)
X_test_codificado = pd.DataFrame(
    encoder_categoricas.transform(X_test),
    columns=encoder_categoricas.get_feature_names_out(),
    index=X_test.index,
)

# El array que devuelve el ColumnTransformer mezcla números con el texto crudo de
# 'remainder__descripcion' en un solo bloque, así que numpy lo tipa todo como dtype=object
# (incluido 'target__barrio', que es un float pero queda encapsulado en un objeto). Se restaura
# el dtype numérico columna por columna, dejando 'descripcion' como texto -- si no, cualquier
# describe()/cálculo posterior sobre las columnas numéricas se rompe silenciosamente.

columnas_numericas_codificadas = [c for c in X_train_codificado.columns if c != 'remainder__descripcion']
X_train_codificado[columnas_numericas_codificadas] = X_train_codificado[columnas_numericas_codificadas].apply(pd.to_numeric)
X_test_codificado[columnas_numericas_codificadas] = X_test_codificado[columnas_numericas_codificadas].apply(pd.to_numeric)

print(f'Columnas tras codificar categóricas: {X_train_codificado.shape[1]} (antes: {X_train.shape[1]})')

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

X_train_final = X_train_codificado[columnas_sin_texto].copy()
X_test_final = X_test_codificado[columnas_sin_texto].copy()

escalador_log = StandardScaler()
X_train_final[columnas_log_scale] = escalador_log.fit_transform(np.log1p(X_train_final[columnas_log_scale]))
X_test_final[columnas_log_scale] = escalador_log.transform(np.log1p(X_test_final[columnas_log_scale]))

escalador_conteos = StandardScaler()
X_train_final[columnas_solo_scale] = escalador_conteos.fit_transform(X_train_final[columnas_solo_scale])
X_test_final[columnas_solo_scale] = escalador_conteos.transform(X_test_final[columnas_solo_scale])

X_train_final[columna_antiguedad] = X_train_final[columna_antiguedad].clip(0, 120)
X_test_final[columna_antiguedad] = X_test_final[columna_antiguedad].clip(0, 120)
escalador_antiguedad = StandardScaler()
X_train_final[[columna_antiguedad]] = escalador_antiguedad.fit_transform(X_train_final[[columna_antiguedad]])
X_test_final[[columna_antiguedad]] = escalador_antiguedad.transform(X_test_final[[columna_antiguedad]])

print(f'X_train_final: {X_train_final.shape}. X_test_final: {X_test_final.shape}.')
print(X_train_final[columnas_log_scale + columnas_solo_scale + [columna_antiguedad]].describe())

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


def agregar_distancias(df: pd.DataFrame, comuna_por_fila: pd.Series, centroide_comuna: pd.DataFrame) -> None:
    for nombre, (lat_punto, lon_punto) in PUNTOS_DE_INTERES.items():
        df[f'distancia_{nombre}_m'] = haversine_m(df['remainder__latitud'], df['remainder__longitud'],
                                                    lat_punto, lon_punto)
    lat_centroide = comuna_por_fila.map(centroide_comuna['remainder__latitud'])
    lon_centroide = comuna_por_fila.map(centroide_comuna['remainder__longitud'])
    df['distancia_centroide_comuna_m'] = haversine_m(df['remainder__latitud'], df['remainder__longitud'],
                                                       lat_centroide, lon_centroide)


centroide_comuna = X_train_final.groupby(X_train['comuna'])[['remainder__latitud', 'remainder__longitud']].mean()
agregar_distancias(X_train_final, X_train['comuna'], centroide_comuna)
agregar_distancias(X_test_final, X_test['comuna'], centroide_comuna)

columnas_distancia = [f'distancia_{nombre}_m' for nombre in PUNTOS_DE_INTERES] + ['distancia_centroide_comuna_m']
print(X_train_final[columnas_distancia].describe())
print('skew:', X_train_final[columnas_distancia].skew().to_dict())

# Con las distancias calculadas, 'latitud'/'longitud' crudas ya cumplieron su propósito y se
# eliminan del set de features -- dejarlas junto con las distancias sería redundante (las
# distancias ya son una transformación de esas mismas dos columnas) y volvería a exponer al
# modelo lineal a la coordenada cruda que se quería evitar.

X_train_final = X_train_final.drop(columns=['remainder__latitud', 'remainder__longitud'])
X_test_final = X_test_final.drop(columns=['remainder__latitud', 'remainder__longitud'])

# log1p + estandarizar, mismo criterio que las superficies en PASO 4d: la distancia responde al
# precio de forma multiplicativa en un modelo hedónico (el mismo argumento que ya se usa para
# Superficie total/útil, ver cabecera del archivo), y el skew medido arriba lo confirma -- 0,55 a
# 0,88 en las tres distancias a puntos fijos, pero 4,06 en la del centroide de la propia comuna
# (por los refugios de Farellones/La Parva, legítimamente lejos de su centroide). Se aplica log1p
# a las 4 por igual para no tratar distinto a features de la misma familia: no perjudica a las
# tres que ya estaban casi simétricas, y corrige la que sí lo necesitaba.

X_train_final[columnas_distancia] = np.log1p(X_train_final[columnas_distancia])
X_test_final[columnas_distancia] = np.log1p(X_test_final[columnas_distancia])

escalador_distancias = StandardScaler()
X_train_final[columnas_distancia] = escalador_distancias.fit_transform(X_train_final[columnas_distancia])
X_test_final[columnas_distancia] = escalador_distancias.transform(X_test_final[columnas_distancia])

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

X_train_final['ratio_construido'] = X_train['Superficie útil'] / X_train['Superficie total']
X_test_final['ratio_construido'] = X_test['Superficie útil'] / X_test['Superficie total']

X_train_final['m2_por_dormitorio'] = X_train['Superficie útil'] / X_train['Dormitorios']
X_test_final['m2_por_dormitorio'] = X_test['Superficie útil'] / X_test['Dormitorios']

X_train_final['banos_por_dormitorio'] = X_train['Baños'] / X_train['Dormitorios']
X_test_final['banos_por_dormitorio'] = X_test['Baños'] / X_test['Dormitorios']

columnas_ratios = ['ratio_construido', 'm2_por_dormitorio', 'banos_por_dormitorio']
print(X_train_final[columnas_ratios].describe())
print('skew:', X_train_final[columnas_ratios].skew().to_dict())

# Escalado con el mismo criterio de forma que PASO 4d, medido sobre train (ver skew impreso
# arriba): ratio_construido (skew 0,75) y baños_por_dormitorio (skew 0,73) son aproximadamente
# simétricos -- StandardScaler directo, igual que los conteos. m2_por_dormitorio (skew 11,7)
# hereda la cola pesada de 'Superficie útil' (mismo numerador) y necesita log1p antes de
# estandarizar, igual que las superficies.

columnas_ratio_simetricos = ['ratio_construido', 'banos_por_dormitorio']
escalador_ratios = StandardScaler()
X_train_final[columnas_ratio_simetricos] = escalador_ratios.fit_transform(X_train_final[columnas_ratio_simetricos])
X_test_final[columnas_ratio_simetricos] = escalador_ratios.transform(X_test_final[columnas_ratio_simetricos])

escalador_m2_dormitorio = StandardScaler()
X_train_final[['m2_por_dormitorio']] = escalador_m2_dormitorio.fit_transform(
    np.log1p(X_train_final[['m2_por_dormitorio']]))
X_test_final[['m2_por_dormitorio']] = escalador_m2_dormitorio.transform(
    np.log1p(X_test_final[['m2_por_dormitorio']]))

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

X_train_final = X_train_final.drop(columns=['remainder__Superficie útil'])
X_test_final = X_test_final.drop(columns=['remainder__Superficie útil'])

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
# 'Gastos comunes' ya entra al modelo como valor continuo (log1p + StandardScaler, ver PASO 4d),
# pero la columna es medio indicador y medio monto: 83,9% de las casas en train vale 0 (sin gasto
# común, típicamente casas fuera de condominio) y el resto trae un monto real. Se agrega
# 'tiene_gastos_comunes' = (Gastos comunes > 0) para que el modelo lineal separe el efecto de
# "está en condominio con administración" del monto en sí, en vez de que ambas señales queden
# mezcladas en un solo coeficiente sobre log1p(Gastos comunes).
#
# Se calcula sobre el valor CRUDO de X_train/X_test (antes del log1p+scale que PASO 4d ya aplicó
# in-place a 'remainder__Gastos comunes' en X_train_final), mismo criterio que los ratios de
# PASO 4f. Es binaria: no se escala (ver PASO 4d).

X_train_final['tiene_gastos_comunes'] = (X_train['Gastos comunes'] > 0).astype(int)
X_test_final['tiene_gastos_comunes'] = (X_test['Gastos comunes'] > 0).astype(int)

print('tiene_gastos_comunes en train:',
      X_train_final['tiene_gastos_comunes'].value_counts(normalize=True).round(3).to_dict())
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


lat0, lon0 = X_train['latitud'].mean(), X_train['longitud'].mean()
coords_train_m = proyectar_metros(X_train['latitud'], X_train['longitud'], lat0, lon0)
coords_test_m = proyectar_metros(X_test['latitud'], X_test['longitud'], lat0, lon0)

arbol_densidad = cKDTree(coords_train_m)

# Cada casa de train está en el propio árbol -- se resta 1 para no contarse a sí misma. Test no
# tiene ese problema: el árbol solo contiene coordenadas de train.
densidad_train = arbol_densidad.query_ball_point(coords_train_m, r=RADIO_DENSIDAD_M, return_length=True) - 1
densidad_test = arbol_densidad.query_ball_point(coords_test_m, r=RADIO_DENSIDAD_M, return_length=True)

X_train_final['densidad_oferta_500m'] = densidad_train
X_test_final['densidad_oferta_500m'] = densidad_test

print(X_train_final['densidad_oferta_500m'].describe())
print('skew:', X_train_final['densidad_oferta_500m'].skew())

# Escalado con el mismo criterio de forma que el resto de PASO 4: skew medido 0,43, por debajo
# del umbral de 1,2 que ya usa PASO 4d para los conteos acotados (Dormitorios/Baños) -- a
# diferencia de las superficies, no tiene cola pesada, así que StandardScaler directo, sin log1p.

escalador_densidad = StandardScaler()
X_train_final[['densidad_oferta_500m']] = escalador_densidad.fit_transform(X_train_final[['densidad_oferta_500m']])
X_test_final[['densidad_oferta_500m']] = escalador_densidad.transform(X_test_final[['densidad_oferta_500m']])

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

