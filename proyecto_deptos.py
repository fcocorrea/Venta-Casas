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
#    barrio, match por correlación en atributos discretos, regresión logística en binarios) deben
#    ajustarse solo con el fold de entrenamiento una vez exista el split -- no con el dataframe
#    completo, como ocurre hoy -- y ninguna debe usar precio/clp/precio unitario como predictor,
#    para no filtrar el target hacia adentro de una feature.
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

import pandas as pd
import numpy as np

GRAFICOS_DIR = 'gráficos'
os.makedirs(GRAFICOS_DIR, exist_ok=True)

deptos_df = pd.read_json('deptos.json')
print('Cantidad de observaciones: {}.\nCantidad de atributos: {}.'.format(*deptos_df.shape))
print('Columnas:', ', '.join(deptos_df.columns.tolist()))

# Limpieza
#
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


def parse_measurement(value):
    """Convierte 'N', 'N a M' (rango, se usa el mínimo) o 'N.NNN unidad' (separador de miles) a
    float. Devuelve NaN si no hay nada parseable (ej. "Precio a consultar")."""
    if pd.isna(value):
        return np.nan
    tokens = str(value).split()
    if not tokens:
        return np.nan
    token = tokens[0].replace('.', '').replace(',', '.')
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
numeric_attributes = [c for c in measurement_columns if deptos_df[c].dtype == object]
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

# Cuando la superficie útil es mayor que la total (no tiene sentido), se usa la útil como total.

deptos_df['Superficie total'] = np.where(deptos_df['Superficie útil'] > deptos_df['Superficie total'],
                                          deptos_df['Superficie útil'], deptos_df['Superficie total'])

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


# Valores extremos bajos: un salto proporcional grande (>50%) yendo desde el límite del rango
# intercuartílico hacia abajo suele indicar error de tipeo en precio o superficie.

lower_cut = cut_after_relative_jump(lower_values.sort_values(ascending=False), threshold=0.5)
if lower_cut is not None:
    delete_lower_values = deptos_df[deptos_df['precio unitario'] <= lower_cut]
    print(f'Eliminamos {len(delete_lower_values)} valores extremos bajos (<= {lower_cut}).')
    deptos_df = deptos_df.drop(delete_lower_values.index)

# Valores extremos altos: aquí suele haber casas amobladas (más caras), publicaciones erróneas o
# de lujo atípico. Igual que abajo, cortamos donde aparece un salto (>100%) en vez de un índice fijo.

upper_cut = cut_after_relative_jump(higher_values.sort_values(), threshold=1.0)
if upper_cut is not None:
    delete_upper_values = deptos_df[deptos_df['precio unitario'] >= upper_cut]
    print(f'Eliminamos {len(delete_upper_values)} valores extremos altos (>= {upper_cut}).')
    deptos_df = deptos_df.drop(delete_upper_values.index)

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

# Imputación de valores nulos en comuna:
# - comuna nula, barrio no: si el barrio identifica una única comuna, la usamos.
# - si un barrio aparece asociado a más de una comuna, se usa la más frecuente; si no está
#   asociado a ninguna, en realidad "barrio" contenía el nombre de la comuna.

barrios = deptos_df.loc[deptos_df.comuna.isna(), 'barrio'].dropna().unique().tolist()
for barrio in barrios:
    comuna_del_barrio = deptos_df.loc[deptos_df.barrio == barrio, 'comuna'].dropna().unique().tolist()
    if not comuna_del_barrio:
        deptos_df.loc[deptos_df.barrio == barrio, 'comuna'] = barrio
        deptos_df.loc[deptos_df.barrio == barrio, 'barrio'] = np.nan
    else:
        comuna = deptos_df.loc[deptos_df.comuna.isin(comuna_del_barrio), 'comuna'].value_counts().idxmax()
        deptos_df.loc[deptos_df.barrio == barrio, 'comuna'] = comuna

print(f'Ahora hay {deptos_df.comuna.isna().sum()} comunas con valores nulos')
comunas = [i for i in deptos_df.comuna.unique() if isinstance(i, str)]
print('Las comunas únicas que quedan son:', ', '.join(comunas))

# Signo de latitud/longitud: Santiago de Chile cae siempre en latitud y longitud negativas
# (hemisferio sur/oeste). Alguna fila llega con el signo invertido (error del scraper, no un dato
# realmente distinto -- la magnitud sigue siendo la correcta para la comuna), así que se corrige
# el signo en vez de anular el valor.

lat_lon_signo_invertido = (deptos_df['latitud'] > 0) | (deptos_df['longitud'] > 0)
if lat_lon_signo_invertido.any():
    print(f'Corregimos el signo de latitud/longitud en {lat_lon_signo_invertido.sum()} filas.')
    deptos_df.loc[lat_lon_signo_invertido, 'latitud'] = -deptos_df.loc[lat_lon_signo_invertido, 'latitud'].abs()
    deptos_df.loc[lat_lon_signo_invertido, 'longitud'] = -deptos_df.loc[lat_lon_signo_invertido, 'longitud'].abs()

# Para los registros donde comuna y/o barrio siguen nulos, se imputan con el valor de la casa
# conocida más cercana en latitud/longitud -- más preciso que fuzzy-matching de texto sobre
# 'dirección' (que además viene vacía en el crawl actual de casas) y aprovecha que casi todas las
# filas sí tienen coordenadas (ver deptos.json: solo 1 de 5603 sin latitud/longitud).

from scipy.spatial import cKDTree


def impute_nearest_by_location(df: pd.DataFrame, column: str) -> None:
    has_coords = df['latitud'].notna() & df['longitud'].notna()
    known = df.loc[df[column].notna() & has_coords]
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
# "Cantidad de pisos" puede tener el mismo tipo de error de tipeo, ej. 2013 en vez de 2).

bad_ranges = [c for c in ['Baños', 'Estacionamientos', 'Bodegas', 'Cantidad de pisos'] if c in deptos_df.columns]

for column in bad_ranges:
    top_20 = deptos_df[column].dropna().nlargest(20).sort_values()
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
    if deptos_df[column].dtype == object or column in coordenadas:
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
# se usa regresión logística. Se mapea "Sí"/"No" a 1/0 en toda la columna de una sola vez (no solo
# en las filas usadas para entrenar) para que la columna quede con un tipo consistente al final --
# antes, las filas imputadas quedaban en 0/1 (int) mientras el resto seguía en 'Sí'/'No' (string).

from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')


def predict_missing_values(X: pd.Series, y: pd.Series) -> pd.Series:
    """Imputa los valores nulos de una variable binaria (0/1) vía regresión logística sobre X.
    Si la columna solo tiene una clase presente (ej. un amenity que en el scrape nunca aparece
    marcado 'No'), no hay nada que un clasificador binario pueda aprender: se rellena con esa
    única clase."""
    known = y.notna()
    na_index = y.loc[y.isna()].index
    if na_index.empty or known.sum() == 0:
        return y
    if y.loc[known].nunique() < 2:
        y.loc[na_index] = y.loc[known].iloc[0]
        return y
    lr = LogisticRegression()
    lr.fit(X.loc[known].values.reshape(-1, 1), y.loc[known].values)
    y.loc[na_index] = lr.predict(X.loc[na_index].values.reshape(-1, 1))
    return y


X = deptos_df['precio unitario']
for binary_feature in binary_features:
    y = deptos_df[binary_feature].map({'Sí': 1, 'No': 0})
    deptos_df[binary_feature] = predict_missing_values(X, y)


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


# Imputación en variables discretas:
# variables numéricas que toman un número finito de valores, propias de una casa (a diferencia del
# departamento no hay "número de piso de la unidad" ni "departamentos por piso").

discrete_values = [c for c in ['Dormitorios', 'Baños', 'Estacionamientos', 'Bodegas',
                                'Cantidad de pisos', 'Antigüedad'] if c in deptos_df.columns]
na_columns = columns_with_missing_values()
print(na_columns[na_columns.index.isin(discrete_values)])


def attribute_correlations(attribute) -> pd.Index:
    """Atributos ordenados de mayor a menor correlación (absoluta) respecto a `attribute`."""
    return corr.loc[corr.index != attribute, attribute].sort_values(ascending=False, key=lambda x: abs(x)).index


def impute_from_discrete_match(attribute: str, match_attribute: str, pending_index: pd.Index) -> pd.Index:
    """Imputa vía la combinación más frecuente en una tabla cruzada con `match_attribute`
    (ej. si 2 dormitorios se combina más seguido con 2 baños, un baño nulo con 2 dormitorios se
    imputa como 2). Devuelve los índices que siguen sin resolver."""
    cross_table = pd.crosstab(deptos_df[match_attribute], deptos_df[attribute])
    modes = cross_table.idxmax(axis=1)
    match_values = deptos_df.loc[pending_index, match_attribute]
    resolvable = match_values[match_values.isin(modes.index)]
    deptos_df.loc[resolvable.index, attribute] = resolvable.map(modes)
    return pending_index.difference(resolvable.index)


def impute_from_continuous_match(attribute: str, match_attribute: str, pending_index: pd.Index) -> pd.Index:
    """Agrupa los valores discretos por la mediana de `match_attribute` (que sí es continuo) y
    asigna el grupo cuya mediana esté más cerca. Devuelve los índices que siguen sin resolver."""
    grouped_medians = deptos_df.groupby(attribute)[match_attribute].median()
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
            mode = deptos_df[attribute].mode()
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
    known = df.loc[df[column].notna() & has_coords]
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

moda_por_barrio = deptos_df.groupby('barrio')['Orientación'].agg(lambda s: s.mode().iat[0] if not s.mode().empty else np.nan)
orientacion_na = deptos_df['Orientación'].isna()
deptos_df.loc[orientacion_na, 'Orientación'] = deptos_df.loc[orientacion_na, 'barrio'].map(moda_por_barrio)

orientacion_na = deptos_df['Orientación'].isna()
if orientacion_na.any():
    moda_global = deptos_df['Orientación'].mode().iat[0]
    print(f'{orientacion_na.sum()} casas sin barrio ni Orientación: se imputan con la moda global ({moda_global}).')
    deptos_df.loc[orientacion_na, 'Orientación'] = moda_global

print(f'Orientación: {deptos_df["Orientación"].isna().sum()} nulos restantes')

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

# Casas candidatas para "el hogar de mis sueños":
# el precio de cada casa viene en UF o en CLP según 'UM'. Para poder filtrar por un mismo umbral en
# UF, se pasan las publicaciones en CLP a UF usando la tasa que el propio sitio usó para calcular
# el equivalente en pesos de las publicaciones en UF (mediana de clp/precio en esas filas, ~40.832
# CLP/UF de forma consistente -- todo el scrape es del mismo día, así que comparten la misma UF).

uf_rate = (deptos_df.loc[deptos_df['UM'] == 'UF', 'clp'] / deptos_df.loc[deptos_df['UM'] == 'UF', 'precio']).median()
deptos_df['precio UF'] = np.where(deptos_df['UM'] == 'UF', deptos_df['precio'], deptos_df['clp'] / uf_rate)

candidatas = deptos_df[
    (deptos_df['precio UF'] < 15000) &
    (deptos_df['Dormitorios'] >= 3) &
    (deptos_df['Baños'] >= 2)
].copy()
candidatas['negociacion'] = np.where(candidatas['precio UF'] > 13500, 'Negociar', 'Precio Ok')

print(f'Casas candidatas: {len(candidatas)} de {len(deptos_df)}')
print(candidatas['negociacion'].value_counts())
candidatas.sort_values('precio unitario').to_excel('casas_candidatas.xlsx', index=False)


"""
=======================================================================================
PASO 4 -- ESTRATEGIA DE PREPROCESAMIENTO PARA EL MODELO (diseño, aún no implementado)
=======================================================================================

Punto de partida medido sobre deptos_limpios.xlsx (5.597 filas x 74 columnas), no supuesto:
sin nulos, pero con tres problemas que condicionan todo lo que sigue.

  (a) 37 de las 74 columnas son CONSTANTES (varianza cero). Son los amenities binarios que
      el scrape solo trae marcados 'Sí' (Piscina no, pero sí Jacuzzi, Sauna, Ascensor,
      Amoblado, Cisterna, ...). No es que la casa no los tenga: el vendedor solo marca lo
      que hay, nunca lo que falta, así que 'nulo' significaba "no informado" y la imputación
      por regresión logística de más arriba los rellenó todos con la única clase presente.
      Quedan 15 binarias con varianza real (Piscina, Terraza, Comedor, Living, Patio,
      Dormitorio en suite, Jardín, Cocina, Aire acondicionado, Calefacción, TV por cable,
      Gas natural, Con conexión para lavarropas, Conserjería, Acceso controlado).
  (b) Asimetría extrema en las superficies: skew de 69 en 'Superficie útil' (máx 250.000 m²)
      y 60 en 'Superficie total' (máx 400.000 m²), más 'Antigüedad' con skew 15 (máx 939
      años) y 'Cantidad de pisos' con 12 (máx 25). El recorte de outliers de más arriba se
      aplicó sobre 'precio unitario', no sobre estas columnas, así que siguen sucias.
  (c) Fuga de target: 'precio', 'clp', 'precio UF', 'precio unitario' y 'UM' son todas el
      target o transformaciones suyas. 'UM' además codifica el tramo de precio (solo las
      publicaciones caras se listan en UF), así que es un proxy y no una feature.

ORDEN DE OPERACIONES (no negociable)
------------------------------------
Primero el split, después todo lo demás. Cada paso que "aprende" algo de los datos (media
y desvío del escalado, categorías del encoder, vocabulario e IDF del texto, medianas de
imputación, umbral de colinealidad) se ajusta SOLO con el fold de entrenamiento y se aplica
al de test. Hoy el script hace lo contrario -- imputa y recorta sobre el dataframe completo
-- así que estos pasos van dentro de un Pipeline/ColumnTransformer de sklearn, no sueltos.
Split estratificado por 'comuna': los tres niveles de precio son muy distintos y hay que
asegurar que ambos folds los representen.

1. ESCALADO DE FEATURES (normalización vs. estandarización)
-----------------------------------------------------------
La decisión no es una sola para todo el dataset, va por forma de la distribución:

  - Superficies ('Superficie útil', 'Superficie total') y 'Gastos comunes' (skew 7,4):
    log1p PRIMERO, StandardScaler DESPUÉS. Estandarizar una distribución con skew 69 no
    la arregla -- deja la masa de los datos aplastada en un rango mínimo y el outlier a 40
    desvíos igual de dominante. El log es lo que corrige la forma; el escalado solo centra.
    Además el log es lo correcto por teoría: en un modelo hedónico el precio responde a la
    superficie de forma multiplicativa, y log(precio) ~ log(superficie) da directamente la
    elasticidad (ver cabecera del archivo).
  - Conteos discretos y acotados ('Dormitorios' 1-14, 'Baños' 1-18, 'Estacionamientos',
    'Bodegas', 'Cantidad de pisos'): StandardScaler a secas. Ya son de rango chico y
    aproximadamente simétricos (skew < 1,2 en dormitorios y baños); el log no aporta.
  - 'Antigüedad': acotar primero a un rango plausible (0-120 años; los 939 son basura que
    sobrevivió a la limpieza) y luego estandarizar.
  - 'latitud'/'longitud': NO escalar por separado ni tratarlas como dos features numéricas
    independientes -- ver punto 4, se convierten en features de distancia.
  - Binarias 0/1: no se escalan, ya están en [0,1].

  Normalización (MinMax) queda descartada como default: es sensible al máximo, y con
  máximos de 400.000 m² comprimiría el 99% de los datos a un rango casi nulo. Solo tendría
  sentido si se cambia a una red neuronal, y aun así después del log. Si tras el log
  quedaran colas pesadas, la alternativa robusta es RobustScaler (usa mediana e IQR, que es
  el mismo criterio de percentil_limits() que ya usa este script).

  Nota: para el modelo de árboles con boosting el escalado es indiferente (los splits son
  invariantes a transformaciones monótonas). Se mantiene igual para que ambos modelos
  compartan el mismo ColumnTransformer, y porque el log sí cambia lo que aprende el lineal.

2. CODIFICACIÓN DE CATEGÓRICAS
-------------------------------
Medida la cardinalidad real, aquí no hay ninguna categórica verdaderamente alta:

  - Baja cardinalidad -> One-Hot con drop='first' (evita la trampa de la variable ficticia
    en el modelo lineal) y handle_unknown='infrequent_if_exist':
      * 'comuna' (3 niveles)
      * 'Tipo de casa' (5: Casa 4.902, Chalet 491, Dúplex 162, Tríplex 41, Cabaña 1)
        -> agrupar Tríplex y Cabaña en una categoría 'Otro'; con 1 sola cabaña, esa columna
        one-hot es un identificador de fila disfrazado de feature.
      * 'Orientación' (8 niveles). Es cíclica en teoría (N-NE-E-...), pero como categórica
        de 8 niveles con 5.597 filas no vale la pena el seno/coseno; one-hot y listo.
  - Cardinalidad media -> 'barrio' (41 niveles) es el caso a decidir. One-Hot lo deja en 41
    columnas para 5.597 filas, y 7 barrios tienen menos de 30 casas (Puente Nuevo tiene 1).
    Estrategia: Target Encoding sobre log(precio) con suavizado bayesiano hacia la media de
    la comuna, ajustado dentro de CV anidado (out-of-fold) para no filtrar el target. El
    barrio es la feature de ubicación más fuerte del dataset, y el target encoding conserva
    ese orden de precios en una sola columna en vez de 41 dispersas. Los barrios con <30
    casas colapsan casi por completo hacia la media de su comuna, que es exactamente el
    comportamiento deseado. Alternativa más simple si el encoding out-of-fold da problemas
    de fuga: One-Hot agrupando los 7 barrios raros en 'Otro' (35 columnas).
  - Ordinal Encoding: no se usa. Ninguna categórica tiene orden natural, y asignarle uno
    arbitrario (barrio 0..40) le inventaría al modelo lineal una relación monótona falsa.
    Los conteos que sí son ordinales ('Dormitorios', 'Baños') ya vienen como enteros.
  - 'UM': se descarta (fuga, ver punto (c) arriba).

3. VECTORIZACIÓN DE TEXTO
--------------------------
Hay dos campos de texto libre: 'titulo' (4.520 valores únicos) y 'descripcion' (5.486).
El texto es la única fuente de señal para atributos que la ficha estructurada no captura:
estado de conservación, remodelaciones, vista, calidad de terminaciones, urgencia de venta.

  - Faltantes: ya resuelto aguas arriba -- 'descripcion' nula se rellena con 'titulo'
    (siempre presente). Para el modelo se concatenan ambos campos en un solo documento
    ('titulo' + ' ' + 'descripcion'), lo que además hace que las filas imputadas no queden
    con el título duplicado pesando doble en el vector.
  - Baseline: TF-IDF con ngram_range=(1,2), min_df=5 (descarta typos y direcciones únicas),
    max_df=0.7 (descarta el boilerplate de la inmobiliaria, que se repite en miles de
    avisos), lista de stopwords en español, y normalización previa de acentos y minúsculas.
    Sobre eso, TruncatedSVD a ~50-100 componentes: la matriz TF-IDF dispersa se lleva mal
    con los modelos de árboles y con una matriz densa de features numéricas.
  - Alternativa a evaluar contra el baseline: embeddings de un modelo multilingüe
    (p. ej. sentence-transformers, paraphrase-multilingual-MiniLM) que capturan sinónimos
    que TF-IDF no ve ("impecable" ~ "excelente estado"). Con 5.597 documentos cortos y en
    español el costo es bajo. Se adopta solo si mejora el MdAPE de forma medible; si no,
    queda TF-IDF, que además es interpretable.
  - Riesgo a controlar: la descripción suele mencionar el precio o la superficie en texto
    ("vendo en 12.000 UF"). Eso es fuga directa del target. Antes de vectorizar hay que
    borrar del documento los patrones numéricos de precio (UF, CLP, $, millones) con una
    regex. Sin ese filtro el modelo "predice" leyendo la respuesta.

4. INGENIERÍA Y SELECCIÓN DE FEATURES
--------------------------------------
Features de dominio a construir:

  - ratio_construido = Superficie útil / Superficie total. Distingue la casa grande en lote
    chico de la casa chica en lote grande, que a igual superficie total valen muy distinto.
    Es la feature que el target 'precio unitario' original borraba por construcción.
  - superficie_terreno = Superficie total - Superficie útil (terreno no construido).
  - baños_por_dormitorio y m2_por_dormitorio: proxies de estándar/calidad de la casa,
    independientes del tamaño absoluto.
  - Distancias reales (haversine) a los polos de valor del sector -- Clínica Alemana,
    Estadio Español, Portal La Dehesa, los colegios del corredor Manquehue-La Dehesa --
    calculadas desde 'latitud'/'longitud'. Esto reemplaza y mejora el antiguo campo
    'cercania' 'Cerca'/'Lejos', que era una lista de barrios escrita a mano: una distancia
    continua en metros conserva el gradiente que la etiqueta binaria tiraba a la basura.
    Ahora es viable porque las coordenadas quedaron completas (99,98%) tras corregir el bug
    del filtro de negativos.
  - Alternativa/complemento a las distancias: distancia al centroide de la comuna, y
    densidad local de la oferta (número de casas en un radio de 500 m vía cKDTree), que
    aproxima "qué tan consolidado está el sector".
  - 'Gastos comunes' > 0 como binaria (indica condominio con administración) además del
    valor continuo: el 50% de las casas tiene 0, así que la columna es medio indicador y
    medio monto.

Selección y eliminación de redundancia:

  - Eliminar de entrada las 37 columnas constantes. No aportan información, inflan la
    matriz y ensucian cualquier ranking de importancia. Un VarianceThreshold(0) lo hace
    automáticamente y sigue funcionando si un crawl futuro les da varianza.
  - Eliminar identificadores: 'url' (5.597 valores únicos = una fila cada uno).
  - Eliminar el bloque de fuga: 'precio', 'clp', 'precio UF', 'precio unitario', 'UM'.
  - Colinealidad: 'Superficie útil' y 'Superficie total' están fuertemente correlacionadas
    entre sí y con 'Dormitorios'/'Baños' (el heatmap ya generado lo muestra). Para el modelo
    lineal, calcular VIF y descartar iterativamente lo que supere ~10, o directamente
    quedarse con log(Superficie total) + ratio_construido en vez de las dos superficies
    crudas (misma información, sin la correlación de 0,9). Para el modelo de árboles la
    colinealidad no rompe las predicciones, pero sí reparte la importancia entre features
    gemelas y hace ilegible el SHAP, así que conviene igual.
  - Selección final: importancia por permutación sobre el fold de validación (no la
    importancia por impureza de sklearn, que sobrevalora las features de alta cardinalidad
    como el target encoding de barrio). Descartar lo que tenga importancia no distinguible
    de cero y reentrenar, verificando que el MdAPE no empeore.
  - Feature de control obligatoria: comparar siempre contra el baseline de mediana de
    precio/m² por barrio. Si toda esta ingeniería no le gana a esa línea de una sola
    columna, el modelo no justifica su complejidad.
"""

