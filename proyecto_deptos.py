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
# con el precio real, para determinar si está sobre o bajo su precio de mercado. Se trata de un
# problema de regresión.
#
# Implementación, en tres grandes pasos:
# 1) Obtención de datos: se extraen vía web scraping con Scrapy (ver deptos_scraper/spiders/deptos.py;
#    se ejecuta con `scrapy crawl deptos -O deptos.json`, documentado en CLAUDE.md).
# 2) Limpieza de datos: homologar y transformar los datos, que no vienen expresados de forma consistente.
# 3) Exploración: entender los datos para adaptarlos a un modelo de regresión.
# 4) Modelos: probar varios modelos y quedarnos con el más adecuado.
# 5) Producción: ubicar las casas en un mapa, coloreadas de rojo (sobrevaloradas) a verde
#    (subvaloradas).

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

# Para los registros donde comuna y barrio siguen nulos, se imputa la comuna por dirección
# (fuzzy match de texto). Esto depende de que el scraper haya podido extraer 'dirección' -- si no
# hay ninguna dirección utilizable (como ocurre en el crawl actual de casas, donde ese selector no
# matchea nada en la ficha de la propiedad) no tiene sentido importar fuzzywuzzy ni intentarlo.

com_barr_null = deptos_df.loc[deptos_df.comuna.isna() & deptos_df.barrio.isna()]
if not com_barr_null.empty and com_barr_null['dirección'].notna().any():
    from fuzzywuzzy import process

    coincidences = {}  # {dirección: [comuna, puntaje de coincidencia, índice]}
    for comuna in comunas:
        address, score, idx = process.extract(comuna, com_barr_null['dirección'], limit=1)[0]
        if address not in coincidences or score > coincidences[address][1]:
            coincidences[address] = [comuna, score, idx]
    for address, features in coincidences.items():
        deptos_df.loc[features[2], 'comuna'] = features[0]
else:
    print(f'Sin direcciones utilizables para imputar comuna por dirección ({len(com_barr_null)} filas pendientes).')

print(f'Ahora hay {deptos_df.comuna.isna().sum()} comunas con valores nulos')

# Barrios:
# mismo problema que con las comunas. Un mismo barrio puede aparecer escrito de formas distintas
# (ej. "las condes" vs "Las Condes"), así que se capitalizan todos.

deptos_df.barrio = deptos_df.barrio.str.title()
print(f'Hay {deptos_df.barrio.isna().sum()} casas sin barrio.')

deptos_without_hood = deptos_df.loc[deptos_df.barrio.isna()]
if not deptos_without_hood.empty and deptos_without_hood['dirección'].notna().any():
    from fuzzywuzzy import process

    deptos_with_hood = deptos_df[~deptos_df.barrio.isna()]
    for idx, direccion in deptos_without_hood['dirección'].items():
        if pd.isna(direccion):
            continue
        _, _, match_idx = process.extract(direccion, deptos_with_hood['dirección'], limit=1)[0]
        deptos_df.loc[idx, 'barrio'] = deptos_with_hood.loc[match_idx, 'barrio']
else:
    print('Sin direcciones utilizables para imputar barrio por dirección; se deja como nulo.')

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

zero_ok_columns = {'Bodegas', 'Estacionamientos', 'Antigüedad'}
for column in summary.columns:
    if deptos_df[column].dtype == object:
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

deptos_df = deptos_df.drop(columns=['dirección'])
deptos_df['Gastos comunes'] = deptos_df['Gastos comunes'].apply(parse_measurement).fillna(0)
deptos_df['Tipo de casa'] = deptos_df['Tipo de casa'].fillna('Casa')

# Orientación: la moda cambia bastante de un barrio a otro (la orientación "buena" depende de cómo
# está trazada la calle), así que imputar con la moda del barrio es más certero que la moda global.
# Los 41 barrios del dataset tienen al menos una Orientación conocida, así que esto resuelve casi
# todo; las filas que además no tienen barrio (72) caen a la moda global como último recurso.

moda_por_barrio = deptos_df.groupby('barrio')['Orientación'].agg(lambda s: s.mode().iat[0] if not s.mode().empty else np.nan)
orientacion_na = deptos_df['Orientación'].isna()
deptos_df.loc[orientacion_na, 'Orientación'] = deptos_df.loc[orientacion_na, 'barrio'].map(moda_por_barrio)

orientacion_na = deptos_df['Orientación'].isna()
if orientacion_na.any():
    moda_global = deptos_df['Orientación'].mode().iat[0]
    print(f'{orientacion_na.sum()} casas sin barrio ni Orientación: se imputan con la moda global ({moda_global}).')
    deptos_df.loc[orientacion_na, 'Orientación'] = moda_global

# Cercanía a Clínica Alemana, Estadio Español, Colegio Monte Tabor y Nazaret, Colegio Las Ursulinas
# y Portal La Dehesa:
# este scrape no tiene lat/lon (se agregó al spider -- ver deptos_scraper/spiders/deptos.py -- pero
# recién sirve para la próxima corrida del crawl, unas 9 horas). Sin coordenadas no hay distancia
# real que calcular, así que se aproxima por barrio: los 5 lugares están todos concentrados en el
# corredor Manquehue-La Dehesa (borde Vitacura / Lo Barnechea), y se listan a mano los barrios del
# dataset que caen en ese corredor. Es una aproximación geográfica basada en conocimiento del
# sector, no una distancia medida -- fácil de ajustar editando el set de abajo. Las casas sin
# barrio conocido (barrio nulo) se clasifican como 'Lejos' por defecto, porque no se puede
# confirmar la cercanía.

barrios_cercanos = {
    'La Dehesa', 'Los Trapenses', 'El Huinganal',  # Lo Barnechea: Portal La Dehesa, Monte Tabor y Nazaret y Las Ursulinas están acá
    'Lo Curro', 'Santa María De Manquehue', 'Estadio Manquehue', 'Estadio Croata', 'La Llavería', 'Jardín Del Este',  # Vitacura: entorno de Clínica Alemana y Estadio Español (eje Manquehue)
    'Los Dominicos',  # Las Condes: borde con La Dehesa por Camino Los Trapenses
}
deptos_df['cercania'] = np.where(deptos_df['barrio'].isin(barrios_cercanos), 'Cerca', 'Lejos')
print('Cercanía a Clínica Alemana / Estadio Español / Monte Tabor y Nazaret / Las Ursulinas / Portal La Dehesa:')
print(deptos_df['cercania'].value_counts())

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
    (deptos_df['Baños'] >= 2) &
    (deptos_df['cercania'] == 'Cerca')
].copy()
candidatas['negociacion'] = np.where(candidatas['precio UF'] > 13500, 'Negociar', 'Precio Ok')

print(f'Casas candidatas: {len(candidatas)} de {len(deptos_df)}')
print(candidatas['negociacion'].value_counts())
candidatas.sort_values('precio unitario').to_excel('casas_candidatas.xlsx', index=False)

