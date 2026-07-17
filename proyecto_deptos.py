# Análisis para el arriendo de departamentos
#
# Problemática:
# Quizás más de una vez nos hemos visto en la situación de tener que arrendar un departamento por
# distintos motivos y no sabemos cuál podría ser la mejor opción para nosotros. Hay muchos ámbitos que
# pueden importar al arrendar un departamento: la cercanía a un parque, al metro, o simplemente un
# barrio tranquilo, algo personal para cada uno. Pero incluso encontrando el departamento de nuestros
# sueños, no siempre sabemos si el precio está sobre o bajo el mercado dadas sus características. Como
# los contratos de arriendo suelen ser por un año, una mala decisión se paga por un año completo. Para
# elegir bien necesitamos datos. Muchos datos.
#
# Contexto:
# El estudio se realiza en Santiago de Chile, en cuatro comunas del sector oriente: Las Condes,
# Providencia, Vitacura y Lo Barnechea. Las ofertas se obtienen de Portal Inmobiliario, el mayor sitio
# de arriendos y ventas de Chile, considerando solo publicaciones de arriendo.
#
# Objetivos:
# Predecir el precio de un departamento a partir de sus características y comparar ese precio predicho
# con el precio real, para determinar si está sobre o bajo su precio de mercado. Se trata de un problema
# de regresión.
#
# Implementación, en tres grandes pasos:
# 1) Obtención de datos: se extraen vía web scraping con Scrapy.
# 2) Limpieza de datos: homologar y transformar los datos, que no vienen expresados de forma consistente.
# 3) Exploración: entender los datos para adaptarlos a un modelo de regresión.
# 4) Modelos: probar varios modelos y quedarnos con el más adecuado.
# 5) Producción: ubicar los departamentos en un mapa, coloreados de rojo (sobrevalorado) a verde
#    (subvalorado).
#
# Extracción de datos:
# Se extraen los datos con Scrapy, siguiendo el tutorial oficial
# (https://docs.scrapy.org/en/latest/intro/tutorial.html):
# 1) Crear un proyecto con scrapy startproject <nombreDeMiProyecto>
# 2) Crear un ambiente de trabajo con Python (por ejemplo, venv)
# 3) Instalar scrapy con pip install Scrapy
# 4) En settings.py, definir USER_AGENT para que el servidor no rechace las peticiones, por ejemplo:
#    USER_AGENT = 'Mozilla/5.0 (compatible; Googlebot/2.1; +https://www.google.com/bot.html)'
# 5) En el mismo archivo, definir DOWNLOAD_DELAY (por ejemplo 1.5) para no ser detectados como bot
# 6) Crear una clase que realice el scraping
# 7) Ejecutar scrapy crawl <nombreDeMiProyecto> -O nombreDeMiProyecto.json para extraer los datos a un
#    json
#
# Para saber qué extraer y cómo (paso 5) hay que explorar la página con XPath o CSS Locators. En Portal
# Inmobiliario nos interesa el título, el precio y la tabla de características del departamento (metros
# cuadrados, tipología, orientación, etc).

import scrapy

class DeptosSpider(scrapy.Spider):
    name = "deptos"

    def start_requests(self):
        urls = ('https://www.portalinmobiliario.com/venta/departamento/vitacura-metropolitana/',
                'https://www.portalinmobiliario.com/venta/casa/vitacura-metropolitana/',
                'https://www.portalinmobiliario.com/venta/departamento/las-condes-metropolitana',
                'https://www.portalinmobiliario.com/venta/casa/las-condes-metropolitana',
                'https://www.portalinmobiliario.com/venta/departamento/lo-barnechea-metropolitana',
                'https://www.portalinmobiliario.com/venta/casa/lo-barnechea-metropolitana'
        )
        
        for url in urls:
            yield scrapy.Request(url, self.parse)
    
    def parse(self, response):
        for url in response.css('a.poly-component__title::attr(href)'):
            yield response.follow(url=url.get(), callback=self.parse_links)
        yield response.follow(url=response.css('a[title="Siguiente"]::attr(href)').get())

    def parse_links(self, response):
        data_table = {}        
        for table in response.css('table'): # extraemos información desde la tabla
            for row in table.css('tr'):
                key = row.css('th div::text').get()
                value = row.css('td span::text').get()
                if key == None: key = ''
                if value == None: value = ''
                data_table[key.strip()] = value.strip()
        
        for text in response.css('body'): # extraemos información básica
            data =  {
                'titulo': text.css('.ui-pdp-title::text').get(),
                'precio': text.css('.andes-money-amount__fraction::text').get(),
                'clp': text.css('.ui-pdp-family--REGULAR .andes-money-amount__fraction::text').get(),
                'dirección': text.css('#location_and_points .ui-pdp-media__title::text').get(),
                'comuna': text.css('div.ui-pdp-breadcrumb>nav>ol>li>a::attr(title)').getall()[-2],
                'barrio': text.css('div.ui-pdp-breadcrumb>nav>ol>li>a::attr(title)').getall()[-1],
                'url': response.url
            }
            data.update(data_table)
            yield data

# La clase anterior hereda de scrapy.Spider. El "Spider" (araña) va tejiendo una telaraña de página a
# página para extraer información, yendo y viniendo entre publicaciones.
#
# Con la araña lista, se ejecuta en el terminal para obtener un json con toda la información:
# scrapy crawl deptos -O deptos.json
# Ese comando extrae los datos a un json que luego cargamos en un dataframe para trabajarlo.

import pandas as pd

deptos_df = pd.read_json('deptos.json')
deptos_df.head()

print('Cantidad de observaciones: {}.\nCantidad de atributos: {}.'.format(*deptos_df.shape))

# Limpieza
#
# Atributos:
# El dataframe tiene muchos atributos y hay que reducir la dimensionalidad para un análisis más
# centrado. Muchos son categóricos binarios (indican si el departamento tiene o no cierta
# característica). Algunos no aportan mucho valor, como el número de torre o si tiene chimenea (en
# Santiago es ilegal prender fuego en chimenea). Otros dependen de qué tan importante sea para cada
# arrendatario, como si el departamento admite mascotas o no.

# Columnas actuales, separando más adelante numéricas de categóricas.

', '.join(deptos_df.columns.tolist())

# Se eliminan primero las columnas que no agregan valor a la hora de elegir un departamento.

remove_columns = ['Chimenea', 'Número de torre', 'Precio estacionamiento desde (UF)', 'Precio bodega desde (UF)', 'Fecha de entrega',
                  'Disponible desde', 'Tipo de seguridad']
deptos_df.drop(remove_columns, axis=1, inplace=True)

# Atributos categóricos binarios:
# indican si el departamento tiene o no una característica (TV por cable, quincho, azotea, etc).

binary_features = deptos_df.loc[:,list(filter(lambda x: len(deptos_df[x].unique()) < 4, deptos_df.columns))]
binary_features

count_binary_attributes = binary_features.shape[1]
count_deptos_attributes = deptos_df.shape[1]
print(f'Los atributos binarios son {count_binary_attributes} y representan el {round((count_binary_attributes/count_deptos_attributes)*100, 4)} % del total de atributos')

# Consideramos binario un atributo con 3 valores únicos o menos (el 3 contempla los nulos). Los
# atributos binarios son más del 70% del total, por lo que más adelante reduciremos esa dimensionalidad
# eliminando los que no inciden en el precio.

# Definir un vector objetivo y transformar los datos numéricos:
# el precio por metro cuadrado es un buen indicador porque tiene una escala homogénea (precio en pesos
# chilenos dividido por los metros cuadrados totales). Se usan los metros cuadrados totales y no los
# útiles porque estos últimos no incluyen jardineras, terrazas ni logias (fuente: http://rb.gy/gce8g).
#
# Creamos el atributo UM (unidad monetaria): en general los departamentos se arriendan en pesos
# chilenos o UF, aunque también hay casos inusuales en dólares. Nota: la UF no es técnicamente una
# unidad monetaria, pero la tratamos como tal para que se sepa bajo qué base se está cobrando.

import numpy as np

um = np.where(deptos_df.precio.str.contains('unidades de fomento'), 'UF',
              np.where(deptos_df.precio.str.contains('lares'), 'USD',  # matchea 'dólares' y 'dolares'
              np.where(deptos_df.precio.str.contains('pesos'), 'CLP',
              'Unknown')))
deptos_df['UM'] = um

# Si no hay ningún valor que no sea pesos, UF o dólares, esta consulta debería dar 0; los casos con
# otra moneda ('Unknown') se revisan uno a uno.

unkown_um = deptos_df[deptos_df.UM == 'Unknown']
print(f'Hay {unkown_um.shape[0]} departamento(s) con una moneda de arriendo distinta a dólares, pesos o UF.')

# Se elimina el texto de la columna precio (usaremos clp en su lugar). Primero identificamos qué
# atributos son numéricos y cuáles siguen como object.

numeric_attributes = ['precio', 'clp', 'Superficie total', 'Superficie útil', 'Superficie de terraza',
                      'Ambientes', 'Dormitorios', 'Baños', 'Estacionamientos',
                      'Cantidad máxima de habitantes', 'Antigüedad']
deptos_df.loc[:,numeric_attributes].dtypes

# Descartamos los que ya son float64; el resto sigue como object y hay que convertirlos.

float_64_attributes = {'Ambientes', 'Estacionamientos', 'Cantidad máxima de habitantes'}
numeric_attributes = list(set(numeric_attributes) - float_64_attributes)

# Los valores en UF con decimales aparecen como 'X unidades de fomento con Y centavos', y las
# superficies como 'número m^2'. Extraemos por regex los dígitos y el punto decimal para poder
# convertirlos a número.

def list_to_number(list_of_numbers):
    return f'{list_of_numbers[0]}.{list_of_numbers[1]}' if len(list_of_numbers) == 2 else list_of_numbers[0]

deptos_df.precio = deptos_df.precio.str.findall(r'\d+').apply(list_to_number)

# findall devuelve una lista: tamaño 1 es un precio en pesos o UF sin decimales, tamaño 2 es una UF con
# decimales (dejamos ambos valores unidos por un punto).

# clp es el precio en pesos chilenos equivalente, y solo viene informado cuando el arriendo está en una
# moneda distinta a CLP. Cuando el arriendo ya está en CLP, ese valor vive en la columna precio.

deptos_df.clp = np.where(deptos_df.clp.isnull(), deptos_df.precio, deptos_df.clp)

# El precio (numerador del precio unitario) ya está listo. Falta la superficie total (denominador).
# Algunas superficies vienen como rango (ej. 45-52 mts2) cuando se arrienda más de un departamento en
# el mismo edificio; en ese caso el precio publicado corresponde al departamento más chico del rango
# (ver imagen rango.PNG).

deptos_df[deptos_df['Superficie útil'].str.find('-') != -1]

# Para los rangos tomamos el valor mínimo (primer elemento tras el split), lo que no afecta a los
# valores que no vienen como rango, así que aplicamos el mismo tratamiento a todos los atributos
# numéricos.

deptos_df['clp'] = deptos_df['clp'].str.replace('.', '')  # quita los separadores de miles antes de convertir a float
for numeric_attribute in numeric_attributes:
    deptos_df[numeric_attribute] = deptos_df[numeric_attribute].str.split().str.get(0).astype(float)

# Puede haber errores de tipeo o superficies en cero por no haber sido completadas.

deptos_df.loc[deptos_df['Superficie total'] == 0].head(10)

# Cuando falta superficie total o superficie útil pero no la otra, rellenamos con la que sí está. Los
# departamentos sin ninguna de las dos se descartan del dataset más abajo, porque no hay forma de saber
# si están con sobreprecio.

deptos_df['Superficie útil'] = np.where((deptos_df['Superficie útil'] == 0.0) & (deptos_df['Superficie total'] != 0.0), 
                                         deptos_df['Superficie total'], deptos_df['Superficie útil'])
deptos_df['Superficie total'] = np.where((deptos_df['Superficie total'] == 0.0) & (deptos_df['Superficie útil'] != 0.0), 
                                         deptos_df['Superficie útil'], deptos_df['Superficie total'])
index_deptos_sin_superficie = deptos_df[deptos_df['Superficie útil'] == 0.0].index
print(f'Departamentos sin superficie: {len(index_deptos_sin_superficie)}')

# Se eliminan estos 18 departamentos: insignificante frente a los más de 4.400 arriendos del dataset.

deptos_df.drop(index_deptos_sin_superficie, inplace=True)
deptos_df.shape

# Cuando la superficie útil es mayor que la total (no tiene sentido), se usa la útil como total.

deptos_df['Superficie total'] = np.where(deptos_df['Superficie útil'] > deptos_df['Superficie total'],
                                        deptos_df['Superficie útil'], deptos_df['Superficie total'])

# Con precio y superficie listos, calculamos la variable objetivo.

deptos_df['precio unitario'] = (deptos_df['clp'] / deptos_df['Superficie total']).round(4)

# Valores extremos:
# revisamos errores de tipeo o departamentos en venta (en lugar de arriendo) antes de seguir con el
# análisis.

import matplotlib.pyplot as plt

plt.hist(deptos_df['precio unitario'])
plt.gca().get_xaxis().get_major_formatter().set_scientific(False)
plt.xlabel('Precio por metro cuadrado')
plt.ylabel('Frecuencia')
plt.title('Histograma de Precio por metro cuadrado.')
plt.xticks(rotation=90)
plt.show()

# El histograma muestra la mayoría de los valores bajo los 500.000 pesos por metro cuadrado, porque
# unos pocos datos extremos llegan a precios mucho más altos. Veamos cómo cambia si nos quedamos solo
# con el rango intercuartílico (percentil 25 al 75).

def percentil_limits(serie: pd.Series, percentils:tuple)->tuple:
    """Límites inferior y superior para considerar un valor extremo."""
    lower_limit = serie.quantile(percentils[0])
    upper_limit = serie.quantile(percentils[1])
    percentil_ranges = upper_limit - lower_limit
    lower_bound = lower_limit - 1.5 * percentil_ranges
    upper_bound = upper_limit + 1.5 * percentil_ranges
    return lower_bound, upper_bound

def get_interquartile_range(serie:pd.Series)->pd.Series:
    """Valores de la serie dentro del rango intercuartílico."""
    lower_bound, upper_bound = percentil_limits(serie, (0.25, 0.75))
    data_within_iqr = serie[(serie >= lower_bound) & (serie <= upper_bound)]
    return data_within_iqr

unitary_price = deptos_df['precio unitario']
interquartile_range = get_interquartile_range(unitary_price)
plt.xlabel('Precio por metro cuadrado')
plt.ylabel('Frecuencia')
plt.title('Histograma de precio por metro cuadrado.')
plt.hist(interquartile_range, bins='sturges')  # regla de Sturges para el número de barras
plt.show()

# Al filtrar los extremos, la distribución se ve mucho más normal: va de los 4.000 a poco más de
# 18.000 pesos por metro cuadrado, concentrada entre los 10.000 y 12.000.

def show_outliers(df:pd.DataFrame)->pd.Series:
    """Valores extremos (bajos y altos) de 'precio unitario'."""
    serie = deptos_df['precio unitario']
    lower_bound, upper_bound = percentil_limits(serie, (0.25, 0.75))
    return serie.loc[serie < lower_bound], serie.loc[serie > upper_bound]

extreme_values = show_outliers(deptos_df)
lower_values = extreme_values[0]
higher_values = extreme_values[1]
extreme_values_df = pd.DataFrame({'valores extremos bajos': len(lower_values),
                                 'valores extremos altos': len(higher_values)},
                                 index = [0])
extreme_values_df

# Hay muchos más valores extremos altos que bajos; revisemos cada grupo más de cerca.
#
# Valores extremos bajos (1.5 veces menores que el primer cuartil): interesa ver cómo se comporta el
# precio unitario en la cola baja.

def plot_sorted_attribute(serie, ascending, ylabel, xlabel, **plot_config):
    serie = serie.sort_values(ascending=ascending)
    plt.ticklabel_format(style='plain', axis='both')  # evita notación científica en los ejes
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.plot(serie.values, **plot_config)
    plt.show()

plot_sorted_attribute(lower_values, False, 'Precio Unitario',
                      'Departamento (N°)', color='darkblue', marker='o',
                      linestyle='dashed', linewidth=2, markersize=8)

# La pendiente empieza a bajar justo antes de los 2.500 pesos por metro cuadrado, así que eliminamos
# los valores bajo ese punto (probables errores de tipeo en precio o superficie). Igual que con los
# extremos altos, guardamos lo eliminado por si hace falta revisarlo después.

delete_lower_values = deptos_df[deptos_df['precio unitario'] < 2500]
deptos_df = deptos_df.drop(delete_lower_values.index)

# Valores extremos altos:
# aquí probablemente hay departamentos en venta (no arriendo) por error de scraping, departamentos
# amoblados (más caros) o publicaciones falsas (ver imagen eliminar.PNG).

plot_sorted_attribute(higher_values, True, 'Precio Unitario',
                      'Departamento (N°)', color='darkblue', linestyle='dashed')

# El precio unitario sube fuerte después del departamento N° 140, probablemente por departamentos en
# venta. Guardamos lo eliminado por si hace falta revisarlo después.

higher_values = higher_values.sort_values().reset_index(drop=True)
print(higher_values[140:142])
threshold = higher_values[141]
delete_upper_values = deptos_df[deptos_df['precio unitario'] >= threshold]
deptos_df = deptos_df.drop(delete_upper_values.index)

# Comunas:
# en la página de origen, el último elemento de la ruta de navegación corresponde al barrio y el
# penúltimo a la comuna. Si el departamento no tiene barrio, la comuna termina ocupando el lugar del
# barrio en el dataset, así que hay que corregirlo.

deptos_df.comuna.unique().tolist()

# Las "comunas" RM (Metropolitana) y Propiedades usadas hay que descartarlas o reemplazarlas.

comunas_a_modificar = deptos_df[(deptos_df.comuna == 'RM (Metropolitana)') | (deptos_df.comuna == 'Propiedades usadas')]
comunas_a_modificar.head()

comunas_a_modificar.barrio.unique().tolist()

# En estos casos el atributo "barrio" en realidad contiene la comuna, y "comuna" contiene la región o
# la sección "propiedades usadas" de la página, porque el barrio real no está detallado. La ruta de
# navegación de origen es: Propiedades usadas > RM (Metropolitana) > Comuna > Barrio.
#
# - Si la comuna es "Propiedades usadas", la propiedad no tiene ni comuna ni barrio: dejamos el barrio
#   como nulo.
# - Para "Propiedades usadas" y "RM (Metropolitana)", la comuna queda como valor nulo (se imputa más
#   abajo).

deptos_df.barrio = np.where(deptos_df.comuna == 'Propiedades usadas', np.nan, deptos_df.barrio)
deptos_df.loc[comunas_a_modificar.index, 'comuna'] = np.nan

print(f'Hay {deptos_df.comuna.isna().sum()} comunas por imputar.')

# Imputación de valores nulos en comuna:
# - comuna nula, barrio no: si el barrio identifica una única comuna, la usamos; si no, recurrimos a
#   la dirección.
# - comuna y barrio nulos: imputamos por dirección.

barrios = deptos_df.loc[deptos_df.comuna.isna(), 'barrio'].unique().tolist()
barrios = [i for i in barrios if isinstance(i, str)]
for barrio in barrios:
    comuna_del_barrio = deptos_df.loc[deptos_df.barrio == barrio, 'comuna'].unique()
    comuna_del_barrio = [i for i in comuna_del_barrio if isinstance(i, str)]
    if comuna_del_barrio == []:
        # "barrio" no tiene ninguna comuna asociada: en realidad contiene el nombre de la comuna
        deptos_df.loc[deptos_df.barrio == barrio, 'comuna'] = barrio
        deptos_df.loc[deptos_df.barrio == barrio, 'barrio'] = np.nan
    else:
        # si un barrio aparece asociado a más de una comuna, se usa la más frecuente
        comuna = deptos_df.loc[deptos_df.comuna.isin(comuna_del_barrio), 'comuna'].value_counts().idxmax()
        deptos_df.loc[deptos_df.barrio == barrio, 'comuna'] = comuna

print(f'Ahora hay {deptos_df.comuna.isna().sum()} comunas con valores nulos')
comunas = [i for i in deptos_df.comuna.unique() if isinstance(i, str)]
print(f'Las comunas unicas que quedan son: ', ', '.join(comunas))

# Para los registros donde comuna y barrio siguen nulos, se imputa la comuna por dirección. Una misma
# calle puede cruzar más de una comuna, y no siempre hay número para ubicar el tramo exacto.

com_barr_null = deptos_df.loc[(deptos_df.comuna.isna()) & (deptos_df.barrio.isna())]
com_barr_null

# En este dataset solo aparece un caso, pero el código siguiente es genérico para cubrir más casos si
# se actualizan los datos.

from fuzzywuzzy import process

def better_coincidence(text:str, choices:pd.Series)->list:
    """Compara un texto con una pd.Series y retorna [(mejor coincidencia, puntaje, índice)]."""
    return process.extract(text, choices, limit=1)

coincidences = {}  # {dirección: [comuna, puntaje de coincidencia, índice]}
for comuna in comunas:
    address, score, idx = better_coincidence(comuna, com_barr_null.dirección)[0]
    if (address in coincidences and score > coincidences[address][1]) or address not in coincidences:
        coincidences[address] = [comuna, score, idx]
for address, features in coincidences.items():
    deptos_df.loc[features[2], 'comuna'] = features[0]

print(f'Ahora hay {deptos_df.comuna.isna().sum()} comunas con valores nulos')

# Barrios:
# mismo problema que con las comunas.

deptos_df.barrio.unique()

# Un mismo barrio puede aparecer escrito de formas distintas (ej. "pedro de valdivia" vs "Pedro de
# Valdivia"), así que se capitalizan todos.

deptos_df.barrio = deptos_df.barrio.str.title()

print(f'Hay {deptos_df.barrio.isna().sum()} departamentos sin barrio.')

# Para imputar el barrio de estos registros: tomamos los departamentos que sí tienen barrio, buscamos
# la dirección con mejor coincidencia usando better_coincidence(), y le asignamos su barrio.

deptos_with_hood = deptos_df[~deptos_df.barrio.isna()]
deptos_without_hood = deptos_df.loc[deptos_df.barrio.isna(), 'dirección'].to_dict()

# Esta imputación por dirección es lenta (son muchas direcciones para comparar una a una), así que el
# resultado ya fue calculado una vez y quedó guardado en 'arriendos_clean.csv'.
deptos_df = pd.read_csv('arriendos_clean.csv', index_col=0)

# Rangos de valores:
# revisamos qué atributos tienen valores inverosímiles (mal tipeados o sin sentido) y los corregimos.

summary = deptos_df.describe()
summary

# Último cuartil:
# en las filas 75% y max, si la diferencia entre ambas es enorme para un atributo numérico, suele ser
# un error de tipeo o de contexto. Por ejemplo, en "Antigüedad" el cuantil 75% son 20 años, pero el
# máximo es 2023: imposible como antigüedad, pero tiene sentido si en realidad es el año de
# construcción del edificio.

from datetime import datetime
year = datetime.now().year
deptos_df['Antigüedad'] = np.where(deptos_df['Antigüedad'] >= 1800, year - deptos_df['Antigüedad'], deptos_df['Antigüedad'])

deptos_df['Antigüedad'].nlargest(5)

# Con la antigüedad corregida, los valores sobre 1.000 años restantes no tienen sentido: se dejan como
# nulos.

deptos_df['Antigüedad'] = np.where(deptos_df['Antigüedad'] > 1000, np.nan, deptos_df['Antigüedad'])

# El mismo tipo de revisión aplica a otras variables cuyos extremos no hacen sentido:
# Superficie de terraza, Baños, Estacionamientos, Cantidad máxima de habitantes, Bodegas,
# Departamentos por piso.

bad_ranges = ('Superficie de terraza', 'Baños', 'Estacionamientos', 'Cantidad máxima de habitantes',
              'Bodegas', 'Departamentos por piso')
data = [deptos_df[column].nlargest(10).tolist() for column in deptos_df.loc[:, bad_ranges]]
data = np.array(data).T
pd.DataFrame(data, columns=bad_ranges)

# Analizando los 10 valores más grandes de cada atributo:
# - Superficie de terraza: salto de más de 7x en el segundo valor más grande; poco creíble que una
#   terraza tenga esa superficie. Se anulan los tres primeros valores para imputarlos según el tamaño
#   del departamento.
# - Baños: el máximo va de 8 a 22 baños, probable error de tipeo (2 en vez de 22).
# - Estacionamientos: salta a 44 y luego a 220; probablemente 4 y 2.
# - Cantidad máxima de habitantes: aumento considerable en los tres valores más grandes, se anulan.
# - Bodegas: mismo patrón que baños y estacionamientos, se anulan por las dudas.
# - Departamentos por piso: podría en realidad ser el total de departamentos del edificio (habría que
#   dividir por el número de pisos); revisar con más detalle.
#
# Repetir este análisis a mano por cada atributo no escala si se actualizan los datos, así que se
# generaliza la estrategia:
# - tomar los 20 valores más grandes de cada atributo,
# - recorrerlos de menor a mayor buscando un salto irrisorio (más del 100% respecto al valor anterior),
# - anular todos los valores desde ese salto en adelante, para imputarlos más adelante.

for column in bad_ranges:
    top_20 = deptos_df[column].nlargest(20)[::-1]
    for i, margin_change in enumerate(top_20.pct_change()):
        if margin_change > 1:
            idx = top_20.tail(20-i).index
            print(f'Asignamos a nan {len(idx)} valores de la columna "{column}"')
            deptos_df.loc[idx, column] = np.nan
            break

# Con los valores irrisorios anulados, empezamos a poblar los valores nulos.
#
# Valores negativos o ceros:
# - no tiene sentido que un departamento tenga 0 baños, dormitorios, ambientes o pisos (nadie podría
#   vivir ahí); hay que decidir si imputarlos por 1 o dejarlos nulos para imputar más adelante.
# - tampoco tienen sentido valores negativos, salvo el número de piso de la unidad, donde -1 es válido
#   (departamentos en subterráneo).
# Como no se puede saber el valor real, se dejan como nulos para imputar después.

min_zero_attr = (summary.loc['min'] <= 0)
min_attr = summary.loc['min', min_zero_attr[min_zero_attr].index].sort_values()
min_attr

# Se excluyen de los ceros las bodegas y estacionamientos (hay departamentos que se arriendan sin
# ellos), y de los negativos el número de piso de la unidad (el -1 es válido).

set_to_na = min_attr.loc[~min_attr.index.isin(['Bodegas', 'Estacionamientos', 'Número de piso de la unidad'])]
for attr, value in set_to_na.items():
    deptos_df.loc[deptos_df[attr] == value, attr] = np.nan

# Valores Nulos:
# al ser datos anotados manualmente por cada arrendador, es normal tener muchos valores nulos,
# especialmente en los atributos categóricos de las tablas de cada publicación.

total_observations = len(deptos_df)
na_values = deptos_df.isnull().sum()
percent_na = ((na_values /  total_observations) * 100).round(2)
na_summary = pd.concat([na_values, percent_na], axis=1)
na_summary.columns = ['Cantidad de Nulos', 'Porcentaje de Nulos']
na_summary = na_summary[na_summary['Cantidad de Nulos'] > 0]
na_summary.sort_values('Cantidad de Nulos', ascending=False, inplace=True)
na_summary

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
plt.figure(figsize=(16, 8))
sns.set(font_scale=.75)
ax = sns.barplot(x='Porcentaje de Nulos', y=na_summary.index, data=na_summary)
plt.title('Número de valores nulos por variable')
plt.xlabel('Porcentaje valores nulos')
plt.ylabel('Atributos')
ax.set_xlim(0, 100)
plt.show()

# Los valores nulos se concentran, para la gran mayoría de los atributos, entre un 20% y un 30%. Para
# poblarlos, conviene separar el problema en tres grupos: variables categóricas binarias (la mayoría),
# variables numéricas discretas y variables numéricas continuas. Donde el atributo tenga alta
# correlación con otro, se puede predecir su valor con un algoritmo de regresión; si no, hay que idear
# otra estrategia.
#
# Imputación de valores nulos en variables categóricas binarias:
# se usa regresión logística, así que primero convertimos "Sí"/"No" a 1/0.

def not_nan_values(X, y):
    """Filtra X e y para conservar solo las filas donde y no es nulo."""
    y = y.loc[y.notnull()]
    X = X.loc[y.index]
    return X, y

def train_split_with_nan_values(X:pd.Series, y:pd.Series)->tuple:
    """Arrays de entrenamiento (X, y) usando solo las filas donde y no es nulo."""
    X, y = not_nan_values(X, y)
    y_train = y.replace('Sí', 1).replace('No', 0).values
    X_train = X.values.reshape(-1, 1)
    return X_train, y_train

from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

def predict_missing_values(X:pd.Series, y:pd.Series)->pd.Series:
    """Imputa los valores nulos de una variable categórica binaria vía regresión logística."""
    X_train, y_train = train_split_with_nan_values(X, y)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    na_index = y.loc[y.isna()].index
    X_predict = X.loc[na_index].values.reshape(-1,1)
    predicted_values = lr.predict(X_predict)
    y.loc[na_index] = predicted_values
    return y

X = deptos_df['precio unitario']
for binary_feature in binary_features:
    y = deptos_df[binary_feature]
    predicted_na_values = predict_missing_values(X, y)
    deptos_df.loc[:, binary_feature] = predicted_na_values

# Valores nulos restantes tras imputar las variables categóricas binarias.

def columns_with_missing_values()->pd.DataFrame:
    null_values = {column:deptos_df[column].isna().sum()
                   for column in deptos_df.columns 
                   if deptos_df[column].isna().sum() > 0}
    null_values = pd.DataFrame(list(null_values.items()), columns=['Atributo', 'Valores Nulos'])
    null_values = null_values.set_index('Atributo').sort_values('Valores Nulos')
    null_values['Tipo de Dato'] = [deptos_df[column].dtype for column in null_values.index]
    return null_values

columns_with_missing_values()

# Relación entre atributos con datos faltantes:
# se revisa la correlación entre los atributos numéricos para poder imputar un atributo nulo a partir
# de otro con alta correlación.

corr = deptos_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap="Blues", annot=True)
plt.show()

# Imputación en variables discretas:
# variables numéricas que toman un número finito de valores. Primero identificamos cuáles tienen
# valores nulos.

discrete_values = ('Estacionamientos', 'Dormitorios', 'Baños', 'Bodegas', 'Número de piso de la unidad',
                       'Departamentos por piso', 'Antigüedad', 'Cantidad de pisos',
                       'Cantidad máxima de habitantes', 'Ambientes')
na_columns = columns_with_missing_values()
na_columns[na_columns.index.isin(discrete_values)]

# Para cada atributo discreto con nulos, ordenamos el resto de atributos de mayor a menor correlación.

def attribute_correlations(attribute)->pd.Index:
    """Atributos ordenados de mayor a menor correlación respecto a `attribute`."""
    return corr.loc[corr.index != attribute, attribute].sort_values(ascending=False, key=lambda x:abs(x)).index

# Llamamos "match" al atributo con mayor correlación respecto al que queremos imputar (ej. el match de
# Baños es Dormitorios, y viceversa). La estrategia depende de si el match es discreto o continuo:
# - Match discreto: tabla cruzada entre ambos atributos, imputando según la combinación más frecuente
#   (ej. si 2 dormitorios se combina más seguido con 2 baños, un baño nulo con 2 dormitorios se imputa
#   como 2).
# - Match continuo: se agrupan los valores discretos por la mediana del atributo continuo, y el valor
#   nulo se imputa según el grupo cuya mediana esté más cerca.
#
# Si tanto el atributo a imputar como su match están nulos en la misma fila, no hay referencia posible,
# así que se prueba con el siguiente atributo de mayor correlación, y así sucesivamente. Si todos los
# match posibles resultan nulos para una fila, se imputa la moda del atributo.

def get_cross_value(value:int, attribute:str, match_attribute:str)->int:
    """Valor a imputar según la tabla cruzada cuando el atributo match es discreto."""
    cross_table = pd.crosstab(deptos_df[match_attribute], deptos_df[attribute])
    try:
        cross_table.loc[value].idxmax()
    except KeyError:  # value no existe en la tabla cruzada
        return None
    return cross_table.loc[value].idxmax()

def match_attribute_values(attribute:str, match_attribute:str)->pd.Series:
    """Valores del atributo match para las filas donde `attribute` es nulo."""
    na_values_idx = deptos_df[deptos_df[attribute].isna()].index
    match_attributes = deptos_df[match_attribute].loc[na_values_idx]
    return match_attributes

def allocate_value_discrete(not_allocated_values:pd.Series, attribute:str, match_attribute:str)->None:
      """Imputa los valores nulos de un atributo numérico discreto."""
      for idx, value in not_allocated_values.items():
            allocate_value = get_cross_value(value, attribute, match_attribute)
            if allocate_value:
                  print(f'{attribute} - {match_attribute} - {idx}-{value} - {allocate_value}')
            else:
                  return False  # no se pudo imputar con este match
      return True  # todos los valores fueron imputados

def get_not_allocated_values(match_values:pd.Series, allocated_values:list)->pd.Series:
    """Valores del atributo match aún no imputados, descartando los que son nulos."""
    not_nan = match_values[~match_values.isna()]
    not_nan_index = not_nan.index
    not_allocated_values = not_nan[~not_nan_index.isin(allocated_values)]
    return not_allocated_values

def allocate_discrete_with_discrete(attribute:str, match_attribute:str, allocated_values:list)->pd.Series:
    """Imputa los valores nulos cuando el atributo match también es discreto."""
    match_values:pd.Series = match_attribute_values(attribute, match_attribute)
    not_allocated_values = get_not_allocated_values(match_values, allocated_values)
    allocate_value = allocate_value_discrete(not_allocated_values, attribute, match_attribute)
    return allocate_value

# Con la estrategia de match discreto lista, falta la de match continuo: se agrupan los valores
# discretos según la mediana del atributo continuo (ej. la mediana del precio de arriendo para cada
# cantidad de dormitorios), y se agrupa por mediana en vez de promedio para no verse afectado por
# valores extremos. Un valor nulo se imputa según qué grupo tiene la mediana más cercana a su valor
# continuo.

def group_by_median(attribute:str, match_attribute:str):
    """Agrupa un atributo discreto contra uno continuo usando la mediana."""
    groupby_median = deptos_df.groupby(attribute)[match_attribute].median()
    return groupby_median

def get_smallest_difference(value, grouped_values):
    """Valor discreto cuyo grupo tiene la mediana más cercana a `value`."""
    differences = (grouped_values - value).abs()
    allocate_value = differences[differences == differences.min()].index[0]
    return allocate_value

def allocate_value_continuous(not_allocated_values:pd.Series, grouped_values, attribute:str, match_attribute:str)->None:
      """Imputa los valores nulos de un atributo numérico discreto según su match continuo."""
      for idx, value in not_allocated_values.items():
            allocate_value = get_smallest_difference(value, grouped_values)
            print(f'{attribute} - {match_attribute} - {idx}-{value} - {allocate_value}')

def allocate_discrete_with_continuous(attribute:str, match_attribute:str, allocated_values:list):
    """Imputa los valores nulos cuando el atributo match es continuo."""
    grouped_values = group_by_median(attribute, match_attribute)
    match_values:pd.Series = match_attribute_values(attribute, match_attribute)
    not_allocated_values = get_not_allocated_values(match_values, allocated_values)
    allocate_value = allocate_value_continuous(not_allocated_values, grouped_values, attribute, match_attribute)
    return allocate_value

# Juntamos ambas estrategias (match discreto y match continuo) en una sola función.

def allocate_values():
    for attribute in discrete_values:
        allocated_values = []
        for match_attribute in attribute_correlations(attribute):
            if match_attribute in discrete_values:
                not_allocated_values = allocate_discrete_with_discrete(attribute, match_attribute, allocated_values)
                if not_allocated_values: break
        not_allocated_values = allocate_discrete_with_continuous(attribute, attribute_correlations[0], allocated_values)
        allocated_values.extend(not_allocated_values.index)

# Bug conocido: falla en el índice 38.0 (Número de piso de la unidad / Cantidad máxima de habitantes).
allocate_values()

deptos_df.Estacionamientos.isna().sum()

# Imputación en variables continuas y en variables categóricas: pendiente.

# Exploración:
# con los datos limpios, exploramos el precio por metro cuadrado por comuna y su relación con la
# superficie.

import matplotlib.pyplot as plt
import seaborn as sns

g = sns.catplot(x='precio unitario', y='comuna',
            data=deptos_df, kind='box', showfliers = False)
plt.show()

# hipótesis: a medida que aumenta la superficie útil, disminuye el precio unitario

sns.relplot(x='precio unitario', y='Superficie total', data=deptos_df, kind='scatter', hue='comuna')
plt.show()

