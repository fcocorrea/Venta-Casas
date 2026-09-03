# Datos sintéticos para las categorías que todavía no se scrapean
#
# Contexto:
# proyecto_casas.py solo modela "casa en venta" en Vitacura, Las Condes y Lo Barnechea -- es lo
# único que casas_scraper/spiders/casas.py recolecta hoy. El sitio (ver web/) va a mostrar cuatro
# categorías (casa venta, casa arriendo, depto venta, depto arriendo) pero recién existe scraper y
# modelo para la primera. Este script rellena las otras tres con datos FABRICADOS -- no son avisos
# reales, son un placeholder de volumen/forma realista para poder construir y probar el mapa
# multi-categoría (filtros, sliders, popups) antes de que exista un scraper propio para cada una.
#
# Diseño -- por qué esto NO vive dentro de proyecto_casas.py ni reentrena nada:
# Cuando cada categoría tenga su propio scraper, el flujo natural es un modelo de cuantiles por
# categoría (mismo PASO 5 de proyecto_casas.py, corrido una vez por categoría) que termine en una
# tabla con las mismas columnas que ya usa el mapa: comuna/latitud/longitud/precio_real/q05/q50/q95/
# residuo_pct/flag/url + atributos físicos. Generar ACÁ filas con esa misma forma (en vez de filas
# "crudas" tipo casas.json que habría que correr por todo el PASO 2-5) significa que el día que
# aparezca un scraper real para, digamos, "depto en venta", basta con reemplazar el JSON que este
# script produce por una tabla real con las mismas columnas -- proyecto_casas.py (PASO 5i) no
# cambia. El precio "justo" y su intervalo se fabrican con una fórmula hedónica simple + ruido
# log-normal (mismo espacio -- log(precio) -- que usa el modelo real, ver encabezado de
# proyecto_casas.py) en vez de entrenar nada: no hay señal real que aprender todavía.
#
# Geografía: en vez de inventar un bounding box por comuna (que podría caer en un parque o un
# cerro), cada punto sintético parte de un punto REAL de casas.json (mismo comuna) y le aplica un
# jitter gaussiano chico (~250 m) -- barrio incluido. casas.json ya viene geocodificado y filtrado a
# las tres comunas del proyecto, así que es la fuente de coordenadas plausibles más barata a mano.
#
# Salida: tres JSON en la raíz del proyecto (gitignoreados vía el `*.json` de .gitignore, igual que
# casas.json -- son datos regenerables, no algo que versionar). Semilla fija por categoría: correr
# este script dos veces produce exactamente los mismos archivos.

import json

import numpy as np

SEMILLA_BASE = 42
COMUNAS_OBJETIVO = ['Vitacura', 'Las Condes', 'Lo Barnechea']
N_POR_COMUNA = 450

# CLP por m² del "precio justo" -- mensual para arriendo, precio total para venta. Números de
# referencia de mercado del sector oriente de Santiago, no calibrados contra ninguna transacción
# real (no existe ese dato todavía para estas tres categorías).
PRECIO_M2_BASE = {
    'casa_arriendo': {'Vitacura': 9_000, 'Las Condes': 8_000, 'Lo Barnechea': 8_500},
    'depto_venta': {'Vitacura': 4_200_000, 'Las Condes': 3_600_000, 'Lo Barnechea': 3_400_000},
    'depto_arriendo': {'Vitacura': 13_000, 'Las Condes': 11_500, 'Lo Barnechea': 11_000},
}

# Rangos de atributos físicos: perfil "casa" (más grande, más dormitorios) vs. perfil "depto".
RANGOS_ATRIBUTOS = {
    'casa_arriendo': dict(dormitorios=(2, 6), banos=(2, 6), estacionamientos=(0, 4),
                           antiguedad=(0, 40), superficie=(100, 600)),
    'depto_venta': dict(dormitorios=(1, 4), banos=(1, 3), estacionamientos=(0, 2),
                         antiguedad=(0, 30), superficie=(35, 180)),
    'depto_arriendo': dict(dormitorios=(1, 4), banos=(1, 3), estacionamientos=(0, 2),
                            antiguedad=(0, 30), superficie=(35, 180)),
}

# Redondeo del precio final -- una publicación real casi nunca pide un CLP exacto al peso.
PASO_REDONDEO = {'casa_arriendo': 10_000, 'depto_venta': 100_000, 'depto_arriendo': 10_000}

JITTER_GRADOS = 0.0025  # ~250 m
# Ruido log-normal sobre el precio "justo": sigma=0.13 con banda 1.645*sigma en cada lado de q50
# reproduce una cobertura del intervalo ~90% (misma banda que PASO 5e usa para el modelo real),
# para que la proporción visual de casos flaggeados en el mapa se vea comparable entre categorías
# reales y sintéticas.
SIGMA_RUIDO = 0.13
Z_90 = 1.645

ARCHIVOS_POR_CATEGORIA = {
    'casa_arriendo': 'casas_arriendo_sintetico.json',
    'depto_venta': 'departamentos_venta_sintetico.json',
    'depto_arriendo': 'departamentos_arriendo_sintetico.json',
}


def cargar_puntos_reales(ruta_casas_json='casas.json'):
    """(barrio, latitud, longitud) de cada casa real en casas.json, agrupados por comuna --
    ancla geográfica para el jitter de los puntos sintéticos (ver encabezado del archivo)."""
    with open(ruta_casas_json, encoding='utf-8') as f:
        filas_reales = json.load(f)

    puntos_por_comuna = {comuna: [] for comuna in COMUNAS_OBJETIVO}
    for fila in filas_reales:
        comuna = fila.get('comuna')
        if comuna not in puntos_por_comuna:
            continue
        try:
            lat, lon = float(fila['latitud']), float(fila['longitud'])
        except (TypeError, ValueError, KeyError):
            continue
        puntos_por_comuna[comuna].append((fila.get('barrio') or comuna, lat, lon))

    for comuna, puntos in puntos_por_comuna.items():
        if not puntos:
            raise ValueError(f'Sin puntos geocodificados reales para "{comuna}" en {ruta_casas_json} '
                              f'-- no hay ancla geográfica para el jitter sintético.')
    return puntos_por_comuna


def generar_categoria(categoria: str, puntos_por_comuna: dict, semilla: int) -> list[dict]:
    """Genera N_POR_COMUNA filas sintéticas por comuna para `categoria`, con la misma forma de
    columnas que `mapa_datos_completo` usa para las casas reales (ver PASO 5i de
    proyecto_casas.py) -- así el loader del mapa no necesita distinguir origen real/sintético
    salvo para el rótulo."""
    rng = np.random.default_rng(semilla)
    rangos = RANGOS_ATRIBUTOS[categoria]
    paso_redondeo = PASO_REDONDEO[categoria]
    filas = []
    contador = 0

    for comuna in COMUNAS_OBJETIVO:
        pool = puntos_por_comuna[comuna]
        precio_m2 = PRECIO_M2_BASE[categoria][comuna]
        for _ in range(N_POR_COMUNA):
            contador += 1
            barrio, lat_ancla, lon_ancla = pool[rng.integers(len(pool))]
            lat = lat_ancla + rng.normal(0, JITTER_GRADOS)
            lon = lon_ancla + rng.normal(0, JITTER_GRADOS)

            dormitorios = int(rng.integers(rangos['dormitorios'][0], rangos['dormitorios'][1] + 1))
            banos = min(int(rng.integers(rangos['banos'][0], rangos['banos'][1] + 1)), dormitorios + 1)
            estacionamientos = int(rng.integers(rangos['estacionamientos'][0], rangos['estacionamientos'][1] + 1))
            antiguedad = int(rng.integers(rangos['antiguedad'][0], rangos['antiguedad'][1] + 1))
            superficie = float(rng.uniform(*rangos['superficie']))

            # Fórmula hedónica simple: precio_m2 de base, + prima por dormitorio sobre 2, -
            # depreciación por antigüedad (piso 70% del valor para que una casa/depto de 40+ años
            # no valga negativo ni casi cero).
            precio_justo = (precio_m2 * superficie
                             * (1 + 0.02 * (dormitorios - 2))
                             * max(0.7, 1 - 0.003 * antiguedad))

            ruido = rng.normal(0, SIGMA_RUIDO)
            precio_real = precio_justo * np.exp(ruido)
            q50 = precio_justo
            q05 = precio_justo * np.exp(-Z_90 * SIGMA_RUIDO)
            q95 = precio_justo * np.exp(Z_90 * SIGMA_RUIDO)
            residuo_pct = (precio_real - q50) / q50 * 100
            flag = ('sobrevalorada' if precio_real > q95 else
                    'subvalorada' if precio_real < q05 else 'dentro_del_intervalo')

            def redondear(valor):
                return round(valor / paso_redondeo) * paso_redondeo

            filas.append({
                'comuna': comuna, 'barrio': barrio,
                'latitud': lat, 'longitud': lon,
                'Dormitorios': dormitorios, 'Baños': banos,
                'Estacionamientos': estacionamientos, 'Antigüedad': antiguedad,
                'Superficie total': round(superficie, 1),
                'precio_real': redondear(precio_real), 'q05': redondear(q05),
                'q50': redondear(q50), 'q95': redondear(q95),
                'residuo_pct': round(residuo_pct, 2), 'flag': flag,
                'url': f'https://demo.zonecheck.cl/sintetico/{categoria}/{contador}',
            })

    return filas


if __name__ == '__main__':
    puntos_por_comuna = cargar_puntos_reales()

    for desplazamiento, (categoria, nombre_archivo) in enumerate(ARCHIVOS_POR_CATEGORIA.items()):
        # Semilla fija (no hash(), que Python aleatoriza por proceso vía PYTHONHASHSEED) -- dos
        # corridas de este script deben producir bit a bit los mismos tres archivos.
        filas = generar_categoria(categoria, puntos_por_comuna, semilla=SEMILLA_BASE + desplazamiento)
        with open(nombre_archivo, 'w', encoding='utf-8') as f:
            json.dump(filas, f, ensure_ascii=False)
        flaggeadas = sum(1 for fila in filas if fila['flag'] != 'dentro_del_intervalo')
        print(f'{nombre_archivo}: {len(filas)} filas generadas '
              f'({flaggeadas} flaggeadas, {flaggeadas / len(filas):.1%}).')
