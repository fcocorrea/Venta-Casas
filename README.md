# Casas en venta — Vitacura, Las Condes y Lo Barnechea

Scraper en Scrapy para casas en venta en las tres comunas (Portal Inmobiliario), más un pipeline
completo de modelamiento que limpia los datos, entrena un modelo de precio con intervalos de
predicción, y arma un ranking de casas sub/sobrevaloradas con export y mapa interactivo.

## Requisitos

Python 3.12. Dependencias congeladas en `requirements.txt`:

```bash
python -m venv venv
./venv/Scripts/python.exe -m pip install -r requirements.txt
```

El `python` del PATH puede no ser el correcto (en Windows, el de la Microsoft Store no trae
`scikit-learn`) — usar siempre `./venv/Scripts/python.exe` para correr el spider y el script de
análisis.

## Uso

```bash
# Correr el spider (sobreescribe el output)
./venv/Scripts/python.exe -m scrapy crawl casas -O casas.json

# Verificar que el spider carga sin errores
./venv/Scripts/python.exe -m scrapy check casas

# Análisis completo: limpieza, features, CV, modelo, intervalos, ranking, export, mapa
./venv/Scripts/python.exe proyecto_casas.py

# Pipeline completo (crawl + análisis), lo mismo que corre la tarea programada
run_daily_crawl.bat
```

`proyecto_casas.py` lee únicamente `casas.json` de punta a punta (sin checkpoints intermedios) y
corre limpio sobre un crawl nuevo. Incluye validación cruzada repetida y búsqueda de
hiperparámetros, así que una corrida completa toma varios minutos, no segundos.

## Estructura

```
casas_scraper/
  spiders/casas.py         # único spider ("casas")
  settings.py               # USER_AGENT, DOWNLOAD_DELAY, etc.
proyecto_casas.py            # limpieza, features, CV, modelo, intervalos, ranking, export, mapa
run_daily_crawl.bat            # crawl -> casas.json -> proyecto_casas.py
requirements.txt                # dependencias congeladas del venv (scraper + análisis, un solo venv)
PENDIENTES.md                   # estado del proyecto, decisiones tomadas, trabajo pendiente
```

### Salidas (se sobreescriben en cada corrida, commiteadas)

- `casas_limpios.xlsx` — dataset completo, limpio e imputado (train + test).
- `casas_candidatas.xlsx` — solo las casas de test cuyo precio real cae fuera de su intervalo de
  predicción (sub o sobrevaloradas), ordenadas por distancia al borde del intervalo.
- `gráficos/mapa_intervalos.html` — mapa interactivo de las casas de test, coloreado por sub/sobre-
  valoración.
- `gráficos/mapa_intervalos_completo.html` — mismo mapa sobre todo el dataset (train + test, con
  las filas de train marcadas como diagnóstico, no como resultado con garantía estadística), con
  barra de filtros por precio, dormitorios, baños, superficie y comuna.
- `gráficos/*.png` — gráficos exploratorios (histogramas, correlaciones, dispersión).

## Automatización

Tarea de Windows Task Scheduler `CasasCrawlDiario`: corre `run_daily_crawl.bat` todas las noches a
las 22:00, con `WakeToRun` activado (despierta el PC desde suspensión, no desde apagado). Vive en
el computador personal (no en este, de trabajo) — no se crea ni se toca desde acá.

```powershell
# En el computador personal:
schtasks /query /tn "CasasCrawlDiario" /v /fo list
```

## Notas

- El botón "Siguiente" de Portal Inmobiliario trae `href=""` (paginación vía JS del lado del
  cliente). El spider construye la URL de la página siguiente él mismo con el patrón `_Desde_N` en
  vez de seguir ese link.
- `USER_AGENT` está fijado como Googlebot y `DOWNLOAD_DELAY=1.5` para evitar bloqueos — no bajar el
  delay sin verificar que el sitio lo sigue tolerando.
- El scrape actual (solo casas, 3 comunas) cubre ~8.900 resultados y toma varias horas sin tope de
  páginas.
- Comuna/barrio se imputan con vecino más cercano sobre `latitud`/`longitud` (`scipy.spatial.cKDTree`),
  no con matching difuso de texto.
- `*.json` y `crawl_log.txt` están en `.gitignore`: el output del scrape y los logs nunca se
  commitean.

Más detalle de arquitectura, decisiones de diseño y estado del proyecto para trabajar con Claude
Code en este repo: [CLAUDE.md](CLAUDE.md) y [PENDIENTES.md](PENDIENTES.md).
