# deptos compra

Scraper en Scrapy para casas en venta en Vitacura, Las Condes y Lo Barnechea (Portal Inmobiliario), más un script de análisis exploratorio / imputación que prepara los datos para un modelo de predicción de precio.

## Requisitos

Python 3.12. No hay `requirements.txt` en el repo; las dependencias usadas son:

```
scrapy pandas numpy matplotlib seaborn scikit-learn fuzzywuzzy
```

`fuzzywuzzy` (usado por `proyecto_deptos.py` para hacer matching difuso de comuna/barrio) no está instalado por defecto — instálalo antes de correr ese script:

```bash
pip install fuzzywuzzy
```

## Uso

```bash
# Correr el spider (sobreescribe el output)
python -m scrapy crawl deptos -O deptos.json

# Verificar que el spider carga sin errores
python -m scrapy check deptos

# Pipeline completo (crawl + análisis), lo mismo que corre la tarea programada
run_daily_crawl.bat
```

`proyecto_deptos.py` no es un script idempotente de punta a punta: a mitad de camino deja de leer `deptos.json` y pasa a leer `arriendos_clean.csv`, un checkpoint de una imputación lenta que no se regenera solo y no está commiteado. Sin ese archivo, el script no corre limpio más allá de ese punto.

## Estructura

```
deptos_scraper/
  spiders/deptos.py   # único spider ("deptos")
  settings.py          # USER_AGENT, DOWNLOAD_DELAY, etc.
proyecto_deptos.py      # EDA, limpieza, imputación, prep para regresión
run_daily_crawl.bat      # crawl -> deptos.json -> proyecto_deptos.py
```

## Automatización

Tarea de Windows Task Scheduler `DeptosCrawlDiario`: corre `run_daily_crawl.bat` todas las noches a las 22:00, con `WakeToRun` activado (despierta el PC desde suspensión, no desde apagado).

```powershell
schtasks /query /tn "DeptosCrawlDiario" /v /fo list
```

## Notas

- El botón "Siguiente" de Portal Inmobiliario trae `href=""` (paginación vía JS del lado del cliente). El spider construye la URL de la página siguiente él mismo con el patrón `_Desde_N` en vez de seguir ese link.
- `USER_AGENT` está fijado como Googlebot y `DOWNLOAD_DELAY=1.5` para evitar bloqueos — no bajar el delay sin verificar que el sitio lo sigue tolerando.
- El scrape actual (solo casas, 3 comunas) cubre ~8.900 resultados y toma varias horas sin tope de páginas.
- `*.json` y `crawl_log.txt` están en `.gitignore`: el output del scrape y los logs nunca se commitean.

Más detalle de arquitectura para trabajar con Claude Code en este repo: [CLAUDE.md](CLAUDE.md).
