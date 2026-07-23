# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Scrapy project that scrapes house listings ("casa") for sale in three Santiago comunas (Vitacura, Las Condes, Lo Barnechea) from Portal Inmobiliario, plus a full modeling pipeline (`proyecto_casas.py`) that cleans the scrape, engineers features, cross-validates, trains a price model with prediction intervals, and produces a ranked export and interactive map of under/over-priced listings. There is no test suite or linter in this repo — everything below reflects the actual current state, not aspirational tooling.

## Commands

```bash
# Run the spider, overwrite output
./venv/Scripts/python.exe -m scrapy crawl casas -O casas.json

# Sanity-check the spider loads (no contracts defined, just checks for import/setup errors)
./venv/Scripts/python.exe -m scrapy check casas

# Debug a specific issue at DEBUG log level, capped so it doesn't run for hours
./venv/Scripts/python.exe -m scrapy crawl casas -L DEBUG -s CLOSESPIDER_ITEMCOUNT=50

# Full analysis: cleaning, features, CV, model, intervals, ranking, export, map. Takes several
# minutes (repeated CV + nested feature-selection CV + hyperparameter search), not seconds.
./venv/Scripts/python.exe proyecto_casas.py

# Full nightly pipeline (what the scheduled task runs)
run_daily_crawl.bat   # crawl -> casas.json, then proyecto_casas.py, both appended to crawl_log.txt
```

Dependencies are frozen in `requirements.txt` (`./venv/Scripts/python.exe -m pip install -r requirements.txt`) — a single venv covers both the scraper and the analysis/modeling script. Historically the scraper ran on a separate global Python install and the analysis script needed its own venv (the global install lacked scikit-learn); that split no longer applies now that scrapy is installed into the project venv alongside everything else.

### Scheduled task (Windows Task Scheduler)

The nightly automation is documented as `CasasCrawlDiario` (daily at 22:00, `WakeToRun` enabled — wakes the PC from sleep, but not from a full shutdown), but **it lives on a different machine** — the user's personal computer (Windows profile `franc`), not this one (a work machine, profile `FranciscoCorrea`). `schtasks /query` for it returns "not found" here, and that's expected: this repo is checked out on both machines, but the scheduled crawl only runs on the personal one. The old `.bat` had a `C:\Users\franc\...` path matching that machine, not this one. Do not "fix" that path assuming migration — `run_daily_crawl.bat` here is updated to this machine's actual path for manual/dev runs, but the personal machine's copy of the task points at its own path independently.

```powershell
# On the personal machine only:
schtasks /query /tn "CasasCrawlDiario" /v /fo list

# schtasks /change hangs waiting on an interactive password prompt in this environment
# (even though the task's logon type is "run only when user is logged on" and needs no
# stored credential) — recreate instead of changing:
schtasks /create /tn "CasasCrawlDiario" /tr '"<path>\run_daily_crawl.bat"' /sc daily /st HH:MM /f
```

## Architecture

**`casas_scraper/spiders/casas.py`** is the only spider (`name = "casas"`). Two-phase crawl:
1. `start_requests` seeds 3 category/comuna listing URLs (casa only — departamento URLs were deliberately removed).
2. `parse()` extracts detail-page links from `a.poly-component__title`, then paginates.

**Pagination is hand-rolled, not link-following** — this is the non-obvious part. Portal Inmobiliario's "Siguiente" button renders with `href=""` (its pagination is client-side JS with no real link in the server HTML), so following it naively deduplicates against the current page and silently stops after page 1. The fix: `parse()` regex-matches `_Desde_\d+` on the current URL, computes `current_offset + PAGE_SIZE (48)`, and builds the next page URL itself (`..._Desde_49`, `..._Desde_97`, ...). It stops only when a page yields zero detail links — there is no reliance on the total-results counter or any pagination widget. Do not "fix" this back to following the Siguiente link.

**`parse_links()`** (the detail-page callback) scrapes two things per listing: a free-form spec `<table>` (key/value pairs merged directly into the item dict — field names vary listing to listing, e.g. `Superficie útil`, `Dormitorios`, `Precio estacionamiento desde (UF)`) and fixed CSS selectors for title/price/comuna/barrio (via the breadcrumb, `getall()[-2]`/`[-1]`). Items are yielded as plain dicts, not `scrapy.Item` — `items.py`/`pipelines.py` are untouched Scrapy scaffolding and are not wired into `settings.py` (`ITEM_PIPELINES` is unset).

**`casas_scraper/settings.py`** pins `USER_AGENT` to Googlebot's UA and `DOWNLOAD_DELAY = 1.5` — both required, empirically, to avoid being blocked; `ROBOTSTXT_OBEY = True` is safe to leave on since Portal Inmobiliario's robots.txt disallows `/propiedades/` and search-filter params but not `/venta/`.

**Volume**: at the time of writing, the 3 casa branches totaled ~8,900 listings combined, and a full uncapped crawl runs ~9+ hours at the current `DOWNLOAD_DELAY`. This is why the schedule is a 22:00 overnight run rather than anything more frequent.

**`proyecto_casas.py`** is a Jupyter-notebook-style script (exported cell-by-cell, not written as a reusable pipeline) that does EDA, outlier detection, missing-value imputation, feature engineering, and price prediction with uncertainty intervals on the scrape output. It reads only `casas.json` end-to-end (no intermediate checkpoint file) and runs clean on a fresh crawl. `comuna`/`barrio` nulls are imputed with a nearest-neighbor lookup on `latitud`/`longitud` (`scipy.spatial.cKDTree`), not fuzzy text matching. All of PASO 2b-4j (everything that fits a parameter from the data — imputation, encoders, scalers, distance/density features) is written as `ajustar_*()`/`aplicar_*()` pairs so it can be refit per fold; PASO 5a's repeated cross-validation (`ejecutar_cv_repetida`) calls these directly instead of reusing the single 80/20 split's fitted parameters. A nested version of that same CV (`medir_importancia_anidada`) drives feature selection too — a column is only dropped if it shows no permutation-importance signal in 100% of folds, which trimmed the feature set from 82 to 75 columns (see `PENDIENTES.md`, P3).

Price intervals (PASO 5e) come from conformalized quantile regression (three `HistGradientBoostingRegressor(loss='quantile')` models at q=0.05/0.50/0.95, calibrated **per price segment** — group-conditional/Mondrian conformal prediction, not a single global correction — because a single correction under-covered the most expensive deciles) rather than a fixed ±% threshold — a fixed threshold was measured to flag 43% of the inventory, indistinguishable from noise. A listing is flagged only if its real price falls outside its own predicted interval (PASO 5f), ranked by distance to the interval edge, not by raw residual.

It writes `casas_limpios.xlsx` (full cleaned dataset), `casas_candidatas.xlsx` (test-set listings flagged as outside their predicted interval — under- **or** over-market, ranked by distance to the edge; listings that fall inside their interval are excluded, and train-set listings are excluded because their predictions are in-sample), plus the charts under `gráficos/` and two interactive maps (`gráficos/mapa_intervalos.html` for test only, `gráficos/mapa_intervalos_completo.html` for the full dataset with a filter toolbar — price range, comuna, dormitorios/baños/superficie minimums, all client-side JS/Leaflet, no external services) — all committed to the repo and overwritten on each run.

## Data flow

```
scrapy crawl casas -O casas.json   (gitignored, one run's worth of raw listings)
        |
        v
proyecto_casas.py   (EDA -> cleaning -> lat/lon imputation -> feature engineering -> CV ->
                      quantile regression intervals -> flag/rank flagged listings)
        |
        v
casas_limpios.xlsx, casas_candidatas.xlsx, gráficos/*.png, gráficos/mapa_*.html
        (committed, overwritten each run)
```

`*.json` and `crawl_log.txt` are gitignored — scrape output and run logs are never committed.
