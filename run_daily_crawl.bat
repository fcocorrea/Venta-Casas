@echo off
cd /d "C:\Users\FranciscoCorrea\OneDrive - Paragon Private Advisors\Mi Pc\Escritorio\PROYECTO\Venta-Deptos"
".\venv\Scripts\python.exe" -m scrapy crawl casas -O casas.json >> crawl_log.txt 2>&1
".\venv\Scripts\python.exe" proyecto_casas.py >> crawl_log.txt 2>&1
