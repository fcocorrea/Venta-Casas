@echo off
cd /d "C:\Users\franc\OneDrive\Escritorio\Proyectos\deptos compra"
C:\Users\franc\AppData\Local\Programs\Python\Python312\python.exe -m scrapy crawl deptos -O deptos.json >> crawl_log.txt 2>&1
C:\Users\franc\AppData\Local\Programs\Python\Python312\python.exe proyecto_deptos.py >> crawl_log.txt 2>&1
