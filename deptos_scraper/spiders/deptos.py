# https://docs.scrapy.org/en/latest/intro/tutorial.html

import re

import scrapy

PAGE_SIZE = 48
DESDE_RE = re.compile(r'_Desde_\d+')
COORDS_RE = re.compile(r'center=(-?[\d.]+)%2C(-?[\d.]+)')


class DeptosSpider(scrapy.Spider):
    name = "deptos"

    def start_requests(self):
        urls = ('https://www.portalinmobiliario.com/venta/casa/vitacura-metropolitana/',
                'https://www.portalinmobiliario.com/venta/casa/las-condes-metropolitana',
                'https://www.portalinmobiliario.com/venta/casa/lo-barnechea-metropolitana'
        )

        for url in urls:
            yield scrapy.Request(url, self.parse)


    def parse(self, response):
        detail_links = response.css('a.poly-component__title::attr(href)').getall()
        for url in detail_links:
            yield response.follow(url=url, callback=self.parse_links)

        # El botón "Siguiente" trae href="" (paginación es client-side JS), así
        # que construimos la página siguiente nosotros mismos vía "_Desde_N".
        # Sin más resultados en esta página, se detiene solo.
        if detail_links:
            current_offset_match = DESDE_RE.search(response.url)
            current_offset = int(current_offset_match.group().split('_')[-1]) if current_offset_match else 1
            next_offset = current_offset + PAGE_SIZE
            base_url = DESDE_RE.sub('', response.url).rstrip('/')
            yield response.follow(f'{base_url}_Desde_{next_offset}', callback=self.parse)

    def parse_links(self, response):
        data_table = {}
        for table in response.css('table'): # extraemos información desde la tabla
            for row in table.css('tr'):
                key = row.css('th div::text').get()
                value = row.css('td span::text').get()
                if key == None: key = ''
                if value == None: value = ''
                data_table[key.strip()] = value.strip()

        # La ficha trae un mapa estático de Google (img data-testid="static-map") cuyo src incluye
        # las coordenadas del centro del mapa, ej. "...&center=-33.3746734%2C-70.4990592&...".
        # No hay lat/lon en ningún otro lado de la página (no hay JSON-LD ni data-attributes con
        # "latitude"), así que se extraen por regex de esa URL.
        static_map_src = response.css('img[data-testid="static-map"]::attr(src)').get()
        coords_match = COORDS_RE.search(static_map_src) if static_map_src else None
        latitud, longitud = coords_match.groups() if coords_match else (None, None)

        for text in response.css('body'): # extraemos información básica
            data =  {
                'titulo': text.css('.ui-pdp-title::text').get(),
                'precio': text.css('.andes-money-amount__fraction::text').get(),
                'clp': text.css('.ui-pdp-family--REGULAR .andes-money-amount__fraction::text').get(),
                'dirección': text.css('.ui-vip-location .ui-pdp-media__title ::text').get(),
                'comuna': text.css('div.ui-pdp-breadcrumb>nav>ol>li>a::attr(title)').getall()[-2],
                'barrio': text.css('div.ui-pdp-breadcrumb>nav>ol>li>a::attr(title)').getall()[-1],
                'latitud': latitud,
                'longitud': longitud,
                'descripcion': text.css('.ui-pdp-description__content::text').get(),
                'url': response.url
            }
            data.update(data_table)
            yield data
