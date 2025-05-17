# Football Data Scraper

## 📊 Descripción
Una potente biblioteca de Python para extraer y analizar datos de fútbol de múltiples fuentes como Transfermarkt, FBRef y FotMob. Diseñada para proporcionar acceso programático a estadísticas detalladas de jugadores, equipos y partidos.

## 🚀 Características
- Extracción de datos de múltiples fuentes de fútbol:
  - Transfermarkt: Información de transferencias y valor de mercado
  - FBRef: Estadísticas detalladas de jugadores y equipos
  - FotMob: Estadísticas y eventos de partidos
- Funciones de visualización integradas
- Manejo de excepciones robusto
- Documentación completa de la API

## 🛠️ Instalación

```bash
pip install football-data-scraper
```

## 📋 Requisitos
- Python 3.x
- BeautifulSoup4
- Pandas
- Requests
- Matplotlib (para visualizaciones)

## 🔧 Uso Básico

```python
from football_data_scraper import TransfermarktScraper, FBRefScraper

# Inicializar scrapers
transfermarkt = TransfermarktScraper()
fbref = FBRefScraper()

# Obtener datos de un jugador
jugador_data = transfermarkt.get_player_data(name="Lionel Messi", player_id="28003")

# Obtener estadísticas de un partido
partido_stats = fbref.get_match_data('https://www.365scores.com/es/football/match/premier-league-7 aston-villa-tottenham-109-114-7#id=4147381')
```

## ⚠️ Notas Importantes
- Respetar los términos de servicio de las fuentes de datos
- Mantener intervalos razonables entre solicitudes
- Verificar la disponibilidad de datos antes de su uso

## 🤝 Contribuir
Las contribuciones son bienvenidas. Por favor:
1. Haz fork del proyecto
2. Crea una rama para tu característica (`git checkout -b feature/AmazingFeature`)
3. Realiza tus cambios y haz commit (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia
Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos
- Transfermarkt, FBRef y FotMob por proporcionar datos valiosos
- La comunidad de análisis de datos de fútbol por su continuo apoyo