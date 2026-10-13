# External open datasets

All external data used by the submissions are listed here. The challenge data themselves are
not redistributed (see the main README).

| File | Content | Source | Licence | Used by |
| :-- | :-- | :-- | :-- | :-- |
| `metar_raw.parquet` | METAR/ASOS reports of the ten airports, 2025-2026 | [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/request/download.phtml), Iowa State University (`src/download_metar.py`) | freely available public data | `src/`, `archive/` |
| `metar_hourly.parquet` | hourly visibility, ceiling, fog, freezing fog, snow and gust flags derived from `metar_raw.parquet` | derived (`src/features_metar.py`) | as above | `src/`, `archive/` |
| `weather_hourly.parquet` | hourly temperature, snowfall, wind and de-icing risk, merged with the METAR features | [Open-Meteo historical weather API](https://open-meteo.com/) (`src/download_weather.py`, `src/merge_metar_weather.py`) | CC BY 4.0 | `src/`, `archive/` |
| `osm_taxiway_distances.parquet` | shortest taxi-route and straight-line distance and tortuosity for 3,844 stand-runway pairs at the ten airports | derived from © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors (`src/features_osm_taxiway.py --all`) | ODbL 1.0 | `src/`, `archive/` |
| `airports.csv` | airport coordinates, elevation and country | [OurAirports](https://github.com/davidmegginson/ourairports-data) | Public domain (Unlicense) | `archive/` |
| `runways.csv` | runway length, width, heading, elevation and displaced thresholds | [OurAirports](https://github.com/davidmegginson/ourairports-data) | Public domain (Unlicense) | `archive/` |
| `aircraft_types.csv` | aircraft type characteristics (dimensions, wing area, weights, speeds, range, fuel capacity) for 591 ICAO type designators | *What is flying?* database by Julien Broutier, [whatisflying-db](https://github.com/laegsgaardTroels/whatisflying-db) (`data/aircraft_types.csv`; the original repository `jbroutier/whatisflying-db` is no longer online) | ODbL 1.0 | `archive/` |

Notes:

- `aircraft_types.csv`, `airports.csv` and `runways.csv` were read by `attach_external_features()` in
  `archive/features.py`. They were used by earlier submissions (for example `v14` and `v54`) that
  are part of the blend `v104`, the anchor of the final submission. They are not used by `src/`.
- `aircraft_types.csv` is an unmodified copy of the file in the mirror above (591 rows, all
  identical). It is redistributed under the Open Database License v1.0
  (<https://opendatacommons.org/licenses/odbl/1-0/>).
- `osm_taxiway_distances.parquet` is a database derived from OpenStreetMap and is therefore made
  available under the Open Database License v1.0. Map data © OpenStreetMap contributors.
- Small lookup tables used by some features (airline alliances, home hubs, national holidays,
  runway headings, RECAT-EU categories by aircraft type) are written directly in the code.
