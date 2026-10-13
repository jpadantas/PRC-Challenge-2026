"""
Build OSM taxiway distance lookup using direct OSM API /api/0.6/map.
Splits large airport bboxes into smaller tiles to stay under 50k node limit.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import requests, json, math, re, xml.etree.ElementTree as ET, time
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
CACHE_DIR = EXTERNAL_DIR / "osm_cache"
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

AIRPORTS = {
    "EDDF": {
        "bbox": (49.997, 8.518, 50.073, 8.650),
        "runways": {"07C":(50.0297,8.5412),"07R":(50.0189,8.5489),"25C":(50.0316,8.6261),"25L":(50.0226,8.6330),"18":(50.0160,8.5988),"36":(50.0567,8.5988)}
    },
    "EDDM": {
        "bbox": (48.330, 11.746, 48.378, 11.838),
        "runways": {"08L":(48.3583,11.7534),"08R":(48.3397,11.7562),"26L":(48.3579,11.8302),"26R":(48.3392,11.8321)}
    },
    "EGLL": {
        "bbox": (51.457, -0.498, 51.493, -0.222),
        "runways": {"09L":(51.4775,-0.4830),"09R":(51.4619,-0.4818),"27L":(51.4774,-0.2343),"27R":(51.4619,-0.2328)}
    },
    "EHAM": {
        "bbox": (52.278, 4.722, 52.358, 4.798),
        "runways": {"06":(52.3047,4.7528),"24":(52.3311,4.7872),"09":(52.3141,4.7394),"27":(52.3141,4.8431),"18L":(52.3508,4.7353),"36R":(52.2776,4.7355),"18R":(52.3508,4.8077),"36C":(52.2776,4.8077)}
    },
    "LEBL": {"bbox":(41.275,2.064,41.306,2.113),"runways":{"07L":(41.2833,2.0701),"07R":(41.2944,2.0765),"25L":(41.2938,2.1077),"25R":(41.2829,2.1073)}},
    "LEMD": {"bbox":(40.436,-3.624,40.518,-3.542),"runways":{"14L":(40.5129,-3.5936),"14R":(40.5052,-3.5739),"32L":(40.4447,-3.5720),"32R":(40.4378,-3.5877),"18L":(40.5145,-3.5749),"36R":(40.4468,-3.5730)}},
    "LFPG": {"bbox":(49.001,2.512,49.039,2.588),"runways":{"08L":(49.0218,2.5268),"08R":(49.0101,2.5161),"09L":(49.0278,2.5527),"09R":(49.0123,2.5543),"26L":(49.0272,2.5847),"26R":(49.0121,2.5845),"27L":(49.0218,2.5659),"27R":(49.0099,2.5776)}},
    "LIRF": {"bbox":(41.782,12.202,41.828,12.268),"runways":{"16L":(41.8222,12.2442),"16R":(41.8240,12.2617),"34L":(41.7932,12.2604),"34R":(41.7924,12.2434),"25":(41.8011,12.2063)}},
    "LSZH": {"bbox":(47.447,8.534,47.483,8.588),"runways":{"10":(47.4600,8.5419),"28":(47.4637,8.5826),"14":(47.4813,8.5574),"32":(47.4598,8.5558),"16":(47.4806,8.5614),"34":(47.4600,8.5608)}},
    "LTFM": {"bbox":(41.252,28.692,41.308,28.783),"runways":{"09L":(41.2764,28.6987),"09R":(41.2659,28.6987),"27L":(41.2762,28.7803),"27R":(41.2657,28.7803)}},
}

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    dphi = math.radians(lat2-lat1); dlam = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlam/2)**2
    return 2*R*math.asin(math.sqrt(a))

def fetch_osm_tile(west, south, east, north, retry=5):
    """Fetch OSM XML for a tile. Returns parsed XML root or None."""
    cache_path = CACHE_DIR / f"tile_{south:.4f}_{west:.4f}_{north:.4f}_{east:.4f}.xml"
    if cache_path.exists():
        return ET.parse(str(cache_path)).getroot()
    url = f"https://api.openstreetmap.org/api/0.6/map?bbox={west},{south},{east},{north}"
    for attempt in range(retry):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                cache_path.write_bytes(r.content)
                return ET.fromstring(r.content)
            elif r.status_code == 400 and "too many nodes" in r.text.lower():
                return "TOO_LARGE"
            elif r.status_code == 509:
                m = re.search(r"try again in (\d+) seconds?", r.text, re.I)
                wait = int(m.group(1)) + 5 if m else 65
                print(f"    HTTP 509: rate-limited, waiting {wait}s ({r.text[:80]})")
                time.sleep(wait)
            else:
                print(f"    HTTP {r.status_code}: {r.text[:100]}")
                time.sleep(2**attempt)
        except Exception as e:
            print(f"    Error: {e}"); time.sleep(2**attempt)
    return None

def fetch_airport_osm(icao, bbox):
    """Fetch OSM for full airport, splitting into tiles if needed."""
    south, west, north, east = bbox
    cache_path = CACHE_DIR / f"{icao}_merged.json"
    if cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
        if "runway_ways" in data:
            return data
        # Old-format cache missing runway geometry: rebuild by re-parsing.
        # Tiles are already on disk so this re-reads from cache, no network calls.
        print(f"  [{icao}] Cache missing runway geometry, rebuilding from cached tiles...")

    # Start with a 2x2 tile grid, expand if needed
    all_nodes = {}
    all_ways_raw = []
    parking_positions = {}
    runway_ways = []

    def parse_root(root):
        for node in root.findall("node"):
            nid = int(node.get("id"))
            lat, lon = float(node.get("lat")), float(node.get("lon"))
            all_nodes[nid] = (lat, lon)
            aeroway = ref = None
            for tag in node.findall("tag"):
                k, v = tag.get("k"), tag.get("v")
                if k == "aeroway": aeroway = v
                if k in ("ref","name"): ref = v
            if aeroway in ("parking_position","gate") and ref:
                parking_positions[str(ref)] = (lat, lon)
        for way in root.findall("way"):
            aeroway = None
            rwy_ref = None
            for tag in way.findall("tag"):
                k, v = tag.get("k"), tag.get("v")
                if k == "aeroway": aeroway = v
                if k in ("ref","name") and rwy_ref is None: rwy_ref = v
            if aeroway in ("taxiway","taxilane","runway"):
                nd_ids = [int(nd.get("ref")) for nd in way.findall("nd")]
                all_ways_raw.append(nd_ids)
                if aeroway == "runway" and rwy_ref and nd_ids:
                    runway_ways.append((rwy_ref, nd_ids[0], nd_ids[-1]))

    def split_and_fetch(s, w, n, e, depth=0):
        if depth > 6:
            print(f"    Max depth reached, skipping tile")
            return
        result = fetch_osm_tile(w, s, e, n)
        if result == "TOO_LARGE":
            # Split into 2 halves
            mid_lat = (s + n) / 2
            mid_lon = (w + e) / 2
            if (n-s) >= (e-w):  # Split horizontally
                split_and_fetch(s, w, mid_lat, e, depth+1)
                split_and_fetch(mid_lat, w, n, e, depth+1)
            else:
                split_and_fetch(s, w, n, mid_lon, depth+1)
                split_and_fetch(s, mid_lon, n, e, depth+1)
        elif result is not None:
            parse_root(result)
            time.sleep(0.5)  # Be polite to OSM servers
        else:
            print(f"    Tile fetch failed permanently: bbox=({w},{s},{e},{n})")

    print(f"  [{icao}] Fetching OSM tiles...")
    split_and_fetch(south, west, north, east)
    print(f"  [{icao}] Nodes: {len(all_nodes)}, Ways: {len(all_ways_raw)}, Stands: {len(parking_positions)}")

    # Derive per-designator runway threshold coordinates from OSM runway way
    # geometry (ref like "16L/34R", way endpoints = the two thresholds).
    runway_coords_osm = {}
    for ref, n_first, n_last in runway_ways:
        parts = [p.strip() for p in ref.split("/") if p.strip()]
        if len(parts) != 2:
            continue
        if n_first in all_nodes and parts[0] not in runway_coords_osm:
            runway_coords_osm[parts[0]] = all_nodes[n_first]
        if n_last in all_nodes and parts[1] not in runway_coords_osm:
            runway_coords_osm[parts[1]] = all_nodes[n_last]
    print(f"  [{icao}] Runway thresholds from OSM: {sorted(runway_coords_osm.keys())}")

    merged = {"nodes": all_nodes, "ways": all_ways_raw, "parking": parking_positions,
              "runway_ways": runway_ways, "runway_coords": runway_coords_osm}
    # Convert int keys to str for JSON serialization
    merged["nodes"] = {str(k): v for k, v in all_nodes.items()}
    if len(all_nodes) == 0:
        print(f"  [{icao}] WARNING: 0 nodes fetched, not caching (will retry next run)")
    else:
        with open(cache_path, "w") as f:
            json.dump(merged, f)
    return merged

def build_graph(merged):
    all_nodes = {int(k): v for k, v in merged["nodes"].items()}
    G = nx.Graph()
    for nid, (lat, lon) in all_nodes.items():
        G.add_node(nid, lat=lat, lon=lon)
    for way_nodes in merged["ways"]:
        valid = [n for n in way_nodes if n in all_nodes]
        for i in range(len(valid)-1):
            a, b = valid[i], valid[i+1]
            d = haversine_m(all_nodes[a][0], all_nodes[a][1], all_nodes[b][0], all_nodes[b][1])
            G.add_edge(a, b, length=d)
    # Keep largest component
    if len(G.nodes) > 0:
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return G

def nearest_node(G, lat, lon):
    best, bd = None, float("inf")
    for nid, d in G.nodes(data=True):
        dist = haversine_m(lat, lon, d["lat"], d["lon"])
        if dist < bd: best, bd = nid, dist
    return best

def fuzzy_match(name, parking):
    name = str(name).strip()

    # 1. Exact match
    if name in parking: return parking[name]

    # 2. Strip leading zeros in numeric suffix (A01 -> A1, A09 -> A9)
    stripped = re.sub(r'(?<=[A-Za-z])0+(\d)', r'\1', name)
    if stripped in parking: return parking[stripped]

    # 3. Add leading zero (A1 -> A01, B9 -> B09)
    padded = re.sub(r'(?<=[A-Za-z])(\d)(?!\d)', r'0\1', name)
    if padded in parking: return parking[padded]

    # 4. Shared stand match: OSM "A09/A10" contains our "A09" or "A10"
    for osm_ref, pos in parking.items():
        parts = re.split(r'[/|,\s]+', osm_ref)
        norm_parts = [re.sub(r'(?<=[A-Za-z])0+(\d)', r'\1', p.strip()) for p in parts]
        norm_name = re.sub(r'(?<=[A-Za-z])0+(\d)', r'\1', name)
        if norm_name in norm_parts or name in parts:
            return pos

    # 5. Pure numeric stand (e.g. EGLL: "508" -> OSM "508")
    if name.isdigit():
        for osm_ref, pos in parking.items():
            if osm_ref.strip() == name or osm_ref.strip().lstrip("0") == name.lstrip("0"):
                return pos

    return None


def process_airport(icao, our_stands, our_runways):
    config = AIRPORTS[icao]
    merged = fetch_airport_osm(icao, config["bbox"])
    G = build_graph(merged)
    parking = merged["parking"]
    # Prefer runway thresholds derived from actual OSM runway geometry over the
    # hand-typed fallback coords (which can go stale, e.g. magnetic-variation
    # runway renumbering, or be wrong for airports we guessed coordinates for).
    rwy_coords = {**config["runways"], **merged.get("runway_coords", {})}
    print(f"  [{icao}] Graph: {len(G.nodes)} nodes, {len(G.edges)} edges | Stands OSM: {len(parking)}")

    rwy_nodes = {}
    for rwy, pos in rwy_coords.items():
        if rwy in our_runways:
            rwy_nodes[rwy] = nearest_node(G, pos[0], pos[1])

    results = []
    n_matched = 0
    for stand in our_stands:
        pos = fuzzy_match(stand, parking)
        if pos is None: continue
        n_matched += 1
        s_node = nearest_node(G, pos[0], pos[1])
        for rwy, r_node in rwy_nodes.items():
            sl = haversine_m(pos[0], pos[1], rwy_coords[rwy][0], rwy_coords[rwy][1])
            try:
                dist = nx.shortest_path_length(G, s_node, r_node, weight="length")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                dist = sl * 1.35
            results.append({"adep":icao,"stand":stand,"runway":rwy,
                            "osm_dist_m":round(dist,1),"sl_dist_m":round(sl,1),
                            "tortuosity":round(dist/max(sl,1.0),3)})

    print(f"  [{icao}] Matched {n_matched}/{len(our_stands)} stands -> {len(results)} routes")
    return pd.DataFrame(results)

if __name__ == "__main__":
    import argparse, polars as pl
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--airport", default="EDDM")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    RAW_DIR = PROJECT_ROOT / "data" / "raw"
    df_raw = pl.read_parquet(RAW_DIR / "ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")

    airports_to_run = list(AIRPORTS.keys()) if args.all else [args.airport]
    all_results = []

    for icao in airports_to_run:
        apt = df_raw.filter(pl.col("ADEP_mvt") == icao)
        our_stands = apt["STAND_mvt"].drop_nulls().cast(str).unique().to_list()
        our_runways = [r for r in apt["RUNWAY_mvt"].drop_nulls().cast(str).unique().to_list() if r != "NA"]
        df_apt = process_airport(icao, our_stands, our_runways)
        if len(df_apt) > 0:
            if args.test:
                print(df_apt.sort_values("osm_dist_m")[["stand","runway","osm_dist_m","sl_dist_m","tortuosity"]].head(10).to_string(index=False))
            all_results.append(df_apt)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        out = EXTERNAL_DIR / "osm_taxiway_distances.parquet"
        combined.to_parquet(out, index=False)
        print(f"\nSaved {len(combined):,} routes -> {out}")
        train_dep = pl.read_parquet(RAW_DIR / "ranking.parquet").filter(
            pl.col("PHASE_mvt") == "DEP"
        ).select(["ADEP_mvt","STAND_mvt","RUNWAY_mvt","TAXITIME_SEC_mvt"]).drop_nulls()
        cpl = pl.from_pandas(combined).rename({"adep":"ADEP_mvt","stand":"STAND_mvt","runway":"RUNWAY_mvt"})
        joined = train_dep.join(cpl, on=["ADEP_mvt","STAND_mvt","RUNWAY_mvt"], how="inner")
        if len(joined) > 500:
            taxi = joined["TAXITIME_SEC_mvt"].to_numpy()
            dist = joined["osm_dist_m"].to_numpy()
            corr = np.corrcoef(taxi, dist)[0,1]
            print(f"VALIDATION: corr(osm_dist_m, taxi_time) = r={corr:.4f}  n={len(joined):,}")
