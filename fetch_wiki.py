"""
fetch_wiki.py — fetch Wikipedia intros for every species in species.json.

Robust version: uses the MediaWiki Query API (not the strict REST summary
endpoint, which 404s on redirect titles). For each species it tries, in order:
  1. English query API by scientific binomial (auto-follows redirects)
  2. English query API by common name
  3. English search API -> best title -> query API
  4. Finnish query API by scientific binomial
  5. Finnish search API -> best title -> query API
This resolves redirects (Cinclus cinclus -> White-throated dipper) and
disambiguation far better, and pulls the intro extract in the SAME call.

  pip install requests tqdm
  python fetch_wiki.py

Output: wiki/<taxonKey>.json per species + wiki_all.json + fetch_report.txt
"""
import json, re, html, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

SPECIES_JSON = "species.json"
OUT = Path("wiki"); OUT.mkdir(exist_ok=True)
WORKERS = 8
TIMEOUT = 20
UA = {"User-Agent": "FieldmarkBot/1.0 (https://github.com/fieldmark; jmlaat@utu.fi) python-requests"}

EN_API = "https://en.wikipedia.org/w/api.php"
FI_API = "https://fi.wikipedia.org/w/api.php"


def clean_sci(sci):
    toks = re.sub(r"[(),]", " ", sci).split()
    out = []
    for t in toks:
        if out and (t[0].isupper() or t[0].isdigit()):
            break
        out.append(t)
        if len(out) == 2:
            break
    return " ".join(out) if out else sci


def query_extract(title, api):
    """MediaWiki query API: resolve redirects + return intro extract in one call."""
    try:
        r = requests.get(api, headers=UA, timeout=TIMEOUT, params={
            "action": "query", "format": "json", "redirects": 1,
            "titles": title, "prop": "extracts|pageimages|info",
            "exintro": 1, "explaintext": 1, "inprop": "url",
            "piprop": "thumbnail", "pithumbsize": 400,
        })
        pages = r.json().get("query", {}).get("pages", {})
        for pid, page in pages.items():
            if pid == "-1" or "missing" in page:
                return None
            extract = (page.get("extract") or "").strip()
            if not extract or len(extract) < 40:
                return None
            # skip disambiguation-ish stubs
            if "may refer to" in extract[:80].lower():
                return None
            return {
                "wiki_title": page.get("title"),
                "wiki_url": page.get("fullurl"),
                "lang": "en" if api == EN_API else "fi",
                "extract": extract,
                "thumbnail": (page.get("thumbnail") or {}).get("source"),
            }
    except Exception:
        return None
    return None


def search_then_query(query, api):
    """Search API -> top title -> query_extract that title."""
    try:
        r = requests.get(api, headers=UA, timeout=TIMEOUT, params={
            "action": "query", "list": "search", "srsearch": query,
            "srlimit": 1, "format": "json"})
        hits = r.json().get("query", {}).get("search", [])
        if hits:
            return query_extract(hits[0]["title"], api)
    except Exception:
        pass
    return None


def first_two_sentences(text):
    text = html.unescape(text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:2]).strip()


def resolve(sp):
    sci = clean_sci(sp["sci"])
    name = sp["name"]
    # 1-2: English query API by sci, then common name
    for cand in (sci, name):
        if cand:
            res = query_extract(cand, EN_API)
            if res:
                return res
    # 3: English search
    for q in (sci, f"{name} bird" if sp["coarse"] == "bird" else f"{name} mammal", name):
        res = search_then_query(q, EN_API)
        if res:
            return res
    # 4: Finnish query API by sci (often present when EN search misses)
    res = query_extract(sci, FI_API)
    if res:
        return res
    # 5: Finnish search by sci
    res = search_then_query(sci, FI_API)
    if res:
        return res
    return None


def fetch_one(sp):
    key = re.sub(r"[^a-z0-9]+", "_", sp["name"].lower()).strip("_")
    rec = {"key": key, "name": sp["name"], "sci": sp["sci"],
           "coarse": sp["coarse"], "taxonKey": sp.get("taxonKey"),
           "obs": sp.get("obs"), "wiki_title": None, "wiki_url": None,
           "lang": None, "summary_2s": None, "extract": None, "thumbnail": None}
    res = resolve(sp)
    if res:
        rec.update(res)
        rec["summary_2s"] = first_two_sentences(res["extract"])
    (OUT / f"{sp.get('taxonKey') or key}.json").write_text(
        json.dumps(rec, indent=2, ensure_ascii=False))
    return rec


def _probe():
    """Run ONE request loudly so real errors (403, network, JSON) are visible
    instead of being silently swallowed by the per-species try/except."""
    import sys
    try:
        r = requests.get(EN_API, headers=UA, timeout=TIMEOUT, params={
            "action":"query","format":"json","redirects":1,
            "titles":"Cinclus cinclus","prop":"extracts","exintro":1,"explaintext":1})
        print(f"[probe] status={r.status_code}")
        if r.status_code != 200:
            print(f"[probe] body: {r.text[:300]}")
            print("[probe] ^ Wikipedia is rejecting requests. Usually User-Agent "
                  "or a proxy/firewall. Fix this before the full run.")
            sys.exit(1)
        pages = r.json().get("query",{}).get("pages",{})
        ok = any(p.get("extract") for p in pages.values())
        print(f"[probe] extract found: {ok}")
        if not ok:
            print("[probe] got 200 but no extract — API param issue.")
            sys.exit(1)
    except Exception as e:
        print(f"[probe] request FAILED: {type(e).__name__}: {e}")
        sys.exit(1)
    print("[probe] OK — Wikipedia reachable, proceeding.\n")


def main():
    _probe()
    species = json.loads(Path(SPECIES_JSON).read_text())
    print(f"{len(species)} species to fetch")
    results, misses = [], []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_one, sp): sp for sp in species}
        for f in tqdm(as_completed(futs), total=len(futs), desc="wikipedia"):
            rec = f.result()
            results.append(rec)
            if not rec["extract"]:
                misses.append(rec["name"])

    results.sort(key=lambda r: -(r.get("obs") or 0))
    (OUT.parent / "wiki_all.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    hit = sum(1 for r in results if r["extract"])
    report = [f"fetched {hit}/{len(results)}", f"missing ({len(misses)}):"] + \
             [f"  - {m}" for m in misses]
    (OUT.parent / "fetch_report.txt").write_text("\n".join(report))
    print(f"\n✅ {hit}/{len(results)} fetched. {len(misses)} missing "
          f"(see fetch_report.txt).")


if __name__ == "__main__":
    main()