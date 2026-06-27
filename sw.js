// Fieldmark service worker.
// Strategy, by asset type:
//   - MODEL + icons      cache-first   (big, rarely changes)
//   - data JSON          stale-while-revalidate (instant load, refresh in bg)
//   - HTML / JS / other  network-first (deploys show up immediately)
//   - Supabase / APIs    never touched (user data must never be cached)
// Bump CACHE to force-drop everything from older versions (e.g. the old v1 model).
const CACHE = 'fieldmark-v17';

// big, rarely-changing assets worth caching aggressively (cache-first)
const STATIC = ['./fieldmark-s-v3.onnx', './icon.svg', './manifest.webmanifest'];

// data files: heavy-ish but read every load. Serve cached instantly, refresh
// in the background so the next load is fresh. NEVER includes user data —
// user data only lives in Supabase, which we skip entirely below.
const DATA = ['./species.json', './wiki_species.json', './content.json'];

self.addEventListener('install', e => {
  self.skipWaiting();   // activate new SW immediately
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(STATIC).catch(()=>{})));
});

self.addEventListener('activate', e => {
  // drop any old caches (old version names, the v1 model, stale html)
  e.waitUntil(
    caches.keys().then(keys => Promise.all(
      keys.filter(k => k !== CACHE).map(k => caches.delete(k))
    )).then(() => self.clients.claim())
  );
});

const endsWithAny = (path, list) => list.some(s => path.endsWith(s.replace('./','')));

self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  const url = new URL(e.request.url);
  // never touch Supabase or any other cross-origin API — user data stays uncached
  if (url.origin !== location.origin) return;

  if (endsWithAny(url.pathname, STATIC)) {
    // cache-first for the model/icons (big, stable)
    e.respondWith(
      caches.match(e.request).then(hit => hit || fetch(e.request).then(res => {
        const copy = res.clone();
        caches.open(CACHE).then(c => c.put(e.request, copy));
        return res;
      }))
    );
    return;
  }

  if (endsWithAny(url.pathname, DATA)) {
    // stale-while-revalidate: answer from cache immediately (no lag),
    // and update the cache in the background for next time.
    e.respondWith(
      caches.open(CACHE).then(c => c.match(e.request).then(hit => {
        const net = fetch(e.request).then(res => { c.put(e.request, res.clone()); return res; })
                                    .catch(() => hit);
        return hit || net;
      }))
    );
    return;
  }

  // network-first for everything else (HTML, JS) — always fresh online,
  // cached copy only as an offline fallback.
  e.respondWith(
    fetch(e.request).then(res => {
      const copy = res.clone();
      caches.open(CACHE).then(c => c.put(e.request, copy));
      return res;
    }).catch(() => caches.match(e.request))
  );
});
