// Fieldmark service worker — network-first for code, cache-first for big assets.
// You should NOT need to bump this version for normal code changes anymore:
// HTML/JS/JSON are fetched fresh from the network when online, so deploys show
// up immediately. The cache is only a fallback for offline use.
const CACHE = 'fieldmark-runtime';

// big, rarely-changing assets worth caching aggressively (cache-first)
const STATIC = ['./fieldmark-s-v1.onnx', './icon.svg', './manifest.webmanifest'];

self.addEventListener('install', e => {
  self.skipWaiting();   // activate new SW immediately
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(STATIC).catch(()=>{})));
});

self.addEventListener('activate', e => {
  // drop any old caches from the previous versioned scheme
  e.waitUntil(
    caches.keys().then(keys => Promise.all(
      keys.filter(k => k !== CACHE).map(k => caches.delete(k))
    )).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  // never touch Supabase or other cross-origin APIs
  if (url.origin !== location.origin) return;

  const isStatic = STATIC.some(s => url.pathname.endsWith(s.replace('./','')));

  if (isStatic) {
    // cache-first for the model/icons (big, stable)
    e.respondWith(
      caches.match(e.request).then(hit => hit || fetch(e.request).then(res => {
        const copy = res.clone();
        caches.open(CACHE).then(c => c.put(e.request, copy));
        return res;
      }))
    );
  } else {
    // network-first for everything else (HTML, JS, JSON) — always fresh online,
    // cached copy only if offline.
    e.respondWith(
      fetch(e.request).then(res => {
        const copy = res.clone();
        caches.open(CACHE).then(c => c.put(e.request, copy));
        return res;
      }).catch(() => caches.match(e.request))
    );
  }
});