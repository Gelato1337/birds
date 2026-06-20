// sw.js — Fieldmark service worker: caches the app so it works OFFLINE
// and installs to the home screen. Bump CACHE when you change files.

const CACHE = 'fieldmark-v1';
const SHELL = [
  './',
  './index.html',
  './fieldmark-sync.js',
  './manifest.webmanifest',
  './icon.svg',
  './yolo11n.onnx',          // the detector model (added once trained)
];

// install: pre-cache the app shell
self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(
      // addAll fails the whole install if one file 404s; cache individually
      // so a missing yolo11n.onnx (before training done) doesn't break install
      SHELL.map((u) => u)
    ).catch(() => {}))
  );
  self.skipWaiting();
});

// activate: drop old caches
self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// fetch strategy:
//  - Supabase / API calls: always go to network (never cache user data)
//  - everything else (app shell, model, fonts): cache-first, fall back to net
self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  // never cache Supabase or other API traffic
  if (url.hostname.includes('supabase') || url.pathname.includes('/rest/') ||
      url.pathname.includes('/auth/')) {
    return; // let it hit the network normally
  }
  e.respondWith(
    caches.match(e.request).then((cached) =>
      cached || fetch(e.request).then((resp) => {
        // cache successful GETs of static assets for next time (offline)
        if (e.request.method === 'GET' && resp.ok) {
          const copy = resp.clone();
          caches.open(CACHE).then((c) => c.put(e.request, copy));
        }
        return resp;
      }).catch(() => cached)
    )
  );
});