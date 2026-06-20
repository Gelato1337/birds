// =====================================================================
// fieldmark-sync.js — Supabase-backed storage with offline fallback.
// =====================================================================
// Replaces the app's old window.storage calls. Provides the SAME sget/sset
// interface the app already uses, but backed by Supabase (durable, synced
// across both phones) with a localStorage mirror so the app works OFFLINE
// and flushes to Supabase when back online.
//
// Security model: nothing syncs unless logged in. The login screen calls
// signIn(); until a session exists, reads/writes stay local-only and queue.
//
// USAGE in fieldmark.html (see fieldmark-supabase.patch.md):
//   <script type="module">
//   import { initSync, sget, sset, pushBoard, signIn, signOut, currentUser }
//     from './fieldmark-sync.js';
//   ...
//   await initSync();              // call before init()
//
// Fill these two from Supabase Settings > API. The anon key is safe to ship.
// =====================================================================
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const SUPABASE_URL = 'https://YOUR-PROJECT.supabase.co';   // <-- fill in
const SUPABASE_ANON_KEY = 'YOUR-ANON-PUBLIC-KEY';          // <-- fill in

const sb = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
let session = null;
let online = navigator.onLine;
window.addEventListener('online', () => { online = true; flushQueue(); });
window.addEventListener('offline', () => { online = false; });

// ---- local mirror (offline cache + write queue) ----
const LK = 'fieldmark_cache';
const QK = 'fieldmark_queue';
function localGet(k){ try{ return JSON.parse(localStorage.getItem(LK)||'{}')[k] ?? null; }catch{ return null; } }
function localSet(k,v){ try{ const c=JSON.parse(localStorage.getItem(LK)||'{}'); c[k]=v; localStorage.setItem(LK,JSON.stringify(c)); }catch{} }
function queueWrite(op){ try{ const q=JSON.parse(localStorage.getItem(QK)||'[]'); q.push(op); localStorage.setItem(QK,JSON.stringify(q)); }catch{} }

// ---- auth ----
export async function initSync(){
  const { data } = await sb.auth.getSession();
  session = data.session;
  sb.auth.onAuthStateChange((_e, s)=>{ session = s; if(s) flushQueue(); });
  return !!session;
}
export async function signIn(email, password){
  const { data, error } = await sb.auth.signInWithPassword({ email, password });
  if(error) throw error;
  session = data.session;
  await flushQueue();
  return session;
}
export async function signOut(){ await sb.auth.signOut(); session = null; }
export function currentUser(){ return session?.user ?? null; }
export function isOnline(){ return online && !!session; }

// ---- the sget/sset the app already uses ----
// keys: 'me', 'log'.  (the shared 'leaderboard' is handled by pushBoard)
export async function sget(key){
  // always try local first (instant, offline-safe), then refresh from cloud
  const local = localGet(key);
  if(session && online){
    try{
      if(key === 'log'){
        const { data } = await sb.from('logbook').select('*').limit(1).maybeSingle();
        if(data){ const v = { points:data.points, species:data.species,
                               cells:data.cells, quests:data.quests,
                               finds: local?.finds || [] };  // finds kept local + in sightings
          localSet(key, v); return v; }
      } else if(key === 'me'){
        const u = currentUser();
        if(u){ const v = local || { id:u.id, name:(u.email||'naturalist').split('@')[0] };
          localSet(key, v); return v; }
      }
    }catch(e){ /* fall back to local */ }
  }
  return local;
}

export async function sset(key, value){
  localSet(key, value);                 // mirror immediately (offline-safe)
  if(!session){ return; }               // not logged in -> stays local only
  const op = { key, value };
  if(!online){ queueWrite(op); return; }
  try{ await cloudWrite(op); }
  catch(e){ queueWrite(op); }
}

async function cloudWrite({ key, value }){
  const u = currentUser(); if(!u) return;
  if(key === 'log'){
    await sb.from('logbook').upsert({
      owner: u.id, points: value.points, species: value.species,
      cells: value.cells, quests: value.quests, updated_at: new Date().toISOString()
    }, { onConflict: 'owner' });
    // also push the newest sighting if present
    const last = value.finds?.[0];
    if(last && !last._synced){
      await sb.from('sightings').insert({
        owner: u.id, ts: new Date(last.t).toISOString(),
        lat: last.lat, lon: last.lon, species: last.species,
        via_sound: !!last.viaSound, thumb: last.thumb || null });
      last._synced = true; localSet('log', value);
    }
  }
  // 'me' needs no cloud write — identity comes from auth.
}

async function flushQueue(){
  if(!session || !online) return;
  let q; try{ q = JSON.parse(localStorage.getItem(QK)||'[]'); }catch{ q=[]; }
  if(!q.length) return;
  const rest = [];
  for(const op of q){ try{ await cloudWrite(op); }catch{ rest.push(op); } }
  localStorage.setItem(QK, JSON.stringify(rest));
}

// ---- leaderboard (shared) ----
export async function pushBoard(me, log){
  if(!session || !online) return;
  try{
    await sb.from('logbook').upsert({
      owner: currentUser().id, points: log.points, species: log.species,
      cells: log.cells, quests: log.quests, updated_at: new Date().toISOString()
    }, { onConflict: 'owner' });
  }catch(e){ /* queued via sset already */ }
}
export async function fetchBoard(){
  if(!session || !online) return [];
  try{
    const { data } = await sb.from('logbook').select('owner, points, species');
    return (data||[]).map(r=>({ id:r.owner, pts:r.points,
      species:Object.keys(r.species||{}).length, name:'naturalist' }));
  }catch{ return []; }
}

// ---- map sightings (durable, from cloud) ----
export async function fetchSightings(){
  if(session && online){
    try{ const { data } = await sb.from('sightings').select('*').order('ts',{ascending:false});
      return data||[]; }catch{}
  }
  return (localGet('log')?.finds) || [];
}