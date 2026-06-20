-- =====================================================================
-- Fieldmark — Supabase schema + security (run in Supabase SQL editor)
-- =====================================================================
-- Model: a single shared login (you + your gf use the same account).
-- Security is enforced by Supabase Auth + Row Level Security (RLS):
-- nothing reads or writes unless the request carries a valid logged-in
-- session. The passphrase in the app is just a UI convenience; THIS is
-- the real gate.
--
-- SETUP (one time):
-- 1. Create a project at supabase.com (free tier).
-- 2. Authentication > Providers: enable Email. For a 2-person app, the
--    simplest is to create ONE user under Authentication > Users
--    (set email + password) and share those credentials.
--    (Turn OFF "Confirm email" under Auth settings so the login just works.)
-- 3. Paste this whole file into SQL Editor and run it.
-- 4. Settings > API: copy the Project URL and the 'anon public' key into
--    fieldmark.html (SUPABASE_URL / SUPABASE_ANON_KEY). The anon key is
--    safe to ship publicly — RLS is what protects the data.
-- =====================================================================

-- ---- app state: one row per logbook (shared account => effectively one row) ----
create table if not exists logbook (
  id          uuid primary key default gen_random_uuid(),
  owner       uuid not null references auth.users(id) default auth.uid(),
  points      int  not null default 0,
  species     jsonb not null default '{}'::jsonb,  -- {species_key: count}
  cells       jsonb not null default '{}'::jsonb,  -- visited 1km cells
  quests      jsonb not null default '{}'::jsonb,
  updated_at  timestamptz not null default now()
);

-- ---- individual sightings (for the map + species detail) ----
create table if not exists sightings (
  id          uuid primary key default gen_random_uuid(),
  owner       uuid not null references auth.users(id) default auth.uid(),
  ts          timestamptz not null default now(),
  lat         double precision,
  lon         double precision,
  species     jsonb not null,            -- array of species keys in this photo
  via_sound   boolean not null default false,
  thumb       text                       -- base64 jpeg (optional; can be null)
);

-- ---- labeling: hand-drawn boxes, so your gf can label from home and you
--      pull them down on the training machine ----
create table if not exists labels (
  stem        text primary key,          -- image id, matches dataset filename
  split       text not null,             -- 'train' | 'val'
  lines       jsonb not null,            -- array of "cls cx cy w h" strings
  labeled_by  uuid not null references auth.users(id) default auth.uid(),
  updated_at  timestamptz not null default now()
);

-- ---------------------------------------------------------------------
-- Row Level Security — the actual protection.
-- Every table: only an authenticated user may touch rows. Because you
-- share one account, you both see the same data; an unauthenticated
-- request sees NOTHING.
-- ---------------------------------------------------------------------
alter table logbook   enable row level security;
alter table sightings enable row level security;
alter table labels    enable row level security;

-- logbook: owner can do everything with their own row
create policy "logbook_owner_all" on logbook
  for all using (auth.uid() = owner) with check (auth.uid() = owner);

-- sightings: same
create policy "sightings_owner_all" on sightings
  for all using (auth.uid() = owner) with check (auth.uid() = owner);

-- labels: any authenticated user may read/write (shared labeling effort).
-- (With a single shared account this is equivalent to owner-only anyway.)
create policy "labels_auth_read"  on labels for select
  using (auth.role() = 'authenticated');
create policy "labels_auth_write" on labels for insert
  with check (auth.role() = 'authenticated');
create policy "labels_auth_update" on labels for update
  using (auth.role() = 'authenticated');

-- helpful index for pulling unlabeled-by-time
create index if not exists labels_updated_idx on labels (updated_at);