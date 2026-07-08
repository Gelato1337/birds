-- =====================================================================
-- Fieldmark — multiplayer migration (run in Supabase SQL editor)
-- =====================================================================
-- Goal: each person has their OWN login (separate accounts), but everyone
-- can SEE everyone else:
--   * the leaderboard shows all players
--   * you can see which species other players have also collected
--   * you can see another player's total number of findings
-- Writes stay private: you can only insert/update/delete YOUR OWN rows.
--
-- This is read-shared, write-isolated. Safe for a small trusted group.
-- Run the whole file once. It is idempotent (safe to re-run).
-- =====================================================================

-- ---- make sure the columns the app writes actually exist ----
alter table logbook add column if not exists name  text;
alter table logbook add column if not exists finds int not null default 0;  -- total sightings count
alter table logbook add column if not exists bonus_points int not null default 0;  -- quest + co-op rewards (entry points are derived from sightings)

-- ---------------------------------------------------------------------
-- LOGBOOK: read-shared, write-own
-- ---------------------------------------------------------------------
drop policy if exists "logbook_owner_all"    on logbook;
drop policy if exists "logbook_read_all"     on logbook;
drop policy if exists "logbook_write_own"    on logbook;
drop policy if exists "logbook_update_own"   on logbook;
drop policy if exists "logbook_delete_own"   on logbook;

-- anyone logged in can read every logbook row (leaderboard + shared dex)
create policy "logbook_read_all" on logbook
  for select using (auth.role() = 'authenticated');
-- but you can only create/change/remove your own row
create policy "logbook_write_own" on logbook
  for insert with check (auth.uid() = owner);
create policy "logbook_update_own" on logbook
  for update using (auth.uid() = owner) with check (auth.uid() = owner);
create policy "logbook_delete_own" on logbook
  for delete using (auth.uid() = owner);

-- ---------------------------------------------------------------------
-- SIGHTINGS: read-shared (for the shared "recent findings" feed + co-op),
--            write-own
-- ---------------------------------------------------------------------
drop policy if exists "sightings_owner_all"  on sightings;
drop policy if exists "sightings_read_all"   on sightings;
drop policy if exists "sightings_write_own"  on sightings;
drop policy if exists "sightings_update_own" on sightings;
drop policy if exists "sightings_delete_own" on sightings;

create policy "sightings_read_all" on sightings
  for select using (auth.role() = 'authenticated');
create policy "sightings_write_own" on sightings
  for insert with check (auth.uid() = owner);
create policy "sightings_update_own" on sightings
  for update using (auth.uid() = owner) with check (auth.uid() = owner);
create policy "sightings_delete_own" on sightings
  for delete using (auth.uid() = owner);

-- helpful index for the cross-player "recent findings" feed
create index if not exists sightings_ts_idx on sightings (ts desc);

-- =====================================================================
-- AFTER RUNNING THIS:
--   * Create a second user under Authentication > Users (own email/pw).
--   * Each person signs in with their own credentials.
--   * Both still see each other on Ranks + shared journal markers.
-- =====================================================================
