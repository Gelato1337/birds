# Setting up the backend (Supabase)

Fieldmark runs entirely in the browser. The backend is only there to **sync
sightings across devices** and drive the **shared leaderboard**. If you just want
to try the app, you can skip all of this and use **"Continue without logging in"**
on the sign-in screen — everything is saved locally on the device and nothing is
synced.

To get login + sync + leaderboard, you need a free Supabase project and the three
SQL scripts in this folder.

## 1. Create a project

1. Sign up at [supabase.com](https://supabase.com) and create a new project (free tier is enough).
2. **Authentication → Providers**: enable **Email**.
3. **Authentication → Users**: create your account(s) (email + password).
   Under Auth settings, turn **off** "Confirm email" so sign-in just works.
   - Two people sharing one account? Create one user and share the credentials.
   - Everyone their own account? Create one user each (see script 2 below).

## 2. Run the SQL scripts

Open **SQL Editor** in the Supabase dashboard and run these **in order**. Each is
idempotent — safe to re-run.

| # | File | What it does | Required? |
|---|------|--------------|-----------|
| 1 | [`01_schema.sql`](01_schema.sql) | Core tables (`logbook`, `sightings`) + Row Level Security. Also the `labels` / `label_items` tables used by the dev labeling pipeline. | **Yes** |
| 2 | [`02_multiplayer.sql`](02_multiplayer.sql) | Switches the leaderboard/dex to **read-shared, write-own** so separate accounts can see each other. Adds the `name`, `finds`, `bonus_points` columns the app writes. | Only if you want **separate accounts** that see each other. Skip for a single shared account. |
| 3 | [`03_points_and_stats.sql`](03_points_and_stats.sql) | DB-side scoring: `species_rarity` + `rarity_points` tables and the `player_stats` view that computes points from the `sightings` table (the app's source of truth for points online). | Recommended. Without it the app falls back to computing points locally. |

## 3. Point the app at your project

In [`../index.html`](../index.html), fill in your project's API details (from
**Settings → API** in the dashboard):

```js
const SUPABASE_URL      = 'https://YOUR-PROJECT.supabase.co';
const SUPABASE_ANON_KEY = 'YOUR-ANON-PUBLIC-KEY';
```

The **anon key is safe to ship publicly** — Row Level Security is what protects the
data. Nothing reads or writes without a valid logged-in session; an unauthenticated
request sees nothing.

## Notes

- **Security model.** RLS is the real gate. `01_schema.sql` locks every table to
  authenticated users; `02_multiplayer.sql` loosens *reads* to all logged-in users
  while keeping *writes* owner-only.
- **The `labels` / `label_items` tables** in script 1 are only used by the dev
  labeling/training pipeline (`dev/`), not by the public app. Harmless to leave in.
- **Local-only mode needs none of this.** The "Continue without logging in" button
  keeps all data in the device's `localStorage` and never contacts Supabase.
