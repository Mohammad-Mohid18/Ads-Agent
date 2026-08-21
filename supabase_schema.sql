-- Run this once in the Supabase SQL Editor for this project.
-- The API stores the full VideoProject payload in project_data and duplicates a
-- few fields below for simple filtering and reporting.

create table if not exists public.video_projects (
  id uuid primary key,
  source_url text,
  aspect_ratio text,
  status text not null default 'processing',
  error text,
  project_data jsonb not null,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create index if not exists video_projects_status_idx on public.video_projects (status);
create index if not exists video_projects_created_at_idx on public.video_projects (created_at desc);

alter table public.video_projects enable row level security;

-- No public policies are created. The backend uses SUPABASE_SERVICE_ROLE_KEY,
-- which bypasses RLS; never expose that key in a browser/frontend application.

-- Normalized copies of each generated component. project_data remains the
-- complete snapshot; these tables make scripts, voice data, assets and renders
-- easy to inspect independently.
create table if not exists public.ad_brand_assets (
  project_id uuid primary key references public.video_projects(id) on delete cascade,
  asset_data jsonb not null,
  updated_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.ad_scripts (
  project_id uuid not null references public.video_projects(id) on delete cascade,
  version integer not null,
  duration double precision not null,
  scenes jsonb not null,
  updated_at timestamptz not null default timezone('utc', now()),
  primary key (project_id, version)
);

create table if not exists public.ad_voiceovers (
  project_id uuid not null references public.video_projects(id) on delete cascade,
  version integer not null,
  total_duration double precision not null,
  segments jsonb not null,
  updated_at timestamptz not null default timezone('utc', now()),
  primary key (project_id, version)
);

create table if not exists public.ad_renders (
  project_id uuid not null references public.video_projects(id) on delete cascade,
  version integer not null,
  storage_path text not null,
  preview_url text not null,
  layers jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default timezone('utc', now()),
  primary key (project_id, version)
);

alter table public.ad_brand_assets enable row level security;
alter table public.ad_scripts enable row level security;
alter table public.ad_voiceovers enable row level security;
alter table public.ad_renders enable row level security;

-- Public bucket so Creatomate can fetch images/audio via clean
-- /storage/v1/object/public/ad-assets/... URLs (no signed ?token= query strings).
insert into storage.buckets (id, name, public)
values ('ad-assets', 'ad-assets', true)
on conflict (id) do update set public = excluded.public;
