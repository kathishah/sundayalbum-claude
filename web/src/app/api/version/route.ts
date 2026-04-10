import { NextResponse } from 'next/server'

const OWNER = 'kathishah'
const REPO = 'sundayalbum-claude'
const GH_BASE = 'https://api.github.com'
const GH_HEADERS = {
  Accept: 'application/vnd.github+json',
  'X-GitHub-Api-Version': '2022-11-28',
}

// In-memory cache — shared across requests within an App Runner instance.
// 10-minute TTL keeps GitHub API usage well under the 60 req/hour unauthenticated limit.
const CACHE_TTL_MS = 10 * 60 * 1000

interface DeployInfo {
  deployed_at: string | null
  sha: string | null
  is_latest: boolean | null
}

interface CacheEntry {
  data: { web: DeployInfo; backend: DeployInfo }
  ts: number
}

let cache: CacheEntry | null = null

const stage = process.env.NEXT_PUBLIC_STAGE ?? 'dev'
const branch = stage === 'prod' ? 'main' : 'dev'

async function ghFetch(path: string): Promise<unknown> {
  const res = await fetch(`${GH_BASE}${path}`, { headers: GH_HEADERS })
  if (!res.ok) throw new Error(`GitHub API ${path} → ${res.status}`)
  return res.json()
}

async function getLastSuccessfulRun(
  workflow: string
): Promise<{ deployed_at: string; sha: string } | null> {
  const data = (await ghFetch(
    `/repos/${OWNER}/${REPO}/actions/workflows/${workflow}/runs?branch=${branch}&status=success&per_page=1`
  )) as { workflow_runs?: { updated_at: string; head_sha: string }[] }
  const run = data.workflow_runs?.[0]
  if (!run) return null
  return { deployed_at: run.updated_at, sha: run.head_sha }
}

// Returns the ISO date of the most recent commit touching any of the given paths.
async function latestCommitDate(paths: string[]): Promise<string | null> {
  const results = await Promise.allSettled(
    paths.map((p) =>
      ghFetch(
        `/repos/${OWNER}/${REPO}/commits?sha=${branch}&path=${encodeURIComponent(p)}&per_page=1`
      ).then((d) => {
        const commits = d as { commit: { committer: { date: string } } }[]
        return commits[0]?.commit.committer.date ?? null
      })
    )
  )
  const dates = results
    .filter((r): r is PromiseFulfilledResult<string | null> => r.status === 'fulfilled')
    .map((r) => r.value)
    .filter((d): d is string => d !== null)
  if (dates.length === 0) return null
  return dates.sort().at(-1)! // ISO strings sort lexicographically
}

function isUpToDate(deployedAt: string | null, latestCommit: string | null): boolean | null {
  if (!deployedAt || !latestCommit) return null
  return new Date(latestCommit) <= new Date(deployedAt)
}

export async function GET() {
  // Serve from cache if fresh
  if (cache && Date.now() - cache.ts < CACHE_TTL_MS) {
    return NextResponse.json(cache.data)
  }

  try {
    const [webRun, backendRun, webCommitDate, backendCommitDate] = await Promise.all([
      getLastSuccessfulRun('deploy-web.yml'),
      getLastSuccessfulRun('deploy-lambda.yml'),
      latestCommitDate(['web']),
      // api + src cover the vast majority of meaningful backend changes
      latestCommitDate(['api', 'src']),
    ])

    const data = {
      web: {
        deployed_at: webRun?.deployed_at ?? null,
        sha: webRun?.sha.slice(0, 7) ?? null,
        is_latest: isUpToDate(webRun?.deployed_at ?? null, webCommitDate),
      },
      backend: {
        deployed_at: backendRun?.deployed_at ?? null,
        sha: backendRun?.sha.slice(0, 7) ?? null,
        is_latest: isUpToDate(backendRun?.deployed_at ?? null, backendCommitDate),
      },
    }

    cache = { data, ts: Date.now() }
    return NextResponse.json(data)
  } catch (err) {
    console.error('Version check failed:', err)
    // Return stale cache if available, otherwise error
    if (cache) return NextResponse.json(cache.data)
    return NextResponse.json({ error: 'unavailable' }, { status: 503 })
  }
}
