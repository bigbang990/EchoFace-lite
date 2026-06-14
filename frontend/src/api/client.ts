export interface ApiClient {
  get: <T>(path: string) => Promise<T>
  post: <T>(path: string, body?: unknown) => Promise<T>
  patch: <T>(path: string, body?: unknown) => Promise<T>
  del: (path: string) => Promise<void>
}

export function createApiClient(baseUrl: string): ApiClient {
  const base = baseUrl.trim().replace(/\/$/, '')
  // ngrok free-tier shows a browser-warning page on first visit; this header skips it
  const isNgrok = base.includes('ngrok')
  const extraHeaders: Record<string, string> = isNgrok ? { 'ngrok-skip-browser-warning': 'true' } : {}

  async function req<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await fetch(`${base}${path}`, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        ...extraHeaders,
        ...(init?.headers ?? {}),
      },
    })
    if (!res.ok) {
      const msg = await res.text().catch(() => res.statusText)
      throw new Error(`${res.status}: ${msg}`)
    }
    if (res.status === 204) return undefined as T
    return res.json() as Promise<T>
  }

  return {
    get: <T>(path: string) => req<T>(path),
    post: <T>(path: string, body?: unknown) =>
      req<T>(path, {
        method: 'POST',
        body: body != null ? JSON.stringify(body) : undefined,
      }),
    patch: <T>(path: string, body?: unknown) =>
      req<T>(path, {
        method: 'PATCH',
        body: body != null ? JSON.stringify(body) : undefined,
      }),
    del: (path: string) => req<void>(path, { method: 'DELETE' }),
  }
}
