import { useCallback, useEffect, useState } from 'react'

interface LogEntry {
  id: number
  timestamp: string
  provider: string
  model: string
  input_tokens: number
  output_tokens: number
  total_tokens: number
  latency_ms: number
  estimated_cost: string
  status: string
  error_type: string | null
}

const KEY_STORAGE = 'llm_gateway_ui_key'
const POLL_MS = 10_000

function StatusBadge({ status, errorType }: { status: string; errorType: string | null }) {
  const isOk = status === 'ok'
  const cls = isOk
    ? 'bg-emerald-500/15 text-emerald-300 ring-emerald-500/30'
    : 'bg-red-500/15 text-red-300 ring-red-500/30'
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ${cls}`}
    >
      <span
        className={`mr-1.5 h-1.5 w-1.5 rounded-full ${isOk ? 'bg-emerald-400' : 'bg-red-400'}`}
      />
      {isOk ? 'ok' : errorType ?? 'error'}
    </span>
  )
}

function fmtTime(iso: string) {
  const d = new Date(iso)
  return d.toLocaleString('pt-BR', { timeZoneName: 'short' })
}

function fmtMs(ms: number) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${ms.toFixed(0)}ms`
}

export default function App() {
  const [key, setKey] = useState<string>(() => localStorage.getItem(KEY_STORAGE) ?? '')
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [error, setError] = useState<string | null>(null)
  const [lastFetch, setLastFetch] = useState<string | null>(null)

  const saveKey = useCallback((value: string) => {
    const trimmed = value.trim()
    setKey(trimmed)
    if (trimmed) localStorage.setItem(KEY_STORAGE, trimmed)
    else localStorage.removeItem(KEY_STORAGE)
  }, [])

  const load = useCallback(async () => {
    if (!key) return
    try {
      const res = await fetch('/api/logs?limit=200', {
        headers: { Authorization: `Bearer ${key}` },
      })
      if (!res.ok) {
        const body = await res.json().catch(() => null)
        setError(`Falha ao carregar logs (HTTP ${res.status}) — ${body?.error?.message ?? ''}`)
        return
      }
      const data = (await res.json()) as { logs: LogEntry[] }
      setLogs(data.logs)
      setError(null)
      setLastFetch(new Date().toLocaleTimeString('pt-BR'))
    } catch (err) {
      setError(`Erro de rede: ${String(err)}`)
    }
  }, [key])

  useEffect(() => {
    if (!key) return
    load()
    const id = setInterval(load, POLL_MS)
    return () => clearInterval(id)
  }, [key, load])

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <header className="border-b border-zinc-800 px-6 py-5">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold text-zinc-50">LLM Gateway</h1>
            <p className="text-sm text-zinc-400">Logs de requisições (atualiza a cada 10s)</p>
          </div>
          <div className="flex items-center gap-3">
            {key && (
              <span className="text-xs text-zinc-400">
                última atualização: {lastFetch ?? '...'}
              </span>
            )}
            <input
              type="password"
              value={key}
              onChange={(e) => saveKey(e.target.value)}
              placeholder="chave virtual (Bearer)"
              className="w-64 rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:border-zinc-500 focus:outline-none"
            />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-6">
        {!key ? (
          <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-8 text-center text-sm text-zinc-400">
            Cole uma chave virtual para ver os logs —{' '}
            <code className="text-zinc-300">docker compose exec gateway llm-gateway create-key</code>
          </div>
        ) : error ? (
          <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
            {error}
            <div className="mt-1 text-xs text-red-400">
              Revise a chave e o status do gateway (verifique se a migração rodou).
            </div>
          </div>
        ) : (
          <div className="overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-zinc-800 bg-zinc-900/80 text-xs uppercase tracking-wide text-zinc-500">
                <tr>
                  <th className="px-4 py-3 font-medium">Timestamp</th>
                  <th className="px-4 py-3 font-medium">Modelo</th>
                  <th className="px-4 py-3 text-right font-medium">Tokens</th>
                  <th className="px-4 py-3 text-right font-medium">Latência</th>
                  <th className="px-4 py-3 font-medium">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800">
                {logs.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="px-4 py-10 text-center text-zinc-500">
                      Nenhuma requisição registrada ainda. Faça uma chamada ao gateway.
                    </td>
                  </tr>
                ) : (
                  logs.map((log) => (
                    <tr key={log.id} className="hover:bg-zinc-800/40">
                      <td className="whitespace-nowrap px-4 py-2.5 text-zinc-300">
                        {fmtTime(log.timestamp)}
                      </td>
                      <td className="px-4 py-2.5">
                        <span className="text-zinc-100">{log.model || '—'}</span>
                        {log.provider && (
                          <span className="ml-2 text-xs text-zinc-500">{log.provider}</span>
                        )}
                      </td>
                      <td className="px-4 py-2.5 text-right tabular-nums text-zinc-300">
                        {log.status === 'ok' ? log.total_tokens : '—'}
                      </td>
                      <td className="px-4 py-2.5 text-right tabular-nums text-zinc-300">
                        {fmtMs(log.latency_ms)}
                      </td>
                      <td className="px-4 py-2.5">
                        <StatusBadge status={log.status} errorType={log.error_type} />
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        )}
      </main>
    </div>
  )
}