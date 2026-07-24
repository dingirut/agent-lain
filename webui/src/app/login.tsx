// Full-screen password gate shown when the console is protected.

import { FormEvent, useState } from 'react'
import { ApiError, api } from '../lib/api'
import { PixelWordmark } from '../components/pixel'
import { Button, TextInput } from '../components/ui'

export default function LoginGate() {
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (!password || busy) return
    setBusy(true)
    setError(null)
    try {
      await api.post('/api/auth/login', { password })
      // Full reload: reconnects the websocket and refetches everything
      // with the fresh session cookie.
      location.reload()
    } catch (err) {
      const apiErr = err as ApiError
      setError(
        apiErr?.status === 429
          ? 'Too many attempts — wait a minute.'
          : apiErr?.status === 401
            ? 'Wrong password.'
            : apiErr?.message || 'Login failed.',
      )
      setBusy(false)
    }
  }

  return (
    <div className="flex h-dvh flex-col items-center justify-center gap-8 bg-page px-6">
      <PixelWordmark w={340} h={80} cell={5} gap={1.5} motion={false} />
      <form onSubmit={submit} className="flex w-full max-w-[320px] flex-col gap-3">
        <TextInput
          autoFocus
          type="password"
          autoComplete="current-password"
          aria-label="Console password"
          placeholder="Console password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          error={!!error}
          className="text-center font-mono"
        />
        {error && <div className="text-center text-[11.5px] text-err">{error}</div>}
        <Button variant="primary" type="submit" loading={busy} disabled={!password}>
          Sign in
        </Button>
      </form>
      <div className="font-mono text-[9.5px] text-faint">web console is password protected</div>
    </div>
  )
}
