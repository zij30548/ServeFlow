import { fileURLToPath } from 'node:url'
import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'

const frontendRoot = fileURLToPath(new URL('.', import.meta.url))

export function resolveBackendOrigin(value) {
  if (value === undefined) return 'http://127.0.0.1:8816'
  try {
    if (typeof value !== 'string' || !value || /\s|\\|[?#]/.test(value)) throw new Error()
    const origin = new URL(value)
    if (!['http:', 'https:'].includes(origin.protocol) || !origin.hostname ||
        origin.username || origin.password || origin.pathname !== '/') throw new Error()
    return origin.origin
  } catch {
    throw new Error('SERVEFLOW_BACKEND_ORIGIN must be an HTTP(S) origin without credentials, path, query or fragment.')
  }
}

export function createFrontendConfig({ mode = 'development' } = {}) {
  const env = loadEnv(mode, frontendRoot, 'SERVEFLOW_')
  const proxy = {
    '/api/python': {
      target: resolveBackendOrigin(env.SERVEFLOW_BACKEND_ORIGIN),
      changeOrigin: true,
      rewrite: path => path.replace(/^\/api\/python/, '')
    }
  }
  return {
    plugins: [vue()],
    // Server configuration never becomes import.meta.env browser configuration.
    envPrefix: [],
    server: { host: '127.0.0.1', port: 5173, strictPort: true, proxy },
    preview: { host: '127.0.0.1', strictPort: true, proxy }
  }
}

export default defineConfig(createFrontendConfig)
