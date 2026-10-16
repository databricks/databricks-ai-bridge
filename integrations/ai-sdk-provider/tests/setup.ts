import fs from 'node:fs'
import path from 'node:path'
import { config as loadEnv } from 'dotenv'

process.env.NODE_ENV = 'test'

const envLocalPath = path.resolve(process.cwd(), '.env.local')

// Load local developer secrets for opt-in live tests without affecting CI defaults.
if (fs.existsSync(envLocalPath)) {
  loadEnv({ path: envLocalPath, quiet: true })
}
