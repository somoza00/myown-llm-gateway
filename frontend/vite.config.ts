import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// base '/ui/' alinha os assets com o StaticFiles mountado em /ui no gateway.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: '/ui/',
})