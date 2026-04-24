import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

const input = process.env.INPUT ?? "library.html";

export default defineConfig({
  plugins: [
    react(),
    viteSingleFile({
      useRecommendedBuildConfig: false,
    }),
  ],
  base: "./",
  server: {
    host: "0.0.0.0",
    port: 5174,
    strictPort: true,
  },
  build: {
    assetsDir: "",
    assetsInlineLimit: () => true,
    chunkSizeWarningLimit: 100000000,
    cssCodeSplit: false,
    outDir: "dist",
    emptyOutDir: false,
    sourcemap: "inline",
    minify: false,
    cssMinify: false,
    rollupOptions: {
      input,
      output: {
        codeSplitting: false,
      },
    },
  },
});
