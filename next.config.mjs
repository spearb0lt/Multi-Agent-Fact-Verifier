/** @type {import('next').NextConfig} */

/**
 * The frontend is exported as static files and served by the Python process.
 *
 * That is one decision with several consequences worth having on purpose.
 * There is one origin, so the event stream needs no CORS and the password
 * header behaves normally. There is one process to deploy, so Render, Docker
 * and an Oracle VM all take the same start command. And there is no build time
 * API address baked into the bundle, which is what made the proxy in the
 * earlier version point at whatever happened to be on port 8000.
 *
 * Every page is a client component that fetches at runtime, so exporting
 * static HTML costs nothing: there is no server rendering to give up.
 */
const nextConfig = {
  output: "export",
  reactStrictMode: true,
  // The export has no image optimiser behind it, and this project ships no
  // images anyway.
  images: { unoptimized: true },
  // Static hosting serves /run/index.html for /run, which is what a plain file
  // server expects to find.
  trailingSlash: true,
};

export default nextConfig;
