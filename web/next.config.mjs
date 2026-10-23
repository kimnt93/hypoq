/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  serverExternalPackages: ["node:sqlite", "hyparquet", "hyparquet-compressors", "arquero"],
};

export default nextConfig;
