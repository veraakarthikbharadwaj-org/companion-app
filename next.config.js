/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverActions: true,
  },
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "avatars.githubusercontent.com",
        port: "",
        pathname: "**",
      },
      // Removed: tjzk.replicate.delivery, replicate.delivery, and a16z.com
      // These domains are NOT_IN_REGISTRY and do not meet foundation model
      // identity, version pinning, or integrity verification requirements.
    ],
  },
};

module.exports = nextConfig;
