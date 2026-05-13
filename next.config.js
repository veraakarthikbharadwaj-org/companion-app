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
      {
        protocol: "https",
        hostname: "tjzk.replicate.delivery",
        port: "",
        pathname: "**",
      },
      // replicate.delivery (NOT_IN_REGISTRY) and a16z.com (NOT_IN_REGISTRY) removed.
      // These hosts are not in the approved model registry and used wildcard pathnames
      // without version pinning or integrity verification. Add only approved,
      // pinned paths here after registry approval.
    ],
  },
};

module.exports = nextConfig;
