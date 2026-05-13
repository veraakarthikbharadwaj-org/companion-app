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
      // tjzk.replicate.delivery (NOT_IN_REGISTRY) removed.
      // This host is not in the approved model registry and used a wildcard pathname
      // without path-level restriction, version pinning, or integrity verification.
      // Add only approved, pinned paths here after registry approval.
      // replicate.delivery (NOT_IN_REGISTRY) and a16z.com (NOT_IN_REGISTRY) removed.
      // These hosts are not in the approved model registry and used wildcard pathnames
      // without version pinning or integrity verification. Add only approved,
      // pinned paths here after registry approval.
    ],
  },
};

module.exports = nextConfig;
