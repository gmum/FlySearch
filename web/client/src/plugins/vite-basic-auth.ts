import type {PluginOption} from 'vite';
import auth from 'basic-auth';

export default function vitePluginHttpBasicAuth(users: { username?: string, password: string }[],
                                                config?: {realm?: string, useInServer?: boolean, useInPreview?: boolean}): PluginOption {
  console.log(`[vite-plugin-http-basic-auth] Initializing with ${users.length} users`)

  // Check if users are defined
  if (!users || users.length === 0) {
    console.warn('[vite-plugin-http-basic-auth] No users defined for vite-plugin-http-basic-auth!')
  }

  users.filter((user, index) => {
    if (!user.username || !user.password) {
      console.warn(`[vite-plugin-http-basic-auth] User on index ${index} does not have defined username and/or password! Ignoring.`)
      return false
    }
    return true;
  })

  function authMiddleware(req, res, next) {
    const user = auth(req);
    if (!user || !user.name || !user.pass) {
      return unauthorized(res, config?.realm || 'Authentication required');
    }
    if (!users.find(u => u.username === user.name && u.password === user.pass)) {
      return unauthorized(res, config?.realm || 'Authentication required');
    }
    next();
  }

  return {
    name: 'vite-plugin-http-basic-auth',
    enforce: 'pre',
    apply: 'serve',

    configureServer(server) {
      if (config?.useInServer === undefined || config?.useInServer)
        server.middlewares.use(authMiddleware);
    },

    configurePreviewServer(server) {
      if (config?.useInServer === undefined || config?.useInPreview)
        server.middlewares.use(authMiddleware);
    },
  }
}

function unauthorized(res, realm: string) {
  res.statusCode = 401;
  res.setHeader('WWW-Authenticate', 'Basic realm="' + realm + '"');
  res.end('Unauthorized');
}
