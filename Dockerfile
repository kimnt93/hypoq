FROM node:22-alpine AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app/web
COPY web/package.json ./
RUN npm install

FROM node:22-alpine AS builder
WORKDIR /app/web
COPY --from=deps /app/web/node_modules ./node_modules
COPY web ./
RUN npm run build

FROM node:22-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

# Copy Next.js standalone application build only
# Data lake, SQLite database, templates, and research folders are mounted as volumes from host
COPY --from=builder --chown=node:node /app/web/.next/standalone ./
COPY --from=builder --chown=node:node /app/web/.next/static ./.next/static

USER node
EXPOSE 3000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://127.0.0.1:3000/api/healthz || exit 1

CMD ["node", "server.js"]
