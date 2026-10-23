FROM node:22-alpine AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app
COPY web/package.json ./
RUN npm install

FROM node:22-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY web ./web
COPY autoresearch.db ./autoresearch.db
COPY data ./data
COPY researches ./researches
WORKDIR /app/web
RUN npm run build

FROM node:22-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

# Copy web files and standalone build
COPY --from=builder --chown=node:node /app/web/.next/standalone ./
COPY --from=builder --chown=node:node /app/web/.next/static ./.next/static

# Copy data, db and researches for backend endpoints
COPY --from=builder --chown=node:node /app/autoresearch.db ./autoresearch.db
COPY --from=builder --chown=node:node /app/data ./data
COPY --from=builder --chown=node:node /app/researches ./researches

USER node
EXPOSE 3000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://127.0.0.1:3000/api/healthz || exit 1

CMD ["node", "server.js"]
