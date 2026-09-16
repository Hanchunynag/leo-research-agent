FROM node:22-alpine AS build
WORKDIR /web
COPY package.json package-lock.json ./
# Keep platform-specific native packages (for example lightningcss) in the
# Linux image even when the lockfile was generated on macOS.
RUN npm ci --include=optional
COPY . ./
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /web/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
