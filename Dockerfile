# GrimmBot — Dockerfile
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

# 1. Swap to Debian to completely bypass the Ubuntu Snap crash.
# 2. Use 'chromium' instead of 'chromium-browser' (Debian naming).
# 3. Add 'dos2unix' to fix Windows line-ending bugs.
# 4. Install 'default-jdk' (Java 17) to maintain autonomous compiling.
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb x11vnc fluxbox xdotool scrot novnc websockify \
    wget gnupg2 ca-certificates \
    build-essential git curl unzip zip dos2unix \
    default-jdk \
    python3 python3-pip python3-dev python3-tk python3-venv \
    pandoc wkhtmltopdf ffmpeg imagemagick poppler-utils \
    tzdata dbus-x11 xclip \
    fonts-liberation fonts-noto-color-emoji \
    libfuse2 wmctrl \
    chromium \
    && rm -rf /var/lib/apt/lists/*

# Manually install Gradle for the agent
ARG GRADLE_VERSION=8.12
RUN wget -q "https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip" -O /tmp/gradle.zip \
    && unzip -q /tmp/gradle.zip -d /opt \
    && ln -s "/opt/gradle-${GRADLE_VERSION}/bin/gradle" /usr/local/bin/gradle \
    && rm /tmp/gradle.zip

# Setup the sandbox user
RUN groupadd -r grimmbot && useradd -r -g grimmbot -G audio,video -m -d /home/grimmbot grimmbot

WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/workspace /app/wormhole \
    /app/data/profiles /app/data/scheduler /app/data/custom_tools \
    /home/grimmbot/.config/chromium/Default /tmp/.X11-unix \
    && chmod 1777 /tmp/.X11-unix \
    && chown -R grimmbot:grimmbot /app /home/grimmbot

# Copy requirements and utilize BuildKit pip cache
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --break-system-packages -r requirements.txt

# Copy the startup script and fix Windows line-endings (\r\n -> \n)
COPY start.sh .
RUN dos2unix start.sh && chmod +x start.sh

# Copy your actual codebase LAST so coding changes don't trigger downloads
COPY core.py agent.py tools.py memory.py scheduler.py screen.py grimmbot.py ./

# Copy the assets directory (frontend UI)
COPY assets/ ./assets/


RUN echo '[]' > /app/data/adaptation.json \
    && chown grimmbot:grimmbot /app/data/adaptation.json

# Setup Fluxbox window manager
RUN mkdir -p /home/grimmbot/.fluxbox \
    && printf '[startup]\n[end]\n' > /home/grimmbot/.fluxbox/apps \
    && printf 'session.screen0.toolbar.visible: false\nsession.screen0.tabs.usePixmap: false\n' > /home/grimmbot/.fluxbox/init \
    && chown -R grimmbot:grimmbot /home/grimmbot/.fluxbox

# Setup default Chromium preferences to disable popups
RUN echo '{"browser":{"has_seen_welcome_page":true,"check_default_browser":false},"bookmark_bar":{"show_on_all_tabs":false},"distribution":{"skip_first_run_ui":true,"suppress_first_run_default_browser_prompt":true},"profile":{"default_content_setting_values":{"notifications":2}}}' > /home/grimmbot/.config/chromium/Default/Preferences \
    && chown -R grimmbot:grimmbot /home/grimmbot/.config/chromium

# Drop to standard user privileges
USER grimmbot
ENV HOME=/home/grimmbot
ENV CHROMIUM_PROFILE_DIR=/home/grimmbot/.config/chromium
ENV WORMHOLE_DIR=/app/wormhole
ENV WORKSPACE_DIR=/app/workspace
ENV DATA_DIR=/app/data

CMD ["/app/start.sh"]
