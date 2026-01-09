#!/bin/bash
# Intentioned - Linux/macOS Installation Script
# Run with: bash install.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default installation path
INSTALL_PATH="${INSTALL_PATH:-$HOME/.local/share/intentioned}"
OPEN_CONFIG=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-config)
            OPEN_CONFIG=false
            shift
            ;;
        --install-path)
            INSTALL_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Intentioned Installer"
            echo ""
            echo "Usage: bash install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --install-path PATH   Set custom installation path"
            echo "  --no-config           Skip opening config tool after install"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         Intentioned - Social Skills Training Platform         ║"
echo "║                  Linux/macOS Installer v1.0                   ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     PLATFORM=Linux;;
    Darwin*)    PLATFORM=macOS;;
    *)          PLATFORM="UNKNOWN:${OS}"
esac
echo -e "${YELLOW}Detected platform: ${PLATFORM}${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for Python
echo -e "\n${YELLOW}[1/7] Checking Python installation...${NC}"
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d' ' -f2 | cut -d'.' -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d' ' -f2 | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
        echo -e "${RED}   Python 3.10+ required. Found: $PYTHON_VERSION${NC}"
        echo -e "${RED}   Please install Python 3.10 or later${NC}"
        exit 1
    fi
    echo -e "${GREEN}   Found: $PYTHON_VERSION ✓${NC}"
else
    echo -e "${RED}   Python3 not found!${NC}"
    if [ "$PLATFORM" = "Linux" ]; then
        echo -e "${YELLOW}   Installing Python3...${NC}"
        if command_exists apt-get; then
            sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv
        elif command_exists dnf; then
            sudo dnf install -y python3 python3-pip
        elif command_exists pacman; then
            sudo pacman -S --noconfirm python python-pip
        else
            echo -e "${RED}   Please install Python 3.10+ manually${NC}"
            exit 1
        fi
    elif [ "$PLATFORM" = "macOS" ]; then
        echo -e "${YELLOW}   Installing Python3 via Homebrew...${NC}"
        if ! command_exists brew; then
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        brew install python@3.12
    fi
fi

# Check for Git
echo -e "\n${YELLOW}[2/7] Checking Git installation...${NC}"
if command_exists git; then
    GIT_VERSION=$(git --version)
    echo -e "${GREEN}   Found: $GIT_VERSION ✓${NC}"
else
    echo -e "${YELLOW}   Git not found. Installing...${NC}"
    if [ "$PLATFORM" = "Linux" ]; then
        if command_exists apt-get; then
            sudo apt-get install -y git
        elif command_exists dnf; then
            sudo dnf install -y git
        elif command_exists pacman; then
            sudo pacman -S --noconfirm git
        fi
    elif [ "$PLATFORM" = "macOS" ]; then
        brew install git
    fi
fi

# Check for ffmpeg
echo -e "\n${YELLOW}[3/7] Checking ffmpeg installation...${NC}"
if command_exists ffmpeg; then
    FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -n1)
    echo -e "${GREEN}   Found: $FFMPEG_VERSION ✓${NC}"
else
    echo -e "${YELLOW}   ffmpeg not found. Installing...${NC}"
    if [ "$PLATFORM" = "Linux" ]; then
        if command_exists apt-get; then
            sudo apt-get install -y ffmpeg
        elif command_exists dnf; then
            sudo dnf install -y ffmpeg
        elif command_exists pacman; then
            sudo pacman -S --noconfirm ffmpeg
        fi
    elif [ "$PLATFORM" = "macOS" ]; then
        brew install ffmpeg
    fi
fi

# Create installation directory
echo -e "\n${YELLOW}[4/7] Creating installation directory...${NC}"
mkdir -p "$INSTALL_PATH"
echo -e "${GREEN}   Path: $INSTALL_PATH ✓${NC}"

# Clone or update repository
echo -e "\n${YELLOW}[5/7] Downloading Intentioned...${NC}"
REPO_PATH="$INSTALL_PATH/intentioned.tech"
if [ -d "$REPO_PATH" ]; then
    echo -e "${YELLOW}   Updating existing installation...${NC}"
    cd "$REPO_PATH"
    git pull
else
    git clone https://github.com/KoJesko/intentioned.tech.git "$REPO_PATH"
fi
echo -e "${GREEN}   Downloaded ✓${NC}"

# Create virtual environment and install dependencies
echo -e "\n${YELLOW}[6/7] Installing Python dependencies...${NC}"
cd "$REPO_PATH"
if [ ! -d "myenv" ]; then
    python3 -m venv myenv
fi
./myenv/bin/pip install --upgrade pip
./myenv/bin/pip install -r requirements.txt
echo -e "${GREEN}   Dependencies installed ✓${NC}"

# Create start scripts
echo -e "\n${YELLOW}[7/7] Creating launch scripts...${NC}"

# Create start script
cat > "$INSTALL_PATH/start-intentioned.sh" << EOF
#!/bin/bash
cd "$REPO_PATH"
source myenv/bin/activate
python server.py
EOF
chmod +x "$INSTALL_PATH/start-intentioned.sh"

# Create config script
cat > "$INSTALL_PATH/config-intentioned.sh" << EOF
#!/bin/bash
cd "$REPO_PATH"
source myenv/bin/activate
python config_tool.py
EOF
chmod +x "$INSTALL_PATH/config-intentioned.sh"

# Create symlinks in user's bin directory
USER_BIN="$HOME/.local/bin"
mkdir -p "$USER_BIN"
ln -sf "$INSTALL_PATH/start-intentioned.sh" "$USER_BIN/intentioned"
ln -sf "$INSTALL_PATH/config-intentioned.sh" "$USER_BIN/intentioned-config"

# Add to PATH if needed
if [[ ":$PATH:" != *":$USER_BIN:"* ]]; then
    echo "export PATH=\"\$PATH:$USER_BIN\"" >> "$HOME/.bashrc"
    echo "export PATH=\"\$PATH:$USER_BIN\"" >> "$HOME/.zshrc" 2>/dev/null || true
    echo -e "${YELLOW}   Added $USER_BIN to PATH (restart shell to apply)${NC}"
fi

# Create desktop entry for Linux
if [ "$PLATFORM" = "Linux" ]; then
    DESKTOP_DIR="$HOME/.local/share/applications"
    mkdir -p "$DESKTOP_DIR"
    
    cat > "$DESKTOP_DIR/intentioned.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Intentioned
Comment=Social Skills Training Platform
Exec=$INSTALL_PATH/start-intentioned.sh
Icon=$REPO_PATH/favicon.ico
Terminal=true
Categories=Education;
EOF

    cat > "$DESKTOP_DIR/intentioned-config.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Intentioned Config
Comment=Intentioned Configuration Tool
Exec=$INSTALL_PATH/config-intentioned.sh
Icon=$REPO_PATH/favicon.ico
Terminal=false
Categories=Education;Settings;
EOF

    echo -e "${GREEN}   Desktop entries created ✓${NC}"
fi

echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                  Installation Complete! 🎉                     ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║  To start Intentioned:                                        ║"
echo "║    intentioned                                                 ║"
echo "║    or: $INSTALL_PATH/start-intentioned.sh                      "
echo "║                                                                ║"
echo "║  To configure:                                                 ║"
echo "║    intentioned-config                                          ║"
echo "║    or: $INSTALL_PATH/config-intentioned.sh                     "
echo "║                                                                ║"
echo "║  Installation path: $INSTALL_PATH                              "
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Offer to open config tool
if [ "$OPEN_CONFIG" = true ]; then
    echo -e "\n${CYAN}Would you like to configure Intentioned now? (Y/n)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Nn]$ ]]; then
        echo -e "${CYAN}Opening Configuration Tool...${NC}"
        cd "$REPO_PATH"
        source myenv/bin/activate
        python config_tool.py
    fi
fi

echo -e "\n${CYAN}Thank you for installing Intentioned!${NC}"
