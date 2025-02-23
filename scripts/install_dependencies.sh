#!/bin/bash

# Function to print error messages
error() {
    echo -e "\e[31mError: $1\e[0m" >&2
    exit 1
}

# Function to print success messages
success() {
    echo -e "\e[32m$1\e[0m"
}

# Function to print info messages
info() {
    echo -e "\e[34m$1\e[0m"
}

# Function to check command existence
check_command() {
    if ! command -v $1 &> /dev/null; then
        return 1
    fi
    return 0
}

# Function to install package manager if not present (for Linux)
install_package_manager() {
    if ! check_command apt-get; then
        if check_command yum; then
            PM="yum"
        elif check_command dnf; then
            PM="dnf"
        else
            error "No supported package manager found"
        fi
    else
        PM="apt-get"
    fi
    echo $PM
}

# Function to install CUDA (if needed)
install_cuda() {
    if ! check_command nvidia-smi; then
        info "Installing CUDA dependencies..."
        case $PM in
            "apt-get")
                sudo $PM install -y nvidia-cuda-toolkit
                ;;
            "yum"|"dnf")
                sudo $PM install -y cuda
                ;;
        esac
    else
        success "CUDA already installed"
    fi
}

# Main installation function
main() {
    # Create necessary directories
    info "Creating necessary directories..."
    mkdir -p prescriptions_output models logs

    # Detect OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        info "Detected macOS system"
        
        # Check if Homebrew is installed
        if ! check_command brew; then
            info "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || error "Failed to install Homebrew"
        fi

        # Install dependencies using Homebrew
        info "Installing dependencies using Homebrew..."
        brew install tesseract || error "Failed to install Tesseract"
        brew install libtesseract || error "Failed to install libtesseract"
        brew install pkg-config || error "Failed to install pkg-config"
        brew install poppler || error "Failed to install poppler"
        brew install ghostscript || error "Failed to install ghostscript"
        brew install opencv || error "Failed to install opencv"

    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        info "Detected Linux system"
        
        # Get package manager
        PM=$(install_package_manager)
        info "Using package manager: $PM"

        # Update package list
        sudo $PM update || error "Failed to update package list"

        # Install dependencies
        info "Installing system dependencies..."
        sudo $PM install -y \
            tesseract-ocr \
            libtesseract-dev \
            tesseract-ocr-eng \
            poppler-utils \
            ghostscript \
            libsm6 \
            libxext6 \
            libxrender-dev \
            libgl1-mesa-glx \
            libglib2.0-0 \
            pkg-config \
            || error "Failed to install system dependencies"

        # Install OpenCV dependencies
        info "Installing OpenCV dependencies..."
        sudo $PM install -y \
            python3-opencv \
            libopencv-dev \
            || error "Failed to install OpenCV dependencies"

        # Install CUDA if system has NVIDIA GPU
        if lspci | grep -i nvidia &> /dev/null; then
            info "NVIDIA GPU detected"
            install_cuda
        fi

    else
        error "Unsupported operating system: $OSTYPE"
    fi

    # Verify installations
    info "Verifying installations..."
    
    # Check Tesseract
    if ! check_command tesseract; then
        error "Tesseract installation failed"
    fi
    
    # Check OpenCV
    if ! python3 -c "import cv2" &> /dev/null; then
        error "OpenCV installation failed"
    fi

    # Set up language data directory
    info "Setting up language data directory..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        TESSDATA_PREFIX=$(brew --prefix tesseract)/share/tessdata
    else
        TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
    fi
    
    # Export TESSDATA_PREFIX
    echo "export TESSDATA_PREFIX=$TESSDATA_PREFIX" >> ~/.bashrc
    echo "export TESSDATA_PREFIX=$TESSDATA_PREFIX" >> ~/.zshrc

    success "All dependencies installed successfully!"
    info "Please run 'source ~/.bashrc' or 'source ~/.zshrc' to update your environment"
}

# Execute main function
main