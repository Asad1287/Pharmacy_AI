#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log errors
error() {
    echo -e "${RED}ERROR: $1${NC}" >&2
    exit 1
}

# Function to log success
success() {
    echo -e "${GREEN}$1${NC}"
}

# Function to log info
info() {
    echo -e "${BLUE}$1${NC}"
}

# Function to check if a command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        return 1
    fi
    return 0
}

# Function to install Tesseract on macOS
install_tesseract_macos() {
    info "Installing Tesseract on macOS..."
    
    # Check if Homebrew is installed
    if ! check_command brew; then
        info "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || \
            error "Failed to install Homebrew"
    fi
    
    # Install Tesseract and dependencies
    brew install tesseract || error "Failed to install Tesseract"
    
    # Install additional languages
    brew install tesseract-lang || error "Failed to install Tesseract languages"
    
    # Set Tesseract data path
    TESSDATA_PREFIX=$(brew --prefix tesseract)/share/tessdata
    echo "export TESSDATA_PREFIX=$TESSDATA_PREFIX" >> ~/.bash_profile
    echo "export TESSDATA_PREFIX=$TESSDATA_PREFIX" >> ~/.zshrc
    
    success "Tesseract installed successfully on macOS"
}

# Function to install Tesseract on Linux
install_tesseract_linux() {
    info "Installing Tesseract on Linux..."
    
    # Detect package manager
    if check_command apt-get; then
        PM="apt-get"
        sudo $PM update
        
        # Install Tesseract and languages
        sudo $PM install -y \
            tesseract-ocr \
            libtesseract-dev \
            tesseract-ocr-eng \
            tesseract-ocr-deu \
            tesseract-ocr-fra \
            tesseract-ocr-spa \
            tesseract-ocr-ita \
            || error "Failed to install Tesseract and languages"
            
    elif check_command yum; then
        PM="yum"
        sudo $PM update
        
        # Install EPEL repository
        sudo $PM install -y epel-release
        
        # Install Tesseract and dependencies
        sudo $PM install -y \
            tesseract \
            tesseract-devel \
            tesseract-langpack-eng \
            tesseract-langpack-deu \
            tesseract-langpack-fra \
            tesseract-langpack-spa \
            tesseract-langpack-ita \
            || error "Failed to install Tesseract and languages"
            
    elif check_command dnf; then
        PM="dnf"
        sudo $PM update
        
        # Install Tesseract and dependencies
        sudo $PM install -y \
            tesseract \
            tesseract-devel \
            tesseract-langpack-eng \
            tesseract-langpack-deu \
            tesseract-langpack-fra \
            tesseract-langpack-spa \
            tesseract-langpack-ita \
            || error "Failed to install Tesseract and languages"
    else
        error "No supported package manager found"
    fi
    
    # Set Tesseract data path
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
    echo "export TESSDATA_PREFIX=$TESSDATA_PREFIX" >> ~/.bashrc
    
    success "Tesseract installed successfully on Linux"
}

# Function to verify Tesseract installation
verify_installation() {
    info "Verifying Tesseract installation..."
    
    # Check if tesseract command is available
    if ! check_command tesseract; then
        error "Tesseract is not properly installed"
    fi
    
    # Check Tesseract version
    TESSERACT_VERSION=$(tesseract --version | grep ^tesseract | cut -d' ' -f2)
    info "Tesseract version: $TESSERACT_VERSION"
    
    # Create test directory
    TEST_DIR="tesseract_test"
    mkdir -p $TEST_DIR
    
    # Create a simple test image
    echo "This is a test image" > "$TEST_DIR/test.txt"
    convert -size 200x50 \
            -background white \
            -fill black \
            -font Arial \
            caption:@"$TEST_DIR/test.txt" \
            "$TEST_DIR/test.png" || \
            error "Failed to create test image"
    
    # Test OCR
    info "Testing OCR functionality..."
    tesseract "$TEST_DIR/test.png" "$TEST_DIR/output" || \
        error "Failed to process test image"
    
    # Check if output file exists and contains text
    if [ ! -f "$TEST_DIR/output.txt" ] || [ ! -s "$TEST_DIR/output.txt" ]; then
        error "OCR test failed - no output generated"
    fi
    
    # Clean up test files
    rm -rf $TEST_DIR
    
    success "Tesseract verification completed successfully"
}

# Function to download and install additional language data
install_additional_languages() {
    info "Installing additional language data..."
    
    # Create temporary directory for downloads
    TEMP_DIR="tessdata_temp"
    mkdir -p $TEMP_DIR
    cd $TEMP_DIR
    
    # List of additional languages to install
    LANGUAGES="eng deu fra spa ita osd equ"
    
    for lang in $LANGUAGES; do
        info "Downloading $lang.traineddata..."
        curl -LO "https://github.com/tesseract-ocr/tessdata/raw/main/$lang.traineddata" || \
            error "Failed to download $lang.traineddata"
        
        # Move to appropriate directory
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sudo mv "$lang.traineddata" "$TESSDATA_PREFIX/" || \
                error "Failed to install $lang.traineddata"
        else
            sudo mv "$lang.traineddata" "/usr/share/tesseract-ocr/4.00/tessdata/" || \
                error "Failed to install $lang.traineddata"
        fi
    done
    
    # Clean up
    cd ..
    rm -rf $TEMP_DIR
    
    success "Additional language data installed successfully"
}

# Main installation function
main() {
    info "Starting Tesseract installation..."
    
    # Check if ImageMagick is installed (needed for verification)
    if ! check_command convert; then
        info "Installing ImageMagick..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install imagemagick || error "Failed to install ImageMagick"
        else
            sudo $PM install -y imagemagick || error "Failed to install ImageMagick"
        fi
    fi
    
    # Install based on OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        install_tesseract_macos
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        install_tesseract_linux
    else
        error "Unsupported operating system: $OSTYPE"
    fi
    
    # Install additional languages
    install_additional_languages
    
    # Verify installation
    verify_installation
    
    success "Tesseract installation and configuration completed successfully!"
    info "Please restart your terminal or run 'source ~/.bashrc' (Linux) or 'source ~/.bash_profile' (macOS) to update your environment"
}

# Execute main function
main