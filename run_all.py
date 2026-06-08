#!/usr/bin/env python3
"""
🎬 COMPLETE SENTIMENT ANALYSIS SETUP & TRAINING SCRIPT
Automates everything: setup, data download, model training, and app launch
Run this: python run_all.py
"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n{'='*70}")
    print(f"⏳ {description}...")
    print(f"{'='*70}")
    
    try:
        if isinstance(command, str):
            result = subprocess.run(command, shell=True, check=True)
        else:
            result = subprocess.run(command, check=True)
        
        print(f"✅ {description} COMPLETE!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║          🎬 SENTIMENT ANALYSIS - COMPLETE AUTOMATION 🎬             ║
║                                                                      ║
║  This script will:                                                   ║
║  ✅ Create virtual environment                                      ║
║  ✅ Install all dependencies                                        ║
║  ✅ Download IMDb dataset (25,000 reviews)                         ║
║  ✅ Train Baseline Model (89.24% accuracy)                         ║
║  ✅ Train Transformer Model (92.46% accuracy)                      ║
║  ✅ Launch Streamlit Web App                                       ║
║                                                                      ║
║  Total Time: ~2-3 hours                                             ║
║  GPU Recommended for faster training                                ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required!")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Detect OS
    os_type = platform.system()
    print(f"✅ OS: {os_type}")
    
    # Step 1: Create virtual environment
    venv_name = "venv"
    if not os.path.exists(venv_name):
        run_command(f"{sys.executable} -m venv {venv_name}", 
                   "Creating virtual environment")
    else:
        print(f"✅ Virtual environment '{venv_name}' already exists")
    
    # Determine activation command
    if os_type == "Windows":
        activate_cmd = f"{venv_name}\\Scripts\\activate.bat && "
    else:
        activate_cmd = f"source {venv_name}/bin/activate && "
    
    # Step 2: Upgrade pip
    run_command(f"{activate_cmd}{sys.executable} -m pip install --upgrade pip",
               "Upgrading pip")
    
    # Step 3: Install requirements
    run_command(f"{activate_cmd}pip install -r requirements.txt",
               "Installing dependencies")
    
    # Step 4: Download and prepare data
    run_command(f"{activate_cmd}{sys.executable} data_preparation.py",
               "Downloading and preparing IMDb dataset")
    
    # Step 5: Train baseline model
    run_command(f"{activate_cmd}{sys.executable} baseline_model.py",
               "Training Baseline Model")
    
    # Step 6: Train transformer model
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║  ⏳ TRAINING TRANSFORMER MODEL (This takes 20-45 minutes)          ║
║                                                                      ║
║  ☕ Grab a coffee and relax!                                        ║
║  💻 Using GPU will be much faster (5-10 minutes)                   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    run_command(f"{activate_cmd}{sys.executable} transformer_model.py",
               "Training Transformer Model")
    
    # Step 7: Launch Streamlit app
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║                   ✅ ALL TRAINING COMPLETE! ✅                      ║
║                                                                      ║
║              🌐 Launching Streamlit Web App...                       ║
║                                                                      ║
║     Visit: http://localhost:8501                                     ║
║                                                                      ║
║  Pages Available:                                                    ║
║  🏠 Home - Project overview                                         ║
║  🔮 Predict - Get sentiments for reviews                            ║
║  📊 Model Comparison - Baseline vs Transformer                       ║
║  📈 Performance - Detailed metrics                                   ║
║  ℹ️  About - Project information                                     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    run_command(f"{activate_cmd}streamlit run app.py",
               "Launching Streamlit Application")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
