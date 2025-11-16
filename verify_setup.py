#!/usr/bin/env python3
"""
Environment verification script for DINO training on TBX11K
Run this before submitting training jobs to verify setup
"""

import sys
import os
from pathlib import Path
import subprocess


def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def print_status(check, status, message=""):
    symbol = "✓" if status else "✗"
    status_text = "PASS" if status else "FAIL"
    color = "\033[92m" if status else "\033[91m"
    reset = "\033[0m"
    print(f"{color}[{symbol}]{reset} {check:.<50} {status_text}")
    if message:
        print(f"    → {message}")
    return status


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    is_ok = version.major == 3 and version.minor >= 8
    return print_status(
        "Python Version",
        is_ok,
        f"Python {version.major}.{version.minor}.{version.micro} (need >= 3.8)"
    )


def check_pytorch():
    """Check PyTorch installation and CUDA"""
    try:
        import torch
        version_ok = torch.__version__ >= "1.13.0"
        cuda_ok = torch.cuda.is_available()
        
        print_status(
            "PyTorch",
            version_ok,
            f"Version {torch.__version__} (need >= 1.13.0)"
        )
        print_status(
            "CUDA Available",
            cuda_ok,
            f"CUDA {torch.version.cuda if cuda_ok else 'N/A'}"
        )
        return version_ok and cuda_ok
    except ImportError:
        return print_status("PyTorch", False, "Not installed")


def check_dependencies():
    """Check required dependencies"""
    all_ok = True
    packages = [
        'torchvision',
        'numpy',
        'scipy',
        'pycocotools',
        'tqdm',
        'Pillow'
    ]
    
    for pkg in packages:
        try:
            __import__(pkg.replace('-', '_'))
            print_status(pkg, True)
        except ImportError:
            print_status(pkg, False, "Not installed")
            all_ok = False
    
    return all_ok


def check_deformable_ops():
    """Check if deformable attention operators are built"""
    try:
        from models.dino.ops.modules import MSDeformAttn
        return print_status(
            "Deformable Attention Ops",
            True,
            "Compiled successfully"
        )
    except Exception as e:
        return print_status(
            "Deformable Attention Ops",
            False,
            f"Not built: {str(e)}"
        )


def check_dataset():
    """Check TBX11K dataset structure"""
    project_root = Path(__file__).parent
    dataset_path = project_root / "TBX11K"
    
    checks = {
        "TBX11K directory": dataset_path.exists(),
        "imgs folder": (dataset_path / "imgs").exists(),
        "annotations folder": (dataset_path / "annotations" / "json").exists(),
        "all_train.json": (dataset_path / "annotations" / "json" / "all_train.json").exists(),
        "all_val.json": (dataset_path / "annotations" / "json" / "all_val.json").exists(),
    }
    
    all_ok = True
    for check, exists in checks.items():
        all_ok &= print_status(check, exists)
    
    # Check JSON file sizes
    if checks["all_train.json"]:
        train_size = (dataset_path / "annotations" / "json" / "all_train.json").stat().st_size
        size_ok = train_size > 1000000  # Should be > 1MB
        print_status(
            "all_train.json size",
            size_ok,
            f"{train_size / 1e6:.1f} MB"
        )
        all_ok &= size_ok
    
    return all_ok


def check_config_files():
    """Check configuration files"""
    project_root = Path(__file__).parent
    
    configs = {
        "Base config": "config/DINO/DINO_4scale.py",
        "TBX11K config": "config/DINO/DINO_4scale_tbx11k.py",
    }
    
    all_ok = True
    for name, path in configs.items():
        exists = (project_root / path).exists()
        all_ok &= print_status(name, exists, path)
    
    return all_ok


def check_scripts():
    """Check SLURM scripts"""
    project_root = Path(__file__).parent
    
    scripts = {
        "Training script": "scripts/train_tbx11k.sbatch",
        "Evaluation script": "scripts/eval_tbx11k.sbatch",
    }
    
    all_ok = True
    for name, path in scripts.items():
        full_path = project_root / path
        exists = full_path.exists()
        
        if exists:
            # Check if executable
            is_executable = os.access(full_path, os.X_OK)
            if not is_executable:
                # Try to make executable
                try:
                    full_path.chmod(0o755)
                    is_executable = True
                except:
                    pass
            
            print_status(name, exists and is_executable, f"{path} {'(executable)' if is_executable else '(not executable)'}")
            all_ok &= is_executable
        else:
            print_status(name, False, path)
            all_ok = False
    
    return all_ok


def check_disk_space():
    """Check available disk space"""
    try:
        result = subprocess.run(['df', '-h', '.'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            parts = lines[1].split()
            available = parts[3]
            print_status("Disk Space", True, f"{available} available")
            return True
    except:
        pass
    return print_status("Disk Space", False, "Could not check")


def check_slurm():
    """Check if SLURM is available"""
    try:
        result = subprocess.run(['squeue', '--version'], capture_output=True, text=True)
        version = result.stdout.strip()
        return print_status("SLURM", True, version)
    except FileNotFoundError:
        return print_status("SLURM", False, "Not found (may not be on compute cluster)")


def main():
    print_header("DINO TBX11K Environment Verification")
    
    print("\n📋 Python Environment")
    py_ok = check_python_version()
    torch_ok = check_pytorch()
    deps_ok = check_dependencies()
    ops_ok = check_deformable_ops()
    
    print("\n📁 Dataset & Files")
    dataset_ok = check_dataset()
    config_ok = check_config_files()
    scripts_ok = check_scripts()
    
    print("\n💻 System")
    disk_ok = check_disk_space()
    slurm_ok = check_slurm()
    
    # Summary
    print_header("Verification Summary")
    
    all_checks = [
        ("Python Environment", py_ok and torch_ok and deps_ok),
        ("Deformable Operators", ops_ok),
        ("Dataset", dataset_ok),
        ("Configuration", config_ok),
        ("SLURM Scripts", scripts_ok),
    ]
    
    all_passed = all(status for _, status in all_checks)
    
    for check, status in all_checks:
        print_status(check, status)
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ All checks passed! Ready to train.")
        print("\nNext steps:")
        print("  1. Review QUICKSTART.md for quick commands")
        print("  2. Edit scripts/train_tbx11k.sbatch (set email)")
        print("  3. Submit training: sbatch scripts/train_tbx11k.sbatch")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        if not ops_ok:
            print("  • Deformable ops: cd models/dino/ops && python setup.py build install")
        if not deps_ok:
            print("  • Dependencies: pip install -r requirements.txt")
        if not dataset_ok:
            print("  • Dataset: Verify TBX11K folder structure")
        if not scripts_ok:
            print("  • Scripts: chmod +x scripts/*.sh")
    
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
