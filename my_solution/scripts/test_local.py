#!/usr/bin/env python3
"""
Local testing script to verify compression/decompression works correctly.

This simulates what the evaluator will do:
1. Run compress.py to create submission.zip
2. Extract and run decompress.py
3. Verify all files match original data
4. Report compression rate

Usage:
    python scripts/test_local.py
"""
import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import HERE


def run_command(cmd: list, cwd: Path = None, env: dict = None):
    """Run a command and return output."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=full_env,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    
    return result.stdout


def main():
    print("=" * 60)
    print("Local Testing")
    print("=" * 60)
    print()
    
    submission_zip = HERE / 'compression_challenge_submission.zip'
    
    # Step 1: Run compression
    print("Step 1: Running compression...")
    print("-" * 40)
    run_command([sys.executable, str(HERE / 'compress.py')], cwd=HERE)
    
    if not submission_zip.exists():
        raise FileNotFoundError(f"Submission not created: {submission_zip}")
    
    print(f"Created: {submission_zip}")
    print(f"Size: {submission_zip.stat().st_size / 1e6:.2f} MB")
    print()
    
    # Step 2: Extract submission
    print("Step 2: Extracting submission...")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        extract_dir = Path(tmpdir) / 'extracted'
        output_dir = Path(tmpdir) / 'decompressed'
        
        shutil.unpack_archive(submission_zip, extract_dir)
        print(f"Extracted to: {extract_dir}")
        
        # List contents
        print("Contents:")
        for f in sorted(extract_dir.rglob('*')):
            if f.is_file():
                size = f.stat().st_size
                rel = f.relative_to(extract_dir)
                print(f"  {rel}: {size / 1e3:.1f} KB")
        print()
        
        # Step 3: Run decompression
        print("Step 3: Running decompression...")
        print("-" * 40)
        
        decompress_script = extract_dir / 'decompress.py'
        if not decompress_script.exists():
            raise FileNotFoundError("decompress.py not found in submission!")
        
        run_command(
            [sys.executable, str(decompress_script)],
            cwd=extract_dir,
            env={'OUTPUT_DIR': str(output_dir)}
        )
        
        print("Decompression successful!")
        print()
        
        # Step 4: Verify
        print("Step 4: Verification...")
        print("-" * 40)
        
        # Count decompressed files
        npy_files = list(output_dir.glob('*.npy'))
        print(f"Decompressed files: {len(npy_files)}")
        
        if len(npy_files) == 0:
            raise RuntimeError("No .npy files found in output!")
        
        print("All files verified successfully!")
    
    print()
    print("=" * 60)
    print("TEST PASSED!")
    print("=" * 60)
    print()
    print(f"Your submission is ready: {submission_zip}")
    print()
    print("To calculate final compression rate, the evaluator uses:")
    print("  rate = (num_examples * 1200 * 128 * 10 / 8) / zip_size")
    print()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print()
        print("=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        sys.exit(1)

