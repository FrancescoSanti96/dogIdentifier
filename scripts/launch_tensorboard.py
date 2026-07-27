#!/usr/bin/env python3
"""
Script per lanciare TensorBoard facilmente
Mostra tutti i training runs disponibili

Usage:
    python scripts/launch_tensorboard.py              # Training razze (default)
    python scripts/launch_tensorboard.py --mydog      # Training binario "È MAGGIE?"
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path

def launch_tensorboard(log_dir="outputs/tensorboard"):
    """Lancia TensorBoard con tutti i logs disponibili"""
    
    tensorboard_dir = Path(log_dir)
    
    if not tensorboard_dir.exists():
        print(f"❌ Directory {log_dir} non trovata!")
        print("   Esegui prima un training con TensorBoard logging")
        return
    
    # Lista tutti i run disponibili
    runs = list(tensorboard_dir.glob('*'))
    
    if not runs:
        print(f"❌ Nessun run TensorBoard trovato in {log_dir}!")
        print("   Esegui prima un training con TensorBoard logging")
        return
    
    training_type = "121 RAZZE" if "tensorboard_my_dog" not in log_dir else "BINARIO (MAGGIE)"
    print(f"📊 TENSORBOARD LAUNCHER - {training_type}")
    print("=" * 50)
    print(f"✅ Trovati {len(runs)} training runs:")
    
    for i, run_dir in enumerate(sorted(runs), 1):
        run_name = run_dir.name
        print(f"   {i}. {run_name}")
    
    print(f"\n🚀 Lanciando TensorBoard su porta 6006...")
    print(f"📂 Log directory: {tensorboard_dir.absolute()}")
    print(f"🌐 URL: http://localhost:6006")
    print(f"⚠️  Premi Ctrl+C per fermare TensorBoard")
    
    try:
        # Lancia TensorBoard
        cmd = [
            sys.executable, '-m', 'tensorboard.main',
            '--logdir', str(tensorboard_dir),
            '--port', '6006',
            '--reload_interval', '1'
        ]
        
        print(f"\n🔄 Eseguendo comando: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print(f"\n🛑 TensorBoard interrotto dall'utente")
    except FileNotFoundError:
        print(f"\n❌ TensorBoard non installato!")
        print(f"   Installa con: pip install tensorboard")
    except Exception as e:
        print(f"\n❌ Errore nel lanciare TensorBoard: {e}")

def main():
    parser = argparse.ArgumentParser(description="Launcher TensorBoard per Dog Breed Identifier")
    parser.add_argument('--mydog', action='store_true', 
                       help='Visualizza training binario "È MAGGIE?" invece di training razze')
    
    args = parser.parse_args()
    
    if args.mydog:
        log_dir = "outputs/tensorboard_my_dog"
    else:
        log_dir = "outputs/tensorboard"
    
    launch_tensorboard(log_dir)

if __name__ == "__main__":
    main()
