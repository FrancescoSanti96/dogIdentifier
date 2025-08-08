#!/usr/bin/env python3
"""
Script per lanciare TensorBoard facilmente
Mostra tutti i training runs disponibili
"""

import os
import subprocess
import sys
from pathlib import Path

def launch_tensorboard():
    """Lancia TensorBoard con tutti i logs disponibili"""
    
    tensorboard_dir = Path('outputs/tensorboard')
    
    if not tensorboard_dir.exists():
        print("❌ Directory outputs/tensorboard non trovata!")
        print("   Esegui prima un training con TensorBoard logging")
        return
    
    # Lista tutti i run disponibili
    runs = list(tensorboard_dir.glob('*'))
    
    if not runs:
        print("❌ Nessun run TensorBoard trovato!")
        print("   Esegui prima un training con TensorBoard logging")
        return
    
    print("📊 TENSORBOARD LAUNCHER")
    print("=" * 40)
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

if __name__ == "__main__":
    launch_tensorboard()
