import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


current_file=Path(__file__).resolve()

INPUT_FILE=current_file.parent.parent.parent/"data"/"train_metadata.csv"

print(INPUT_FILE)
def analizar_corte():
    try:
        df = pd.read_csv(INPUT_FILE)
    except:
        print("❌ Falta el archivo train_metadata.csv")
        return

    # 1. Contamos audios por especie
    conteo = df['primary_label'].value_counts().reset_index()
    conteo.columns = ['Especie', 'Num_Audios']
    
    # 2. Ordenamos de mayor a menor
    conteo = conteo.sort_values('Num_Audios', ascending=False).reset_index(drop=True)
    
    # 3. Gráfica del Codo
    plt.figure(figsize=(12, 6))
    
    # Eje X: Número de especie (1ª, 2ª, 3ª...)
    # Eje Y: Cantidad de audios
    plt.plot(conteo.index, conteo['Num_Audios'], color='blue', linewidth=2, label='Distribución')
    
    # --- DIBUJAR LÍNEAS DE CORTE POTENCIALES ---
    cortes = [20, 50, 100]
    colors = ['red', 'orange', 'green']
    
    for corte, color in zip(cortes, colors):
        # Cuántas especies sobreviven a este corte?
        num_especies = len(conteo[conteo['Num_Audios'] >= corte])
        
        plt.axhline(y=corte, color=color, linestyle='--', alpha=0.7)
        plt.text(len(conteo), corte, f' Corte > {corte} audios\n ({num_especies} especies)', 
                 color=color, verticalalignment='center')

    plt.title('Método del Codo: Cantidad de Audios por Especie')
    plt.xlabel('Especies (Ordenadas por popularidad)')
    plt.ylabel('Número de Audios')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    print("Cierra la gráfica para ver los datos...")
    plt.show()
    
    # Imprimir tabla de decisión
    print("\n--- 📋 TABLA DE DECISIÓN ---")
    print("| Umbral Min. Audios | Especies Resultantes | ¿Es viable? |")
    print("|--------------------|----------------------|-------------|")
    for corte in [10, 20, 30, 50, 100]:
        n = len(conteo[conteo['Num_Audios'] >= corte])
        viable = "✅ Sí" if n > 20 else "⚠️ Pocas"
        if n > 150: viable = "⚠️ Demasiadas (Lento)"
        print(f"| > {corte:<16} | {n:<20} | {viable:<11} |")

if __name__ == "__main__":
    analizar_corte()