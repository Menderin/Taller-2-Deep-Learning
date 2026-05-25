import pandas as pd

ruta_archivo = 'data/raw/15 atributos R0-R5.sav' 
try:
    df = pd.read_spss(ruta_archivo)
    
    print(" Archivo cargado exitosamente.\n")
    print("--- filas del dataset (df.head()) ---")
    print(df.head(22))
    print("-" * 50 + "\n")

    # 2. Mostrar información general (tipos de datos, valores nulos, etc.)
    print("--- Información general (df.info()) ---")
    df.info()
    print("-" * 50 + "\n")

    print("--- Tamaño del dataset (df.shape) ---")
    print(f"El dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas.")
    print("-" * 50 + "\n")

    # 4. Mostrar estadísticas descriptivas básicas
    print("--- Estadísticas descriptivas (df.describe()) ---")
    # include='all' para ver también información de columnas no numéricas
    print(df.describe(include='all'))

except FileNotFoundError:
    print(f"❌ Error: No se encontró el archivo '{ruta_archivo}'. Verifica la ruta.")
except Exception as e:
    print(f"❌ Ocurrió un error al leer el archivo: {e}")