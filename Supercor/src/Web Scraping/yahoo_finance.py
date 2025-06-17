import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# Current dir
current_dir = os.path.dirname(os.path.abspath(__file__))

# Opciones de driver
options = webdriver.ChromeOptions()
options.page_load_strategy = "normal"
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

# https://finance.yahoo.com/quote/CL%3DF/history/?period1=1296259200&period2=1463875200

# POC
# https://finance.yahoo.com/quote/WTI/history/?period1=1514678400&period2=1525824000

# URL objetivo
url = "https://finance.yahoo.com/quote/CL%3DF/history/?period1=1296259200&period2=1463875200"

def get_dataframe(data_body):
    list_stock = []
    for row in data_body.find_elements(By.CSS_SELECTOR, "tr"):
        cells = row.find_elements(By.CSS_SELECTOR, "td") 
        list_stock.append([c.text for c in cells])
        print("Procesando fila...")
    return list_stock

try:
    driver.get(url)
    
    # Rechazar cookies
    print("Buscando botón de consentimiento...")
    try:
        consent_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn.secondary.reject-all"))
        )
        # Hacer scroll hacia el botón
        driver.execute_script("arguments[0].scrollIntoView(true);", consent_button)
        time.sleep(1)  # Espera pequeña tras hacer scroll
        consent_button.click()
        print("Botón 'Rechazar Todo' clickeado")
        time.sleep(2)
    except TimeoutException:
        print("No se mostró el popup de cookies")

    # Esperar la tabla
    data_head = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.table-container table.table thead"))
    )
    data_body = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.table-container table.table tbody"))
    )

    # Obtener cabeceras
    headers = []
    for row in data_head.find_elements(By.CSS_SELECTOR, "tr"):
        print(row.text)
        header = row.find_elements(By.CSS_SELECTOR, "th")
        headers = [h.text for h in header]

    # Obtener cuerpo de tabla
    stocks = get_dataframe(data_body)

    # Crear DataFrame
    df = pd.DataFrame(stocks, columns=headers)
    print(df)

    # Guardar CSV
    brent_oil_path = os.path.join(current_dir, '../../data/raw/macrodata')
    os.makedirs(brent_oil_path, exist_ok=True)
    brent_oil_data = os.path.join(brent_oil_path, "wti_crude_oil.csv")
    df.to_csv(brent_oil_data, index=False)
    print(f"Datos guardados en: {brent_oil_data}")

except Exception as e:
    print(f"Error durante la ejecución: {e}")

finally:
    driver.quit()

