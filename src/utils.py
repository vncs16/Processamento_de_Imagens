# src/utils.py
import cv2
import numpy as np
# from PIL import Image # Não precisamos mais importar PIL.Image aqui para a conversão para QPixmap

from PySide6.QtGui import QImage, QPixmap # Alterado de PyQt5/PySide2
from PySide6.QtCore import Qt # Necessário para QImage.Format e Qt.KeepAspectRatio, etc.

def load_image_cv2(file_path):
    """
    Carrega uma imagem usando OpenCV.
    Retorna a imagem em formato BGR (padrão do OpenCV).
    """
    try:
        img = cv2.imread(file_path)
        if img is None:
            raise FileNotFoundError(f"Não foi possível carregar a imagem de: {file_path}")
        return img
    except Exception as e:
        print(f"Erro ao carregar a imagem: {e}")
        return None

def save_image_cv2(image_data, file_path):
    """
    Salva uma imagem usando OpenCV.
    image_data deve ser um array NumPy.
    """
    try:
        cv2.imwrite(file_path, image_data)
        return True
    except Exception as e:
        print(f"Erro ao salvar a imagem: {e}")
        return False

def convert_to_grayscale(image_bgr):
    """
    Converte uma imagem BGR (OpenCV) para níveis de cinza.
    Se a imagem já for em níveis de cinza, retorna como está.
    """
    if len(image_bgr.shape) == 3 and image_bgr.shape[2] == 3:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return image_bgr # Já é grayscale ou tem canal alfa, trata como grayscale


def convert_cv2_to_qpixmap(cv_image):
    """
    Converte uma imagem OpenCV (NumPy array) para QPixmap para exibição na GUI.
    Suporta imagens em níveis de cinza e coloridas (BGR).
    """
    if cv_image is None:
        return None

    h, w = cv_image.shape[:2]
    
    if len(cv_image.shape) == 2:  # Grayscale
        bytes_per_line = w
        # QImage.Format_Grayscale8 para imagens em escala de cinza de 8 bits
        q_image = QImage(cv_image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
    elif len(cv_image.shape) == 3:  # BGR (color)
        # OpenCV usa BGR, QImage espera RGB ou BGRX/RGBX. Converter BGR para RGB é o mais seguro.
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * w # 3 bytes por pixel (RGB)
        # QImage.Format_RGB888 para imagens RGB de 24 bits
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    else:
        return None # Formato de imagem não suportado

    return QPixmap.fromImage(q_image)