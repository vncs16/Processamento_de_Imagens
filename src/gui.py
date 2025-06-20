# src/gui.py
import sys
from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QLabel, QFileDialog, QSizePolicy, QLineEdit, QDialog, QMessageBox, QInputDialog,
    QGroupBox, QScrollArea # QScrollArea já importado
)
from PySide6.QtGui import QPixmap, QImage, QIcon
from PySide6.QtCore import Qt
import cv2
import numpy as np
from .image_processor import ImageProcessor
from .utils import load_image_cv2, save_image_cv2, convert_to_grayscale, convert_cv2_to_qpixmap


class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Processamento Digital de Imagens")
        self.setGeometry(100, 100, 1200, 800) # Tamanho da janela
        
        self.image_processor = ImageProcessor()
        self.current_image_bgr = None 
        self.current_image_gray = None 
        self.display_image = None 

        self.init_ui()
        self.apply_dark_styles()

    def apply_dark_styles(self):
        self.setStyleSheet("""
            /* Estilos Globais */
            QMainWindow {
                background-color: #2e2e2e; /* Cinza escuro de fundo */
                color: #e0e0e0; /* Texto claro */
            }

            /* Estilo para GroupBoxes (para agrupar seções de filtros) */
            QGroupBox {
                background-color: #3c3c3c; /* Fundo do groupbox um pouco mais claro que o da janela */
                border: 1px solid #5a5a5a;
                border-radius: 8px;
                margin-top: 10px; /* Espaço para o título */
                font-weight: bold;
                color: #e0e0e0; /* Cor do texto do groupbox */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; /* Título centralizado no topo */
                padding: 0 10px;
                background-color: #2e2e2e; /* Fundo do título igual ao da janela */
                border-radius: 5px;
            }

            /* Estilo para Botões */
            QPushButton {
                background-color: #4a90e2; /* Azul vibrante */
                color: white;
                border-radius: 5px;
                padding: 10px 15px;
                font-size: 14px;
                font-weight: bold;
                border: none;
                margin: 5px 0; /* Margem vertical para espaçamento */
            }
            QPushButton:hover {
                background-color: #5c9cea; /* Azul mais claro no hover */
            }
            QPushButton:pressed {
                background-color: #3b7ad1; /* Azul mais escuro quando pressionado */
            }

            /* Estilo para QLabel que exibe a imagem */
            QLabel#image_display_label { /* Renomeado para não conflitar com outros QLabels */
                border: 2px dashed #757575; /* Borda pontilhada cinza */
                background-color: #1e1e1e; /* Fundo bem escuro para a área da imagem */
                color: #cccccc;
                font-style: italic;
                padding: 20px;
                border-radius: 8px;
            }

            /* Estilo para outros QLabels (títulos de seção, etc.) */
            QLabel {
                font-weight: bold;
                color: #e0e0e0;
                margin-top: 5px;
                margin-bottom: 5px;
            }

            /* Estilo para QLineEdit (caixas de entrada de texto) */
            QLineEdit {
                border: 1px solid #5a5a5a;
                border-radius: 5px;
                padding: 5px;
                background-color: #4a4a4a; /* Fundo escuro para a caixa de texto */
                color: #e0e0e0; /* Texto claro */
            }

            /* Estilo para QDialog (caixas de diálogo de input) */
            QDialog {
                background-color: #3c3c3c;
                color: #e0e0e0;
            }
            QDialog QLabel {
                color: #e0e0e0;
            }
            QDialog QPushButton {
                background-color: #4a90e2;
                color: white;
            }

            /* Estilo da barra de rolagem */
            QScrollArea {
                border: none; /* Remover borda padrão do scroll area */
            }
            QScrollBar:vertical {
                border: 1px solid #4a4a4a;
                background: #3c3c3c;
                width: 12px;
                margin: 0px 0px 0px 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #6a6a6a;
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            QScrollBar:horizontal {
                border: 1px solid #4a4a4a;
                background: #3c3c3c;
                height: 12px;
                margin: 0px 0px 0px 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background: #6a6a6a;
                min-width: 20px;
                border-radius: 6px;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                border: none;
                background: none;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
            }
        """)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20) # Margens maiores
        main_layout.setSpacing(20) # Mais espaço entre os painéis

        # Painel Esquerdo: Área de Exibição da Imagem e Botões de Arquivo
        image_panel = QVBoxLayout()
        
        # --- QScrollArea para a imagem ---
        self.image_scroll_area = QScrollArea()
        self.image_scroll_area.setWidgetResizable(True) # Permitir que o widget interno seja redimensionado
        self.image_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.image_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.image_label = QLabel("Carregue uma imagem para começar")
        self.image_label.setObjectName("image_display_label") # Nome único para o QSS
        self.image_label.setAlignment(Qt.AlignCenter)
        # self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Removido para usar o scroll area

        # Adiciona o QLabel ao QScrollArea
        self.image_scroll_area.setWidget(self.image_label)
        image_panel.addWidget(self.image_scroll_area) # Adiciona a área de rolagem ao painel da imagem

        file_buttons_layout = QHBoxLayout()
        try:
            self.load_button = QPushButton(QIcon("assets/open_icon.png"), "Carregar Imagem") 
            self.save_button = QPushButton(QIcon("assets/save_icon.png"), "Salvar Imagem")
        except:
            self.load_button = QPushButton("Carregar Imagem")
            self.save_button = QPushButton("Salvar Imagem")

        self.load_button.clicked.connect(self.load_image)
        self.save_button.clicked.connect(self.save_image)
        file_buttons_layout.addWidget(self.load_button)
        file_buttons_layout.addWidget(self.save_button)
        image_panel.addLayout(file_buttons_layout)
        
        main_layout.addLayout(image_panel, 2) # 2/3 da largura para a imagem

        # Painel Direito: Controles (divididos em GroupBoxes e encapsulados em QScrollArea)
        controls_panel_scroll_area = QScrollArea() # Cria a área de rolagem
        controls_panel_scroll_area.setWidgetResizable(True) # Faz o widget dentro dela ser redimensionável
        controls_panel_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff) # Desliga a barra horizontal
        controls_panel_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded) # Barra vertical conforme necessário

        controls_panel_container = QWidget() # Este widget conterá todos os GroupBoxes
        controls_panel_container.setObjectName("controls_panel_widget") # Nome para QSS
        controls_panel = QVBoxLayout(controls_panel_container)
        controls_panel.setAlignment(Qt.AlignTop)
        controls_panel.setSpacing(10) # Espaçamento entre os group boxes

        # Adiciona o container de controles à área de rolagem
        controls_panel_scroll_area.setWidget(controls_panel_container)

        # --- Seções com QGroupBox ---
        self.add_group_box(controls_panel, "Histograma", [
            ("Calcular Histograma", self.action_calculate_histogram)
        ])

        self.add_group_box(controls_panel, "Transformações de Intensidade", [
            ("Alargamento de Contraste", self.action_contrast_stretching),
            ("Equalização de Histograma", self.action_histogram_equalization)
        ])

        self.add_group_box(controls_panel, "Filtros Passa-Baixa", [
            ("Média", self.action_mean_filter),
            ("Mediana", self.action_median_filter),
            ("Gaussiano", self.action_gaussian_filter),
            ("Máximo", self.action_max_filter),
            ("Mínimo", self.action_min_filter)
        ])
        
        self.add_group_box(controls_panel, "Filtros Passa-Alta", [
            ("Laplaciano", self.action_laplacian_filter),
            ("Roberts", self.action_roberts_filter),
            ("Prewitt", self.action_prewitt_filter),
            ("Sobel", self.action_sobel_filter)
        ])

        self.add_group_box(controls_panel, "Convolução na Frequência", [
            ("Filtro Passa-Baixa (Frequência)", lambda: self.action_frequency_domain_filter("lowpass")),
            ("Filtro Passa-Alta (Frequência)", lambda: self.action_frequency_domain_filter("highpass"))
        ])

        self.add_group_box(controls_panel, "Espectro de Fourier", [
            ("Exibir Espectro de Fourier", self.action_display_fourier_spectrum)
        ])

        self.add_group_box(controls_panel, "Morfologia Matemática Básica", [
            ("Erosão", self.action_erosion),
            ("Dilatação", self.action_dilation)
        ])

        self.add_group_box(controls_panel, "Morfologia Matemática Avançada", [
            ("Abertura", self.action_opening),
            ("Fechamento", self.action_closing),
            ("Gradiente Morfológico", self.action_morphological_gradient)
        ])
        
        self.add_group_box(controls_panel, "Segmentação", [
            ("Limiarização (Otsu)", self.action_otsu_thresholding)
        ])
        
        main_layout.addWidget(controls_panel_scroll_area, 1) 

    def add_group_box(self, parent_layout, title, buttons_info):
        group_box = QGroupBox(title)
        group_layout = QVBoxLayout(group_box)
        group_layout.setContentsMargins(10, 25, 10, 10) 
        group_layout.setSpacing(5) 
        
        for button_text, action_func in buttons_info:
            button = QPushButton(button_text) 
            group_layout.addWidget(button)
            button.clicked.connect(action_func)
        
        parent_layout.addWidget(group_box)

    def display_image_on_label(self, cv_image):
        if cv_image is None:
            self.image_label.setText("Erro ao exibir imagem.")
            return

        q_pixmap = convert_cv2_to_qpixmap(cv_image)
        if q_pixmap:
            # Não escalamos mais para o tamanho do QLabel, mas para o tamanho do QScrollArea (viewport)
            # ou para manter o tamanho original se for menor.
            # Se você quer que a imagem sempre preencha o espaço disponível do scrollarea, use:
            # scaled_pixmap = q_pixmap.scaled(self.image_scroll_area.viewport().size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            # Ou para manter o tamanho original e usar scrollbar se for maior:
            self.image_label.setPixmap(q_pixmap) # Define a imagem em seu tamanho original
            self.image_label.adjustSize() # Ajusta o tamanho do QLabel ao tamanho da imagem
            self.image_label.setText("") # Remove o texto "Nenhuma imagem carregada"
        else:
            self.image_label.setText("Falha ao converter imagem para exibição.")


    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Carregar Imagem", "",
                                                "Imagens (*.png *.jpg *.jpeg *.bmp *.tif);;Todos os Arquivos (*)", options=options)
        if file_name:
            loaded_image_bgr = load_image_cv2(file_name)
            if loaded_image_bgr is not None:
                self.current_image_bgr = loaded_image_bgr
                self.current_image_gray = convert_to_grayscale(self.current_image_bgr)
                self.display_image = self.current_image_gray
                self.display_image_on_label(self.display_image)
            else:
                QMessageBox.warning(self, "Erro de Carregamento", "Não foi possível carregar a imagem selecionada.")

    def save_image(self):
        if self.display_image is None:
            QMessageBox.warning(self, "Erro de Salvamento", "Nenhuma imagem para salvar.")
            return

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Salvar Imagem", "imagem_processada.png",
                                                "Imagens (*.png *.jpg *.bmp *.tif);;Todos os Arquivos (*)", options=options)
        if file_name:
            if save_image_cv2(self.display_image, file_name):
                QMessageBox.information(self, "Sucesso", "Imagem salva com sucesso!")
            else:
                QMessageBox.warning(self, "Erro de Salvamento", "Não foi possível salvar a imagem.")

    def check_image_loaded(self):
        if self.current_image_gray is None:
            QMessageBox.warning(self, "Erro", "Por favor, carregue uma imagem primeiro.")
            return False
        return True

    def action_calculate_histogram(self):
        if not self.check_image_loaded(): return
        hist = self.image_processor.calculate_histogram(self.current_image_gray)
        self.image_processor.display_histogram(hist)

    def action_contrast_stretching(self):
        if not self.check_image_loaded(): return
        r1, s1, r2, s2 = 50, 0, 150, 255 
        result_image = self.image_processor.apply_contrast_stretching(self.current_image_gray, r1, s1, r2, s2)
        
        self.current_image_gray = result_image
        self.display_image = result_image
        self.display_image_on_label(self.display_image)

    def action_histogram_equalization(self):
        if not self.check_image_loaded(): return
        result_image = self.image_processor.apply_histogram_equalization(self.current_image_gray)
        
        self.current_image_gray = result_image
        self.display_image = result_image
        self.display_image_on_label(self.display_image)

    def get_kernel_size(self, default_size=3):
        text, ok = QInputDialog.getText(self, "Tamanho do Kernel", "Insira o tamanho do kernel (ímpar):",
                                         QLineEdit.Normal, str(default_size))
        if ok and text.isdigit():
            size = int(text)
            if size % 2 == 1 and size > 0:
                return size
            else:
                QMessageBox.warning(self, "Entrada Inválida", "O tamanho do kernel deve ser um número inteiro ímpar e positivo.")
        return None

    def action_mean_filter(self):
        if not self.check_image_loaded(): return
        kernel_size = self.get_kernel_size()
        if kernel_size:
            result_image = self.image_processor.apply_mean_filter(self.current_image_gray, kernel_size)
            
            self.current_image_gray = result_image
            self.display_image = result_image
            self.display_image_on_label(self.display_image)

    def action_median_filter(self):
        if not self.check_image_loaded(): return
        kernel_size = self.get_kernel_size()
        if kernel_size:
            result_image = self.image_processor.apply_median_filter(self.current_image_gray, kernel_size)
            
            self.current_image_gray = result_image
            self.display_image = result_image
            self.display_image_on_label(self.display_image)

    def action_gaussian_filter(self):
        if not self.check_image_loaded(): return
        kernel_size = self.get_kernel_size()
        if kernel_size:
            sigma_text, ok = QInputDialog.getText(self, "Sigma Gaussiano", "Insira o valor de Sigma:",
                                                 QLineEdit.Normal, "0")
            if ok and sigma_text.replace('.', '', 1).isdigit():
                sigma = float(sigma_text)
                result_image = self.image_processor.apply_gaussian_filter(self.current_image_gray, kernel_size, sigma)
                
                self.current_image_gray = result_image
                self.display_image = result_image
                self.display_image_on_label(self.display_image)
            else:
                QMessageBox.warning(self, "Entrada Inválida", "Sigma deve ser um número válido.")

    def action_max_filter(self):
        if not self.check_image_loaded(): return
        kernel_size = self.get_kernel_size()
        if kernel_size:
            result_image = self.image_processor.apply_max_filter(self.current_image_gray, kernel_size)
            
            self.current_image_gray = result_image
            self.display_image = result_image
            self.display_image_on_label(self.display_image)

    def action_min_filter(self):
        if not self.check_image_loaded(): return
        kernel_size = self.get_kernel_size()
        if kernel_size:
            result_image = self.image_processor.apply_min_filter(self.current_image_gray, kernel_size)
            
            self.current_image_gray = result_image
            self.display_image = result_image
            self.display_image_on_label(self.display_image)

    def action_laplacian_filter(self):
        if not self.check_image_loaded(): return
        result_image = self.image_processor.apply_laplacian_filter(self.current_image_gray)
        
        self.current_image_gray = result_image
        self.display_image = result_image
        self.display_image_on_label(self.display_image)

    def action_roberts_filter(self):
        if not self.check_image_loaded(): return
        result_image = self.image_processor.apply_roberts_filter(self.current_image_gray)
        
        self.current_image_gray = result_image
        self.display_image = result_image
        self.display_image_on_label(self.display_image)

    def action_prewitt_filter(self):
        if not self.check_image_loaded(): return
        result_image = self.image_processor.apply_prewitt_filter(self.current_image_gray)
        
        self.current_image_gray = result_image
        self.display_image = result_image
        self.display_image_on_label(self.display_image)

    def action_sobel_filter(self):
        if not self.check_image_loaded(): return
        result_image = self.image_processor.apply_sobel_filter(self.current_image_gray)
        
        self.current_image_gray = result_image
        self.display_image = result_image
        self.display_image_on_label(self.display_image)

    def action_frequency_domain_filter(self, filter_type):
        if not self.check_image_loaded(): return
        
        cutoff_text, ok = QInputDialog.getText(self, "Frequência de Corte", "Insira a frequência de corte:",
                                             QLineEdit.Normal, "30")
        if ok and cutoff_text.isdigit():
            cutoff_frequency = int(cutoff_text)
            result_image = self.image_processor.apply_frequency_domain_filter(self.current_image_gray, filter_type, cutoff_frequency)
            
            self.current_image_gray = result_image
            self.display_image = result_image
            self.display_image_on_label(self.display_image)
        else:
            QMessageBox.warning(self, "Entrada Inválida", "A frequência de corte deve ser um número inteiro.")

    def action_display_fourier_spectrum(self):
        if not self.check_image_loaded(): return
        # Esta operação não altera a imagem, apenas exibe um dado dela
        self.image_processor.display_fourier_spectrum(self.current_image_gray)

    def action_erosion(self):
        if not self.check_image_loaded(): return
        kernel_size_text, ok = QInputDialog.getText(self, "Tamanho do Kernel para Erosão", "Insira o tamanho do kernel (ex: 3 para 3x3):",
                                         QLineEdit.Normal, "3")
        if ok and kernel_size_text.isdigit():
            kernel_size = int(kernel_size_text)
            if kernel_size > 0:
                result_image = self.image_processor.apply_erosion(self.current_image_gray, (kernel_size, kernel_size))
                
                self.current_image_gray = result_image
                self.display_image = result_image
                self.display_image_on_label(self.display_image)
            else:
                QMessageBox.warning(self, "Entrada Inválida", "O tamanho do kernel deve ser um número positivo.")
        else:
            QMessageBox.warning(self, "Entrada Inválida", "Tamanho do kernel inválido.")

    def action_dilation(self):
        if not self.check_image_loaded(): return
        kernel_size_text, ok = QInputDialog.getText(self, "Tamanho do Kernel para Dilatação", "Insira o tamanho do kernel (ex: 3 para 3x3):",
                                         QLineEdit.Normal, "3")
        if ok and kernel_size_text.isdigit():
            kernel_size = int(kernel_size_text)
            if kernel_size > 0:
                result_image = self.image_processor.apply_dilation(self.current_image_gray, (kernel_size, kernel_size))
                
                self.current_image_gray = result_image
                self.display_image = result_image
                self.display_image_on_label(self.display_image)
            else:
                QMessageBox.warning(self, "Entrada Inválida", "O tamanho do kernel deve ser um número positivo.")
        else:
            QMessageBox.warning(self, "Entrada Inválida", "Tamanho do kernel inválido.")

    def action_otsu_thresholding(self):
        if not self.check_image_loaded(): return
        result_image = self.image_processor.apply_otsu_thresholding(self.current_image_gray)
        
        self.current_image_gray = result_image
        self.display_image = result_image
        self.display_image_on_label(self.display_image)

    # --- NOVAS AÇÕES PARA ABERTURA, FECHAMENTO E GRADIENTE ---
    def action_opening(self):
        if not self.check_image_loaded(): return
        kernel_size_text, ok = QInputDialog.getText(self, "Tamanho do Kernel para Abertura", "Insira o tamanho do kernel (ex: 3 para 3x3):",
                                         QLineEdit.Normal, "3")
        if ok and kernel_size_text.isdigit():
            kernel_size = int(kernel_size_text)
            if kernel_size > 0:
                result_image = self.image_processor.apply_opening(self.current_image_gray, (kernel_size, kernel_size))
                self.current_image_gray = result_image
                self.display_image = result_image
                self.display_image_on_label(self.display_image)
            else:
                QMessageBox.warning(self, "Entrada Inválida", "O tamanho do kernel deve ser um número positivo.")
        else:
            QMessageBox.warning(self, "Entrada Inválida", "Tamanho do kernel inválido.")

    def action_closing(self):
        if not self.check_image_loaded(): return
        kernel_size_text, ok = QInputDialog.getText(self, "Tamanho do Kernel para Fechamento", "Insira o tamanho do kernel (ex: 3 para 3x3):",
                                         QLineEdit.Normal, "3")
        if ok and kernel_size_text.isdigit():
            kernel_size = int(kernel_size_text)
            if kernel_size > 0:
                result_image = self.image_processor.apply_closing(self.current_image_gray, (kernel_size, kernel_size))
                self.current_image_gray = result_image
                self.display_image = result_image
                self.display_image_on_label(self.display_image)
            else:
                QMessageBox.warning(self, "Entrada Inválida", "O tamanho do kernel deve ser um número positivo.")
        else:
            QMessageBox.warning(self, "Entrada Inválida", "Tamanho do kernel inválido.")

    def action_morphological_gradient(self):
        if not self.check_image_loaded(): return
        kernel_size_text, ok = QInputDialog.getText(self, "Tamanho do Kernel para Gradiente", "Insira o tamanho do kernel (ex: 3 para 3x3):",
                                         QLineEdit.Normal, "3")
        if ok and kernel_size_text.isdigit():
            kernel_size = int(kernel_size_text)
            if kernel_size > 0:
                result_image = self.image_processor.apply_morphological_gradient(self.current_image_gray, (kernel_size, kernel_size))
                self.current_image_gray = result_image
                self.display_image = result_image
                self.display_image_on_label(self.display_image)
            else:
                QMessageBox.warning(self, "Entrada Inválida", "O tamanho do kernel deve ser um número positivo.")
        else:
            QMessageBox.warning(self, "Entrada Inválida", "Tamanho do kernel inválido.")