# src/image_processor.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self):
        pass

    def calculate_histogram(self, image_gray):
        """
        Calcula o histograma de uma imagem em níveis de cinza.
        """
        if image_gray is None:
            return None
        hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
        return hist

    def display_histogram(self, hist_data, title="Histograma da Imagem"):
        """
        Exibe o histograma usando Matplotlib.
        """
        if hist_data is None:
            print("Dados do histograma não disponíveis.")
            return

        plt.figure()
        plt.title(title)
        plt.xlabel("Nível de Intensidade")
        plt.ylabel("Número de Pixels")
        plt.plot(hist_data)
        plt.xlim([0, 256])
        plt.show()

    def apply_contrast_stretching(self, image_gray, r1, s1, r2, s2):
        """
        Aplica alargamento de contraste.
        Para simplificar, uma versão linear por partes.
        """
        if image_gray is None:
            return None

        # Certifica-se de que a imagem é float para cálculos
        img_float = image_gray.astype(np.float32)

        # Mapeamento linear por partes
        def map_pixel(pixel):
            if 0 <= pixel <= r1:
                return (s1 / r1) * pixel
            elif r1 < pixel <= r2:
                return ((s2 - s1) / (r2 - r1)) * (pixel - r1) + s1
            else: # r2 < pixel <= 255
                return ((255 - s2) / (255 - r2)) * (pixel - r2) + s2

        # Aplica a função de mapeamento a cada pixel
        stretched_image = np.vectorize(map_pixel)(img_float)

        # Clipa os valores para garantir que estão entre 0 e 255 e converte para uint8
        stretched_image = np.clip(stretched_image, 0, 255).astype(np.uint8)
        return stretched_image


    def apply_histogram_equalization(self, image_gray):
        """
        Aplica equalização de histograma.
        """
        if image_gray is None:
            return None
        return cv2.equalizeHist(image_gray)

    # --- Filtros Passa-Baixa ---
    def apply_mean_filter(self, image_gray, kernel_size):
        """
        Aplica filtro da média.
        """
        if image_gray is None:
            return None
        return cv2.blur(image_gray, (kernel_size, kernel_size))

    def apply_median_filter(self, image_gray, kernel_size):
        """
        Aplica filtro da mediana.
        """
        if image_gray is None:
            return None
        # Kernel size deve ser ímpar
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.medianBlur(image_gray, kernel_size)

    def apply_gaussian_filter(self, image_gray, kernel_size, sigma):
        """
        Aplica filtro Gaussiano.
        """
        if image_gray is None:
            return None
        # Kernel size deve ser ímpar
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image_gray, (kernel_size, kernel_size), sigmaX=sigma)

    def apply_max_filter(self, image_gray, kernel_size):
        """
        Aplica filtro de Máximo (dilatação morfológica com kernel quadrado).
        """
        if image_gray is None:
            return None
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(image_gray, kernel)

    def apply_min_filter(self, image_gray, kernel_size):
        """
        Aplica filtro de Mínimo (erosão morfológica com kernel quadrado).
        """
        if image_gray is None:
            return None
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.erode(image_gray, kernel)


    # --- Filtros Passa-Alta ---
    def apply_laplacian_filter(self, image_gray):
        """
        Aplica filtro Laplaciano.
        """
        if image_gray is None:
            return None
        laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
        return cv2.convertScaleAbs(laplacian)

    def apply_roberts_filter(self, image_gray):
        """
        Aplica o filtro Roberts.
        """
        if image_gray is None:
            return None
        
        img_float = np.float32(image_gray)
        
        kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        
        grad_x = cv2.filter2D(img_float, cv2.CV_32F, kernel_x)
        grad_y = cv2.filter2D(img_float, cv2.CV_32F, kernel_y)
        
        roberts_edge = np.sqrt(grad_x**2 + grad_y**2)
        
        roberts_edge = cv2.normalize(roberts_edge, None, 0, 255, cv2.NORM_MINMAX)
        return np.uint8(roberts_edge)


    def apply_prewitt_filter(self, image_gray):
        """
        Aplica o filtro Prewitt.
        """
        if image_gray is None:
            return None
        
        img_float = np.float32(image_gray)

        kernel_x = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -1, -1],
                             [0, 0, 0],
                             [1, 1, 1]], dtype=np.float32)

        grad_x = cv2.filter2D(img_float, cv2.CV_32F, kernel_x)
        grad_y = cv2.filter2D(img_float, cv2.CV_32F, kernel_y)

        prewitt_edge = np.sqrt(grad_x**2 + grad_y**2)

        prewitt_edge = cv2.normalize(prewitt_edge, None, 0, 255, cv2.NORM_MINMAX)
        return np.uint8(prewitt_edge)


    def apply_sobel_filter(self, image_gray):
        """
        Aplica o filtro Sobel.
        """
        if image_gray is None:
            return None
        sobelx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
        
        sobel_edge = np.sqrt(sobelx**2 + sobely**2)
        
        sobel_edge = cv2.normalize(sobel_edge, None, 0, 255, cv2.NORM_MINMAX)
        return np.uint8(sobel_edge)

    # --- Convolução no Domínio da Frequência ---
    def apply_frequency_domain_filter(self, image_gray, filter_type="lowpass", cutoff_frequency=30):
        """
        Implementa filtros passa-alta/passa-baixa no domínio da frequência.
        filter_type: "lowpass" (passa-baixa) ou "highpass" (passa-alta).
        cutoff_frequency: Frequência de corte para o filtro.
        """
        if image_gray is None:
            return None

        rows, cols = image_gray.shape
        
        dft = cv2.dft(np.float32(image_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        center_row, center_col = rows // 2, cols // 2
        mask = np.zeros((rows, cols, 2), np.float32)

        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt((i - center_row)**2 + (j - center_col)**2)
                if filter_type == "lowpass":
                    if distance <= cutoff_frequency:
                        mask[i, j] = 1
                elif filter_type == "highpass":
                    if distance > cutoff_frequency:
                        mask[i, j] = 1

        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        return np.uint8(img_back)

    def display_fourier_spectrum(self, image_gray):
        """
        Exibe a imagem com seu espectro de Fourier.
        """
        if image_gray is None:
            print("Imagem não disponível para exibir o espectro de Fourier.")
            return

        dft = cv2.dft(np.float32(image_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

        plt.figure(figsize=(10, 5))

        plt.subplot(121), plt.imshow(image_gray, cmap='gray')
        plt.title('Imagem Original'), plt.xticks([]), plt.yticks([])

        plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Espectro de Fourier (Magnitude)'), plt.xticks([]), plt.yticks([])

        plt.show()

    # --- Morfologia Matemática Básica ---
    def apply_erosion(self, image_gray, kernel_shape=(3, 3)):
        """
        Aplica operação de erosão.
        kernel_shape: Tupla (altura, largura) para o kernel retangular.
        """
        if image_gray is None:
            return None
        kernel = np.ones(kernel_shape, np.uint8)
        return cv2.erode(image_gray, kernel, iterations=1)

    def apply_dilation(self, image_gray, kernel_shape=(3, 3)):
        """
        Aplica operação de dilatação.
        kernel_shape: Tupla (altura, largura) para o kernel retangular.
        """
        if image_gray is None:
            return None
        kernel = np.ones(kernel_shape, np.uint8)
        return cv2.dilate(image_gray, kernel, iterations=1)

    # --- Segmentação ---
    def apply_otsu_thresholding(self, image_gray):
        """
        Aplica o método de Otsu para limiarização.
        Retorna a imagem binária.
        """
        if image_gray is None:
            return None
        ret, otsu_image = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"Limiar de Otsu encontrado: {ret}")
        return otsu_image

    # --- NOVAS FUNÇÕES: Morfologia Matemática Avançada ---
    def apply_opening(self, image_gray, kernel_shape=(3, 3)):
        """
        Aplica a operação de Abertura (Erosão seguida de Dilatação).
        Útil para remover pequenos objetos e suavizar contornos.
        """
        if image_gray is None:
            return None
        kernel = np.ones(kernel_shape, np.uint8)
        return cv2.morphologyEx(image_gray, cv2.MORPH_OPEN, kernel)

    def apply_closing(self, image_gray, kernel_shape=(3, 3)):
        """
        Aplica a operação de Fechamento (Dilatação seguida de Erosão).
        Útil para preencher pequenos buracos e fechar lacunas.
        """
        if image_gray is None:
            return None
        kernel = np.ones(kernel_shape, np.uint8)
        return cv2.morphologyEx(image_gray, cv2.MORPH_CLOSE, kernel)

    def apply_morphological_gradient(self, image_gray, kernel_shape=(3, 3)):
        """
        Aplica a operação de Gradiente Morfológico (Dilatação - Erosão).
        Realça as bordas e os contornos.
        """
        if image_gray is None:
            return None
        kernel = np.ones(kernel_shape, np.uint8)
        return cv2.morphologyEx(image_gray, cv2.MORPH_GRADIENT, kernel)