import sys
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PySide6.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QPushButton, QFileDialog
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PIL import Image
import cv2

# Classe principal da aplicação
class ObjectRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Inicialização da interface de usuário
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle("Reconhecimento de Objetos")

        # Configuração da exibição da imagem
        self.view = QGraphicsView(self)
        self.view.setGeometry(10, 10, 780, 500)
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)

        # Botão para carregar uma imagem
        self.loadImageButton = QPushButton("SELECIONAR IMAGEM", self)
        self.loadImageButton.setGeometry(10, 520, 180, 40)
        self.loadImageButton.clicked.connect(self.loadImage)

        self.selectedImagePath = None
        self.model = None

        # Verifica se a GPU está disponível
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def loadModel(self):
        # Carregamento do modelo Faster R-CNN pré-treinado
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

    def loadImage(self):
        # Diálogo para selecionar uma imagem
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        filePath, _ = QFileDialog.getOpenFileName(self, "SELECIONAR IMAGEM", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)", options=options)

        if filePath:
            self.selectedImagePath = filePath
            if not self.model:
                self.loadModel()
            image = Image.open(self.selectedImagePath)
            self.processImage(image)

    def processImage(self, image):
        if self.model:
            # Converte a imagem em um tensor PyTorch
            input_tensor = T.ToTensor()(image).unsqueeze(0).to(self.device)
            
            # Realiza detecções de objetos na imagem usando o modelo carregado
            with torch.no_grad():
                predictions = self.model(input_tensor)
            
            # Desenha caixas delimitadoras ao redor dos objetos detectados
            image_with_boxes = self.drawBoundingBoxes(np.array(image), predictions[0])

            # Exibe a imagem com as caixas delimitadoras
            self.displayImage(image_with_boxes)

    def drawBoundingBoxes(self, image, predictions):
        image_with_boxes = image.copy()
        boxes = predictions["boxes"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        threshold = 0.5

        for box, label, score in zip(boxes, labels, scores):
            if score >= threshold:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_with_boxes, f"Class: {label}, Score: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image_with_boxes

    def displayImage(self, image):
        # Verifica o número de canais na imagem
        if len(image.shape) == 2:
            # Se for uma imagem em escala de cinza, converte-a em RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        h, w, ch = image.shape
        q_img = QImage(image.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        item = QGraphicsPixmapItem(pixmap)
        self.scene.clear()
        self.scene.addItem(item)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectRecognitionApp()
    window.show()
    sys.exit(app.exec())
