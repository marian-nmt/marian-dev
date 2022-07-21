import sys
import pymarian
from sacremoses import MosesPunctNormalizer
from sentence_splitter import SentenceSplitter, split_text_into_sentences

from PyQt5.QtWidgets import *

# marian = pymarian.Translator(
#     model="model.bin", vocabs=["enu.spm", "enu.spm"], 
#     beam_search=2, normalize=1, mini_batch=1, maxi_batch=1, 
#     cpu_threads=1, output_approx_knn=[128, 1024]
# ) ---> Translator("--model model.bin --vocabs enu.spm enu.spm --beam-search 2 --normalize 1 --mini-batch 1 --maxi-batch 1 --cpu-threads 1 --output-approx-knn 128 1024")

norm = MosesPunctNormalizer(lang="en")
splitter = SentenceSplitter("en")

class Example(QWidget):
    
    def __init__(self):
        super().__init__()
                    
        self.setWindowTitle("Live Translator")
        # setting the geometry of window
        self.setGeometry(300, 300, 500, 500)

        # centering
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

        self.marian = None

        self.input = QPlainTextEdit(self)
        self.input.textChanged.connect(self.onChanged)
        self.output = QPlainTextEdit(self)
        
        hbox = QHBoxLayout()
        self.cli = QLineEdit(self)
        self.cli.setText("-c decoder.yml --cpu-threads 8")

        self.reload = QPushButton("Reload")
        self.reload.clicked.connect(self.onClicked)

        hbox.addWidget(self.cli)
        hbox.addWidget(self.reload)

        layout = QVBoxLayout()
        layout.addLayout(hbox)
        layout.addWidget(self.input)
        layout.addWidget(self.output)
        self.setLayout(layout)

        self.show()

    def onChanged(self):
        self.reloadMarian()
        inputText = norm.normalize(self.input.toPlainText())
        inputLines = splitter.split(inputText)
        if self.marian:
            outputLines = self.marian.translate(inputLines)
            outputText = "\n".join(outputLines)
            self.output.setPlainText(outputText)

    def onClicked(self):
        self.reloadMarian()
        
    def reloadMarian(self):
        if not self.marian:
            command = self.cli.text()
            print(command)
            self.marian = pymarian.Translator(command)
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
