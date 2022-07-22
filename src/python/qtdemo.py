import sys
import pymarian
from sacremoses import MosesPunctNormalizer
from sentence_splitter import SentenceSplitter

from PyQt5.QtWidgets import *

class Example(QWidget):
    
    def __init__(self):
        super().__init__()

        self.cache = dict()
        self.norm = MosesPunctNormalizer(lang="en")
        self.splitter = SentenceSplitter("en")
                    
        self.setWindowTitle("Live Translator")
        # setting the geometry of window
        self.setGeometry(300, 300, 1200, 800)

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
        self.cli.setText("-c models/enu.deu.yml --cpu-threads 8")

        self.reload = QPushButton("Reload")
        self.reload.clicked.connect(self.onClicked)
        self.run = QPushButton("Translate")
        self.run.clicked.connect(self.onChanged)

        hbox.addWidget(self.cli)
        hbox.addWidget(self.reload)        
        hbox.addWidget(self.run)

        layout = QVBoxLayout()
        layout.addLayout(hbox)
        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.input)
        hbox2.addWidget(self.output)
        layout.addLayout(hbox2)
        self.setLayout(layout)

        self.reloadMarian()
        self.show()

    def onChanged(self):
        inputText = self.input.toPlainText()
        if not self.current:
            self.reloadMarian()
        if self.current:
            outputText = self.translate(inputText)
            self.output.setPlainText(outputText)

    def onClicked(self):
        self.reloadMarian()
        
    def reloadMarian(self):
        command = self.cli.text()
        print(command)
        if command not in self.cache:
            self.cache[command] = dict()
            self.cache[command]["#MODEL#"] = pymarian.Translator(command)
        self.current = self.cache[command]

    def translate(self, inputText):
        inputLines = [self.splitter.split(p) 
            for p in inputText.split("\n")]

        candidates = []
        for paragraph in inputLines:
            for line in paragraph:
                if line not in self.current:
                    candidates.append(line)
    
        outputLines = self.current["#MODEL#"].translate(
            [self.norm.normalize(c) for c in candidates]
        )

        for (src, trg) in zip(candidates, outputLines):
            self.current[src] = trg 
        
        return "\n".join([
            " ".join([self.current[src] for src in paragraph]) 
            for paragraph in inputLines
        ])
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
