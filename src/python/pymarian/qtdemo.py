import sys
import time

import pymarian
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from sacremoses import MosesPunctNormalizer, MosesTokenizer
from sentence_splitter import SentenceSplitter


class Example(QWidget):
    def __init__(self):
        super().__init__()

        self.cache = dict()
        self.norm = MosesPunctNormalizer(lang="en")
        self.tok = MosesTokenizer(lang="en")
        self.splitter = SentenceSplitter("en")

        self.setWindowTitle("Live Translator")
        self.setFont(QFont(self.font().family(), 11))
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
        self.cli.setText(
            "-c models/enu.deu.yml --cpu-threads 8 -b1 --mini-batch-words 256 --maxi-batch 100 --maxi-batch-sort src"
        )

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

        self.statusBar = QStatusBar()
        layout.addWidget(self.statusBar)

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
        self.cache = dict()  # clean instead of caching
        if command not in self.cache:
            self.cache[command] = dict()
            self.cache[command]["#MODEL#"] = pymarian.Translator(command)
        self.current = self.cache[command]

    def translate(self, inputText):
        t0 = time.perf_counter()

        inputLines = [self.splitter.split(p) for p in inputText.split("\n")]

        unseenLines = []
        for paragraph in inputLines:
            for line in paragraph:
                if line not in self.current:
                    unseenLines.append(line)

        normLines = [self.norm.normalize(c) for c in unseenLines]

        t1 = time.perf_counter()
        outputLines = self.current["#MODEL#"].translate(normLines)
        t2 = time.perf_counter()

        totalStat = sum([len(self.tok.tokenize(line)) for line in unseenLines])

        if totalStat:
            self.statusBar.showMessage(
                f"Translated {totalStat} tokens ({len(unseenLines)} lines) in {t2 - t1:.2f} second ({totalStat / (t2 - t1):.2f} tokens per second). Preprocessing took {t1 - t0:.2f} seconds. Total: {t2 - t0:.2f} seconds"
            )

        for src, trg in zip(unseenLines, outputLines):
            self.current[src] = trg

        return "\n".join([" ".join([self.current[src] for src in paragraph]) for paragraph in inputLines])


def main():
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
