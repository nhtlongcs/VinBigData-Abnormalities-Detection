import os

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPaintEvent, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QApplication, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget


def check_im_file(x):
    return os.path.splitext(x)[-1] in ['.jpg', '.png']


class ImageView(QLabel):
    def __init__(self):
        super().__init__()
        self.bboxes = []

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)

        self.painter = QPainter()
        self.painter.begin(self)
        self.drawBboxes()
        self.painter.end()

    def drawSingleBbox(self, class_id, x_min, x_max, y_min, y_max):
        self.painter.setPen(QPen(Qt.red))
        self.painter.drawRect(x_min, y_min, x_max - x_min, y_max - y_min)
        self.painter.drawText(x_min, y_min, str(class_id))

    def drawBboxes(self):
        for class_id, x_min, x_max, y_min, y_max in self.bboxes:
            self.drawSingleBbox(class_id, x_min, x_max, y_min, y_max)


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(1280, 720)

        self.imlist = []
        self.im_id = 0
        self.imdir = ''

        self.metafn = ''
        self.meta_df = None

        self.annfn = ''
        self.ann_df = None

        self.reffn = ''
        self.ref_df = None

        self.conf = 0.5

        # ==>

        w_imview = QWidget()
        l_imview = QVBoxLayout()
        w_imview.setLayout(l_imview)

        # ====>

        w_imdir = QWidget()
        l_imdir = QHBoxLayout()
        w_imdir.setLayout(l_imdir)

        w_imdir_btn = QPushButton('Browse')
        w_imdir_btn.setFixedWidth(100)
        w_imdir_btn.clicked.connect(self.btnClicked_imdir)
        l_imdir.addWidget(w_imdir_btn)

        self.w_imdir_txt = QLabel('Choose the directory containing the images')
        l_imdir.addWidget(self.w_imdir_txt)

        l_imview.addWidget(w_imdir)

        # ====>

        w_imshow = QScrollArea(self)
        w_imshow.setFixedWidth(1000)

        self.img = ImageView()

        w_imshow.setWidget(self.img)

        l_imview.addWidget(w_imshow)

        # ====>

        self.w_impick = QWidget()
        self.w_impick.setDisabled(True)
        l_impick = QHBoxLayout()
        self.w_impick.setLayout(l_impick)

        w_impick_leftbtn = QPushButton('<')
        w_impick_leftbtn.setFixedWidth(30)
        w_impick_leftbtn.clicked.connect(self.btnClicked_imdir_leftbtn)
        l_impick.addWidget(w_impick_leftbtn)

        self.w_impick_idtxt = QLineEdit()
        self.w_impick_idtxt.setFixedWidth(100)
        self.w_impick_idtxt.returnPressed.connect(self.txtEdited_impick_id_txt)
        l_impick.addWidget(self.w_impick_idtxt)

        w_impick_rightbtn = QPushButton('>')
        w_impick_rightbtn.setFixedWidth(30)
        w_impick_rightbtn.clicked.connect(self.btnClicked_imdir_rightbtn)
        l_impick.addWidget(w_impick_rightbtn)

        self.w_impick_fntxt = QLabel()
        l_impick.addWidget(self.w_impick_fntxt)

        l_imview.addWidget(self.w_impick)

        # ======= Configuration =======

        w_cfg = QWidget()
        w_cfg.setFixedWidth(280)

        l_cfg = QVBoxLayout()
        w_cfg.setLayout(l_cfg)

        # .......

        w_loadcsv = QWidget()
        w_loadcsv.setFixedHeight(150)
        l_loadcsv = QVBoxLayout()
        w_loadcsv.setLayout(l_loadcsv)

        w_metafn = QWidget()
        l_metafn = QHBoxLayout()
        w_metafn.setLayout(l_metafn)

        w_metafn_btn = QPushButton('Browse')
        w_metafn_btn.setFixedWidth(100)
        w_metafn_btn.clicked.connect(self.btnClicked_metafn)
        l_metafn.addWidget(w_metafn_btn)

        self.w_metafn_txt = QLabel('Path to metadata of annotation file')
        l_metafn.addWidget(self.w_metafn_txt)

        l_loadcsv.addWidget(w_metafn)

        # .......

        w_annfn = QWidget()
        l_annfn = QHBoxLayout()
        w_annfn.setLayout(l_annfn)

        w_annfn_btn = QPushButton('Browse')
        w_annfn_btn.setFixedWidth(100)
        w_annfn_btn.clicked.connect(self.btnClicked_annfn)
        l_annfn.addWidget(w_annfn_btn)

        self.w_annfn_txt = QLabel('Path to annotation file')
        l_annfn.addWidget(self.w_annfn_txt)

        l_loadcsv.addWidget(w_annfn)

        # .......

        w_reffn = QWidget()
        l_reffn = QHBoxLayout()
        w_reffn.setLayout(l_reffn)

        w_reffn_btn = QPushButton('Browse')
        w_reffn_btn.setFixedWidth(100)
        w_reffn_btn.clicked.connect(self.btnClicked_reffn)
        l_reffn.addWidget(w_reffn_btn)

        self.w_reffn_txt = QLabel('Path to reference file')
        l_reffn.addWidget(self.w_reffn_txt)

        l_loadcsv.addWidget(w_reffn)

        l_cfg.addWidget(w_loadcsv)

        # ..........

        w_conf = QWidget()
        w_conf.setFixedHeight(100)

        l_conf = QVBoxLayout()
        w_conf.setLayout(l_conf)

        self.txt_conf = QLabel('Confidence threshold: 0.5')
        l_conf.addWidget(self.txt_conf)

        self.sl_conf = QSlider(Qt.Horizontal)
        self.sl_conf.setMinimum(0)
        self.sl_conf.setMaximum(100)
        self.sl_conf.setValue(50)
        self.sl_conf.setTickInterval(1)
        self.sl_conf.valueChanged.connect(self.slEdited_conf)
        l_conf.addWidget(self.sl_conf)

        l_cfg.addWidget(w_conf)

        self.l_root = QHBoxLayout()
        self.l_root.addWidget(w_imview)
        self.l_root.addWidget(w_cfg)

        self.setLayout(self.l_root)

    def slEdited_conf(self):
        self.conf = self.sl_conf.value()
        self.notify_conf_changed()

    def btnClicked_imdir(self):
        picked_dir = QFileDialog.getExistingDirectory(self, 'Select Folder')

        if picked_dir != '':
            self.imdir = picked_dir
            self.w_imdir_txt.setText(self.imdir)
            self.imlist = list(filter(check_im_file, os.listdir(self.imdir)))

            self.im_id = 0
            self.notify_imid_changed()

            self.w_impick.setDisabled(False)

    def btnClicked_annfn(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        picked, _ = QFileDialog.getOpenFileName(self,
                                                "QFileDialog.getOpenFileName()",
                                                "",
                                                "CSV Files (*.csv)", options=options)
        if picked != '':
            self.annfn = picked
            self.w_annfn_txt.setText(os.path.basename(self.annfn))

            self.ann_df = pd.read_csv(self.annfn)

    def btnClicked_metafn(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        picked, _ = QFileDialog.getOpenFileName(self,
                                                "QFileDialog.getOpenFileName()",
                                                "",
                                                "CSV Files (*.csv)", options=options)
        if picked != '':
            self.metafn = picked
            self.w_metafn_txt.setText(os.path.basename(self.metafn))

            self.meta_df = pd.read_csv(self.metafn)
            self.meta_df = self.meta_df.set_index('image_id')

    def btnClicked_reffn(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        picked, _ = QFileDialog.getOpenFileName(self,
                                                "QFileDialog.getOpenFileName()",
                                                "",
                                                "CSV Files (*.csv)", options=options)
        if picked != '':
            self.reffn = picked
            self.w_reffn_txt.setText(os.path.basename(self.reffn))

    def notify_imid_changed(self):
        self.w_impick_idtxt.setText(str(self.im_id))
        self.w_impick_fntxt.setText(self.imlist[self.im_id])
        self.setImage(self.imdir + '/' + self.imlist[self.im_id])

        self.drawBboxes()

    def notify_conf_changed(self):
        self.txt_conf.setText(f'Confidence threshold: {self.conf / 100}')

    def btnClicked_imdir_leftbtn(self):
        if self.im_id > 0:
            self.im_id -= 1
            self.notify_imid_changed()

    def btnClicked_imdir_rightbtn(self):
        if self.im_id < len(self.imlist) - 1:
            self.im_id += 1
            self.notify_imid_changed()

    def txtEdited_impick_id_txt(self):
        if self.w_impick_idtxt.text().isdigit():
            new_im_id = int(self.w_impick_idtxt.text())
            if 0 <= new_im_id < len(self.imlist):
                self.im_id = new_im_id
        self.notify_imid_changed()

    def setImage(self, image_fn):
        self.img.setPixmap(QPixmap(image_fn))
        self.img.adjustSize()

    def drawBboxes(self):
        if self.ann_df is None:
            return

        pid = os.path.splitext(self.imlist[self.im_id])[0]

        bboxes = self.ann_df[self.ann_df['image_id'] == pid]

        w, h = self.meta_df.loc[pid].width, self.meta_df.loc[pid].height
        nw, nh = self.img.width(), self.img.height()

        def process_bbox(bb, w, h, nw, nh):
            class_id = int(bb.class_id)
            x_min = int(bb.x_min / w * nw)
            x_max = int(bb.x_max / w * nw)
            y_min = int(bb.y_min / h * nh)
            y_max = int(bb.y_max / h * nh)
            return [class_id, x_min, x_max, y_min, y_max]

        self.img.bboxes = [
            process_bbox(bb, w, h, nw, nh)
            for i, bb in bboxes.iterrows()
            if 0 <= int(bb.class_id) < 14
        ]


app = QApplication([])
window = MyApp()
window.show()
app.exec()
