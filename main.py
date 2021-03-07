import os
import numpy as np

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPaintEvent, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QApplication, QCheckBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget

DEBUG = False


def check_im_file(x):
    return os.path.splitext(x)[-1] in ['.jpg', '.png']


def process_bbox(bb, nw, nh):
    x_min = int(bb.x_min * nw)
    x_max = int(bb.x_max * nw)
    y_min = int(bb.y_min * nh)
    y_max = int(bb.y_max * nh)
    return [bb.class_id, x_min, x_max, y_min, y_max]


def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def iou_mat(A, B):
    n_i = len(A)
    n_j = len(B)
    iou_mat = np.empty((n_i, n_j))
    for i in range(n_i):
        for j in range(n_j):
            iou_mat[i, j] = intersection_over_union(A[i], B[j])
    return iou_mat


COORD_LABELS = ['x_min', 'y_min', 'x_max', 'y_max']


class ImageView(QLabel):
    def __init__(self):
        super().__init__()
        self.ann_bboxes = []
        self.ref_bboxes = []

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)

        self.painter = QPainter()
        self.painter.begin(self)
        self.drawBboxes()
        self.painter.end()

    def drawSingleBbox(self, class_id, x_min, x_max, y_min, y_max, pen):
        self.painter.setPen(pen)
        self.painter.drawRect(x_min, y_min, x_max - x_min, y_max - y_min)
        self.painter.drawText(x_min, y_min+10, str(class_id))

    def drawBboxes(self):
        for class_id, x_min, x_max, y_min, y_max in self.ann_bboxes:
            self.drawSingleBbox(class_id,
                                x_min, x_max,
                                y_min, y_max,
                                QPen(Qt.yellow))

        for class_id, x_min, x_max, y_min, y_max in self.ref_bboxes:
            self.drawSingleBbox(class_id,
                                x_min, x_max,
                                y_min, y_max,
                                QPen(Qt.green))


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(1280, 720)

        self.imlist = []
        self.im_id = 0
        self.imdir = ''

        self.meta_df = None

        self.ann_df = None
        self.useann = False

        self.ref_df = None
        self.useref = False

        self.conf = 0.5
        self.clsfilter = list(range(14))

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
        w_imshow.setFixedWidth(950)

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
        w_cfg.setFixedWidth(275)

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

        l_metafn.addWidget(QLabel('Metadata'))

        w_metafn_btn = QPushButton('...')
        w_metafn_btn.setFixedWidth(20)
        w_metafn_btn.clicked.connect(self.btnClicked_metafn)
        l_metafn.addWidget(w_metafn_btn)

        self.w_metafn_txt = QLabel()
        l_metafn.addWidget(self.w_metafn_txt)

        l_loadcsv.addWidget(w_metafn)

        # .......

        w_annfn = QWidget()
        l_annfn = QHBoxLayout(w_annfn)

        self.cb_useann = QCheckBox()
        self.cb_useann.setFixedWidth(20)
        self.cb_useann.stateChanged.connect(self.cbChecked_useann)
        l_annfn.addWidget(self.cb_useann)

        l_annfn.addWidget(QLabel('Prediction'))
        w_annfn_btn = QPushButton('...')
        w_annfn_btn.setFixedWidth(20)
        w_annfn_btn.clicked.connect(self.btnClicked_annfn)
        l_annfn.addWidget(w_annfn_btn)

        self.w_annfn_txt = QLabel()
        l_annfn.addWidget(self.w_annfn_txt)

        l_loadcsv.addWidget(w_annfn)

        # .......

        w_reffn = QWidget()
        l_reffn = QHBoxLayout(w_reffn)

        self.cb_useref = QCheckBox()
        self.cb_useref.setFixedWidth(20)
        self.cb_useref.stateChanged.connect(self.cbChecked_useref)
        l_reffn.addWidget(self.cb_useref)

        l_reffn.addWidget(QLabel('Ground truth'))
        w_reffn_btn = QPushButton('...')
        w_reffn_btn.setFixedWidth(20)
        w_reffn_btn.clicked.connect(self.btnClicked_reffn)
        l_reffn.addWidget(w_reffn_btn)

        self.w_reffn_txt = QLabel()
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
        self.sl_conf.setMaximum(1000)
        self.sl_conf.setValue(500)
        self.sl_conf.setTickInterval(1)
        self.sl_conf.valueChanged.connect(self.slEdited_conf)
        l_conf.addWidget(self.sl_conf)

        l_cfg.addWidget(w_conf)

        # .....

        w_scroll_clsfilter = QScrollArea()
        w_scroll_clsfilter.setFixedHeight(250)
        w_scroll_clsfilter.setWidgetResizable(True)

        w_clsfilter = QWidget()
        w_scroll_clsfilter.setWidget(w_clsfilter)
        l_clsfilter = QVBoxLayout(w_clsfilter)

        self.w_gb_clsfilter = QGroupBox('Class filter')
        l_gb_clsfilter = QVBoxLayout(self.w_gb_clsfilter)
        for i in range(14):
            item = QCheckBox(str(i), self.w_gb_clsfilter)
            item.setChecked(True)
            item.stateChanged.connect(self.cbToggled_clsfilter)
            l_gb_clsfilter.addWidget(item)
        l_clsfilter.addWidget(self.w_gb_clsfilter)

        l_cfg.addWidget(w_scroll_clsfilter)

        # .....

        self.l_root = QHBoxLayout(self)
        self.l_root.addWidget(w_imview)
        self.l_root.addWidget(w_cfg)

    def cbChecked_useann(self):
        self.useann = self.cb_useann.isChecked()
        self.drawBboxes()

    def cbChecked_useref(self):
        self.useref = self.cb_useref.isChecked()
        self.drawBboxes()

    def cbToggled_clsfilter(self):
        self.clsfilter = [
            int(box.text())
            for box in self.w_gb_clsfilter.findChildren(QCheckBox)
            if box.isChecked()
        ]
        self.notify_clsfilter_changed()

    def slEdited_conf(self):
        self.conf = self.sl_conf.value() * 1.0 / self.sl_conf.maximum()
        self.notify_conf_changed()

    def btnClicked_imdir(self):
        picked = QFileDialog.getExistingDirectory(self, 'Select Folder')

        if DEBUG:
            picked = 'VinBigData-Abnormalities-Detection/data/vinbigdata-512/train'
        if picked != '':
            self.imdir = picked
            self.w_imdir_txt.setText(self.imdir)
            self.imlist = list(filter(check_im_file, os.listdir(self.imdir)))

            self.im_id = 0
            self.notify_imid_changed()

            self.w_impick.setDisabled(False)

    def btnClicked_metafn(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        picked, _ = QFileDialog.getOpenFileName(self,
                                                "QFileDialog.getOpenFileName()",
                                                "",
                                                "CSV Files (*.csv)", options=options)

        if DEBUG:
            picked = 'VinBigData-Abnormalities-Detection/data/meta/train_info.csv'
        if picked != '':
            self.metafn = picked
            self.w_metafn_txt.setText(os.path.basename(self.metafn))

            self.meta_df = pd.read_csv(self.metafn)
            self.meta_df = self.meta_df.set_index('image_id')

    def btnClicked_annfn(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        picked, _ = QFileDialog.getOpenFileName(self,
                                                "QFileDialog.getOpenFileName()",
                                                "",
                                                "CSV Files (*.csv)", options=options)

        if DEBUG:
            picked = '0.csv'
        if picked != '':
            self.annfn = picked
            self.w_annfn_txt.setText(os.path.basename(self.annfn))

            self.ann_df = pd.read_csv(self.annfn)
            self.ann_df['class_id'] = self.ann_df['class_id'].astype(int)

    def btnClicked_reffn(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        picked, _ = QFileDialog.getOpenFileName(self,
                                                "QFileDialog.getOpenFileName()",
                                                "",
                                                "CSV Files (*.csv)", options=options)

        if DEBUG:
            picked = 'VinBigData-Abnormalities-Detection/data/raw/train.csv'
        if picked != '':
            self.reffn = picked
            self.w_reffn_txt.setText(os.path.basename(self.reffn))

            self.ref_df = pd.read_csv(self.reffn)
            self.ref_df['class_id'] = self.ref_df['class_id'].astype(int)

            self.ref_df = self.ref_df.merge(
                self.meta_df,
                left_on='image_id', right_on='image_id',
                suffixes=('_', None)
            )
            self.ref_df.x_min /= self.ref_df.width
            self.ref_df.x_max /= self.ref_df.width
            self.ref_df.y_min /= self.ref_df.height
            self.ref_df.y_max /= self.ref_df.height

    def notify_imid_changed(self):
        self.w_impick_idtxt.setText(str(self.im_id))
        self.w_impick_fntxt.setText(self.imlist[self.im_id])
        self.setImage(self.imdir + '/' + self.imlist[self.im_id])

        self.drawBboxes()

    def notify_conf_changed(self):
        self.txt_conf.setText(f'Confidence threshold: {self.conf}')

        self.drawBboxes()

    def notify_clsfilter_changed(self):
        self.drawBboxes()

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
        if self.imlist is None or self.meta_df is None:
            return

        pid = os.path.splitext(self.imlist[self.im_id])[0]
        nw, nh = self.img.width(), self.img.height()

        if self.useann and self.ann_df is not None:
            pred = self.ann_df[self.ann_df['image_id'] == pid]
            pred = pred[pred.score >= self.conf]
            pred = pred[pred.class_id.isin(self.clsfilter)]

            self.img.ann_bboxes = [
                process_bbox(bb, nw, nh)
                for _, bb in pred.iterrows()
            ]
        else:
            self.img.ann_bboxes = []

        if self.useref and self.ref_df is not None:
            gt = self.ref_df[self.ref_df['image_id'] == pid]
            gt = gt[gt['class_id'].isin(self.clsfilter)]

            self.img.ref_bboxes = [
                process_bbox(bb, nw, nh)
                for _, bb in gt.iterrows()
            ]

            if False:
                p = pred[COORD_LABELS].values
                g = gt[COORD_LABELS].values
                ious = iou_mat(p, g)

                ii, _ = np.where(ious > 0.4)

                self.img.ann_bboxes = [
                    process_bbox(bb, nw, nh)
                    for i, (_, bb) in enumerate(pred.iterrows())
                    if i in ii
                ]
        else:
            self.img.ref_bboxes = []

        self.img.update()


app = QApplication([])
window = MyApp()
window.show()
app.exec()
